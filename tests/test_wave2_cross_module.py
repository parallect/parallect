"""Wave-2: Cross-module integration + chaos tests.

Tests the seams between wave-1 subsystems: FilesystemPlugin + embeddings,
plugin fan-out with real plugin resolution, PriorResearchCache chaining,
and chaos scenarios (500s, malformed JSON, corrupted stores, etc.).
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from parallect.backends import BackendSpec
from parallect.backends.adapters import BackendError
from parallect.plugins.data_sources import (
    DataSource,
    Document,
    IndexStats,
    PluginError,
    SourceSpec,
    get_registry,
    register,
    reset_registry,
)
from parallect.plugins.data_sources.filesystem import FilesystemPlugin
from parallect.plugins.data_sources.prior_research import PriorResearchCache
from parallect.plugins.data_sources.prxhub import PrxhubPlugin
from parallect.orchestrator.plugin_sources import (
    PluginFanOutResult,
    run_plugin_sources,
)
from parallect.providers import ProviderResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed_factory(mapping: dict[str, list[float]], default_dim: int = 4):
    async def fake_embed(texts, **_kwargs):
        return [mapping.get(t, [0.0] * default_dim) for t in texts]
    return fake_embed


def _patch_embed(monkeypatch, fake):
    import parallect.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "embed", fake)


def _patch_embed_dimensions(monkeypatch, dims: int | None):
    import parallect.embeddings as emb_mod

    async def fake_dims(**_kwargs):
        if dims is None:
            raise RuntimeError("no dims")
        return dims

    monkeypatch.setattr(emb_mod, "embed_dimensions", fake_dims)


def _patch_httpx_prxhub(monkeypatch, handler):
    orig = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return orig(*args, **kwargs)

    monkeypatch.setattr(
        "parallect.plugins.data_sources.prxhub.httpx.AsyncClient", patched
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture(autouse=True)
def _reset_embed_cache():
    import parallect.embeddings as emb_mod
    emb_mod._reset_caches()
    yield
    emb_mod._reset_caches()


# ---------------------------------------------------------------------------
# 1. Cross-module integration
# ---------------------------------------------------------------------------


class TestFilesystemWithVaryingEmbeddings:
    """FilesystemPlugin + mock backend returning varying vectors."""

    async def test_varying_vectors_ranked_correctly(
        self, tmp_path: Path, monkeypatch
    ):
        """Backend returns distinct vectors per chunk; search ranks correctly."""
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "a.md").write_text("# A\n\nalpha content about consensus")
        (notes / "b.md").write_text("# B\n\nbeta content about coffee")

        mapping = {
            "# A\n\nalpha content about consensus": [1.0, 0.0, 0.0, 0.0],
            "# B\n\nbeta content about coffee": [0.0, 1.0, 0.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure({
            "name": "test",
            "path": str(notes),
            "index_dir": str(tmp_path / "idx"),
        })
        await p.index()

        # Query close to "a" vector
        async def q_embed(texts, **_):
            return [[0.95, 0.05, 0.0, 0.0]]

        _patch_embed(monkeypatch, q_embed)
        docs = await p.search("consensus", limit=2)
        assert len(docs) == 2
        assert "a.md" in docs[0].metadata["path"]

    async def test_embed_returns_empty_for_query(
        self, tmp_path: Path, monkeypatch
    ):
        """If embed() returns empty vectors for a query, search returns []."""
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "a.md").write_text("# A\n\nalpha")

        _patch_embed(
            monkeypatch,
            _fake_embed_factory({"# A\n\nalpha": [1.0, 0.0, 0.0, 0.0]}),
        )
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure({
            "name": "test",
            "path": str(notes),
            "index_dir": str(tmp_path / "idx"),
        })
        await p.index()

        async def empty_embed(texts, **_):
            return []

        _patch_embed(monkeypatch, empty_embed)
        docs = await p.search("anything")
        assert docs == []


class TestPluginFanOutEndToEnd:
    """--sources fan-out with mocked providers + real plugin resolution."""

    async def test_multiple_sources_fan_out(self, tmp_path: Path, monkeypatch):
        """Two plugins registered; fan-out returns results from both."""

        class FakePlugin:
            name = "alpha"
            display_name = "alpha"
            requires_index = False

            async def configure(self, config):
                pass

            async def search(self, query, *, limit=10):
                return [Document(id="a1", content="alpha doc", title="Alpha")]

            async def index(self, *, force=False):
                return IndexStats(source_name="alpha", documents_indexed=0,
                                  documents_skipped=0, elapsed_seconds=0.0)

            async def is_fresh(self):
                return True

            async def fetch(self, doc_id):
                return None

            async def health_check(self):
                return {"status": "ok"}

        class FakePlugin2:
            name = "beta"
            display_name = "beta"
            requires_index = False

            async def configure(self, config):
                pass

            async def search(self, query, *, limit=10):
                return [Document(id="b1", content="beta doc", title="Beta")]

            async def index(self, *, force=False):
                return IndexStats(source_name="beta", documents_indexed=0,
                                  documents_skipped=0, elapsed_seconds=0.0)

            async def is_fresh(self):
                return True

            async def fetch(self, doc_id):
                return None

            async def health_check(self):
                return {"status": "ok"}

        register(FakePlugin())
        register(FakePlugin2())

        results = await run_plugin_sources("test query", "alpha,beta")
        assert len(results) == 2
        assert results[0].result is not None
        assert results[1].result is not None
        assert results[0].result.provider == "alpha"
        assert results[1].result.provider == "beta"
        assert "alpha doc" in results[0].result.report_markdown
        assert "beta doc" in results[1].result.report_markdown


class TestDimensionMismatchOnSearch:
    """Index with model A dims, search with model B dims -> clear error."""

    async def test_dim_mismatch_raises_on_search(
        self, tmp_path: Path, monkeypatch
    ):
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "a.md").write_text("# A\n\nalpha content")

        _patch_embed(
            monkeypatch,
            _fake_embed_factory({"# A\n\nalpha content": [1.0, 0.0, 0.0, 0.0]}),
        )
        _patch_embed_dimensions(monkeypatch, 4)

        from parallect.config_mod.settings import ParallectSettings
        monkeypatch.setattr(
            ParallectSettings, "load",
            classmethod(lambda cls: cls(embeddings_model="model-a")),
        )

        p = FilesystemPlugin()
        await p.configure({
            "name": "test", "path": str(notes),
            "index_dir": str(tmp_path / "idx"),
        })
        await p.index()

        async def wrong_dim_embed(texts, **_):
            return [[0.1] * 8]

        _patch_embed(monkeypatch, wrong_dim_embed)
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            await p.search("anything")


class TestPriorResearchChain:
    """Append a run, then search finds it."""

    async def test_append_then_search_finds_prior(
        self, tmp_path: Path, monkeypatch
    ):
        db = tmp_path / "cache.db"
        key = "consensus algorithms\n\nPaxos is a leader-based"
        mapping = {key: [1.0, 0.0, 0.0, 0.0]}
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))

        cache = PriorResearchCache()
        await cache.configure({"db_path": str(db)})
        await cache.append(
            query="consensus algorithms",
            synthesis_md="Paxos is a leader-based",
            sources_json="[]",
            bundle_id="bnd-chain-1",
        )

        async def q_embed(texts, **_):
            return [[0.9, 0.1, 0.0, 0.0]]

        _patch_embed(monkeypatch, q_embed)
        docs = await cache.search("consensus comparison")
        assert len(docs) == 1
        assert docs[0].metadata["bundle_id"] == "bnd-chain-1"
        assert "Paxos" in docs[0].content


class TestTwoFilesystemInstances:
    """Two filesystem instances contribute independently."""

    async def test_two_instances_independent(
        self, tmp_path: Path, monkeypatch
    ):
        notes_a = tmp_path / "a"
        notes_a.mkdir()
        (notes_a / "doc.md").write_text("# A\n\nalpha only")

        notes_b = tmp_path / "b"
        notes_b.mkdir()
        (notes_b / "doc.md").write_text("# B\n\nbeta only")

        mapping = {
            "# A\n\nalpha only": [1.0, 0.0, 0.0, 0.0],
            "# B\n\nbeta only": [0.0, 1.0, 0.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        pa = FilesystemPlugin()
        await pa.configure({
            "name": "inst_a", "path": str(notes_a),
            "index_dir": str(tmp_path / "idx_a"),
        })
        await pa.index()

        pb = FilesystemPlugin()
        await pb.configure({
            "name": "inst_b", "path": str(notes_b),
            "index_dir": str(tmp_path / "idx_b"),
        })
        await pb.index()

        async def q_embed_a(texts, **_):
            return [[0.9, 0.1, 0.0, 0.0]]

        _patch_embed(monkeypatch, q_embed_a)

        docs_a = await pa.search("alpha", limit=5)
        docs_b = await pb.search("alpha", limit=5)

        assert len(docs_a) == 1
        assert "a" in docs_a[0].metadata["path"].lower()
        assert len(docs_b) == 1
        assert "b" in docs_b[0].metadata["path"].lower()


class TestPluginCollisionOverwrite:
    """Last registration wins when same name is registered twice."""

    async def test_last_registration_wins(self):

        class FakeA:
            name = "collision"
            display_name = "A"
            requires_index = False

            async def configure(self, config):
                pass

            async def search(self, query, *, limit=10):
                return [Document(id="a1", content="from A")]

            async def index(self, *, force=False):
                return IndexStats(source_name="collision", documents_indexed=0,
                                  documents_skipped=0, elapsed_seconds=0.0)

            async def is_fresh(self):
                return True

            async def fetch(self, doc_id):
                return None

            async def health_check(self):
                return {"status": "ok"}

        class FakeB:
            name = "collision"
            display_name = "B"
            requires_index = False

            async def configure(self, config):
                pass

            async def search(self, query, *, limit=10):
                return [Document(id="b1", content="from B")]

            async def index(self, *, force=False):
                return IndexStats(source_name="collision", documents_indexed=0,
                                  documents_skipped=0, elapsed_seconds=0.0)

            async def is_fresh(self):
                return True

            async def fetch(self, doc_id):
                return None

            async def health_check(self):
                return {"status": "ok"}

        register(FakeA())
        register(FakeB())

        results = await run_plugin_sources("q", "collision")
        assert len(results) == 1
        assert "from B" in results[0].result.report_markdown


# ---------------------------------------------------------------------------
# 3. Chaos tests
# ---------------------------------------------------------------------------


class TestEmbedding500MidIndex:
    """Embedding endpoint 500 mid-index raises."""

    async def test_embed_500_raises_on_index(
        self, tmp_path: Path, monkeypatch
    ):
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "a.md").write_text("# A\n\nalpha")
        (notes / "b.md").write_text("# B\n\nbeta")

        call_count = 0

        async def flaky_embed(texts, **_):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("embed endpoint 500")
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        _patch_embed(monkeypatch, flaky_embed)
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure({
            "name": "flaky", "path": str(notes),
            "index_dir": str(tmp_path / "idx"),
        })
        with pytest.raises(RuntimeError, match="embed endpoint 500"):
            await p.index()


class TestPrxhub429:
    """PrxhubPlugin 429 -> surfaces error."""

    async def test_429_surfaces_error(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "rate limited"})

        _patch_httpx_prxhub(monkeypatch, handler)
        p = PrxhubPlugin()
        await p.configure({"api_url": "http://hub.test"})

        with pytest.raises(httpx.HTTPStatusError):
            await p.search("query")


class TestSynthesisMalformedJson:
    """Synthesis backend returns 500 -> BackendError."""

    async def test_backend_500_raises(self, monkeypatch):
        from parallect.backends.adapters import call_openai_compat_chat

        spec = BackendSpec(
            kind="custom", base_url="http://broken.test/v1",
            api_key="k", model="m", api_key_env="",
        )

        fake_request = httpx.Request("POST", "http://broken.test/v1/chat/completions")
        resp = httpx.Response(500, text="Internal Server Error", request=fake_request)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            with pytest.raises(BackendError):
                await call_openai_compat_chat(spec, "prompt", "system")


class TestSqliteStoreCorrupted:
    """Corrupted SQLite store -> detect error."""

    async def test_corrupted_db_raises(self, tmp_path: Path, monkeypatch):
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "a.md").write_text("# A\n\nalpha")

        _patch_embed(
            monkeypatch,
            _fake_embed_factory({"# A\n\nalpha": [1.0, 0.0, 0.0, 0.0]}),
        )
        _patch_embed_dimensions(monkeypatch, 4)

        idx_dir = tmp_path / "idx"
        idx_dir.mkdir(parents=True)
        (idx_dir / "index.db").write_bytes(b"NOT A SQLITE DATABASE " * 100)

        p = FilesystemPlugin()
        await p.configure({
            "name": "corrupt", "path": str(notes),
            "index_dir": str(idx_dir),
        })

        with pytest.raises((sqlite3.DatabaseError, RuntimeError)):
            await p.index()


class TestEmbeddingWrongDimBatch:
    """Embedding returns wrong number of vectors -> mismatch detected."""

    async def test_wrong_count_in_batch_raises(
        self, tmp_path: Path, monkeypatch
    ):
        notes = tmp_path / "notes"
        notes.mkdir()

        content = "# Big\n\n" + "\n\n".join(["paragraph " * 200 for _ in range(5)])
        (notes / "big.md").write_text(content)

        async def miscount_embed(texts, **_):
            return [[0.1, 0.2, 0.3, 0.4]]

        _patch_embed(monkeypatch, miscount_embed)
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure({
            "name": "mismatch", "path": str(notes),
            "index_dir": str(tmp_path / "idx"),
        })

        with pytest.raises(RuntimeError, match="embed.*returned.*vectors"):
            await p.index()


class TestUnconfiguredSourceError:
    """--sources references unconfigured plugin -> clear error."""

    async def test_unknown_source_raises(self):
        with pytest.raises(PluginError, match="unknown data source"):
            await run_plugin_sources("q", "nonexistent-plugin")

    async def test_filesystem_without_path_raises(self, monkeypatch):
        register(FilesystemPlugin())
        monkeypatch.setattr(
            "parallect.orchestrator.plugin_sources._extract_plugin_configs",
            lambda _: {},
        )
        with pytest.raises(PluginError, match="has no `path` configured"):
            await run_plugin_sources("q", "filesystem")


class TestPluginEntryPointImportError:
    """Plugin entry point raises during import -> logged, not crashed."""

    def test_entry_point_error_logged_not_crashed(self, monkeypatch):
        import parallect.plugins.data_sources as ds_mod

        ds_mod._ENTRY_POINTS_LOADED = False

        class FakeEP:
            name = "broken-plugin"

            def load(self):
                raise ImportError("simulated broken plugin")

        def fake_entry_points(group=""):
            return [FakeEP()]

        monkeypatch.setattr(
            "parallect.plugins.data_sources.entry_points", fake_entry_points
        )

        ds_mod._load_entry_points()
        assert "broken-plugin" not in ds_mod._REGISTRY


class TestSynthesisLlmTimeout:
    """LLM synthesis timeout -> graceful error propagation."""

    async def test_timeout_propagates(self, monkeypatch):
        from parallect.synthesis.llm import synthesize

        results = [
            ProviderResult(
                provider="test", status="completed",
                report_markdown="test report",
            )
        ]

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mpost:
            mpost.side_effect = httpx.ReadTimeout("timeout")
            with pytest.raises(httpx.ReadTimeout):
                await synthesize(
                    "test query", results, model="custom",
                    base_url="http://slow.test/v1",
                )


class TestFanOutTimeout:
    """Plugin fan-out with timeout -> error result, not crash."""

    async def test_slow_plugin_times_out(self):
        import asyncio

        class SlowPlugin:
            name = "slow"
            display_name = "slow"
            requires_index = False

            async def configure(self, config):
                pass

            async def search(self, query, *, limit=10):
                await asyncio.sleep(10)
                return []

            async def index(self, *, force=False):
                return IndexStats(source_name="slow", documents_indexed=0,
                                  documents_skipped=0, elapsed_seconds=0.0)

            async def is_fresh(self):
                return True

            async def fetch(self, doc_id):
                return None

            async def health_check(self):
                return {"status": "ok"}

        register(SlowPlugin())
        results = await run_plugin_sources("q", "slow", timeout=0.1)
        assert len(results) == 1
        assert results[0].error is not None
        assert "timed out" in results[0].error


class TestPriorResearchCacheForceClears:
    """PriorResearchCache index --force clears all entries."""

    async def test_force_clears_cache(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "cache.db"

        async def fake_embed(texts, **_):
            return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

        import parallect.embeddings as emb_mod
        monkeypatch.setattr(emb_mod, "embed", fake_embed)

        cache = PriorResearchCache()
        await cache.configure({"db_path": str(db)})
        await cache.append(
            query="q1", synthesis_md="s1",
            sources_json="[]", bundle_id="b1",
        )
        h1 = await cache.health_check()
        assert h1["total_runs"] == 1

        await cache.index(force=True)
        h2 = await cache.health_check()
        assert h2["total_runs"] == 0


class TestFilesystemFreshness:
    """FilesystemPlugin.is_fresh transitions correctly."""

    async def test_fresh_after_index_stale_after_edit(
        self, tmp_path: Path, monkeypatch
    ):
        import os
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "a.md").write_text("# A\n\nalpha")

        _patch_embed(
            monkeypatch,
            _fake_embed_factory({"# A\n\nalpha": [1.0, 0.0, 0.0, 0.0]}),
        )
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure({
            "name": "fresh", "path": str(notes),
            "index_dir": str(tmp_path / "idx"),
        })

        # Not fresh before first index (no DB)
        assert await p.is_fresh() is False

        await p.index()
        # Fresh right after index
        assert await p.is_fresh() is True

        # Edit a file and bump mtime into the future
        import time
        time.sleep(0.01)
        (notes / "a.md").write_text("# A\n\nalpha modified")
        future = time.time() + 100
        os.utime(notes / "a.md", (future, future))

        # Now stale
        assert await p.is_fresh() is False


class TestPrxhubFetchById:
    """PrxhubPlugin fetch by bundle: and claim: ids."""

    async def test_fetch_bundle_returns_doc(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            if "/api/bundles/b123" in str(request.url):
                return httpx.Response(200, json={
                    "id": "b123", "title": "Test Bundle",
                    "query": "test", "summary": "A test bundle",
                })
            return httpx.Response(404)

        _patch_httpx_prxhub(monkeypatch, handler)
        p = PrxhubPlugin()
        await p.configure({"api_url": "http://hub.test"})

        doc = await p.fetch("bundle:b123")
        assert doc is not None
        assert doc.id == "bundle:b123"
        assert "Test Bundle" in (doc.title or "")

    async def test_fetch_unknown_returns_none(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        _patch_httpx_prxhub(monkeypatch, handler)
        p = PrxhubPlugin()
        await p.configure({"api_url": "http://hub.test"})

        assert await p.fetch("bundle:missing") is None
        assert await p.fetch("bad-format") is None
