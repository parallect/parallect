"""Wave-2: End-to-end smoke tests.

Gated behind PARALLECT_INTEGRATION=1. These tests exercise the full
pipeline with mocked LLM/embedding backends but real plugin I/O.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _reset_registry():
    from parallect.plugins.data_sources import reset_registry
    reset_registry()
    yield
    reset_registry()


@pytest.fixture(autouse=True)
def _reset_embed_cache():
    import parallect.embeddings as emb_mod
    emb_mod._reset_caches()
    yield
    emb_mod._reset_caches()


def _fake_embed_factory(default_dim: int = 4):
    async def fake_embed(texts, **_kwargs):
        vectors = []
        for t in texts:
            h = hash(t) % 10000
            vec = [(h % 7) / 7.0, (h % 11) / 11.0,
                   (h % 13) / 13.0, (h % 17) / 17.0]
            vectors.append(vec[:default_dim])
        return vectors
    return fake_embed


def _patch_embed(monkeypatch, dim: int = 4):
    import parallect.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "embed", _fake_embed_factory(dim))

    async def fake_dims(**_kwargs):
        return dim
    monkeypatch.setattr(emb_mod, "embed_dimensions", fake_dims)


class TestFullFilesystemResearchSmoke:
    async def test_filesystem_index_search_roundtrip(self, tmp_path: Path, monkeypatch):
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "consensus.md").write_text(
            "# Consensus Algorithms\n\nPaxos and Raft are the two most "
            "widely studied consensus algorithms. Paxos requires 2f+1 nodes."
        )
        (notes / "coffee.md").write_text(
            "# Coffee\n\nArabica beans are grown at higher altitudes."
        )

        _patch_embed(monkeypatch)

        from parallect.plugins.data_sources.filesystem import FilesystemPlugin

        plugin = FilesystemPlugin()
        await plugin.configure({
            "name": "smoke-notes", "path": str(notes),
            "index_dir": str(tmp_path / "idx"),
        })

        stats = await plugin.index()
        assert stats.documents_indexed >= 2
        assert Path(stats.index_path).exists()

        docs = await plugin.search("consensus", limit=5)
        assert len(docs) >= 1
        for d in docs:
            assert d.id
            assert d.content
            assert d.metadata.get("kind") == "filesystem"

        health = await plugin.health_check()
        assert health["status"] == "ok"
        assert health["file_count"] >= 2


class TestPluginsStatusSmoke:
    async def test_index_then_health_shows_ok(self, tmp_path: Path, monkeypatch):
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "a.md").write_text("# A\n\nalpha content")

        _patch_embed(monkeypatch)

        from parallect.plugins.data_sources.filesystem import FilesystemPlugin

        plugin = FilesystemPlugin()
        await plugin.configure({
            "name": "status-test", "path": str(notes),
            "index_dir": str(tmp_path / "idx"),
        })

        h1 = await plugin.health_check()
        assert h1["status"] == "not_indexed"

        await plugin.index()
        h2 = await plugin.health_check()
        assert h2["status"] == "ok"
        assert h2["chunk_count"] >= 1


class TestPriorResearchChainSmoke:
    async def test_second_run_finds_first(self, tmp_path: Path, monkeypatch):
        _patch_embed(monkeypatch)

        from parallect.plugins.data_sources.prior_research import PriorResearchCache

        db_path = tmp_path / "cache.db"
        cache = PriorResearchCache()
        await cache.configure({"db_path": str(db_path)})

        await cache.append(
            query="consensus algorithms",
            synthesis_md="Paxos and Raft are consensus algorithms.",
            sources_json='[{"provider": "perplexity"}]',
            bundle_id="run-1",
        )

        docs = await cache.search("consensus comparison", limit=5)
        assert len(docs) >= 1
        assert docs[0].metadata["bundle_id"] == "run-1"
        assert "Paxos" in docs[0].content

        full = await cache.fetch("prior:run-1")
        assert full is not None
        assert full.metadata["sources_json"] == '[{"provider": "perplexity"}]'

        await cache.append(
            query="consensus comparison",
            synthesis_md="Raft simplifies Paxos by using an explicit leader.",
            sources_json='[{"provider": "gemini"}]',
            bundle_id="run-2",
        )

        docs2 = await cache.search("consensus", limit=10)
        bundle_ids = {d.metadata["bundle_id"] for d in docs2}
        assert "run-1" in bundle_ids
        assert "run-2" in bundle_ids
