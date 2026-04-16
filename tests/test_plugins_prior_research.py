"""PriorResearchCache tests — append + search roundtrip."""

from __future__ import annotations

from pathlib import Path

import pytest

from parallect.plugins.data_sources.prior_research import PriorResearchCache


def _fake_embed_factory(mapping: dict[str, list[float]], default_dim: int = 4):
    async def fake_embed(texts, **_kwargs):
        return [mapping.get(t, [0.0] * default_dim) for t in texts]

    return fake_embed


def _patch_embed(monkeypatch, fake):
    import parallect.embeddings as emb_mod

    monkeypatch.setattr(emb_mod, "embed", fake)


class TestAppendSearchRoundtrip:
    async def test_append_then_search(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "cache.db"
        # Known mapping: the "past run" embedding will match the query embedding.
        key = "first query\n\nfirst synthesis body"
        mapping = {key: [1.0, 0.0, 0.0, 0.0]}
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))

        p = PriorResearchCache()
        await p.configure({"db_path": str(db)})

        await p.append(
            query="first query",
            synthesis_md="first synthesis body",
            sources_json="[]",
            bundle_id="bnd-1",
        )

        # Query with a highly-similar vector
        async def query_embed(texts, **_):
            return [[0.9, 0.1, 0.0, 0.0]]

        monkeypatch.setattr("parallect.embeddings.embed", query_embed)
        docs = await p.search("first query variation")
        assert len(docs) == 1
        assert docs[0].metadata["bundle_id"] == "bnd-1"
        assert docs[0].metadata["past_query"] == "first query"
        assert docs[0].content == "first synthesis body"

    async def test_multiple_runs_ranked_by_similarity(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "cache.db"
        mapping = {
            "q1\n\ns1": [1.0, 0.0, 0.0, 0.0],
            "q2\n\ns2": [0.0, 1.0, 0.0, 0.0],
            "q3\n\ns3": [0.0, 0.0, 1.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))

        p = PriorResearchCache()
        await p.configure({"db_path": str(db)})
        for n in (1, 2, 3):
            await p.append(
                query=f"q{n}",
                synthesis_md=f"s{n}",
                sources_json="[]",
                bundle_id=f"bnd-{n}",
            )

        # Query that's closest to q2's vector
        async def query_embed(texts, **_):
            return [[0.0, 0.95, 0.05, 0.0]]

        monkeypatch.setattr("parallect.embeddings.embed", query_embed)
        docs = await p.search("anything", limit=3)
        assert docs[0].metadata["bundle_id"] == "bnd-2"

    async def test_fetch_by_id(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "cache.db"
        _patch_embed(monkeypatch, _fake_embed_factory({"q\n\nbody": [1.0, 0.0, 0.0, 0.0]}))

        p = PriorResearchCache()
        await p.configure({"db_path": str(db)})
        await p.append(
            query="q", synthesis_md="body", sources_json='{"x":1}', bundle_id="b1"
        )
        doc = await p.fetch("prior:b1")
        assert doc is not None
        assert doc.content == "body"
        assert doc.metadata["sources_json"] == '{"x":1}'

    async def test_fetch_missing(self, tmp_path: Path):
        p = PriorResearchCache()
        await p.configure({"db_path": str(tmp_path / "empty.db")})
        assert await p.fetch("prior:nope") is None
        assert await p.fetch("bad-format") is None


class TestHealthCheck:
    async def test_empty_cache(self, tmp_path: Path):
        p = PriorResearchCache()
        await p.configure({"db_path": str(tmp_path / "c.db")})
        h = await p.health_check()
        assert h["status"] == "ok"
        assert h["total_runs"] == 0

    async def test_counts_runs(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "c.db"
        _patch_embed(monkeypatch, _fake_embed_factory({}))
        p = PriorResearchCache()
        await p.configure({"db_path": str(db)})
        for n in range(3):
            await p.append(
                query=f"q{n}", synthesis_md="body", sources_json="[]", bundle_id=f"b{n}"
            )
        h = await p.health_check()
        assert h["total_runs"] == 3
        assert h["latest_timestamp"] is not None


class TestIndexNoop:
    async def test_index_reports_count(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "c.db"
        _patch_embed(monkeypatch, _fake_embed_factory({}))
        p = PriorResearchCache()
        await p.configure({"db_path": str(db)})
        await p.append(
            query="q", synthesis_md="body", sources_json="[]", bundle_id="b1"
        )
        stats = await p.index()
        assert stats.documents_skipped == 1

    async def test_force_clears(self, tmp_path: Path, monkeypatch):
        db = tmp_path / "c.db"
        _patch_embed(monkeypatch, _fake_embed_factory({}))
        p = PriorResearchCache()
        await p.configure({"db_path": str(db)})
        await p.append(
            query="q", synthesis_md="body", sources_json="[]", bundle_id="b1"
        )
        stats = await p.index(force=True)
        # After force, no records remain
        h = await p.health_check()
        assert h["total_runs"] == 0

    async def test_is_fresh_always_true(self):
        p = PriorResearchCache()
        assert await p.is_fresh() is True
