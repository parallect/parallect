"""ObsidianPlugin tests.

Covers frontmatter parsing, wikilink resolution, backlink graph,
Smart Connections detection, graph-aware ranking, and end-to-end
index+search.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parallect.plugins.data_sources.obsidian import (
    ObsidianPlugin,
    extract_wikilinks,
    parse_frontmatter,
    strip_frontmatter,
)


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


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestFrontmatter:
    def test_array_tags(self):
        text = "---\ntags:\n  - python\n  - ai\n---\nbody"
        fm = parse_frontmatter(text)
        assert fm is not None
        assert fm["tags"] == ["python", "ai"]

    def test_string_tags(self):
        text = "---\ntags: single-tag\n---\nbody"
        fm = parse_frontmatter(text)
        assert fm is not None
        assert fm["tags"] == "single-tag"

    def test_aliases(self):
        text = "---\naliases:\n  - AI\n  - Artificial Intelligence\n---\nbody"
        fm = parse_frontmatter(text)
        assert fm is not None
        assert fm["aliases"] == ["AI", "Artificial Intelligence"]

    def test_missing_frontmatter(self):
        assert parse_frontmatter("no frontmatter here") is None

    def test_malformed_yaml(self):
        text = "---\n: : : bad\n---\nbody"
        fm = parse_frontmatter(text)
        assert fm is None or isinstance(fm, dict)

    def test_scalar_keys_prefixed(self):
        text = "---\nauthor: Alice\ndate: 2025-01-01\ntags:\n  - t\n---\nbody"
        fm = parse_frontmatter(text)
        assert fm is not None
        assert fm["author"] == "Alice"

    def test_strip_frontmatter(self):
        text = "---\ntags:\n  - a\n---\nbody content"
        stripped = strip_frontmatter(text)
        assert stripped == "body content"

    def test_strip_no_frontmatter(self):
        text = "just body"
        assert strip_frontmatter(text) == text


# ---------------------------------------------------------------------------
# Wikilinks
# ---------------------------------------------------------------------------


class TestWikilinks:
    def test_simple_link(self):
        links = extract_wikilinks("See [[Target Note]] for details.")
        assert links == ["Target Note"]

    def test_piped_link(self):
        links = extract_wikilinks("See [[Target|display text]] here.")
        assert links == ["Target"]

    def test_nested_path(self):
        links = extract_wikilinks("Link to [[subfolder/Note]].")
        assert links == ["subfolder/Note"]

    def test_ambiguous_dedup(self):
        links = extract_wikilinks("[[A]] and [[A]] again")
        assert links == ["A"]

    def test_broken_link(self):
        links = extract_wikilinks("[[]] empty target")
        assert links == []

    def test_multiple_links(self):
        text = "[[A]] then [[B|bee]] and [[C]]"
        links = extract_wikilinks(text)
        assert links == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Backlink graph
# ---------------------------------------------------------------------------


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    (v / "a.md").write_text("---\ntags:\n  - core\n---\n# A\n\nLinks to [[b]] and [[c]]")
    (v / "b.md").write_text("# B\n\nLinks to [[a]]")
    (v / "c.md").write_text("# C\n\nLinks to [[a]] and [[b]]")
    sub = v / "notes"
    sub.mkdir()
    (sub / "d.md").write_text("# D\n\nLinks to [[a]]")
    (v / "orphan.md").write_text("# Orphan\n\nNo links here")
    return v


class TestBacklinkGraph:
    async def test_build_and_invert(self, vault_dir: Path, tmp_path: Path, monkeypatch):
        mapping: dict[str, list[float]] = {}
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        p = ObsidianPlugin()
        await p.configure({
            "name": "vault",
            "path": str(vault_dir),
            "index_dir": str(tmp_path / "idx"),
        })
        p._build_stem_index()
        p._build_link_graph()

        assert "b.md" in p._links_out["a.md"]
        assert "c.md" in p._links_out["a.md"]
        assert "a.md" in p._links_out["b.md"]
        assert "b.md" in p._links_in["a.md"]
        assert "c.md" in p._links_in["a.md"]
        assert "notes/d.md" in p._links_in["a.md"]
        assert "c.md" in p._links_in["b.md"]


# ---------------------------------------------------------------------------
# Smart Connections
# ---------------------------------------------------------------------------


class TestSmartConnections:
    async def test_detect_and_reuse(self, tmp_path: Path, monkeypatch):
        v = tmp_path / "sc_vault"
        v.mkdir()
        (v / "note.md").write_text("# Note\n\nsome text")
        sc_dir = v / ".smart-env" / "multi"
        sc_dir.mkdir(parents=True)
        entry = {"key": "test", "vec": [0.1, 0.2, 0.3, 0.4]}
        (sc_dir / "embeddings.ajson").write_text(json.dumps(entry) + "\n")

        _patch_embed(monkeypatch, _fake_embed_factory({}))
        _patch_embed_dimensions(monkeypatch, 4)

        p = ObsidianPlugin()
        await p.configure({
            "name": "sc",
            "path": str(v),
            "index_dir": str(tmp_path / "idx"),
            "reuse_smart_connections": True,
        })
        p._try_reuse_smart_connections()

    async def test_dim_mismatch_skips(self, tmp_path: Path, monkeypatch):
        v = tmp_path / "sc_vault2"
        v.mkdir()
        (v / "note.md").write_text("# Note\n\nsome text")
        sc_dir = v / ".smart-env" / "multi"
        sc_dir.mkdir(parents=True)
        entry = {"key": "test", "vec": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
        (sc_dir / "embeddings.ajson").write_text(json.dumps(entry) + "\n")

        mapping = {"# Note\n\nsome text": [1.0, 0.0, 0.0, 0.0]}
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        p = ObsidianPlugin()
        await p.configure({
            "name": "sc2",
            "path": str(v),
            "index_dir": str(tmp_path / "idx"),
            "reuse_smart_connections": True,
        })
        await p.index()
        # SC has dim=8, index has dim=4; should warn but not raise
        p._try_reuse_smart_connections()

    async def test_no_sc_dir(self, tmp_path: Path, monkeypatch):
        v = tmp_path / "nosc"
        v.mkdir()

        _patch_embed(monkeypatch, _fake_embed_factory({}))
        _patch_embed_dimensions(monkeypatch, 4)

        p = ObsidianPlugin()
        await p.configure({
            "name": "nosc",
            "path": str(v),
            "index_dir": str(tmp_path / "idx"),
            "reuse_smart_connections": True,
        })
        p._try_reuse_smart_connections()

    async def test_empty_ajson(self, tmp_path: Path, monkeypatch):
        v = tmp_path / "sc_empty"
        v.mkdir()
        sc_dir = v / ".smart-env" / "multi"
        sc_dir.mkdir(parents=True)
        (sc_dir / "embeddings.ajson").write_text("")

        _patch_embed(monkeypatch, _fake_embed_factory({}))
        _patch_embed_dimensions(monkeypatch, 4)

        p = ObsidianPlugin()
        await p.configure({
            "name": "sce",
            "path": str(v),
            "index_dir": str(tmp_path / "idx"),
            "reuse_smart_connections": True,
        })
        p._try_reuse_smart_connections()


# ---------------------------------------------------------------------------
# Graph-aware ranking
# ---------------------------------------------------------------------------


class TestGraphRanking:
    async def test_backlinks_boost_score(self, vault_dir: Path, tmp_path: Path, monkeypatch):
        # a.md (stripped) = "# A\n\nLinks to [[b]] and [[c]]"
        # b.md = "# B\n\nLinks to [[a]]"
        # c.md = "# C\n\nLinks to [[a]] and [[b]]"
        # d.md = "# D\n\nLinks to [[a]]"
        # orphan.md = "# Orphan\n\nNo links here"
        # a is linked by b, c, d (3 backlinks)
        mapping = {
            "# A\n\nLinks to [[b]] and [[c]]": [0.9, 0.0, 0.0, 0.0],
            "# B\n\nLinks to [[a]]": [0.8, 0.1, 0.0, 0.0],
            "# C\n\nLinks to [[a]] and [[b]]": [0.7, 0.2, 0.0, 0.0],
            "# D\n\nLinks to [[a]]": [0.6, 0.3, 0.0, 0.0],
            "# Orphan\n\nNo links here": [0.5, 0.4, 0.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        p = ObsidianPlugin()
        await p.configure({
            "name": "rank",
            "path": str(vault_dir),
            "index_dir": str(tmp_path / "idx"),
            "graph_rank_boost": 0.05,
        })
        await p.index()

        async def query_embed(texts, **_):
            return [[0.95, 0.05, 0.0, 0.0]]

        monkeypatch.setattr("parallect.embeddings.embed", query_embed)

        docs = await p.search("test", limit=5)
        assert len(docs) >= 1
        assert "a.md" in docs[0].metadata["path"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestObsidianConfigure:
    async def test_default_name(self, tmp_path: Path):
        p = ObsidianPlugin()
        await p.configure({
            "path": str(tmp_path),
            "index_dir": str(tmp_path / "idx"),
        })
        assert p.name == "obsidian"
        assert p.display_name == "obsidian"

    async def test_custom_name(self, tmp_path: Path):
        p = ObsidianPlugin()
        await p.configure({
            "name": "vault",
            "path": str(tmp_path),
            "index_dir": str(tmp_path / "idx"),
        })
        assert p.name == "vault"
        assert p.display_name == "obsidian:vault"

    async def test_config_flags(self, tmp_path: Path):
        p = ObsidianPlugin()
        await p.configure({
            "path": str(tmp_path),
            "index_dir": str(tmp_path / "idx"),
            "parse_frontmatter": False,
            "resolve_wikilinks": False,
            "reuse_smart_connections": True,
            "graph_rank_boost": 0.1,
        })
        assert p._parse_frontmatter is False
        assert p._resolve_wikilinks is False
        assert p._reuse_smart_connections is True
        assert p._graph_rank_boost == 0.1


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    async def test_index_and_search(self, vault_dir: Path, tmp_path: Path, monkeypatch):
        # After frontmatter stripping, a.md content = "# A\n\nLinks to [[b]] and [[c]]"
        mapping = {
            "# A\n\nLinks to [[b]] and [[c]]": [1.0, 0.0, 0.0, 0.0],
            "# B\n\nLinks to [[a]]": [0.0, 1.0, 0.0, 0.0],
            "# C\n\nLinks to [[a]] and [[b]]": [0.0, 0.0, 1.0, 0.0],
            "# D\n\nLinks to [[a]]": [0.0, 0.0, 0.0, 1.0],
            "# Orphan\n\nNo links here": [0.5, 0.5, 0.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        p = ObsidianPlugin()
        await p.configure({
            "name": "e2e",
            "path": str(vault_dir),
            "index_dir": str(tmp_path / "idx"),
        })
        stats = await p.index()
        assert stats.documents_indexed == 5
        assert stats.source_name == "e2e"

        async def query_embed(texts, **_):
            return [[0.9, 0.1, 0.0, 0.0]]

        monkeypatch.setattr("parallect.embeddings.embed", query_embed)

        docs = await p.search("alpha", limit=5)
        assert len(docs) == 5
        assert "a.md" in docs[0].metadata["path"]
        assert "frontmatter_tags" in docs[0].metadata
        assert docs[0].metadata["frontmatter_tags"] == ["core"]
        assert "links_out" in docs[0].metadata
        assert "links_in" in docs[0].metadata
        assert len(docs[0].metadata["links_in"]) >= 2  # b, c, d link to a

    async def test_search_empty_index(self, tmp_path: Path, monkeypatch):
        v = tmp_path / "empty"
        v.mkdir()

        _patch_embed(monkeypatch, _fake_embed_factory({}))
        _patch_embed_dimensions(monkeypatch, 4)

        p = ObsidianPlugin()
        await p.configure({
            "name": "empty",
            "path": str(v),
            "index_dir": str(tmp_path / "idx"),
        })
        docs = await p.search("anything")
        assert docs == []

    async def test_frontmatter_stripped_from_embeddings(
        self, tmp_path: Path, monkeypatch
    ):
        v = tmp_path / "fm_strip"
        v.mkdir()
        (v / "note.md").write_text("---\ntags:\n  - test\n---\n# Hello\n\nWorld")

        embedded_texts: list[str] = []

        async def capture_embed(texts, **_kwargs):
            embedded_texts.extend(texts)
            return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

        _patch_embed(monkeypatch, capture_embed)
        _patch_embed_dimensions(monkeypatch, 4)

        p = ObsidianPlugin()
        await p.configure({
            "name": "strip",
            "path": str(v),
            "index_dir": str(tmp_path / "idx"),
        })
        await p.index()

        assert len(embedded_texts) == 1
        assert "---" not in embedded_texts[0]
        assert "# Hello" in embedded_texts[0]
