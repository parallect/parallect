"""FilesystemPlugin tests.

All tests mock ``parallect.embeddings.embed`` so they don't need a real
embeddings backend. The mocks return deterministic small vectors so
cosine-similarity ranking is testable.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from parallect.plugins.data_sources.filesystem import (
    FilesystemPlugin,
    _chunk_text,
    _extract_title,
)


def _fake_embed_factory(mapping: dict[str, list[float]], default_dim: int = 4):
    """Return an async embed() that returns mapped vectors (or zeros)."""

    async def fake_embed(texts, **_kwargs):
        return [mapping.get(t, [0.0] * default_dim) for t in texts]

    return fake_embed


def _patch_embed(monkeypatch, fake):
    """Patch both `parallect.embeddings.embed` and the plugin-local import."""
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
# Chunking
# ---------------------------------------------------------------------------


class TestChunking:
    def test_empty_returns_empty(self):
        assert _chunk_text("") == []
        assert _chunk_text("   \n  ") == []

    def test_single_paragraph(self):
        chunks = _chunk_text("hello world")
        assert chunks == ["hello world"]

    def test_multi_paragraph_merged(self):
        text = "p1 short.\n\np2 also short.\n\np3."
        chunks = _chunk_text(text)
        assert len(chunks) == 1
        assert "p1" in chunks[0] and "p3" in chunks[0]

    def test_long_content_splits(self):
        para = "x" * 1500
        text = "\n\n".join([para, para, para])
        chunks = _chunk_text(text)
        # Three ~1500-char paragraphs, target 2000 -> should split.
        assert len(chunks) >= 2

    def test_oversized_paragraph_sentence_split(self):
        giant = ". ".join([f"Sentence number {i}" for i in range(200)]) + "."
        chunks = _chunk_text(giant)
        # Giant single paragraph must split on sentences.
        assert len(chunks) >= 2


class TestTitleExtraction:
    def test_first_h1_heading(self):
        content = "# The Title\n\nbody\n\n# another"
        assert _extract_title(content, "fallback.md") == "The Title"

    def test_no_heading_uses_fallback(self):
        assert _extract_title("just text", "notes.md") == "notes.md"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "notes"
    d.mkdir()
    (d / "a.md").write_text("# A\n\nalpha content")
    (d / "b.md").write_text("# B\n\nbeta content")
    # Excluded file
    excluded = d / ".git"
    excluded.mkdir()
    (excluded / "config").write_text("ignore me")
    return d


class TestFilesystemConfigure:
    async def test_requires_path(self, tmp_path: Path):
        p = FilesystemPlugin()
        with pytest.raises(ValueError, match="requires a `path`"):
            await p.configure({})

    async def test_nonexistent_path(self, tmp_path: Path):
        p = FilesystemPlugin()
        with pytest.raises(ValueError, match="does not exist"):
            await p.configure({"path": str(tmp_path / "nope")})

    async def test_instance_sets_name_and_display(self, tmp_path: Path):
        p = FilesystemPlugin()
        await p.configure(
            {"name": "notes", "path": str(tmp_path), "index_dir": str(tmp_path / "idx")}
        )
        assert p.name == "notes"
        assert p.display_name == "filesystem:notes"

    async def test_default_instance_keeps_filesystem_name(self, tmp_path: Path):
        p = FilesystemPlugin()
        await p.configure({"path": str(tmp_path), "index_dir": str(tmp_path / "idx")})
        assert p.name == "filesystem"


class TestFilesystemIndex:
    async def test_first_index_embeds_and_stores(
        self, notes_dir: Path, tmp_path: Path, monkeypatch
    ):
        mapping = {
            "# A\n\nalpha content": [1.0, 0.0, 0.0, 0.0],
            "# B\n\nbeta content": [0.0, 1.0, 0.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure(
            {
                "name": "notes",
                "path": str(notes_dir),
                "index_dir": str(tmp_path / "idx"),
            }
        )

        stats = await p.index()
        assert stats.documents_indexed == 2
        assert stats.documents_skipped == 0
        assert Path(stats.index_path).exists()

    async def test_skip_unchanged_files(
        self, notes_dir: Path, tmp_path: Path, monkeypatch
    ):
        mapping = {
            "# A\n\nalpha content": [1.0, 0.0, 0.0, 0.0],
            "# B\n\nbeta content": [0.0, 1.0, 0.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure(
            {
                "name": "notes",
                "path": str(notes_dir),
                "index_dir": str(tmp_path / "idx"),
            }
        )
        await p.index()
        # Second index run — nothing changed
        stats2 = await p.index()
        assert stats2.documents_indexed == 0
        assert stats2.documents_skipped == 2

    async def test_reindex_after_content_change(
        self, notes_dir: Path, tmp_path: Path, monkeypatch
    ):
        mapping = {
            "# A\n\nalpha content": [1.0, 0.0, 0.0, 0.0],
            "# B\n\nbeta content": [0.0, 1.0, 0.0, 0.0],
        }
        embed = _fake_embed_factory(mapping)
        _patch_embed(monkeypatch, embed)
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure(
            {
                "name": "notes",
                "path": str(notes_dir),
                "index_dir": str(tmp_path / "idx"),
            }
        )
        await p.index()

        # Edit a file + bump mtime
        (notes_dir / "a.md").write_text("# A\n\nalpha new")
        future = (notes_dir / "a.md").stat().st_mtime + 10
        os.utime(notes_dir / "a.md", (future, future))

        mapping["# A\n\nalpha new"] = [0.5, 0.5, 0.0, 0.0]
        stats = await p.index()
        assert stats.documents_indexed == 1
        assert stats.documents_skipped == 1

    async def test_dimension_mismatch_blocks_without_force(
        self, notes_dir: Path, tmp_path: Path, monkeypatch
    ):
        # First index with 4-dim vectors
        mapping = {
            "# A\n\nalpha content": [1.0, 0.0, 0.0, 0.0],
            "# B\n\nbeta content": [0.0, 1.0, 0.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping, default_dim=4))
        _patch_embed_dimensions(monkeypatch, 4)

        # Configure settings with a model/dim so metadata is persisted.
        from parallect.config_mod.settings import ParallectSettings

        monkeypatch.setattr(
            ParallectSettings, "load", classmethod(lambda cls: cls(embeddings_model="model-a"))
        )

        p = FilesystemPlugin()
        await p.configure(
            {"name": "notes", "path": str(notes_dir), "index_dir": str(tmp_path / "idx")}
        )
        await p.index()

        # Now pretend dimensions changed to 8
        monkeypatch.setattr(ParallectSettings, "load", classmethod(
            lambda cls: cls(embeddings_model="model-b")
        ))
        _patch_embed_dimensions(monkeypatch, 8)

        with pytest.raises(RuntimeError, match="embedding model or dimensions changed"):
            await p.index()

        # --force proceeds
        _patch_embed(monkeypatch, _fake_embed_factory({}, default_dim=8))
        _patch_embed_dimensions(monkeypatch, 8)
        stats = await p.index(force=True)
        assert stats.documents_indexed >= 0  # didn't raise

    async def test_search_roundtrip(
        self, notes_dir: Path, tmp_path: Path, monkeypatch
    ):
        mapping = {
            "# A\n\nalpha content": [1.0, 0.0, 0.0, 0.0],
            "# B\n\nbeta content": [0.0, 1.0, 0.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure(
            {"name": "notes", "path": str(notes_dir), "index_dir": str(tmp_path / "idx")}
        )
        await p.index()

        # Query matching the "alpha" vector should rank a.md first.
        async def query_embed(texts, **_):
            return [[0.9, 0.1, 0.0, 0.0]]

        monkeypatch.setattr("parallect.embeddings.embed", query_embed)

        docs = await p.search("alpha", limit=5)
        assert len(docs) == 2
        assert "a.md" in docs[0].metadata["path"]
        assert docs[0].score is not None and docs[0].score > (docs[1].score or -1)

    async def test_excludes_hidden_dirs(
        self, notes_dir: Path, tmp_path: Path, monkeypatch
    ):
        """Files under excluded patterns (.git/) aren't indexed."""
        # Add a real md under .git/
        (notes_dir / ".git" / "leaked.md").write_text("# leaked\nshould not index")

        mapping = {
            "# A\n\nalpha content": [1.0, 0.0, 0.0, 0.0],
            "# B\n\nbeta content": [0.0, 1.0, 0.0, 0.0],
            "# leaked\n\nshould not index": [0.0, 0.0, 1.0, 0.0],
        }
        _patch_embed(monkeypatch, _fake_embed_factory(mapping))
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure(
            {"name": "notes", "path": str(notes_dir), "index_dir": str(tmp_path / "idx")}
        )
        stats = await p.index()
        # Only a.md + b.md should be indexed (leaked.md excluded).
        assert stats.documents_indexed == 2

    async def test_max_file_kb_skips_large_files(
        self, tmp_path: Path, monkeypatch
    ):
        d = tmp_path / "big"
        d.mkdir()
        (d / "ok.md").write_text("# ok\nsmall")
        (d / "huge.md").write_text("x" * (600 * 1024))  # 600 KB -> skipped at 500

        _patch_embed(monkeypatch, _fake_embed_factory({"# ok\n\nsmall": [1.0, 0.0, 0.0, 0.0]}))
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure(
            {"name": "big", "path": str(d), "index_dir": str(tmp_path / "idx"), "max_file_kb": 500}
        )
        stats = await p.index()
        assert stats.documents_indexed == 1
        assert stats.documents_skipped >= 1


class TestFilesystemHealth:
    async def test_unconfigured(self):
        p = FilesystemPlugin()
        h = await p.health_check()
        assert h["status"] == "unconfigured"

    async def test_not_indexed(self, tmp_path: Path):
        p = FilesystemPlugin()
        await p.configure(
            {"name": "x", "path": str(tmp_path), "index_dir": str(tmp_path / "idx")}
        )
        h = await p.health_check()
        assert h["status"] == "not_indexed"

    async def test_ok_after_index(self, tmp_path: Path, monkeypatch):
        d = tmp_path / "notes"
        d.mkdir()
        (d / "a.md").write_text("# A\nalpha")
        _patch_embed(monkeypatch, _fake_embed_factory({"# A\n\nalpha": [1.0, 0.0, 0.0, 0.0]}))
        _patch_embed_dimensions(monkeypatch, 4)

        p = FilesystemPlugin()
        await p.configure({"name": "n", "path": str(d), "index_dir": str(tmp_path / "idx")})
        await p.index()
        h = await p.health_check()
        assert h["status"] == "ok"
        assert h["file_count"] == 1
        assert h["chunk_count"] >= 1


class TestFilesystemFetch:
    async def test_fetch_returns_full_content(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# T\n\nfull body here")
        p = FilesystemPlugin()
        doc = await p.fetch(f"{f}#0")
        assert doc is not None
        assert doc.content == "# T\n\nfull body here"
        assert doc.title == "T"

    async def test_fetch_missing_file(self):
        p = FilesystemPlugin()
        doc = await p.fetch("/nonexistent.md#0")
        assert doc is None

    async def test_fetch_bad_id(self):
        p = FilesystemPlugin()
        assert await p.fetch("no-hash-here") is None
