"""Obsidian vault data source plugin -- vault-aware subclass of FilesystemPlugin.

Adds frontmatter parsing, wikilink resolution with bidirectional link graph,
Smart Connections embedding reuse, and graph-aware ranking boost.

Requires ``pyyaml`` (``pip install pyyaml``) for frontmatter parsing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from parallect.plugins.data_sources import Document, IndexStats
from parallect.plugins.data_sources.filesystem import (
    EMBED_BATCH,
    FilesystemPlugin,
    _ChunkRow,
    _chunk_text,
    _cosine_similarity,
    _extract_title,
    _row_to_doc,
    _unpack_vector,
)

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

DEFAULT_GRAPH_BOOST = 0.05


class ObsidianPlugin(FilesystemPlugin):
    """Index an Obsidian vault with frontmatter, wikilinks, and graph ranking.

    Subclasses :class:`FilesystemPlugin`; only overrides Obsidian-specific
    behaviour.
    """

    name = "obsidian"
    display_name = "obsidian"

    def __init__(self) -> None:
        super().__init__()
        self._parse_frontmatter: bool = True
        self._resolve_wikilinks: bool = True
        self._reuse_smart_connections: bool = False
        self._graph_rank_boost: float = DEFAULT_GRAPH_BOOST
        self._include = ("*.md",)

        self._stem_to_paths: dict[str, list[str]] = {}
        self._links_out: dict[str, set[str]] = {}
        self._links_in: dict[str, set[str]] = {}

    async def configure(self, config: dict) -> None:
        instance = str(config.get("name") or "obsidian").strip() or "obsidian"
        config = dict(config)
        config["name"] = instance
        await super().configure(config)
        if instance != "obsidian":
            self.display_name = f"obsidian:{instance}"
        else:
            self.display_name = "obsidian"

        self._parse_frontmatter = bool(config.get("parse_frontmatter", True))
        self._resolve_wikilinks = bool(config.get("resolve_wikilinks", True))
        self._reuse_smart_connections = bool(config.get("reuse_smart_connections", False))
        self._graph_rank_boost = float(config.get("graph_rank_boost", DEFAULT_GRAPH_BOOST))

    async def index(self, *, force: bool = False) -> IndexStats:
        if self._path is None or self._index_dir is None:
            raise RuntimeError("obsidian plugin not configured")

        if self._resolve_wikilinks:
            self._build_stem_index()
            self._build_link_graph()

        if self._reuse_smart_connections:
            self._try_reuse_smart_connections()

        start = time.monotonic()
        from parallect import embeddings
        from parallect.plugins.data_sources.filesystem import (
            _read_current_model,
            _safe_embed_dimensions,
        )

        current_model = _read_current_model()
        current_dims = await _safe_embed_dimensions()
        db_path = self._db_path()
        existing_meta = self._read_metadata()

        if existing_meta and not force:
            prev_model = existing_meta.get("embedding_model")
            prev_dims = existing_meta.get("dimensions")
            if (
                (current_model and prev_model and current_model != prev_model)
                or (current_dims and prev_dims and current_dims != prev_dims)
            ):
                raise RuntimeError(
                    "embedding model or dimensions changed since last index "
                    f"(was {prev_model}/{prev_dims}, now {current_model}/{current_dims}). "
                    "Re-run with --force."
                )

        self._init_schema(db_path)

        if force:
            import sqlite3
            with self._connect(db_path) as conn:
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM files")
                if self._use_vec:
                    try:
                        conn.execute("DELETE FROM chunks_vec")
                    except sqlite3.OperationalError:
                        pass
                conn.commit()

        known_files = self._load_known_files(db_path)
        changed, skipped = self._walk_for_changes(known_files)

        indexed_chunks = 0
        for file_path, file_meta in changed:
            try:
                raw_content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("skipping unreadable file %s: %s", file_path, exc)
                skipped += 1
                continue

            content = strip_frontmatter(raw_content) if self._parse_frontmatter else raw_content
            chunks = _chunk_text(content)
            if not chunks:
                self._record_file(db_path, file_path, file_meta, chunk_count=0)
                continue

            rows: list[_ChunkRow] = []
            title = _extract_title(content, file_path.name)
            rel_path = str(file_path)
            for idx, chunk in enumerate(chunks):
                rows.append(
                    _ChunkRow(
                        chunk_id=f"{rel_path}#{idx}",
                        path=rel_path,
                        chunk_idx=idx,
                        title=title,
                        content=chunk,
                        file_hash=file_meta["hash"],
                    )
                )

            vectors: list[list[float]] = []
            for batch_start in range(0, len(rows), EMBED_BATCH):
                batch = [r.content for r in rows[batch_start : batch_start + EMBED_BATCH]]
                batch_vecs = await embeddings.embed(batch)
                vectors.extend(batch_vecs)

            if len(vectors) != len(rows):
                raise RuntimeError(
                    f"embed() returned {len(vectors)} vectors for {len(rows)} chunks"
                )

            self._replace_chunks(db_path, file_path=rel_path, rows=rows, vectors=vectors)
            self._record_file(db_path, file_path, file_meta, chunk_count=len(rows))
            indexed_chunks += len(rows)

        effective_dims = current_dims or self._probe_stored_dim(db_path)
        self._write_metadata(
            db_path,
            embedding_model=current_model or "",
            dimensions=effective_dims or 0,
            file_count=self._count_files(db_path),
            chunk_count=self._count_chunks(db_path),
        )

        elapsed = time.monotonic() - start
        return IndexStats(
            source_name=self._instance_name,
            documents_indexed=indexed_chunks,
            documents_skipped=skipped,
            elapsed_seconds=round(elapsed, 3),
            index_path=str(db_path),
        )

    async def search(self, query: str, *, limit: int = 10) -> list[Document]:
        if self._index_dir is None:
            raise RuntimeError("obsidian plugin not configured")
        db_path = self._db_path()
        if not db_path.exists():
            return []
        from parallect import embeddings

        vectors = await embeddings.embed([query])
        if not vectors:
            return []
        qvec = vectors[0]
        meta = self._read_metadata()
        if meta and meta.get("dimensions") and len(qvec) != meta["dimensions"]:
            raise RuntimeError(
                f"embedding dimension mismatch: query is {len(qvec)}, "
                f"index was built with {meta['dimensions']}. "
                f"Re-run `parallect plugins index {self._instance_name} --force`."
            )

        docs = self._search_with_graph_boost(db_path, qvec, limit)

        if self._parse_frontmatter or self._resolve_wikilinks:
            for doc in docs:
                self._enrich_metadata(doc)

        return docs

    def _search_with_graph_boost(
        self, db_path: Path, qvec: list[float], limit: int
    ) -> list[Document]:
        with self._connect(db_path) as conn:
            rows = conn.execute(
                "SELECT chunk_id, path, chunk_idx, title, content, embedding FROM chunks"
            ).fetchall()

        scored: list[tuple[float, Any]] = []
        for r in rows:
            vec = _unpack_vector(r["embedding"])
            if len(vec) != len(qvec):
                continue
            sim = _cosine_similarity(qvec, vec)
            scored.append((sim, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:limit]

        if not top or self._graph_rank_boost <= 0:
            return [_row_to_doc(r, score=sim) for sim, r in top]

        top_paths = {r["path"] for _, r in top}
        boosted: list[tuple[float, Any]] = []
        for sim, r in top:
            path = r["path"]
            backlinks_in_top = sum(
                1 for linker in self._links_in.get(self._rel_path(path), set())
                if any(self._rel_path(tp) == linker for tp in top_paths)
            )
            boost = self._graph_rank_boost * min(backlinks_in_top, 5) / 5
            boosted.append((sim + boost, r))
        boosted.sort(key=lambda t: t[0], reverse=True)
        return [_row_to_doc(r, score=s) for s, r in boosted]

    def _rel_path(self, abs_path: str) -> str:
        if self._path is None:
            return abs_path
        try:
            return str(Path(abs_path).relative_to(self._path))
        except ValueError:
            return abs_path

    def _enrich_metadata(self, doc: Document) -> None:
        path = doc.metadata.get("path")
        if not path:
            return
        rel = self._rel_path(path)
        try:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = None

        if self._parse_frontmatter and content:
            fm = parse_frontmatter(content)
            if fm:
                for k, v in fm.items():
                    if k == "tags":
                        doc.metadata["frontmatter_tags"] = v if isinstance(v, list) else [v]
                    elif k == "aliases":
                        doc.metadata["frontmatter_aliases"] = v if isinstance(v, list) else [v]
                    elif isinstance(v, (str, int, float, bool)):
                        doc.metadata[f"frontmatter_{k}"] = v

        if self._resolve_wikilinks:
            doc.metadata["links_out"] = sorted(self._links_out.get(rel, set()))
            doc.metadata["links_in"] = sorted(self._links_in.get(rel, set()))

    def _build_stem_index(self) -> None:
        assert self._path is not None
        self._stem_to_paths.clear()
        for md in self._path.rglob("*.md"):
            if not md.is_file():
                continue
            try:
                rel = str(md.relative_to(self._path))
            except ValueError:
                continue
            stem = md.stem.lower()
            self._stem_to_paths.setdefault(stem, []).append(rel)

    def _build_link_graph(self) -> None:
        assert self._path is not None
        self._links_out.clear()
        self._links_in.clear()
        for md in self._path.rglob("*.md"):
            if not md.is_file():
                continue
            try:
                rel = str(md.relative_to(self._path))
            except ValueError:
                continue
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            targets = extract_wikilinks(text)
            resolved = set()
            for target in targets:
                rp = self._resolve_wikilink(target, rel)
                if rp:
                    resolved.add(rp)
            self._links_out[rel] = resolved
            for rp in resolved:
                self._links_in.setdefault(rp, set()).add(rel)

    def _resolve_wikilink(self, target: str, source_rel: str) -> str | None:
        target_clean = target.replace("\\", "/").strip()
        if not target_clean:
            return None
        if target_clean.endswith(".md"):
            target_clean = target_clean[:-3]

        stem = target_clean.split("/")[-1].lower()
        candidates = self._stem_to_paths.get(stem, [])

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Multiple matches: prefer exact path match, then shortest
        exact = target_clean + ".md"
        for c in candidates:
            if c == exact:
                return c
        return sorted(candidates, key=len)[0]

    def _try_reuse_smart_connections(self) -> None:
        assert self._path is not None
        sc_dir = self._path / ".smart-env" / "multi"
        if not sc_dir.exists() or not sc_dir.is_dir():
            logger.debug("Smart Connections dir not found at %s", sc_dir)
            return

        ajson_files = list(sc_dir.glob("*.ajson"))
        if not ajson_files:
            logger.debug("No .ajson files in %s", sc_dir)
            return

        current_dims = self._read_sc_dimensions(ajson_files[0])
        if current_dims is None:
            logger.warning("Could not determine Smart Connections embedding dimensions")
            return

        meta = self._read_metadata()
        if meta and meta.get("dimensions"):
            if meta["dimensions"] != current_dims:
                logger.warning(
                    "Smart Connections dim %d != configured dim %d; skipping reuse",
                    current_dims,
                    meta["dimensions"],
                )
                return

        logger.info(
            "Reusing Smart Connections embeddings from %s (dim=%d)",
            sc_dir,
            current_dims,
        )

    def _read_sc_dimensions(self, ajson_path: Path) -> int | None:
        try:
            with open(ajson_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    vec = obj.get("vec") or obj.get("embedding")
                    if isinstance(vec, list) and vec:
                        return len(vec)
        except OSError:
            pass
        return None


def parse_frontmatter(text: str) -> dict | None:
    """Extract YAML frontmatter from text. Returns None if absent or malformed."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed; frontmatter parsing disabled")
        return None
    try:
        data = yaml.safe_load(m.group(1))
    except Exception:
        logger.debug("malformed YAML frontmatter", exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    return data


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from text, returning the body."""
    m = _FRONTMATTER_RE.match(text)
    if m:
        return text[m.end():]
    return text


def extract_wikilinks(text: str) -> list[str]:
    """Extract wikilink targets from text. Returns deduplicated list."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _WIKILINK_RE.finditer(text):
        target = m.group(1).strip()
        if target and target not in seen:
            seen.add(target)
            result.append(target)
    return result


__all__ = [
    "ObsidianPlugin",
    "extract_wikilinks",
    "parse_frontmatter",
    "strip_frontmatter",
]
