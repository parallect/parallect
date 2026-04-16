"""Filesystem data source plugin — index local directories for semantic search.

Design notes
------------

**sqlite-vec** is used opportunistically. If the user has it installed
(``pip install sqlite-vec``) we use its ``vec0`` virtual table for
sub-linear ANN search. Otherwise we fall back to a linear-scan SQLite
table that stores vectors as raw ``float32`` BLOBs and scores them in
Python. Both paths share the same schema for chunks + metadata; only the
vector column differs.

Rationale: sqlite-vec is ~30KB, one C extension, zero-config, and its
wheels cover macOS/Linux/Windows for cpython 3.9-3.12 — but it is an
optional dep because (a) the wheels aren't on every mirror, and (b) the
linear path is perfectly fine for the <50k-chunk indexes we expect in the
wave-1 target use case (a personal notes folder, a repo of docs).
Anything bigger should use the dedicated backend; this plugin is designed
to be the easy path, not the scaling path.

**Chunking** is paragraph-aware: split on blank lines, then merge
adjacent paragraphs until adding the next one would exceed
``CHUNK_TARGET_CHARS`` (~500 tokens @ 4 chars/token); then start a new
chunk. Oversized paragraphs (rare — long code blocks) get split on
sentence boundaries.

**Freshness** uses mtime + size for the quick-check. Only if the quick
check says "different" do we hash to be sure. For an unchanged file this
is O(1); for a fully-touched folder it degrades to one stat call per
file, which is still cheap.

**Dimension guarding** — the index stores the embedding model name and
dimensions it was built with. If the current ``[embeddings]`` config
disagrees, :meth:`index` (without ``--force``) raises a clear error so
the user knows to re-run with ``--force``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from parallect.plugins.data_sources import Document, IndexStats

logger = logging.getLogger(__name__)

DEFAULT_INCLUDE = ("*.md", "*.txt")
DEFAULT_EXCLUDE = (".git/", "node_modules/", ".DS_Store", ".venv/", "__pycache__/")
DEFAULT_MAX_FILE_KB = 500
CHUNK_TARGET_CHARS = 2000  # ~500 tokens @ 4 chars/token
EMBED_BATCH = 96


def _has_sqlite_vec() -> bool:
    """Probe whether sqlite-vec is importable."""
    try:
        import sqlite_vec  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class _ChunkRow:
    chunk_id: str
    path: str
    chunk_idx: int
    title: str
    content: str
    file_hash: str


class FilesystemPlugin:
    """Index one local directory and answer semantic search queries.

    Multiple instances are supported via the ``name`` config key
    (e.g. ``[[plugins.filesystem]] name = "notes"``). Each instance has
    its own SQLite index under ``~/.parallect/indexes/<instance>/index.db``.
    """

    name = "filesystem"
    display_name = "filesystem"
    requires_index = True

    def __init__(self) -> None:
        self._instance_name: str = "default"
        self._path: Path | None = None
        self._include: tuple[str, ...] = DEFAULT_INCLUDE
        self._exclude: tuple[str, ...] = DEFAULT_EXCLUDE
        self._max_file_bytes: int = DEFAULT_MAX_FILE_KB * 1024
        self._index_dir: Path | None = None
        self._use_vec: bool = _has_sqlite_vec()

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    async def configure(self, config: dict) -> None:
        """Accepts ``{name, path, include, exclude, max_file_kb, index_dir}``.

        ``name`` is the instance id; it also becomes ``self.name`` so that
        multi-instance registries can distinguish ``filesystem:notes`` from
        ``filesystem:repo``.
        """
        instance = str(config.get("name") or "filesystem").strip() or "filesystem"
        self._instance_name = instance
        self.name = instance if instance != "filesystem" else "filesystem"
        self.display_name = (
            f"filesystem:{instance}" if instance != "filesystem" else "filesystem"
        )

        raw_path = config.get("path")
        if not raw_path:
            raise ValueError("filesystem plugin requires a `path` config")
        self._path = Path(str(raw_path)).expanduser().resolve()
        if not self._path.exists():
            raise ValueError(f"filesystem path does not exist: {self._path}")

        self._include = tuple(config.get("include") or DEFAULT_INCLUDE)
        self._exclude = tuple(config.get("exclude") or DEFAULT_EXCLUDE)
        self._max_file_bytes = int(config.get("max_file_kb") or DEFAULT_MAX_FILE_KB) * 1024

        override_dir = config.get("index_dir")
        if override_dir:
            self._index_dir = Path(str(override_dir)).expanduser().resolve()
        else:
            self._index_dir = Path.home() / ".parallect" / "indexes" / instance
        self._index_dir.mkdir(parents=True, exist_ok=True)

    async def search(self, query: str, *, limit: int = 10) -> list[Document]:
        """Embed the query and return top-``limit`` chunks by cosine sim."""
        if self._index_dir is None:
            raise RuntimeError("filesystem plugin not configured")
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
        return self._search_index(db_path, qvec, limit)

    async def index(self, *, force: bool = False) -> IndexStats:
        """Walk the configured path, embed changed chunks, store to SQLite."""
        if self._path is None or self._index_dir is None:
            raise RuntimeError("filesystem plugin not configured")
        start = time.monotonic()
        from parallect import embeddings

        # Resolve the active embeddings model + dimensions via the shared
        # backends module. Doing this up-front (one probe) lets us detect a
        # dim/model change before doing any work.
        current_model = _read_current_model()
        current_dims = await _safe_embed_dimensions()
        db_path = self._db_path()
        existing_meta = self._read_metadata()

        # Dimension/model mismatch gate.
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

        # Progress bar — visible when stdout is a terminal
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
        )
        with progress:
            task = progress.add_task(
                f"Indexing {len(changed)} files", total=len(changed)
            )
            for file_path, file_meta in changed:
                progress.update(task, description=f"[dim]{file_path.name}[/dim]")
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("skipping unreadable file %s: %s", file_path, exc)
                    skipped += 1
                    progress.advance(task)
                    continue
                chunks = _chunk_text(content)
                if not chunks:
                    self._record_file(db_path, file_path, file_meta, chunk_count=0)
                    progress.advance(task)
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
                progress.advance(task)

        # Persist embeddings metadata so later runs can detect dim/model
        # changes. Fall back to the dims of the most recently stored vector
        # when settings don't carry that info.
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

    async def is_fresh(self) -> bool:
        """True if no file under ``path`` is newer than the last indexed_at."""
        if self._index_dir is None or self._path is None:
            return False
        db_path = self._db_path()
        if not db_path.exists():
            return False
        meta = self._read_metadata()
        if not meta or not meta.get("indexed_at"):
            return False
        cutoff = float(meta["indexed_at"])
        for file_path in self._iter_candidate_files():
            try:
                if file_path.stat().st_mtime > cutoff:
                    return False
            except OSError:
                continue
        return True

    async def fetch(self, doc_id: str) -> Document | None:
        """Fetch full file content for a ``<path>#<chunk_idx>`` doc id."""
        if "#" not in doc_id:
            return None
        path_str, _, _idx = doc_id.rpartition("#")
        file_path = Path(path_str)
        if not file_path.exists():
            return None
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        title = _extract_title(content, file_path.name)
        return Document(
            id=doc_id,
            content=content,
            title=title,
            source_url=f"file://{file_path}",
            metadata={"path": str(file_path), "kind": "filesystem"},
        )

    async def health_check(self) -> dict:
        if self._index_dir is None or self._path is None:
            return {"status": "unconfigured"}
        db_path = self._db_path()
        if not db_path.exists():
            return {
                "status": "not_indexed",
                "path": str(self._path),
                "index_path": str(db_path),
            }
        meta = self._read_metadata() or {}
        return {
            "status": "ok",
            "path": str(self._path),
            "index_path": str(db_path),
            "file_count": meta.get("file_count"),
            "chunk_count": meta.get("chunk_count"),
            "embedding_model": meta.get("embedding_model"),
            "dimensions": meta.get("dimensions"),
            "indexed_at": meta.get("indexed_at"),
            "backend": "sqlite-vec" if self._use_vec else "linear-scan",
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _db_path(self) -> Path:
        assert self._index_dir is not None
        return self._index_dir / "index.db"

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        if self._use_vec:
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception:
                self._use_vec = False
        return conn

    def _init_schema(self, db_path: Path) -> None:
        with self._connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path        TEXT PRIMARY KEY,
                    size        INTEGER NOT NULL,
                    mtime       REAL NOT NULL,
                    hash        TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at  REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id   TEXT PRIMARY KEY,
                    path       TEXT NOT NULL,
                    chunk_idx  INTEGER NOT NULL,
                    title      TEXT,
                    content    TEXT NOT NULL,
                    embedding  BLOB,
                    FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_metadata (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _ensure_vec_table(self, conn: sqlite3.Connection, dim: int) -> None:
        if not self._use_vec:
            return
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
        ).fetchone()
        if row:
            return
        conn.execute(
            f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{dim}])"
        )

    def _replace_chunks(
        self,
        db_path: Path,
        *,
        file_path: str,
        rows: list[_ChunkRow],
        vectors: list[list[float]],
    ) -> None:
        with self._connect(db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE path = ?", (file_path,))
            if self._use_vec and vectors:
                self._ensure_vec_table(conn, dim=len(vectors[0]))
            for row, vec in zip(rows, vectors):
                blob = _pack_vector(vec)
                conn.execute(
                    "INSERT INTO chunks(chunk_id, path, chunk_idx, title, content, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (row.chunk_id, row.path, row.chunk_idx, row.title, row.content, blob),
                )
                if self._use_vec:
                    rowid = conn.execute(
                        "SELECT rowid FROM chunks WHERE chunk_id = ?", (row.chunk_id,)
                    ).fetchone()[0]
                    try:
                        conn.execute(
                            "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                            (rowid, blob),
                        )
                    except sqlite3.OperationalError:
                        self._use_vec = False
            conn.commit()

    def _record_file(
        self,
        db_path: Path,
        file_path: Path,
        file_meta: dict,
        *,
        chunk_count: int,
    ) -> None:
        with self._connect(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO files(path, size, mtime, hash, chunk_count, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(file_path),
                    file_meta["size"],
                    file_meta["mtime"],
                    file_meta["hash"],
                    chunk_count,
                    time.time(),
                ),
            )
            conn.commit()

    def _load_known_files(self, db_path: Path) -> dict[str, dict]:
        if not db_path.exists():
            return {}
        try:
            with self._connect(db_path) as conn:
                rows = conn.execute("SELECT path, size, mtime, hash FROM files").fetchall()
        except sqlite3.OperationalError:
            return {}
        return {
            r["path"]: {"size": r["size"], "mtime": r["mtime"], "hash": r["hash"]}
            for r in rows
        }

    def _iter_candidate_files(self) -> Iterable[Path]:
        assert self._path is not None
        root = self._path
        for pattern in self._include:
            yield from _iter_globs(root, pattern, self._exclude)

    def _walk_for_changes(
        self, known: dict[str, dict]
    ) -> tuple[list[tuple[Path, dict]], int]:
        changed: list[tuple[Path, dict]] = []
        skipped = 0
        for path in self._iter_candidate_files():
            try:
                st = path.stat()
            except OSError:
                skipped += 1
                continue
            if st.st_size > self._max_file_bytes:
                skipped += 1
                continue
            prev = known.get(str(path))
            if prev and prev["size"] == st.st_size and prev["mtime"] == st.st_mtime:
                skipped += 1
                continue
            try:
                file_hash = _file_hash(path)
            except OSError:
                skipped += 1
                continue
            if prev and prev["hash"] == file_hash:
                skipped += 1
                continue
            changed.append(
                (path, {"size": st.st_size, "mtime": st.st_mtime, "hash": file_hash})
            )
        return changed, skipped

    def _search_index(self, db_path: Path, qvec: list[float], limit: int) -> list[Document]:
        with self._connect(db_path) as conn:
            if self._use_vec:
                qblob = _pack_vector(qvec)
                try:
                    rows = conn.execute(
                        """
                        SELECT c.chunk_id, c.path, c.chunk_idx, c.title, c.content,
                               v.distance AS distance
                        FROM chunks_vec v
                        JOIN chunks c ON c.rowid = v.rowid
                        WHERE v.embedding MATCH ?
                        ORDER BY v.distance
                        LIMIT ?
                        """,
                        (qblob, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = None
                if rows is not None:
                    return [_row_to_doc(r, score=_distance_to_score(r["distance"])) for r in rows]
            rows = conn.execute(
                "SELECT chunk_id, path, chunk_idx, title, content, embedding FROM chunks"
            ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            vec = _unpack_vector(r["embedding"])
            if len(vec) != len(qvec):
                continue
            sim = _cosine_similarity(qvec, vec)
            scored.append((sim, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [_row_to_doc(r, score=sim) for sim, r in scored[:limit]]

    def _read_metadata(self) -> dict | None:
        db_path = self._db_path()
        if not db_path.exists():
            return None
        try:
            with self._connect(db_path) as conn:
                rows = conn.execute("SELECT key, value FROM index_metadata").fetchall()
        except sqlite3.OperationalError:
            return None
        if not rows:
            return None
        out: dict[str, Any] = {}
        for r in rows:
            k = r["key"]
            v = r["value"]
            if k in ("dimensions", "file_count", "chunk_count"):
                try:
                    out[k] = int(v)
                except ValueError:
                    out[k] = v
            elif k == "indexed_at":
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            else:
                out[k] = v
        return out

    def _write_metadata(
        self,
        db_path: Path,
        *,
        embedding_model: str,
        dimensions: int,
        file_count: int,
        chunk_count: int,
    ) -> None:
        with self._connect(db_path) as conn:
            rows = [
                ("embedding_model", embedding_model),
                ("dimensions", str(dimensions)),
                ("file_count", str(file_count)),
                ("chunk_count", str(chunk_count)),
                ("indexed_at", str(time.time())),
            ]
            for k, v in rows:
                conn.execute(
                    "INSERT INTO index_metadata(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (k, v),
                )
            conn.commit()

    def _probe_stored_dim(self, db_path: Path) -> int | None:
        """Inspect the first stored embedding blob to infer its dimension.

        Used when ``[embeddings]`` config is absent but we've just stored
        vectors anyway -- we still want metadata written so freshness /
        health_check surface accurate counts.
        """
        with self._connect(db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1"
            ).fetchone()
        if not row or not row[0]:
            return None
        return len(row[0]) // 4

    def _count_files(self, db_path: Path) -> int:
        with self._connect(db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])

    def _count_chunks(self, db_path: Path) -> int:
        with self._connect(db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])


# ---------------------------------------------------------------------------
# Helpers — chunking, hashing, vector codec
# ---------------------------------------------------------------------------


def _iter_globs(root: Path, pattern: str, exclude: tuple[str, ...]) -> Iterable[Path]:
    for p in root.rglob(pattern):
        if not p.is_file():
            continue
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)
        skip = False
        for ex in exclude:
            if ex in rel or rel.startswith(ex) or f"/{ex}" in f"/{rel}":
                skip = True
                break
        if skip:
            continue
        yield p


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _extract_title(content: str, fallback: str) -> str:
    m = _HEADING_RE.search(content)
    if m:
        return m.group(1).strip()
    return fallback


def _chunk_text(text: str) -> list[str]:
    """Paragraph-aware chunking with a soft ~500-token target."""
    text = text.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in paragraphs:
        if len(para) > CHUNK_TARGET_CHARS:
            if buf:
                chunks.append("\n\n".join(buf))
                buf = []
                buf_len = 0
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sbuf: list[str] = []
            sbuf_len = 0
            for s in sentences:
                if sbuf_len + len(s) > CHUNK_TARGET_CHARS and sbuf:
                    chunks.append(" ".join(sbuf))
                    sbuf = []
                    sbuf_len = 0
                sbuf.append(s)
                sbuf_len += len(s) + 1
            if sbuf:
                chunks.append(" ".join(sbuf))
            continue
        if buf_len + len(para) > CHUNK_TARGET_CHARS and buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0
        buf.append(para)
        buf_len += len(para) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(blob: bytes) -> list[float]:
    if not blob:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


def _distance_to_score(distance: float | None) -> float | None:
    if distance is None:
        return None
    try:
        return 1.0 - float(distance)
    except (TypeError, ValueError):
        return None


def _row_to_doc(r: sqlite3.Row, *, score: float | None) -> Document:
    path = r["path"]
    content = r["content"]
    snippet = content[:400]
    return Document(
        id=r["chunk_id"],
        content=snippet,
        title=r["title"] or Path(path).name,
        source_url=f"file://{path}",
        score=score,
        metadata={
            "kind": "filesystem",
            "path": path,
            "chunk_idx": r["chunk_idx"],
        },
    )


__all__ = ["FilesystemPlugin"]


# ---------------------------------------------------------------------------
# Embeddings resolution helpers
# ---------------------------------------------------------------------------


def _read_current_model() -> str | None:
    """Best-effort read of the currently-configured embeddings model name.

    Returns ``None`` if settings cannot be loaded or the model isn't set.
    Used for the dimension-mismatch gate; a missing model name just skips
    that half of the check, which is the safe default.
    """
    try:
        from parallect.config_mod.settings import ParallectSettings

        settings = ParallectSettings.load()
    except Exception:
        return None
    return getattr(settings, "embeddings_model", "") or None


async def _safe_embed_dimensions() -> int | None:
    """Probe the embeddings backend for output dimensions.

    Returns ``None`` on any failure so we don't block indexing when the
    probe can't succeed (e.g. offline environment). The dim-mismatch gate
    simply won't fire in that case, which matches the "warn if we can,
    otherwise proceed" posture of the rest of the CLI.
    """
    try:
        from parallect import embeddings

        # embed_dimensions is the new wave-1 public API.
        if hasattr(embeddings, "embed_dimensions"):
            dims = await embeddings.embed_dimensions()
            return int(dims) if dims else None
    except Exception:
        return None
    return None

