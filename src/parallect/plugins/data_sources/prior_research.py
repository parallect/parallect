"""Prior research cache plugin.

Stores every ``parallect research`` run (query, synthesis markdown, sources
JSON, timestamp, bundle id) in ``~/.parallect/research-cache.db`` along with
an embedding of the query+synthesis, so later runs can retrieve semantically
similar past work.

This plugin's index is *incremental* — there's no ``parallect plugins index
prior-research`` step. :meth:`append` is called by the orchestrator after
each run; :meth:`search` embeds the query and returns the top matches.
``requires_index`` is therefore ``False`` even though we maintain a SQLite
store on disk (the convention in this codebase is "does the user need to
run index manually?").
"""

from __future__ import annotations

import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

from parallect.plugins.data_sources import Document, IndexStats


DEFAULT_DB_PATH = Path.home() / ".parallect" / "research-cache.db"


class PriorResearchCache:
    """Semantic cache over your own prior research bundles."""

    name = "prior-research"
    display_name = "prior research"
    requires_index = False

    def __init__(self) -> None:
        self._db_path: Path = DEFAULT_DB_PATH
        self._max_results: int = 10

    async def configure(self, config: dict) -> None:
        override = config.get("db_path")
        if override:
            self._db_path = Path(str(override)).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_results = int(config.get("max_results") or 10)
        self._init_schema()

    async def search(self, query: str, *, limit: int = 10) -> list[Document]:
        """Embed the query; return the top-``limit`` past runs by cosine sim."""
        self._init_schema()
        from parallect import embeddings

        vectors = await embeddings.embed([query])
        if not vectors:
            return []
        qvec = vectors[0]
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT bundle_id, past_query, synthesis_md, sources_json, "
                "       timestamp, embedding FROM entries"
            ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            vec = _unpack_vector(r["embedding"])
            if len(vec) != len(qvec):
                continue
            sim = _cosine_similarity(qvec, vec)
            scored.append((sim, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[Document] = []
        for sim, r in scored[:limit]:
            synthesis = r["synthesis_md"] or ""
            out.append(
                Document(
                    id=f"prior:{r['bundle_id']}",
                    content=synthesis,
                    title=r["past_query"],
                    source_url=None,
                    score=sim,
                    metadata={
                        "kind": "prior-research",
                        "bundle_id": r["bundle_id"],
                        "past_query": r["past_query"],
                        "timestamp": r["timestamp"],
                    },
                )
            )
        return out

    async def index(self, *, force: bool = False) -> IndexStats:
        """No-op (cache is incremental). ``--force`` clears the cache."""
        self._init_schema()
        if force:
            with self._connect() as conn:
                conn.execute("DELETE FROM entries")
                conn.commit()
        with self._connect() as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
        return IndexStats(
            source_name=self.name,
            documents_indexed=0,
            documents_skipped=count,
            elapsed_seconds=0.0,
            index_path=str(self._db_path),
        )

    async def is_fresh(self) -> bool:
        # Always fresh — the cache is append-only.
        return True

    async def fetch(self, doc_id: str) -> Document | None:
        if not doc_id.startswith("prior:"):
            return None
        bundle_id = doc_id.removeprefix("prior:")
        self._init_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT bundle_id, past_query, synthesis_md, sources_json, timestamp "
                "FROM entries WHERE bundle_id = ?",
                (bundle_id,),
            ).fetchone()
        if not row:
            return None
        return Document(
            id=doc_id,
            content=row["synthesis_md"] or "",
            title=row["past_query"],
            source_url=None,
            metadata={
                "kind": "prior-research",
                "bundle_id": row["bundle_id"],
                "past_query": row["past_query"],
                "timestamp": row["timestamp"],
                "sources_json": row["sources_json"],
            },
        )

    async def health_check(self) -> dict:
        self._init_schema()
        with self._connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
            latest = conn.execute(
                "SELECT timestamp FROM entries ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return {
            "status": "ok",
            "total_runs": total,
            "latest_timestamp": latest["timestamp"] if latest else None,
            "db_path": str(self._db_path),
        }

    # ------------------------------------------------------------------
    # Incremental append — called by the orchestrator after each run
    # ------------------------------------------------------------------

    async def append(
        self,
        *,
        query: str,
        synthesis_md: str,
        sources_json: str,
        bundle_id: str,
    ) -> None:
        """Append a completed research run to the cache.

        Embeds ``query`` + the first 1KB of synthesis to avoid blowing token
        limits for long reports. The embedding is used for the similarity
        search in :meth:`search`.
        """
        self._init_schema()
        from parallect import embeddings

        text = f"{query}\n\n{(synthesis_md or '')[:1024]}"
        vectors = await embeddings.embed([text])
        vec = vectors[0] if vectors else []
        blob = _pack_vector(vec)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO entries("
                "  bundle_id, past_query, synthesis_md, sources_json, timestamp, embedding"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (bundle_id, query, synthesis_md, sources_json, time.time(), blob),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    bundle_id    TEXT PRIMARY KEY,
                    past_query   TEXT NOT NULL,
                    synthesis_md TEXT,
                    sources_json TEXT,
                    timestamp    REAL NOT NULL,
                    embedding    BLOB
                )
                """
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Shared vector helpers (kept module-local to avoid cross-plugin imports)
# ---------------------------------------------------------------------------


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(blob: Any) -> list[float]:
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


__all__ = ["PriorResearchCache"]
