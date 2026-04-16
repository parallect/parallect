"""Integration test: FilesystemPlugin end-to-end against a real embeddings server.

Skipped unless a real embeddings endpoint is reachable (LM Studio on 1234 or
a LiteLLM dev container). Wave-1 docker-compose in tests/integration/ brings
up LiteLLM -- when that's running this test verifies that index + search
actually produce coherent results with real embeddings.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from parallect.plugins.data_sources.filesystem import FilesystemPlugin


def _reachable(url: str) -> bool:
    try:
        with httpx.Client(timeout=0.3) as c:
            r = c.get(url)
            return r.status_code == 200
    except Exception:
        return False


@pytest.fixture
def real_embeddings_available() -> bool:
    # Prefer explicit opt-in (integration test runner sets this).
    if os.environ.get("PARALLECT_INTEGRATION_EMBEDDINGS") == "1":
        return True
    # Fallback: probe LM Studio / Ollama defaults.
    return _reachable("http://localhost:1234/v1/models") or _reachable(
        "http://localhost:11434/api/tags"
    )


@pytest.mark.asyncio
async def test_real_embed_index_and_search(
    tmp_path: Path, real_embeddings_available: bool
):
    if not real_embeddings_available:
        pytest.skip("no real embeddings endpoint reachable")

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "paxos.md").write_text(
        "# Paxos\n\nPaxos is a consensus algorithm requiring 2f+1 nodes to "
        "tolerate f failures. It has two phases: prepare and accept."
    )
    (notes / "raft.md").write_text(
        "# Raft\n\nRaft is a consensus algorithm with an explicit leader. "
        "It simplifies state machine replication compared to Paxos."
    )
    (notes / "coffee.md").write_text(
        "# Coffee\n\nArabica beans are grown at higher altitudes."
    )

    plugin = FilesystemPlugin()
    await plugin.configure(
        {"name": "notes-int", "path": str(notes), "index_dir": str(tmp_path / "idx")}
    )
    stats = await plugin.index()
    assert stats.documents_indexed >= 3

    docs = await plugin.search("leader election consensus", limit=3)
    # The top result for a consensus query should be Paxos or Raft, not coffee.
    assert docs
    top_titles = {(d.title or "").lower() for d in docs[:2]}
    assert any("paxos" in t or "raft" in t for t in top_titles)
