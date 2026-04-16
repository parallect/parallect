"""Prxhub data source plugin.

Queries the public prxhub search API for bundles + claims. Read-only, no
auth required for the public endpoint — but supports an API key for
authenticated/private hub instances.

Response shape from ``GET /api/search``::

    {
      "bundles": {"results": [{id, title, query, trust_score, divergence,
                               attestation_count, providers, ...}], ...},
      "claims":  {"results": [{id, content, confidence, bundle_id, ...}], ...}
    }

Both sides are normalised into :class:`Document` with ``metadata.kind`` set
to ``"bundle"`` or ``"claim"``, so synthesis can treat them uniformly while
preserving the hub's richer trust signals.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from parallect.plugins.data_sources import Document, IndexStats

DEFAULT_API_URL = "https://prxhub.com"
DEFAULT_TIMEOUT = 20.0


class PrxhubPlugin:
    """Read-only search against the prxhub public API."""

    name = "prxhub"
    display_name = "prxhub"
    requires_index = False

    def __init__(self) -> None:
        self._api_url: str = DEFAULT_API_URL
        self._api_key: str | None = None
        self._timeout: float = DEFAULT_TIMEOUT
        self._configured: bool = False

    async def configure(self, config: dict) -> None:
        """Accepts ``{api_url, api_key, timeout}``; all optional."""
        self._api_url = str(config.get("api_url") or DEFAULT_API_URL).rstrip("/")
        self._api_key = config.get("api_key") or None
        self._timeout = float(config.get("timeout") or DEFAULT_TIMEOUT)
        self._configured = True

    def _headers(self) -> dict:
        headers = {"Accept": "application/json", "User-Agent": "parallect-cli"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def search(self, query: str, *, limit: int = 10) -> list[Document]:
        """Search the hub, returning bundles + claims (up to ``limit`` each)."""
        params = {"q": query, "limit": str(limit)}
        url = f"{self._api_url}/api/search"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        return _parse_search_response(data, api_url=self._api_url)

    async def index(self, *, force: bool = False) -> IndexStats:
        """No-op: prxhub search is live, no local index."""
        return IndexStats(
            source_name=self.name,
            documents_indexed=0,
            documents_skipped=0,
            elapsed_seconds=0.0,
            index_path=None,
        )

    async def is_fresh(self) -> bool:
        """Always "fresh" — there's no local cache to stale."""
        return True

    async def fetch(self, doc_id: str) -> Document | None:
        """Fetch a single bundle or claim by id.

        ``doc_id`` format: ``bundle:<id>`` or ``claim:<id>``. Matches the
        format produced by :func:`_parse_search_response`.
        """
        if ":" not in doc_id:
            return None
        kind, _, raw_id = doc_id.partition(":")
        path_map = {"bundle": "/api/bundles/", "claim": "/api/claims/"}
        if kind not in path_map:
            return None
        url = f"{self._api_url}{path_map[kind]}{raw_id}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError:
            return None
        if kind == "bundle":
            return _bundle_to_doc(data, api_url=self._api_url)
        return _claim_to_doc(data, api_url=self._api_url)

    async def health_check(self) -> dict:
        """Hit ``GET /api/health`` and return the result (or the error)."""
        start = time.monotonic()
        url = f"{self._api_url}/api/health"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            return {
                "status": "error",
                "error": str(exc),
                "api_url": self._api_url,
                "elapsed_ms": int((time.monotonic() - start) * 1000),
            }
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if resp.status_code >= 400:
            return {
                "status": "error",
                "http_status": resp.status_code,
                "api_url": self._api_url,
                "elapsed_ms": elapsed_ms,
            }
        body: Any = {}
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:200]}
        return {
            "status": "ok",
            "http_status": resp.status_code,
            "api_url": self._api_url,
            "elapsed_ms": elapsed_ms,
            "body": body,
        }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_search_response(data: dict, *, api_url: str) -> list[Document]:
    docs: list[Document] = []
    bundles = ((data or {}).get("bundles") or {}).get("results") or []
    claims = ((data or {}).get("claims") or {}).get("results") or []
    for b in bundles:
        docs.append(_bundle_to_doc(b, api_url=api_url))
    for c in claims:
        docs.append(_claim_to_doc(c, api_url=api_url))
    return docs


def _bundle_to_doc(b: dict, *, api_url: str) -> Document:
    bundle_id = str(b.get("id") or b.get("bundle_id") or "")
    title = b.get("title") or b.get("query")
    content_parts = []
    if b.get("query"):
        content_parts.append(f"Query: {b['query']}")
    if b.get("summary"):
        content_parts.append(str(b["summary"]))
    if b.get("synthesis_md"):
        content_parts.append(str(b["synthesis_md"]))
    content = "\n\n".join(content_parts) or (title or "")
    return Document(
        id=f"bundle:{bundle_id}",
        content=content,
        title=title,
        source_url=f"{api_url}/bundles/{bundle_id}" if bundle_id else None,
        score=_coerce_float(b.get("score")),
        metadata={
            "kind": "bundle",
            "bundle_id": bundle_id or None,
            "trust_score": _coerce_float(b.get("trust_score")),
            "divergence": _coerce_float(b.get("divergence")),
            "attestation_count": _coerce_int(b.get("attestation_count")),
            "providers": list(b.get("providers") or []),
        },
    )


def _claim_to_doc(c: dict, *, api_url: str) -> Document:
    claim_id = str(c.get("id") or c.get("claim_id") or "")
    bundle_id = str(c.get("bundle_id") or "")
    content = str(c.get("content") or c.get("text") or "")
    source_url = (
        f"{api_url}/bundles/{bundle_id}#claim-{claim_id}"
        if bundle_id and claim_id
        else None
    )
    return Document(
        id=f"claim:{claim_id}",
        content=content,
        title=(content[:80] + "...") if len(content) > 80 else (content or None),
        source_url=source_url,
        score=_coerce_float(c.get("score")),
        metadata={
            "kind": "claim",
            "bundle_id": bundle_id or None,
            "confidence": _coerce_float(c.get("confidence")),
            "trust_score": _coerce_float(c.get("trust_score")),
            "divergence": _coerce_float(c.get("divergence")),
            "attestation_count": _coerce_int(c.get("attestation_count")),
            "providers": list(c.get("providers") or []),
        },
    )


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


__all__ = ["PrxhubPlugin"]
