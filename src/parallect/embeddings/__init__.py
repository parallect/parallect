"""Embeddings abstraction.

Wave-1 public API:

    from parallect.embeddings import embed, embed_dimensions

    vectors = await embed(["hello", "world"])
    dim = await embed_dimensions()

Plugins and downstream Parallect modules (claim clustering, evidence graph,
hub search) all depend on this thin surface. Keep it stable.
"""

from __future__ import annotations

import asyncio

from parallect.backends import (
    KNOWN_EMBEDDING_DIMS,
    OPENAI_COMPAT_BACKENDS,
    BackendSpec,
    resolve_embeddings_backend,
)
from parallect.backends.adapters import (
    call_gemini_embeddings,
    call_openai_compat_embeddings,
)

# Module-level cache of probed embedding dimensions, keyed by
# (backend_kind, model_name). Avoids round-tripping to the server on every
# dimension lookup -- embeddings tend to be called from hot paths.
_DIMENSION_CACHE: dict[tuple[str, str], int] = {}
_DIMENSION_LOCK: asyncio.Lock | None = None


def _ensure_lock() -> asyncio.Lock:
    global _DIMENSION_LOCK
    if _DIMENSION_LOCK is None:
        _DIMENSION_LOCK = asyncio.Lock()
    return _DIMENSION_LOCK


async def embed(
    texts: list[str],
    *,
    settings: object | None = None,
    backend: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    """Produce embedding vectors for the given texts.

    Args:
        texts: list of strings to embed. Empty list returns [].
        settings: ParallectSettings-like object (used to resolve backend).
        backend/model/base_url: per-call overrides; normally omitted.

    Returns one vector per input, preserving input order.
    """
    if not texts:
        return []

    spec = resolve_embeddings_backend(
        settings=settings,
        override_backend=backend,
        override_model=model,
        override_base_url=base_url,
    )

    return await _dispatch_embed(spec, texts)


async def embed_dimensions(
    *,
    settings: object | None = None,
    backend: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> int:
    """Return the output dimension of the configured embeddings model.

    First call probes the backend with a one-token input; subsequent calls
    hit the cache.
    """
    spec = resolve_embeddings_backend(
        settings=settings,
        override_backend=backend,
        override_model=model,
        override_base_url=base_url,
    )

    cache_key = (spec.kind, spec.model)
    if cache_key in _DIMENSION_CACHE:
        return _DIMENSION_CACHE[cache_key]

    # Fast path: known model.
    known = KNOWN_EMBEDDING_DIMS.get(spec.model)
    if known is not None:
        _DIMENSION_CACHE[cache_key] = known
        return known

    # Slow path: probe with a sentinel input under a lock so concurrent
    # callers don't all hammer the server on the first request.
    lock = _ensure_lock()
    async with lock:
        if cache_key in _DIMENSION_CACHE:
            return _DIMENSION_CACHE[cache_key]
        vectors = await _dispatch_embed(spec, ["."])
        if not vectors or not vectors[0]:
            raise RuntimeError(
                f"Embedding probe returned an empty vector for "
                f"backend={spec.kind} model={spec.model}."
            )
        dim = len(vectors[0])
        _DIMENSION_CACHE[cache_key] = dim
        return dim


async def _dispatch_embed(spec: BackendSpec, texts: list[str]) -> list[list[float]]:
    if spec.kind == "gemini":
        return await call_gemini_embeddings(spec, texts)
    if spec.kind in OPENAI_COMPAT_BACKENDS:
        return await call_openai_compat_embeddings(spec, texts)
    raise ValueError(f"Unsupported embeddings backend: {spec.kind}")


def _reset_caches() -> None:
    """Test helper -- clear dimension cache between tests."""
    _DIMENSION_CACHE.clear()


__all__ = ["embed", "embed_dimensions"]
