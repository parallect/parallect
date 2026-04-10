"""SHA-256 response hashing for provider integrity verification."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from parallect.providers import ProviderResult


def hash_response(body: str) -> str:
    """Compute SHA-256 hex digest of a response body."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def attach_response_hash(result: ProviderResult, raw_body: str) -> ProviderResult:
    """Attach response hash, size, and timestamp to a ProviderResult.

    Returns the same result object with hash fields populated.
    """
    result.response_hash = hash_response(raw_body)
    result.raw_response_size = len(raw_body.encode("utf-8"))
    result.received_at = datetime.now(timezone.utc)
    return result
