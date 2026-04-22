"""Sentinel module for the legacy ``parallect`` PyPI distribution.

This module is intentionally empty. Its presence in ``sys.path`` tells the
canonical ``parallect-cli`` package (via its ``parallect/__init__.py``) that
the caller installed through the legacy ``pip install parallect`` path, at
which point a ``DeprecationWarning`` is emitted pointing them at
``pip install parallect-cli``.

Do not import from this module directly. Do not rely on its existence.
It will be removed when the legacy ``parallect`` distribution is flipped
from "shim" to "hard-error" in a future release.
"""

__all__: list[str] = []
