"""parallect: Open-source multi-provider AI deep research (BYOK).

Public API:
    - research(): High-level fan-out, synthesize, extract claims, write .prx
    - Providers: PerplexityProvider, GeminiProvider, OpenAIDRProvider, etc.
    - Synthesis: synthesize, concatenate
    - Claims: extract_claims
    - Config: ParallectSettings

Note on distribution name:
    Starting with 0.3.0, this package's canonical PyPI distribution is
    ``parallect-cli``. The legacy ``parallect`` distribution still exists on
    PyPI as a compatibility shim that depends on ``parallect-cli`` and
    installs a sentinel module (``parallect_deprecation``); if that sentinel
    is importable we emit a DeprecationWarning below so users migrate their
    ``pip install`` command. The import name, CLI command, config dir, and
    public API are unchanged.
"""

import warnings as _warnings
from importlib.util import find_spec as _find_spec

if _find_spec("parallect_deprecation") is not None:
    _warnings.warn(
        "The 'parallect' PyPI distribution has been renamed to 'parallect-cli'. "
        "Please update your install command: `pip install parallect-cli`. "
        "The 'parallect' name will continue to work as a compatibility shim, "
        "but will hard-error in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )

from prx_spec import (
    BundleData,
    ProviderData,
    ValidationResult,
    generate_keypair,
    read_bundle,
    sign_attestation,
    validate_archive,
    validate_bundle,
    verify_attestation,
    write_bundle,
)

from parallect.config_mod.settings import ParallectSettings
from parallect.orchestrator.parallel import research
from parallect.providers.base import AsyncResearchProvider, ResearchProvider
from parallect.synthesis.concat import concatenate
from parallect.synthesis.llm import synthesize

__version__ = "0.3.0"

__all__ = [
    "AsyncResearchProvider",
    "BundleData",
    "ParallectSettings",
    "ProviderData",
    "ResearchProvider",
    "ValidationResult",
    "concatenate",
    "generate_keypair",
    "read_bundle",
    "research",
    "sign_attestation",
    "synthesize",
    "validate_archive",
    "validate_bundle",
    "verify_attestation",
    "write_bundle",
]
