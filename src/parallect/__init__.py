"""parallect: Open-source multi-provider AI deep research (BYOK).

Public API:
    - research(): High-level fan-out, synthesize, extract claims, write .prx
    - Providers: PerplexityProvider, GeminiProvider, OpenAIDRProvider, etc.
    - Synthesis: synthesize, concatenate
    - Claims: extract_claims
    - Config: ParallectSettings
"""

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

__version__ = "0.2.0"

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
