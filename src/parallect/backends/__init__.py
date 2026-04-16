"""Shared backend resolution for synthesis + embeddings.

Both synthesis and embeddings share the same set of OpenAI-compatible servers
(LiteLLM, OpenRouter, self-hosted OpenAI-compat, vLLM, Ollama, LM Studio, ...)
plus two dedicated paths (Anthropic, Gemini) for synthesis only.

`BackendSpec` is the resolved triple (backend name, base_url, api_key) that
callers hand to the correct adapter. Resolution obeys the standard Parallect
precedence order: explicit argument > environment variable > config > default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

SynthesisBackend = Literal[
    "openai",
    "openrouter",
    "litellm",
    "gemini",
    "anthropic",
    "ollama",
    "lmstudio",
    "custom",
]

EmbeddingsBackend = Literal[
    "openai",
    "openrouter",
    "litellm",
    "gemini",
    "ollama",
    "lmstudio",
    "custom",
]

# Backends that speak the OpenAI /v1/chat/completions + /v1/embeddings shape.
# They all share a single adapter and differ only in base_url and api_key_env.
OPENAI_COMPAT_BACKENDS: frozenset[str] = frozenset(
    {"openai", "openrouter", "litellm", "ollama", "lmstudio", "custom"}
)

# Default base URLs per backend. `custom` has no default -- user must supply it.
DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "litellm": "http://localhost:4000",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
}

# Default env var name holding the API key for each backend. Local backends
# have no key requirement, so we fall back to a dummy placeholder downstream.
DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "litellm": "LITELLM_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "lmstudio": "LMSTUDIO_API_KEY",
    "custom": "CUSTOM_LLM_API_KEY",
}

# Default synthesis/embedding models per backend. Only used when the user has
# not picked one explicitly. Picked to be cheap, fast, and BYOK-friendly.
DEFAULT_SYNTHESIS_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "openrouter": "google/gemini-2.5-flash-lite-preview",
    "litellm": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3.2",
    "lmstudio": "default",
    "custom": "",
}

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "openrouter": "openai/text-embedding-3-small",
    "litellm": "text-embedding-3-small",
    "gemini": "text-embedding-004",
    "ollama": "nomic-embed-text",
    "lmstudio": "text-embedding-nomic-embed-text-v1.5",
    "custom": "",
}

# Backends that do NOT support embeddings. Anthropic has no embeddings endpoint.
EMBEDDINGS_UNSUPPORTED: frozenset[str] = frozenset({"anthropic"})

# Known embedding output dimensions. Used as a fast-path before probing.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-004": 768,
    "nomic-embed-text": 768,
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendSpec:
    """Resolved backend info ready for a concrete HTTP adapter.

    `kind` is either the canonical backend name (openai, ollama, ...) or
    `"anthropic"` / `"gemini"` for their dedicated paths. Adapters dispatch
    on this field.
    """

    kind: str
    base_url: str
    api_key: str
    model: str
    api_key_env: str


def _env_lookup(name: str) -> str:
    """Fetch env var safely (empty string if unset)."""
    return os.environ.get(name, "") or ""


def resolve_synthesis_backend(
    *,
    cli_base_url: str | None = None,
    cli_model: str | None = None,
    settings: object | None = None,
) -> BackendSpec:
    """Resolve the synthesis backend using full precedence.

    Precedence (highest to lowest):
      1. `cli_base_url` / `cli_model` (explicit flags)
      2. `PARALLECT_SYNTHESIS_BASE_URL` env
      3. `[synthesis]` config in `settings`
      4. Hard defaults (anthropic + claude-sonnet-4)
    """
    backend, base_url, api_key_env, model = _read_synthesis_from_settings(settings)

    # Env override (PARALLECT_SYNTHESIS_BASE_URL) implies a custom/openai-compat backend.
    env_base_url = _env_lookup("PARALLECT_SYNTHESIS_BASE_URL")
    if env_base_url and not cli_base_url:
        base_url = env_base_url
        # When the user points at a URL without naming a backend, treat it as
        # OpenAI-compatible -- that's the whole point of this env var.
        if backend == "anthropic":
            backend = "custom"

    # CLI flag wins.
    if cli_base_url:
        base_url = cli_base_url
        if backend in ("anthropic", "gemini"):
            backend = "custom"
    if cli_model:
        model = cli_model

    # Fill in defaults for anything still blank.
    if not base_url:
        base_url = DEFAULT_BASE_URLS.get(backend, "")
    if not model:
        model = DEFAULT_SYNTHESIS_MODELS.get(backend, "")

    if backend == "custom" and not base_url:
        raise ValueError(
            "Backend 'custom' requires an explicit base_url "
            "(CLI --synthesis-base-url, env PARALLECT_SYNTHESIS_BASE_URL, "
            "or [synthesis].base_url in config)."
        )

    api_key = _resolve_api_key(backend, api_key_env, settings)

    return BackendSpec(
        kind=backend,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        api_key_env=api_key_env or DEFAULT_API_KEY_ENV.get(backend, ""),
    )


def resolve_embeddings_backend(
    *,
    settings: object | None = None,
    override_backend: str | None = None,
    override_model: str | None = None,
    override_base_url: str | None = None,
) -> BackendSpec:
    """Resolve the embeddings backend.

    Embeddings doesn't yet have a CLI surface; the primary precedence is env
    + config + defaults. `override_*` exist so `parallect.embeddings.embed()`
    callers can force a particular backend per call.
    """
    backend, base_url, api_key_env, model = _read_embeddings_from_settings(settings)

    if override_backend:
        backend = override_backend
    if override_base_url:
        base_url = override_base_url
    if override_model:
        model = override_model

    if backend in EMBEDDINGS_UNSUPPORTED:
        raise ValueError(
            f"Backend '{backend}' does not support embeddings -- "
            "use openai, openrouter, litellm, gemini, ollama, lmstudio, or custom."
        )

    if not base_url:
        base_url = DEFAULT_BASE_URLS.get(backend, "")
    if not model:
        model = DEFAULT_EMBEDDING_MODELS.get(backend, "")

    if backend == "custom" and not base_url:
        raise ValueError(
            "Backend 'custom' requires an explicit base_url "
            "([embeddings].base_url in config or override_base_url=...)."
        )

    api_key = _resolve_api_key(backend, api_key_env, settings)

    return BackendSpec(
        kind=backend,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        api_key_env=api_key_env or DEFAULT_API_KEY_ENV.get(backend, ""),
    )


def _read_synthesis_from_settings(
    settings: object | None,
) -> tuple[str, str, str, str]:
    if settings is None:
        try:
            from parallect.config_mod.settings import ParallectSettings
            settings = ParallectSettings.load()
        except Exception:
            return ("anthropic", "", "", "")
    backend = getattr(settings, "synthesis_backend", "") or "anthropic"
    base_url = getattr(settings, "synthesis_base_url", "") or ""
    api_key_env = getattr(settings, "synthesis_api_key_env", "") or ""
    model = getattr(settings, "synthesis_model", "") or ""
    return (backend, base_url, api_key_env, model)


def _read_embeddings_from_settings(
    settings: object | None,
) -> tuple[str, str, str, str]:
    if settings is None:
        try:
            from parallect.config_mod.settings import ParallectSettings
            settings = ParallectSettings.load()
        except Exception:
            return ("openai", "", "", "")
    backend = getattr(settings, "embeddings_backend", "") or "openai"
    base_url = getattr(settings, "embeddings_base_url", "") or ""
    api_key_env = getattr(settings, "embeddings_api_key_env", "") or ""
    model = getattr(settings, "embeddings_model", "") or ""
    return (backend, base_url, api_key_env, model)


def _resolve_api_key(
    backend: str,
    api_key_env: str,
    settings: object | None,
) -> str:
    """Resolve the API key to use, preferring explicit env vars over settings."""
    # Explicit api_key_env always wins -- user said exactly where the key lives.
    if api_key_env:
        value = _env_lookup(api_key_env)
        if value:
            return value

    # Fall back to the conventional env var for this backend.
    default_env = DEFAULT_API_KEY_ENV.get(backend, "")
    if default_env:
        value = _env_lookup(default_env)
        if value:
            return value

    # Fall back to settings fields (covers the BYOK case where the CLI already
    # holds the key from ~/.config/parallect/config.toml).
    if settings is not None:
        settings_field_map = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "gemini": "google_api_key",
            "openrouter": "openrouter_api_key",
            "litellm": "litellm_api_key",
        }
        attr = settings_field_map.get(backend)
        if attr:
            value = getattr(settings, attr, "") or ""
            if value:
                return value

    # Local backends don't need a key; return dummy token accepted by LM Studio / Ollama.
    if backend in {"ollama", "lmstudio"}:
        return "not-needed"

    return ""


# ---------------------------------------------------------------------------
# Validation helpers (used by config + tests)
# ---------------------------------------------------------------------------


def is_valid_synthesis_backend(name: str) -> bool:
    return name in {
        "openai", "openrouter", "litellm", "gemini",
        "anthropic", "ollama", "lmstudio", "custom",
    }


def is_valid_embeddings_backend(name: str) -> bool:
    return name in {
        "openai", "openrouter", "litellm", "gemini",
        "ollama", "lmstudio", "custom",
    }
