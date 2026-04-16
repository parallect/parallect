"""LLM-based synthesis of multiple provider reports.

Backends: OpenAI / OpenRouter / LiteLLM / any OpenAI-compat server share one
code path (see `parallect.backends.adapters.call_openai_compat_chat`), plus
dedicated adapters for Anthropic and Gemini.

Resolution precedence for the synthesis backend:
    1. `--synthesis-base-url` CLI flag (+ `--synthesize-with` model)
    2. `PARALLECT_SYNTHESIS_BASE_URL` env var
    3. `[synthesis]` section in config.toml
    4. Hard defaults (anthropic, claude-sonnet-4)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from parallect.backends import (
    OPENAI_COMPAT_BACKENDS,
    BackendSpec,
    resolve_synthesis_backend,
)
from parallect.backends.adapters import (
    call_anthropic_chat,
    call_gemini_chat,
    call_openai_compat_chat,
)
from parallect.providers import ProviderResult

SYNTHESIS_SYSTEM_PROMPT = """\
You are a research synthesis expert. You will receive multiple research reports \
on the same topic from different AI providers. Your task is to produce a single \
unified report that:

1. Identifies key agreements across providers
2. Highlights disagreements or contradictions
3. Synthesizes the strongest arguments and evidence
4. Provides a balanced, comprehensive overview
5. Uses markdown formatting with clear sections

Do not attribute statements to specific providers. Write as a unified report. \
If providers disagree on a point, present both perspectives."""


@dataclass
class SynthesisResult:
    """Result of synthesis."""

    report_markdown: str
    model: str | None = None
    cost_usd: float | None = None
    duration_seconds: float | None = None
    tokens: dict | None = None


def _build_synthesis_prompt(query: str, results: list[ProviderResult]) -> str:
    """Build the synthesis prompt from provider results."""
    sections = []
    for r in results:
        sections.append(f"### Report from {r.provider}\n\n{r.report_markdown}")

    reports_text = "\n\n---\n\n".join(sections)

    return (
        f"# Research Query\n\n{query}\n\n"
        f"# Provider Reports\n\n{reports_text}\n\n"
        f"# Task\n\n"
        f"Synthesize the above reports into a single unified research report."
    )


# Legacy model->backend mapping used for back-compat with the old string-based
# `synthesize_with="anthropic"` / `synthesize_with="ollama/llama3.2"` API.
_LEGACY_SHORT_NAMES: dict[str, str] = {
    "openai": "openai",
    "gemini": "gemini",
    "anthropic": "anthropic",
    "grok": "openrouter",          # no native grok; pragmatic default
    "perplexity": "openrouter",    # same
    "openrouter": "openrouter",
    "litellm": "litellm",
    "ollama": "ollama",
    "lmstudio": "lmstudio",
}

_LEGACY_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-20250514",
    "grok": "grok-3",
    "perplexity": "sonar",
    "ollama": "llama3.2",
    "lmstudio": "default",
}


def _spec_from_legacy_model(
    model: str,
    api_key: str | None,
    settings: object | None,
) -> BackendSpec:
    """Map the legacy `synthesize_with="anthropic"` string to a BackendSpec.

    The historical CLI calls this module with strings like `"anthropic"`,
    `"ollama/llama3.2"`, `"gpt-4o"`, etc. We preserve that contract while
    routing through the new BackendSpec type so all paths share one adapter.
    """
    # Split off `ollama/llama3.2` -> ("ollama", "llama3.2")
    prefix, _, suffix = model.partition("/")
    backend_key = _LEGACY_SHORT_NAMES.get(prefix)

    if backend_key is not None:
        resolved_model = suffix or _LEGACY_DEFAULT_MODELS.get(prefix, model)
        spec = resolve_synthesis_backend(
            cli_base_url=None,
            cli_model=resolved_model,
            settings=_synthetic_settings_for(backend_key, settings),
        )
        if api_key:
            spec = _with_api_key(spec, api_key)
        return spec

    # Unknown model string -> treat it as an OpenAI-compat model name, route
    # through the default OpenAI base URL (preserves the pre-wave1 fallback).
    spec = resolve_synthesis_backend(
        cli_base_url=None,
        cli_model=model,
        settings=_synthetic_settings_for("openai", settings),
    )
    if api_key:
        spec = _with_api_key(spec, api_key)
    return spec


def _synthetic_settings_for(backend: str, base: object | None) -> object:
    """Produce a settings-like object with the requested backend pinned.

    We need this because the legacy call signature passes only a model string,
    not a settings object. The new resolver reads the backend off settings, so
    we fake a minimal shim.
    """
    class _Shim:
        synthesis_backend = backend
        synthesis_base_url = ""
        synthesis_api_key_env = ""
        synthesis_model = ""

        def __getattr__(self, name):  # pragma: no cover - passthrough
            if base is not None:
                return getattr(base, name, "")
            return ""

    return _Shim()


def _with_api_key(spec: BackendSpec, api_key: str) -> BackendSpec:
    """Return a new BackendSpec with api_key replaced."""
    return BackendSpec(
        kind=spec.kind,
        base_url=spec.base_url,
        api_key=api_key,
        model=spec.model,
        api_key_env=spec.api_key_env,
    )


async def synthesize(
    query: str,
    provider_results: list[ProviderResult],
    model: str = "anthropic",
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    settings: object | None = None,
) -> SynthesisResult:
    """Produce a unified synthesis report from multiple provider reports.

    Preserves the pre-wave1 signature (positional `model` + `api_key`). The
    new `base_url` kwarg + `settings` kwarg are additive.
    """
    if not provider_results:
        return SynthesisResult(report_markdown="No provider results to synthesize.")

    prompt = _build_synthesis_prompt(query, provider_results)
    start = time.monotonic()

    # Prefer full backend resolution when CLI/settings info is supplied;
    # otherwise fall back to the legacy model-string mapping.
    if base_url is not None or _env_has("PARALLECT_SYNTHESIS_BASE_URL") or (
        settings is not None and getattr(settings, "synthesis_backend", "")
    ):
        spec = resolve_synthesis_backend(
            cli_base_url=base_url,
            cli_model=model if "/" not in model and model not in _LEGACY_SHORT_NAMES else None,
            settings=settings,
        )
        if api_key:
            spec = _with_api_key(spec, api_key)
    else:
        spec = _spec_from_legacy_model(model, api_key, settings)

    content_block = await _dispatch_chat(spec, prompt)

    duration = round(time.monotonic() - start, 2)
    return SynthesisResult(
        report_markdown=content_block["content"],
        model=content_block.get("model") or spec.model,
        cost_usd=_estimate_cost(spec, content_block.get("tokens") or {}),
        duration_seconds=duration,
        tokens=content_block.get("tokens"),
    )


def _env_has(name: str) -> bool:
    return bool(os.environ.get(name, ""))


async def _dispatch_chat(spec: BackendSpec, prompt: str) -> dict:
    if spec.kind == "anthropic":
        return await call_anthropic_chat(spec, prompt, SYNTHESIS_SYSTEM_PROMPT)
    if spec.kind == "gemini":
        return await call_gemini_chat(spec, prompt, SYNTHESIS_SYSTEM_PROMPT)
    if spec.kind in OPENAI_COMPAT_BACKENDS:
        return await call_openai_compat_chat(spec, prompt, SYNTHESIS_SYSTEM_PROMPT)
    raise ValueError(f"Unsupported synthesis backend: {spec.kind}")


def _estimate_cost(spec: BackendSpec, tokens: dict) -> float | None:
    # Rough per-backend heuristic only -- precise pricing is tracked upstream.
    if not tokens:
        return None
    if spec.kind == "anthropic":
        return 0.03
    if spec.kind in ("openai", "openrouter", "litellm", "custom"):
        return 0.05
    if spec.kind == "gemini":
        return 0.01
    return None


# ---------------------------------------------------------------------------
# Legacy internal helpers (retained for tests that patch them directly)
# ---------------------------------------------------------------------------


async def _synthesize_anthropic(prompt: str, api_key: str | None = None) -> SynthesisResult:
    """Retained for back-compat. New callers should use `synthesize()`."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("Anthropic API key required for synthesis")
    spec = BackendSpec(
        kind="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key=key,
        model="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
    )
    result = await call_anthropic_chat(spec, prompt, SYNTHESIS_SYSTEM_PROMPT)
    return SynthesisResult(
        report_markdown=result["content"],
        model=result.get("model"),
        cost_usd=0.03,
        tokens=result.get("tokens"),
    )


async def _synthesize_openai_compat(
    prompt: str, model: str, api_key: str | None = None
) -> SynthesisResult:
    """Retained for back-compat."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    spec = BackendSpec(
        kind="openai",
        base_url="https://api.openai.com/v1",
        api_key=key,
        model=model,
        api_key_env="OPENAI_API_KEY",
    )
    result = await call_openai_compat_chat(spec, prompt, SYNTHESIS_SYSTEM_PROMPT)
    return SynthesisResult(
        report_markdown=result["content"],
        model=result.get("model"),
        cost_usd=0.05,
        tokens=result.get("tokens"),
    )
