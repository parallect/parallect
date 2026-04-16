"""Integration: hit a real LiteLLM container via the OpenAI-compat adapter.

Marked with `pytest.mark.integration` so it's skipped unless
`PARALLECT_INTEGRATION=1` is set (see `conftest.py`).
"""

from __future__ import annotations

import os

import pytest

from parallect.backends import resolve_synthesis_backend
from parallect.backends.adapters import call_openai_compat_chat

pytestmark = pytest.mark.integration


def _litellm_url() -> str:
    return os.environ.get("LITELLM_URL", "http://localhost:14000")


def _litellm_model() -> str:
    return os.environ.get("LITELLM_MODEL", "test-llama")


class _StubSettings:
    synthesis_backend = "litellm"
    synthesis_model = ""
    synthesis_base_url = ""
    synthesis_api_key_env = "LITELLM_API_KEY"
    embeddings_backend = ""
    embeddings_model = ""
    embeddings_base_url = ""
    embeddings_api_key_env = ""
    openai_api_key = ""
    anthropic_api_key = ""
    google_api_key = ""
    openrouter_api_key = ""
    litellm_api_key = ""


@pytest.mark.asyncio
async def test_resolver_points_at_litellm(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-parallect-test")
    monkeypatch.setenv("PARALLECT_SYNTHESIS_BASE_URL", _litellm_url())
    settings = _StubSettings()
    spec = resolve_synthesis_backend(
        cli_model=_litellm_model(),
        settings=settings,
    )
    assert spec.base_url == _litellm_url()
    assert spec.api_key == "sk-parallect-test"


@pytest.mark.asyncio
async def test_chat_roundtrip(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-parallect-test")
    monkeypatch.setenv("PARALLECT_SYNTHESIS_BASE_URL", _litellm_url())
    settings = _StubSettings()
    spec = resolve_synthesis_backend(
        cli_model=_litellm_model(),
        settings=settings,
    )
    out = await call_openai_compat_chat(
        spec,
        prompt="Say hello in one word.",
        system_prompt="Reply with exactly one word.",
        timeout=60.0,
    )
    assert out["content"], "LiteLLM returned an empty response"
    assert isinstance(out["tokens"], dict)
