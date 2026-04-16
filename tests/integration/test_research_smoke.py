"""Integration: end-to-end `research()` with mocked providers + real LiteLLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

from parallect.config_mod.settings import ParallectSettings
from parallect.orchestrator.parallel import research
from parallect.providers import ProviderResult


class _StubProvider:
    """Provider that returns canned markdown and skips claim extraction."""

    def __init__(self, name: str, text: str):
        self._name = name
        self._text = text

    @property
    def name(self) -> str:
        return self._name

    async def research(self, query: str) -> ProviderResult:
        return ProviderResult(
            provider=self._name,
            status="completed",
            report_markdown=f"# {self._name}\n\n{self._text}\n\nQuery: {query}",
            model="stub-model",
            duration_seconds=0.01,
            cost_usd=0.0,
        )

    def is_available(self) -> bool:
        return True

    def estimate_cost(self, query: str) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_research_end_to_end_with_litellm(tmp_path, monkeypatch):
    litellm_url = os.environ.get("LITELLM_URL", "http://localhost:14000")
    monkeypatch.setenv("PARALLECT_SYNTHESIS_BASE_URL", litellm_url)
    monkeypatch.setenv("LITELLM_API_KEY", "sk-parallect-test")

    settings = ParallectSettings(
        synthesis_backend="litellm",
        synthesis_api_key_env="LITELLM_API_KEY",
        synthesis_model=os.environ.get("LITELLM_MODEL", "test-llama"),
    )

    providers: list[Any] = [
        _StubProvider("alpha", "Alpha believes the sky is blue."),
        _StubProvider("beta", "Beta thinks the sky might be green at dusk."),
    ]

    out_path = tmp_path / "out.prx"
    bundle = await research(
        query="What color is the sky?",
        providers=providers,
        synthesize_with="litellm",
        synthesis_base_url=litellm_url,
        no_synthesis=False,
        extract_claims_flag=False,
        budget_cap_usd=None,
        timeout_per_provider=30.0,
        output=out_path,
        no_sign=True,
        settings=settings,
    )
    assert out_path.exists()
    assert bundle.manifest.has_synthesis
    assert bundle.synthesis_md
