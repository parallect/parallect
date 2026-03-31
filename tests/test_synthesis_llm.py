"""Tests for LLM synthesis module (backend routing and mocked API calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from parallect.providers import ProviderResult
from parallect.synthesis.llm import (
    SynthesisResult,
    _build_synthesis_prompt,
    synthesize,
)

_FAKE_REQUEST = httpx.Request("POST", "http://test")


def _make_results() -> list[ProviderResult]:
    return [
        ProviderResult(
            provider="alpha",
            status="completed",
            report_markdown="# Alpha\n\nAlpha says yes.",
        ),
        ProviderResult(
            provider="beta",
            status="completed",
            report_markdown="# Beta\n\nBeta says no.",
        ),
    ]


# ---------------------------------------------------------------------------
# _build_synthesis_prompt
# ---------------------------------------------------------------------------


class TestBuildSynthesisPrompt:
    def test_includes_query(self):
        prompt = _build_synthesis_prompt("quantum computing", _make_results())
        assert "quantum computing" in prompt

    def test_includes_all_provider_reports(self):
        prompt = _build_synthesis_prompt("test", _make_results())
        assert "Alpha says yes" in prompt
        assert "Beta says no" in prompt

    def test_includes_provider_names(self):
        prompt = _build_synthesis_prompt("test", _make_results())
        assert "alpha" in prompt
        assert "beta" in prompt

    def test_empty_results(self):
        prompt = _build_synthesis_prompt("test", [])
        assert "test" in prompt


# ---------------------------------------------------------------------------
# synthesize routing
# ---------------------------------------------------------------------------


def _openai_compat_response(content: str = "# Synthesized") -> dict:
    return {
        "choices": [{"message": {"content": content}}],
    }


def _anthropic_response(content: str = "# Synthesized") -> dict:
    return {
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": 100, "output_tokens": 200},
    }


class TestSynthesizeRouting:
    @pytest.mark.asyncio
    async def test_empty_results_returns_no_data(self):
        result = await synthesize("test", [], model="anthropic")
        assert "No provider results" in result.report_markdown

    @pytest.mark.asyncio
    async def test_anthropic_backend(self):
        resp_json = _anthropic_response("# Unified Report")
        mock_resp = httpx.Response(200, json=resp_json, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await synthesize(
                "test query",
                _make_results(),
                model="anthropic",
                api_key="test-key",
            )
        assert isinstance(result, SynthesisResult)
        assert "Unified Report" in result.report_markdown
        assert result.duration_seconds is not None

    @pytest.mark.asyncio
    async def test_ollama_backend(self):
        resp_json = _openai_compat_response("# Ollama Synth")
        mock_resp = httpx.Response(200, json=resp_json, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await synthesize(
                "test query",
                _make_results(),
                model="ollama/llama3.2",
            )
        assert "Ollama Synth" in result.report_markdown
        assert result.model == "llama3.2"

    @pytest.mark.asyncio
    async def test_lmstudio_backend(self):
        resp_json = _openai_compat_response("# LMS Synth")
        mock_resp = httpx.Response(200, json=resp_json, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await synthesize(
                "test query",
                _make_results(),
                model="lmstudio/gpt-oss-120b",
            )
        assert "LMS Synth" in result.report_markdown
        assert result.model == "gpt-oss-120b"

    @pytest.mark.asyncio
    async def test_lmstudio_default_model(self):
        resp_json = _openai_compat_response("# Default")
        mock_resp = httpx.Response(200, json=resp_json, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await synthesize(
                "test query",
                _make_results(),
                model="lmstudio",
            )
        assert result.model == "default"

    @pytest.mark.asyncio
    async def test_openai_compat_fallback(self):
        resp_json = _openai_compat_response("# OpenAI")
        mock_resp = httpx.Response(200, json=resp_json, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await synthesize(
                "test query",
                _make_results(),
                model="gpt-4o",
                api_key="test-key",
            )
        assert "OpenAI" in result.report_markdown

    @pytest.mark.asyncio
    async def test_anthropic_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key"):
                await synthesize(
                    "test query",
                    _make_results(),
                    model="anthropic",
                    api_key=None,
                )
