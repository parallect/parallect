"""Tests for OpenAI-compatible and LM Studio provider adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from parallect.providers import ProviderResult
from parallect.providers.lmstudio import LMStudioProvider
from parallect.providers.openai_compat import OpenAICompatibleProvider

_FAKE_REQUEST = httpx.Request("POST", "http://test")


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider
# ---------------------------------------------------------------------------


def _compat_response(content: str = "# Report\n\nHello.", model: str = "test-model") -> dict:
    """Build a mock /v1/chat/completions response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
        },
    }


class TestOpenAICompatibleProvider:
    def test_construction(self):
        p = OpenAICompatibleProvider(
            name="test-provider",
            base_url="http://localhost:9999",
            model="test-model",
        )
        assert p.name == "test-provider"

    def test_estimate_cost_free(self):
        p = OpenAICompatibleProvider(
            name="test", base_url="http://localhost:9999", model="m"
        )
        assert p.estimate_cost("any query") == 0.0

    @pytest.mark.asyncio
    async def test_research_success(self):
        mock_resp = httpx.Response(200, json=_compat_response(), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            p = OpenAICompatibleProvider(
                name="test-provider",
                base_url="http://localhost:9999",
                model="test-model",
            )
            result = await p.research("What is AI?")

        assert isinstance(result, ProviderResult)
        assert result.provider == "test-provider"
        assert result.status == "completed"
        assert "Report" in result.report_markdown
        assert result.tokens is not None
        assert result.tokens["total"] == 150

    @pytest.mark.asyncio
    async def test_research_http_error(self):
        mock_resp = httpx.Response(500, text="Internal Server Error", request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            p = OpenAICompatibleProvider(
                name="test-provider",
                base_url="http://localhost:9999",
                model="m",
            )
            result = await p.research("test")
        # Provider catches errors and returns a failed ProviderResult
        assert result.status == "failed"
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_research_connection_error(self):
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            p = OpenAICompatibleProvider(
                name="test-provider",
                base_url="http://localhost:9999",
                model="m",
            )
            result = await p.research("test")
        # Provider catches errors and returns a failed ProviderResult
        assert result.status == "failed"
        assert "Connection refused" in result.error

    def test_is_available_true(self):
        mock_resp = httpx.Response(200, json={"data": [{"id": "m1"}]})
        with patch("httpx.Client.get", return_value=mock_resp):
            p = OpenAICompatibleProvider(
                name="test", base_url="http://localhost:9999", model="m"
            )
            assert p.is_available() is True

    def test_is_available_connection_error(self):
        with patch("httpx.Client.get", side_effect=httpx.ConnectError("refused")):
            p = OpenAICompatibleProvider(
                name="test", base_url="http://localhost:9999", model="m"
            )
            assert p.is_available() is False


# ---------------------------------------------------------------------------
# LMStudioProvider
# ---------------------------------------------------------------------------


class TestLMStudioProvider:
    def test_default_construction(self):
        p = LMStudioProvider()
        assert p.name == "lmstudio"
        assert "1234" in p.base_url

    def test_custom_model_and_host(self):
        p = LMStudioProvider(model="gpt-oss-120b", host="http://myhost:5555")
        assert p.name == "lmstudio"

    def test_inherits_estimate_cost(self):
        p = LMStudioProvider()
        assert p.estimate_cost("any") == 0.0

    @pytest.mark.asyncio
    async def test_research_delegates_to_parent(self):
        resp_data = _compat_response(model="gpt-oss-120b")
        mock_resp = httpx.Response(200, json=resp_data, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            p = LMStudioProvider(model="gpt-oss-120b")
            result = await p.research("What is LM Studio?")

        assert result.provider == "lmstudio"
        assert result.status == "completed"
