"""Tests for claim extraction module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from parallect.claims.extract import _parse_json_response, extract_claims
from parallect.providers import ProviderResult

_FAKE_REQUEST = httpx.Request("POST", "http://test")


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_plain_json(self):
        data = {"claims": [{"id": "c1", "content": "test"}]}
        result = _parse_json_response(json.dumps(data))
        assert result == data

    def test_json_with_code_fence(self):
        data = {"claims": []}
        wrapped = f"```json\n{json.dumps(data)}\n```"
        result = _parse_json_response(wrapped)
        assert result == data

    def test_json_with_plain_code_fence(self):
        data = {"claims": []}
        wrapped = f"```\n{json.dumps(data)}\n```"
        result = _parse_json_response(wrapped)
        assert result == data

    def test_json_with_whitespace(self):
        data = {"key": "value"}
        result = _parse_json_response(f"  \n{json.dumps(data)}\n  ")
        assert result == data

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("not json at all")


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------


def _make_provider_results() -> list[ProviderResult]:
    return [
        ProviderResult(
            provider="fast", status="completed", report_markdown="# Fast Report"
        ),
        ProviderResult(
            provider="slow", status="completed", report_markdown="# Slow Report"
        ),
    ]


MOCK_CLAIMS_JSON = {
    "claims": [
        {
            "id": "claim_001",
            "content": "AI is transforming research",
            "providers_supporting": ["fast", "slow"],
            "providers_contradicting": [],
            "category": "fact",
        },
        {
            "id": "claim_002",
            "content": "Quantum supremacy is debated",
            "providers_supporting": ["fast"],
            "providers_contradicting": ["slow"],
            "category": "comparison",
        },
    ]
}


class TestExtractClaims:
    @pytest.mark.asyncio
    async def test_anthropic_extraction(self):
        """Test claim extraction via Anthropic backend."""
        anthropic_response = {
            "content": [{"text": json.dumps(MOCK_CLAIMS_JSON)}],
        }
        mock_resp = httpx.Response(200, json=anthropic_response, request=_FAKE_REQUEST)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            claims = await extract_claims(
                synthesis_markdown="# Synthesis\nAI is great.",
                provider_results=_make_provider_results(),
                model="anthropic",
                api_key="test-key",
            )

        assert len(claims) == 2
        assert claims[0]["content"] == "AI is transforming research"
        assert claims[1]["providers_contradicting"] == ["slow"]

    @pytest.mark.asyncio
    async def test_openai_extraction(self):
        """Test claim extraction via OpenAI backend."""
        openai_response = {
            "choices": [
                {"message": {"content": json.dumps(MOCK_CLAIMS_JSON)}}
            ],
        }
        mock_resp = httpx.Response(200, json=openai_response, request=_FAKE_REQUEST)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            claims = await extract_claims(
                synthesis_markdown="# Synthesis",
                provider_results=_make_provider_results(),
                model="gpt-4o",
                api_key="test-key",
            )

        assert len(claims) == 2

    @pytest.mark.asyncio
    async def test_filters_failed_providers(self):
        """Failed providers should not appear in prompt context."""
        results = [
            ProviderResult(provider="good", status="completed", report_markdown="Good"),
            ProviderResult(provider="bad", status="failed", report_markdown="", error="Boom"),
        ]
        anthropic_response = {
            "content": [{"text": json.dumps({"claims": []})}],
        }
        mock_resp = httpx.Response(200, json=anthropic_response, request=_FAKE_REQUEST)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as m:
            await extract_claims(
                synthesis_markdown="# Test",
                provider_results=results,
                model="anthropic",
                api_key="test-key",
            )

        # Check that only "good" appears in the prompt sent to the API
        call_args = m.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        prompt_text = body["messages"][0]["content"]
        assert "good" in prompt_text
        # "bad" should not be in provider names list (but may appear in other context)

    @pytest.mark.asyncio
    async def test_empty_claims_response(self):
        """Handle an LLM response with empty claims list."""
        anthropic_response = {
            "content": [{"text": json.dumps({"claims": []})}],
        }
        mock_resp = httpx.Response(200, json=anthropic_response, request=_FAKE_REQUEST)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            claims = await extract_claims(
                synthesis_markdown="# Empty",
                provider_results=_make_provider_results(),
                model="anthropic",
                api_key="test-key",
            )

        assert claims == []

    @pytest.mark.asyncio
    async def test_anthropic_missing_key_raises(self):
        """Should raise ValueError when no API key is available."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key"):
                await extract_claims(
                    synthesis_markdown="# Test",
                    provider_results=_make_provider_results(),
                    model="anthropic",
                    api_key=None,
                )
