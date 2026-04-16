"""Tests for parallect.backends.adapters -- the HTTP layer.

We mock `httpx.AsyncClient.post` end-to-end. The goal is to lock in:

  - Correct URL / headers / payload shape per backend
  - Error paths (401, 429, 500) surface as `BackendError`
  - The normalized response dict shape for downstream callers
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from parallect.backends import BackendSpec
from parallect.backends.adapters import (
    BackendError,
    call_anthropic_chat,
    call_gemini_chat,
    call_gemini_embeddings,
    call_openai_compat_chat,
    call_openai_compat_embeddings,
)

_FAKE_REQUEST = httpx.Request("POST", "http://test")


def _openai_chat_response(content="# Hello", model="gpt-4o-mini"):
    return {
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _anthropic_chat_response(content="# Hello"):
    return {
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": 5, "output_tokens": 15},
    }


def _gemini_chat_response(content="# Hello"):
    return {
        "candidates": [
            {"content": {"parts": [{"text": content}]}},
        ],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 22,
            "totalTokenCount": 33,
        },
    }


def _openai_embed_response(n=2, dim=3):
    return {
        "model": "text-embedding-3-small",
        "data": [
            {"index": i, "embedding": [0.1 * (i + 1)] * dim}
            for i in range(n)
        ],
    }


def _spec_openai_compat(kind="openai", base="https://api.openai.com/v1"):
    return BackendSpec(
        kind=kind,
        base_url=base,
        api_key="sk-test",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    )


def _spec_anthropic():
    return BackendSpec(
        kind="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test",
        model="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
    )


def _spec_gemini(api_key="g-test"):
    return BackendSpec(
        kind="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key=api_key,
        model="gemini-2.5-flash",
        api_key_env="GOOGLE_API_KEY",
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible chat
# ---------------------------------------------------------------------------


class TestOpenAICompatChat:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        resp = httpx.Response(200, json=_openai_chat_response("# Result"), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            out = await call_openai_compat_chat(
                _spec_openai_compat(),
                prompt="hello",
                system_prompt="sys",
            )
        assert out["content"] == "# Result"
        assert out["tokens"] == {"input": 10, "output": 20, "total": 30}
        mpost.assert_called_once()
        args, kwargs = mpost.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
        assert kwargs["json"]["model"] == "gpt-4o-mini"
        assert any(m["role"] == "system" for m in kwargs["json"]["messages"])

    @pytest.mark.asyncio
    async def test_openrouter_uses_openrouter_base(self):
        spec = _spec_openai_compat(kind="openrouter", base="https://openrouter.ai/api/v1")
        resp = httpx.Response(200, json=_openai_chat_response(), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            await call_openai_compat_chat(spec, "p", "s")
        args, _kwargs = mpost.call_args
        assert args[0] == "https://openrouter.ai/api/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_local_backend_omits_auth_when_key_empty(self):
        spec = BackendSpec(
            kind="ollama",
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3.2",
            api_key_env="OLLAMA_API_KEY",
        )
        resp = httpx.Response(200, json=_openai_chat_response(), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            await call_openai_compat_chat(spec, "p", "s")
        _args, kwargs = mpost.call_args
        assert "Authorization" not in kwargs["headers"]

    @pytest.mark.asyncio
    async def test_extra_headers_merge(self):
        resp = httpx.Response(200, json=_openai_chat_response(), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            await call_openai_compat_chat(
                _spec_openai_compat(),
                "p",
                "s",
                extra_headers={"HTTP-Referer": "https://parallect.ai"},
            )
        _args, kwargs = mpost.call_args
        assert kwargs["headers"]["HTTP-Referer"] == "https://parallect.ai"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403, 429, 500, 502, 503])
    async def test_error_statuses_raise(self, status):
        resp = httpx.Response(status, json={"error": "nope"}, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            with pytest.raises(BackendError) as exc:
                await call_openai_compat_chat(_spec_openai_compat(), "p", "s")
        assert exc.value.status == status
        assert exc.value.backend == "openai"


# ---------------------------------------------------------------------------
# OpenAI-compatible embeddings
# ---------------------------------------------------------------------------


class TestOpenAICompatEmbeddings:
    @pytest.mark.asyncio
    async def test_returns_vectors_in_order(self):
        payload = {
            "model": "text-embedding-3-small",
            "data": [
                {"index": 1, "embedding": [0.2, 0.2, 0.2]},
                {"index": 0, "embedding": [0.1, 0.1, 0.1]},
            ],
        }
        resp = httpx.Response(200, json=payload, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            out = await call_openai_compat_embeddings(
                _spec_openai_compat(),
                texts=["a", "b"],
            )
        assert out[0] == [0.1, 0.1, 0.1]
        assert out[1] == [0.2, 0.2, 0.2]

    @pytest.mark.asyncio
    async def test_posts_to_embeddings_endpoint(self):
        resp = httpx.Response(200, json=_openai_embed_response(), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            await call_openai_compat_embeddings(_spec_openai_compat(), ["a", "b"])
        args, kwargs = mpost.call_args
        assert args[0] == "https://api.openai.com/v1/embeddings"
        assert kwargs["json"]["input"] == ["a", "b"]
        assert kwargs["json"]["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_error_raises(self):
        resp = httpx.Response(429, json={"error": "rate limit"}, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            with pytest.raises(BackendError) as exc:
                await call_openai_compat_embeddings(_spec_openai_compat(), ["a"])
        assert exc.value.status == 429


# ---------------------------------------------------------------------------
# Anthropic chat
# ---------------------------------------------------------------------------


class TestAnthropicChat:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        resp = httpx.Response(200, json=_anthropic_chat_response("# Anthropic"), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            out = await call_anthropic_chat(_spec_anthropic(), "p", "s")
        assert "# Anthropic" in out["content"]
        assert out["tokens"]["input"] == 5
        _args, kwargs = mpost.call_args
        assert kwargs["headers"]["x-api-key"] == "sk-ant-test"
        assert kwargs["headers"]["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    async def test_missing_key_raises(self):
        spec = BackendSpec(
            kind="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key="",
            model="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
        )
        with pytest.raises(ValueError, match="Anthropic API key"):
            await call_anthropic_chat(spec, "p", "s")

    @pytest.mark.asyncio
    async def test_error_raises_backend_error(self):
        resp = httpx.Response(500, json={"error": "boom"}, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            with pytest.raises(BackendError) as exc:
                await call_anthropic_chat(_spec_anthropic(), "p", "s")
        assert exc.value.backend == "anthropic"


# ---------------------------------------------------------------------------
# Gemini chat + embeddings
# ---------------------------------------------------------------------------


class TestGeminiChat:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        resp = httpx.Response(200, json=_gemini_chat_response("# G"), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            out = await call_gemini_chat(_spec_gemini(), "p", "s")
        assert "# G" in out["content"]
        args, _kwargs = mpost.call_args
        assert "key=g-test" in args[0]
        assert "generateContent" in args[0]

    @pytest.mark.asyncio
    async def test_missing_key_raises(self):
        spec = _spec_gemini(api_key="")
        with pytest.raises(ValueError, match="Gemini backend requires an API key"):
            await call_gemini_chat(spec, "p", "s")


class TestGeminiEmbeddings:
    @pytest.mark.asyncio
    async def test_batch_embed(self):
        payload = {
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]},
                {"values": [0.4, 0.5, 0.6]},
            ],
        }
        resp = httpx.Response(200, json=payload, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            out = await call_gemini_embeddings(_spec_gemini(), ["a", "b"])
        assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        args, _kwargs = mpost.call_args
        assert "batchEmbedContents" in args[0]

    @pytest.mark.asyncio
    async def test_missing_key_raises(self):
        spec = _spec_gemini(api_key="")
        with pytest.raises(ValueError, match="Gemini backend requires an API key"):
            await call_gemini_embeddings(spec, ["a"])
