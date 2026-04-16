"""Tests for parallect.embeddings.{embed, embed_dimensions}."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

import parallect.embeddings as embeddings_mod
from parallect.embeddings import embed, embed_dimensions

_FAKE_REQUEST = httpx.Request("POST", "http://test")


class _Settings:
    def __init__(self, **kw):
        defaults = {
            "synthesis_backend": "",
            "synthesis_model": "",
            "synthesis_base_url": "",
            "synthesis_api_key_env": "",
            "embeddings_backend": "openai",
            "embeddings_model": "",
            "embeddings_base_url": "",
            "embeddings_api_key_env": "",
            "openai_api_key": "sk-test",
            "google_api_key": "",
            "openrouter_api_key": "",
            "litellm_api_key": "",
            "anthropic_api_key": "",
        }
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


def _openai_embed_response(n=2, dim=4):
    return {
        "model": "text-embedding-3-small",
        "data": [
            {"index": i, "embedding": [0.1 * (i + 1)] * dim}
            for i in range(n)
        ],
    }


@pytest.fixture(autouse=True)
def _reset_cache():
    embeddings_mod._reset_caches()
    yield
    embeddings_mod._reset_caches()


class TestEmbed:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        out = await embed([])
        assert out == []

    @pytest.mark.asyncio
    async def test_default_openai_backend(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        resp = httpx.Response(200, json=_openai_embed_response(2, 4), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            vectors = await embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 4

    @pytest.mark.asyncio
    async def test_uses_settings_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        s = _Settings(embeddings_backend="openai", embeddings_model="text-embedding-3-large")
        resp = httpx.Response(200, json=_openai_embed_response(1, 3), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            await embed(["hi"], settings=s)
        _args, kwargs = mpost.call_args
        assert kwargs["json"]["model"] == "text-embedding-3-large"

    @pytest.mark.asyncio
    async def test_gemini_backend_uses_gemini_endpoint(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "g-1")
        s = _Settings(embeddings_backend="gemini")
        payload = {
            "embeddings": [
                {"values": [0.1, 0.2]},
            ],
        }
        resp = httpx.Response(200, json=payload, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            vectors = await embed(["hi"], settings=s)
        assert vectors == [[0.1, 0.2]]
        args, _kwargs = mpost.call_args
        assert "batchEmbedContents" in args[0]

    @pytest.mark.asyncio
    async def test_per_call_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        resp = httpx.Response(200, json=_openai_embed_response(1, 2), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            await embed(
                ["hi"],
                backend="custom",
                base_url="http://localhost:4000/v1",
                model="my-embed",
            )
        args, kwargs = mpost.call_args
        assert args[0] == "http://localhost:4000/v1/embeddings"
        assert kwargs["json"]["model"] == "my-embed"


class TestEmbedDimensions:
    @pytest.mark.asyncio
    async def test_known_model_skips_probe(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        # No HTTP call expected for known models.
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mpost:
            dim = await embed_dimensions()
        assert dim == 1536
        mpost.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_model_probes_and_caches(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        s = _Settings(
            embeddings_backend="custom",
            embeddings_base_url="http://local:9/v1",
            embeddings_model="mystery-model",
        )
        resp = httpx.Response(200, json=_openai_embed_response(1, 7), request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp) as mpost:
            dim1 = await embed_dimensions(settings=s)
            dim2 = await embed_dimensions(settings=s)
        assert dim1 == 7
        assert dim2 == 7
        # Second call is served from the cache.
        assert mpost.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_vector_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        s = _Settings(
            embeddings_backend="custom",
            embeddings_base_url="http://local:9/v1",
            embeddings_model="broken",
        )
        payload = {"model": "broken", "data": [{"index": 0, "embedding": []}]}
        resp = httpx.Response(200, json=payload, request=_FAKE_REQUEST)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            with pytest.raises(RuntimeError, match="empty vector"):
                await embed_dimensions(settings=s)

    @pytest.mark.asyncio
    async def test_anthropic_refused(self):
        s = _Settings(embeddings_backend="anthropic")
        with pytest.raises(ValueError, match="does not support embeddings"):
            await embed(["hi"], settings=s)
