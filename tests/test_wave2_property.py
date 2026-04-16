"""Wave-2: Property-based tests using hypothesis.

Tests that backend resolution, parse_sources, document metadata, embedding
dimension caching, and prior research cache behave correctly across a wide
range of inputs.
"""

from __future__ import annotations

import string
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from hypothesis import given, settings as hsettings, HealthCheck
from hypothesis import strategies as st

from parallect.backends import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URLS,
    resolve_embeddings_backend,
    resolve_synthesis_backend,
)
from parallect.plugins.data_sources import (
    Document,
    PluginError,
    parse_sources,
)


_SYNTHESIS_BACKENDS = ["openai", "openrouter", "litellm", "gemini",
                       "anthropic", "ollama", "lmstudio"]
_EMBEDDINGS_BACKENDS = ["openai", "openrouter", "litellm", "gemini",
                        "ollama", "lmstudio"]

_valid_name_chars = string.ascii_lowercase + string.digits + "-_"
_name_st = st.text(
    alphabet=_valid_name_chars, min_size=1, max_size=30,
).filter(lambda s: s[0] not in "-_" and ":" not in s)

_SUPPRESS_FIXTURE = [HealthCheck.function_scoped_fixture]


class TestBackendResolutionProperty:
    @given(
        backend=st.sampled_from(_SYNTHESIS_BACKENDS),
        cli_model=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
    )
    @hsettings(max_examples=50, deadline=2000, suppress_health_check=_SUPPRESS_FIXTURE)
    def test_synthesis_resolution_never_crashes(self, backend, cli_model, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        env_name = DEFAULT_API_KEY_ENV.get(backend, "")
        if env_name:
            monkeypatch.setenv(env_name, "test-key")

        class S:
            synthesis_backend = backend
            synthesis_model = ""
            synthesis_base_url = ""
            synthesis_api_key_env = ""
            def __getattr__(self, name):
                return ""

        spec = resolve_synthesis_backend(settings=S(), cli_model=cli_model)
        assert spec.kind == backend
        if cli_model:
            assert spec.model == cli_model

    @given(backend=st.sampled_from(_EMBEDDINGS_BACKENDS))
    @hsettings(max_examples=30, deadline=2000, suppress_health_check=_SUPPRESS_FIXTURE)
    def test_embeddings_resolution_never_crashes(self, backend, monkeypatch):
        env_name = DEFAULT_API_KEY_ENV.get(backend, "")
        if env_name:
            monkeypatch.setenv(env_name, "test-key")

        class S:
            embeddings_backend = backend
            embeddings_model = ""
            embeddings_base_url = ""
            embeddings_api_key_env = ""
            def __getattr__(self, name):
                return ""

        spec = resolve_embeddings_backend(settings=S())
        assert spec.kind == backend
        assert spec.base_url == DEFAULT_BASE_URLS.get(backend, "")

    @given(has_cli=st.booleans(), has_env=st.booleans(), has_config=st.booleans())
    @hsettings(max_examples=20, deadline=2000, suppress_health_check=_SUPPRESS_FIXTURE)
    def test_synthesis_precedence_ordering(self, has_cli, has_env, has_config, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "k")

        cli_url = "http://cli:1/v1" if has_cli else None
        if has_env:
            monkeypatch.setenv("PARALLECT_SYNTHESIS_BASE_URL", "http://env:2/v1")
        config_url = "http://cfg:3/v1" if has_config else ""

        class S:
            synthesis_backend = "openai"
            synthesis_model = ""
            synthesis_base_url = config_url
            synthesis_api_key_env = ""
            openai_api_key = "k"
            def __getattr__(self, name):
                return ""

        spec = resolve_synthesis_backend(settings=S(), cli_base_url=cli_url)

        if has_cli:
            assert spec.base_url == "http://cli:1/v1"
        elif has_env:
            assert spec.base_url == "http://env:2/v1"
        elif has_config:
            assert spec.base_url == "http://cfg:3/v1"
        else:
            assert spec.base_url == DEFAULT_BASE_URLS["openai"]


class TestParseSourcesProperty:
    @given(names=st.lists(_name_st, min_size=1, max_size=5))
    @hsettings(max_examples=50, deadline=2000)
    def test_roundtrip_names(self, names):
        raw = ",".join(names)
        specs = parse_sources(raw)
        assert len(specs) == len(names)
        for spec, name in zip(specs, names):
            assert spec.name == name

    @given(name=_name_st, instance=_name_st)
    @hsettings(max_examples=50, deadline=2000)
    def test_roundtrip_with_instance(self, name, instance):
        raw = f"{name}:{instance}"
        specs = parse_sources(raw)
        assert len(specs) == 1
        assert specs[0].name == name
        assert specs[0].instance == instance

    def test_empty_returns_empty(self):
        assert parse_sources(None) == []
        assert parse_sources("") == []

    def test_colon_only_name_raises(self):
        with pytest.raises(PluginError, match="invalid"):
            parse_sources(":instance")


class TestDocumentMetadataProperty:
    @given(
        metadata=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.text(max_size=50), st.integers(),
                st.floats(allow_nan=False), st.booleans(), st.none(),
            ),
            max_size=10,
        ),
    )
    @hsettings(max_examples=50, deadline=2000)
    def test_metadata_survives_roundtrip(self, metadata):
        doc = Document(id="test-id", content="test content", metadata=metadata)
        assert doc.metadata == metadata
        for key, value in metadata.items():
            assert doc.metadata[key] == value


class TestEmbedDimensionsCacheProperty:
    @given(dim=st.integers(min_value=1, max_value=4096))
    @hsettings(max_examples=20, deadline=10000, suppress_health_check=_SUPPRESS_FIXTURE)
    async def test_dimension_cache_consistent(self, dim, monkeypatch):
        import asyncio
        import parallect.embeddings as emb_mod

        emb_mod._reset_caches()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")

        fake_request = httpx.Request("POST", "http://test")
        resp_data = {
            "model": "test-model",
            "data": [{"index": 0, "embedding": [0.1] * dim}],
        }
        resp = httpx.Response(200, json=resp_data, request=fake_request)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
            from parallect.embeddings import embed_dimensions

            class S:
                embeddings_backend = "custom"
                embeddings_model = f"dim-test-model-{dim}"
                embeddings_base_url = "http://local:9/v1"
                embeddings_api_key_env = ""
                openai_api_key = "sk-1"
                def __getattr__(self, name):
                    return ""

            results = await asyncio.gather(
                embed_dimensions(settings=S()),
                embed_dimensions(settings=S()),
                embed_dimensions(settings=S()),
            )
            assert all(r == dim for r in results)

        emb_mod._reset_caches()


class TestPriorResearchCacheProperty:
    @given(
        query=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
        synthesis=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @hsettings(max_examples=20, deadline=10000, suppress_health_check=_SUPPRESS_FIXTURE)
    async def test_append_always_retrievable(self, query, synthesis, tmp_path, monkeypatch):
        from parallect.plugins.data_sources.prior_research import PriorResearchCache
        import parallect.embeddings as emb_mod

        async def fake_embed(texts, **_):
            return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

        monkeypatch.setattr(emb_mod, "embed", fake_embed)

        db_path = tmp_path / f"prop-cache-{hash(query) % 10000}.db"
        cache = PriorResearchCache()
        await cache.configure({"db_path": str(db_path)})

        await cache.append(
            query=query, synthesis_md=synthesis,
            sources_json="[]", bundle_id="prop-bnd",
        )

        doc = await cache.fetch("prior:prop-bnd")
        assert doc is not None
        assert doc.content == synthesis
        assert doc.metadata["past_query"] == query
