"""Tests for parallect.backends: synthesis + embeddings backend resolution."""

from __future__ import annotations

import pytest

from parallect.backends import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URLS,
    DEFAULT_SYNTHESIS_MODELS,
    DEFAULT_EMBEDDING_MODELS,
    OPENAI_COMPAT_BACKENDS,
    is_valid_embeddings_backend,
    is_valid_synthesis_backend,
    resolve_embeddings_backend,
    resolve_synthesis_backend,
)


class _Settings:
    """Minimal duck-typed settings used for resolver tests."""

    def __init__(self, **kwargs):
        defaults = {
            "synthesis_backend": "",
            "synthesis_model": "",
            "synthesis_base_url": "",
            "synthesis_api_key_env": "",
            "embeddings_backend": "",
            "embeddings_model": "",
            "embeddings_base_url": "",
            "embeddings_api_key_env": "",
            "openai_api_key": "",
            "anthropic_api_key": "",
            "google_api_key": "",
            "openrouter_api_key": "",
            "litellm_api_key": "",
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Synthesis backend resolution
# ---------------------------------------------------------------------------


class TestSynthesisResolutionDefaults:
    def test_no_settings_defaults_to_anthropic(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
        # Mock auto-loaded settings so the test doesn't pick up the dev config
        _fake = type("S", (), {
            "synthesis_backend": "", "synthesis_model": "",
            "synthesis_base_url": "", "synthesis_api_key_env": "",
        })()
        monkeypatch.setattr(
            "parallect.config_mod.settings.ParallectSettings",
            type("F", (), {"load": classmethod(lambda cls: _fake)}),
        )
        spec = resolve_synthesis_backend()
        assert spec.kind == "anthropic"
        assert spec.base_url == DEFAULT_BASE_URLS["anthropic"]
        assert spec.api_key == "sk-ant-123"
        assert spec.model == DEFAULT_SYNTHESIS_MODELS["anthropic"]

    def test_settings_override_backend(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        s = _Settings(synthesis_backend="openrouter")
        spec = resolve_synthesis_backend(settings=s)
        assert spec.kind == "openrouter"
        assert spec.base_url == DEFAULT_BASE_URLS["openrouter"]

    def test_settings_override_model(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-123")
        s = _Settings(synthesis_backend="openai", synthesis_model="gpt-4o")
        spec = resolve_synthesis_backend(settings=s)
        assert spec.model == "gpt-4o"


class TestSynthesisPrecedence:
    def test_cli_base_url_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("PARALLECT_SYNTHESIS_BASE_URL", "http://env:1/v1")
        spec = resolve_synthesis_backend(cli_base_url="http://cli:2/v1")
        assert spec.base_url == "http://cli:2/v1"

    def test_env_wins_over_settings(self, monkeypatch):
        monkeypatch.setenv("PARALLECT_SYNTHESIS_BASE_URL", "http://envset:1/v1")
        s = _Settings(synthesis_backend="openai", synthesis_base_url="http://cfg:2/v1")
        spec = resolve_synthesis_backend(settings=s)
        assert spec.base_url == "http://envset:1/v1"

    def test_settings_wins_over_default(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        s = _Settings(
            synthesis_backend="openai",
            synthesis_base_url="http://cfg:2/v1",
        )
        spec = resolve_synthesis_backend(settings=s)
        assert spec.base_url == "http://cfg:2/v1"

    def test_cli_model_wins(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        s = _Settings(synthesis_backend="openai", synthesis_model="gpt-4o")
        spec = resolve_synthesis_backend(settings=s, cli_model="gpt-5")
        assert spec.model == "gpt-5"


class TestSynthesisCustomBackend:
    def test_custom_requires_base_url(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        s = _Settings(synthesis_backend="custom")
        with pytest.raises(ValueError, match="requires an explicit base_url"):
            resolve_synthesis_backend(settings=s)

    def test_custom_with_base_url_ok(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        s = _Settings(
            synthesis_backend="custom",
            synthesis_base_url="http://host:9999/v1",
            synthesis_model="my-model",
        )
        spec = resolve_synthesis_backend(settings=s)
        assert spec.kind == "custom"
        assert spec.base_url == "http://host:9999/v1"
        assert spec.model == "my-model"

    def test_env_base_url_upgrades_anthropic_to_custom(self, monkeypatch):
        monkeypatch.setenv("PARALLECT_SYNTHESIS_BASE_URL", "http://envurl:1/v1")
        s = _Settings(synthesis_backend="anthropic")
        spec = resolve_synthesis_backend(settings=s)
        assert spec.kind == "custom"
        assert spec.base_url == "http://envurl:1/v1"


class TestSynthesisAllBackends:
    @pytest.mark.parametrize(
        "backend",
        ["openai", "openrouter", "litellm", "gemini", "anthropic", "ollama", "lmstudio"],
    )
    def test_each_backend_resolves(self, backend, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        env_name = DEFAULT_API_KEY_ENV[backend]
        monkeypatch.setenv(env_name, "test-key")
        s = _Settings(synthesis_backend=backend)
        spec = resolve_synthesis_backend(settings=s)
        assert spec.kind == backend
        assert spec.base_url == DEFAULT_BASE_URLS[backend]
        assert spec.model == DEFAULT_SYNTHESIS_MODELS[backend]


class TestSynthesisApiKey:
    def test_custom_api_key_env(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        monkeypatch.setenv("MY_CUSTOM_KEY", "xyz")
        s = _Settings(
            synthesis_backend="openai",
            synthesis_api_key_env="MY_CUSTOM_KEY",
        )
        spec = resolve_synthesis_backend(settings=s)
        assert spec.api_key == "xyz"

    def test_local_backends_no_key_required(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        s = _Settings(synthesis_backend="ollama")
        spec = resolve_synthesis_backend(settings=s)
        assert spec.api_key == "not-needed"

    def test_fallback_to_settings_api_key(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_SYNTHESIS_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = _Settings(synthesis_backend="openai", openai_api_key="sk-from-settings")
        spec = resolve_synthesis_backend(settings=s)
        assert spec.api_key == "sk-from-settings"


# ---------------------------------------------------------------------------
# Embeddings backend resolution
# ---------------------------------------------------------------------------


class TestEmbeddingsResolutionDefaults:
    def test_default_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-123")
        _fake = type("S", (), {
            "embeddings_backend": "", "embeddings_model": "",
            "embeddings_base_url": "", "embeddings_api_key_env": "",
        })()
        monkeypatch.setattr(
            "parallect.config_mod.settings.ParallectSettings",
            type("F", (), {"load": classmethod(lambda cls: _fake)}),
        )
        spec = resolve_embeddings_backend()
        assert spec.kind == "openai"
        assert spec.base_url == DEFAULT_BASE_URLS["openai"]
        assert spec.model == DEFAULT_EMBEDDING_MODELS["openai"]

    def test_settings_override_backend(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
        s = _Settings(embeddings_backend="gemini")
        spec = resolve_embeddings_backend(settings=s)
        assert spec.kind == "gemini"

    def test_settings_override_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-123")
        s = _Settings(embeddings_backend="openai", embeddings_model="text-embedding-3-large")
        spec = resolve_embeddings_backend(settings=s)
        assert spec.model == "text-embedding-3-large"


class TestEmbeddingsAllBackends:
    @pytest.mark.parametrize(
        "backend",
        ["openai", "openrouter", "litellm", "gemini", "ollama", "lmstudio"],
    )
    def test_each_backend_resolves(self, backend, monkeypatch):
        monkeypatch.setenv(DEFAULT_API_KEY_ENV[backend], "k")
        s = _Settings(embeddings_backend=backend)
        spec = resolve_embeddings_backend(settings=s)
        assert spec.kind == backend
        assert spec.base_url == DEFAULT_BASE_URLS[backend]

    def test_anthropic_forbidden(self):
        s = _Settings(embeddings_backend="anthropic")
        with pytest.raises(ValueError, match="does not support embeddings"):
            resolve_embeddings_backend(settings=s)


class TestEmbeddingsCustomBackend:
    def test_custom_requires_base_url(self):
        s = _Settings(embeddings_backend="custom")
        with pytest.raises(ValueError, match="requires an explicit base_url"):
            resolve_embeddings_backend(settings=s)

    def test_custom_with_base_url_ok(self):
        s = _Settings(
            embeddings_backend="custom",
            embeddings_base_url="http://host:8000/v1",
            embeddings_model="my-embed",
        )
        spec = resolve_embeddings_backend(settings=s)
        assert spec.kind == "custom"
        assert spec.base_url == "http://host:8000/v1"

    def test_per_call_override_backend(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "k")
        s = _Settings(embeddings_backend="openai", openai_api_key="sk-1")
        spec = resolve_embeddings_backend(settings=s, override_backend="ollama")
        assert spec.kind == "ollama"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidators:
    def test_valid_synthesis_backends(self):
        for name in ("openai", "openrouter", "litellm", "gemini",
                     "anthropic", "ollama", "lmstudio", "custom"):
            assert is_valid_synthesis_backend(name)

    def test_invalid_synthesis_backend(self):
        assert not is_valid_synthesis_backend("made-up")

    def test_valid_embeddings_backends(self):
        for name in ("openai", "openrouter", "litellm", "gemini",
                     "ollama", "lmstudio", "custom"):
            assert is_valid_embeddings_backend(name)

    def test_embeddings_anthropic_invalid(self):
        assert not is_valid_embeddings_backend("anthropic")

    def test_openai_compat_registry_consistency(self):
        for name in OPENAI_COMPAT_BACKENDS:
            assert name in DEFAULT_API_KEY_ENV
            if name != "custom":
                assert name in DEFAULT_BASE_URLS
