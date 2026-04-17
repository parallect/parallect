"""Tests for configuration settings."""

from __future__ import annotations

from parallect.config_mod.settings import ParallectSettings


class TestSettings:
    def test_defaults(self):
        settings = ParallectSettings()
        assert settings.providers == ["perplexity", "gemini", "openai"]
        assert settings.synthesize_with == "anthropic"
        assert settings.budget_cap_usd == 2.00
        assert settings.timeout_per_provider == 900.0
        assert settings.auto_sign is True

    def test_override_via_init(self):
        settings = ParallectSettings(
            providers=["ollama"],
            synthesize_with="ollama",
            budget_cap_usd=0.0,
        )
        assert settings.providers == ["ollama"]
        assert settings.synthesize_with == "ollama"
        assert settings.budget_cap_usd == 0.0

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("PARALLECT_PERPLEXITY_API_KEY", "test-key")
        settings = ParallectSettings()
        assert settings.perplexity_api_key == "test-key"

    def test_empty_api_keys_by_default(self):
        settings = ParallectSettings()
        assert settings.perplexity_api_key == ""
        assert settings.google_api_key == ""
        assert settings.openai_api_key == ""
        assert settings.anthropic_api_key == ""
        assert settings.xai_api_key == ""


class TestPluginManager:
    def test_register_and_run_hooks(self):
        from parallect.plugins import PluginManager

        manager = PluginManager()

        class MyHook:
            async def pre_research(self, query, providers):
                return query + " (modified)"

        manager.register_hook(MyHook())

        import asyncio

        result = asyncio.run(manager.run_pre_research("test", ["a"]))
        assert result == "test (modified)"

    def test_no_hooks_passthrough(self):
        from parallect.plugins import PluginManager

        manager = PluginManager()

        import asyncio

        result = asyncio.run(manager.run_pre_research("test", ["a"]))
        assert result == "test"
