"""Tests for the new [synthesis] and [embeddings] TOML sections."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from parallect.config_mod import settings as settings_mod
from parallect.config_mod.settings import ParallectSettings


@pytest.fixture
def isolated_tmp(tmp_path, monkeypatch):
    """Run config loading with no real user/project config visible."""
    # Redirect user config dir + project path to tmp.
    monkeypatch.setattr(
        settings_mod, "_user_config_path", lambda: tmp_path / "user.toml"
    )
    monkeypatch.setattr(
        settings_mod, "_project_config_path", lambda: tmp_path / "project.toml"
    )
    # Clear env vars that would leak into tests.
    for name in (
        "PARALLECT_SYNTHESIS_BASE_URL",
        "PARALLECT_OPENAI_API_KEY",
        "PARALLECT_ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


class TestSynthesisSection:
    def test_reads_backend_model_baseurl_apikey_env(self, isolated_tmp):
        toml = dedent("""\
            [synthesis]
            backend = "openrouter"
            model = "google/gemini-2.5-flash-lite-preview"
            base_url = "https://openrouter.ai/api/v1"
            api_key_env = "OPENROUTER_API_KEY"
        """)
        (isolated_tmp / "user.toml").write_text(toml)
        s = ParallectSettings.load()
        assert s.synthesis_backend == "openrouter"
        assert s.synthesis_model == "google/gemini-2.5-flash-lite-preview"
        assert s.synthesis_base_url == "https://openrouter.ai/api/v1"
        assert s.synthesis_api_key_env == "OPENROUTER_API_KEY"

    def test_project_overrides_user(self, isolated_tmp):
        (isolated_tmp / "user.toml").write_text(
            '[synthesis]\nbackend = "openai"\n'
        )
        (isolated_tmp / "project.toml").write_text(
            '[synthesis]\nbackend = "litellm"\n'
        )
        s = ParallectSettings.load()
        assert s.synthesis_backend == "litellm"


class TestEmbeddingsSection:
    def test_reads_full_shape(self, isolated_tmp):
        toml = dedent("""\
            [embeddings]
            backend = "gemini"
            model = "text-embedding-004"
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            api_key_env = "GOOGLE_API_KEY"
        """)
        (isolated_tmp / "user.toml").write_text(toml)
        s = ParallectSettings.load()
        assert s.embeddings_backend == "gemini"
        assert s.embeddings_model == "text-embedding-004"
        assert s.embeddings_base_url == "https://generativelanguage.googleapis.com/v1beta"
        assert s.embeddings_api_key_env == "GOOGLE_API_KEY"

    def test_empty_when_absent(self, isolated_tmp):
        s = ParallectSettings.load()
        assert s.embeddings_backend == ""
        assert s.embeddings_model == ""


class TestBackwardsCompatibility:
    def test_old_providers_section_still_works(self, isolated_tmp):
        toml = dedent("""\
            [providers]
            openai_api_key = "sk-abc"
            openrouter_api_key = "or-xyz"

            [synthesis]
            backend = "openrouter"
        """)
        (isolated_tmp / "user.toml").write_text(toml)
        s = ParallectSettings.load()
        assert s.openai_api_key == "sk-abc"
        assert s.openrouter_api_key == "or-xyz"
        assert s.synthesis_backend == "openrouter"
