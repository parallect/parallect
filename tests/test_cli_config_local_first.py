"""Tests for the local-first default behaviour in `parallect config`.

We exercise the decision tree inside `config_cmd` at the unit level. The
typer prompts are driven via `typer.testing.CliRunner`, the local probe is
monkeypatched, and the resulting config.toml is parsed back for assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

from parallect.backends import probe as probe_mod
from parallect.cli import config as cli_config_mod
from parallect.backends.probe import LocalProbeResult
from parallect.cli import app

# Empty-string answers for every API key prompt in `parallect config`.
# Order is fragile w.r.t. the prompt flow: perplexity, google, openai, xai,
# anthropic, openrouter, parallect. The optional local-default confirmation
# comes first when a local backend is detected.
_SEVEN_EMPTY_KEYS = "\n" * 7


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Redirect platformdirs to a tmp dir so we don't touch the real config."""
    monkeypatch.setattr(
        "platformdirs.user_config_dir",
        lambda name: str(tmp_path / name),
    )
    return tmp_path


class TestLocalFirst:
    def test_lmstudio_reachable_sets_backend(self, isolated_config, monkeypatch):
        fake = lambda timeout=0.4: LocalProbeResult(lmstudio_reachable=True, ollama_reachable=False)
        monkeypatch.setattr(probe_mod, "probe_local_backends", fake)
        monkeypatch.setattr(cli_config_mod, "probe_local_backends", fake)
        runner = CliRunner()
        # First prompt: "Default synthesis + embeddings to lmstudio?" -> accept default (yes)
        input_stream = "\n" + _SEVEN_EMPTY_KEYS
        result = runner.invoke(app, ["config"], input=input_stream)
        assert result.exit_code == 0, result.output

        toml_path = isolated_config / "parallect" / "config.toml"
        assert toml_path.exists()
        data = _load_toml(toml_path)
        assert data["synthesis"]["backend"] == "lmstudio"
        assert data["embeddings"]["backend"] == "lmstudio"

    def test_only_ollama_reachable_sets_backend(self, isolated_config, monkeypatch):
        fake = lambda timeout=0.4: LocalProbeResult(lmstudio_reachable=False, ollama_reachable=True)
        monkeypatch.setattr(probe_mod, "probe_local_backends", fake)
        monkeypatch.setattr(cli_config_mod, "probe_local_backends", fake)
        runner = CliRunner()
        input_stream = "\n" + _SEVEN_EMPTY_KEYS
        result = runner.invoke(app, ["config"], input=input_stream)
        assert result.exit_code == 0, result.output

        data = _load_toml(isolated_config / "parallect" / "config.toml")
        assert data["synthesis"]["backend"] == "ollama"
        assert data["embeddings"]["backend"] == "ollama"

    def test_user_declines_local_default(self, isolated_config, monkeypatch):
        fake = lambda timeout=0.4: LocalProbeResult(lmstudio_reachable=True, ollama_reachable=False)
        monkeypatch.setattr(probe_mod, "probe_local_backends", fake)
        monkeypatch.setattr(cli_config_mod, "probe_local_backends", fake)
        runner = CliRunner()
        # 'n' declines local; all keys empty, no cloud config written either.
        input_stream = "n\n" + _SEVEN_EMPTY_KEYS
        result = runner.invoke(app, ["config"], input=input_stream)
        assert result.exit_code == 0, result.output

        data = _load_toml(isolated_config / "parallect" / "config.toml")
        # No [synthesis] or [embeddings] section when user declined AND
        # entered no cloud keys -- or the section exists but with nothing.
        assert "synthesis" not in data or data["synthesis"] == {}
        assert "embeddings" not in data or data["embeddings"] == {}

    def test_neither_reachable_prompts_for_cloud(self, isolated_config, monkeypatch):
        fake = lambda timeout=0.4: LocalProbeResult(lmstudio_reachable=False, ollama_reachable=False)
        monkeypatch.setattr(probe_mod, "probe_local_backends", fake)
        monkeypatch.setattr(cli_config_mod, "probe_local_backends", fake)
        runner = CliRunner()
        # No local prompt, but we enter an Anthropic key.
        # Prompt order: perplexity, google, openai, xai, anthropic, openrouter, parallect
        input_stream = "\n\n\n\nsk-ant-test\n\n\n"
        result = runner.invoke(app, ["config"], input=input_stream)
        assert result.exit_code == 0, result.output

        data = _load_toml(isolated_config / "parallect" / "config.toml")
        assert data["providers"]["anthropic_api_key"] == "sk-ant-test"
        assert data["synthesis"]["backend"] == "anthropic"
        # Embeddings should NOT pick anthropic (no embeddings endpoint).
        assert data.get("embeddings", {}).get("backend", "") != "anthropic"

    def test_openai_key_also_picks_embeddings_default(self, isolated_config, monkeypatch):
        fake = lambda timeout=0.4: LocalProbeResult(lmstudio_reachable=False, ollama_reachable=False)
        monkeypatch.setattr(probe_mod, "probe_local_backends", fake)
        monkeypatch.setattr(cli_config_mod, "probe_local_backends", fake)
        runner = CliRunner()
        # perplexity, google, openai, xai, anthropic, openrouter, parallect
        input_stream = "\n\nsk-openai\n\n\n\n\n"
        result = runner.invoke(app, ["config"], input=input_stream)
        assert result.exit_code == 0, result.output

        data = _load_toml(isolated_config / "parallect" / "config.toml")
        assert data["providers"]["openai_api_key"] == "sk-openai"
        assert data["synthesis"]["backend"] == "openai"
        assert data["embeddings"]["backend"] == "openai"
