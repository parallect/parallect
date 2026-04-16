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

# Menu-driven config: after first-run probe, user enters "s" to save.
_SAVE_AND_EXIT = "s\n"


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
        monkeypatch.setattr("parallect.backends.probe.probe_local_backends", fake)
        runner = CliRunner()
        # Accept local default (yes) → save and exit
        input_stream = "\n" + _SAVE_AND_EXIT
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
        monkeypatch.setattr("parallect.backends.probe.probe_local_backends", fake)
        runner = CliRunner()
        input_stream = "\n" + _SAVE_AND_EXIT
        result = runner.invoke(app, ["config"], input=input_stream)
        assert result.exit_code == 0, result.output

        data = _load_toml(isolated_config / "parallect" / "config.toml")
        assert data["synthesis"]["backend"] == "ollama"
        assert data["embeddings"]["backend"] == "ollama"

    def test_user_declines_local_default(self, isolated_config, monkeypatch):
        fake = lambda timeout=0.4: LocalProbeResult(lmstudio_reachable=True, ollama_reachable=False)
        monkeypatch.setattr(probe_mod, "probe_local_backends", fake)
        monkeypatch.setattr("parallect.backends.probe.probe_local_backends", fake)
        runner = CliRunner()
        # 'n' declines local → save and exit
        input_stream = "n\n" + _SAVE_AND_EXIT
        result = runner.invoke(app, ["config"], input=input_stream)
        assert result.exit_code == 0, result.output

        data = _load_toml(isolated_config / "parallect" / "config.toml")
        assert "synthesis" not in data or not data.get("synthesis", {}).get("backend")
        assert "embeddings" not in data or not data.get("embeddings", {}).get("backend")

    def test_neither_reachable_save_empty(self, isolated_config, monkeypatch):
        """No local detected, user just saves and exits → empty config is valid."""
        fake = lambda timeout=0.4: LocalProbeResult(lmstudio_reachable=False, ollama_reachable=False)
        monkeypatch.setattr(probe_mod, "probe_local_backends", fake)
        monkeypatch.setattr("parallect.backends.probe.probe_local_backends", fake)
        runner = CliRunner()
        input_stream = _SAVE_AND_EXIT
        result = runner.invoke(app, ["config"], input=input_stream)
        assert result.exit_code == 0, result.output
        toml_path = isolated_config / "parallect" / "config.toml"
        assert toml_path.exists()

    def test_menu_configure_synthesis(self, isolated_config, monkeypatch):
        """User picks synthesis backend via menu option 1."""
        fake = lambda timeout=0.4: LocalProbeResult(lmstudio_reachable=False, ollama_reachable=False)
        monkeypatch.setattr(probe_mod, "probe_local_backends", fake)
        monkeypatch.setattr("parallect.backends.probe.probe_local_backends", fake)
        runner = CliRunner()
        # Menu: 1 (synthesis) → 3 (anthropic) → model prompt → save
        input_stream = "1\n3\nclaude-sonnet-4\nANTHROPIC_API_KEY\ns\n"
        result = runner.invoke(app, ["config"], input=input_stream)
        assert result.exit_code == 0, result.output
        data = _load_toml(isolated_config / "parallect" / "config.toml")
        assert data["synthesis"]["backend"] == "anthropic"
        assert data["synthesis"]["model"] == "claude-sonnet-4"
