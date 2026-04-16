"""Tests for the local-first default behaviour in ``parallect config``.

These tests exercise the first-run probe and menu config flows using
Textual's ``App.run_test()`` + pilot for headless deterministic testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

from parallect.backends.probe import LocalProbeResult
from parallect.cli.config_app import (
    BackendScreen,
    ConfigApp,
    FirstRunDialog,
    load_toml,
    write_toml,
)


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _make_app(
    tmp_path: Path,
    probe_fn=None,
    existing_data: dict | None = None,
    first_run: bool = False,
) -> ConfigApp:
    config_path = tmp_path / "parallect" / "config.toml"
    if existing_data is not None:
        write_toml(config_path, existing_data)
    elif not first_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# Parallect CLI configuration\n")
    return ConfigApp(config_path=config_path, probe_fn=probe_fn)


class TestLocalFirst:
    @pytest.mark.asyncio
    async def test_lmstudio_reachable_sets_backend(self, tmp_path: Path) -> None:
        probe_fn = lambda: LocalProbeResult(lmstudio_reachable=True, ollama_reachable=False)
        app = _make_app(tmp_path, probe_fn=probe_fn, first_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, FirstRunDialog)
            await pilot.click("#first-run-yes")
            await pilot.pause()
            await pilot.press("s")

        toml_path = tmp_path / "parallect" / "config.toml"
        assert toml_path.exists()
        data = _load_toml(toml_path)
        assert data["synthesis"]["backend"] == "lmstudio"
        assert data["embeddings"]["backend"] == "lmstudio"

    @pytest.mark.asyncio
    async def test_only_ollama_reachable_sets_backend(self, tmp_path: Path) -> None:
        probe_fn = lambda: LocalProbeResult(lmstudio_reachable=False, ollama_reachable=True)
        app = _make_app(tmp_path, probe_fn=probe_fn, first_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, FirstRunDialog)
            await pilot.click("#first-run-yes")
            await pilot.pause()
            await pilot.press("s")

        data = _load_toml(tmp_path / "parallect" / "config.toml")
        assert data["synthesis"]["backend"] == "ollama"
        assert data["embeddings"]["backend"] == "ollama"

    @pytest.mark.asyncio
    async def test_user_declines_local_default(self, tmp_path: Path) -> None:
        probe_fn = lambda: LocalProbeResult(lmstudio_reachable=True, ollama_reachable=False)
        app = _make_app(tmp_path, probe_fn=probe_fn, first_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, FirstRunDialog)
            await pilot.click("#first-run-no")
            await pilot.pause()
            await pilot.press("s")

        data = _load_toml(tmp_path / "parallect" / "config.toml")
        assert "synthesis" not in data or not data.get("synthesis", {}).get("backend")
        assert "embeddings" not in data or not data.get("embeddings", {}).get("backend")

    @pytest.mark.asyncio
    async def test_neither_reachable_save_empty(self, tmp_path: Path) -> None:
        """No local detected, user just saves and exits -> empty config is valid."""
        probe_fn = lambda: LocalProbeResult(lmstudio_reachable=False, ollama_reachable=False)
        app = _make_app(tmp_path, probe_fn=probe_fn, first_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Dismiss the "nothing detected" dialog
            await pilot.click("#nothing-ok")
            await pilot.pause()
            await pilot.press("s")

        toml_path = tmp_path / "parallect" / "config.toml"
        assert toml_path.exists()

    @pytest.mark.asyncio
    async def test_menu_configure_synthesis(self, tmp_path: Path) -> None:
        """User picks synthesis backend via menu."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            from textual.widgets import Input, OptionList, RadioSet

            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, BackendScreen)

            # Select anthropic (index 2)
            radio_set = app.screen.query_one("#backend-radios", RadioSet)
            radio_set.focus()
            for _ in range(2):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            # Set model
            model_input = app.screen.query_one("#model-input", Input)
            model_input.value = "claude-sonnet-4"

            # Set API key env
            env_input = app.screen.query_one("#api-key-env-input", Input)
            env_input.value = "ANTHROPIC_API_KEY"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _load_toml(tmp_path / "parallect" / "config.toml")
        assert data["synthesis"]["backend"] == "anthropic"
        assert data["synthesis"]["model"] == "claude-sonnet-4"
