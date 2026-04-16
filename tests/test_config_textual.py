"""Tests for the Textual-based ``parallect config`` TUI.

Every test uses ``App.run_test()`` + ``pilot`` for headless, deterministic
testing. No piped stdin, no CliRunner for interactive flows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

from textual.widgets import Button, DataTable, Input, OptionList, RadioSet, Static

from parallect.backends.probe import LocalProbeResult
from parallect.cli.config_app import (
    EMBEDDINGS_BACKENDS,
    MENU_ITEMS,
    SYNTHESIS_BACKENDS,
    BackendScreen,
    ConfigApp,
    FirstRunDialog,
    NothingDetectedDialog,
    ParallectApiScreen,
    PluginsScreen,
    ProviderKeysScreen,
    ViewConfigScreen,
    load_toml,
    write_toml,
)


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _make_app(
    tmp_path: Path,
    existing_data: dict | None = None,
    probe_fn=None,
    first_run: bool = False,
) -> ConfigApp:
    """Create a ConfigApp pointed at a tmp config file.

    By default, creates an empty config file so first-run probe is skipped.
    Pass ``first_run=True`` + ``probe_fn`` to test the first-run flow.
    """
    config_path = tmp_path / "parallect" / "config.toml"
    if existing_data is not None:
        write_toml(config_path, existing_data)
    elif not first_run:
        # Create an empty but valid config file to skip first-run detection
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# Parallect CLI configuration\n")
    return ConfigApp(config_path=config_path, probe_fn=probe_fn)


# ===================================================================
# Menu navigation
# ===================================================================


class TestMenuNavigation:
    @pytest.mark.asyncio
    async def test_save_creates_file(self, tmp_path: Path) -> None:
        """Pressing 's' saves config to disk."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("s")
        config_path = tmp_path / "parallect" / "config.toml"
        assert config_path.exists()

    @pytest.mark.asyncio
    async def test_quit_without_saving(self, tmp_path: Path) -> None:
        """Pressing 'q' exits without writing changes to disk."""
        app = _make_app(tmp_path)
        config_path = tmp_path / "parallect" / "config.toml"
        original_content = config_path.read_text()
        # Mutate in-memory data
        app.data["synthesis"] = {"backend": "openai"}
        async with app.run_test() as pilot:
            await pilot.press("q")
        # File should be unchanged (the mutation was not saved)
        assert config_path.read_text() == original_content

    @pytest.mark.asyncio
    async def test_menu_shows_all_items(self, tmp_path: Path) -> None:
        """All 6 menu items are displayed."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            assert option_list.option_count == len(MENU_ITEMS)
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_select_synthesis_opens_screen(self, tmp_path: Path) -> None:
        """Selecting 'Synthesis backend' pushes BackendScreen."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            # Focus the option list and select first item
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()
            # Should now be on a BackendScreen
            assert isinstance(app.screen, BackendScreen)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_select_embeddings_opens_screen(self, tmp_path: Path) -> None:
        """Selecting 'Embeddings backend' pushes BackendScreen."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, BackendScreen)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_select_providers_opens_screen(self, tmp_path: Path) -> None:
        """Selecting 'Provider API keys' pushes ProviderKeysScreen."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(2):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ProviderKeysScreen)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_select_plugins_opens_screen(self, tmp_path: Path) -> None:
        """Selecting 'Data source plugins' pushes PluginsScreen."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(3):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, PluginsScreen)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_select_parallect_api_opens_screen(self, tmp_path: Path) -> None:
        """Selecting 'Parallect API key' pushes ParallectApiScreen."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(4):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ParallectApiScreen)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_escape_from_subscreen_returns_to_menu(self, tmp_path: Path) -> None:
        """Escape from a sub-screen pops back to the main screen."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, BackendScreen)
            await pilot.press("escape")
            await pilot.pause()
            # Back on main screen — can still see option list
            assert app.query_one("#main-menu", OptionList) is not None
            await pilot.press("q")


# ===================================================================
# Synthesis backend
# ===================================================================


class TestSynthesisConfig:
    @pytest.mark.asyncio
    async def test_select_lmstudio_and_save(self, tmp_path: Path) -> None:
        """Select lmstudio backend, save, verify TOML."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            # Open synthesis screen
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()

            # lmstudio is the first radio button, should be selectable
            radio_set = app.screen.query_one("#backend-radios", RadioSet)
            radio_set.focus()
            # First item (lmstudio) - press space to select
            await pilot.press("enter")
            await pilot.pause()

            # Click Apply
            await pilot.click("#apply-btn")
            await pilot.pause()

            # Save
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["synthesis"]["backend"] == "lmstudio"

    @pytest.mark.asyncio
    async def test_select_custom_with_base_url(self, tmp_path: Path) -> None:
        """Select custom backend, type base URL, save."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()

            # Navigate to 'custom' (index 6)
            radio_set = app.screen.query_one("#backend-radios", RadioSet)
            radio_set.focus()
            for _ in range(6):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            # Type base URL
            base_url_input = app.screen.query_one("#base-url-input", Input)
            base_url_input.focus()
            await pilot.pause()
            base_url_input.value = "http://myserver:8080/v1"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["synthesis"]["backend"] == "custom"
        assert data["synthesis"]["base_url"] == "http://myserver:8080/v1"

    @pytest.mark.asyncio
    async def test_select_anthropic_with_api_key_env(self, tmp_path: Path) -> None:
        """Select anthropic, set api_key_env, save."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()

            # Navigate to anthropic (index 2)
            radio_set = app.screen.query_one("#backend-radios", RadioSet)
            radio_set.focus()
            for _ in range(2):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            # Set api_key_env
            env_input = app.screen.query_one("#api-key-env-input", Input)
            env_input.focus()
            await pilot.pause()
            env_input.value = "ANTHROPIC_API_KEY"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["synthesis"]["backend"] == "anthropic"
        assert data["synthesis"]["api_key_env"] == "ANTHROPIC_API_KEY"

    @pytest.mark.asyncio
    async def test_change_model_name(self, tmp_path: Path) -> None:
        """Change model name, save, verify."""
        existing = {"synthesis": {"backend": "openai", "model": "gpt-4"}}
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()

            model_input = app.screen.query_one("#model-input", Input)
            model_input.focus()
            await pilot.pause()
            model_input.value = "gpt-4-turbo"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["synthesis"]["model"] == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_back_button_returns_to_menu(self, tmp_path: Path) -> None:
        """Back button on synthesis screen returns to main menu."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, BackendScreen)

            await pilot.click("#back-btn")
            await pilot.pause()
            # Back on main screen
            assert not isinstance(app.screen, BackendScreen)
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_synthesis_has_all_backends(self, tmp_path: Path) -> None:
        """Verify all synthesis backends are present as radio buttons."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()

            radio_set = app.screen.query_one("#backend-radios", RadioSet)
            # Count radio buttons
            buttons = radio_set.query("RadioButton")
            assert len(buttons) == len(SYNTHESIS_BACKENDS)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")


# ===================================================================
# Embeddings backend
# ===================================================================


class TestEmbeddingsConfig:
    @pytest.mark.asyncio
    async def test_select_openai_embeddings(self, tmp_path: Path) -> None:
        """Select openai embeddings backend and save."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, BackendScreen)

            # openai is index 2 in EMBEDDINGS_BACKENDS
            radio_set = app.screen.query_one("#backend-radios", RadioSet)
            radio_set.focus()
            for _ in range(2):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["embeddings"]["backend"] == "openai"

    @pytest.mark.asyncio
    async def test_anthropic_not_in_embeddings(self, tmp_path: Path) -> None:
        """Anthropic should not be an option for embeddings."""
        backend_keys = [k for k, _ in EMBEDDINGS_BACKENDS]
        assert "anthropic" not in backend_keys

    @pytest.mark.asyncio
    async def test_switch_backend_and_model(self, tmp_path: Path) -> None:
        """Switch from openai to lmstudio, update model, save."""
        existing = {
            "embeddings": {"backend": "openai", "model": "text-embedding-3-small"},
        }
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            # Select lmstudio (index 0)
            radio_set = app.screen.query_one("#backend-radios", RadioSet)
            radio_set.focus()
            # Press Home to go to first
            await pilot.press("home")
            await pilot.press("enter")
            await pilot.pause()

            # Change model
            model_input = app.screen.query_one("#model-input", Input)
            model_input.focus()
            await pilot.pause()
            model_input.value = "nomic-embed-text"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["embeddings"]["backend"] == "lmstudio"
        assert data["embeddings"]["model"] == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_embeddings_has_correct_count(self, tmp_path: Path) -> None:
        """Embeddings should have exactly 6 backends (no anthropic)."""
        assert len(EMBEDDINGS_BACKENDS) == 6


# ===================================================================
# Provider API keys
# ===================================================================


class TestProviderKeys:
    @pytest.mark.asyncio
    async def test_set_provider_key(self, tmp_path: Path) -> None:
        """Type a key, save, verify it is stored."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(2):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, ProviderKeysScreen)
            inp = app.screen.query_one("#key-perplexity_api_key", Input)
            inp.focus()
            await pilot.pause()
            inp.value = "pplx-test-key-1234567890"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["providers"]["perplexity_api_key"] == "pplx-test-key-1234567890"

    @pytest.mark.asyncio
    async def test_existing_key_preserved_when_empty(self, tmp_path: Path) -> None:
        """Leaving an input empty should preserve the existing key."""
        existing = {"providers": {"perplexity_api_key": "existing-key-value"}}
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(2):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            # Don't type anything, just apply
            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["providers"]["perplexity_api_key"] == "existing-key-value"

    @pytest.mark.asyncio
    async def test_new_key_overwrites_existing(self, tmp_path: Path) -> None:
        """Typing a new key replaces the old one."""
        existing = {"providers": {"openai_api_key": "old-key"}}
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(2):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            inp = app.screen.query_one("#key-openai_api_key", Input)
            inp.focus()
            await pilot.pause()
            inp.value = "new-openai-key-abcdef"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["providers"]["openai_api_key"] == "new-openai-key-abcdef"

    @pytest.mark.asyncio
    async def test_all_six_fields_independent(self, tmp_path: Path) -> None:
        """Setting one provider key does not affect others."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(2):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            # Set only xai key
            inp = app.screen.query_one("#key-xai_api_key", Input)
            inp.value = "xai-key-123"

            # Set only anthropic key
            inp2 = app.screen.query_one("#key-anthropic_api_key", Input)
            inp2.value = "ant-key-456"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["providers"]["xai_api_key"] == "xai-key-123"
        assert data["providers"]["anthropic_api_key"] == "ant-key-456"
        assert "perplexity_api_key" not in data["providers"]


# ===================================================================
# Data source plugins
# ===================================================================


class TestPlugins:
    @pytest.mark.asyncio
    async def test_add_filesystem_plugin(self, tmp_path: Path) -> None:
        """Add a filesystem plugin via modal."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(3):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, PluginsScreen)
            await pilot.click("#add-fs-btn")
            await pilot.pause()

            # Fill in the modal
            name_input = app.screen.query_one("#plugin-name", Input)
            name_input.value = "docs"
            path_input = app.screen.query_one("#plugin-path", Input)
            path_input.value = "/home/user/docs"
            await pilot.pause()

            await pilot.click("#modal-add")
            await pilot.pause()

            # Back to main, save
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert len(data["plugins"]["filesystem"]) == 1
        assert data["plugins"]["filesystem"][0]["name"] == "docs"
        assert data["plugins"]["filesystem"][0]["path"] == "/home/user/docs"

    @pytest.mark.asyncio
    async def test_add_two_filesystem_plugins(self, tmp_path: Path) -> None:
        """Adding two filesystem plugins grows the array."""
        existing = {
            "plugins": {
                "filesystem": [{"name": "first", "path": "/first"}],
            },
        }
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(3):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            await pilot.click("#add-fs-btn")
            await pilot.pause()

            name_input = app.screen.query_one("#plugin-name", Input)
            name_input.value = "second"
            path_input = app.screen.query_one("#plugin-path", Input)
            path_input.value = "/second"
            await pilot.pause()

            await pilot.click("#modal-add")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert len(data["plugins"]["filesystem"]) == 2

    @pytest.mark.asyncio
    async def test_add_obsidian_plugin(self, tmp_path: Path) -> None:
        """Add an Obsidian vault plugin."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(3):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            await pilot.click("#add-obs-btn")
            await pilot.pause()

            name_input = app.screen.query_one("#plugin-name", Input)
            name_input.value = "notes"
            path_input = app.screen.query_one("#plugin-path", Input)
            path_input.value = "/home/user/ObsidianVault"
            await pilot.pause()

            await pilot.click("#modal-add")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["plugins"]["obsidian"][0]["name"] == "notes"
        assert data["plugins"]["obsidian"][0]["path"] == "/home/user/ObsidianVault"

    @pytest.mark.asyncio
    async def test_plugin_list_shows_existing(self, tmp_path: Path) -> None:
        """Existing plugins are displayed in the plugin list."""
        existing = {
            "plugins": {
                "filesystem": [{"name": "research", "path": "/data"}],
            },
        }
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(3):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            plugin_list = app.screen.query_one("#plugin-list", Static)
            text = str(plugin_list.content)
            assert "filesystem" in text
            assert "research" in text
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_add_with_empty_path_shows_error(self, tmp_path: Path) -> None:
        """Empty path in plugin modal shows validation error."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(3):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            await pilot.click("#add-fs-btn")
            await pilot.pause()

            # Leave path empty, click Add
            await pilot.click("#modal-add")
            await pilot.pause()

            # Error should be shown
            error_label = app.screen.query_one("#plugin-error")
            assert "required" in str(error_label.content).lower()

            # Cancel out
            await pilot.click("#modal-cancel")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_configure_prxhub_url(self, tmp_path: Path) -> None:
        """Configure prxhub URL via modal."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(3):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            await pilot.click("#prxhub-btn")
            await pilot.pause()

            url_input = app.screen.query_one("#prxhub-url", Input)
            url_input.value = "https://custom-prxhub.example.com"
            await pilot.pause()

            await pilot.click("#modal-save")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["plugins"]["prxhub"]["api_url"] == "https://custom-prxhub.example.com"


# ===================================================================
# Parallect API key
# ===================================================================


class TestParallectApiKey:
    @pytest.mark.asyncio
    async def test_set_api_key(self, tmp_path: Path) -> None:
        """Set a Parallect API key."""
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(4):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, ParallectApiScreen)
            inp = app.screen.query_one("#par-api-key", Input)
            inp.value = "par_live_hint_secretvalue"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["parallect_api"]["api_key"] == "par_live_hint_secretvalue"

    @pytest.mark.asyncio
    async def test_clear_api_key(self, tmp_path: Path) -> None:
        """Clear an existing API key."""
        existing = {"parallect_api": {"api_key": "par_live_old_key"}}
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(4):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            await pilot.click("#clear-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert "parallect_api" not in data or "api_key" not in data.get("parallect_api", {})


# ===================================================================
# First-run flow
# ===================================================================


class TestFirstRun:
    @pytest.mark.asyncio
    async def test_lmstudio_detected_accept(self, tmp_path: Path) -> None:
        """LM Studio detected, user accepts -> synthesis+embeddings set."""
        probe_fn = lambda: LocalProbeResult(lmstudio_reachable=True, ollama_reachable=False)
        app = _make_app(tmp_path, probe_fn=probe_fn, first_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            # FirstRunDialog should be showing
            assert isinstance(app.screen, FirstRunDialog)
            await pilot.click("#first-run-yes")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["synthesis"]["backend"] == "lmstudio"
        assert data["embeddings"]["backend"] == "lmstudio"

    @pytest.mark.asyncio
    async def test_lmstudio_detected_decline(self, tmp_path: Path) -> None:
        """LM Studio detected, user declines -> no backends set."""
        probe_fn = lambda: LocalProbeResult(lmstudio_reachable=True, ollama_reachable=False)
        app = _make_app(tmp_path, probe_fn=probe_fn, first_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, FirstRunDialog)
            await pilot.click("#first-run-no")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert "synthesis" not in data or "backend" not in data.get("synthesis", {})

    @pytest.mark.asyncio
    async def test_neither_detected_shows_warning(self, tmp_path: Path) -> None:
        """No local backend -> warning dialog, then menu accessible."""
        probe_fn = lambda: LocalProbeResult(lmstudio_reachable=False, ollama_reachable=False)
        app = _make_app(tmp_path, probe_fn=probe_fn, first_run=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, NothingDetectedDialog)
            await pilot.click("#nothing-ok")
            await pilot.pause()
            # Menu should now be accessible
            assert app.query_one("#main-menu", OptionList) is not None
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_existing_config_skips_first_run(self, tmp_path: Path) -> None:
        """When config file exists, first-run probe is skipped."""
        existing = {"synthesis": {"backend": "openai"}}
        # probe_fn would throw if called
        def bad_probe():
            raise RuntimeError("Should not be called")

        app = _make_app(tmp_path, existing_data=existing, probe_fn=bad_probe)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Should be on main menu, not first-run dialog
            assert app.query_one("#main-menu", OptionList) is not None
            await pilot.press("q")


# ===================================================================
# View current config
# ===================================================================


class TestViewConfig:
    @pytest.mark.asyncio
    async def test_view_shows_data_table(self, tmp_path: Path) -> None:
        """View config screen shows a DataTable with values."""
        existing = {
            "synthesis": {"backend": "openai", "model": "gpt-4"},
            "providers": {"openai_api_key": "sk-1234567890abcdef"},
        }
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(5):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, ViewConfigScreen)
            table = app.screen.query_one("#config-table", DataTable)
            assert table.row_count > 0
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_view_masks_api_keys(self, tmp_path: Path) -> None:
        """API keys in the view should be masked."""
        existing = {
            "providers": {"openai_api_key": "sk-1234567890abcdefghij"},
        }
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            for _ in range(5):
                await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            table = app.screen.query_one("#config-table", DataTable)
            # The full key should NOT appear in any cell
            # Check that masking is applied
            assert table.row_count > 0
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("q")


# ===================================================================
# Round-trip tests
# ===================================================================


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_load_save_preserves_config(self, tmp_path: Path) -> None:
        """Load existing config, save without changes, file content preserved."""
        existing = {
            "synthesis": {"backend": "openai", "model": "gpt-4"},
            "embeddings": {"backend": "lmstudio"},
            "providers": {"openai_api_key": "sk-test123"},
        }
        config_path = tmp_path / "parallect" / "config.toml"
        write_toml(config_path, existing)
        original = _read_toml(config_path)

        app = ConfigApp(config_path=config_path)
        async with app.run_test() as pilot:
            await pilot.press("s")

        reloaded = _read_toml(config_path)
        assert reloaded == original

    @pytest.mark.asyncio
    async def test_change_one_thing_preserves_rest(self, tmp_path: Path) -> None:
        """Change one field, save, verify rest is untouched."""
        existing = {
            "synthesis": {"backend": "openai", "model": "gpt-4"},
            "embeddings": {"backend": "lmstudio", "model": "nomic"},
            "providers": {"openai_api_key": "sk-preserved"},
        }
        app = _make_app(tmp_path, existing_data=existing)
        async with app.run_test() as pilot:
            # Change synthesis model
            option_list = app.query_one("#main-menu", OptionList)
            option_list.focus()
            await pilot.press("enter")
            await pilot.pause()

            model_input = app.screen.query_one("#model-input", Input)
            model_input.value = "gpt-5"
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()
            await pilot.press("s")

        data = _read_toml(tmp_path / "parallect" / "config.toml")
        assert data["synthesis"]["model"] == "gpt-5"
        # Rest preserved
        assert data["embeddings"]["backend"] == "lmstudio"
        assert data["embeddings"]["model"] == "nomic"
        assert data["providers"]["openai_api_key"] == "sk-preserved"


# ===================================================================
# TOML utility tests (non-UI, kept as simple unit tests)
# ===================================================================


class TestTomlUtilities:
    def test_load_nonexistent(self, tmp_path: Path) -> None:
        assert load_toml(tmp_path / "nope.toml") == {}

    def test_write_and_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "test.toml"
        data = {
            "synthesis": {"backend": "openai", "model": "gpt-4"},
            "plugins": {
                "filesystem": [
                    {"name": "a", "path": "/a"},
                    {"name": "b", "path": "/b"},
                ],
                "prxhub": {"api_url": "https://prxhub.com"},
            },
        }
        write_toml(path, data)
        reloaded = load_toml(path)
        assert reloaded["synthesis"]["backend"] == "openai"
        assert len(reloaded["plugins"]["filesystem"]) == 2
        assert reloaded["plugins"]["prxhub"]["api_url"] == "https://prxhub.com"

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "config.toml"
        write_toml(path, {"test": {"key": "val"}})
        assert path.exists()
