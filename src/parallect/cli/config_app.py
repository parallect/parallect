"""Textual TUI for `parallect config`.

Replaces the old Rich/Typer interactive prompt flow with a proper Textual app
that can be tested headlessly via ``App.run_test()`` + pilot.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import platformdirs
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    RadioButton,
    RadioSet,
    Static,
)

# ---------------------------------------------------------------------------
# TOML helpers (reused from config.py)
# ---------------------------------------------------------------------------


def _default_config_path() -> Path:
    return Path(platformdirs.user_config_dir("parallect")) / "config.toml"


def load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def _toml_val(v: object) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        inner = ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in v)
        return f"[{inner}]"
    return f'"{v}"'


def write_toml(path: Path, data: dict) -> None:
    """Write config dict as TOML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Parallect CLI configuration\n"]
    for section, values in data.items():
        if section == "plugins":
            for plugin_name, instances in values.items():
                if isinstance(instances, list):
                    for inst in instances:
                        lines.append(f"[[plugins.{plugin_name}]]")
                        for k, v in inst.items():
                            lines.append(f"{k} = {_toml_val(v)}")
                        lines.append("")
                elif isinstance(instances, dict):
                    lines.append(f"[plugins.{plugin_name}]")
                    for k, v in instances.items():
                        lines.append(f"{k} = {_toml_val(v)}")
                    lines.append("")
        elif isinstance(values, dict):
            lines.append(f"[{section}]")
            for k, v in values.items():
                lines.append(f"{k} = {_toml_val(v)}")
            lines.append("")
        else:
            lines.append(f"{section} = {_toml_val(values)}")
    path.write_text("\n".join(lines))


def _mask(key: str) -> str:
    if not key or len(key) < 8:
        return key or "(not set)"
    return key[:8] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# Synthesis / Embeddings backend choices
# ---------------------------------------------------------------------------

SYNTHESIS_BACKENDS = [
    ("lmstudio", "LM Studio (local, private, free)"),
    ("ollama", "Ollama (local, private, free)"),
    ("anthropic", "Anthropic / Claude (cloud)"),
    ("openai", "OpenAI / GPT (cloud)"),
    ("gemini", "Google Gemini (cloud, cheap)"),
    ("openrouter", "OpenRouter (cloud, any model)"),
    ("custom", "Custom OpenAI-compatible URL"),
]

EMBEDDINGS_BACKENDS = [
    ("lmstudio", "LM Studio (local)"),
    ("ollama", "Ollama (local)"),
    ("openai", "OpenAI text-embedding-3-small"),
    ("gemini", "Google text-embedding-004"),
    ("openrouter", "OpenRouter"),
    ("custom", "Custom OpenAI-compatible URL"),
]

PROVIDER_KEYS = [
    ("perplexity_api_key", "Perplexity"),
    ("google_api_key", "Google AI (Gemini)"),
    ("openai_api_key", "OpenAI"),
    ("xai_api_key", "xAI (Grok)"),
    ("anthropic_api_key", "Anthropic"),
    ("openrouter_api_key", "OpenRouter"),
]

CLOUD_BACKENDS = {"anthropic", "openai", "gemini", "openrouter"}


# ---------------------------------------------------------------------------
# Modal dialogs
# ---------------------------------------------------------------------------


class EditPluginModal(ModalScreen[tuple | None]):
    """Modal for editing or deleting an existing plugin entry."""

    def __init__(self, ptype: str, pname: str | None, path: str) -> None:
        super().__init__()
        self.ptype = ptype
        self.pname = pname
        self.current_path = path

    def compose(self) -> ComposeResult:
        display = f"{self.ptype}:{self.pname}" if self.pname else self.ptype
        with Vertical(id="modal-container"):
            yield Label(f"Edit: {display}", classes="screen-title")
            yield Label("Path", classes="form-label")
            yield Input(value=self.current_path, id="edit-path")
            with Horizontal(classes="button-bar"):
                yield Button("Save", variant="primary", id="edit-save")
                yield Button("Delete", variant="error", id="edit-delete")
                yield Button("Cancel", id="edit-cancel")

    @on(Button.Pressed, "#edit-save")
    def _on_save(self) -> None:
        new_path = self.query_one("#edit-path", Input).value.strip()
        if new_path:
            self.dismiss(("edit", self.ptype, self.pname, new_path))

    @on(Button.Pressed, "#edit-delete")
    def _on_delete(self) -> None:
        self.dismiss(("delete", self.ptype, self.pname, ""))

    @on(Button.Pressed, "#edit-cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)


class AddPluginModal(ModalScreen[dict | None]):
    """Modal for adding a filesystem or obsidian plugin instance."""

    def __init__(self, plugin_type: str) -> None:
        super().__init__()
        self.plugin_type = plugin_type

    def compose(self) -> ComposeResult:
        path_label = "Vault path" if self.plugin_type == "obsidian" else "Directory path"
        default_name = "notes" if self.plugin_type == "obsidian" else "research"
        with Vertical(id="modal-container"):
            yield Label(f"Add {self.plugin_type} plugin", classes="screen-title")
            yield Label("Instance name", classes="form-label")
            yield Input(value=default_name, id="plugin-name")
            yield Label(path_label, classes="form-label")
            yield Input(placeholder="/path/to/directory", id="plugin-path")
            yield Label("", id="plugin-error", classes="validation-error")
            with Horizontal(classes="button-bar"):
                yield Button("Add", variant="primary", id="modal-add")
                yield Button("Cancel", id="modal-cancel")

    @on(Button.Pressed, "#modal-add")
    def _on_add(self) -> None:
        name = self.query_one("#plugin-name", Input).value.strip()
        path = self.query_one("#plugin-path", Input).value.strip()
        if not path:
            self.query_one("#plugin-error", Label).update("Path is required")
            return
        self.dismiss({"name": name or "default", "path": path})

    @on(Button.Pressed, "#modal-cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)


class PrxhubModal(ModalScreen[str | None]):
    """Modal for configuring prxhub URL."""

    def __init__(self, current_url: str = "") -> None:
        super().__init__()
        self._current_url = current_url

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("Configure prxhub", classes="screen-title")
            yield Label("prxhub URL", classes="form-label")
            yield Input(
                value=self._current_url or "https://prxhub.com",
                id="prxhub-url",
            )
            with Horizontal(classes="button-bar"):
                yield Button("Save", variant="primary", id="modal-save")
                yield Button("Cancel", id="modal-cancel")

    @on(Button.Pressed, "#modal-save")
    def _on_save(self) -> None:
        url = self.query_one("#prxhub-url", Input).value.strip()
        self.dismiss(url or "https://prxhub.com")

    @on(Button.Pressed, "#modal-cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)


class FirstRunDialog(ModalScreen[bool]):
    """Shown on first run when a local backend is detected."""

    def __init__(self, backend_name: str) -> None:
        super().__init__()
        self._backend = backend_name

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label(f"Found {self._backend}", classes="screen-title")
            yield Label(
                f"Use {self._backend} for synthesis and embeddings?",
                classes="form-label",
            )
            with Horizontal(classes="button-bar"):
                yield Button("Yes", variant="primary", id="first-run-yes")
                yield Button("No", id="first-run-no")

    @on(Button.Pressed, "#first-run-yes")
    def _on_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#first-run-no")
    def _on_no(self) -> None:
        self.dismiss(False)


class NothingDetectedDialog(ModalScreen[None]):
    """Shown on first run when no local backend is found."""

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("No local LLM detected", classes="screen-title")
            yield Label(
                "No LM Studio or Ollama server found. "
                "Configure a cloud backend from the menu.",
                classes="form-label",
            )
            with Horizontal(classes="button-bar"):
                yield Button("OK", variant="primary", id="nothing-ok")

    @on(Button.Pressed, "#nothing-ok")
    def _on_ok(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Sub-screens
# ---------------------------------------------------------------------------


class BackendScreen(Screen):
    """Configure synthesis or embeddings backend."""

    BINDINGS = [("escape", "go_back", "Back")]

    def __init__(
        self,
        section_key: str,
        backends: list[tuple[str, str]],
        data: dict,
    ) -> None:
        super().__init__()
        self._section_key = section_key
        self._backends = backends
        self._data = data

    def compose(self) -> ComposeResult:
        title = "Synthesis" if self._section_key == "synthesis" else "Embeddings"
        yield Label(f"{title} Backend", classes="screen-title")

        section = self._data.get(self._section_key, {})
        current_backend = section.get("backend", "")

        with VerticalScroll(classes="form-group"):
            # Radio buttons for backend selection
            pressed_idx = -1
            with RadioSet(id="backend-radios"):
                for i, (key, label) in enumerate(self._backends):
                    btn = RadioButton(label, value=(key == current_backend))
                    if key == current_backend:
                        pressed_idx = i
                    yield btn

            yield Label("Model name", classes="form-label")
            yield Input(
                value=section.get("model", ""),
                placeholder="e.g. claude-sonnet-4",
                id="model-input",
            )

            yield Label("Base URL (for custom backend)", classes="form-label")
            yield Input(
                value=section.get("base_url", ""),
                placeholder="http://localhost:1234/v1",
                id="base-url-input",
            )

            yield Label("API key env var (for cloud backends)", classes="form-label")
            yield Input(
                value=section.get("api_key_env", ""),
                placeholder="e.g. ANTHROPIC_API_KEY",
                id="api-key-env-input",
            )

        with Horizontal(classes="button-bar"):
            yield Button("Apply", variant="primary", id="apply-btn")
            yield Button("Back", id="back-btn")

    @on(Button.Pressed, "#apply-btn")
    def _on_apply(self) -> None:
        self._save_to_data()
        self.app.pop_screen()

    @on(Button.Pressed, "#back-btn")
    def _on_back(self) -> None:
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def _save_to_data(self) -> None:
        section = self._data.setdefault(self._section_key, {})
        radio_set = self.query_one("#backend-radios", RadioSet)
        if radio_set.pressed_index >= 0:
            backend_key = self._backends[radio_set.pressed_index][0]
            section["backend"] = backend_key

        model = self.query_one("#model-input", Input).value.strip()
        if model:
            section["model"] = model
        elif "model" in section:
            del section["model"]

        base_url = self.query_one("#base-url-input", Input).value.strip()
        if base_url:
            section["base_url"] = base_url
        elif "base_url" in section:
            del section["base_url"]

        api_key_env = self.query_one("#api-key-env-input", Input).value.strip()
        if api_key_env:
            section["api_key_env"] = api_key_env
        elif "api_key_env" in section:
            del section["api_key_env"]


class ProviderKeysScreen(Screen):
    """Configure provider API keys."""

    BINDINGS = [("escape", "go_back", "Back")]

    def __init__(self, data: dict) -> None:
        super().__init__()
        self._data = data

    def compose(self) -> ComposeResult:
        yield Label("Provider API Keys", classes="screen-title")
        yield Label(
            "Only needed for BYOK web research (not for local-only use).",
            classes="screen-subtitle",
        )

        section = self._data.get("providers", {})

        with VerticalScroll(classes="form-group"):
            for field, label in PROVIDER_KEYS:
                current = section.get(field, "")
                placeholder = _mask(current) if current else "paste key here"
                yield Label(label, classes="form-label")
                yield Input(
                    value="",
                    placeholder=placeholder,
                    password=True,
                    id=f"key-{field}",
                )

        with Horizontal(classes="button-bar"):
            yield Button("Apply", variant="primary", id="apply-btn")
            yield Button("Back", id="back-btn")

    @on(Button.Pressed, "#apply-btn")
    def _on_apply(self) -> None:
        section = self._data.setdefault("providers", {})
        for field, _ in PROVIDER_KEYS:
            inp = self.query_one(f"#key-{field}", Input)
            val = inp.value.strip()
            if val:
                section[field] = val
            # If empty, preserve existing (don't overwrite)
        self.app.pop_screen()

    @on(Button.Pressed, "#back-btn")
    def _on_back(self) -> None:
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()


class PluginsScreen(Screen):
    """Configure data source plugins."""

    BINDINGS = [("escape", "go_back", "Back")]

    def __init__(self, data: dict) -> None:
        super().__init__()
        self._data = data

    def compose(self) -> ComposeResult:
        yield Label("Data Source Plugins", classes="screen-title")
        yield Label("[dim]Select an entry to edit or delete it.[/dim]")

        yield OptionList(*self._build_entries(), id="plugin-entries")

        with Horizontal(classes="button-bar"):
            yield Button("Add filesystem", id="add-fs-btn")
            yield Button("Add Obsidian", id="add-obs-btn")
            yield Button("Configure prxhub", id="prxhub-btn")
            yield Button("Back", id="back-btn")

    def _build_entries(self) -> list[str]:
        plugins = self._data.get("plugins", {})
        if not plugins:
            return ["(no plugins configured)"]
        entries = []
        for pname, pval in plugins.items():
            if isinstance(pval, list):
                for inst in pval:
                    entries.append(f"{pname}:{inst.get('name', '?')}  →  {inst.get('path', '?')}")
            elif isinstance(pval, dict):
                if pname == "prxhub":
                    entries.append(f"prxhub  →  {pval.get('api_url', 'https://prxhub.com')}")
                else:
                    entries.append(f"{pname}  →  {pval}")
        return entries if entries else ["(no plugins configured)"]

    def _refresh_list(self) -> None:
        ol = self.query_one("#plugin-entries", OptionList)
        ol.clear_options()
        for entry in self._build_entries():
            ol.add_option(entry)

    @on(OptionList.OptionSelected, "#plugin-entries")
    def _on_entry_selected(self, event: OptionList.OptionSelected) -> None:
        label = str(event.option.prompt)
        if label.startswith("("):
            return
        # Parse "type:name  →  path" back to identify the entry
        parts = label.split("  →  ", 1)
        if len(parts) != 2:
            return
        spec, path = parts[0].strip(), parts[1].strip()
        if ":" in spec:
            ptype, pname = spec.split(":", 1)
        else:
            ptype, pname = spec, None

        # Show edit/delete options
        self.app.push_screen(
            EditPluginModal(ptype, pname, path),
            self._on_edit_result,
        )

    def _on_edit_result(self, result: tuple | None) -> None:
        if result is None:
            return
        action, ptype, pname, new_path = result
        plugins = self._data.get("plugins", {})
        if action == "delete":
            if ptype in plugins:
                val = plugins[ptype]
                if isinstance(val, list):
                    plugins[ptype] = [i for i in val if i.get("name") != pname]
                    if not plugins[ptype]:
                        del plugins[ptype]
                elif isinstance(val, dict):
                    del plugins[ptype]
        elif action == "edit":
            if ptype in plugins:
                val = plugins[ptype]
                if isinstance(val, list):
                    for inst in val:
                        if inst.get("name") == pname:
                            inst["path"] = new_path
                            break
                elif isinstance(val, dict):
                    val["path"] = new_path
        self._refresh_list()

    @on(Button.Pressed, "#add-fs-btn")
    def _on_add_fs(self) -> None:
        self.app.push_screen(AddPluginModal("filesystem"), self._on_fs_result)

    def _on_fs_result(self, result: dict | None) -> None:
        if result:
            plugins = self._data.setdefault("plugins", {})
            instances = plugins.setdefault("filesystem", [])
            if not isinstance(instances, list):
                instances = [instances]
                plugins["filesystem"] = instances
            instances.append(result)
            self._refresh_list()

    @on(Button.Pressed, "#add-obs-btn")
    def _on_add_obs(self) -> None:
        self.app.push_screen(AddPluginModal("obsidian"), self._on_obs_result)

    def _on_obs_result(self, result: dict | None) -> None:
        if result:
            plugins = self._data.setdefault("plugins", {})
            instances = plugins.setdefault("obsidian", [])
            if not isinstance(instances, list):
                instances = [instances]
                plugins["obsidian"] = instances
            instances.append(result)
            self._refresh_list()

    @on(Button.Pressed, "#prxhub-btn")
    def _on_prxhub(self) -> None:
        plugins = self._data.get("plugins", {})
        current = plugins.get("prxhub", {})
        url = current.get("api_url", "") if isinstance(current, dict) else ""
        self.app.push_screen(PrxhubModal(url), self._on_prxhub_result)

    def _on_prxhub_result(self, result: str | None) -> None:
        if result:
            plugins = self._data.setdefault("plugins", {})
            prxhub = plugins.setdefault("prxhub", {})
            prxhub["api_url"] = result
            self._refresh_list()

    @on(Button.Pressed, "#back-btn")
    def _on_back(self) -> None:
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ParallectApiScreen(Screen):
    """Configure Parallect API key."""

    BINDINGS = [("escape", "go_back", "Back")]

    def __init__(self, data: dict) -> None:
        super().__init__()
        self._data = data

    def compose(self) -> ComposeResult:
        yield Label("Parallect API Key (SaaS mode)", classes="screen-title")

        section = self._data.get("parallect_api", {})
        current = section.get("api_key", "")
        placeholder = _mask(current) if current else "par_live_..."

        with VerticalScroll(classes="form-group"):
            yield Label("API key", classes="form-label")
            yield Input(
                value="",
                placeholder=placeholder,
                password=True,
                id="par-api-key",
            )

        with Horizontal(classes="button-bar"):
            yield Button("Apply", variant="primary", id="apply-btn")
            yield Button("Clear", variant="warning", id="clear-btn")
            yield Button("Back", id="back-btn")

    @on(Button.Pressed, "#apply-btn")
    def _on_apply(self) -> None:
        val = self.query_one("#par-api-key", Input).value.strip()
        if val:
            self._data.setdefault("parallect_api", {})["api_key"] = val
        self.app.pop_screen()

    @on(Button.Pressed, "#clear-btn")
    def _on_clear(self) -> None:
        section = self._data.get("parallect_api", {})
        section.pop("api_key", None)
        if not section:
            self._data.pop("parallect_api", None)
        self.app.pop_screen()

    @on(Button.Pressed, "#back-btn")
    def _on_back(self) -> None:
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ViewConfigScreen(Screen):
    """Read-only view of current configuration."""

    BINDINGS = [("escape", "go_back", "Back")]

    def __init__(self, data: dict) -> None:
        super().__init__()
        self._data = data

    def compose(self) -> ComposeResult:
        yield Label("Current Configuration", classes="screen-title")
        table = DataTable(id="config-table")
        table.add_columns("Section", "Key", "Value")
        yield table
        with Horizontal(classes="button-bar"):
            yield Button("Back", id="back-btn")

    def on_mount(self) -> None:
        table = self.query_one("#config-table", DataTable)
        for section, values in self._data.items():
            if section == "plugins":
                for pname, pval in values.items():
                    if isinstance(pval, list):
                        for inst in pval:
                            table.add_row(
                                f"plugins.{pname}",
                                inst.get("name", "?"),
                                inst.get("path", str(pval)),
                            )
                    else:
                        table.add_row(f"plugins.{pname}", "", str(pval))
            elif isinstance(values, dict):
                for k, v in values.items():
                    display = _mask(str(v)) if "key" in k.lower() else str(v)
                    table.add_row(section, k, display)

    @on(Button.Pressed, "#back-btn")
    def _on_back(self) -> None:
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Main menu items
# ---------------------------------------------------------------------------

MENU_ITEMS = [
    "Synthesis backend\n  Which LLM writes your research reports",
    "Embeddings backend\n  Which model indexes your local files for search",
    "Provider API keys\n  API keys for BYOK web research (Perplexity, OpenAI, etc.)",
    "Data source plugins\n  Local directories, Obsidian vaults, prxhub",
    "Parallect API key\n  Connect to parallect.ai for SaaS-mode research",
    "View current config\n  See all settings at a glance",
]


# ---------------------------------------------------------------------------
# ConfigApp
# ---------------------------------------------------------------------------


class ConfigApp(App):
    """Textual app for ``parallect config``."""

    CSS_PATH = "config_app.tcss"

    BINDINGS = [
        ("s", "save", "Save"),
        ("q", "quit_app", "Quit"),
    ]

    def __init__(
        self,
        config_path: Path | None = None,
        probe_fn: Any | None = None,
    ) -> None:
        super().__init__()
        self._config_path = config_path or _default_config_path()
        self._probe_fn = probe_fn  # injectable for tests
        self.data: dict = {}
        self._saved = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Label("Parallect Configuration", classes="screen-title")
        yield Label(str(self._config_path), classes="screen-subtitle")
        yield OptionList(*MENU_ITEMS, id="main-menu")
        yield Footer()

    def on_mount(self) -> None:
        self.data = load_toml(self._config_path)
        # First-run detection
        if not self._config_path.exists():
            self._run_first_time_probe()

    def _run_first_time_probe(self) -> None:
        if self._probe_fn is not None:
            probe = self._probe_fn()
        else:
            from parallect.backends.probe import probe_local_backends

            probe = probe_local_backends(timeout=0.4)

        if probe.preferred_backend:
            self.push_screen(
                FirstRunDialog(probe.preferred_backend),
                self._on_first_run_result,
            )
        else:
            self.push_screen(NothingDetectedDialog())

    def _on_first_run_result(self, accepted: bool) -> None:
        if accepted:
            from parallect.backends.probe import probe_local_backends

            if self._probe_fn is not None:
                probe = self._probe_fn()
            else:
                probe = probe_local_backends(timeout=0.4)
            backend = probe.preferred_backend
            if backend:
                self.data.setdefault("synthesis", {})["backend"] = backend
                self.data.setdefault("embeddings", {})["backend"] = backend

    @on(OptionList.OptionSelected, "#main-menu")
    def _on_menu_select(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        if idx == 0:
            self.push_screen(
                BackendScreen("synthesis", SYNTHESIS_BACKENDS, self.data)
            )
        elif idx == 1:
            self.push_screen(
                BackendScreen("embeddings", EMBEDDINGS_BACKENDS, self.data)
            )
        elif idx == 2:
            self.push_screen(ProviderKeysScreen(self.data))
        elif idx == 3:
            self.push_screen(PluginsScreen(self.data))
        elif idx == 4:
            self.push_screen(ParallectApiScreen(self.data))
        elif idx == 5:
            self.push_screen(ViewConfigScreen(self.data))

    def action_save(self) -> None:
        write_toml(self._config_path, self.data)
        self._saved = True
        self.notify("Configuration saved")
        self.exit()

    def action_quit_app(self) -> None:
        self.exit()
