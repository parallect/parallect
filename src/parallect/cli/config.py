"""parallect config -- Textual TUI for interactive configuration.

The Typer command launches a Textual app. TOML utilities are re-exported
from config_app for backward compatibility.
"""

from __future__ import annotations

from parallect.cli.config_app import (  # noqa: F401 -- re-export
    _default_config_path as _config_path,
    load_toml as _load_toml,
    write_toml as _write_toml,
    _mask,
)


def config_cmd() -> None:
    """Interactive setup for API keys and preferences."""
    from parallect.cli.config_app import ConfigApp

    app = ConfigApp()
    app.run()
