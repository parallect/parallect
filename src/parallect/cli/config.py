"""parallect config -- interactive configuration setup."""

from __future__ import annotations

from pathlib import Path

import platformdirs
import typer
from rich.console import Console

console = Console()


def config_cmd() -> None:
    """Interactive setup for API keys and preferences."""
    config_dir = Path(platformdirs.user_config_dir("parallect"))
    config_path = config_dir / "config.toml"

    console.print("[bold]Parallect Configuration[/bold]\n")
    console.print(f"Config file: {config_path}\n")

    if config_path.exists():
        console.print("[dim]Existing config found. Values shown as defaults.[/dim]\n")

    # Collect API keys
    sections: dict[str, dict[str, str]] = {"providers": {}, "defaults": {}, "signing": {}}

    perplexity_key = typer.prompt("Perplexity API key", default="", show_default=False)
    if perplexity_key:
        sections["providers"]["perplexity_api_key"] = perplexity_key

    google_key = typer.prompt("Google AI API key", default="", show_default=False)
    if google_key:
        sections["providers"]["google_api_key"] = google_key

    openai_key = typer.prompt("OpenAI API key", default="", show_default=False)
    if openai_key:
        sections["providers"]["openai_api_key"] = openai_key

    xai_key = typer.prompt("xAI (Grok) API key", default="", show_default=False)
    if xai_key:
        sections["providers"]["xai_api_key"] = xai_key

    anthropic_key = typer.prompt("Anthropic API key", default="", show_default=False)
    if anthropic_key:
        sections["providers"]["anthropic_api_key"] = anthropic_key

    parallect_key = typer.prompt(
        "Parallect API key (for enhance/publish)", default="", show_default=False
    )

    # Write config
    config_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Parallect CLI configuration\n"]

    if sections["providers"]:
        lines.append("[providers]")
        for k, v in sections["providers"].items():
            lines.append(f'{k} = "{v}"')
        lines.append("")

    if parallect_key:
        lines.append("[parallect_api]")
        lines.append(f'api_key = "{parallect_key}"')
        lines.append("")

    lines.append("[defaults]")
    lines.append('providers = ["perplexity", "gemini", "openai"]')
    lines.append('synthesize_with = "anthropic"')
    lines.append("budget_cap_usd = 2.00")
    lines.append("")

    config_path.write_text("\n".join(lines))
    console.print(f"\n[green]Config saved to {config_path}[/green]")
