"""parallect config -- interactive configuration setup.

First-time setup is local-first: we probe LM Studio (port 1234) and Ollama
(port 11434) before prompting for any cloud API keys. When a local server
is found, synthesis + embeddings default to that backend. Otherwise we
prompt the user to configure a cloud backend.
"""

from __future__ import annotations

from pathlib import Path

import platformdirs
import typer
from rich.console import Console

from parallect.backends.probe import probe_local_backends

console = Console()


def config_cmd() -> None:
    """Interactive setup for API keys and preferences."""
    config_dir = Path(platformdirs.user_config_dir("parallect"))
    config_path = config_dir / "config.toml"

    console.print("[bold]Parallect Configuration[/bold]\n")
    console.print(f"Config file: {config_path}\n")

    first_run = not config_path.exists()

    if not first_run:
        console.print("[dim]Existing config found. Values shown as defaults.[/dim]\n")

    # --- Local-first detection -------------------------------------------
    local_defaults: dict[str, str] = {}
    if first_run:
        console.print("[dim]Probing for local LLM servers...[/dim]")
        probe = probe_local_backends(timeout=0.4)
        if probe.lmstudio_reachable:
            console.print("[green]Found LM Studio at http://localhost:1234[/green]")
        if probe.ollama_reachable:
            console.print("[green]Found Ollama at http://localhost:11434[/green]")

        if probe.preferred_backend:
            pref = probe.preferred_backend
            use_local = typer.confirm(
                f"Default synthesis + embeddings to {pref}?",
                default=True,
            )
            if use_local:
                local_defaults["synthesis.backend"] = pref
                local_defaults["embeddings.backend"] = pref
        elif not probe.any_reachable:
            console.print(
                "[yellow]No local LLM detected. Install Ollama or LM Studio for "
                "private research, or configure a cloud provider below.[/yellow]\n"
            )

    # --- Collect API keys ------------------------------------------------
    sections: dict[str, dict[str, object]] = {
        "providers": {},
        "defaults": {},
        "signing": {},
        "synthesis": {},
        "embeddings": {},
    }

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

    openrouter_key = typer.prompt(
        "OpenRouter API key (aggregator; optional)", default="", show_default=False
    )
    if openrouter_key:
        sections["providers"]["openrouter_api_key"] = openrouter_key

    parallect_key = typer.prompt(
        "Parallect API key (for enhance/publish)", default="", show_default=False
    )

    # --- Decide synthesis + embeddings defaults --------------------------
    # If local-first set something, honor it. Otherwise, pick from whatever
    # keys the user just entered.
    synthesis_backend = local_defaults.get("synthesis.backend", "")
    embeddings_backend = local_defaults.get("embeddings.backend", "")

    if not synthesis_backend:
        if anthropic_key:
            synthesis_backend = "anthropic"
        elif openrouter_key:
            synthesis_backend = "openrouter"
        elif openai_key:
            synthesis_backend = "openai"
        elif google_key:
            synthesis_backend = "gemini"

    if not embeddings_backend:
        # Anthropic has no embeddings endpoint -- pick openai or gemini.
        if openai_key:
            embeddings_backend = "openai"
        elif google_key:
            embeddings_backend = "gemini"
        elif openrouter_key:
            embeddings_backend = "openrouter"

    if synthesis_backend:
        sections["synthesis"]["backend"] = synthesis_backend
    if embeddings_backend:
        sections["embeddings"]["backend"] = embeddings_backend

    # --- Write config ----------------------------------------------------
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

    if sections["synthesis"]:
        lines.append("[synthesis]")
        for k, v in sections["synthesis"].items():
            lines.append(f'{k} = "{v}"')
        lines.append("")

    if sections["embeddings"]:
        lines.append("[embeddings]")
        for k, v in sections["embeddings"].items():
            lines.append(f'{k} = "{v}"')
        lines.append("")

    config_path.write_text("\n".join(lines))
    console.print(f"\n[green]Config saved to {config_path}[/green]")
