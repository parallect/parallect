"""parallect config -- menu-driven configuration.

Replaces the old "walk through every key" flow with a numbered menu so
users can configure exactly what they need without being prompted for
everything. Non-interactive `parallect config set key=value` for scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import platformdirs
import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _config_path() -> Path:
    return Path(platformdirs.user_config_dir("parallect")) / "config.toml"


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def _write_toml(path: Path, data: dict) -> None:
    """Write a flat-ish dict back as TOML. Handles one level of nesting +
    arrays-of-tables for plugins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Parallect CLI configuration\n"]
    for section, values in data.items():
        if section == "plugins":
            for plugin_name, instances in values.items():
                if isinstance(instances, list):
                    for inst in instances:
                        lines.append(f"[[plugins.{plugin_name}]]")
                        for k, v in inst.items():
                            lines.append(f'{k} = {_toml_val(v)}')
                        lines.append("")
                elif isinstance(instances, dict):
                    lines.append(f"[plugins.{plugin_name}]")
                    for k, v in instances.items():
                        lines.append(f'{k} = {_toml_val(v)}')
                    lines.append("")
        elif isinstance(values, dict):
            lines.append(f"[{section}]")
            for k, v in values.items():
                lines.append(f'{k} = {_toml_val(v)}')
            lines.append("")
        else:
            lines.append(f'{section} = {_toml_val(values)}')
    path.write_text("\n".join(lines))


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


def _mask(key: str) -> str:
    if not key or len(key) < 8:
        return key or "(not set)"
    return key[:8] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# Menu options
# ---------------------------------------------------------------------------


def _configure_synthesis(data: dict) -> None:
    section = data.setdefault("synthesis", {})
    console.print("\n[bold]Synthesis Backend[/bold]")
    console.print("  Current:", section.get("backend", "(not set)"),
                  "·", section.get("model", "(default)"))
    console.print()
    console.print("  [1] lmstudio  — local LM Studio (private, free)")
    console.print("  [2] ollama    — local Ollama (private, free)")
    console.print("  [3] anthropic — Claude (cloud)")
    console.print("  [4] openai    — GPT (cloud)")
    console.print("  [5] gemini    — Gemini (cloud, cheap)")
    console.print("  [6] openrouter — OpenRouter (cloud, any model)")
    console.print("  [7] custom    — any OpenAI-compatible URL")
    console.print("  [0] back")

    choice = typer.prompt("Choice", default="0")
    backends = {"1": "lmstudio", "2": "ollama", "3": "anthropic", "4": "openai",
                "5": "gemini", "6": "openrouter", "7": "custom"}
    if choice in backends:
        section["backend"] = backends[choice]
        if choice == "7":
            section["base_url"] = typer.prompt("Base URL", default=section.get("base_url", ""))
        model = typer.prompt("Model name", default=section.get("model", ""))
        if model:
            section["model"] = model
        if choice in ("3", "4", "5", "6"):
            env = typer.prompt("API key env var", default=section.get("api_key_env", ""))
            if env:
                section["api_key_env"] = env
        console.print(f"[green]Synthesis set to {section['backend']}[/green]")


def _configure_embeddings(data: dict) -> None:
    section = data.setdefault("embeddings", {})
    console.print("\n[bold]Embeddings Backend[/bold]")
    console.print("  Current:", section.get("backend", "(not set)"),
                  "·", section.get("model", "(default)"))
    console.print()
    console.print("  [1] lmstudio  — local (nomic-embed-text)")
    console.print("  [2] ollama    — local")
    console.print("  [3] openai    — text-embedding-3-small")
    console.print("  [4] gemini    — text-embedding-004")
    console.print("  [5] openrouter")
    console.print("  [6] custom")
    console.print("  [0] back")

    choice = typer.prompt("Choice", default="0")
    backends = {"1": "lmstudio", "2": "ollama", "3": "openai",
                "4": "gemini", "5": "openrouter", "6": "custom"}
    if choice in backends:
        section["backend"] = backends[choice]
        if choice == "6":
            section["base_url"] = typer.prompt("Base URL", default=section.get("base_url", ""))
        model = typer.prompt("Model name", default=section.get("model", ""))
        if model:
            section["model"] = model
        console.print(f"[green]Embeddings set to {section['backend']}[/green]")


def _configure_providers(data: dict) -> None:
    section = data.setdefault("providers", {})
    console.print("\n[bold]Provider API Keys[/bold]")
    console.print("  Only needed for BYOK web research (not for local-only use).\n")

    keys = [
        ("perplexity_api_key", "Perplexity"),
        ("google_api_key", "Google AI (Gemini)"),
        ("openai_api_key", "OpenAI"),
        ("xai_api_key", "xAI (Grok)"),
        ("anthropic_api_key", "Anthropic"),
        ("openrouter_api_key", "OpenRouter"),
    ]
    for field, label in keys:
        current = section.get(field, "")
        prompt_text = f"  {label}"
        if current:
            prompt_text += f" [{_mask(current)}]"
        val = typer.prompt(prompt_text, default="", show_default=False)
        if val:
            section[field] = val
        elif current:
            section[field] = current


def _configure_plugins(data: dict) -> None:
    plugins = data.setdefault("plugins", {})
    console.print("\n[bold]Data Source Plugins[/bold]\n")
    console.print("  [1] Add a local directory (filesystem)")
    console.print("  [2] Add an Obsidian vault")
    console.print("  [3] Configure prxhub")
    console.print("  [4] View current plugins")
    console.print("  [0] back")

    choice = typer.prompt("Choice", default="0")

    if choice == "1":
        name = typer.prompt("Instance name", default="research")
        path = typer.prompt("Directory path")
        instances = plugins.setdefault("filesystem", [])
        if not isinstance(instances, list):
            instances = [instances]
            plugins["filesystem"] = instances
        instances.append({"name": name, "path": path})
        console.print(f"[green]Added filesystem:{name} → {path}[/green]")

    elif choice == "2":
        name = typer.prompt("Instance name", default="notes")
        path = typer.prompt("Vault path")
        instances = plugins.setdefault("obsidian", [])
        if not isinstance(instances, list):
            instances = [instances]
            plugins["obsidian"] = instances
        instances.append({"name": name, "path": path})
        console.print(f"[green]Added obsidian:{name} → {path}[/green]")

    elif choice == "3":
        prxhub = plugins.setdefault("prxhub", {})
        url = typer.prompt("prxhub URL", default=prxhub.get("api_url", "https://prxhub.com"))
        prxhub["api_url"] = url
        console.print(f"[green]prxhub → {url}[/green]")

    elif choice == "4":
        if not plugins:
            console.print("  (none configured)")
        for pname, pval in plugins.items():
            if isinstance(pval, list):
                for inst in pval:
                    console.print(f"  {pname}:{inst.get('name', '?')} → {inst.get('path', '?')}")
            elif isinstance(pval, dict):
                console.print(f"  {pname} → {pval}")


def _configure_parallect_api(data: dict) -> None:
    section = data.setdefault("parallect_api", {})
    console.print("\n[bold]Parallect API (SaaS mode)[/bold]")
    current = section.get("api_key", "")
    if current:
        console.print(f"  Current: {_mask(current)}")
    val = typer.prompt("  API key (par_live_...)", default="", show_default=False)
    if val:
        section["api_key"] = val
    elif current:
        section["api_key"] = current


def _show_current(data: dict) -> None:
    console.print("\n[bold]Current Configuration[/bold]\n")
    t = Table(show_header=True)
    t.add_column("Section", style="cyan")
    t.add_column("Key")
    t.add_column("Value")
    for section, values in data.items():
        if section == "plugins":
            for pname, pval in values.items():
                if isinstance(pval, list):
                    for inst in pval:
                        t.add_row(f"plugins.{pname}", inst.get("name", "?"), inst.get("path", str(pval)))
                else:
                    t.add_row(f"plugins.{pname}", "", str(pval))
        elif isinstance(values, dict):
            for k, v in values.items():
                display = _mask(str(v)) if "key" in k.lower() else str(v)
                t.add_row(section, k, display)
    console.print(t)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def config_cmd() -> None:
    """Interactive setup for API keys and preferences."""
    path = _config_path()
    data = _load_toml(path)

    console.print(f"[bold]Parallect Configuration[/bold]")
    console.print(f"[dim]{path}[/dim]\n")

    # First-run local detection
    if not path.exists():
        from parallect.backends.probe import probe_local_backends
        console.print("[dim]Probing for local LLM servers...[/dim]")
        probe = probe_local_backends(timeout=0.4)
        if probe.lmstudio_reachable:
            console.print("[green]Found LM Studio[/green]")
        if probe.ollama_reachable:
            console.print("[green]Found Ollama[/green]")
        if probe.preferred_backend:
            pref = probe.preferred_backend
            if typer.confirm(f"Default synthesis + embeddings to {pref}?", default=True):
                data.setdefault("synthesis", {})["backend"] = pref
                data.setdefault("embeddings", {})["backend"] = pref
        console.print()

    while True:
        console.print("\n  [bold]What do you want to configure?[/bold]\n")
        console.print("  [1] Synthesis backend (which LLM writes your reports)")
        console.print("  [2] Embeddings backend (which model indexes your files)")
        console.print("  [3] Provider API keys (for BYOK web research)")
        console.print("  [4] Data source plugins (local files, Obsidian, prxhub)")
        console.print("  [5] Parallect API key (SaaS mode)")
        console.print("  [6] View current config")
        console.print("  [s] Save and exit")
        console.print("  [q] Quit without saving")

        choice = typer.prompt("\nChoice", default="s")

        if choice == "1":
            _configure_synthesis(data)
        elif choice == "2":
            _configure_embeddings(data)
        elif choice == "3":
            _configure_providers(data)
        elif choice == "4":
            _configure_plugins(data)
        elif choice == "5":
            _configure_parallect_api(data)
        elif choice == "6":
            _show_current(data)
        elif choice == "s":
            _write_toml(path, data)
            console.print(f"\n[green]Saved to {path}[/green]")
            break
        elif choice == "q":
            console.print("[yellow]Exiting without saving.[/yellow]")
            break
