"""`parallect plugins` — manage data source plugins.

Subcommands:

  - ``parallect plugins list``                       -- show discovered plugins
  - ``parallect plugins status [name]``              -- health-check one or all
  - ``parallect plugins index <name> [--force]``     -- trigger (re)index
  - ``parallect plugins config <name>``              -- interactive config
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import platformdirs
import typer
from rich.console import Console
from rich.table import Table

from parallect.plugins.data_sources import PluginError, get, get_registry

console = Console()

plugins_app = typer.Typer(
    name="plugins",
    help="Manage data source plugins.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@plugins_app.command("list")
def list_cmd(
    json_out: bool = typer.Option(
        False, "--json", help="Output JSON instead of a rich table."
    ),
) -> None:
    """List all discovered data source plugins."""
    reg = get_registry()
    if json_out:
        payload = [
            {
                "name": p.name,
                "display_name": getattr(p, "display_name", p.name),
                "requires_index": bool(getattr(p, "requires_index", False)),
            }
            for p in reg.values()
        ]
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Data source plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Display")
    table.add_column("Needs index", justify="center")
    for p in reg.values():
        table.add_row(
            p.name,
            getattr(p, "display_name", p.name),
            "yes" if getattr(p, "requires_index", False) else "no",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@plugins_app.command("status")
def status_cmd(
    name: str | None = typer.Argument(None, help="Plugin name (omit for all)."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Run ``health_check()`` for one or all plugins."""
    reg = get_registry()
    if name:
        if name not in reg:
            console.print(f"[red]unknown plugin:[/red] {name}")
            raise typer.Exit(1)
        targets = {name: reg[name]}
    else:
        targets = reg

    results: dict[str, dict] = {}
    for n, plugin in targets.items():
        try:
            results[n] = asyncio.run(plugin.health_check())
        except Exception as exc:
            results[n] = {"status": "error", "error": str(exc)}

    if json_out:
        typer.echo(json.dumps(results, indent=2, default=str))
        return

    for n, info in results.items():
        status = info.get("status", "unknown")
        color = {
            "ok": "green",
            "error": "red",
            "not_indexed": "yellow",
            "unconfigured": "yellow",
        }.get(status, "white")
        console.print(f"[{color}]{n}[/{color}]: {status}")
        for k, v in info.items():
            if k == "status":
                continue
            console.print(f"  [dim]{k}:[/dim] {v}")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@plugins_app.command("index")
def index_cmd(
    name: str = typer.Argument(..., help="Plugin name."),
    force: bool = typer.Option(False, "--force", help="Re-index even if fresh."),
) -> None:
    """(Re)build a plugin's local index."""
    try:
        plugin = get(name)
    except PluginError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if not getattr(plugin, "requires_index", False):
        console.print(
            f"[yellow]{name} does not require an index; nothing to do.[/yellow]"
        )
        return

    console.print(f"[cyan]Indexing {name}...[/cyan]")
    try:
        stats = asyncio.run(plugin.index(force=force))
    except Exception as exc:
        console.print(f"[red]index failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"[green]Indexed {stats.documents_indexed} chunks[/green]")
    console.print(f"  skipped: {stats.documents_skipped}")
    console.print(f"  elapsed: {stats.elapsed_seconds:.2f}s")
    if stats.index_path:
        console.print(f"  index:   {stats.index_path}")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@plugins_app.command("config")
def config_cmd(
    name: str = typer.Argument(..., help="Plugin name."),
) -> None:
    """Interactively configure a plugin and persist to user config.toml.

    Appends a ``[plugins.<name>]`` section to the user's config file. The
    orchestrator reads those sections at research time and passes them to
    :meth:`plugin.configure`.
    """
    try:
        plugin = get(name)
    except PluginError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    config_path = Path(platformdirs.user_config_dir("parallect")) / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if name == "prxhub":
        api_url = typer.prompt("prxhub API URL", default="https://prxhub.com")
        api_key = typer.prompt("prxhub API key (optional)", default="", show_default=False)
        lines.append("\n[plugins.prxhub]")
        lines.append(f'api_url = "{api_url}"')
        if api_key:
            lines.append(f'api_key = "{api_key}"')
    elif name == "filesystem":
        instance = typer.prompt("Instance name", default="default")
        path = typer.prompt("Path to index")
        include = typer.prompt(
            "Include globs (comma-separated)", default="*.md,*.txt"
        )
        exclude = typer.prompt(
            "Exclude substrings (comma-separated)",
            default=".git/,node_modules/,.DS_Store",
        )
        max_kb = typer.prompt("Max file size (KB)", default="500")
        lines.append("\n[[plugins.filesystem]]")
        lines.append(f'name = "{instance}"')
        lines.append(f'path = "{path}"')
        lines.append(
            "include = [" + ", ".join(f'"{s.strip()}"' for s in include.split(",") if s.strip()) + "]"
        )
        lines.append(
            "exclude = [" + ", ".join(f'"{s.strip()}"' for s in exclude.split(",") if s.strip()) + "]"
        )
        lines.append(f"max_file_kb = {int(max_kb)}")
    elif name == "prior-research":
        db_path = typer.prompt(
            "Cache DB path",
            default=str(Path.home() / ".parallect" / "research-cache.db"),
        )
        lines.append("\n[plugins.prior-research]")
        lines.append(f'db_path = "{db_path}"')
    else:
        console.print(
            f"[yellow]No interactive config defined for '{name}'. "
            f"Edit {config_path} manually.[/yellow]"
        )
        return

    with open(config_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    console.print(f"[green]Appended config for '{name}' to {config_path}[/green]")


__all__ = ["plugins_app"]
