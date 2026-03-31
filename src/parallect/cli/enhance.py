"""parallect enhance -- enhance via Parallect API."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

console = Console()


def enhance_cmd(
    bundle_path: str = typer.Argument(help="Path to .prx bundle"),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Parallect API key (par_live_...)"
    ),
    tier: str = typer.Option("standard", "--tier", help="Enhancement tier: standard, premium"),
) -> None:
    """Enhance a bundle via the Parallect API (adds sources, evidence graph, confidence)."""
    asyncio.run(_enhance_async(bundle_path, api_key, tier))


async def _enhance_async(bundle_path: str, api_key: str | None, tier: str) -> None:
    from parallect.api import enhance_bundle
    from parallect.config_mod.settings import ParallectSettings

    settings = ParallectSettings.load()
    key = api_key or settings.parallect_api_key

    if not key:
        console.print(
            "[red]Parallect API key required. "
            "Set via --api-key or in config.[/red]"
        )
        raise typer.Exit(1)

    path = Path(bundle_path)
    if not path.exists():
        console.print(f"[red]File not found: {bundle_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Enhancing:[/bold] {path.name}")
    console.print(f"[dim]Tier: {tier}[/dim]")

    try:
        job = await enhance_bundle(path, key, tier=tier)
        console.print("[green]Enhancement complete![/green]")
        if job.enhanced_path:
            console.print(f"  Output: {job.enhanced_path}")
    except Exception as e:
        console.print(f"[red]Enhancement failed: {e}[/red]")
        raise typer.Exit(1)
