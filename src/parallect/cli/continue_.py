"""parallect continue -- follow-on research linked to parent."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def continue_cmd(
    bundle_path: str = typer.Argument(help="Path to parent .prx bundle"),
    query: str | None = typer.Argument(
        None, help="Follow-on query (or pick from suggestions)"
    ),
    providers: str | None = typer.Option(
        None, "--providers", "-p", help="Comma-separated provider names"
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Output path"),
    timeout: float = typer.Option(
        900.0, "--timeout",
        help="Per-provider timeout in seconds. Default 900s (15 min) matches "
             "`parallect research` so follow-on runs behave the same way as "
             "the initial query.",
    ),
) -> None:
    """Run follow-on research linked to a parent bundle."""
    from prx_spec import read_bundle

    path = Path(bundle_path)
    if not path.exists():
        console.print(f"[red]File not found: {bundle_path}[/red]")
        raise typer.Exit(1)

    parent = read_bundle(path)
    parent_id = parent.manifest.id

    console.print(f"[dim]Continuing from: {parent_id}[/dim]")
    console.print(
        f"[dim]Original query: {parent.manifest.query}[/dim]"
    )

    # If no query provided, offer follow-on suggestions from the parent
    follow_on_query = query
    if not follow_on_query:
        follow_on_query = _pick_follow_on(parent)

    if not follow_on_query:
        console.print("[red]No query provided and no follow-ons available.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Follow-on query:[/bold] {follow_on_query}")

    # Build parent context summary for providers
    parent_context = _build_parent_context(parent)

    asyncio.run(_continue_async(
        query=follow_on_query,
        parent_bundle_id=parent_id,
        parent_context=parent_context,
        providers_str=providers,
        output=output,
        timeout=timeout,
    ))


def _pick_follow_on(parent: object) -> str | None:
    """Display follow-on suggestions and let the user pick one."""
    follow_ons = getattr(parent, "follow_ons", None)
    if not follow_ons:
        return None

    # follow_ons may be a list of objects with .query attr or dicts
    items: list[str] = []
    for fo in follow_ons:
        q = getattr(fo, "query", None) or (
            fo.get("query") if isinstance(fo, dict) else None
        )
        if q:
            items.append(q)

    if not items:
        return None

    console.print("\n[bold]Suggested follow-on questions:[/bold]")
    for i, item in enumerate(items, 1):
        console.print(f"  [cyan]{i}.[/cyan] {item}")
    console.print(f"  [cyan]{len(items) + 1}.[/cyan] [dim]Enter custom query[/dim]")

    choice = Prompt.ask(
        "\nSelect a follow-on",
        default="1",
    )

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(items):
            return items[idx]
    except ValueError:
        # Treat the input as a custom query
        return choice

    # Custom query option
    return Prompt.ask("Enter your follow-on query")


def _build_parent_context(parent: object) -> str:
    """Build a concise context summary from the parent bundle."""
    parts: list[str] = []

    # Include original query
    if hasattr(parent, "manifest") and parent.manifest.query:
        parts.append(f"Original research question: {parent.manifest.query}")

    # Include synthesis summary (first 500 chars)
    synthesis_md = getattr(parent, "synthesis_md", None)
    if synthesis_md:
        summary = synthesis_md[:500]
        if len(synthesis_md) > 500:
            summary += "..."
        parts.append(f"Previous synthesis summary:\n{summary}")

    # Include key claims if available
    claims = getattr(parent, "claims", None)
    if claims:
        claim_list = getattr(claims, "claims", None) or claims
        if hasattr(claim_list, "__iter__"):
            claim_texts = []
            for c in list(claim_list)[:10]:
                text = getattr(c, "content", None) or (
                    c.get("content") if isinstance(c, dict) else str(c)
                )
                if text:
                    claim_texts.append(f"- {text}")
            if claim_texts:
                parts.append(
                    "Key claims from previous research:\n"
                    + "\n".join(claim_texts)
                )

    return "\n\n".join(parts) if parts else ""


async def _continue_async(
    query: str,
    parent_bundle_id: str,
    parent_context: str,
    providers_str: str | None,
    output: str | None,
    timeout: float = 900.0,
) -> None:
    from parallect.cli.research import _resolve_providers
    from parallect.config_mod.settings import ParallectSettings
    from parallect.orchestrator.parallel import research

    settings = ParallectSettings.load()
    provider_instances = _resolve_providers(
        providers_str, False, settings, timeout=timeout,
    )

    if not provider_instances:
        console.print(
            "[red]No providers available. "
            "Run 'parallect config' to set up API keys.[/red]"
        )
        raise typer.Exit(1)

    synth_model = settings.synthesize_with

    console.print(
        f"[dim]Providers: "
        f"{', '.join(p.name for p in provider_instances)}[/dim]"
    )

    bundle = await research(
        query=query,
        providers=provider_instances,
        synthesize_with=synth_model,
        parent_bundle_id=parent_bundle_id,
        parent_context=parent_context,
        output=output,
        timeout_per_provider=timeout,
    )

    console.print(f"\n[green]Bundle created:[/green] {bundle.manifest.id}")
    console.print(f"  Parent: {parent_bundle_id}")
    console.print(
        f"  Providers: {', '.join(bundle.manifest.providers_used)}"
    )
    if output:
        console.print(f"  Saved to: {output}")
