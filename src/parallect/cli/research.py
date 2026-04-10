"""parallect research -- run multi-provider research."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from parallect.config_mod.settings import ParallectSettings

console = Console()


def research_cmd(
    query: str = typer.Argument(help="Research query"),
    providers: str | None = typer.Option(
        None, "--providers", "-p", help="Comma-separated provider names, or 'all'"
    ),
    synthesize_with: str | None = typer.Option(
        None, "--synthesize-with", "-s",
        help="Model for synthesis (e.g. anthropic, ollama/llama3.2)",
    ),
    no_synthesis: bool = typer.Option(False, "--no-synthesis", help="Skip synthesis step"),
    budget_cap: float | None = typer.Option(
        None, "--budget-cap", help="Maximum cost in USD"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output path for .prx file"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", help="Output directory for .prx file"
    ),
    local: bool = typer.Option(False, "--local", help="Use only local providers (Ollama)"),
    no_sign: bool = typer.Option(False, "--no-sign", help="Skip bundle signing"),
    timeout: float = typer.Option(120.0, "--timeout", help="Per-provider timeout in seconds"),
    deep: bool = typer.Option(
        False, "--deep",
        help="Use premium deep-research models (slower, more thorough, higher cost)",
    ),
) -> None:
    """Run multi-provider deep research and produce a .prx bundle."""
    asyncio.run(_research_async(
        query=query,
        providers_str=providers,
        synthesize_with=synthesize_with,
        no_synthesis=no_synthesis,
        budget_cap=budget_cap,
        output=output,
        output_dir=output_dir,
        local=local,
        no_sign=no_sign,
        timeout=timeout,
        deep=deep,
    ))


async def _research_async(
    query: str,
    providers_str: str | None,
    synthesize_with: str | None,
    no_synthesis: bool,
    budget_cap: float | None,
    output: str | None,
    output_dir: str | None,
    local: bool,
    no_sign: bool,
    timeout: float,
    deep: bool = False,
) -> None:
    from parallect.config_mod.settings import ParallectSettings

    settings = ParallectSettings.load()

    # Resolve providers
    provider_instances = _resolve_providers(providers_str, local, settings, deep=deep)

    if not provider_instances:
        console.print(
            "[red]No providers available. Run 'parallect config' to set up API keys.[/red]"
        )
        raise typer.Exit(1)

    synth_model = synthesize_with or settings.synthesize_with
    if no_synthesis:
        synth_model = None

    # Resolve output path (fall back to settings.output_dir so bundles are always saved)
    out_path = output or output_dir or settings.output_dir

    console.print(f"\n[bold]Researching:[/bold] {query}")
    if deep:
        console.print("[bold yellow]Deep research mode[/bold yellow] — premium models, longer runtime")
    console.print(f"[dim]Providers: {', '.join(p.name for p in provider_instances)}[/dim]")
    if synth_model:
        console.print(f"[dim]Synthesis: {synth_model}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running research...", total=None)

        from parallect.orchestrator.parallel import research

        try:
            bundle = await research(
                query=query,
                providers=provider_instances,
                synthesize_with=synth_model,
                no_synthesis=no_synthesis,
                budget_cap_usd=budget_cap,
                timeout_per_provider=timeout,
                output=out_path,
                no_sign=no_sign,
                settings=settings,
            )
        except RuntimeError as exc:
            progress.stop()
            console.print(f"\n[red bold]Research failed:[/red bold] {exc}")
            raise typer.Exit(1)

        progress.update(task, description="Done!")

    console.print(f"\n[green]Bundle created:[/green] {bundle.manifest.id}")
    console.print(f"  Providers: {', '.join(bundle.manifest.providers_used)}")
    console.print(f"  Synthesis: {'yes' if bundle.manifest.has_synthesis else 'no'}")
    if bundle.attestations:
        console.print(f"  Signed: yes ({len(bundle.attestations)} attestation(s))")
    if bundle.manifest.total_cost_usd:
        console.print(f"  Cost: ${bundle.manifest.total_cost_usd:.4f}")
    if out_path:
        console.print(f"  Saved to: {out_path}")


def _resolve_providers(
    providers_str: str | None,
    local: bool,
    settings: ParallectSettings,
    *,
    deep: bool = False,
) -> list:
    """Resolve provider names to provider instances."""
    from parallect.providers.anthropic import AnthropicProvider
    from parallect.providers.gemini import GeminiProvider
    from parallect.providers.grok import GrokProvider
    from parallect.providers.ldr import LDRProvider
    from parallect.providers.lmstudio import LMStudioProvider
    from parallect.providers.ollama import OllamaProvider
    from parallect.providers.openai_dr import OpenAIDRProvider
    from parallect.providers.perplexity import PerplexityProvider

    if local:
        return [OllamaProvider(model=settings.ollama_default_model, host=settings.ollama_host)]

    if providers_str:
        names = [n.strip() for n in providers_str.split(",")]
    else:
        names = settings.providers

    if "all" in names:
        names = ["perplexity", "gemini", "openai", "grok", "anthropic"]

    instances = []
    provider_map = {
        "perplexity": lambda: PerplexityProvider(api_key=settings.perplexity_api_key),
        "gemini": lambda: GeminiProvider(api_key=settings.google_api_key, deep=deep),
        "openai": lambda: OpenAIDRProvider(api_key=settings.openai_api_key, deep=deep),
        "grok": lambda: GrokProvider(api_key=settings.xai_api_key, deep=deep),
        "anthropic": lambda: AnthropicProvider(api_key=settings.anthropic_api_key, deep=deep),
        "ollama": lambda: OllamaProvider(
            model=settings.ollama_default_model, host=settings.ollama_host
        ),
        "lmstudio": lambda: LMStudioProvider(
            model=settings.lmstudio_default_model, host=settings.lmstudio_host
        ),
        "ldr": lambda: LDRProvider(),
    }

    for name in names:
        if name in provider_map:
            instance = provider_map[name]()
            if instance.is_available():
                instances.append(instance)
            else:
                console.print(
                    f"[yellow]Provider '{name}' not available (missing API key?)[/yellow]"
                )
        else:
            console.print(f"[yellow]Unknown provider: {name}[/yellow]")

    return instances
