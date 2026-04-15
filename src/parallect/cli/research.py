"""parallect research -- run multi-provider research (BYOK or SaaS)."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from parallect.config_mod.settings import ParallectSettings

console = Console()


# ---------------------------------------------------------------------------
# Tier definitions (shared between BYOK and SaaS modes)
# ---------------------------------------------------------------------------

VALID_TIERS = ("nano", "lite", "normal", "deep", "max")


@dataclass(frozen=True)
class TierConfig:
    name: str
    budget_cap_usd: float
    deep: bool
    providers: tuple[str, ...] | None  # None = respect user/settings default


TIER_CONFIGS: dict[str, TierConfig] = {
    "nano": TierConfig("nano", 0.25, False, ("perplexity",)),
    "lite": TierConfig("lite", 0.75, False, ("perplexity", "gemini")),
    "normal": TierConfig("normal", 2.00, False, None),
    "deep": TierConfig("deep", 6.00, True, None),
    "max": TierConfig("max", 15.00, True, ("perplexity", "gemini", "openai", "grok", "anthropic")),
}


def resolve_tier(tier: str | None) -> TierConfig:
    if tier is None:
        return TIER_CONFIGS["normal"]
    if tier not in TIER_CONFIGS:
        raise typer.BadParameter(
            f"Invalid tier '{tier}'. Valid tiers: {', '.join(VALID_TIERS)}"
        )
    return TIER_CONFIGS[tier]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteDecision:
    mode: str  # "saas" or "byok"
    api_key: str | None
    reason: str


def decide_route(
    *, api_key_flag: str | None, local_flag: bool, env_key: str | None
) -> RouteDecision:
    """Pick SaaS vs BYOK.

    - ``--local`` always forces BYOK.
    - Otherwise: if an API key is provided via flag or env, use SaaS.
    - Else BYOK.
    """
    if local_flag:
        return RouteDecision("byok", None, "--local forced")
    if api_key_flag:
        return RouteDecision("saas", api_key_flag, "--api-key flag")
    if env_key:
        return RouteDecision("saas", env_key, "PARALLECT_API_KEY env")
    return RouteDecision("byok", None, "no API key")


def _short_key(api_key: str) -> str:
    """Return only the prefix of an API key (par_live_<hint>)."""
    parts = api_key.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return parts[0] if parts else "***"


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


def research_cmd(
    query: str = typer.Argument(help="Research query"),
    providers: str | None = typer.Option(
        None, "--providers", "-p", help="Comma-separated provider names, or 'all'"
    ),
    tier: str | None = typer.Option(
        None, "--tier",
        help="Tier: nano | lite | normal | deep | max (default: normal)",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", envvar="PARALLECT_API_KEY",
        help="Parallect API key (par_live_...). Routes to SaaS mode.",
    ),
    api_url: str | None = typer.Option(
        None, "--api-url", envvar="PARALLECT_API_URL",
        help="Override Parallect API base URL (SaaS mode).",
    ),
    local: bool = typer.Option(
        False, "--local",
        help="Force BYOK mode even if an API key is present.",
    ),
    synthesize_with: str | None = typer.Option(
        None, "--synthesize-with", "-s",
        help="[BYOK] Model for synthesis (e.g. anthropic, ollama/llama3.2)",
    ),
    no_synthesis: bool = typer.Option(
        False, "--no-synthesis", help="[BYOK] Skip synthesis step"
    ),
    no_sign: bool = typer.Option(False, "--no-sign", help="[BYOK] Skip bundle signing"),
    budget_cap: float | None = typer.Option(
        None, "--budget-cap", help="Maximum cost in USD"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output path for .prx file"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", help="Output directory for .prx file"
    ),
    timeout: float = typer.Option(
        120.0, "--timeout", help="Per-provider timeout (BYOK) / request timeout (SaaS) in seconds"
    ),
    poll_interval: float = typer.Option(
        15.0, "--poll-interval", help="[SaaS] Seconds between status polls"
    ),
    deep: bool = typer.Option(
        False, "--deep",
        help="[DEPRECATED] Alias for --tier deep. Use --tier instead.",
    ),
) -> None:
    """Run multi-provider deep research and produce a .prx bundle."""
    # --deep deprecation shim
    if deep:
        console.print(
            "[yellow]warning: --deep is deprecated; use --tier deep instead.[/yellow]"
        )
        if tier is None:
            tier = "deep"

    tier_cfg = resolve_tier(tier)

    route = decide_route(
        api_key_flag=api_key,
        local_flag=local,
        env_key=os.environ.get("PARALLECT_API_KEY"),
    )

    if route.mode == "saas":
        asyncio.run(_run_saas(
            query=query,
            api_key=route.api_key or "",
            api_url=api_url,
            tier_cfg=tier_cfg,
            providers_str=providers,
            budget_cap=budget_cap,
            output=output,
            output_dir=output_dir,
            timeout=timeout,
            poll_interval=poll_interval,
        ))
    else:
        asyncio.run(_run_byok(
            query=query,
            providers_str=providers,
            tier_cfg=tier_cfg,
            synthesize_with=synthesize_with,
            no_synthesis=no_synthesis,
            budget_cap=budget_cap,
            output=output,
            output_dir=output_dir,
            local=local,
            no_sign=no_sign,
            timeout=timeout,
        ))


# ---------------------------------------------------------------------------
# SaaS path
# ---------------------------------------------------------------------------


async def _run_saas(
    *,
    query: str,
    api_key: str,
    api_url: str | None,
    tier_cfg: TierConfig,
    providers_str: str | None,
    budget_cap: float | None,
    output: str | None,
    output_dir: str | None,
    timeout: float,
    poll_interval: float,
) -> None:
    from parallect.api import (
        InsufficientBalanceError,
        JobFailedError,
        JobTimeoutError,
        ParallectAPIClient,
        ParallectAPIError,
        RateLimitError,
        ServerError,
        UnauthorizedError,
    )
    from parallect.config_mod.settings import ParallectSettings

    settings = ParallectSettings.load()
    base_url = api_url or settings.parallect_api_url

    console.print(
        f"[cyan]→ Running on Parallect API[/cyan] "
        f"(tier: {tier_cfg.name}, account: {_short_key(api_key)})"
    )
    console.print(f"[bold]Query:[/bold] {query}")

    client = ParallectAPIClient(api_key=api_key, api_url=base_url, timeout=timeout)

    provider_list = None
    if providers_str:
        provider_list = [p.strip() for p in providers_str.split(",") if p.strip()]

    try:
        submission = await client.submit_thread(
            message=query, tier=tier_cfg.name, providers=provider_list
        )
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {e}")
        raise typer.Exit(1)
    except InsufficientBalanceError as e:
        console.print(
            f"[red]Insufficient balance:[/red] {e}\n"
            "Top up at https://parallect.ai/billing."
        )
        raise typer.Exit(2)
    except RateLimitError as e:
        console.print(f"[red]Rate limited:[/red] {e}")
        raise typer.Exit(1)
    except ServerError as e:
        console.print(f"[red]Server error:[/red] {e}")
        raise typer.Exit(1)
    except ParallectAPIError as e:
        console.print(f"[red]API error:[/red] {e}")
        raise typer.Exit(1)

    thread = submission.get("thread") or {}
    job = submission.get("job") or {}
    thread_id = thread.get("id") or submission.get("threadId")
    job_id = job.get("id") or submission.get("jobId")
    if not job_id:
        console.print("[red]API did not return a job id.[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]thread: {thread_id}  job: {job_id}[/dim]")

    started = time.monotonic()
    last_status = {"status": None}

    def fmt_status(s) -> str:  # noqa: ANN001
        elapsed = int(time.monotonic() - started)
        mm, ss = divmod(elapsed, 60)
        parts = [f"[{mm}m {ss:02d}s]"]
        if s.pipeline_phase:
            parts.append(f"phase: {s.pipeline_phase}")
        if s.source_count is not None:
            parts.append(f"sources: {s.source_count}")
        if s.synthesis_chars is not None:
            parts.append(f"chars: {s.synthesis_chars:,}")
        parts.append(f"status: {s.status}")
        return " · ".join(parts)

    try:
        with Live(console=console, refresh_per_second=4) as live:
            def cb(s) -> None:  # noqa: ANN001
                last_status["status"] = s
                live.update(fmt_status(s))

            final = await client.poll_until_done(
                job_id, poll_interval=poll_interval, on_update=cb
            )
    except JobFailedError as e:
        console.print(f"[red]Job failed:[/red] {e}")
        raise typer.Exit(1)
    except JobTimeoutError as e:
        console.print(f"[red]Timeout:[/red] {e}")
        raise typer.Exit(1)
    except ParallectAPIError as e:
        console.print(f"[red]API error while polling:[/red] {e}")
        raise typer.Exit(1)

    # Decide output path
    out_path = _resolve_saas_output(output, output_dir, job_id)

    ok, _ = await client.download_bundle(job_id, out_path)
    if not ok:
        console.print(
            "[yellow]warning: /api/v1/jobs/{id}/prx returned 404; "
            "falling back to a minimal bundle from poll data.[/yellow]"
        )
        _write_minimal_bundle(out_path, query=query, job=final)

    elapsed = time.monotonic() - started
    cost_usd = (final.total_cost_cents or 0) / 100 if final.total_cost_cents else None
    console.print(f"[green]Bundle saved:[/green] {out_path}")
    console.print(f"  Duration: {int(elapsed // 60)}m {int(elapsed % 60):02d}s")
    if cost_usd is not None:
        console.print(f"  Cost: ${cost_usd:.4f}")


def _resolve_saas_output(
    output: str | None, output_dir: str | None, job_id: str
) -> Path:
    if output:
        p = Path(output)
        if not p.suffix:
            p = p / f"{job_id}.prx"
        return p
    if output_dir:
        return Path(output_dir) / f"{job_id}.prx"
    return Path.cwd() / f"{job_id}.prx"


def _write_minimal_bundle(path: Path, *, query: str, job) -> None:  # noqa: ANN001
    """Write a minimal .prx bundle when the full download endpoint is unavailable.

    Uses prx-spec's writer to produce a valid zip bundle containing whatever
    synthesis markdown / evidence graph we already have.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from prx_spec import Manifest, write_bundle  # type: ignore

        manifest = Manifest(
            id=job.job_id or "unknown",
            query=query,
            providers_used=[],
            has_synthesis=bool(job.synthesis_markdown),
        )
        write_bundle(
            path,
            manifest=manifest,
            synthesis_md=job.synthesis_markdown or "",
            evidence_graph_json=job.evidence_graph_json or "",
        )
    except Exception:
        # Last-resort: write markdown alongside a .prx suffix; better than nothing.
        path.write_text(job.synthesis_markdown or "", encoding="utf-8")


# ---------------------------------------------------------------------------
# BYOK path (preserves existing behaviour)
# ---------------------------------------------------------------------------


async def _run_byok(
    *,
    query: str,
    providers_str: str | None,
    tier_cfg: TierConfig,
    synthesize_with: str | None,
    no_synthesis: bool,
    budget_cap: float | None,
    output: str | None,
    output_dir: str | None,
    local: bool,
    no_sign: bool,
    timeout: float,
) -> None:
    from parallect.config_mod.settings import ParallectSettings

    settings = ParallectSettings.load()

    # Tier may override providers + budget if user didn't pass them explicitly
    effective_providers_str = providers_str
    if effective_providers_str is None and tier_cfg.providers is not None:
        effective_providers_str = ",".join(tier_cfg.providers)

    effective_budget = budget_cap if budget_cap is not None else tier_cfg.budget_cap_usd

    provider_instances = _resolve_providers(
        effective_providers_str, local, settings, deep=tier_cfg.deep
    )

    if not provider_instances:
        console.print(
            "[red]No providers available. Run 'parallect config' to set up API keys.[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"[cyan]→ Running locally[/cyan] "
        f"(BYOK, {len(provider_instances)} provider"
        f"{'s' if len(provider_instances) != 1 else ''}, tier: {tier_cfg.name})"
    )

    synth_model = synthesize_with or settings.synthesize_with
    if no_synthesis:
        synth_model = None

    out_path = output or output_dir or settings.output_dir

    console.print(f"\n[bold]Researching:[/bold] {query}")
    if tier_cfg.deep:
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
                budget_cap_usd=effective_budget,
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
