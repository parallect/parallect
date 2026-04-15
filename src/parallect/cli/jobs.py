"""`parallect jobs` subcommands — inspect and download SaaS jobs."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import typer

from parallect.api import (
    InsufficientBalanceError,
    ParallectAPIClient,
    RateLimitError,
    ServerError,
    UnauthorizedError,
)

jobs_app = typer.Typer(
    name="jobs",
    help="Inspect and download Parallect hosted research jobs.",
    no_args_is_help=True,
)


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("PARALLECT_API_KEY")
    if not key:
        typer.echo(
            "No Parallect API key found. Pass --api-key or set PARALLECT_API_KEY.",
            err=True,
        )
        raise typer.Exit(code=2)
    return key


def _resolve_url(api_url: str | None) -> str:
    return api_url or os.environ.get("PARALLECT_API_URL") or "https://parallect.ai"


@jobs_app.command("status")
def status_cmd(
    job_id: str = typer.Argument(..., help="Job id (uuid) from a prior research run."),
    api_key: str | None = typer.Option(None, "--api-key", help="Parallect API key (or PARALLECT_API_KEY env)."),
    api_url: str | None = typer.Option(None, "--api-url", help="Override API base URL (or PARALLECT_API_URL env)."),
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON instead of the pretty one-liner."),
) -> None:
    """Fetch the current status of a hosted research job."""
    key = _resolve_key(api_key)
    url = _resolve_url(api_url)

    async def _run() -> dict:
        client = ParallectAPIClient(api_key=key, api_url=url)
        job = await client.get_job(job_id)
        return job.raw if hasattr(job, "raw") else {"status": job.status}

    try:
        payload = asyncio.run(_run())
    except UnauthorizedError:
        typer.echo("Unauthorized — check your API key.", err=True)
        raise typer.Exit(code=1)
    except RateLimitError:
        typer.echo("Rate limited — try again shortly.", err=True)
        raise typer.Exit(code=1)
    except ServerError as e:
        typer.echo(f"Server error: {e}", err=True)
        raise typer.Exit(code=1)
    except httpx.HTTPStatusError as e:
        typer.echo(f"HTTP {e.response.status_code}: {e.response.text[:200]}", err=True)
        raise typer.Exit(code=1)

    # Normalize whether the API returns {job: {...}} or a flat object.
    job = payload.get("job") if isinstance(payload, dict) and "job" in payload else payload

    if as_json:
        typer.echo(json.dumps(job, indent=2, default=str))
        return

    status = job.get("status")
    phase = job.get("pipelinePhase") or "—"
    tier = job.get("budgetTier") or "—"
    mode = job.get("researchMode") or "—"
    synth_mode = job.get("synthesisMode") or "—"
    duration = job.get("durationSeconds")
    raw_cost = (job.get("totalRawProviderCostCents") or 0) / 100
    synth_cost = (job.get("synthesisCostCents") or 0) / 100
    total_cost = raw_cost + synth_cost

    parts = [
        f"status={status}",
        f"phase={phase}",
        f"tier={tier}",
        f"mode={mode}",
        f"synth={synth_mode}",
    ]
    if duration is not None:
        parts.append(f"duration={duration}s")
    parts.append(f"cost=${total_cost:.2f}")
    typer.echo(" ".join(parts))


@jobs_app.command("download")
def download_cmd(
    job_id: str = typer.Argument(..., help="Job id (uuid) to download."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file path. Default: <jobId>.prx in cwd."),
    api_key: str | None = typer.Option(None, "--api-key", help="Parallect API key (or PARALLECT_API_KEY env)."),
    api_url: str | None = typer.Option(None, "--api-url", help="Override API base URL (or PARALLECT_API_URL env)."),
) -> None:
    """Download the signed .prx bundle for a completed job."""
    key = _resolve_key(api_key)
    url = _resolve_url(api_url)
    dest = output or Path(f"{job_id}.prx")

    async def _run() -> int:
        client = ParallectAPIClient(api_key=key, api_url=url, timeout=120.0)
        data = await client.download_bundle(job_id)
        dest.write_bytes(data)
        return len(data)

    try:
        size = asyncio.run(_run())
    except UnauthorizedError:
        typer.echo("Unauthorized — check your API key.", err=True)
        raise typer.Exit(code=1)
    except InsufficientBalanceError as e:
        typer.echo(f"Insufficient balance: {e}", err=True)
        raise typer.Exit(code=1)
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 404:
            typer.echo(
                "Bundle not available (404). Either the job isn't complete, doesn't belong "
                "to this key's project, or the server hasn't deployed the /prx endpoint yet.",
                err=True,
            )
        else:
            typer.echo(f"HTTP {code}: {e.response.text[:200]}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Wrote {dest} ({size:,} bytes)")
