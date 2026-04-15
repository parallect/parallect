"""Typer CLI application for parallect — the research engine."""

from __future__ import annotations

import typer

from parallect.cli.config import config_cmd
from parallect.cli.continue_ import continue_cmd
from parallect.cli.enhance import enhance_cmd
from parallect.cli.research import research_cmd

app = typer.Typer(
    name="parallect",
    help="Multi-provider AI deep research from the command line.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


AGENT_HELP = """\
# parallect — agent reference

`parallect` is a multi-provider deep research CLI. It fans a query out to
several research providers in parallel and emits a signed, verifiable .prx
bundle (zip) containing per-provider reports, a synthesis, an evidence graph,
and source attestations. It runs in two modes:

  - BYOK (default): you supply provider API keys; everything runs locally.
  - SaaS: you supply a Parallect API key and the hosted service does the work.

## Routing rules

  - `--api-key par_live_...` OR `PARALLECT_API_KEY` env set -> SaaS mode
  - `--local` -> always BYOK (overrides the env key)
  - Neither -> BYOK mode

## Common commands

Run normal-tier research, save to a directory:
    parallect research "query" --tier normal --output-dir ./research

Run max-tier research against the hosted API, write to a specific file
(requires PARALLECT_API_KEY in env or --api-key):
    parallect research "query" --tier max --output out.prx

Force local BYOK with specific providers:
    parallect research "query" --local --providers perplexity,gemini

Continue from a prior bundle with a follow-on question:
    parallect continue out.prx "follow-up"

Enhance an existing bundle through the hosted API:
    parallect enhance out.prx --api-key par_live_...

## `parallect research` flags

  --tier nano|lite|normal|deep|max   (default: normal)
  --providers p1,p2,...              (or 'all')
  --api-key par_live_...             (routes to SaaS; env: PARALLECT_API_KEY)
  --api-url URL                      (override; env: PARALLECT_API_URL)
  --local                            (force BYOK)
  --output PATH / --output-dir DIR
  --budget-cap USD
  --timeout SECONDS
  --poll-interval SECONDS            (SaaS only)
  --synthesize-with MODEL            (BYOK only)
  --no-synthesis                     (BYOK only)
  --no-sign                          (BYOK only)

## Environment variables

  PARALLECT_API_KEY   - hosted API key (par_live_...), routes to SaaS
  PARALLECT_API_URL   - override API base (default: https://api.parallect.ai)
  PERPLEXITY_API_KEY  - BYOK provider key
  GOOGLE_API_KEY      - BYOK Gemini key
  OPENAI_API_KEY      - BYOK OpenAI key
  XAI_API_KEY         - BYOK Grok key
  ANTHROPIC_API_KEY   - BYOK Anthropic key

## Output format

Every command produces a `.prx` bundle. It is a zip with:
  - manifest.json         job metadata, providers used, costs
  - synthesis.md          cross-provider synthesis
  - evidence-graph.json   claims + sources + confidence
  - providers/*.json      raw per-provider reports
  - attestations/*.jws    optional Ed25519 signatures

Read bundles with the `prx` CLI, or with python: `prx_spec.read_bundle(path)`.

## Common failures and exit codes

  1    generic error (bad tier, no providers available, server error, timeout)
  2    insufficient account balance (SaaS only)

SaaS-mode error signals:
  401  bad or missing API key
  402  insufficient balance (top up at https://parallect.ai/billing)
  429  rate limited — retry with backoff
  5xx  server error
  timeout  polling exceeded --timeout

## Checking status / resuming

SaaS jobs are durable. The CLI prints the job id on submission; if the CLI
disconnects you can poll manually:
    curl -H "Authorization: Bearer $PARALLECT_API_KEY" \\
         https://api.parallect.ai/api/v1/jobs/<job_id>
and download the bundle once `status == completed`:
    curl -H "Authorization: Bearer $PARALLECT_API_KEY" \\
         https://api.parallect.ai/api/v1/jobs/<job_id>/prx -o out.prx
"""


def _agent_help_callback(value: bool) -> None:
    if value:
        typer.echo(AGENT_HELP)
        raise typer.Exit()


@app.callback()
def _root(
    agent_help: bool = typer.Option(
        False,
        "--agent-help",
        hidden=True,
        is_eager=True,
        callback=_agent_help_callback,
        help="Print an LLM-friendly reference for agents.",
    ),
) -> None:
    """Multi-provider AI deep research from the command line."""


app.command("research")(research_cmd)
app.command("continue")(continue_cmd)
app.command("enhance")(enhance_cmd)
app.command("config")(config_cmd)


if __name__ == "__main__":
    app()
