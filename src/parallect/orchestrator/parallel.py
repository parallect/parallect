"""Parallel fan-out orchestrator."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from prx_spec import BundleData, ProviderData, write_bundle
from prx_spec.models import Manifest, Producer

from parallect.providers import ProviderResult
from parallect.providers.base import AsyncResearchProvider


@dataclass
class ProviderOutcome:
    """Outcome of a single provider call (success or failure)."""

    provider: str
    result: ProviderResult | None = None
    error: str | None = None


async def fan_out(
    query: str,
    providers: list[AsyncResearchProvider],
    timeout_per_provider: float = 120.0,
) -> list[ProviderOutcome]:
    """Fan out query to all providers in parallel.

    Partial failures are captured, not raised. Each provider gets its own
    timeout. One provider failing does not cancel the others.
    """

    async def run_one(provider: AsyncResearchProvider) -> ProviderOutcome:
        try:
            async with asyncio.timeout(timeout_per_provider):
                result = await provider.research(query)
                return ProviderOutcome(provider=provider.name, result=result)
        except TimeoutError:
            return ProviderOutcome(
                provider=provider.name,
                error=f"Timed out after {timeout_per_provider}s",
            )
        except Exception as e:
            return ProviderOutcome(provider=provider.name, error=str(e))

    outcomes = await asyncio.gather(
        *[run_one(p) for p in providers],
        return_exceptions=False,
    )
    return list(outcomes)


async def research(
    query: str,
    providers: list[AsyncResearchProvider],
    synthesize_with: str | None = "anthropic",
    extract_claims_flag: bool = True,
    budget_cap_usd: float | None = None,
    timeout_per_provider: float = 120.0,
    output: str | Path | None = None,
    no_synthesis: bool = False,
    parent_bundle_id: str | None = None,
    parent_context: str | None = None,
) -> BundleData:
    """High-level research: fan out, collect, optionally synthesize, write .prx.

    This is the primary entry point for both the CLI and programmatic use.
    """
    from parallect.orchestrator.budget import BudgetEstimator

    # Budget check
    if budget_cap_usd is not None:
        estimator = BudgetEstimator()
        estimate = estimator.estimate(query, providers)
        estimator.check_cap(estimate, budget_cap_usd)

    # Fan out to all providers (augment query with parent context if continuing)
    effective_query = query
    if parent_context:
        effective_query = (
            f"{query}\n\n---\n"
            f"Context from previous research:\n{parent_context}"
        )
    outcomes = await fan_out(effective_query, providers, timeout_per_provider)

    # Build provider data from successful results
    provider_data: list[ProviderData] = []
    providers_used: list[str] = []
    total_cost = 0.0
    total_duration = 0.0

    for outcome in outcomes:
        if outcome.result and outcome.result.status != "failed":
            pd = ProviderData(name=outcome.provider, report_md=outcome.result.report_markdown)
            provider_data.append(pd)
            providers_used.append(outcome.provider)
            if outcome.result.cost_usd:
                total_cost += outcome.result.cost_usd
            if outcome.result.duration_seconds:
                total_duration = max(total_duration, outcome.result.duration_seconds)

    # Build bundle
    bundle_id = f"prx_{secrets.token_hex(4)}"
    now = datetime.now(timezone.utc).isoformat()

    has_synthesis = False
    synthesis_md = None

    # Synthesize if requested and we have results
    if not no_synthesis and synthesize_with and provider_data:
        try:
            from parallect.synthesis.llm import synthesize

            results = [
                o.result for o in outcomes if o.result and o.result.status != "failed"
            ]
            synth_result = await synthesize(query, results, model=synthesize_with)
            synthesis_md = synth_result.report_markdown
            has_synthesis = True
            if synth_result.cost_usd:
                total_cost += synth_result.cost_usd
        except Exception:
            # Synthesis failure is non-fatal
            pass

    manifest = Manifest(
        spec_version="1.0",
        id=bundle_id,
        query=query,
        created_at=now,
        producer=Producer(name="parallect-oss", version="0.1.0"),
        providers_used=providers_used,
        has_synthesis=has_synthesis,
        has_claims=False,
        has_sources=False,
        has_evidence_graph=False,
        has_follow_ons=False,
        total_cost_usd=round(total_cost, 4) if total_cost else None,
        total_duration_seconds=round(total_duration, 2) if total_duration else None,
        parent_bundle_id=parent_bundle_id,
    )

    bundle = BundleData(
        manifest=manifest,
        query_md=f"# Research Query\n\n{query}",
        providers=provider_data,
        synthesis_md=synthesis_md,
    )

    # Write to disk if output path given
    if output:
        output_path = Path(output)
        if output_path.is_dir():
            output_path = output_path / f"{bundle_id}.prx"
        write_bundle(bundle, output_path)

    return bundle
