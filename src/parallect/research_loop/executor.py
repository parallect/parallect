"""Executor: fan out sub-queries to providers and plugins in parallel."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from parallect.orchestrator.parallel import ProviderOutcome, fan_out
from parallect.orchestrator.plugin_sources import PluginFanOutResult, run_plugin_sources
from parallect.providers import ProviderResult
from parallect.providers.base import AsyncResearchProvider
from parallect.research_loop.planner import SubQuery

logger = logging.getLogger(__name__)


@dataclass
class ExecutorResult:
    """Aggregated results from executing one iteration's sub-queries."""

    provider_outcomes: list[ProviderOutcome] = field(default_factory=list)
    plugin_outcomes: list[PluginFanOutResult] = field(default_factory=list)
    cost_usd: float = 0.0
    duration_seconds: float = 0.0

    @property
    def all_results(self) -> list[ProviderResult]:
        results = []
        for o in self.provider_outcomes:
            if o.result and o.result.status != "failed":
                results.append(o.result)
        for o in self.plugin_outcomes:
            if o.result and o.result.status != "failed":
                results.append(o.result)
        return results

    @property
    def results_summary(self) -> str:
        lines = []
        for r in self.all_results:
            preview = r.report_markdown[:300] if r.report_markdown else "(empty)"
            lines.append(f"### {r.provider}\n{preview}\n")
        return "\n".join(lines) if lines else "(no results)"


async def execute(
    sub_queries: list[SubQuery],
    providers: list[AsyncResearchProvider],
    *,
    sources_raw: str | None = None,
    settings: object | None = None,
    timeout: float = 120.0,
) -> ExecutorResult:
    """Execute all sub-queries against providers and plugins in parallel.

    Each sub-query is sent to all providers. Plugin sources are queried
    once with the combined sub-query text for relevance.
    """
    start = time.monotonic()

    combined_query = "\n\n".join(sq.query for sq in sub_queries)

    provider_task = asyncio.create_task(
        fan_out(combined_query, providers, timeout)
    )
    plugin_task = asyncio.create_task(
        run_plugin_sources(combined_query, sources_raw, settings=settings, timeout=timeout)
    )

    provider_outcomes = await provider_task
    plugin_outcomes = await plugin_task

    total_cost = 0.0
    for o in provider_outcomes:
        if o.result and o.result.cost_usd:
            total_cost += o.result.cost_usd
    for o in plugin_outcomes:
        if o.result and o.result.cost_usd:
            total_cost += o.result.cost_usd

    elapsed = time.monotonic() - start

    return ExecutorResult(
        provider_outcomes=provider_outcomes,
        plugin_outcomes=plugin_outcomes,
        cost_usd=total_cost,
        duration_seconds=round(elapsed, 3),
    )
