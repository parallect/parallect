"""Orchestrator: planner -> executor -> evaluator -> loop -> synthesizer.

Returns a ResearchLoopResult containing the final synthesis and full
provenance for each iteration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from parallect.backends import resolve_synthesis_backend
from parallect.providers import ProviderResult
from parallect.providers.base import AsyncResearchProvider
from parallect.research_loop.budget import BudgetExhausted, IterationBudget
from parallect.research_loop.evaluator import EvaluatorResult, evaluate
from parallect.research_loop.executor import ExecutorResult, execute
from parallect.research_loop.planner import PlannerResult, SubQuery, plan
from parallect.research_loop.synthesizer import synthesize_iterations
from parallect.synthesis.llm import SynthesisResult

logger = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    """Provenance for a single iteration."""

    iteration: int
    sub_queries: list[str]
    results_count: int
    evaluator_decision: str
    evaluator_rationale: str
    evaluator_gaps: list[str]
    cost_usd: float
    duration_seconds: float


@dataclass
class ResearchLoopResult:
    """Final output of the agentic research loop."""

    query: str
    iterations: list[IterationRecord]
    all_results: list[ProviderResult]
    synthesis: SynthesisResult | None = None
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    stop_reason: str = ""

    def provenance_dict(self) -> dict:
        """Serializable provenance for loop.json in the .prx bundle."""
        return {
            "query": self.query,
            "total_iterations": len(self.iterations),
            "stop_reason": self.stop_reason,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "iterations": [
                {
                    "iteration": it.iteration,
                    "sub_queries": it.sub_queries,
                    "results_count": it.results_count,
                    "evaluator_decision": it.evaluator_decision,
                    "evaluator_rationale": it.evaluator_rationale,
                    "evaluator_gaps": it.evaluator_gaps,
                    "cost_usd": round(it.cost_usd, 4),
                    "duration_seconds": round(it.duration_seconds, 2),
                }
                for it in self.iterations
            ],
        }


def _resolve_loop_backend(
    model: str | None,
    settings: object | None,
):
    """Resolve the backend spec for planner/evaluator LLM calls."""
    return resolve_synthesis_backend(
        cli_model=model,
        settings=settings,
    )


async def run_loop(
    query: str,
    providers: list[AsyncResearchProvider],
    *,
    max_iterations: int = 3,
    budget_cap_usd: float = 5.0,
    planner_model: str | None = None,
    evaluator_model: str | None = None,
    synthesize_with: str | None = "anthropic",
    synthesis_base_url: str | None = None,
    sources: str | None = None,
    settings: object | None = None,
    timeout: float = 900.0,
    on_status: Callable[[str], None] | None = None,
) -> ResearchLoopResult:
    """Run the iterative research loop.

    Phases per iteration:
      1. Planner decomposes query into sub-queries
      2. Executor fans out to providers + plugins
      3. Evaluator decides stop/continue
      4. If continue, loop with accumulated context

    After the loop, the synthesizer produces a final report over all
    accumulated results.

    Stop conditions: max iterations, evaluator says stop, budget cap,
    or no new results in an iteration.
    """
    loop_start = time.monotonic()
    budget = IterationBudget(cap_usd=budget_cap_usd)

    planner_spec = _resolve_loop_backend(planner_model, settings)
    evaluator_spec = _resolve_loop_backend(evaluator_model, settings)

    source_names = [p.name for p in providers]
    if sources:
        source_names.extend(s.strip() for s in sources.split(",") if s.strip())

    all_results: list[ProviderResult] = []
    iterations: list[IterationRecord] = []
    prior_iteration_ctx: list[dict] = []
    stop_reason = ""

    def _emit(msg: str) -> None:
        if on_status:
            on_status(msg)

    for i in range(1, max_iterations + 1):
        it_start = time.monotonic()
        _emit(f"[it {i}/{max_iterations}] Planning...")

        # Plan
        try:
            planner_result: PlannerResult = await plan(
                query,
                source_names,
                planner_spec,
                prior_iterations=prior_iteration_ctx,
                timeout=timeout,
            )
        except Exception:
            logger.warning("Planner failed on iteration %d; using original query", i)
            planner_result = PlannerResult(sub_queries=[SubQuery(query=query)])

        budget.record(planner_result.cost_usd, f"planner-it{i}")

        sq_texts = [sq.query for sq in planner_result.sub_queries]
        _emit(
            f"[it {i}/{max_iterations}] Planner -> {len(sq_texts)} sub-queries: "
            + "; ".join(sq_texts[:3])
        )

        # Execute
        _emit(f"[it {i}/{max_iterations}] Executing...")
        try:
            exec_result: ExecutorResult = await execute(
                planner_result.sub_queries,
                providers,
                sources_raw=sources,
                settings=settings,
                timeout=timeout,
            )
        except Exception:
            logger.warning("Executor failed on iteration %d", i, exc_info=True)
            stop_reason = "executor_failure"
            break

        try:
            budget.record(exec_result.cost_usd, f"executor-it{i}")
        except BudgetExhausted:
            stop_reason = "budget_exhausted"
            _emit(f"[stop] Budget exhausted after iteration {i}")
            break

        new_results = exec_result.all_results
        if not new_results:
            stop_reason = "no_new_results"
            _emit(f"[stop] No new results in iteration {i}")
            break

        all_results.extend(new_results)

        provider_durations = []
        for o in exec_result.provider_outcomes:
            name = o.provider
            dur = o.result.duration_seconds if o.result and o.result.duration_seconds else 0
            provider_durations.append(f"{name} ({dur:.1f}s)")
        for o in exec_result.plugin_outcomes:
            name = o.spec.display
            dur = o.result.duration_seconds if o.result and o.result.duration_seconds else 0
            provider_durations.append(f"{name} ({dur:.1f}s)")
        _emit(f"[it {i}/{max_iterations}] Executor -> {', '.join(provider_durations)}")

        # Evaluate (skip on last iteration)
        if i == max_iterations:
            eval_result = EvaluatorResult(decision="stop", rationale="max iterations reached")
        else:
            _emit(f"[it {i}/{max_iterations}] Evaluating...")
            eval_result = await evaluate(
                query,
                exec_result.results_summary,
                evaluator_spec,
                iteration=i,
                max_iterations=max_iterations,
                budget_remaining=budget.remaining,
                budget_total=budget.cap_usd,
                timeout=timeout,
            )
            try:
                budget.record(eval_result.cost_usd, f"evaluator-it{i}")
            except BudgetExhausted:
                eval_result = EvaluatorResult(
                    decision="stop", rationale="budget exhausted after evaluation"
                )

        it_elapsed = time.monotonic() - it_start
        it_cost = planner_result.cost_usd + exec_result.cost_usd + eval_result.cost_usd

        iterations.append(IterationRecord(
            iteration=i,
            sub_queries=sq_texts,
            results_count=len(new_results),
            evaluator_decision=eval_result.decision,
            evaluator_rationale=eval_result.rationale,
            evaluator_gaps=eval_result.gaps,
            cost_usd=it_cost,
            duration_seconds=round(it_elapsed, 3),
        ))

        prior_iteration_ctx.append({
            "sub_queries": sq_texts,
            "evaluator_rationale": eval_result.rationale,
            "gaps": eval_result.gaps,
        })

        _emit(
            f"[it {i}/{max_iterations}] Evaluator -> {eval_result.decision}"
            + (f" (gap: {', '.join(eval_result.gaps[:2])})" if eval_result.gaps else "")
        )

        if eval_result.decision == "stop":
            if i == max_iterations:
                stop_reason = stop_reason or "max_iterations"
            else:
                stop_reason = stop_reason or "evaluator_stop"
            break

        if budget.exhausted:
            stop_reason = "budget_exhausted"
            _emit("[stop] Budget exhausted")
            break

    if not stop_reason:
        stop_reason = "max_iterations"

    # Synthesize
    synthesis: SynthesisResult | None = None
    if all_results and synthesize_with:
        _emit(f"[stop] Synthesizing across {len(iterations)} iteration(s)...")
        try:
            synthesis = await synthesize_iterations(
                query,
                all_results,
                model=synthesize_with,
                base_url=synthesis_base_url,
                settings=settings,
            )
            if synthesis.cost_usd:
                budget.record(synthesis.cost_usd, "synthesis")
        except BudgetExhausted:
            pass
        except Exception:
            logger.warning("Final synthesis failed", exc_info=True)

    total_elapsed = time.monotonic() - loop_start

    return ResearchLoopResult(
        query=query,
        iterations=iterations,
        all_results=all_results,
        synthesis=synthesis,
        total_cost_usd=budget.spent,
        total_duration_seconds=round(total_elapsed, 3),
        stop_reason=stop_reason,
    )
