"""Planner: decompose a research query into sub-queries via LLM."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from parallect.backends import BackendSpec, OPENAI_COMPAT_BACKENDS
from parallect.backends.adapters import (
    call_anthropic_chat,
    call_gemini_chat,
    call_openai_compat_chat,
)
from parallect.research_loop.prompts import PLANNER_SYSTEM, PLANNER_USER

logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    """A single sub-query produced by the planner."""

    query: str
    target_sources: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class PlannerResult:
    """Output of a planner call."""

    sub_queries: list[SubQuery]
    cost_usd: float = 0.0
    tokens: dict | None = None


def _format_prior_iterations(prior: list[dict]) -> str:
    if not prior:
        return "None (first iteration)"
    lines = []
    for i, it in enumerate(prior, 1):
        lines.append(f"## Iteration {i}")
        for sq in it.get("sub_queries", []):
            lines.append(f"- {sq}")
        if it.get("evaluator_rationale"):
            lines.append(f"Evaluator: {it['evaluator_rationale']}")
        lines.append("")
    return "\n".join(lines)


async def plan(
    query: str,
    sources: list[str],
    spec: BackendSpec,
    *,
    prior_iterations: list[dict] | None = None,
    timeout: float = 60.0,
) -> PlannerResult:
    """Call the LLM to produce sub-queries for one iteration.

    On failure (parse error, empty response), returns a single-query
    fallback so the loop degrades gracefully rather than crashing.
    """
    prompt = PLANNER_USER.format(
        query=query,
        sources=", ".join(sources),
        prior_iterations=_format_prior_iterations(prior_iterations or []),
    )

    try:
        raw = await _dispatch(spec, prompt, timeout=timeout)
    except Exception:
        logger.warning("Planner LLM call failed; falling back to single query", exc_info=True)
        return PlannerResult(sub_queries=[SubQuery(query=query)])

    content = raw.get("content", "")
    tokens = raw.get("tokens")
    cost = _estimate_planner_cost(spec, tokens)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Planner returned invalid JSON; falling back to single query")
        return PlannerResult(sub_queries=[SubQuery(query=query)], cost_usd=cost, tokens=tokens)

    sub_queries_raw = parsed.get("sub_queries", [])
    if not sub_queries_raw:
        return PlannerResult(sub_queries=[SubQuery(query=query)], cost_usd=cost, tokens=tokens)

    sub_queries = [
        SubQuery(
            query=sq.get("query", query),
            target_sources=sq.get("target_sources", []),
            rationale=sq.get("rationale", ""),
        )
        for sq in sub_queries_raw
    ]

    # Clamp to 2-5
    if len(sub_queries) < 2:
        sub_queries.append(SubQuery(query=query))
    sub_queries = sub_queries[:5]

    return PlannerResult(sub_queries=sub_queries, cost_usd=cost, tokens=tokens)


async def _dispatch(spec: BackendSpec, prompt: str, *, timeout: float) -> dict:
    if spec.kind == "anthropic":
        return await call_anthropic_chat(spec, prompt, PLANNER_SYSTEM, timeout=timeout)
    if spec.kind == "gemini":
        return await call_gemini_chat(spec, prompt, PLANNER_SYSTEM, timeout=timeout)
    if spec.kind in OPENAI_COMPAT_BACKENDS:
        return await call_openai_compat_chat(spec, prompt, PLANNER_SYSTEM, timeout=timeout)
    raise ValueError(f"Unsupported backend for planner: {spec.kind}")


def _estimate_planner_cost(spec: BackendSpec, tokens: dict | None) -> float:
    if not tokens:
        return 0.0
    if spec.kind == "anthropic":
        return 0.005
    if spec.kind == "gemini":
        return 0.002
    return 0.003
