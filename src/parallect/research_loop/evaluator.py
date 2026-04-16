"""Evaluator: decide whether to continue iterating or stop."""

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
from parallect.research_loop.prompts import EVALUATOR_SYSTEM, EVALUATOR_USER

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorResult:
    """Output of an evaluator call."""

    decision: str  # "stop" or "continue"
    rationale: str = ""
    gaps: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens: dict | None = None


async def evaluate(
    query: str,
    results_summary: str,
    spec: BackendSpec,
    *,
    iteration: int,
    max_iterations: int,
    budget_remaining: float,
    budget_total: float,
    timeout: float = 60.0,
) -> EvaluatorResult:
    """Call the LLM to decide whether to continue or stop.

    On any failure (LLM error, parse error), returns stop (conservative).
    """
    prompt = EVALUATOR_USER.format(
        query=query,
        iteration=iteration,
        max_iterations=max_iterations,
        results_summary=results_summary,
        budget_remaining=budget_remaining,
        budget_total=budget_total,
    )

    try:
        raw = await _dispatch(spec, prompt, timeout=timeout)
    except Exception:
        logger.warning("Evaluator LLM call failed; defaulting to stop", exc_info=True)
        return EvaluatorResult(decision="stop", rationale="evaluator call failed")

    content = raw.get("content", "")
    tokens = raw.get("tokens")
    cost = _estimate_evaluator_cost(spec, tokens)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Evaluator returned invalid JSON; defaulting to stop")
        return EvaluatorResult(
            decision="stop",
            rationale="evaluator returned invalid JSON",
            cost_usd=cost,
            tokens=tokens,
        )

    decision = parsed.get("decision", "stop")
    if decision not in ("stop", "continue"):
        decision = "stop"

    return EvaluatorResult(
        decision=decision,
        rationale=parsed.get("rationale", ""),
        gaps=parsed.get("gaps", []),
        suggested_queries=parsed.get("suggested_queries", []),
        cost_usd=cost,
        tokens=tokens,
    )


async def _dispatch(spec: BackendSpec, prompt: str, *, timeout: float) -> dict:
    if spec.kind == "anthropic":
        return await call_anthropic_chat(spec, prompt, EVALUATOR_SYSTEM, timeout=timeout)
    if spec.kind == "gemini":
        return await call_gemini_chat(spec, prompt, EVALUATOR_SYSTEM, timeout=timeout)
    if spec.kind in OPENAI_COMPAT_BACKENDS:
        return await call_openai_compat_chat(spec, prompt, EVALUATOR_SYSTEM, timeout=timeout)
    raise ValueError(f"Unsupported backend for evaluator: {spec.kind}")


def _estimate_evaluator_cost(spec: BackendSpec, tokens: dict | None) -> float:
    if not tokens:
        return 0.0
    if spec.kind == "anthropic":
        return 0.005
    if spec.kind == "gemini":
        return 0.002
    return 0.003
