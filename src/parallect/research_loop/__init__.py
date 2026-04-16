"""Agentic iterative research loop.

Planner -> Executor -> Evaluator -> Synthesizer, with configurable
iteration budget and stop conditions.
"""

from parallect.research_loop.budget import IterationBudget
from parallect.research_loop.loop import ResearchLoopResult, run_loop

__all__ = ["IterationBudget", "ResearchLoopResult", "run_loop"]
