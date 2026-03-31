"""Budget estimation and cap enforcement."""

from __future__ import annotations

from parallect.providers.base import AsyncResearchProvider


class BudgetExceededError(Exception):
    """Raised when estimated cost exceeds budget cap."""


class BudgetEstimator:
    """Estimate total cost before running and enforce caps."""

    def estimate(self, query: str, providers: list[AsyncResearchProvider]) -> float:
        """Estimate total cost across all providers."""
        return sum(p.estimate_cost(query) for p in providers)

    def check_cap(self, estimate: float, cap: float) -> None:
        """Raise if estimate exceeds cap."""
        if estimate > cap:
            raise BudgetExceededError(
                f"Estimated ${estimate:.2f} exceeds budget cap ${cap:.2f}"
            )
