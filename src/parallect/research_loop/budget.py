"""Iteration budget tracking for the research loop.

Tracks cumulative cost across iterations and enforces a hard USD cap.
Separate from the single-pass BudgetEstimator in parallect.orchestrator.budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExhausted(Exception):
    """Raised when cumulative cost exceeds the iteration budget cap."""

    def __init__(self, spent: float, cap: float) -> None:
        super().__init__(f"Budget exhausted: ${spent:.4f} spent of ${cap:.4f} cap")
        self.spent = spent
        self.cap = cap


@dataclass
class IterationBudget:
    """Track cumulative cost across research loop iterations."""

    cap_usd: float
    _spent: float = field(default=0.0, init=False)
    _breakdown: list[dict] = field(default_factory=list, init=False)

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self._spent)

    @property
    def exhausted(self) -> bool:
        return self._spent >= self.cap_usd

    @property
    def breakdown(self) -> list[dict]:
        return list(self._breakdown)

    def record(self, cost_usd: float, label: str = "") -> None:
        """Record a cost and check the cap. Raises BudgetExhausted if exceeded."""
        self._spent += cost_usd
        self._breakdown.append({"cost_usd": cost_usd, "label": label})
        if self._spent > self.cap_usd:
            raise BudgetExhausted(self._spent, self.cap_usd)

    def can_afford(self, estimated_cost: float) -> bool:
        """Check whether the estimated cost fits within the remaining budget."""
        return (self._spent + estimated_cost) <= self.cap_usd
