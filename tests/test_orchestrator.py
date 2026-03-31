"""Tests for the async orchestrator."""

from __future__ import annotations

import asyncio

import pytest

from parallect.orchestrator.budget import BudgetEstimator, BudgetExceededError
from parallect.orchestrator.parallel import fan_out
from parallect.providers import ProviderResult


class SlowProvider:
    @property
    def name(self) -> str:
        return "slow"

    async def research(self, query: str) -> ProviderResult:
        await asyncio.sleep(10)
        return ProviderResult(provider="slow", status="completed", report_markdown="slow")

    def estimate_cost(self, query: str) -> float:
        return 0.0

    def is_available(self) -> bool:
        return True


class FastProvider:
    @property
    def name(self) -> str:
        return "fast"

    async def research(self, query: str) -> ProviderResult:
        return ProviderResult(
            provider="fast",
            status="completed",
            report_markdown="# Fast Report\n\nThis is fast.",
            cost_usd=0.01,
            duration_seconds=0.1,
        )

    def estimate_cost(self, query: str) -> float:
        return 0.01

    def is_available(self) -> bool:
        return True


class FailingProvider:
    @property
    def name(self) -> str:
        return "failing"

    async def research(self, query: str) -> ProviderResult:
        raise RuntimeError("Provider exploded")

    def estimate_cost(self, query: str) -> float:
        return 0.05

    def is_available(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_fan_out_single_provider():
    outcomes = await fan_out("test query", [FastProvider()])
    assert len(outcomes) == 1
    assert outcomes[0].provider == "fast"
    assert outcomes[0].result is not None
    assert outcomes[0].result.status == "completed"
    assert outcomes[0].error is None


@pytest.mark.asyncio
async def test_fan_out_multiple_providers():
    outcomes = await fan_out("test", [FastProvider(), FastProvider()])
    assert len(outcomes) == 2
    assert all(o.result is not None for o in outcomes)


@pytest.mark.asyncio
async def test_fan_out_captures_failures():
    outcomes = await fan_out("test", [FastProvider(), FailingProvider()])
    assert len(outcomes) == 2

    fast = next(o for o in outcomes if o.provider == "fast")
    failing = next(o for o in outcomes if o.provider == "failing")

    assert fast.result is not None
    assert failing.error is not None
    assert "exploded" in failing.error


@pytest.mark.asyncio
async def test_fan_out_timeout():
    outcomes = await fan_out("test", [SlowProvider()], timeout_per_provider=0.1)
    assert len(outcomes) == 1
    assert outcomes[0].error is not None
    assert "Timed out" in outcomes[0].error


@pytest.mark.asyncio
async def test_fan_out_partial_timeout():
    """One provider times out, the other succeeds."""
    outcomes = await fan_out(
        "test", [FastProvider(), SlowProvider()], timeout_per_provider=0.5
    )
    assert len(outcomes) == 2

    fast = next(o for o in outcomes if o.provider == "fast")
    slow = next(o for o in outcomes if o.provider == "slow")

    assert fast.result is not None
    assert slow.error is not None


class TestBudgetEstimator:
    def test_estimate(self):
        estimator = BudgetEstimator()
        providers = [FastProvider(), FailingProvider()]
        estimate = estimator.estimate("test", providers)
        assert estimate == pytest.approx(0.06)

    def test_check_cap_passes(self):
        estimator = BudgetEstimator()
        estimator.check_cap(0.05, 1.00)  # Should not raise

    def test_check_cap_fails(self):
        estimator = BudgetEstimator()
        with pytest.raises(BudgetExceededError, match="exceeds"):
            estimator.check_cap(2.00, 1.00)
