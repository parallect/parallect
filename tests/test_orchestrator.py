"""Tests for the async orchestrator."""

from __future__ import annotations

import asyncio

import pytest

from parallect.orchestrator.budget import BudgetEstimator, BudgetExceededError
from parallect.orchestrator.parallel import ProviderOutcome, fan_out, research
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


class FailedStatusProvider:
    """Provider whose research() returns a ProviderResult(status='failed').

    This is the shape every built-in provider uses when an upstream API
    call raises (see e.g. PerplexityProvider.research's except branch).
    """

    @property
    def name(self) -> str:
        return "failed_status"

    async def research(self, query: str) -> ProviderResult:
        return ProviderResult(
            provider="failed_status",
            status="failed",
            error="HTTP 401 Unauthorized",
            duration_seconds=0.1,
        )

    def estimate_cost(self, query: str) -> float:
        return 0.01

    def is_available(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_research_surfaces_failed_status_outcome():
    """A provider returning ProviderResult(status='failed') is not silently
    dropped — the on_provider_failure callback fires and the bundle only
    contains successful providers."""
    failures: list[ProviderOutcome] = []

    bundle = await research(
        query="hi",
        providers=[FastProvider(), FailedStatusProvider()],
        synthesize_with=None,
        extract_claims_flag=False,
        no_synthesis=True,
        on_provider_failure=failures.append,
    )

    assert bundle.manifest.providers_used == ["fast"]
    assert [o.provider for o in failures] == ["failed_status"]
    assert failures[0].result is not None
    assert "401" in (failures[0].result.error or "")


@pytest.mark.asyncio
async def test_research_surfaces_exception_failures():
    """A provider that raises is reported via the callback with the
    exception text."""
    failures: list[ProviderOutcome] = []

    bundle = await research(
        query="hi",
        providers=[FastProvider(), FailingProvider()],
        synthesize_with=None,
        extract_claims_flag=False,
        no_synthesis=True,
        on_provider_failure=failures.append,
    )

    assert bundle.manifest.providers_used == ["fast"]
    assert [o.provider for o in failures] == ["failing"]
    assert "exploded" in (failures[0].error or "")


@pytest.mark.asyncio
async def test_research_surfaces_timeout_failures():
    """A provider that times out is reported via the callback with a
    'Timed out' message, and the successful provider still ends up in
    the bundle."""
    failures: list[ProviderOutcome] = []

    bundle = await research(
        query="hi",
        providers=[FastProvider(), SlowProvider()],
        synthesize_with=None,
        extract_claims_flag=False,
        no_synthesis=True,
        timeout_per_provider=0.1,
        on_provider_failure=failures.append,
    )

    assert bundle.manifest.providers_used == ["fast"]
    assert [o.provider for o in failures] == ["slow"]
    assert "Timed out" in (failures[0].error or "")


@pytest.mark.asyncio
async def test_research_logs_failures_when_no_callback(caplog):
    """Even without a callback, failed providers are logged at WARNING
    level via the 'parallect.orchestrator' logger, so tooling that wires
    up logging can surface them."""
    import logging as _logging

    with caplog.at_level(_logging.WARNING, logger="parallect.orchestrator"):
        await research(
            query="hi",
            providers=[FastProvider(), FailingProvider()],
            synthesize_with=None,
            extract_claims_flag=False,
            no_synthesis=True,
        )

    msgs = [r.getMessage() for r in caplog.records]
    assert any("failing" in m and "exploded" in m for m in msgs), msgs


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
