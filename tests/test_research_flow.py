"""Tests for the full research() orchestration flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from parallect.orchestrator.budget import BudgetExceededError
from parallect.orchestrator.parallel import research
from parallect.providers import ProviderResult

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class MockProvider:
    """A fast mock provider that returns canned results."""

    def __init__(self, name: str = "mock", content: str = "# Mock Report\n\nDone."):
        self._name = name
        self._content = content

    @property
    def name(self) -> str:
        return self._name

    async def research(self, query: str) -> ProviderResult:
        return ProviderResult(
            provider=self._name,
            status="completed",
            report_markdown=self._content,
            cost_usd=0.01,
            duration_seconds=0.1,
        )

    def estimate_cost(self, query: str) -> float:
        return 0.01

    def is_available(self) -> bool:
        return True


class FailingMockProvider:
    """A provider that always fails."""

    @property
    def name(self) -> str:
        return "failing"

    async def research(self, query: str) -> ProviderResult:
        raise RuntimeError("Provider crash")

    def estimate_cost(self, query: str) -> float:
        return 0.05

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# research() tests
# ---------------------------------------------------------------------------


class TestResearchFunction:
    @pytest.mark.asyncio
    async def test_single_provider_no_synthesis(self):
        bundle = await research(
            query="test query",
            providers=[MockProvider()],
            no_synthesis=True,
        )
        assert bundle.manifest.query == "test query"
        assert len(bundle.providers) == 1
        assert bundle.providers[0].name == "mock"
        assert bundle.synthesis_md is None
        assert bundle.manifest.has_synthesis is False

    @pytest.mark.asyncio
    async def test_multi_provider(self):
        providers = [
            MockProvider("alpha", "# Alpha Report"),
            MockProvider("beta", "# Beta Report"),
        ]
        bundle = await research(
            query="multi test",
            providers=providers,
            no_synthesis=True,
        )
        assert len(bundle.providers) == 2
        names = {p.name for p in bundle.providers}
        assert names == {"alpha", "beta"}
        assert bundle.manifest.providers_used == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """One provider fails; the other's result is still captured."""
        providers = [MockProvider("good"), FailingMockProvider()]
        bundle = await research(
            query="test",
            providers=providers,
            no_synthesis=True,
        )
        assert len(bundle.providers) == 1
        assert bundle.providers[0].name == "good"

    @pytest.mark.asyncio
    async def test_all_fail_empty_providers(self):
        """When all providers fail, Manifest validation rejects empty providers_used."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await research(
                query="test",
                providers=[FailingMockProvider()],
                no_synthesis=True,
            )

    @pytest.mark.asyncio
    async def test_with_synthesis(self):
        """Verify synthesis is called when not disabled."""
        from parallect.synthesis.llm import SynthesisResult

        mock_synth = AsyncMock(
            return_value=SynthesisResult(
                report_markdown="# Unified Report",
                model="test-model",
            )
        )

        with patch("parallect.synthesis.llm.synthesize", mock_synth):
            bundle = await research(
                query="synth test",
                providers=[MockProvider("p1"), MockProvider("p2")],
                synthesize_with="anthropic",
                no_synthesis=False,
            )

        assert bundle.synthesis_md == "# Unified Report"
        assert bundle.manifest.has_synthesis is True
        mock_synth.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesis_failure_non_fatal(self):
        """If synthesis fails, the bundle should still be created without synthesis."""
        mock_synth = AsyncMock(side_effect=RuntimeError("Synthesis exploded"))

        with patch("parallect.synthesis.llm.synthesize", mock_synth):
            bundle = await research(
                query="test",
                providers=[MockProvider()],
                synthesize_with="anthropic",
                no_synthesis=False,
            )

        assert bundle.synthesis_md is None
        assert bundle.manifest.has_synthesis is False

    @pytest.mark.asyncio
    async def test_budget_cap_enforced(self):
        """Should raise BudgetExceededError when estimate exceeds cap."""
        # MockProvider.estimate_cost returns 0.01 each, so 2 providers = 0.02
        with pytest.raises(BudgetExceededError):
            await research(
                query="test",
                providers=[MockProvider("a"), MockProvider("b")],
                budget_cap_usd=0.001,  # too low
                no_synthesis=True,
            )

    @pytest.mark.asyncio
    async def test_budget_cap_passes(self):
        """Should succeed when estimate is within cap."""
        bundle = await research(
            query="test",
            providers=[MockProvider()],
            budget_cap_usd=1.0,  # plenty of room
            no_synthesis=True,
        )
        assert len(bundle.providers) == 1

    @pytest.mark.asyncio
    async def test_output_path_writes_file(self, tmp_path):
        """When output is given, bundle should be written to disk."""
        out_file = tmp_path / "result.prx"
        await research(
            query="write test",
            providers=[MockProvider()],
            no_synthesis=True,
            output=str(out_file),
        )
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_output_dir_writes_file(self, tmp_path):
        """When output is a directory, bundle should be saved inside it."""
        await research(
            query="dir test",
            providers=[MockProvider()],
            no_synthesis=True,
            output=str(tmp_path),
        )
        prx_files = list(tmp_path.glob("*.prx"))
        assert len(prx_files) == 1

    @pytest.mark.asyncio
    async def test_bundle_metadata(self):
        """Verify manifest fields are correctly populated."""
        bundle = await research(
            query="metadata test",
            providers=[MockProvider("p1")],
            no_synthesis=True,
        )
        assert bundle.manifest.id.startswith("prx_")
        assert bundle.manifest.spec_version == "1.0"
        assert bundle.manifest.producer.name == "parallect-oss"
        assert bundle.manifest.total_cost_usd is not None
        assert "metadata test" in bundle.query_md

    @pytest.mark.asyncio
    async def test_cost_aggregation(self):
        """Cost should be summed across all successful providers."""
        providers = [
            MockProvider("a"),  # cost_usd=0.01
            MockProvider("b"),  # cost_usd=0.01
        ]
        bundle = await research(
            query="cost test",
            providers=providers,
            no_synthesis=True,
        )
        assert bundle.manifest.total_cost_usd == pytest.approx(0.02, abs=0.001)
