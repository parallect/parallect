"""Per-provider citation deduplication in the parallel orchestrator."""

from __future__ import annotations

import pytest

from parallect.orchestrator.parallel import research
from parallect.providers import ProviderResult


class CitingMockProvider:
    """Mock provider that returns a caller-supplied citations list."""

    def __init__(self, name: str, citations: list[dict]):
        self._name = name
        self._citations = citations

    @property
    def name(self) -> str:
        return self._name

    async def research(self, query: str) -> ProviderResult:
        return ProviderResult(
            provider=self._name,
            status="completed",
            report_markdown="# Mock Report\n\nBody.",
            citations=list(self._citations),
            cost_usd=0.01,
            duration_seconds=0.1,
        )

    def estimate_cost(self, query: str) -> float:
        return 0.01

    def is_available(self) -> bool:
        return True


class TestProviderCitationDedup:
    @pytest.mark.asyncio
    async def test_duplicate_urls_collapsed_with_contiguous_index(self):
        """Two citations pointing at the same URL collapse to one, index 1."""
        citations = [
            {"url": "https://Example.com/Page/", "title": "First", "snippet": "short"},
            {"url": "https://example.com/Page", "title": "Second", "snippet": "longer snippet with more context"},
            {"url": "https://other.com/a", "title": "Other", "snippet": "x"},
        ]
        provider = CitingMockProvider("mock", citations)

        bundle = await research(
            query="dedupe test",
            providers=[provider],
            no_synthesis=True,
        )

        assert len(bundle.providers) == 1
        out = bundle.providers[0].citations
        assert out is not None
        # The two example.com variants collapse; other.com stays.
        assert len(out) == 2
        # Indices are contiguous starting at 1.
        assert [c.index for c in out] == [1, 2]
        # First occurrence's title is kept.
        example = next(c for c in out if "example.com" in c.url.lower())
        assert example.title == "First"
        # Later, strictly-longer snippet is merged in.
        assert example.snippet == "longer snippet with more context"

    @pytest.mark.asyncio
    async def test_shorter_later_snippet_is_ignored(self):
        citations = [
            {"url": "https://example.com/doc", "title": "A", "snippet": "this is the long original snippet"},
            {"url": "https://example.com/doc/", "title": "B", "snippet": "short"},
        ]
        provider = CitingMockProvider("mock", citations)

        bundle = await research(
            query="dedupe test",
            providers=[provider],
            no_synthesis=True,
        )
        out = bundle.providers[0].citations
        assert out is not None
        assert len(out) == 1
        assert out[0].snippet == "this is the long original snippet"
        assert out[0].title == "A"

    @pytest.mark.asyncio
    async def test_empty_urls_skipped_and_index_stays_contiguous(self):
        citations = [
            {"url": "", "title": "Skip me"},
            {"url": "https://a.com/x", "title": "A"},
            {"url": "https://b.com/y", "title": "B"},
        ]
        provider = CitingMockProvider("mock", citations)

        bundle = await research(
            query="dedupe test",
            providers=[provider],
            no_synthesis=True,
        )
        out = bundle.providers[0].citations
        assert out is not None
        assert [c.index for c in out] == [1, 2]
        assert {c.url for c in out} == {"https://a.com/x", "https://b.com/y"}

    @pytest.mark.asyncio
    async def test_per_provider_scope(self):
        """Dedup is per-provider — cross-provider URL overlap is not collapsed here."""
        p1 = CitingMockProvider(
            "p1",
            [
                {"url": "https://same.com/a", "title": "P1"},
                {"url": "https://same.com/a", "title": "P1 dup"},
            ],
        )
        p2 = CitingMockProvider(
            "p2",
            [{"url": "https://same.com/a", "title": "P2"}],
        )

        bundle = await research(
            query="dedupe test",
            providers=[p1, p2],
            no_synthesis=True,
        )

        by_name = {p.name: p for p in bundle.providers}
        assert len(by_name["p1"].citations or []) == 1
        assert len(by_name["p2"].citations or []) == 1
        assert (by_name["p1"].citations or [])[0].index == 1
        assert (by_name["p2"].citations or [])[0].index == 1
