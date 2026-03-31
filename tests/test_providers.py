"""Tests for provider protocol and registry."""

from __future__ import annotations

import pytest

from parallect.providers import AsyncResearchProvider, ProviderResult, ResearchProvider
from parallect.providers.registry import ProviderRegistry, validate_provider


class MockSyncProvider:
    """A minimal mock that satisfies ResearchProvider protocol."""

    @property
    def name(self) -> str:
        return "mock-sync"

    def research(self, query: str) -> ProviderResult:
        return ProviderResult(provider="mock-sync", status="completed", report_markdown="# Test")

    def estimate_cost(self, query: str) -> float:
        return 0.01

    def is_available(self) -> bool:
        return True


class MockAsyncProvider:
    """A minimal mock that satisfies AsyncResearchProvider protocol."""

    @property
    def name(self) -> str:
        return "mock-async"

    async def research(self, query: str) -> ProviderResult:
        return ProviderResult(provider="mock-async", status="completed", report_markdown="# Test")

    def estimate_cost(self, query: str) -> float:
        return 0.02

    def is_available(self) -> bool:
        return True


class TestProviderProtocol:
    def test_sync_provider_matches_protocol(self):
        provider = MockSyncProvider()
        assert isinstance(provider, ResearchProvider)

    def test_async_provider_matches_protocol(self):
        provider = MockAsyncProvider()
        assert isinstance(provider, AsyncResearchProvider)

    def test_validate_provider(self):
        assert validate_provider(MockSyncProvider())
        assert validate_provider(MockAsyncProvider())
        assert not validate_provider("not a provider")
        assert not validate_provider(42)


class TestProviderRegistry:
    def test_register_and_get(self):
        registry = ProviderRegistry()
        provider = MockSyncProvider()
        registry.register(provider)

        assert registry.has("mock-sync")
        assert registry.get("mock-sync") is provider

    def test_list_available(self):
        registry = ProviderRegistry()
        registry.register(MockSyncProvider())
        registry.register(MockAsyncProvider())

        available = registry.list_available()
        assert available == ["mock-async", "mock-sync"]

    def test_get_missing_raises(self):
        registry = ProviderRegistry()
        with pytest.raises(KeyError, match="not-here"):
            registry.get("not-here")

    def test_register_invalid_raises(self):
        registry = ProviderRegistry()
        with pytest.raises(TypeError):
            registry.register("not a provider")


class TestProviderResult:
    def test_basic_result(self):
        result = ProviderResult(
            provider="test",
            status="completed",
            report_markdown="# Test Report",
        )
        assert result.provider == "test"
        assert result.status == "completed"
        assert result.error is None
        assert result.cost_usd is None

    def test_failed_result(self):
        result = ProviderResult(
            provider="test",
            status="failed",
            error="Connection timeout",
        )
        assert result.status == "failed"
        assert result.error == "Connection timeout"
        assert result.report_markdown == ""
