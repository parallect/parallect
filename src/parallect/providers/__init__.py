"""Provider protocol and base classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ProviderResult:
    """Result from a single provider research call."""

    provider: str
    status: str  # "completed", "partial", "failed"
    report_markdown: str = ""
    citations: list[dict] = field(default_factory=list)
    model: str | None = None
    cost_usd: float | None = None
    duration_seconds: float | None = None
    tokens: dict | None = None
    error: str | None = None


@runtime_checkable
class ResearchProvider(Protocol):
    """Protocol for synchronous research providers."""

    @property
    def name(self) -> str: ...

    def research(self, query: str) -> ProviderResult: ...

    def estimate_cost(self, query: str) -> float: ...

    def is_available(self) -> bool: ...


@runtime_checkable
class AsyncResearchProvider(Protocol):
    """Protocol for async research providers."""

    @property
    def name(self) -> str: ...

    async def research(self, query: str) -> ProviderResult: ...

    def estimate_cost(self, query: str) -> float: ...

    def is_available(self) -> bool: ...
