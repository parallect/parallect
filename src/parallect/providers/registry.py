"""Provider registry: discovery, registration, and validation."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from parallect.providers import AsyncResearchProvider, ResearchProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Central registry for research providers.

    Providers can be registered explicitly, loaded from entry points,
    or configured via settings.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ResearchProvider | AsyncResearchProvider] = {}

    def register(self, provider: ResearchProvider | AsyncResearchProvider) -> None:
        """Register a provider instance. Validates protocol compliance."""
        if not isinstance(provider, (ResearchProvider, AsyncResearchProvider)):
            raise TypeError(f"{provider} does not implement ResearchProvider protocol")
        self._providers[provider.name] = provider

    def discover_entry_points(self) -> None:
        """Load providers from parallect.providers entry point group."""
        for ep in entry_points(group="parallect.providers"):
            try:
                provider_cls = ep.load()
                logger.debug("Discovered provider entry point: %s", ep.name)
                self._providers[ep.name] = provider_cls
            except Exception:
                logger.warning("Failed to load provider entry point: %s", ep.name, exc_info=True)

    def get(self, name: str) -> ResearchProvider | AsyncResearchProvider:
        """Get a registered provider by name."""
        if name not in self._providers:
            available = ", ".join(sorted(self._providers.keys()))
            raise KeyError(f"Provider '{name}' not found. Available: {available}")
        return self._providers[name]

    def list_available(self) -> list[str]:
        """Return names of all registered providers."""
        return sorted(self._providers.keys())

    def has(self, name: str) -> bool:
        """Check if a provider is registered."""
        return name in self._providers


def validate_provider(obj: Any) -> bool:
    """Check if an object implements the ResearchProvider protocol."""
    return isinstance(obj, (ResearchProvider, AsyncResearchProvider))
