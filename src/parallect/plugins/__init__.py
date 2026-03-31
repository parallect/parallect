"""Plugin system for parallect.

Three discovery layers:
1. Entry points (pip-installable packages)
2. Config (user-registered in config.toml)
3. Explicit imports (programmatic use)
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class PluginHooks(Protocol):
    """Optional hook points for plugins. Implement any subset."""

    async def pre_research(self, query: str, providers: list[str]) -> str: ...
    async def post_provider(self, provider: str, result: Any) -> Any: ...
    async def post_synthesis(self, synthesis: Any) -> Any: ...
    async def post_bundle(self, bundle: Any) -> Any: ...


class PluginManager:
    """Manages plugin discovery, loading, and hook execution."""

    def __init__(self) -> None:
        self._hooks: list[Any] = []

    def register_hook(self, hook: Any) -> None:
        """Register a plugin hook object."""
        self._hooks.append(hook)

    def discover_entry_points(self) -> None:
        """Load hooks from parallect.hooks entry point group."""
        for ep in entry_points(group="parallect.hooks"):
            try:
                hook = ep.load()
                self._hooks.append(hook)
                logger.debug("Loaded hook plugin: %s", ep.name)
            except Exception:
                logger.warning("Failed to load hook plugin: %s", ep.name, exc_info=True)

    async def run_pre_research(self, query: str, providers: list[str]) -> str:
        """Run all pre_research hooks in order. Returns (possibly modified) query."""
        for hook in self._hooks:
            if hasattr(hook, "pre_research"):
                query = await hook.pre_research(query, providers)
        return query

    async def run_post_provider(self, provider: str, result: Any) -> Any:
        """Run all post_provider hooks in order."""
        for hook in self._hooks:
            if hasattr(hook, "post_provider"):
                result = await hook.post_provider(provider, result)
        return result

    async def run_post_synthesis(self, synthesis: Any) -> Any:
        """Run all post_synthesis hooks in order."""
        for hook in self._hooks:
            if hasattr(hook, "post_synthesis"):
                synthesis = await hook.post_synthesis(synthesis)
        return synthesis

    async def run_post_bundle(self, bundle: Any) -> Any:
        """Run all post_bundle hooks in order."""
        for hook in self._hooks:
            if hasattr(hook, "post_bundle"):
                bundle = await hook.post_bundle(bundle)
        return bundle
