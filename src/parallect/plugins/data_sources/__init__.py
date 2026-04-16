"""Data source plugin protocol + discovery.

A *data source* is anything that can contribute :class:`Document` objects to
a research run: a semantic index over local notes, a hit against the public
prxhub API, a cache of your own prior runs, or a third-party plugin
distributed on PyPI.

Discovery layers, highest precedence first:

1. Explicit registration via :func:`register` (programmatic).
2. In-core built-ins (auto-registered on first :func:`get_registry` call):
   ``prxhub``, ``filesystem``, ``prior-research``.
3. Entry points under the ``parallect.data_sources`` group (third-party).

The CLI surfaces plugins via ``parallect plugins list|status|index|config``
and the orchestrator consumes them via ``--sources``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """One result from a data source.

    ``score`` is the source's own relevance signal (cosine sim, hub rank,
    ...). The orchestrator treats this as a hint only; synthesis sees all
    returned docs regardless of score.
    """

    id: str
    content: str
    title: str | None = None
    source_url: str | None = None
    metadata: dict = field(default_factory=dict)
    score: float | None = None


@dataclass
class IndexStats:
    """Summary returned by :meth:`DataSource.index`."""

    source_name: str
    documents_indexed: int
    documents_skipped: int
    elapsed_seconds: float
    index_path: str | None = None


@runtime_checkable
class DataSource(Protocol):
    """Protocol every data source plugin must satisfy.

    Using ``runtime_checkable`` so that ``isinstance(plugin, DataSource)``
    works in tests. Plugins don't need to subclass — a plain class with
    matching attributes/methods is enough.
    """

    name: str
    display_name: str
    requires_index: bool

    async def configure(self, config: dict) -> None: ...
    async def search(self, query: str, *, limit: int = 10) -> list[Document]: ...
    async def index(self, *, force: bool = False) -> IndexStats: ...
    async def is_fresh(self) -> bool: ...
    async def fetch(self, doc_id: str) -> Document | None: ...
    async def health_check(self) -> dict: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PluginError(RuntimeError):
    """Plugin-specific error (unknown source, misconfigured, etc)."""


_REGISTRY: dict[str, DataSource] = {}
_INCORE_LOADED = False
_ENTRY_POINTS_LOADED = False


def register(plugin: DataSource) -> None:
    """Register a data source plugin by its ``.name``.

    The last registration wins. This lets tests swap in mocks and lets users
    override a built-in by installing a third-party plugin with the same name.
    """
    if not hasattr(plugin, "name") or not plugin.name:
        raise PluginError(f"plugin {plugin!r} has no .name")
    _REGISTRY[plugin.name] = plugin


def _load_incore() -> None:
    """Register the three in-core plugins.

    Imported lazily so that importing this module doesn't pull in sqlite,
    httpx, or other heavier deps until a caller actually wants the registry.
    """
    global _INCORE_LOADED
    if _INCORE_LOADED:
        return
    _INCORE_LOADED = True
    from parallect.plugins.data_sources.filesystem import FilesystemPlugin
    from parallect.plugins.data_sources.prior_research import PriorResearchCache
    from parallect.plugins.data_sources.obsidian import ObsidianPlugin
    from parallect.plugins.data_sources.prxhub import PrxhubPlugin

    for plugin in (PrxhubPlugin(), FilesystemPlugin(), ObsidianPlugin(), PriorResearchCache()):
        # Don't clobber an already-registered instance (tests register first).
        _REGISTRY.setdefault(plugin.name, plugin)


def _load_entry_points() -> None:
    """Load third-party plugins from the ``parallect.data_sources`` group.

    Entry-point targets must either be a ``DataSource`` instance or a zero-arg
    callable that returns one. Failures are logged, not raised — a broken
    plugin shouldn't kill the CLI.
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        eps = entry_points(group="parallect.data_sources")
    except Exception:
        logger.warning("failed to enumerate entry points", exc_info=True)
        return
    for ep in eps:
        try:
            target = ep.load()
            plugin = target() if callable(target) else target
            if not isinstance(plugin, DataSource):
                logger.warning(
                    "entry point %s does not satisfy DataSource protocol", ep.name
                )
                continue
            register(plugin)
        except Exception:
            logger.warning("failed to load data source plugin %s", ep.name, exc_info=True)


def get_registry() -> dict[str, DataSource]:
    """Return the current plugin registry (in-core + entry points)."""
    _load_incore()
    _load_entry_points()
    return dict(_REGISTRY)


def get(name: str) -> DataSource:
    """Look up a plugin by name. Raises :class:`PluginError` if unknown."""
    reg = get_registry()
    if name not in reg:
        available = ", ".join(sorted(reg.keys())) or "(none)"
        raise PluginError(f"unknown data source '{name}'. Available: {available}")
    return reg[name]


def reset_registry() -> None:
    """Test-only: clear the registry so tests can seed their own plugins."""
    global _INCORE_LOADED, _ENTRY_POINTS_LOADED
    _REGISTRY.clear()
    _INCORE_LOADED = False
    _ENTRY_POINTS_LOADED = False


# ---------------------------------------------------------------------------
# --sources parsing (shared by CLI + orchestrator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """A parsed ``--sources`` entry.

    ``name`` is the plugin name (e.g. ``filesystem``). ``instance`` is the
    optional instance id for multi-instance plugins (e.g. ``notes`` in
    ``filesystem:notes``). ``None`` means "the default instance".
    """

    name: str
    instance: str | None = None

    @property
    def display(self) -> str:
        return f"{self.name}:{self.instance}" if self.instance else self.name


def parse_sources(raw: str | None) -> list[SourceSpec]:
    """Parse a ``--sources`` CLI string into specs.

    Accepts ``None``/empty (returns ``[]``), a comma-separated list, or
    ``plugin:instance`` entries. Empty tokens are skipped. Raises
    :class:`PluginError` on an empty plugin name (e.g. ``":notes"``).
    """
    if not raw:
        return []
    specs: list[SourceSpec] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            name, instance = token.split(":", 1)
            name = name.strip()
            instance = instance.strip() or None
        else:
            name, instance = token, None
        if not name:
            raise PluginError(f"invalid --sources token: '{token}'")
        specs.append(SourceSpec(name=name, instance=instance))
    return specs


__all__ = [
    "DataSource",
    "Document",
    "IndexStats",
    "PluginError",
    "SourceSpec",
    "get",
    "get_registry",
    "parse_sources",
    "register",
    "reset_registry",
]
