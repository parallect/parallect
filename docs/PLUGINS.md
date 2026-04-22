# Plugin Guide

Extend parallect with custom hooks that run during the research pipeline.

## Hook Points

Parallect exposes 4 hook points in the research lifecycle:

| Hook | When | Can Modify |
|------|------|-----------|
| `pre_research` | Before fan-out to providers | Query text |
| `post_provider` | After each provider returns | Provider result |
| `post_synthesis` | After synthesis completes | Synthesis markdown |
| `post_bundle` | After bundle is assembled | Full bundle |

## Writing a Plugin

Create a class that implements any subset of the hooks:

```python
class MyPlugin:
    """Example plugin that adds a disclaimer to synthesis."""

    async def post_synthesis(self, synthesis: str) -> str:
        return synthesis + "\n\n---\n*Processed by MyPlugin*"
```

You don't need to implement all hooks — only the ones you need.

## Hook Signatures

```python
from typing import Protocol, Any

class PluginHooks(Protocol):
    async def pre_research(self, query: str, providers: list[str]) -> str:
        """Modify the query before it's sent to providers."""
        ...

    async def post_provider(self, provider: str, result: Any) -> Any:
        """Process or filter a provider's result."""
        ...

    async def post_synthesis(self, synthesis: Any) -> Any:
        """Modify synthesis output."""
        ...

    async def post_bundle(self, bundle: Any) -> Any:
        """Modify the assembled bundle before writing."""
        ...
```

## Registering Plugins

### Via Entry Points (recommended for pip packages)

In your package's `pyproject.toml`:

```toml
[project.entry-points."parallect.hooks"]
my-plugin = "my_package:MyPlugin"
```

When your package is installed, parallect automatically discovers and loads it.

### Programmatic Registration

```python
from parallect.plugins import PluginManager

mgr = PluginManager()
mgr.register_hook(MyPlugin())
```

## Example: Query Logger

```python
import logging

logger = logging.getLogger(__name__)

class QueryLogger:
    """Log every research query."""

    async def pre_research(self, query: str, providers: list[str]) -> str:
        logger.info("Research: %s (providers: %s)", query, providers)
        return query  # Don't modify
```

## Example: Cost Monitor

```python
class CostMonitor:
    """Track cumulative cost across research runs."""

    def __init__(self):
        self.total_cost = 0.0

    async def post_provider(self, provider: str, result):
        if hasattr(result, 'cost_usd') and result.cost_usd:
            self.total_cost += result.cost_usd
            if self.total_cost > 5.0:
                print(f"Warning: cumulative cost ${self.total_cost:.2f}")
        return result
```

## Example: Content Filter

```python
class ContentFilter:
    """Filter provider results that are too short."""

    MIN_LENGTH = 500

    async def post_provider(self, provider: str, result):
        if len(result.report_markdown) < self.MIN_LENGTH:
            result.status = "partial"
        return result
```

## Plugin Discovery Order

1. **Entry points** — installed packages with `parallect.hooks` entry points
2. **Config** — plugins listed in `config.toml` (future)
3. **Explicit** — programmatically registered via `PluginManager.register_hook()`

Hooks from all sources run in registration order.

## LDR Plugin

The Local Deep Research integration is a built-in plugin:

```bash
pip install "parallect-cli[ldr]"
parallect research "my query" -p ldr
```

See `parallect.providers.ldr` for the implementation pattern.
