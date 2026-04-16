"""Fan-out to data source plugins as part of a research run.

This runs in parallel with the web provider fan-out and folds the plugin
results into the bundle as :class:`ProviderData` entries so synthesis picks
them up with zero extra work. Each plugin becomes a distinct "virtual
provider" in the bundle: it gets its own section in the manifest under
``providers_used`` and its own entry in ``providers/`` inside the .prx.

Parsing rules for ``--sources``:
  - ``prxhub`` -> lookup plugin by name, use default config
  - ``filesystem:notes`` -> instance ``notes`` under ``[[plugins.filesystem]]``
  - unknown source name -> raise (CLI reports to user)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import platformdirs

from parallect.plugins.data_sources import (
    DataSource,
    Document,
    PluginError,
    SourceSpec,
    get as get_plugin,
    parse_sources,
)
from parallect.providers import ProviderResult

logger = logging.getLogger(__name__)


@dataclass
class PluginFanOutResult:
    spec: SourceSpec
    result: ProviderResult | None = None
    error: str | None = None
    documents: list[Document] = field(default_factory=list)


def _extract_plugin_configs(settings: object | None) -> dict[str, list[dict]]:
    """Pull per-plugin config dicts out of user/project TOML.

    Our :class:`ParallectSettings` currently maps fixed TOML sections to
    flat attrs. For plugin config we parse the raw TOML files ourselves,
    because the schema is heterogeneous (filesystem supports an array of
    instances, the others are single objects).

    Returns a mapping plugin_name -> list of config dicts. For plugins
    that only allow one instance (prxhub, prior-research) the list has
    at most one entry.
    """
    import sys
    from pathlib import Path

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]

    out: dict[str, list[dict]] = {}

    paths = [
        Path(platformdirs.user_config_dir("parallect")) / "config.toml",
        Path.cwd() / "parallect.toml",
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        except Exception:
            logger.warning("failed to parse plugin config at %s", path, exc_info=True)
            continue
        plugins_section = raw.get("plugins") or {}
        for plugin_name, blob in plugins_section.items():
            if isinstance(blob, dict):
                out.setdefault(plugin_name, []).append(dict(blob))
            elif isinstance(blob, list):
                for item in blob:
                    if isinstance(item, dict):
                        out.setdefault(plugin_name, []).append(dict(item))
    return out


def _select_config(
    plugin_name: str,
    instance: str | None,
    plugin_configs: dict[str, list[dict]],
) -> dict:
    """Pick the config dict for a given ``SourceSpec``.

    - Unknown plugin with no config -> empty dict (plugin gets defaults)
    - Multi-instance plugin (filesystem) with ``instance`` -> match by
      ``name`` field; raise if no match.
    - Single-instance plugin with ``instance`` -> warn and ignore instance.
    """
    configs = plugin_configs.get(plugin_name) or []
    if not configs:
        if instance:
            return {"name": instance}
        return {}
    if plugin_name == "filesystem":
        if instance:
            for cfg in configs:
                if str(cfg.get("name") or "default") == instance:
                    return dict(cfg)
            raise PluginError(
                f"filesystem instance '{instance}' not found in config. "
                f"Available: {[c.get('name') for c in configs]}"
            )
        # No instance requested -> first configured instance wins.
        return dict(configs[0])
    # Single-instance plugin: just take the first.
    return dict(configs[0])


async def _run_one(
    spec: SourceSpec,
    plugin: DataSource,
    config: dict,
    query: str,
    *,
    limit: int = 10,
    timeout: float = 60.0,
) -> PluginFanOutResult:
    start = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            await plugin.configure(config)
            docs = await plugin.search(query, limit=limit)
    except TimeoutError:
        return PluginFanOutResult(
            spec=spec, result=None, error=f"timed out after {timeout}s", documents=[]
        )
    except Exception as exc:
        return PluginFanOutResult(spec=spec, result=None, error=str(exc), documents=[])
    elapsed = time.monotonic() - start
    report_md = _render_docs(spec, docs)
    citations = [
        {
            "url": d.source_url or "",
            "title": d.title,
            "snippet": (d.content[:200] if d.content else None),
            "domain": None,
        }
        for d in docs
        if d.source_url
    ]
    result = ProviderResult(
        provider=spec.display,
        status="completed",
        report_markdown=report_md,
        citations=citations,
        model=None,
        cost_usd=0.0,
        duration_seconds=round(elapsed, 3),
        tokens=None,
    )
    return PluginFanOutResult(spec=spec, result=result, documents=list(docs))


def _render_docs(spec: SourceSpec, docs: list[Document]) -> str:
    """Format docs into provider-report-style markdown.

    Synthesis reads this as free-form prose. Source URLs and scores are
    included so later steps can cite them even if the structured citations
    list is dropped.
    """
    if not docs:
        return f"# Source: {spec.display}\n\n*No matching documents.*\n"
    lines = [f"# Source: {spec.display}", ""]
    for i, d in enumerate(docs, 1):
        header = d.title or f"Document {i}"
        url = f" ({d.source_url})" if d.source_url else ""
        score = f" [score={d.score:.3f}]" if d.score is not None else ""
        lines.append(f"## {i}. {header}{url}{score}")
        if d.metadata:
            kind = d.metadata.get("kind")
            if kind:
                lines.append(f"*kind:* {kind}")
        lines.append("")
        lines.append(d.content or "")
        lines.append("")
    return "\n".join(lines)


async def run_plugin_sources(
    query: str,
    sources_raw: str | None,
    *,
    settings: object | None = None,
    limit: int = 10,
    timeout: float = 60.0,
) -> list[PluginFanOutResult]:
    """Parse ``--sources``, look up plugins, run them in parallel.

    Returns one :class:`PluginFanOutResult` per spec, in input order.
    Unknown sources raise :class:`PluginError` from the parser/resolver so
    the CLI fails loudly — silent drops are a nightmare to debug.
    """
    specs = parse_sources(sources_raw)
    if not specs:
        return []

    plugin_configs = _extract_plugin_configs(settings)

    resolved: list[tuple[SourceSpec, DataSource, dict]] = []
    for spec in specs:
        plugin = get_plugin(spec.name)  # raises PluginError on unknown
        config = _select_config(spec.name, spec.instance, plugin_configs)
        # filesystem requires an explicit path; fail loudly early.
        if spec.name == "filesystem" and not config.get("path"):
            raise PluginError(
                f"filesystem source '{spec.display}' has no `path` configured. "
                "Add `[[plugins.filesystem]] path = \"...\"` to config.toml or "
                "run `parallect plugins config filesystem`."
            )
        resolved.append((spec, plugin, config))

    results = await asyncio.gather(
        *[
            _run_one(spec, plugin, config, query, limit=limit, timeout=timeout)
            for spec, plugin, config in resolved
        ]
    )
    return list(results)


__all__ = ["PluginFanOutResult", "run_plugin_sources"]
