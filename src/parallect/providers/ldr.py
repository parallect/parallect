"""Local Deep Research (LDR) integration plugin.

Wraps the local-deep-research Python package as a parallect provider.
Install with: pip install parallect[ldr]
"""

from __future__ import annotations

import json
import time

from parallect.providers import AsyncResearchProvider, ProviderResult
from parallect.providers.hash_response import attach_response_hash


class LDRProvider:
    """Provider adapter for Local Deep Research (LDR).

    LDR runs entirely locally using Ollama or other local LLM backends,
    performing iterative deep research with web search.

    Requires: pip install local-deep-research
    """

    def __init__(
        self,
        model: str = "llama3.2",
        max_iterations: int = 5,
    ) -> None:
        self._model = model
        self._max_iterations = max_iterations

    @property
    def name(self) -> str:
        return "ldr"

    async def research(self, query: str) -> ProviderResult:
        """Run LDR research and return a ProviderResult."""
        try:
            from local_deep_research.api import quick_summary
        except ImportError:
            return ProviderResult(
                provider="ldr",
                status="failed",
                error=(
                    "local-deep-research not installed. "
                    "Install with: pip install local-deep-research"
                ),
            )

        start = time.monotonic()

        try:
            result = quick_summary(
                query=query,
                model=self._model,
                max_iterations=self._max_iterations,
            )

            duration = time.monotonic() - start

            # LDR returns a dict with 'summary' and optionally 'sources'
            # Serialize the raw result for hashing
            raw_body = result if isinstance(result, str) else json.dumps(result, default=str)
            report_md = result if isinstance(result, str) else result.get("summary", str(result))

            citations = []
            if isinstance(result, dict) and "sources" in result:
                for src in result["sources"]:
                    citations.append({
                        "url": src.get("url", ""),
                        "title": src.get("title", ""),
                    })

            provider_result = ProviderResult(
                provider="ldr",
                status="completed",
                report_markdown=report_md,
                citations=citations,
                model=self._model,
                cost_usd=0.0,  # Local — no API cost
                duration_seconds=round(duration, 2),
            )
            return attach_response_hash(provider_result, raw_body)
        except Exception as e:
            return ProviderResult(
                provider="ldr",
                status="failed",
                error=str(e),
                duration_seconds=round(time.monotonic() - start, 2),
            )

    def estimate_cost(self, query: str) -> float:
        """LDR is local, so cost is always $0."""
        return 0.0

    def is_available(self) -> bool:
        """Check if local-deep-research is installed."""
        try:
            import local_deep_research  # noqa: F401
            return True
        except ImportError:
            return False


class LDRHooks:
    """Plugin hooks entry point for LDR.

    Registers the LDR provider automatically when the plugin is installed.
    """

    async def pre_research(self, query: str, providers: list[str]) -> str:
        """No-op — LDR doesn't modify queries."""
        return query
