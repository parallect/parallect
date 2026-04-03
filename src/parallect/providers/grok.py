"""xAI Grok provider with web search + X/Twitter search.

Two tiers:
  default  — grok-3 via Responses API (no server-side tools, fast)
  --deep   — grok-4 via Responses API + web_search + x_search tools

Note: xAI only supports server-side tools (web_search, x_search) with
the grok-4 family of models. grok-3 uses the Responses API without tools.

Citations extracted from output annotations and top-level citations array.
Cost from cost_in_usd_ticks when available, otherwise calculated.
"""

from __future__ import annotations

import time

import json

import httpx

from parallect.providers import ProviderResult
from parallect.providers.hash_response import attach_response_hash

XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
DEFAULT_MODEL = "grok-3"
DEEP_MODEL = "grok-4"


class GrokProvider:
    """Adapter for the xAI Grok Responses API with web + X search."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
        deep: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = DEEP_MODEL if deep else model
        self.timeout = timeout
        self._deep = deep

    @property
    def name(self) -> str:
        return "grok"

    async def research(self, query: str) -> ProviderResult:
        start = time.monotonic()
        try:
            prompt = (
                "Conduct thorough research on the following topic. "
                "Provide comprehensive analysis with sources.\n\n"
                f"{query}"
            )
            body: dict = {
                "model": self.model,
                "input": prompt,
            }
            # Server-side tools only supported on grok-4 family
            if self._deep or "grok-4" in self.model:
                body["tools"] = [
                    {"type": "web_search"},
                    {"type": "x_search"},
                ]

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    XAI_RESPONSES_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                response.raise_for_status()
                raw_text = response.text
                data = json.loads(raw_text)

            duration = time.monotonic() - start
            return self._parse(data, duration, raw_text)

        except Exception as e:
            duration = time.monotonic() - start
            return ProviderResult(
                provider="grok",
                status="failed",
                error=str(e),
                duration_seconds=round(duration, 2),
            )

    def _parse(self, data: dict, duration: float, raw_text: str = "") -> ProviderResult:
        """Parse Responses API output into ProviderResult."""
        markdown = ""
        citations: list[dict] = []
        seen_urls: set[str] = set()

        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text" and block.get("text"):
                        markdown += block["text"]
                    for ann in block.get("annotations", []):
                        url = ann.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            citations.append({
                                "url": url,
                                "title": ann.get("title"),
                            })

        # Top-level citations array (fallback / additional)
        for url in data.get("citations", []):
            if url not in seen_urls:
                seen_urls.add(url)
                citations.append({"url": url})

        usage = data.get("usage", {})
        tool_usage = data.get("server_side_tool_usage", {})

        # Prefer API-reported cost when available
        ticks = usage.get("cost_in_usd_ticks")
        if ticks is not None:
            cost = ticks / 10_000_000_000
        else:
            cost = self._calculate_cost(usage, tool_usage, self.model)

        result = ProviderResult(
            provider="grok",
            status="completed",
            report_markdown=markdown,
            citations=citations,
            model=data.get("model", self.model),
            cost_usd=cost,
            duration_seconds=round(duration, 2),
            tokens={
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "total": usage.get("total_tokens", 0),
                "cached": usage.get("cached_tokens", 0),
                "reasoning": usage.get("reasoning_tokens", 0),
                "web_searches": tool_usage.get("web_search", 0),
                "x_searches": tool_usage.get("x_search", 0),
            },
        )
        if raw_text:
            return attach_response_hash(result, raw_text)
        return result

    @staticmethod
    def _calculate_cost(
        usage: dict, tool_usage: dict, model: str
    ) -> float:
        """Token-based cost with tool search pricing."""
        is_premium = "grok-4-0709" in model or "grok-3" in model
        is_beta = "4.20" in model or "beta" in model
        if is_premium:
            input_rate, cached_rate, output_rate = 3.0, 0.75, 15.0
        elif is_beta:
            input_rate, cached_rate, output_rate = 2.0, 0.2, 6.0
        else:
            input_rate, cached_rate, output_rate = 0.2, 0.05, 0.5

        cached = usage.get("cached_tokens", 0)
        billable_input = usage.get("input_tokens", 0) - cached
        input_cost = (billable_input / 1_000_000) * input_rate
        cached_cost = (cached / 1_000_000) * cached_rate
        output_cost = (usage.get("output_tokens", 0) / 1_000_000) * output_rate
        web_cost = (tool_usage.get("web_search", 0) / 1000) * 5.0
        x_cost = (tool_usage.get("x_search", 0) / 1000) * 5.0
        return input_cost + cached_cost + output_cost + web_cost + x_cost

    def estimate_cost(self, query: str) -> float:
        return 0.50

    def is_available(self) -> bool:
        return bool(self.api_key)
