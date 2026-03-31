"""OpenAI Deep Research provider.

Two tiers:
  default  — gpt-4o-mini  + web_search_preview (fast, cheap)
  --deep   — o3-deep-research, background polling up to 60 min

Uses the Responses API (/v1/responses) with url_citation annotations.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from parallect.providers import ProviderResult

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4o-mini"
DEEP_MODEL = "o3-deep-research"
POLL_INTERVAL_S = 10
MAX_POLL_S = 60 * 60  # 60 minutes


class OpenAIDRProvider:
    """Adapter for the OpenAI Responses API with web search."""

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
        return "openai"

    async def research(self, query: str) -> ProviderResult:
        start = time.monotonic()
        try:
            is_deep = "deep-research" in self.model
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            body: dict = {
                "model": self.model,
                "input": query,
                "tools": [{"type": "web_search_preview"}],
            }
            if is_deep:
                body["background"] = True

            async with httpx.AsyncClient(
                timeout=self.timeout if not is_deep else 30.0
            ) as client:
                response = await client.post(
                    OPENAI_RESPONSES_URL, headers=headers, json=body,
                )
                response.raise_for_status()
                data = response.json()

            # Deep research requires polling
            if is_deep and data.get("status") != "completed":
                data = await self._poll(data["id"], headers)

            duration = time.monotonic() - start
            return self._parse(data, duration)

        except Exception as e:
            duration = time.monotonic() - start
            return ProviderResult(
                provider="openai",
                status="failed",
                error=str(e),
                duration_seconds=round(duration, 2),
            )

    async def _poll(self, response_id: str, headers: dict) -> dict:
        """Poll until deep-research job completes."""
        deadline = time.monotonic() + MAX_POLL_S
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.monotonic() < deadline:
                await asyncio.sleep(POLL_INTERVAL_S)
                resp = await client.get(
                    f"{OPENAI_RESPONSES_URL}/{response_id}",
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status")
                if status == "completed":
                    return data
                if status in ("failed", "cancelled"):
                    raise RuntimeError(f"OpenAI deep research {status}")
        raise TimeoutError("OpenAI deep research timed out after 60 minutes")

    def _parse(self, data: dict, duration: float) -> ProviderResult:
        """Parse Responses API output into ProviderResult."""
        markdown = ""
        citations: list[dict] = []
        web_search_count = 0

        for item in data.get("output", []):
            if item.get("type") == "web_search_call":
                web_search_count += 1
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text" and block.get("text"):
                        markdown += block["text"]
                    for ann in block.get("annotations", []):
                        if ann.get("type") == "url_citation" and ann.get("url"):
                            citations.append({
                                "url": ann["url"],
                                "title": ann.get("title"),
                            })

        usage = data.get("usage", {})
        cost = self._calculate_cost(usage, self.model, web_search_count)

        return ProviderResult(
            provider="openai",
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
                "cached": usage.get("input_tokens_details", {}).get("cached_tokens", 0),
                "reasoning": usage.get("output_tokens_details", {}).get("reasoning_tokens", 0),
            },
        )

    @staticmethod
    def _calculate_cost(usage: dict, model: str, web_search_count: int) -> float:
        """Token-based cost using OpenAI pricing tiers."""
        if "o4-mini" in model:
            input_rate, cached_rate, output_rate = 1.10, 0.275, 4.40
        elif "o3" in model:
            input_rate, cached_rate, output_rate = 10.0, 2.5, 40.0
        elif "gpt-4o-mini" in model:
            input_rate, cached_rate, output_rate = 0.15, 0.075, 0.60
        elif "gpt-4o" in model:
            input_rate, cached_rate, output_rate = 2.5, 1.25, 10.0
        else:
            input_rate, cached_rate, output_rate = 2.0, 0.5, 8.0

        cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        billable_input = usage.get("input_tokens", 0) - cached
        input_cost = (billable_input / 1_000_000) * input_rate
        cached_cost = (cached / 1_000_000) * cached_rate
        output_cost = (usage.get("output_tokens", 0) / 1_000_000) * output_rate
        search_cost = (web_search_count / 1000) * 10.0
        return input_cost + cached_cost + output_cost + search_cost

    def estimate_cost(self, query: str) -> float:
        if "deep-research" in self.model:
            return 7.50
        return 0.05

    def is_available(self) -> bool:
        return bool(self.api_key)
