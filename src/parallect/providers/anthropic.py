"""Anthropic Claude provider with web search + web fetch tools.

Two tiers:
  default  — claude-sonnet-4 + web_search(20) + web_fetch(10), agentic loop
  --deep   — claude-opus-4 + same tools, higher step limit

Uses the Messages API with server-side tool_use for web research.
Citations extracted from web_search tool results.
"""

from __future__ import annotations

import time

import httpx

from parallect.providers import ProviderResult

ANTHROPIC_API_URL = "https://api.anthropic.com/v1"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEEP_MODEL = "claude-opus-4-20250918"
MAX_STEPS = 15
DEEP_MAX_STEPS = 25


class AnthropicProvider:
    """Adapter for the Anthropic Messages API with web search tools."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        deep: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = DEEP_MODEL if deep else model
        self.timeout = timeout
        self._deep = deep
        self._max_steps = DEEP_MAX_STEPS if deep else MAX_STEPS

    @property
    def name(self) -> str:
        return "anthropic"

    async def research(self, query: str) -> ProviderResult:
        start = time.monotonic()
        try:
            result = await self._agentic_loop(query)
            duration = time.monotonic() - start
            result.duration_seconds = round(duration, 2)
            return result
        except Exception as e:
            duration = time.monotonic() - start
            return ProviderResult(
                provider="anthropic",
                status="failed",
                error=str(e),
                duration_seconds=round(duration, 2),
            )

    async def _agentic_loop(self, query: str) -> ProviderResult:
        """Run multi-step agentic loop with tool use."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "web-search-2025-03-05",
            "Content-Type": "application/json",
        }
        messages: list[dict] = [{"role": "user", "content": query}]
        all_text: list[str] = []
        citations: list[dict] = []
        seen_urls: set[str] = set()
        total_input = 0
        total_output = 0
        search_requests = 0
        fetch_requests = 0

        for _step in range(self._max_steps):
            body: dict = {
                "model": self.model,
                "max_tokens": 16384,
                "temperature": 0.3,
                "system": (
                    "You are a thorough research analyst. Use web search and web fetch "
                    "tools to find current, authoritative information. Search multiple "
                    "queries to build a comprehensive picture. For each claim, cite "
                    "your sources with URLs. Synthesize findings into a well-structured "
                    "markdown report with clear sections, evidence, and source citations."
                ),
                "messages": messages,
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 20,
                    },
                    {
                        "type": "web_fetch_20250910",
                        "name": "web_fetch",
                        "max_uses": 10,
                    },
                ],
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{ANTHROPIC_API_URL}/messages",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()

            usage = data.get("usage", {})
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)

            # Track server-side tool usage
            server_tool = (
                data.get("usage", {}).get("server_tool_use", {})
            )
            search_requests += server_tool.get("web_search_requests", 0)

            # Collect text and tool results
            content_blocks = data.get("content", [])
            has_tool_use = False
            tool_results: list[dict] = []

            for block in content_blocks:
                if block.get("type") == "text" and block.get("text"):
                    all_text.append(block["text"])
                elif block.get("type") == "web_search_tool_result":
                    search_requests += 1  # fallback count
                    for item in block.get("content", []):
                        if item.get("type") == "web_search_result":
                            url = item.get("url", "")
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                citations.append({
                                    "url": url,
                                    "title": item.get("title"),
                                    "snippet": (item.get("snippet") or "")[:200],
                                })
                elif block.get("type") == "web_fetch_tool_result":
                    fetch_requests += 1
                elif block.get("type") == "tool_use":
                    has_tool_use = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": "Continue with analysis.",
                    })

            # If model wants to use tools, add assistant + tool_result messages
            if has_tool_use and data.get("stop_reason") == "tool_use":
                messages.append({"role": "assistant", "content": content_blocks})
                messages.append({"role": "user", "content": tool_results})
            else:
                # Model finished (end_turn or stop)
                break

        markdown = "\n".join(all_text)
        cost = self._calculate_cost(
            total_input, total_output, search_requests, self.model
        )

        return ProviderResult(
            provider="anthropic",
            status="completed",
            report_markdown=markdown,
            citations=citations,
            model=self.model,
            cost_usd=cost,
            tokens={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
                "search_requests": search_requests,
                "fetch_requests": fetch_requests,
            },
        )

    @staticmethod
    def _calculate_cost(
        input_tokens: int,
        output_tokens: int,
        search_requests: int,
        model: str,
    ) -> float:
        """Token-based cost plus per-search pricing."""
        is_opus = "opus" in model
        input_rate = 5.0 if is_opus else 3.0
        output_rate = 25.0 if is_opus else 15.0
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        search_cost = (search_requests / 1000) * 10.0
        return input_cost + output_cost + search_cost

    def estimate_cost(self, query: str) -> float:
        return 4.50 if self._deep else 0.50

    def is_available(self) -> bool:
        return bool(self.api_key)
