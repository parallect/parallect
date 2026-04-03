"""Google Gemini provider with grounded web search.

Two tiers:
  default  — gemini-2.5-flash + googleSearch tool (fast, grounded)
  --deep   — deep-research-pro-preview-12-2025 via Interactions API (async)

Citations extracted from groundingMetadata.groundingChunks (default)
or interaction output citations (deep).
"""

from __future__ import annotations

import asyncio
import time

import json

import httpx

from parallect.providers import ProviderResult
from parallect.providers.hash_response import attach_response_hash

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"
DEEP_AGENT = "deep-research-pro-preview-12-2025"
POLL_INTERVAL_S = 10
MAX_POLL_S = 60 * 60


class GeminiProvider:
    """Adapter for the Google Gemini API with web search grounding.

    Uses generateContent with googleSearch tool for default mode,
    or the Interactions API for deep research.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
        deep: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._deep = deep

    @property
    def name(self) -> str:
        return "gemini"

    async def research(self, query: str) -> ProviderResult:
        if self._deep:
            return await self._deep_research(query)
        return await self._grounded_research(query)

    # ----- Default: generateContent + googleSearch -----

    async def _grounded_research(self, query: str) -> ProviderResult:
        start = time.monotonic()
        try:
            prompt = (
                "Conduct thorough deep research on the following topic. "
                "Provide comprehensive analysis with citations.\n\n"
                f"{query}"
            )
            url = (
                f"{GEMINI_API_URL}/models/{self.model}:generateContent"
                f"?key={self.api_key}"
            )
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 65536,
                        },
                        "tools": [{"googleSearch": {}}],
                    },
                )
                response.raise_for_status()
                raw_text = response.text
                data = json.loads(raw_text)

            duration = time.monotonic() - start
            candidates = data.get("candidates", [])
            if not candidates:
                return ProviderResult(
                    provider="gemini",
                    status="failed",
                    error="No candidates in response",
                    duration_seconds=round(duration, 2),
                )

            parts = candidates[0].get("content", {}).get("parts", [])
            content = "\n".join(p.get("text", "") for p in parts)

            # Extract grounding citations
            citations: list[dict] = []
            grounding = candidates[0].get("groundingMetadata", {})
            for chunk in grounding.get("groundingChunks", []):
                web = chunk.get("web", {})
                if web.get("uri"):
                    citations.append({
                        "url": web["uri"],
                        "title": web.get("title"),
                    })

            usage = data.get("usageMetadata", {})
            cost = self._calculate_cost(usage)

            result = ProviderResult(
                provider="gemini",
                status="completed",
                report_markdown=content,
                citations=citations,
                model=self.model,
                cost_usd=cost,
                duration_seconds=round(duration, 2),
                tokens={
                    "input": usage.get("promptTokenCount", 0),
                    "output": usage.get("candidatesTokenCount", 0),
                    "total": usage.get("totalTokenCount", 0),
                    "thoughts": usage.get("thoughtsTokenCount", 0),
                },
            )
            return attach_response_hash(result, raw_text)
        except Exception as e:
            duration = time.monotonic() - start
            return ProviderResult(
                provider="gemini",
                status="failed",
                error=str(e),
                duration_seconds=round(duration, 2),
            )

    # ----- Deep: Interactions API with polling -----

    async def _deep_research(self, query: str) -> ProviderResult:
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{GEMINI_API_URL}/interactions?key={self.api_key}",
                    json={
                        "input": query,
                        "agent": DEEP_AGENT,
                        "background": True,
                    },
                )
                resp.raise_for_status()
                created = resp.json()

            data, raw_text = await self._poll(created["id"])
            duration = time.monotonic() - start
            result = self._parse_interaction(data, duration, query)
            return attach_response_hash(result, raw_text)

        except Exception as e:
            duration = time.monotonic() - start
            return ProviderResult(
                provider="gemini",
                status="failed",
                error=str(e),
                duration_seconds=round(duration, 2),
            )

    async def _poll(self, interaction_id: str) -> tuple[dict, str]:
        """Poll Interactions API until completion.

        Returns (parsed_data, raw_response_text) tuple.
        """
        deadline = time.monotonic() + MAX_POLL_S
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.monotonic() < deadline:
                await asyncio.sleep(POLL_INTERVAL_S)
                resp = await client.get(
                    f"{GEMINI_API_URL}/interactions/{interaction_id}"
                    f"?key={self.api_key}",
                )
                resp.raise_for_status()
                raw_text = resp.text
                data = json.loads(raw_text)
                status = data.get("status")
                if status == "completed":
                    return data, raw_text
                if status == "failed":
                    raise RuntimeError(
                        f"Gemini Deep Research failed: {data.get('error', 'unknown')}"
                    )
        raise TimeoutError("Gemini Deep Research timed out")

    def _parse_interaction(self, data: dict, duration: float, query: str) -> ProviderResult:
        """Parse Interactions API response."""
        markdown = ""
        citations: list[dict] = []
        seen_urls: set[str] = set()

        for output in data.get("outputs", []):
            if output.get("text"):
                markdown += output["text"]
            for c in output.get("citations", []):
                url = c.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    citations.append({"url": url, "title": c.get("title")})

        # Try both camelCase and snake_case usage formats
        usage_cm = data.get("usageMetadata") or data.get("usage_metadata") or {}
        usage_sn = data.get("usage", {})

        if usage_cm.get("totalTokenCount", 0) > 0:
            usage_data = usage_cm
        elif usage_sn.get("total_tokens", 0) > 0:
            usage_data = {
                "promptTokenCount": usage_sn.get("total_input_tokens", 0),
                "candidatesTokenCount": usage_sn.get("total_output_tokens", 0),
                "totalTokenCount": usage_sn.get("total_tokens", 0),
                "thoughtsTokenCount": usage_sn.get("total_thought_tokens", 0),
            }
        else:
            usage_data = {}

        cost = self._calculate_deep_cost(usage_data) if usage_data else 0.0

        return ProviderResult(
            provider="gemini",
            status="completed",
            report_markdown=markdown,
            citations=citations,
            model=DEEP_AGENT,
            cost_usd=cost,
            duration_seconds=round(duration, 2),
            tokens={
                "input": usage_data.get("promptTokenCount", 0),
                "output": usage_data.get("candidatesTokenCount", 0),
                "total": usage_data.get("totalTokenCount", 0),
                "thoughts": usage_data.get("thoughtsTokenCount", 0),
            },
        )

    @staticmethod
    def _calculate_cost(usage: dict) -> float:
        """Cost for generateContent (flash-lite tier)."""
        cached = usage.get("cachedContentTokenCount", 0)
        billable_input = usage.get("promptTokenCount", 0) - cached
        input_cost = (billable_input / 1_000_000) * 0.15
        cached_cost = (cached / 1_000_000) * 0.0375
        output_cost = (usage.get("candidatesTokenCount", 0) / 1_000_000) * 0.60
        thought_cost = (usage.get("thoughtsTokenCount", 0) / 1_000_000) * 0.60
        return input_cost + cached_cost + output_cost + thought_cost

    @staticmethod
    def _calculate_deep_cost(usage: dict) -> float:
        """Cost for deep research (2.5-pro tier, context-aware)."""
        cached = usage.get("cachedContentTokenCount", 0)
        billable_input = usage.get("promptTokenCount", 0) - cached
        is_large = usage.get("promptTokenCount", 0) > 200_000
        input_rate = 4.0 if is_large else 2.0
        cached_rate = 0.4 if is_large else 0.2
        output_rate = 18.0 if is_large else 12.0
        thought_rate = 18.0 if is_large else 12.0
        input_cost = (billable_input / 1_000_000) * input_rate
        cached_cost = (cached / 1_000_000) * cached_rate
        output_cost = (usage.get("candidatesTokenCount", 0) / 1_000_000) * output_rate
        thought_cost = (usage.get("thoughtsTokenCount", 0) / 1_000_000) * thought_rate
        return input_cost + cached_cost + output_cost + thought_cost

    def estimate_cost(self, query: str) -> float:
        return 3.0 if self._deep else 0.15

    def is_available(self) -> bool:
        return bool(self.api_key)
