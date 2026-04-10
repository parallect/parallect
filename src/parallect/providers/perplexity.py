"""Perplexity Sonar Deep Research provider."""

from __future__ import annotations

import time

import json

import httpx

from parallect.providers import ProviderResult
from parallect.providers.hash_response import attach_response_hash

PERPLEXITY_API_URL = "https://api.perplexity.ai"
DEFAULT_MODEL = "sonar-deep-research"
COST_PER_QUERY_ESTIMATE = 0.05


class PerplexityProvider:
    """Adapter for the Perplexity Sonar API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "perplexity"

    async def research(self, query: str) -> ProviderResult:
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{PERPLEXITY_API_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a research assistant. Provide a thorough, "
                                    "well-sourced research report."
                                ),
                            },
                            {"role": "user", "content": query},
                        ],
                    },
                )
                response.raise_for_status()
                raw_text = response.text
                data = json.loads(raw_text)

            duration = time.monotonic() - start
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            citations = []
            for i, url in enumerate(data.get("citations", []), 1):
                citations.append({"index": i, "url": url})

            result = ProviderResult(
                provider="perplexity",
                status="completed",
                report_markdown=content,
                citations=citations,
                model=data.get("model", self.model),
                cost_usd=COST_PER_QUERY_ESTIMATE,
                duration_seconds=round(duration, 2),
                tokens={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                },
            )
            return attach_response_hash(result, raw_text)
        except Exception as e:
            duration = time.monotonic() - start
            return ProviderResult(
                provider="perplexity",
                status="failed",
                error=str(e),
                duration_seconds=round(duration, 2),
            )

    def estimate_cost(self, query: str) -> float:
        return COST_PER_QUERY_ESTIMATE

    def is_available(self) -> bool:
        return bool(self.api_key)
