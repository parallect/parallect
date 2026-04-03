"""Base adapter for any OpenAI-compatible API endpoint."""

from __future__ import annotations

import time

import json

import httpx

from parallect.providers import ProviderResult
from parallect.providers.hash_response import attach_response_hash


class OpenAICompatibleProvider:
    """Provider adapter for OpenAI-compatible chat completion APIs.

    Works with any server that implements the /v1/chat/completions endpoint
    (Ollama, LM Studio, vLLM, LocalAI, etc).
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        system_prompt: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or (
            "You are a research assistant. Provide a thorough, well-structured "
            "research report in markdown format with citations where possible."
        )
        self.timeout = timeout

    @property
    def name(self) -> str:
        return self._name

    async def research(self, query: str) -> ProviderResult:
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": query},
                        ],
                    },
                )
                response.raise_for_status()
                raw_text = response.text
                data = json.loads(raw_text)

            duration = time.monotonic() - start
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})

            result = ProviderResult(
                provider=self._name,
                status="completed",
                report_markdown=content,
                model=data.get("model", self.model),
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
                provider=self._name,
                status="failed",
                error=str(e),
                duration_seconds=round(duration, 2),
            )

    def estimate_cost(self, query: str) -> float:
        return 0.0  # Local models are free

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.base_url}/v1/models")
                return r.status_code == 200
        except Exception:
            return False
