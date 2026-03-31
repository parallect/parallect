"""Ollama provider (wraps OpenAI-compatible endpoint)."""

from __future__ import annotations

from parallect.providers.openai_compat import OpenAICompatibleProvider

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


class OllamaProvider(OpenAICompatibleProvider):
    """Adapter for Ollama's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(
            name="ollama",
            base_url=host,
            model=model,
            api_key="ollama",
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        import httpx

        try:
            with httpx.Client(timeout=3.0) as client:
                r = client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False
