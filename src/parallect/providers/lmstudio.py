"""LM Studio provider (wraps OpenAI-compatible endpoint)."""

from __future__ import annotations

from parallect.providers.openai_compat import OpenAICompatibleProvider

DEFAULT_HOST = "http://localhost:1234"
DEFAULT_MODEL = "default"


class LMStudioProvider(OpenAICompatibleProvider):
    """Adapter for LM Studio's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(
            name="lmstudio",
            base_url=host,
            model=model,
            api_key="lm-studio",
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return "lmstudio"
