"""HTTP adapters for each backend family.

Three adapter functions, all sharing the same response-shape contract:

  - `call_openai_compat_chat(spec, prompt, system_prompt) -> dict`
  - `call_anthropic_chat(spec, prompt, system_prompt) -> dict`
  - `call_gemini_chat(spec, prompt, system_prompt) -> dict`

Each returns a normalized result:
  {
    "content": str,               # the markdown/text body
    "model": str | None,
    "tokens": {"input": int, "output": int, "total": int} | None,
    "raw": dict,                  # the full upstream JSON
  }

Synthesis + embeddings callers consume this and build their own dataclasses.
"""

from __future__ import annotations

from typing import Any

import httpx

from parallect.backends import BackendSpec


# ---------------------------------------------------------------------------
# Shared error handling
# ---------------------------------------------------------------------------


class BackendError(RuntimeError):
    """Raised when an upstream backend returns a non-2xx response."""

    def __init__(self, status: int, message: str, *, backend: str) -> None:
        super().__init__(f"[{backend} HTTP {status}] {message}")
        self.status = status
        self.backend = backend


def _raise_for_status(response: httpx.Response, *, backend: str) -> None:
    if response.status_code >= 400:
        try:
            body = response.text[:500]
        except Exception:
            body = "<no body>"
        raise BackendError(response.status_code, body, backend=backend)


# ---------------------------------------------------------------------------
# OpenAI-compatible chat adapter
# ---------------------------------------------------------------------------


async def call_openai_compat_chat(
    spec: BackendSpec,
    prompt: str,
    system_prompt: str,
    *,
    timeout: float = 120.0,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST /chat/completions on any OpenAI-compatible server.

    Works for OpenAI, OpenRouter, LiteLLM, vLLM, Ollama, LM Studio, and any
    user-supplied `custom` endpoint. The caller is responsible for picking
    the right model name for the backend.
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if spec.api_key:
        headers["Authorization"] = f"Bearer {spec.api_key}"
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{spec.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        _raise_for_status(response, backend=spec.kind)
        data = response.json()

    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content", "")
    usage = data.get("usage") or {}
    tokens = {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }

    return {
        "content": content,
        "model": data.get("model", spec.model),
        "tokens": tokens,
        "raw": data,
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible embeddings adapter
# ---------------------------------------------------------------------------


async def call_openai_compat_embeddings(
    spec: BackendSpec,
    texts: list[str],
    *,
    timeout: float = 60.0,
    extra_headers: dict[str, str] | None = None,
) -> list[list[float]]:
    """POST /embeddings on any OpenAI-compatible server.

    Most servers accept a list of strings. Some (e.g. some vLLM builds) only
    accept a single string per request -- we batch 1:1 in that case.
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if spec.api_key:
        headers["Authorization"] = f"Bearer {spec.api_key}"
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": spec.model,
        "input": texts,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{spec.base_url}/embeddings",
            headers=headers,
            json=payload,
        )
        _raise_for_status(response, backend=spec.kind)
        data = response.json()

    items = data.get("data") or []
    # Keep the original order (api docs guarantee this, but be defensive).
    items_sorted = sorted(items, key=lambda e: e.get("index", 0))
    return [item.get("embedding", []) for item in items_sorted]


# ---------------------------------------------------------------------------
# Anthropic chat adapter (synthesis only)
# ---------------------------------------------------------------------------


async def call_anthropic_chat(
    spec: BackendSpec,
    prompt: str,
    system_prompt: str,
    *,
    timeout: float = 120.0,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    if not spec.api_key:
        raise ValueError(
            "Anthropic API key required for synthesis. "
            f"Set {spec.api_key_env or 'ANTHROPIC_API_KEY'}."
        )

    headers = {
        "x-api-key": spec.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": spec.model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{spec.base_url}/messages",
            headers=headers,
            json=payload,
        )
        _raise_for_status(response, backend="anthropic")
        data = response.json()

    blocks = data.get("content", [])
    content = "\n".join(
        b["text"] for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    )
    usage = data.get("usage") or {}
    tokens = {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "total": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }

    return {
        "content": content,
        "model": data.get("model", spec.model),
        "tokens": tokens,
        "raw": data,
    }


# ---------------------------------------------------------------------------
# Gemini chat adapter (synthesis only)
# ---------------------------------------------------------------------------


async def call_gemini_chat(
    spec: BackendSpec,
    prompt: str,
    system_prompt: str,
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    if not spec.api_key:
        raise ValueError(
            "Gemini backend requires an API key. "
            f"Set {spec.api_key_env or 'GOOGLE_API_KEY'}."
        )

    # Gemini uses the ?key= query param, not a bearer header.
    url = (
        f"{spec.base_url}/models/{spec.model}:generateContent?key={spec.api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        _raise_for_status(response, backend="gemini")
        data = response.json()

    candidates = data.get("candidates") or []
    content = ""
    if candidates:
        parts = (candidates[0].get("content") or {}).get("parts") or []
        content = "".join(p.get("text", "") for p in parts)

    usage = data.get("usageMetadata") or {}
    tokens = {
        "input": usage.get("promptTokenCount", 0),
        "output": usage.get("candidatesTokenCount", 0),
        "total": usage.get("totalTokenCount", 0),
    }

    return {
        "content": content,
        "model": spec.model,
        "tokens": tokens,
        "raw": data,
    }


# ---------------------------------------------------------------------------
# Gemini embeddings adapter
# ---------------------------------------------------------------------------


async def call_gemini_embeddings(
    spec: BackendSpec,
    texts: list[str],
    *,
    timeout: float = 60.0,
) -> list[list[float]]:
    if not spec.api_key:
        raise ValueError(
            "Gemini backend requires an API key. "
            f"Set {spec.api_key_env or 'GOOGLE_API_KEY'}."
        )

    # Gemini's batch endpoint is `:batchEmbedContents`.
    url = (
        f"{spec.base_url}/models/{spec.model}:batchEmbedContents?key={spec.api_key}"
    )
    payload = {
        "requests": [
            {
                "model": f"models/{spec.model}",
                "content": {"parts": [{"text": t}]},
            }
            for t in texts
        ],
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        _raise_for_status(response, backend="gemini")
        data = response.json()

    vectors = []
    for entry in data.get("embeddings", []):
        vectors.append(entry.get("values", []))
    return vectors
