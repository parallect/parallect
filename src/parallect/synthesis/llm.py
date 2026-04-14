"""LLM-based synthesis of multiple provider reports."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from parallect.providers import ProviderResult

SYNTHESIS_SYSTEM_PROMPT = """\
You are a research synthesis expert. You will receive multiple research reports \
on the same topic from different AI providers. Your task is to produce a single \
unified report that:

1. Identifies key agreements across providers
2. Highlights disagreements or contradictions
3. Synthesizes the strongest arguments and evidence
4. Provides a balanced, comprehensive overview
5. Uses markdown formatting with clear sections

Do not attribute statements to specific providers. Write as a unified report. \
If providers disagree on a point, present both perspectives."""


@dataclass
class SynthesisResult:
    """Result of synthesis."""

    report_markdown: str
    model: str | None = None
    cost_usd: float | None = None
    duration_seconds: float | None = None
    tokens: dict | None = None


def _build_synthesis_prompt(query: str, results: list[ProviderResult]) -> str:
    """Build the synthesis prompt from provider results."""
    sections = []
    for r in results:
        sections.append(f"### Report from {r.provider}\n\n{r.report_markdown}")

    reports_text = "\n\n---\n\n".join(sections)

    return (
        f"# Research Query\n\n{query}\n\n"
        f"# Provider Reports\n\n{reports_text}\n\n"
        f"# Task\n\n"
        f"Synthesize the above reports into a single unified research report."
    )


async def synthesize(
    query: str,
    provider_results: list[ProviderResult],
    model: str = "anthropic",
    api_key: str | None = None,
) -> SynthesisResult:
    """Produce a unified synthesis report from multiple provider reports.

    Currently supports Anthropic and OpenAI-compatible endpoints for synthesis.
    """
    if not provider_results:
        return SynthesisResult(report_markdown="No provider results to synthesize.")

    prompt = _build_synthesis_prompt(query, provider_results)
    start = time.monotonic()

    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
        "grok": "grok-3",
        "perplexity": "sonar",
    }

    if model.startswith("anthropic") or model == "anthropic":
        result = await _synthesize_anthropic(prompt, api_key)
    elif model.startswith("ollama"):
        model_name = model.split("/", 1)[1] if "/" in model else "llama3.2"
        result = await _synthesize_ollama(prompt, model_name)
    elif model.startswith("lmstudio"):
        model_name = model.split("/", 1)[1] if "/" in model else "default"
        result = await _synthesize_lmstudio(prompt, model_name)
    else:
        resolved_model = DEFAULT_MODELS.get(model, model)
        result = await _synthesize_openai_compat(prompt, resolved_model, api_key)

    result.duration_seconds = round(time.monotonic() - start, 2)
    return result


async def _synthesize_anthropic(
    prompt: str, api_key: str | None = None
) -> SynthesisResult:
    """Synthesize using Anthropic Claude."""
    import os

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("Anthropic API key required for synthesis")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 8192,
                "system": SYNTHESIS_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

    content_blocks = data.get("content", [])
    content = "\n".join(
        block["text"] for block in content_blocks if block.get("type") == "text"
    )
    usage = data.get("usage", {})

    return SynthesisResult(
        report_markdown=content,
        model=data.get("model"),
        cost_usd=0.03,  # estimate
        tokens={
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "total": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    )


async def _synthesize_ollama(prompt: str, model: str) -> SynthesisResult:
    """Synthesize using a local Ollama model."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    return SynthesisResult(report_markdown=content, model=model)


async def _synthesize_lmstudio(prompt: str, model: str) -> SynthesisResult:
    """Synthesize using a local LM Studio model."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Authorization": "Bearer lm-studio"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    return SynthesisResult(report_markdown=content, model=model)


async def _synthesize_openai_compat(
    prompt: str, model: str, api_key: str | None = None
) -> SynthesisResult:
    """Synthesize using any OpenAI-compatible endpoint."""
    import os

    key = api_key or os.environ.get("OPENAI_API_KEY", "")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return SynthesisResult(
        report_markdown=content,
        model=data.get("model", model),
        cost_usd=0.05,
        tokens={
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
        },
    )
