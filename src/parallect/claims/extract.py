"""Basic claim extraction using LLM structured output."""

from __future__ import annotations

import json
import os

import httpx

from parallect.providers import ProviderResult

CLAIM_EXTRACTION_PROMPT = """\
Extract atomic, verifiable claims from the following research synthesis. \
For each claim, identify which providers support it and which contradict it.

Return a JSON object with this exact structure:
{
  "claims": [
    {
      "id": "claim_001",
      "content": "The specific claim text",
      "providers_supporting": ["provider1", "provider2"],
      "providers_contradicting": [],
      "category": "fact|comparison|prediction|methodology|opinion"
    }
  ]
}

Provider names in the reports: %s

Synthesis report:
%s
"""


async def extract_claims(
    synthesis_markdown: str,
    provider_results: list[ProviderResult],
    model: str = "anthropic",
    api_key: str | None = None,
) -> list[dict]:
    """Extract atomic claims from a synthesis report.

    Returns a list of claim dicts with id, content, providers_supporting,
    providers_contradicting, and category fields.
    """
    provider_names = [r.provider for r in provider_results if r.status != "failed"]
    prompt = CLAIM_EXTRACTION_PROMPT % (
        ", ".join(provider_names),
        synthesis_markdown,
    )

    if model.startswith("anthropic") or model == "anthropic":
        claims_json = await _extract_anthropic(prompt, api_key)
    else:
        claims_json = await _extract_openai(prompt, model, api_key)

    return claims_json.get("claims", [])


async def _extract_anthropic(prompt: str, api_key: str | None = None) -> dict:
    """Extract claims using Anthropic Claude."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("Anthropic API key required for claim extraction")

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
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

    content = data["content"][0]["text"]
    # Extract JSON from response (may be wrapped in markdown code block)
    return _parse_json_response(content)


async def _extract_openai(prompt: str, model: str, api_key: str | None = None) -> dict:
    """Extract claims using OpenAI-compatible API."""
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
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    return _parse_json_response(content)


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (code fences)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
    return json.loads(content)
