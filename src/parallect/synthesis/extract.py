"""Extract claims and sources from provider results."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from urllib.parse import urlparse

import httpx

from parallect.providers import ProviderResult

logger = logging.getLogger("parallect")

CLAIMS_SYSTEM_PROMPT = """\
You are a claims extraction expert. Given multiple research reports on the same \
topic, extract atomic factual claims and note which providers support or contradict \
each claim.

Return valid JSON (no markdown fences) with the following structure:
{
  "claims": [
    {
      "id": "claim_001",
      "content": "The factual claim as a single sentence",
      "providers_supporting": ["provider_name"],
      "providers_contradicting": [],
      "category": "one of: fact, statistic, prediction, opinion, methodology"
    }
  ]
}

Extract 10-30 claims covering the key findings. Each claim must be an atomic \
statement (one fact per claim). Focus on substantive findings, not meta-statements \
about the report itself."""


async def extract_claims(
    query: str,
    provider_results: list[ProviderResult],
    api_key: str | None = None,
    model: str = "anthropic",
) -> list[dict]:
    """Extract claims from provider results using an LLM.

    Returns a list of claim dicts compatible with prx_spec BasicClaim.
    """
    if not provider_results:
        return []

    sections = []
    for r in provider_results:
        sections.append(f"### Report from {r.provider}\n\n{r.report_markdown}")
    reports_text = "\n\n---\n\n".join(sections)

    prompt = (
        f"# Research Query\n\n{query}\n\n"
        f"# Provider Reports\n\n{reports_text}\n\n"
        f"# Task\n\nExtract atomic claims from the above reports."
    )

    try:
        if model.startswith("anthropic") or model == "anthropic":
            raw = await _call_anthropic(prompt, CLAIMS_SYSTEM_PROMPT, api_key)
        else:
            raw = await _call_openai_compat(prompt, CLAIMS_SYSTEM_PROMPT, model, api_key)

        data = json.loads(raw)
        return data.get("claims", [])
    except Exception as exc:
        logger.warning("Claims extraction failed: %s", exc)
        return []


FOLLOW_ONS_SYSTEM_PROMPT = """\
You are a research planning expert. Given a research query and the synthesis of \
multiple provider reports, suggest 3-5 follow-on research questions that would \
deepen understanding of the topic.

Return valid JSON (no markdown fences) with the following structure:
{
  "follow_ons": [
    {
      "query": "The follow-on research question",
      "rationale": "Why this follow-on would be valuable",
      "estimated_providers": ["provider1", "provider2"]
    }
  ]
}

Focus on gaps, contradictions, or areas that deserve deeper investigation."""


async def extract_follow_ons(
    query: str,
    synthesis_md: str,
    api_key: str | None = None,
    model: str = "anthropic",
) -> list[dict]:
    """Generate follow-on research questions from the synthesis.

    Returns a list of dicts compatible with prx_spec FollowOn.
    """
    if not synthesis_md:
        return []

    prompt = (
        f"# Original Research Query\n\n{query}\n\n"
        f"# Synthesis Report\n\n{synthesis_md}\n\n"
        f"# Task\n\nSuggest follow-on research questions."
    )

    try:
        if model.startswith("anthropic") or model == "anthropic":
            raw = await _call_anthropic(prompt, FOLLOW_ONS_SYSTEM_PROMPT, api_key)
        else:
            raw = await _call_openai_compat(prompt, FOLLOW_ONS_SYSTEM_PROMPT, model, api_key)

        data = json.loads(raw)
        return data.get("follow_ons", [])
    except Exception as exc:
        logger.warning("Follow-on extraction failed: %s", exc)
        return []


def extract_sources(provider_results: list[ProviderResult]) -> list[dict]:
    """Build a deduplicated source registry from provider citations.

    Parses the citations list from each provider result and deduplicates
    by canonical URL.
    """
    seen: dict[str, dict] = {}

    for result in provider_results:
        if not result.citations:
            continue
        for cite in result.citations:
            url = cite.get("url", "")
            if not url:
                continue
            canonical = _canonical_url(url)
            canonical_id = hashlib.sha256(canonical.encode()).hexdigest()[:12]

            if canonical_id in seen:
                entry = seen[canonical_id]
                if result.provider not in entry["cited_by_providers"]:
                    entry["cited_by_providers"].append(result.provider)
                    entry["citation_count"] += 1
            else:
                parsed = urlparse(url)
                seen[canonical_id] = {
                    "id": f"src_{canonical_id}",
                    "url": url,
                    "url_canonical": canonical,
                    "canonical_id": canonical_id,
                    "title": cite.get("title"),
                    "domain": parsed.netloc,
                    "source_type": "web",
                    "cited_by_providers": [result.provider],
                    "citation_count": 1,
                    "first_seen_provider": result.provider,
                }

    return list(seen.values())


def _canonical_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}".lower()


async def _call_anthropic(prompt: str, system: str, api_key: str | None) -> str:
    import os

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("Anthropic API key required for claims extraction")

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
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()

    blocks = data.get("content", [])
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


async def _call_openai_compat(
    prompt: str, system: str, model: str, api_key: str | None
) -> str:
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
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()

    return data["choices"][0]["message"]["content"]
