"""Extract claims and sources from provider results."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from urllib.parse import urlparse

from parallect.backends import (
    OPENAI_COMPAT_BACKENDS,
    BackendSpec,
    resolve_synthesis_backend,
)
from parallect.backends.adapters import (
    call_anthropic_chat,
    call_gemini_chat,
    call_openai_compat_chat,
)
from parallect.providers import ProviderResult

logger = logging.getLogger("parallect")


def _clean_json(raw: str) -> str:
    """Strip markdown fences and whitespace so json.loads succeeds."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


async def _dispatch(spec: BackendSpec, system_prompt: str, user_prompt: str) -> str:
    """Route a system+user chat call to the right backend adapter."""
    if spec.kind == "anthropic":
        result = await call_anthropic_chat(spec, user_prompt, system_prompt)
    elif spec.kind == "gemini":
        result = await call_gemini_chat(spec, user_prompt, system_prompt)
    elif spec.kind in OPENAI_COMPAT_BACKENDS:
        result = await call_openai_compat_chat(spec, user_prompt, system_prompt)
    else:
        raise ValueError(f"Unsupported backend for extraction: {spec.kind}")
    return result.get("content", "")

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
    model: str | None = None,
    *,
    base_url: str | None = None,
    settings: object | None = None,
) -> list[dict]:
    """Extract claims from provider results using an LLM.

    The extraction model follows the same resolution as synthesis: CLI overrides,
    then `[synthesis]` in config, then defaults. Local backends (lmstudio,
    ollama) work without an API key.

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
        cli_model = model if model and "/" not in model else None
        spec = resolve_synthesis_backend(
            cli_base_url=base_url,
            cli_model=cli_model,
            settings=settings,
        )
        if api_key:
            spec = BackendSpec(
                kind=spec.kind,
                base_url=spec.base_url,
                api_key=api_key,
                model=spec.model,
                api_key_env=spec.api_key_env,
            )
        raw = await _dispatch(spec, CLAIMS_SYSTEM_PROMPT, prompt)
        data = json.loads(_clean_json(raw))
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
    model: str | None = None,
    *,
    base_url: str | None = None,
    settings: object | None = None,
) -> list[dict]:
    """Generate follow-on research questions from the synthesis.

    Uses the same backend resolution as synthesis. Returns a list of dicts
    compatible with prx_spec FollowOn.
    """
    if not synthesis_md:
        return []

    prompt = (
        f"# Original Research Query\n\n{query}\n\n"
        f"# Synthesis Report\n\n{synthesis_md}\n\n"
        f"# Task\n\nSuggest follow-on research questions."
    )

    try:
        cli_model = model if model and "/" not in model else None
        spec = resolve_synthesis_backend(
            cli_base_url=base_url,
            cli_model=cli_model,
            settings=settings,
        )
        if api_key:
            spec = BackendSpec(
                kind=spec.kind,
                base_url=spec.base_url,
                api_key=api_key,
                model=spec.model,
                api_key_env=spec.api_key_env,
            )
        raw = await _dispatch(spec, FOLLOW_ONS_SYSTEM_PROMPT, prompt)
        data = json.loads(_clean_json(raw))
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


