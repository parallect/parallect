"""Parallel fan-out orchestrator."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from prx_spec import BundleData, ProviderData, write_bundle
from prx_spec.models import Manifest, Producer, ProviderBreakdown
from prx_spec.models.provider import Citation, ProviderMeta, TokenUsage
from prx_spec.models.synthesis import SynthesisMeta

from parallect.providers import ProviderResult
from parallect.providers.base import AsyncResearchProvider


def _try_sign_bundle(
    bundle: BundleData,
    settings: object,
    input_report_hashes: dict[str, str] | None = None,
) -> BundleData:
    """Attempt to sign bundle attestations using the local key.

    Creates a bundle-level attestation, signs it, and attaches it to the bundle.
    Returns the bundle (mutated with attestations) if signing succeeds,
    or the original bundle if no key is available.
    """
    from prx_spec.attestation.keys import load_private_key, get_key_id, DEFAULT_KEY_DIR
    from prx_spec.attestation.signing import sign_attestation, compute_file_hash
    from prx_spec.models.attestation_models import Attestation, Signer, Subject

    key_path_str = getattr(settings, "key_path", "")
    identity = getattr(settings, "identity", "") or "anonymous"
    key_dir = Path(key_path_str) if key_path_str else DEFAULT_KEY_DIR

    try:
        signing_key = load_private_key(key_dir)
    except Exception:
        return bundle

    verify_key = signing_key.verify_key
    key_id = get_key_id(verify_key)

    # Build a bundle-level attestation
    manifest_json = bundle.manifest.model_dump_json(exclude_none=True)
    manifest_hash = compute_file_hash(manifest_json.encode("utf-8"))

    from prx_spec.models.attestation_models import AttestationContext

    context = AttestationContext(
        manifest_sha256=manifest_hash,
        input_report_hashes=input_report_hashes if input_report_hashes else None,
    )

    attestation = Attestation(
        version="1.0",
        type="bundle",
        signer=Signer(
            type="researcher",
            identity=identity,
            key_id=key_id,
            public_key_url=f"local://{key_dir / 'prx_signing.pub'}",
        ),
        subject=Subject(
            file="manifest.json",
            sha256=manifest_hash,
        ),
        context=context,
    )

    signed = sign_attestation(attestation, signing_key)
    bundle.attestations[f"bundle.researcher.{key_id}.sig.json"] = signed
    return bundle


def _default_extraction_model(settings: object | None) -> str:
    """Resolve a sensible model string for claims/follow-on extraction when no
    explicit model is supplied. Reads `[synthesis]` config first so local
    backends (lmstudio, ollama) work BYOK-free."""
    if settings is not None:
        backend = getattr(settings, "synthesis_backend", "") or ""
        model = getattr(settings, "synthesis_model", "") or ""
        if backend and model:
            return f"{backend}/{model}"
        if backend:
            return backend
    return "anthropic"


def _resolve_synth_key(synthesize_with: str, settings: object | None) -> str | None:
    """Pick the right API key for the synthesis provider from settings."""
    if settings is None:
        return None
    key_map = {
        "anthropic": "anthropic_api_key",
        "openai": "openai_api_key",
        "gemini": "google_api_key",
        "grok": "xai_api_key",
        "perplexity": "perplexity_api_key",
    }
    for prefix, attr in key_map.items():
        if synthesize_with.startswith(prefix) or synthesize_with == prefix:
            return getattr(settings, attr, None) or None
    # Local backends don't need a real key
    if any(synthesize_with.startswith(p) for p in ("lmstudio", "ollama")):
        return "not-needed"
    return None


@dataclass
class ProviderOutcome:
    """Outcome of a single provider call (success or failure)."""

    provider: str
    result: ProviderResult | None = None
    error: str | None = None


async def fan_out(
    query: str,
    providers: list[AsyncResearchProvider],
    timeout_per_provider: float = 120.0,
) -> list[ProviderOutcome]:
    """Fan out query to all providers in parallel.

    Partial failures are captured, not raised. Each provider gets its own
    timeout. One provider failing does not cancel the others.
    """

    async def run_one(provider: AsyncResearchProvider) -> ProviderOutcome:
        try:
            async with asyncio.timeout(timeout_per_provider):
                result = await provider.research(query)
                return ProviderOutcome(provider=provider.name, result=result)
        except TimeoutError:
            return ProviderOutcome(
                provider=provider.name,
                error=f"Timed out after {timeout_per_provider}s",
            )
        except Exception as e:
            return ProviderOutcome(provider=provider.name, error=str(e))

    outcomes = await asyncio.gather(
        *[run_one(p) for p in providers],
        return_exceptions=False,
    )
    return list(outcomes)


async def research(
    query: str,
    providers: list[AsyncResearchProvider],
    synthesize_with: str | None = "anthropic",
    synthesis_base_url: str | None = None,
    extract_claims_flag: bool = True,
    budget_cap_usd: float | None = None,
    timeout_per_provider: float = 120.0,
    output: str | Path | None = None,
    no_synthesis: bool = False,
    parent_bundle_id: str | None = None,
    parent_context: str | None = None,
    no_sign: bool = False,
    settings: object | None = None,
    sources: str | None = None,
) -> BundleData:
    """High-level research: fan out, collect, optionally synthesize, write .prx.

    This is the primary entry point for both the CLI and programmatic use.
    """
    import time as _time

    from parallect.orchestrator.budget import BudgetEstimator
    from parallect.plugins import PluginManager

    run_start = _time.monotonic()

    # Initialize plugins
    plugin_mgr = PluginManager()
    plugin_mgr.discover_entry_points()

    # Budget check
    if budget_cap_usd is not None:
        estimator = BudgetEstimator()
        estimate = estimator.estimate(query, providers)
        estimator.check_cap(estimate, budget_cap_usd)

    # Hook: pre_research (may modify query)
    query = await plugin_mgr.run_pre_research(
        query, [p.name for p in providers]
    )

    # Fan out to all providers (augment query with parent context if continuing)
    effective_query = query
    if parent_context:
        effective_query = (
            f"{query}\n\n---\n"
            f"Context from previous research:\n{parent_context}"
        )

    # Web providers + data-source plugins run concurrently so a slow
    # plugin never blocks the web fan-out (and vice-versa).
    from parallect.orchestrator.plugin_sources import run_plugin_sources

    provider_task = asyncio.create_task(
        fan_out(effective_query, providers, timeout_per_provider)
    )
    plugins_task = asyncio.create_task(
        run_plugin_sources(effective_query, sources, settings=settings)
    )
    outcomes = await provider_task
    plugin_outcomes = await plugins_task

    # Fold plugin results into outcomes as synthetic ProviderOutcome entries.
    for pr in plugin_outcomes:
        if pr.result is None:
            outcomes.append(ProviderOutcome(
                provider=pr.spec.display, error=pr.error or "plugin failed"
            ))
        else:
            outcomes.append(ProviderOutcome(
                provider=pr.spec.display, result=pr.result
            ))

    # Build provider data from successful results
    provider_data: list[ProviderData] = []
    providers_used: list[str] = []
    total_cost = 0.0
    total_duration = 0.0

    failed_outcomes: list[ProviderOutcome] = []
    input_report_hashes: dict[str, str] = {}

    for outcome in outcomes:
        if outcome.error or not outcome.result or outcome.result.status == "failed":
            failed_outcomes.append(outcome)
            continue
        # Hook: post_provider
        outcome.result = await plugin_mgr.run_post_provider(
            outcome.provider, outcome.result
        )
        r = outcome.result

        # Build per-provider citations list (dedupe by canonical URL).
        # Perplexity and others sometimes chunk the same URL across multiple
        # citation indices; deduplicate here so the source registry and any
        # downstream readers see a clean, contiguous 1-based index.
        citations = None
        if r.citations:
            from parallect.synthesis.extract import _canonical_url

            seen: dict[str, dict] = {}
            order: list[str] = []
            for c in r.citations:
                url = c.get("url", "")
                if not url:
                    continue
                canonical = _canonical_url(url)
                if canonical in seen:
                    # Merge: keep first title/snippet, but swap in a strictly
                    # longer snippet from a later chunk of the same source.
                    existing = seen[canonical]
                    later_snippet = c.get("snippet")
                    if (
                        later_snippet
                        and (
                            not existing.get("snippet")
                            or len(later_snippet) > len(existing.get("snippet") or "")
                        )
                    ):
                        existing["snippet"] = later_snippet
                    continue
                seen[canonical] = {
                    "url": url,
                    "title": c.get("title"),
                    "snippet": c.get("snippet"),
                    "domain": c.get("domain"),
                }
                order.append(canonical)

            citations = [
                Citation(
                    index=i + 1,
                    url=seen[canonical]["url"],
                    title=seen[canonical]["title"],
                    snippet=seen[canonical]["snippet"],
                    domain=seen[canonical]["domain"],
                )
                for i, canonical in enumerate(order)
            ]

        # Build per-provider meta
        tokens = None
        if r.tokens:
            tokens = TokenUsage(
                input=r.tokens.get("input", 0),
                output=r.tokens.get("output", 0),
                total=r.tokens.get("total", 0),
            )
        meta = ProviderMeta(
            provider=outcome.provider,
            model=r.model,
            cost_usd=r.cost_usd,
            duration_seconds=r.duration_seconds,
            tokens=tokens,
            status=r.status,  # type: ignore[arg-type]
        )

        pd = ProviderData(
            name=outcome.provider,
            report_md=r.report_markdown,
            citations=citations or None,
            meta=meta,
        )
        provider_data.append(pd)
        providers_used.append(outcome.provider)
        if r.cost_usd:
            total_cost += r.cost_usd
        if r.duration_seconds:
            total_duration = max(total_duration, r.duration_seconds)
        if r.response_hash:
            input_report_hashes[outcome.provider] = r.response_hash

    if not providers_used:
        error_lines: list[str] = []
        for o in failed_outcomes:
            err = o.error or (o.result.error if o.result else None) or "unknown error"
            error_lines.append(f"  • {o.provider}: {err}")
        raise RuntimeError(
            "All providers failed — no results to bundle.\n" + "\n".join(error_lines)
        )

    # Build bundle
    bundle_id = f"prx_{secrets.token_hex(4)}"
    now = datetime.now(timezone.utc).isoformat()

    has_synthesis = False
    synthesis_md = None
    synthesis_meta = None

    # Synthesize if requested and we have results
    if not no_synthesis and synthesize_with and provider_data:
        try:
            from parallect.synthesis.llm import synthesize

            synth_api_key = _resolve_synth_key(synthesize_with, settings)
            results = [
                o.result for o in outcomes if o.result and o.result.status != "failed"
            ]
            synth_result = await synthesize(
                query,
                results,
                model=synthesize_with,
                api_key=synth_api_key,
                base_url=synthesis_base_url,
                settings=settings,
            )
            synthesis_md = synth_result.report_markdown
            has_synthesis = True
            synthesis_meta = SynthesisMeta(
                provider=synthesize_with,
                model=synth_result.model,
                cost_usd=synth_result.cost_usd,
                duration_seconds=synth_result.duration_seconds,
                tokens=synth_result.tokens,
            )
            if synth_result.cost_usd:
                total_cost += synth_result.cost_usd
            # Hook: post_synthesis
            synthesis_md = await plugin_mgr.run_post_synthesis(synthesis_md)
        except Exception as exc:
            import logging
            logging.getLogger("parallect").warning("Synthesis failed: %s", exc)

    # Extract claims from provider results
    claims_file = None
    if extract_claims_flag and provider_data:
        try:
            from parallect.synthesis.extract import extract_claims
            from prx_spec.models.synthesis import BasicClaim, ClaimsFile

            # Claims extraction reuses the same backend resolution as synthesis:
            # explicit model > [synthesis] config in settings > default. Never
            # falls back to hardcoded anthropic, so local backends work BYOK-free.
            extraction_model = synthesize_with or _default_extraction_model(settings)
            synth_api_key = _resolve_synth_key(extraction_model, settings)
            successful_results = [
                o.result for o in outcomes if o.result and o.result.status != "failed"
            ]
            raw_claims = await extract_claims(
                query, successful_results,
                api_key=synth_api_key,
                model=synthesize_with,
                base_url=synthesis_base_url,
                settings=settings,
            )
            if raw_claims:
                parsed_claims = [
                    BasicClaim(
                        id=c.get("id", f"claim_{i:03d}"),
                        content=c["content"],
                        providers_supporting=c.get("providers_supporting", []),
                        providers_contradicting=c.get("providers_contradicting", []),
                        category=c.get("category"),
                    )
                    for i, c in enumerate(raw_claims)
                    if c.get("content")
                ]
                if parsed_claims:
                    claims_file = ClaimsFile(
                        extraction_model=extraction_model,
                        extraction_version="0.1.0",
                        claims=parsed_claims,
                    )
        except Exception as exc:
            import logging
            logging.getLogger("parallect").warning("Claims extraction failed: %s", exc)

    # Extract follow-on research questions
    follow_ons = None
    if has_synthesis and synthesis_md:
        try:
            from parallect.synthesis.extract import extract_follow_ons
            from prx_spec.models.synthesis import FollowOn

            extraction_model = synthesize_with or _default_extraction_model(settings)
            synth_api_key = _resolve_synth_key(extraction_model, settings)
            raw_follow_ons = await extract_follow_ons(
                query, synthesis_md,
                api_key=synth_api_key,
                model=synthesize_with,
                base_url=synthesis_base_url,
                settings=settings,
            )
            if raw_follow_ons:
                follow_ons = [
                    FollowOn(
                        query=fo["query"],
                        rationale=fo.get("rationale"),
                        estimated_providers=fo.get("estimated_providers", []),
                        related_claims=fo.get("related_claims", []),
                    )
                    for fo in raw_follow_ons
                    if fo.get("query")
                ]
        except Exception as exc:
            import logging
            logging.getLogger("parallect").warning("Follow-on extraction failed: %s", exc)

    # Build source registry from provider citations
    sources_registry = None
    if provider_data:
        try:
            from parallect.synthesis.extract import extract_sources
            from prx_spec.models.sources import Source, SourcesRegistry

            successful_results = [
                o.result for o in outcomes if o.result and o.result.status != "failed"
            ]
            raw_sources = extract_sources(successful_results)
            if raw_sources:
                parsed_sources = [Source(**s) for s in raw_sources]
                sources_registry = SourcesRegistry(sources=parsed_sources)
        except Exception as exc:
            import logging
            logging.getLogger("parallect").warning("Source extraction failed: %s", exc)

    provider_breakdown = [
        ProviderBreakdown(
            provider=o.provider,
            status=(o.result.status if o.result else "failed"),
            duration_seconds=(o.result.duration_seconds if o.result else None),
            cost_usd=(o.result.cost_usd if o.result else None),
        )
        for o in outcomes
    ]

    run_duration = round(_time.monotonic() - run_start, 2)

    manifest = Manifest(
        spec_version="1.1",
        id=bundle_id,
        query=query,
        created_at=now,
        producer=Producer(name="parallect-oss", version="0.1.0"),
        providers_used=providers_used,
        provider_breakdown=provider_breakdown,
        has_synthesis=has_synthesis,
        has_claims=claims_file is not None,
        has_sources=sources_registry is not None,
        has_evidence_graph=False,
        has_follow_ons=bool(follow_ons),
        total_cost_usd=round(total_cost, 4) if total_cost else None,
        total_duration_seconds=run_duration,
        parent_bundle_id=parent_bundle_id,
    )

    bundle = BundleData(
        manifest=manifest,
        query_md=f"# Research Query\n\n{query}",
        providers=provider_data,
        synthesis_md=synthesis_md,
        synthesis_meta=synthesis_meta,
        claims=claims_file,
        follow_ons=follow_ons,
        sources=sources_registry,
    )

    # Hook: post_bundle
    bundle = await plugin_mgr.run_post_bundle(bundle)

    # Auto-sign if settings allow and key is available
    if not no_sign and settings and getattr(settings, "auto_sign", False):
        bundle = _try_sign_bundle(bundle, settings, input_report_hashes)
        # The signer mutates bundle.attestations; flip the manifest flag to match.
        if bundle.attestations:
            bundle.manifest = bundle.manifest.model_copy(update={"has_attestations": True})

    # Write to disk if output path given
    if output:
        output_path = Path(output)
        if output_path.is_dir():
            output_path = output_path / f"{bundle_id}.prx"
        write_bundle(bundle, output_path)

    return bundle
