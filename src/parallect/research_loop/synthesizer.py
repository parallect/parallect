"""Synthesizer: final synthesis across all iterations.

Delegates to parallect.synthesis.llm.synthesize() so the same backend
resolution, prompts, and cost tracking apply.
"""

from __future__ import annotations

from parallect.providers import ProviderResult
from parallect.synthesis.llm import SynthesisResult, synthesize


async def synthesize_iterations(
    query: str,
    all_results: list[ProviderResult],
    *,
    model: str = "anthropic",
    api_key: str | None = None,
    base_url: str | None = None,
    settings: object | None = None,
) -> SynthesisResult:
    """Produce a final synthesis report from all iterations' results.

    This is a thin wrapper that passes accumulated results to the existing
    synthesis pipeline. The synthesis prompt already handles multiple
    provider reports; iteration boundaries are transparent to it.
    """
    if not all_results:
        return SynthesisResult(report_markdown="No results to synthesize.")

    return await synthesize(
        query,
        all_results,
        model=model,
        api_key=api_key,
        base_url=base_url,
        settings=settings,
    )
