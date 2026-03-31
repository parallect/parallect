"""No-LLM concatenation mode for synthesis."""

from __future__ import annotations

from parallect.providers import ProviderResult
from parallect.synthesis.llm import SynthesisResult


def concatenate(
    query: str,
    provider_results: list[ProviderResult],
) -> SynthesisResult:
    """No-LLM mode: concatenate reports with section headers.

    Produces a synthesis report by joining all provider reports
    under headings, with no model calls.
    """
    sections = [
        f"## {r.provider}\n\n{r.report_markdown}" for r in provider_results if r.report_markdown
    ]
    report = f"# {query}\n\n" + "\n\n---\n\n".join(sections)
    return SynthesisResult(report_markdown=report, model=None)
