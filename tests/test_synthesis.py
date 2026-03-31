"""Tests for synthesis engines."""

from __future__ import annotations

from parallect.providers import ProviderResult
from parallect.synthesis.concat import concatenate


class TestConcatenate:
    def test_concatenate_basic(self):
        results = [
            ProviderResult(
                provider="provider_a",
                status="completed",
                report_markdown="Report from A.",
            ),
            ProviderResult(
                provider="provider_b",
                status="completed",
                report_markdown="Report from B.",
            ),
        ]
        synth = concatenate("test query", results)

        assert "# test query" in synth.report_markdown
        assert "## provider_a" in synth.report_markdown
        assert "Report from A." in synth.report_markdown
        assert "## provider_b" in synth.report_markdown
        assert "Report from B." in synth.report_markdown
        assert synth.model is None

    def test_concatenate_empty(self):
        synth = concatenate("test", [])
        assert "# test" in synth.report_markdown

    def test_concatenate_skips_empty_reports(self):
        results = [
            ProviderResult(provider="a", status="completed", report_markdown="Content."),
            ProviderResult(provider="b", status="failed", report_markdown=""),
        ]
        synth = concatenate("test", results)
        assert "## a" in synth.report_markdown
        assert "## b" not in synth.report_markdown
