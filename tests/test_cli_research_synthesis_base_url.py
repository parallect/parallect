"""Tests that the `--synthesis-base-url` CLI flag reaches the orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from parallect.cli import app
from prx_spec import BundleData
from prx_spec.models import Manifest, Producer


def _fake_bundle() -> BundleData:
    return BundleData(
        manifest=Manifest(
            spec_version="1.0",
            id="prx_abcd1234",
            query="q",
            created_at="2026-01-01T00:00:00Z",
            producer=Producer(name="parallect-oss", version="0.1.0"),
            providers_used=["stub"],
            has_synthesis=False,
            has_claims=False,
            has_sources=False,
            has_evidence_graph=False,
            has_follow_ons=False,
        ),
        query_md="# q",
        providers=[],
    )


class TestSynthesisBaseUrlFlag:
    def test_flag_threaded_to_research(self, tmp_path, monkeypatch):
        runner = CliRunner()

        # The CLI's _resolve_providers path needs at least one "available"
        # provider or it exits. Stub the provider resolver.
        class _StubProvider:
            name = "stub"
            def is_available(self):
                return True
            async def research(self, q):  # pragma: no cover - not used
                return None

        monkeypatch.setattr(
            "parallect.cli.research._resolve_providers",
            lambda *a, **kw: [_StubProvider()],
        )
        # Force BYOK route.
        monkeypatch.delenv("PARALLECT_API_KEY", raising=False)

        captured = {}

        async def fake_research(*args, **kwargs):
            captured.update(kwargs)
            return _fake_bundle()

        monkeypatch.setattr("parallect.orchestrator.parallel.research", fake_research)

        result = runner.invoke(
            app,
            [
                "research",
                "test query",
                "--local",
                "--synthesis-base-url",
                "http://localhost:4000/v1",
                "--no-sign",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured.get("synthesis_base_url") == "http://localhost:4000/v1"

    def test_env_var_also_threads(self, tmp_path, monkeypatch):
        runner = CliRunner()

        class _StubProvider:
            name = "stub"
            def is_available(self):
                return True
            async def research(self, q):  # pragma: no cover
                return None

        monkeypatch.setattr(
            "parallect.cli.research._resolve_providers",
            lambda *a, **kw: [_StubProvider()],
        )
        monkeypatch.delenv("PARALLECT_API_KEY", raising=False)
        monkeypatch.setenv("PARALLECT_SYNTHESIS_BASE_URL", "http://envllm:1/v1")

        captured = {}

        async def fake_research(*args, **kwargs):
            captured.update(kwargs)
            return _fake_bundle()

        monkeypatch.setattr("parallect.orchestrator.parallel.research", fake_research)

        result = runner.invoke(
            app,
            [
                "research",
                "test query",
                "--local",
                "--no-sign",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0, result.output
        # Typer's envvar= binding surfaces the env var as the flag value.
        assert captured.get("synthesis_base_url") == "http://envllm:1/v1"
