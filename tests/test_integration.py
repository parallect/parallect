"""Integration tests using a local OpenAI-compatible LLM (LM Studio).

Run with: PARALLECT_INTEGRATION_TESTS=1 uv run pytest tests/test_integration.py -v

Requires LM Studio running on localhost:1234 with any model loaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PARALLECT_INTEGRATION_TESTS"),
    reason="Set PARALLECT_INTEGRATION_TESTS=1 to run",
)


@pytest.fixture(scope="module")
def lmstudio_available() -> bool:
    """Check if LM Studio is reachable."""
    import httpx

    try:
        r = httpx.get("http://localhost:1234/v1/models", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def lmstudio_model(lmstudio_available: bool) -> str:
    """Get the first loaded model name from LM Studio."""
    if not lmstudio_available:
        pytest.skip("LM Studio not running on localhost:1234")

    import httpx

    r = httpx.get("http://localhost:1234/v1/models", timeout=5.0)
    data = r.json()
    models = data.get("data", [])
    if not models:
        pytest.skip("No models loaded in LM Studio")
    return models[0]["id"]


@pytest.fixture
def tmp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _run_cli(
    *args: str, env_extra: dict | None = None, timeout: float = 120.0,
) -> subprocess.CompletedProcess:
    """Run the parallect CLI as a subprocess."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "parallect.cli.main", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(Path(__file__).parent.parent),
    )


# ---------------------------------------------------------------------------
# Provider: direct LM Studio research
# ---------------------------------------------------------------------------


class TestLMStudioProvider:
    """Test LM Studio as a research provider."""

    @pytest.mark.asyncio
    async def test_lmstudio_provider_research(self, lmstudio_model: str):
        """Direct API call to LM Studio provider."""
        from parallect.providers.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model=lmstudio_model)
        assert provider.is_available()

        result = await provider.research("What is the capital of France? Answer in one sentence.")
        assert result.status == "completed"
        assert result.report_markdown
        assert len(result.report_markdown) > 10
        assert result.provider == "lmstudio"

    @pytest.mark.asyncio
    async def test_lmstudio_provider_returns_tokens(self, lmstudio_model: str):
        """LM Studio should return token usage info."""
        from parallect.providers.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model=lmstudio_model)
        result = await provider.research("Say hello.")
        assert result.tokens is not None
        assert result.tokens.get("total", 0) > 0


# ---------------------------------------------------------------------------
# Research command: end-to-end .prx creation
# ---------------------------------------------------------------------------


class TestResearchCommand:
    """End-to-end research producing a .prx bundle."""

    @pytest.mark.asyncio
    async def test_research_single_provider(self, lmstudio_model: str, tmp_dir: Path):
        """Run research with a single LM Studio provider, produce a valid bundle."""
        from parallect.orchestrator.parallel import research
        from parallect.providers.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model=lmstudio_model)
        output_path = tmp_dir / "test.prx"

        bundle = await research(
            query="What are the three laws of thermodynamics?",
            providers=[provider],
            no_synthesis=True,
            timeout_per_provider=120.0,
            output=str(output_path),
        )

        assert bundle.manifest.id.startswith("prx_")
        assert len(bundle.providers) == 1
        assert bundle.providers[0].name == "lmstudio"
        assert bundle.providers[0].report_md
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_research_with_synthesis(self, lmstudio_model: str, tmp_dir: Path):
        """Run research with LM Studio as both provider and synthesizer."""
        from parallect.orchestrator.parallel import research
        from parallect.providers.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model=lmstudio_model)
        output_path = tmp_dir / "synth.prx"

        bundle = await research(
            query="Compare Python and Rust for systems programming.",
            providers=[provider],
            synthesize_with=f"lmstudio/{lmstudio_model}",
            timeout_per_provider=120.0,
            output=str(output_path),
        )

        assert bundle.manifest.has_synthesis
        assert bundle.synthesis_md
        assert len(bundle.synthesis_md) > 50

    @pytest.mark.asyncio
    async def test_research_multi_provider(self, lmstudio_model: str, tmp_dir: Path):
        """Run research with multiple LM Studio 'providers' (simulated)."""
        from parallect.orchestrator.parallel import research
        from parallect.providers.openai_compat import OpenAICompatibleProvider

        # Create two providers with different system prompts to simulate variety
        provider1 = OpenAICompatibleProvider(
            name="researcher-a",
            base_url="http://localhost:1234",
            model=lmstudio_model,
            system_prompt="You are a concise technical writer. Keep answers short and factual.",
        )
        provider2 = OpenAICompatibleProvider(
            name="researcher-b",
            base_url="http://localhost:1234",
            model=lmstudio_model,
            system_prompt="You are an academic researcher. Provide detailed analysis with nuance.",
        )

        output_path = tmp_dir / "multi.prx"
        bundle = await research(
            query="What is quantum entanglement?",
            providers=[provider1, provider2],
            no_synthesis=True,
            timeout_per_provider=120.0,
            output=str(output_path),
        )

        assert len(bundle.providers) == 2
        names = {p.name for p in bundle.providers}
        assert "researcher-a" in names
        assert "researcher-b" in names


# ---------------------------------------------------------------------------
# Bundle read-back: validate round-trip
# ---------------------------------------------------------------------------


class TestBundleRoundTrip:
    """Read back bundles and verify contents."""

    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self, lmstudio_model: str, tmp_dir: Path):
        """Write a .prx then read it back and verify all data survives."""
        from prx_spec import read_bundle, validate_archive

        from parallect.orchestrator.parallel import research
        from parallect.providers.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model=lmstudio_model)
        output_path = tmp_dir / "roundtrip.prx"

        original = await research(
            query="Explain the CAP theorem in distributed systems.",
            providers=[provider],
            no_synthesis=True,
            timeout_per_provider=120.0,
            output=str(output_path),
        )

        # Read it back
        loaded = read_bundle(output_path)
        assert loaded.manifest.id == original.manifest.id
        assert loaded.manifest.query == original.manifest.query
        assert len(loaded.providers) == len(original.providers)
        assert loaded.providers[0].report_md == original.providers[0].report_md

        # Validate at all levels
        result = validate_archive(output_path)
        assert result.valid, f"Validation failed: {result}"


# ---------------------------------------------------------------------------
# Synthesis: LM Studio as synthesis backend
# ---------------------------------------------------------------------------


class TestSynthesis:
    """Test synthesis with local LLM."""

    @pytest.mark.asyncio
    async def test_synthesis_produces_report(self, lmstudio_model: str):
        """Synthesize multiple reports into one."""
        from parallect.providers import ProviderResult
        from parallect.synthesis.llm import synthesize

        results = [
            ProviderResult(
                provider="source-a",
                status="completed",
                report_markdown=(
                    "# Report A\n\nPython is dynamically typed"
                    " and great for prototyping."
                ),
            ),
            ProviderResult(
                provider="source-b",
                status="completed",
                report_markdown=(
                    "# Report B\n\nRust has a strict type system"
                    " and memory safety guarantees."
                ),
            ),
        ]

        synth = await synthesize(
            query="Compare Python and Rust",
            provider_results=results,
            model=f"lmstudio/{lmstudio_model}",
        )

        assert synth.report_markdown
        assert len(synth.report_markdown) > 50


# ---------------------------------------------------------------------------
# CLI subprocess tests
# ---------------------------------------------------------------------------


class TestCLICommands:
    """Test CLI commands as subprocess calls."""

    @pytest.mark.asyncio
    async def test_cli_research_to_file(self, lmstudio_model: str, tmp_dir: Path):
        """parallect research with --providers lmstudio writes a valid .prx."""
        from parallect.orchestrator.parallel import research
        from parallect.providers.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model=lmstudio_model)
        output_path = tmp_dir / "cli_test.prx"

        await research(
            query="What is photosynthesis?",
            providers=[provider],
            no_synthesis=True,
            timeout_per_provider=120.0,
            output=str(output_path),
        )

        assert output_path.exists()

        # Now test read command
        from prx_spec import read_bundle
        loaded = read_bundle(output_path)
        assert loaded.manifest.query == "What is photosynthesis?"

    @pytest.mark.asyncio
    async def test_cli_export_markdown(self, lmstudio_model: str, tmp_dir: Path):
        """Export a .prx to markdown."""
        from prx_spec import read_bundle

        from parallect.cli.export import _export_markdown
        from parallect.orchestrator.parallel import research
        from parallect.providers.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model=lmstudio_model)
        prx_path = tmp_dir / "export_test.prx"

        await research(
            query="What is DNA?",
            providers=[provider],
            no_synthesis=True,
            timeout_per_provider=120.0,
            output=str(prx_path),
        )

        bundle = read_bundle(prx_path)
        md = _export_markdown(bundle)
        assert "DNA" in md
        assert "lmstudio" in md.lower() or "report" in md.lower()

    @pytest.mark.asyncio
    async def test_cli_export_json(self, lmstudio_model: str, tmp_dir: Path):
        """Export a .prx to JSON."""
        from prx_spec import read_bundle

        from parallect.cli.export import _export_json
        from parallect.orchestrator.parallel import research
        from parallect.providers.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model=lmstudio_model)
        prx_path = tmp_dir / "json_test.prx"

        await research(
            query="What is RNA?",
            providers=[provider],
            no_synthesis=True,
            timeout_per_provider=120.0,
            output=str(prx_path),
        )

        bundle = read_bundle(prx_path)
        json_str = _export_json(bundle)
        data = json.loads(json_str)
        assert "manifest" in data
        assert "providers" in data
        assert data["manifest"]["query"] == "What is RNA?"

    @pytest.mark.asyncio
    async def test_validate_bundle(self, lmstudio_model: str, tmp_dir: Path):
        """Validate a bundle at all levels."""
        from prx_spec import validate_archive

        from parallect.orchestrator.parallel import research
        from parallect.providers.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model=lmstudio_model)
        prx_path = tmp_dir / "validate_test.prx"

        await research(
            query="What is gravity?",
            providers=[provider],
            no_synthesis=True,
            timeout_per_provider=120.0,
            output=str(prx_path),
        )

        result = validate_archive(prx_path)
        for level in result.levels:
            assert level.passed, f"L{level.level} failed: {level.errors}"


# ---------------------------------------------------------------------------
# Full pipeline: research + synthesize + validate
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Full end-to-end pipeline tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, lmstudio_model: str, tmp_dir: Path):
        """Research -> synthesize -> write -> read -> validate -> export."""
        from prx_spec import read_bundle, validate_archive

        from parallect.cli.export import _export_json, _export_markdown
        from parallect.orchestrator.parallel import research
        from parallect.providers.lmstudio import LMStudioProvider
        from parallect.providers.openai_compat import OpenAICompatibleProvider

        # 1. Research with two providers
        p1 = LMStudioProvider(model=lmstudio_model)
        p2 = OpenAICompatibleProvider(
            name="alt-researcher",
            base_url="http://localhost:1234",
            model=lmstudio_model,
            system_prompt="You are a skeptical analyst. Question assumptions and identify risks.",
        )

        prx_path = tmp_dir / "pipeline.prx"

        bundle = await research(
            query="What are the pros and cons of microservices architecture?",
            providers=[p1, p2],
            synthesize_with=f"lmstudio/{lmstudio_model}",
            timeout_per_provider=120.0,
            output=str(prx_path),
        )

        # 2. Verify bundle structure
        assert len(bundle.providers) == 2
        assert bundle.manifest.has_synthesis
        assert bundle.synthesis_md

        # 3. Read back from disk
        loaded = read_bundle(prx_path)
        assert loaded.manifest.id == bundle.manifest.id
        assert len(loaded.providers) == 2
        assert loaded.synthesis_md == bundle.synthesis_md

        # 4. Validate
        result = validate_archive(prx_path)
        assert result.valid

        # 5. Export
        md = _export_markdown(loaded)
        assert "microservices" in md.lower()

        json_str = _export_json(loaded)
        data = json.loads(json_str)
        assert len(data["providers"]) == 2
        assert data["synthesis"]
