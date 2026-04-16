"""Tests for the full research() orchestration flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from parallect.orchestrator.budget import BudgetExceededError
from parallect.orchestrator.parallel import research
from parallect.providers import ProviderResult

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class MockProvider:
    """A fast mock provider that returns canned results."""

    def __init__(self, name: str = "mock", content: str = "# Mock Report\n\nDone."):
        self._name = name
        self._content = content

    @property
    def name(self) -> str:
        return self._name

    async def research(self, query: str) -> ProviderResult:
        return ProviderResult(
            provider=self._name,
            status="completed",
            report_markdown=self._content,
            cost_usd=0.01,
            duration_seconds=0.1,
        )

    def estimate_cost(self, query: str) -> float:
        return 0.01

    def is_available(self) -> bool:
        return True


class FailingMockProvider:
    """A provider that always fails."""

    @property
    def name(self) -> str:
        return "failing"

    async def research(self, query: str) -> ProviderResult:
        raise RuntimeError("Provider crash")

    def estimate_cost(self, query: str) -> float:
        return 0.05

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# research() tests
# ---------------------------------------------------------------------------


class TestResearchFunction:
    @pytest.mark.asyncio
    async def test_single_provider_no_synthesis(self):
        bundle = await research(
            query="test query",
            providers=[MockProvider()],
            no_synthesis=True,
        )
        assert bundle.manifest.query == "test query"
        assert len(bundle.providers) == 1
        assert bundle.providers[0].name == "mock"
        assert bundle.synthesis_md is None
        assert bundle.manifest.has_synthesis is False

    @pytest.mark.asyncio
    async def test_multi_provider(self):
        providers = [
            MockProvider("alpha", "# Alpha Report"),
            MockProvider("beta", "# Beta Report"),
        ]
        bundle = await research(
            query="multi test",
            providers=providers,
            no_synthesis=True,
        )
        assert len(bundle.providers) == 2
        names = {p.name for p in bundle.providers}
        assert names == {"alpha", "beta"}
        assert bundle.manifest.providers_used == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """One provider fails; the other's result is still captured."""
        providers = [MockProvider("good"), FailingMockProvider()]
        bundle = await research(
            query="test",
            providers=providers,
            no_synthesis=True,
        )
        assert len(bundle.providers) == 1
        assert bundle.providers[0].name == "good"

    @pytest.mark.asyncio
    async def test_all_fail_empty_providers(self):
        """When all providers fail, orchestrator raises RuntimeError before reaching Manifest."""
        with pytest.raises(RuntimeError, match="All providers failed"):
            await research(
                query="test",
                providers=[FailingMockProvider()],
                no_synthesis=True,
            )

    @pytest.mark.asyncio
    async def test_with_synthesis(self):
        """Verify synthesis is called when not disabled."""
        from parallect.synthesis.llm import SynthesisResult

        mock_synth = AsyncMock(
            return_value=SynthesisResult(
                report_markdown="# Unified Report",
                model="test-model",
            )
        )

        with patch("parallect.synthesis.llm.synthesize", mock_synth):
            bundle = await research(
                query="synth test",
                providers=[MockProvider("p1"), MockProvider("p2")],
                synthesize_with="anthropic",
                no_synthesis=False,
            )

        assert bundle.synthesis_md == "# Unified Report"
        assert bundle.manifest.has_synthesis is True
        mock_synth.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesis_failure_non_fatal(self):
        """If synthesis fails, the bundle should still be created without synthesis."""
        mock_synth = AsyncMock(side_effect=RuntimeError("Synthesis exploded"))

        with patch("parallect.synthesis.llm.synthesize", mock_synth):
            bundle = await research(
                query="test",
                providers=[MockProvider()],
                synthesize_with="anthropic",
                no_synthesis=False,
            )

        assert bundle.synthesis_md is None
        assert bundle.manifest.has_synthesis is False

    @pytest.mark.asyncio
    async def test_budget_cap_enforced(self):
        """Should raise BudgetExceededError when estimate exceeds cap."""
        # MockProvider.estimate_cost returns 0.01 each, so 2 providers = 0.02
        with pytest.raises(BudgetExceededError):
            await research(
                query="test",
                providers=[MockProvider("a"), MockProvider("b")],
                budget_cap_usd=0.001,  # too low
                no_synthesis=True,
            )

    @pytest.mark.asyncio
    async def test_budget_cap_passes(self):
        """Should succeed when estimate is within cap."""
        bundle = await research(
            query="test",
            providers=[MockProvider()],
            budget_cap_usd=1.0,  # plenty of room
            no_synthesis=True,
        )
        assert len(bundle.providers) == 1

    @pytest.mark.asyncio
    async def test_output_path_writes_file(self, tmp_path):
        """When output is given, bundle should be written to disk."""
        out_file = tmp_path / "result.prx"
        await research(
            query="write test",
            providers=[MockProvider()],
            no_synthesis=True,
            output=str(out_file),
        )
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_output_dir_writes_file(self, tmp_path):
        """When output is a directory, bundle should be saved inside it."""
        await research(
            query="dir test",
            providers=[MockProvider()],
            no_synthesis=True,
            output=str(tmp_path),
        )
        prx_files = list(tmp_path.glob("*.prx"))
        assert len(prx_files) == 1

    @pytest.mark.asyncio
    async def test_bundle_metadata(self):
        """Verify manifest fields are correctly populated."""
        bundle = await research(
            query="metadata test",
            providers=[MockProvider("p1")],
            no_synthesis=True,
        )
        assert bundle.manifest.id.startswith("prx_")
        assert bundle.manifest.spec_version == "1.1"
        assert bundle.manifest.producer.name == "parallect-oss"
        assert bundle.manifest.total_cost_usd is not None
        assert "metadata test" in bundle.query_md

    @pytest.mark.asyncio
    async def test_cost_aggregation(self):
        """Cost should be summed across all successful providers."""
        providers = [
            MockProvider("a"),  # cost_usd=0.01
            MockProvider("b"),  # cost_usd=0.01
        ]
        bundle = await research(
            query="cost test",
            providers=providers,
            no_synthesis=True,
        )
        assert bundle.manifest.total_cost_usd == pytest.approx(0.02, abs=0.001)


class TestManifestFields:
    """Regression tests for manifest correctness after the v1.1 alignment."""

    @pytest.mark.asyncio
    async def test_spec_version_is_1_1(self):
        bundle = await research(
            query="spec test",
            providers=[MockProvider()],
            no_synthesis=True,
        )
        assert bundle.manifest.spec_version == "1.1"

    @pytest.mark.asyncio
    async def test_provider_breakdown_populated_on_success(self):
        """Every successful provider should appear in provider_breakdown."""
        providers = [MockProvider("alpha"), MockProvider("beta")]
        bundle = await research(
            query="breakdown test",
            providers=providers,
            no_synthesis=True,
        )
        pb = bundle.manifest.provider_breakdown
        assert len(pb) == 2
        names = {e.provider for e in pb}
        assert names == {"alpha", "beta"}
        for entry in pb:
            assert entry.status == "completed"
            assert entry.duration_seconds is not None
            assert entry.cost_usd == pytest.approx(0.01, abs=0.001)

    @pytest.mark.asyncio
    async def test_provider_breakdown_includes_failures(self):
        """Failed providers should also be recorded so operators can diagnose."""
        providers = [MockProvider("good"), FailingMockProvider()]
        bundle = await research(
            query="failure-visibility test",
            providers=providers,
            no_synthesis=True,
        )
        pb = bundle.manifest.provider_breakdown
        statuses = {e.provider: e.status for e in pb}
        assert statuses["good"] == "completed"
        # FailingMockProvider raises; outcome.result is None -> status="failed"
        assert "failing" in statuses
        assert statuses["failing"] == "failed"

    @pytest.mark.asyncio
    async def test_total_duration_exceeds_provider_duration(self):
        """total_duration_seconds must cover the whole orchestrator run, not
        just the slowest provider. MockProvider reports 0.1s; the overall
        run always takes longer than that."""
        bundle = await research(
            query="duration test",
            providers=[MockProvider()],
            no_synthesis=True,
        )
        assert bundle.manifest.total_duration_seconds is not None
        assert bundle.manifest.total_duration_seconds >= 0.1

    @pytest.mark.asyncio
    async def test_has_attestations_flips_when_signed(self, tmp_path):
        """After auto-sign produces an attestation, the manifest flag must
        reflect that — the previous behavior silently left it False."""
        from prx_spec.attestation.keys import generate_keypair

        # generate_keypair writes private+public keys under key_dir; passing a
        # fresh tmp path guarantees we don't collide with an existing key.
        key_dir = tmp_path / "keys"
        generate_keypair(key_dir)

        class _FakeSettings:
            auto_sign = True
            key_path = str(key_dir)
            identity = "test-user"
            synthesis_backend = ""
            synthesis_model = ""

        bundle = await research(
            query="signing test",
            providers=[MockProvider()],
            no_synthesis=True,
            settings=_FakeSettings(),
        )
        assert bundle.attestations, "Expected at least one attestation"
        assert bundle.manifest.has_attestations is True

    @pytest.mark.asyncio
    async def test_has_attestations_false_without_signing_key(self):
        """If no signing key is available, manifest flag stays False and no
        attestations are added (baseline — not a regression test, but a
        complement to the flips-on-sign test)."""
        bundle = await research(
            query="no-sign test",
            providers=[MockProvider()],
            no_synthesis=True,
            no_sign=True,
        )
        assert bundle.manifest.has_attestations is False
        assert not bundle.attestations


# ---------------------------------------------------------------------------
# Evidence graph tests
# ---------------------------------------------------------------------------


class CitingProvider:
    """Mock provider that returns a caller-supplied citations list."""

    def __init__(self, name: str, citations: list[dict]):
        self._name = name
        self._citations = citations

    @property
    def name(self) -> str:
        return self._name

    async def research(self, query: str) -> ProviderResult:
        return ProviderResult(
            provider=self._name,
            status="completed",
            report_markdown=f"# Report from {self._name}",
            citations=list(self._citations),
            cost_usd=0.01,
            duration_seconds=0.1,
        )

    def estimate_cost(self, query: str) -> float:
        return 0.01

    def is_available(self) -> bool:
        return True


class TestEvidenceGraph:
    """Provider-level evidence graph linking claims to sources."""

    @pytest.mark.asyncio
    async def test_graph_has_one_edge_per_claim_provider_source(self):
        """One edge per (claim × supporting provider × that provider's sources)."""
        alpha_citations = [
            {"url": "https://a.example.com/one", "title": "A1"},
            {"url": "https://a.example.com/two", "title": "A2"},
        ]
        beta_citations = [
            {"url": "https://b.example.com/one", "title": "B1"},
        ]

        # Mock claims extraction: 2 claims, different supporting providers.
        canned_claims = [
            {
                "id": "claim_001",
                "content": "Alpha-only claim",
                "providers_supporting": ["alpha"],
                "providers_contradicting": [],
                "category": "fact",
            },
            {
                "id": "claim_002",
                "content": "Both providers support this",
                "providers_supporting": ["alpha", "beta"],
                "providers_contradicting": [],
                "category": "fact",
            },
        ]

        mock_extract = AsyncMock(return_value=canned_claims)

        with patch(
            "parallect.synthesis.extract.extract_claims", mock_extract
        ):
            bundle = await research(
                query="evidence test",
                providers=[
                    CitingProvider("alpha", alpha_citations),
                    CitingProvider("beta", beta_citations),
                ],
                no_synthesis=True,
            )

        # Sanity: claims + sources were both built.
        assert bundle.claims is not None
        assert len(bundle.claims.claims) == 2
        assert bundle.sources is not None
        assert len(bundle.sources.sources) == 3  # a1, a2, b1 (no URL overlap)

        # Evidence graph should be present.
        assert bundle.evidence_graph is not None
        assert bundle.manifest.has_evidence_graph is True

        edges = bundle.evidence_graph.evidence
        # claim_001: alpha supports, alpha cited 2 sources → 2 edges
        # claim_002: alpha + beta support. alpha cited 2, beta cited 1 → 3 edges
        assert len(edges) == 5

        # All relations are "supports".
        assert all(e.relation == "supports" for e in edges)

        # Every edge's discovered_by_provider is one of the supporting providers
        # for its claim.
        claim_supporters = {c.id: set(c.providers_supporting) for c in bundle.claims.claims}
        for e in edges:
            assert e.discovered_by_provider in claim_supporters[e.claim_id]

        # Every edge's source_id resolves to a source that was cited by that edge's provider.
        source_index = {s.id: s for s in bundle.sources.sources}
        for e in edges:
            src = source_index[e.source_id]
            assert e.discovered_by_provider in (src.cited_by_providers or [])

    @pytest.mark.asyncio
    async def test_no_graph_when_claims_missing(self):
        """If claims extraction returns nothing, no evidence graph is emitted."""
        mock_extract = AsyncMock(return_value=[])

        with patch(
            "parallect.synthesis.extract.extract_claims", mock_extract
        ):
            bundle = await research(
                query="evidence test",
                providers=[
                    CitingProvider(
                        "alpha", [{"url": "https://example.com/x", "title": "X"}]
                    ),
                ],
                no_synthesis=True,
            )

        assert bundle.claims is None
        assert bundle.evidence_graph is None
        assert bundle.manifest.has_evidence_graph is False

    @pytest.mark.asyncio
    async def test_no_graph_when_sources_missing(self):
        """If no provider emits citations, there are no sources and no graph."""
        canned_claims = [
            {
                "id": "claim_001",
                "content": "A claim",
                "providers_supporting": ["alpha"],
                "providers_contradicting": [],
                "category": "fact",
            }
        ]
        mock_extract = AsyncMock(return_value=canned_claims)

        with patch(
            "parallect.synthesis.extract.extract_claims", mock_extract
        ):
            bundle = await research(
                query="evidence test",
                providers=[MockProvider("alpha")],  # no citations
                no_synthesis=True,
            )

        assert bundle.sources is None
        assert bundle.evidence_graph is None
        assert bundle.manifest.has_evidence_graph is False


class TestResearchOnStatus:
    """The ``on_status`` callback fires messages at each pipeline phase."""

    @pytest.mark.asyncio
    async def test_status_sequence_includes_fan_out_extraction_and_write(
        self, tmp_path
    ):
        messages: list[str] = []

        canned_claims = [
            {
                "id": "claim_001",
                "content": "Something claim",
                "providers_supporting": ["mock"],
                "providers_contradicting": [],
                "category": "fact",
            }
        ]

        with patch(
            "parallect.synthesis.extract.extract_claims",
            AsyncMock(return_value=canned_claims),
        ):
            out = tmp_path / "bundle.prx"
            await research(
                query="status test",
                providers=[MockProvider("mock")],
                no_synthesis=True,
                output=str(out),
                on_status=messages.append,
            )

        # Fan-out phase should be announced.
        assert any(m.startswith("Fanning out to") for m in messages), messages
        # Per-provider landing line.
        assert any(m.startswith("mock: complete") for m in messages), messages
        # Claims extraction phase.
        assert any(m.startswith("Extracting claims with") for m in messages), messages
        # Bundle write.
        assert any(m.startswith("Writing ") and m.endswith(".prx...") for m in messages), messages

    @pytest.mark.asyncio
    async def test_callback_errors_are_swallowed(self):
        """A misbehaving on_status hook must not kill the research run."""

        def boom(_msg: str) -> None:
            raise RuntimeError("UI is on fire")

        bundle = await research(
            query="resilience test",
            providers=[MockProvider()],
            no_synthesis=True,
            on_status=boom,
        )
        assert bundle is not None
        assert len(bundle.providers) == 1

    @pytest.mark.asyncio
    async def test_on_status_optional(self):
        """Omitting on_status is fine — no regression from baseline behaviour."""
        bundle = await research(
            query="baseline test",
            providers=[MockProvider()],
            no_synthesis=True,
        )
        assert bundle is not None
        assert len(bundle.providers) == 1
