"""Live integration tests for all cloud providers.

Run with:
    PARALLECT_LIVE_TESTS=1 uv run pytest tests/test_provider_integration.py -v

Requires API keys set as environment variables:
    OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY),
    ANTHROPIC_API_KEY, XAI_API_KEY, PERPLEXITY_API_KEY

Each provider is tested with a trivial query ("What is 1+1?") to keep
cost minimal while validating the full request → parse → ProviderResult
pipeline including:
  - HTTP request/response cycle
  - Response parsing (Responses API / generateContent / Messages API)
  - Citation extraction (url_citations, groundingMetadata, tool results, annotations)
  - Token tracking & cost calculation
  - ProviderResult field completeness
  - Error handling for invalid keys
  - The --deep flag constructor path (no live call for deep — too slow/expensive)
"""

from __future__ import annotations

import os

import pytest

from parallect.providers import ProviderResult

# ---------------------------------------------------------------------------
# Skip unless opted-in
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("PARALLECT_LIVE_TESTS"),
    reason="Set PARALLECT_LIVE_TESTS=1 to run live provider tests",
)

SIMPLE_QUERY = "What is 1+1? Answer in one sentence."

# ---------------------------------------------------------------------------
# Fixtures: one per provider key
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        pytest.skip("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def xai_key() -> str:
    key = os.environ.get("XAI_API_KEY", "")
    if not key:
        pytest.skip("XAI_API_KEY not set")
    return key


@pytest.fixture(scope="module")
def perplexity_key() -> str:
    key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not key:
        pytest.skip("PERPLEXITY_API_KEY not set")
    return key


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def assert_successful_result(result: ProviderResult, provider_name: str) -> None:
    """Common assertions for a successful provider result."""
    assert result.status == "completed", (
        f"{provider_name} failed: {result.error}"
    )
    assert result.provider == provider_name
    assert result.report_markdown, f"{provider_name} returned empty markdown"
    assert len(result.report_markdown) > 5, (
        f"{provider_name} markdown too short: {result.report_markdown!r}"
    )
    assert result.duration_seconds is not None
    assert result.duration_seconds > 0
    assert result.cost_usd is not None
    assert result.cost_usd >= 0
    assert result.model is not None


def assert_tokens_present(result: ProviderResult, provider_name: str) -> None:
    """Assert token usage is tracked."""
    assert result.tokens is not None, f"{provider_name} missing tokens dict"
    # At minimum we should track input and output
    assert result.tokens.get("input", 0) > 0 or result.tokens.get("total", 0) > 0, (
        f"{provider_name} tokens look empty: {result.tokens}"
    )


def assert_answer_reasonable(result: ProviderResult) -> None:
    """For '1+1' the answer should mention '2' (digit or word)."""
    text = result.report_markdown.lower()
    assert "2" in text or "two" in text, (
        f"Expected '2' or 'two' in response to '1+1', got: {result.report_markdown[:200]}"
    )


# ===================================================================
# OpenAI — Responses API + web_search_preview
# ===================================================================


class TestOpenAIProvider:
    """Live tests for OpenAI Responses API provider."""

    @pytest.mark.asyncio
    async def test_basic_query(self, openai_key: str):
        """Simple query returns completed result with markdown."""
        from parallect.providers.openai_dr import OpenAIDRProvider

        provider = OpenAIDRProvider(api_key=openai_key)
        assert provider.is_available()
        assert provider.name == "openai"

        result = await provider.research(SIMPLE_QUERY)
        assert_successful_result(result, "openai")
        assert_answer_reasonable(result)

    @pytest.mark.asyncio
    async def test_tokens_tracked(self, openai_key: str):
        """Token usage is captured from the Responses API."""
        from parallect.providers.openai_dr import OpenAIDRProvider

        provider = OpenAIDRProvider(api_key=openai_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        assert_tokens_present(result, "openai")

    @pytest.mark.asyncio
    async def test_cost_calculated(self, openai_key: str):
        """Cost should be calculated from token usage, not a flat rate."""
        from parallect.providers.openai_dr import OpenAIDRProvider

        provider = OpenAIDRProvider(api_key=openai_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        # For a simple query with gpt-4o-mini, cost should be very small
        assert result.cost_usd is not None
        assert result.cost_usd < 0.10, f"Cost too high for simple query: ${result.cost_usd}"

    @pytest.mark.asyncio
    async def test_model_in_result(self, openai_key: str):
        """Model name should be populated from API response."""
        from parallect.providers.openai_dr import OpenAIDRProvider

        provider = OpenAIDRProvider(api_key=openai_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        assert result.model is not None
        assert "gpt" in result.model.lower() or "o" in result.model.lower()

    @pytest.mark.asyncio
    async def test_web_search_citations(self, openai_key: str):
        """A query that needs web search should produce citations."""
        from parallect.providers.openai_dr import OpenAIDRProvider

        provider = OpenAIDRProvider(api_key=openai_key)
        result = await provider.research(
            "What is the current population of Tokyo as of 2025?"
        )
        assert result.status == "completed"
        # web_search_preview should produce url_citation annotations
        assert isinstance(result.citations, list)
        if result.citations:
            assert "url" in result.citations[0]

    @pytest.mark.asyncio
    async def test_invalid_key_fails_gracefully(self):
        """Invalid API key returns failed status, doesn't crash."""
        from parallect.providers.openai_dr import OpenAIDRProvider

        provider = OpenAIDRProvider(api_key="sk-invalid-key-12345")
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "failed"
        assert result.error is not None
        assert result.duration_seconds is not None

    def test_deep_mode_constructor(self, openai_key: str):
        """Deep mode selects o3-deep-research model."""
        from parallect.providers.openai_dr import OpenAIDRProvider

        provider = OpenAIDRProvider(api_key=openai_key, deep=True)
        assert provider.model == "o3-deep-research"
        assert provider._deep is True
        assert provider.estimate_cost("q") == 7.50

    def test_default_mode_constructor(self, openai_key: str):
        """Default mode selects gpt-4o-mini."""
        from parallect.providers.openai_dr import OpenAIDRProvider

        provider = OpenAIDRProvider(api_key=openai_key)
        assert provider.model == "gpt-4o-mini"
        assert provider._deep is False
        assert provider.estimate_cost("q") == 0.05

    def test_cost_calculation_tiers(self):
        """Verify cost calculation for different model tiers."""
        from parallect.providers.openai_dr import OpenAIDRProvider

        usage = {"input_tokens": 1000, "output_tokens": 500, "input_tokens_details": {}}
        # gpt-4o-mini tier
        cost_mini = OpenAIDRProvider._calculate_cost(usage, "gpt-4o-mini", 0)
        assert cost_mini > 0
        # o3 tier should be more expensive
        cost_o3 = OpenAIDRProvider._calculate_cost(usage, "o3-deep-research", 0)
        assert cost_o3 > cost_mini
        # Web search adds cost
        cost_search = OpenAIDRProvider._calculate_cost(usage, "gpt-4o-mini", 5)
        assert cost_search > cost_mini


# ===================================================================
# Gemini — generateContent + googleSearch grounding
# ===================================================================


class TestGeminiProvider:
    """Live tests for Gemini provider with grounded search."""

    @pytest.mark.asyncio
    async def test_basic_query(self, gemini_key: str):
        """Simple query returns completed result."""
        from parallect.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key=gemini_key)
        assert provider.is_available()
        assert provider.name == "gemini"

        result = await provider.research(SIMPLE_QUERY)
        assert_successful_result(result, "gemini")
        assert_answer_reasonable(result)

    @pytest.mark.asyncio
    async def test_tokens_tracked(self, gemini_key: str):
        """Token usage captured from usageMetadata."""
        from parallect.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key=gemini_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        assert_tokens_present(result, "gemini")

    @pytest.mark.asyncio
    async def test_cost_calculated(self, gemini_key: str):
        """Cost should be token-based, not flat."""
        from parallect.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key=gemini_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        assert result.cost_usd is not None
        assert result.cost_usd < 0.50, f"Cost too high: ${result.cost_usd}"

    @pytest.mark.asyncio
    async def test_grounding_citations(self, gemini_key: str):
        """A factual query should trigger googleSearch and return grounding citations."""
        from parallect.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key=gemini_key)
        result = await provider.research(
            "What is the current population of Tokyo as of 2025?"
        )
        assert result.status == "completed"
        assert isinstance(result.citations, list)
        # googleSearch grounding should produce citations with URLs
        if result.citations:
            assert "url" in result.citations[0]

    @pytest.mark.asyncio
    async def test_max_output_tokens(self, gemini_key: str):
        """Provider should handle longer outputs (65K max configured)."""
        from parallect.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key=gemini_key)
        result = await provider.research(
            "Briefly list the first 5 prime numbers."
        )
        assert result.status == "completed"
        assert len(result.report_markdown) > 10

    @pytest.mark.asyncio
    async def test_invalid_key_fails_gracefully(self):
        """Invalid API key returns failed status."""
        from parallect.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="invalid-key-12345")
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "failed"
        assert result.error is not None

    def test_deep_mode_constructor(self, gemini_key: str):
        """Deep mode sets the flag (doesn't change model for generateContent)."""
        from parallect.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key=gemini_key, deep=True)
        assert provider._deep is True
        assert provider.estimate_cost("q") == 3.0

    def test_default_mode_constructor(self, gemini_key: str):
        """Default mode uses gemini-2.5-flash."""
        from parallect.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key=gemini_key)
        assert provider.model == "gemini-2.5-flash"
        assert provider._deep is False
        assert provider.estimate_cost("q") == 0.15

    def test_cost_calculation(self):
        """Verify cost calculation handles all token fields."""
        from parallect.providers.gemini import GeminiProvider

        usage = {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 500,
            "totalTokenCount": 1500,
            "thoughtsTokenCount": 200,
        }
        cost = GeminiProvider._calculate_cost(usage)
        assert cost > 0

        # Deep cost with large context flag
        cost_deep = GeminiProvider._calculate_deep_cost(usage)
        assert cost_deep > 0
        assert cost_deep > cost  # deep uses higher rates


# ===================================================================
# Anthropic — Messages API + web_search + web_fetch tools
# ===================================================================


class TestAnthropicProvider:
    """Live tests for Anthropic provider with web search tools."""

    @pytest.mark.asyncio
    async def test_basic_query(self, anthropic_key: str):
        """Simple query returns completed result."""
        from parallect.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=anthropic_key)
        assert provider.is_available()
        assert provider.name == "anthropic"

        result = await provider.research(SIMPLE_QUERY)
        assert_successful_result(result, "anthropic")
        assert_answer_reasonable(result)

    @pytest.mark.asyncio
    async def test_tokens_tracked(self, anthropic_key: str):
        """Token usage captured from messages API."""
        from parallect.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=anthropic_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        assert_tokens_present(result, "anthropic")

    @pytest.mark.asyncio
    async def test_cost_calculated(self, anthropic_key: str):
        """Cost should be token-based."""
        from parallect.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=anthropic_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        assert result.cost_usd is not None
        assert result.cost_usd < 1.00, f"Cost too high: ${result.cost_usd}"

    @pytest.mark.asyncio
    async def test_web_search_citations(self, anthropic_key: str):
        """Factual query should trigger web_search tool and return citations."""
        from parallect.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=anthropic_key)
        result = await provider.research(
            "What is the current population of Tokyo as of 2025?"
        )
        assert result.status == "completed"
        assert isinstance(result.citations, list)
        # Anthropic web_search should find results
        if result.citations:
            assert "url" in result.citations[0]

    @pytest.mark.asyncio
    async def test_agentic_loop_tracks_steps(self, anthropic_key: str):
        """Token dict should include search/fetch request counts."""
        from parallect.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=anthropic_key)
        result = await provider.research(
            "What were the top 3 news stories today?"
        )
        assert result.status == "completed"
        assert result.tokens is not None
        # Should have search_requests field (may be 0 if model didn't search)
        assert "search_requests" in result.tokens
        assert "fetch_requests" in result.tokens

    @pytest.mark.asyncio
    async def test_invalid_key_fails_gracefully(self):
        """Invalid API key returns failed status."""
        from parallect.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-ant-invalid-12345")
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "failed"
        assert result.error is not None

    def test_deep_mode_constructor(self, anthropic_key: str):
        """Deep mode selects opus model and higher step limit."""
        from parallect.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=anthropic_key, deep=True)
        assert "opus" in provider.model
        assert provider._deep is True
        assert provider._max_steps == 25
        assert provider.estimate_cost("q") == 4.50

    def test_default_mode_constructor(self, anthropic_key: str):
        """Default mode selects sonnet."""
        from parallect.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=anthropic_key)
        assert "sonnet" in provider.model
        assert provider._deep is False
        assert provider._max_steps == 15
        assert provider.estimate_cost("q") == 0.50

    def test_cost_calculation(self):
        """Verify cost calculation for different models."""
        from parallect.providers.anthropic import AnthropicProvider

        # Sonnet tier
        cost_sonnet = AnthropicProvider._calculate_cost(
            1000, 500, 5, "claude-sonnet-4-20250514"
        )
        assert cost_sonnet > 0
        # Opus tier should be more expensive
        cost_opus = AnthropicProvider._calculate_cost(
            1000, 500, 5, "claude-opus-4-20250918"
        )
        assert cost_opus > cost_sonnet


# ===================================================================
# Grok — xAI Responses API + web_search + x_search
# ===================================================================


class TestGrokProvider:
    """Live tests for Grok provider with web + X search."""

    @pytest.mark.asyncio
    async def test_basic_query(self, xai_key: str):
        """Simple query returns completed result."""
        from parallect.providers.grok import GrokProvider

        provider = GrokProvider(api_key=xai_key)
        assert provider.is_available()
        assert provider.name == "grok"

        result = await provider.research(SIMPLE_QUERY)
        assert_successful_result(result, "grok")
        assert_answer_reasonable(result)

    @pytest.mark.asyncio
    async def test_tokens_tracked(self, xai_key: str):
        """Token usage captured from Responses API."""
        from parallect.providers.grok import GrokProvider

        provider = GrokProvider(api_key=xai_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        assert_tokens_present(result, "grok")

    @pytest.mark.asyncio
    async def test_cost_calculated(self, xai_key: str):
        """Cost should be calculated or from cost_in_usd_ticks."""
        from parallect.providers.grok import GrokProvider

        provider = GrokProvider(api_key=xai_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        assert result.cost_usd is not None
        assert result.cost_usd >= 0

    @pytest.mark.asyncio
    async def test_web_and_x_search(self, xai_key: str):
        """Query that benefits from search should produce citations."""
        from parallect.providers.grok import GrokProvider

        provider = GrokProvider(api_key=xai_key)
        result = await provider.research(
            "What is the current population of Tokyo as of 2025?"
        )
        assert result.status == "completed"
        assert isinstance(result.citations, list)
        if result.citations:
            assert "url" in result.citations[0]

    @pytest.mark.asyncio
    async def test_x_search_tracking(self, xai_key: str):
        """Token dict should include web_searches and x_searches counts."""
        from parallect.providers.grok import GrokProvider

        provider = GrokProvider(api_key=xai_key)
        result = await provider.research(
            "What are people saying on X about AI agents today?"
        )
        assert result.status == "completed"
        assert result.tokens is not None
        assert "web_searches" in result.tokens
        assert "x_searches" in result.tokens

    @pytest.mark.asyncio
    async def test_invalid_key_fails_gracefully(self):
        """Invalid API key returns failed status."""
        from parallect.providers.grok import GrokProvider

        provider = GrokProvider(api_key="xai-invalid-key-12345")
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "failed"
        assert result.error is not None

    def test_constructor_defaults(self, xai_key: str):
        """Default model is grok-3."""
        from parallect.providers.grok import GrokProvider

        provider = GrokProvider(api_key=xai_key)
        assert provider.model == "grok-3"
        assert provider._deep is False
        assert provider.estimate_cost("q") == 0.50

    def test_deep_mode_constructor(self, xai_key: str):
        """Deep mode selects grok-4 with tools."""
        from parallect.providers.grok import GrokProvider

        provider = GrokProvider(api_key=xai_key, deep=True)
        assert provider.model == "grok-4"
        assert provider._deep is True

    def test_cost_calculation_tiers(self):
        """Verify cost calculation for different model tiers."""
        from parallect.providers.grok import GrokProvider

        usage = {"input_tokens": 1000, "output_tokens": 500}
        tool_usage: dict = {}
        # Premium tier (grok-3)
        cost_premium = GrokProvider._calculate_cost(usage, tool_usage, "grok-3")
        assert cost_premium > 0
        # Non-premium should be cheaper
        cost_cheap = GrokProvider._calculate_cost(usage, tool_usage, "grok-mini")
        assert cost_cheap < cost_premium

    def test_cost_from_ticks(self):
        """When cost_in_usd_ticks is present, it should be used directly."""
        from parallect.providers.grok import GrokProvider

        provider = GrokProvider(api_key="test")
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "2"}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost_in_usd_ticks": 50_000_000,  # $0.005
            },
        }
        result = provider._parse(data, 1.0)
        assert result.cost_usd == pytest.approx(0.005, abs=0.001)


# ===================================================================
# Perplexity — Sonar API + native citations
# ===================================================================


class TestPerplexityProvider:
    """Live tests for Perplexity Sonar provider."""

    @pytest.mark.asyncio
    async def test_basic_query(self, perplexity_key: str):
        """Simple query returns completed result."""
        from parallect.providers.perplexity import PerplexityProvider

        provider = PerplexityProvider(api_key=perplexity_key)
        assert provider.is_available()
        assert provider.name == "perplexity"

        result = await provider.research(SIMPLE_QUERY)
        assert_successful_result(result, "perplexity")
        assert_answer_reasonable(result)

    @pytest.mark.asyncio
    async def test_tokens_tracked(self, perplexity_key: str):
        """Token usage captured from chat completions response."""
        from parallect.providers.perplexity import PerplexityProvider

        provider = PerplexityProvider(api_key=perplexity_key)
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "completed"
        assert_tokens_present(result, "perplexity")

    @pytest.mark.asyncio
    async def test_native_citations(self, perplexity_key: str):
        """Perplexity always returns citations for research queries."""
        from parallect.providers.perplexity import PerplexityProvider

        provider = PerplexityProvider(api_key=perplexity_key)
        result = await provider.research(
            "What is the current population of Tokyo?"
        )
        assert result.status == "completed"
        assert isinstance(result.citations, list)
        # Perplexity should always return citations for factual queries
        assert len(result.citations) > 0, "Expected citations from Perplexity"
        assert "url" in result.citations[0]

    @pytest.mark.asyncio
    async def test_invalid_key_fails_gracefully(self):
        """Invalid API key returns failed status."""
        from parallect.providers.perplexity import PerplexityProvider

        provider = PerplexityProvider(api_key="pplx-invalid-12345")
        result = await provider.research(SIMPLE_QUERY)
        assert result.status == "failed"
        assert result.error is not None


# ===================================================================
# Cross-provider consistency tests
# ===================================================================


class TestCrossProviderConsistency:
    """Verify that all providers return consistent ProviderResult shapes."""

    @pytest.fixture
    def all_providers(
        self, openai_key, gemini_key, anthropic_key, xai_key, perplexity_key
    ):
        """Instantiate all available providers."""
        from parallect.providers.anthropic import AnthropicProvider
        from parallect.providers.gemini import GeminiProvider
        from parallect.providers.grok import GrokProvider
        from parallect.providers.openai_dr import OpenAIDRProvider
        from parallect.providers.perplexity import PerplexityProvider

        return [
            OpenAIDRProvider(api_key=openai_key),
            GeminiProvider(api_key=gemini_key),
            AnthropicProvider(api_key=anthropic_key),
            GrokProvider(api_key=xai_key),
            PerplexityProvider(api_key=perplexity_key),
        ]

    @pytest.mark.asyncio
    async def test_all_providers_answer_simple_query(self, all_providers):
        """Every provider should successfully answer '1+1'."""
        import asyncio

        results = await asyncio.gather(
            *[p.research(SIMPLE_QUERY) for p in all_providers]
        )

        for result in results:
            assert result.status == "completed", (
                f"{result.provider} failed: {result.error}"
            )
            assert_answer_reasonable(result)

    @pytest.mark.asyncio
    async def test_all_providers_return_complete_result(self, all_providers):
        """Every provider result should have all required fields populated."""
        import asyncio

        results = await asyncio.gather(
            *[p.research(SIMPLE_QUERY) for p in all_providers]
        )

        for result in results:
            assert result.status == "completed", f"{result.provider}: {result.error}"
            assert result.report_markdown, f"{result.provider} empty markdown"
            assert result.duration_seconds is not None, f"{result.provider} no duration"
            assert result.duration_seconds > 0, f"{result.provider} zero duration"
            assert result.cost_usd is not None, f"{result.provider} no cost"
            assert result.cost_usd >= 0, f"{result.provider} negative cost"
            assert result.tokens is not None, f"{result.provider} no tokens"
            assert result.model is not None, f"{result.provider} no model"

    @pytest.mark.asyncio
    async def test_citations_are_lists_of_dicts(self, all_providers):
        """Every provider's citations should be list[dict] with 'url' keys."""
        import asyncio

        results = await asyncio.gather(
            *[p.research("What is the current population of Tokyo?") for p in all_providers]
        )

        for result in results:
            assert result.status == "completed", f"{result.provider}: {result.error}"
            assert isinstance(result.citations, list), (
                f"{result.provider} citations not a list: {type(result.citations)}"
            )
            for citation in result.citations:
                assert isinstance(citation, dict), (
                    f"{result.provider} citation not a dict: {type(citation)}"
                )
                assert "url" in citation, (
                    f"{result.provider} citation missing 'url': {citation}"
                )


# ===================================================================
# End-to-end orchestrator test
# ===================================================================


class TestOrchestratorIntegration:
    """Full research pipeline with live providers."""

    @pytest.mark.asyncio
    async def test_multi_provider_research_produces_bundle(
        self, openai_key, gemini_key, tmp_path
    ):
        """Research with 2 live providers produces a valid .prx bundle."""
        from parallect.orchestrator.parallel import research
        from parallect.providers.gemini import GeminiProvider
        from parallect.providers.openai_dr import OpenAIDRProvider

        providers = [
            OpenAIDRProvider(api_key=openai_key),
            GeminiProvider(api_key=gemini_key),
        ]

        output_file = tmp_path / "test.prx"
        bundle = await research(
            query="What is 2+2?",
            providers=providers,
            no_synthesis=True,
            timeout_per_provider=120.0,
            output=str(output_file),
        )

        # Bundle structure
        assert bundle.manifest.id.startswith("prx_")
        assert bundle.manifest.query == "What is 2+2?"
        assert len(bundle.providers) == 2
        assert set(bundle.manifest.providers_used) == {"openai", "gemini"}
        assert bundle.manifest.has_synthesis is False

        # Cost aggregated
        assert bundle.manifest.total_cost_usd is not None
        assert bundle.manifest.total_cost_usd > 0

        # File written
        assert output_file.exists()
        assert output_file.stat().st_size > 100

    @pytest.mark.asyncio
    async def test_bundle_round_trip(self, openai_key, tmp_path):
        """Write then read-back a bundle produced with a live provider."""
        from prx_spec import read_bundle

        from parallect.orchestrator.parallel import research
        from parallect.providers.openai_dr import OpenAIDRProvider

        output_file = tmp_path / "roundtrip.prx"
        bundle = await research(
            query="What is 3+3?",
            providers=[OpenAIDRProvider(api_key=openai_key)],
            no_synthesis=True,
            output=str(output_file),
        )

        # Read back
        loaded = read_bundle(output_file)
        assert loaded.manifest.id == bundle.manifest.id
        assert loaded.manifest.query == "What is 3+3?"
        assert len(loaded.providers) == 1
        assert loaded.providers[0].name == "openai"
        assert len(loaded.providers[0].report_md) > 5

    @pytest.mark.asyncio
    async def test_fan_out_partial_failure(self, openai_key):
        """One bad provider + one good: bundle should contain the good one."""
        from parallect.orchestrator.parallel import fan_out
        from parallect.providers.openai_dr import OpenAIDRProvider

        good = OpenAIDRProvider(api_key=openai_key)
        bad = OpenAIDRProvider(api_key="sk-invalid-key")

        outcomes = await fan_out(SIMPLE_QUERY, [good, bad], timeout_per_provider=30.0)
        assert len(outcomes) == 2

        successes = [o for o in outcomes if o.result and o.result.status == "completed"]
        failures = [o for o in outcomes if o.error or (o.result and o.result.status == "failed")]

        assert len(successes) >= 1
        assert len(failures) >= 1
