"""Tests for the agentic iterative research loop."""

from __future__ import annotations

import json

import pytest

from parallect.providers import ProviderResult
from parallect.research_loop.budget import BudgetExhausted, IterationBudget
from parallect.research_loop.evaluator import evaluate
from parallect.research_loop.executor import execute
from parallect.research_loop.loop import run_loop
from parallect.research_loop.planner import SubQuery, plan
from parallect.research_loop.prompts import PROMPT_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProvider:
    def __init__(self, name: str = "mock", report: str = "# Mock\n\nContent."):
        self._name = name
        self._report = report

    @property
    def name(self) -> str:
        return self._name

    async def research(self, query: str) -> ProviderResult:
        return ProviderResult(
            provider=self._name,
            status="completed",
            report_markdown=self._report,
            cost_usd=0.01,
            duration_seconds=0.05,
        )

    def estimate_cost(self, query: str) -> float:
        return 0.01

    def is_available(self) -> bool:
        return True


class FailingMockProvider:
    @property
    def name(self) -> str:
        return "failing"

    async def research(self, query: str) -> ProviderResult:
        raise RuntimeError("boom")

    def estimate_cost(self, query: str) -> float:
        return 0.01

    def is_available(self) -> bool:
        return True


def _mock_backend_spec():
    from parallect.backends import BackendSpec
    return BackendSpec(
        kind="openai",
        base_url="http://localhost:9999/v1",
        api_key="test-key",
        model="test-model",
        api_key_env="TEST_KEY",
    )


def _make_planner_response(sub_queries: list[str]) -> dict:
    payload = {
        "sub_queries": [
            {"query": sq, "target_sources": ["mock"], "rationale": "test"}
            for sq in sub_queries
        ]
    }
    return {
        "content": json.dumps(payload),
        "model": "test-model",
        "tokens": {"input": 100, "output": 50, "total": 150},
        "raw": {},
    }


def _make_evaluator_response(decision: str, gaps: list[str] | None = None) -> dict:
    payload = {
        "decision": decision,
        "rationale": f"test {decision}",
        "gaps": gaps or [],
        "suggested_queries": [],
    }
    return {
        "content": json.dumps(payload),
        "model": "test-model",
        "tokens": {"input": 100, "output": 50, "total": 150},
        "raw": {},
    }


# ---------------------------------------------------------------------------
# Budget tests
# ---------------------------------------------------------------------------


class TestIterationBudget:
    def test_initial_state(self):
        b = IterationBudget(cap_usd=1.0)
        assert b.spent == 0.0
        assert b.remaining == 1.0
        assert not b.exhausted
        assert b.breakdown == []

    def test_record(self):
        b = IterationBudget(cap_usd=1.0)
        b.record(0.3, "planner")
        assert b.spent == pytest.approx(0.3)
        assert b.remaining == pytest.approx(0.7)
        assert len(b.breakdown) == 1

    def test_record_multiple(self):
        b = IterationBudget(cap_usd=1.0)
        b.record(0.3, "a")
        b.record(0.4, "b")
        assert b.spent == pytest.approx(0.7)
        assert len(b.breakdown) == 2

    def test_budget_exhausted_raises(self):
        b = IterationBudget(cap_usd=0.5)
        with pytest.raises(BudgetExhausted, match="exhausted"):
            b.record(0.6, "too much")

    def test_exhausted_flag(self):
        b = IterationBudget(cap_usd=0.5)
        b.record(0.5, "exact")
        assert b.exhausted

    def test_can_afford(self):
        b = IterationBudget(cap_usd=1.0)
        b.record(0.8, "x")
        assert b.can_afford(0.2)
        assert not b.can_afford(0.3)

    def test_zero_cost_record(self):
        b = IterationBudget(cap_usd=1.0)
        b.record(0.0, "free")
        assert b.spent == 0.0
        assert not b.exhausted


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


class TestPlanner:
    @pytest.mark.asyncio
    async def test_plan_produces_sub_queries(self, monkeypatch):
        response = _make_planner_response(["sub1", "sub2", "sub3"])

        async def mock_dispatch(*args, **kwargs):
            return response

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_dispatch)

        result = await plan("test query", ["mock"], _mock_backend_spec())
        assert len(result.sub_queries) >= 2
        assert len(result.sub_queries) <= 5
        assert all(isinstance(sq, SubQuery) for sq in result.sub_queries)

    @pytest.mark.asyncio
    async def test_plan_references_declared_sources(self, monkeypatch):
        response = _make_planner_response(["about mock data"])

        async def mock_dispatch(*args, **kwargs):
            return response

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_dispatch)

        result = await plan("test", ["mock", "perplexity"], _mock_backend_spec())
        assert len(result.sub_queries) >= 2

    @pytest.mark.asyncio
    async def test_plan_clamps_to_2_min(self, monkeypatch):
        payload = {"sub_queries": [{"query": "only one"}]}
        response = {
            "content": json.dumps(payload),
            "tokens": {"input": 10, "output": 10, "total": 20},
            "raw": {},
        }

        async def mock_dispatch(*args, **kwargs):
            return response

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_dispatch)

        result = await plan("test", ["mock"], _mock_backend_spec())
        assert len(result.sub_queries) >= 2

    @pytest.mark.asyncio
    async def test_plan_clamps_to_5_max(self, monkeypatch):
        sqs = [f"sq{i}" for i in range(8)]
        response = _make_planner_response(sqs)

        async def mock_dispatch(*args, **kwargs):
            return response

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_dispatch)

        result = await plan("test", ["mock"], _mock_backend_spec())
        assert len(result.sub_queries) <= 5

    @pytest.mark.asyncio
    async def test_plan_fallback_on_invalid_json(self, monkeypatch):
        async def mock_dispatch(*args, **kwargs):
            return {"content": "not json at all", "tokens": None, "raw": {}}

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_dispatch)

        result = await plan("fallback query", ["mock"], _mock_backend_spec())
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0].query == "fallback query"

    @pytest.mark.asyncio
    async def test_plan_fallback_on_llm_failure(self, monkeypatch):
        async def mock_dispatch(*args, **kwargs):
            raise RuntimeError("LLM down")

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_dispatch)

        result = await plan("fallback query", ["mock"], _mock_backend_spec())
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0].query == "fallback query"

    @pytest.mark.asyncio
    async def test_plan_with_prior_iterations(self, monkeypatch):
        response = _make_planner_response(["gap filler 1", "gap filler 2"])

        async def mock_dispatch(*args, **kwargs):
            return response

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_dispatch)

        prior = [{"sub_queries": ["old q"], "evaluator_rationale": "gap: banking"}]
        result = await plan("test", ["mock"], _mock_backend_spec(), prior_iterations=prior)
        assert len(result.sub_queries) >= 2


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate_stop(self, monkeypatch):
        response = _make_evaluator_response("stop")

        async def mock_dispatch(*args, **kwargs):
            return response

        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_dispatch)

        result = await evaluate(
            "test", "results here", _mock_backend_spec(),
            iteration=1, max_iterations=3, budget_remaining=4.0, budget_total=5.0,
        )
        assert result.decision == "stop"

    @pytest.mark.asyncio
    async def test_evaluate_continue(self, monkeypatch):
        response = _make_evaluator_response("continue", gaps=["banking adoption"])

        async def mock_dispatch(*args, **kwargs):
            return response

        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_dispatch)

        result = await evaluate(
            "test", "results here", _mock_backend_spec(),
            iteration=1, max_iterations=3, budget_remaining=4.0, budget_total=5.0,
        )
        assert result.decision == "continue"
        assert "banking" in result.gaps[0]

    @pytest.mark.asyncio
    async def test_evaluate_malformed_json_defaults_stop(self, monkeypatch):
        async def mock_dispatch(*args, **kwargs):
            return {"content": "{{bad json", "tokens": None, "raw": {}}

        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_dispatch)

        result = await evaluate(
            "test", "results", _mock_backend_spec(),
            iteration=1, max_iterations=3, budget_remaining=4.0, budget_total=5.0,
        )
        assert result.decision == "stop"

    @pytest.mark.asyncio
    async def test_evaluate_failure_defaults_stop(self, monkeypatch):
        async def mock_dispatch(*args, **kwargs):
            raise RuntimeError("LLM down")

        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_dispatch)

        result = await evaluate(
            "test", "results", _mock_backend_spec(),
            iteration=1, max_iterations=3, budget_remaining=4.0, budget_total=5.0,
        )
        assert result.decision == "stop"

    @pytest.mark.asyncio
    async def test_evaluate_invalid_decision_defaults_stop(self, monkeypatch):
        payload = {"decision": "maybe", "rationale": "dunno"}

        async def mock_dispatch(*args, **kwargs):
            return {"content": json.dumps(payload), "tokens": None, "raw": {}}

        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_dispatch)

        result = await evaluate(
            "test", "results", _mock_backend_spec(),
            iteration=1, max_iterations=3, budget_remaining=4.0, budget_total=5.0,
        )
        assert result.decision == "stop"


# ---------------------------------------------------------------------------
# Executor tests
# ---------------------------------------------------------------------------


class TestExecutor:
    @pytest.mark.asyncio
    async def test_execute_multiple_providers(self):
        providers = [MockProvider("p1"), MockProvider("p2")]
        sub_queries = [SubQuery(query="q1"), SubQuery(query="q2")]
        result = await execute(sub_queries, providers, timeout=10.0)
        assert len(result.all_results) == 2
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_execute_merge_results(self):
        providers = [MockProvider("a", "Report A"), MockProvider("b", "Report B")]
        sub_queries = [SubQuery(query="test")]
        result = await execute(sub_queries, providers, timeout=10.0)
        assert len(result.all_results) == 2
        summaries = result.results_summary
        assert "a" in summaries or "b" in summaries

    @pytest.mark.asyncio
    async def test_execute_with_failing_provider(self):
        providers = [MockProvider("ok"), FailingMockProvider()]
        sub_queries = [SubQuery(query="test")]
        result = await execute(sub_queries, providers, timeout=10.0)
        assert len(result.all_results) == 1
        assert result.all_results[0].provider == "ok"

    @pytest.mark.asyncio
    async def test_execute_no_sources(self):
        providers = [MockProvider("p1")]
        sub_queries = [SubQuery(query="test")]
        result = await execute(sub_queries, providers, sources_raw=None, timeout=10.0)
        assert len(result.all_results) >= 1

    @pytest.mark.asyncio
    async def test_execute_duration_tracked(self):
        providers = [MockProvider("fast")]
        sub_queries = [SubQuery(query="test")]
        result = await execute(sub_queries, providers, timeout=10.0)
        assert result.duration_seconds >= 0


# ---------------------------------------------------------------------------
# Loop tests
# ---------------------------------------------------------------------------


class TestLoop:
    @pytest.mark.asyncio
    async def test_max_iterations_stop(self, monkeypatch):
        planner_resp = _make_planner_response(["sq1", "sq2"])
        eval_resp = _make_evaluator_response("continue", gaps=["more"])

        async def mock_plan_dispatch(*args, **kwargs):
            return planner_resp

        async def mock_eval_dispatch(*args, **kwargs):
            return eval_resp

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_plan_dispatch)
        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_eval_dispatch)

        # Mock synthesis
        async def mock_synth(*args, **kwargs):
            from parallect.synthesis.llm import SynthesisResult
            return SynthesisResult(report_markdown="# Synthesis", model="mock", cost_usd=0.01)

        monkeypatch.setattr("parallect.research_loop.synthesizer.synthesize", mock_synth)

        result = await run_loop(
            "test query",
            [MockProvider("p1")],
            max_iterations=2,
            budget_cap_usd=10.0,
            synthesize_with="anthropic",
            timeout=10.0,
        )
        assert len(result.iterations) == 2
        assert result.stop_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_evaluator_stop(self, monkeypatch):
        planner_resp = _make_planner_response(["sq1", "sq2"])
        eval_resp = _make_evaluator_response("stop")

        async def mock_plan_dispatch(*args, **kwargs):
            return planner_resp

        async def mock_eval_dispatch(*args, **kwargs):
            return eval_resp

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_plan_dispatch)
        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_eval_dispatch)

        async def mock_synth(*args, **kwargs):
            from parallect.synthesis.llm import SynthesisResult
            return SynthesisResult(report_markdown="# Synthesis", model="mock", cost_usd=0.01)

        monkeypatch.setattr("parallect.research_loop.synthesizer.synthesize", mock_synth)

        result = await run_loop(
            "test query",
            [MockProvider("p1")],
            max_iterations=5,
            budget_cap_usd=10.0,
            synthesize_with="anthropic",
            timeout=10.0,
        )
        assert len(result.iterations) == 1
        assert result.stop_reason == "evaluator_stop"

    @pytest.mark.asyncio
    async def test_budget_stop(self, monkeypatch):
        planner_resp = _make_planner_response(["sq1", "sq2"])
        eval_resp = _make_evaluator_response("continue")

        async def mock_plan_dispatch(*args, **kwargs):
            return planner_resp

        async def mock_eval_dispatch(*args, **kwargs):
            return eval_resp

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_plan_dispatch)
        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_eval_dispatch)

        async def mock_synth(*args, **kwargs):
            from parallect.synthesis.llm import SynthesisResult
            return SynthesisResult(report_markdown="# Synthesis", model="mock", cost_usd=0.0)

        monkeypatch.setattr("parallect.research_loop.synthesizer.synthesize", mock_synth)

        result = await run_loop(
            "test query",
            [MockProvider("p1")],
            max_iterations=10,
            budget_cap_usd=0.02,
            synthesize_with="anthropic",
            timeout=10.0,
        )
        assert result.stop_reason in ("budget_exhausted", "evaluator_stop", "max_iterations")
        assert result.total_cost_usd <= 0.05  # small tolerance

    @pytest.mark.asyncio
    async def test_no_synthesis_when_disabled(self, monkeypatch):
        planner_resp = _make_planner_response(["sq1", "sq2"])
        eval_resp = _make_evaluator_response("stop")

        async def mock_plan_dispatch(*args, **kwargs):
            return planner_resp

        async def mock_eval_dispatch(*args, **kwargs):
            return eval_resp

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_plan_dispatch)
        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_eval_dispatch)

        result = await run_loop(
            "test query",
            [MockProvider("p1")],
            max_iterations=3,
            budget_cap_usd=10.0,
            synthesize_with=None,
            timeout=10.0,
        )
        assert result.synthesis is None

    @pytest.mark.asyncio
    async def test_streaming_callbacks(self, monkeypatch):
        planner_resp = _make_planner_response(["sq1", "sq2"])
        eval_resp = _make_evaluator_response("stop")

        async def mock_plan_dispatch(*args, **kwargs):
            return planner_resp

        async def mock_eval_dispatch(*args, **kwargs):
            return eval_resp

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_plan_dispatch)
        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_eval_dispatch)

        messages = []
        await run_loop(
            "test query",
            [MockProvider("p1")],
            max_iterations=3,
            budget_cap_usd=10.0,
            synthesize_with=None,
            timeout=10.0,
            on_status=lambda msg: messages.append(msg),
        )
        assert len(messages) > 0
        assert any("[it 1/" in m for m in messages)

    @pytest.mark.asyncio
    async def test_no_status_callback_is_noop(self, monkeypatch):
        planner_resp = _make_planner_response(["sq1", "sq2"])
        eval_resp = _make_evaluator_response("stop")

        async def mock_plan_dispatch(*args, **kwargs):
            return planner_resp

        async def mock_eval_dispatch(*args, **kwargs):
            return eval_resp

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_plan_dispatch)
        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_eval_dispatch)

        # Should not raise even without on_status
        result = await run_loop(
            "test query",
            [MockProvider("p1")],
            max_iterations=1,
            budget_cap_usd=10.0,
            synthesize_with=None,
            timeout=10.0,
        )
        assert len(result.iterations) == 1


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------


class TestProvenance:
    @pytest.mark.asyncio
    async def test_provenance_dict_shape(self, monkeypatch):
        planner_resp = _make_planner_response(["sq1", "sq2"])
        eval_resp = _make_evaluator_response("stop")

        async def mock_plan_dispatch(*args, **kwargs):
            return planner_resp

        async def mock_eval_dispatch(*args, **kwargs):
            return eval_resp

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_plan_dispatch)
        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_eval_dispatch)

        result = await run_loop(
            "test query",
            [MockProvider("p1")],
            max_iterations=2,
            budget_cap_usd=10.0,
            synthesize_with=None,
            timeout=10.0,
        )
        prov = result.provenance_dict()
        assert prov["query"] == "test query"
        assert prov["total_iterations"] >= 1
        assert "stop_reason" in prov
        assert "iterations" in prov
        assert isinstance(prov["iterations"], list)

        it0 = prov["iterations"][0]
        assert "iteration" in it0
        assert "sub_queries" in it0
        assert "results_count" in it0
        assert "evaluator_decision" in it0
        assert "cost_usd" in it0
        assert "duration_seconds" in it0

    @pytest.mark.asyncio
    async def test_provenance_multiple_iterations(self, monkeypatch):
        planner_resp = _make_planner_response(["sq1", "sq2"])

        call_count = {"n": 0}

        async def mock_plan_dispatch(*args, **kwargs):
            return planner_resp

        async def mock_eval_dispatch(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                return _make_evaluator_response("stop")
            return _make_evaluator_response("continue", gaps=["gap1"])

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_plan_dispatch)
        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_eval_dispatch)

        result = await run_loop(
            "test query",
            [MockProvider("p1")],
            max_iterations=5,
            budget_cap_usd=10.0,
            synthesize_with=None,
            timeout=10.0,
        )
        prov = result.provenance_dict()
        assert prov["total_iterations"] >= 2


# ---------------------------------------------------------------------------
# End-to-end test (mocked LLMs + in-memory plugins)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_three_iterations_with_synthesis(self, monkeypatch):
        planner_resp = _make_planner_response(["sq1", "sq2"])

        call_count = {"n": 0}

        async def mock_plan_dispatch(*args, **kwargs):
            return planner_resp

        async def mock_eval_dispatch(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                return _make_evaluator_response("stop")
            return _make_evaluator_response("continue", gaps=[f"gap{call_count['n']}"])

        monkeypatch.setattr("parallect.research_loop.planner._dispatch", mock_plan_dispatch)
        monkeypatch.setattr("parallect.research_loop.evaluator._dispatch", mock_eval_dispatch)

        async def mock_synth(query, results, **kwargs):
            from parallect.synthesis.llm import SynthesisResult
            return SynthesisResult(
                report_markdown=f"# Synthesis\n\nBased on {len(results)} results.",
                model="mock",
                cost_usd=0.02,
            )

        monkeypatch.setattr("parallect.research_loop.synthesizer.synthesize", mock_synth)

        providers = [MockProvider("p1", "Report 1"), MockProvider("p2", "Report 2")]
        messages = []

        result = await run_loop(
            "comprehensive test query",
            providers,
            max_iterations=5,
            budget_cap_usd=10.0,
            synthesize_with="anthropic",
            timeout=10.0,
            on_status=lambda msg: messages.append(msg),
        )

        assert len(result.iterations) == 3
        assert result.synthesis is not None
        assert "Synthesis" in result.synthesis.report_markdown
        assert len(result.all_results) == 6  # 2 providers * 3 iterations
        assert result.total_cost_usd > 0
        assert result.stop_reason == "evaluator_stop"
        assert len(messages) > 0

        prov = result.provenance_dict()
        assert prov["total_iterations"] == 3
        assert len(prov["iterations"]) == 3


# ---------------------------------------------------------------------------
# Prompts version test
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_prompt_version_exists(self):
        assert PROMPT_VERSION == "1.0"
