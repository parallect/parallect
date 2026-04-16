"""CLI-level tests for `parallect research` (SaaS + BYOK paths) and agent-help."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from parallect.cli import AGENT_HELP, app


runner = CliRunner()


# ---------------------------------------------------------------------------
# --agent-help
# ---------------------------------------------------------------------------


class TestAgentHelp:
    def test_agent_help_prints_reference(self):
        result = runner.invoke(app, ["--agent-help"])
        assert result.exit_code == 0
        assert "parallect" in result.stdout
        assert "Routing rules" in result.stdout
        assert "PARALLECT_API_KEY" in result.stdout
        assert "--tier" in result.stdout

    def test_agent_help_hidden_from_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--agent-help" not in result.stdout

    def test_agent_help_reasonable_length(self):
        # under ~150 lines target
        assert AGENT_HELP.count("\n") < 160


# ---------------------------------------------------------------------------
# SaaS path — end-to-end with mocked HTTP
# ---------------------------------------------------------------------------


class _MockDriver:
    """Sequences responses per-path for SaaS tests."""

    def __init__(self, handler):
        self.handler = handler

    def patch(self, monkeypatch):
        orig = httpx.AsyncClient
        transport = httpx.MockTransport(self.handler)

        def patched(*args, **kwargs):
            kwargs["transport"] = transport
            return orig(*args, **kwargs)

        monkeypatch.setattr("parallect.api.httpx.AsyncClient", patched)


def _seq_handler(job_states: list[dict], *, bundle_status: int = 200, bundle_body: bytes = b"PRX"):
    """Returns a handler that walks through job_states for /jobs/<id> calls."""
    idx = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/threads":
            return httpx.Response(
                200,
                json={"thread": {"id": "t_1"}, "job": {"id": "j_1", "status": "queued"}},
            )
        if path == "/api/v1/jobs/j_1":
            i = min(idx["n"], len(job_states) - 1)
            idx["n"] += 1
            return httpx.Response(200, json=job_states[i])
        if path == "/api/v1/jobs/j_1/prx":
            if bundle_status == 200:
                return httpx.Response(200, content=bundle_body)
            return httpx.Response(bundle_status, json={"error": "n/a"})
        return httpx.Response(404, json={"error": "unknown"})

    return handler


class TestSaasPath:
    def test_success_downloads_bundle(self, tmp_path: Path, monkeypatch):
        handler = _seq_handler(
            job_states=[
                {"id": "j_1", "status": "running", "pipelinePhase": "authoring"},
                {"id": "j_1", "status": "completed", "totalCostCents": 42},
            ],
        )
        _MockDriver(handler).patch(monkeypatch)

        out = tmp_path / "out.prx"
        result = runner.invoke(
            app,
            [
                "research", "test query",
                "--tier", "max",
                "--api-key", "par_live_abcd1234_secret",
                "--api-url", "http://api.test",
                "--output", str(out),
                "--poll-interval", "0",
            ],
        )
        assert result.exit_code == 0, result.stdout + "\n" + (result.stderr or "")
        assert out.exists()
        assert out.read_bytes() == b"PRX"
        # Banner shows short key, not full secret
        assert "par_live_abcd1234" in result.stdout
        assert "secret" not in result.stdout

    def test_404_bundle_falls_back_to_minimal(self, tmp_path: Path, monkeypatch):
        handler = _seq_handler(
            job_states=[
                {
                    "id": "j_1",
                    "status": "completed",
                    "synthesisMarkdown": "# hello",
                    "evidenceGraphJson": "{}",
                },
            ],
            bundle_status=404,
        )
        _MockDriver(handler).patch(monkeypatch)

        out = tmp_path / "out.prx"
        result = runner.invoke(
            app,
            [
                "research", "q",
                "--tier", "normal",
                "--api-key", "par_live_xy_z",
                "--api-url", "http://api.test",
                "--output", str(out),
                "--poll-interval", "0",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "falling back" in result.stdout.lower() or "404" in result.stdout
        assert out.exists()

    def test_401_unauthorized(self, tmp_path: Path, monkeypatch):
        def handler(request):
            return httpx.Response(401, json={"error": "bad key"})

        _MockDriver(handler).patch(monkeypatch)
        result = runner.invoke(
            app,
            [
                "research", "q",
                "--api-key", "par_live_bad_x",
                "--api-url", "http://api.test",
            ],
        )
        assert result.exit_code == 1
        assert "unauthorized" in result.stdout.lower()

    def test_402_insufficient_balance(self, tmp_path: Path, monkeypatch):
        def handler(request):
            return httpx.Response(402, json={"error": "no credits"})

        _MockDriver(handler).patch(monkeypatch)
        result = runner.invoke(
            app,
            [
                "research", "q",
                "--api-key", "par_live_x_y",
                "--api-url", "http://api.test",
            ],
        )
        assert result.exit_code == 2
        assert "insufficient" in result.stdout.lower()

    def test_500_server_error(self, monkeypatch):
        def handler(request):
            return httpx.Response(500, json={"error": "boom"})

        _MockDriver(handler).patch(monkeypatch)
        result = runner.invoke(
            app,
            [
                "research", "q",
                "--api-key", "par_live_x_y",
                "--api-url", "http://api.test",
            ],
        )
        assert result.exit_code == 1

    def test_poll_timeout(self, tmp_path: Path, monkeypatch):
        def handler(request):
            path = request.url.path
            if path == "/api/v1/threads":
                return httpx.Response(
                    200,
                    json={"thread": {"id": "t"}, "job": {"id": "j_1", "status": "queued"}},
                )
            if path == "/api/v1/jobs/j_1":
                return httpx.Response(200, json={"id": "j_1", "status": "running"})
            return httpx.Response(404)

        _MockDriver(handler).patch(monkeypatch)

        # Patch poll_until_done to use a tiny max_wait_s
        from parallect.api import ParallectAPIClient
        orig = ParallectAPIClient.poll_until_done

        async def quick(self, job_id, *, poll_interval=15.0, max_wait_s=None, on_update=None):
            return await orig(self, job_id, poll_interval=0, max_wait_s=0.1, on_update=on_update)

        monkeypatch.setattr(ParallectAPIClient, "poll_until_done", quick)

        result = runner.invoke(
            app,
            [
                "research", "q",
                "--api-key", "par_live_x_y",
                "--api-url", "http://api.test",
                "--poll-interval", "0",
            ],
        )
        assert result.exit_code == 1
        assert "timeout" in result.stdout.lower()


# ---------------------------------------------------------------------------
# BYOK path — regression + --local override
# ---------------------------------------------------------------------------


class TestByokPath:
    def test_local_flag_forces_byok_even_with_env_key(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("PARALLECT_API_KEY", "par_live_abc_sekret")

        # Patch the BYOK entry point so we can assert it was called (no real providers).
        called = {"byok": False}

        async def fake_byok(**kwargs):
            called["byok"] = True
            # Simulate "no providers" early exit
            from typer import Exit
            raise Exit(code=0)

        monkeypatch.setattr("parallect.cli.research._run_byok", fake_byok)

        result = runner.invoke(
            app,
            ["research", "q", "--local", "--tier", "normal"],
        )
        assert called["byok"] is True
        assert result.exit_code == 0

    def test_no_key_routes_byok(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_API_KEY", raising=False)
        called = {"byok": False, "saas": False}

        async def fake_byok(**kwargs):
            called["byok"] = True
            from typer import Exit
            raise Exit(code=0)

        async def fake_saas(**kwargs):
            called["saas"] = True
            from typer import Exit
            raise Exit(code=0)

        monkeypatch.setattr("parallect.cli.research._run_byok", fake_byok)
        monkeypatch.setattr("parallect.cli.research._run_saas", fake_saas)

        result = runner.invoke(app, ["research", "q"])
        assert result.exit_code == 0
        assert called["byok"] is True
        assert called["saas"] is False

    def test_api_key_routes_saas(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_API_KEY", raising=False)
        called = {"byok": False, "saas": False}

        async def fake_byok(**kwargs):
            called["byok"] = True
            from typer import Exit
            raise Exit(code=0)

        async def fake_saas(**kwargs):
            called["saas"] = True
            from typer import Exit
            raise Exit(code=0)

        monkeypatch.setattr("parallect.cli.research._run_byok", fake_byok)
        monkeypatch.setattr("parallect.cli.research._run_saas", fake_saas)

        result = runner.invoke(
            app, ["research", "q", "--api-key", "par_live_a_b"]
        )
        assert result.exit_code == 0
        assert called["saas"] is True
        assert called["byok"] is False


# ---------------------------------------------------------------------------
# Deprecation shim for --deep
# ---------------------------------------------------------------------------


class TestDeepDeprecation:
    def test_deep_warns_and_routes_to_deep_tier(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_API_KEY", raising=False)
        captured = {}

        async def fake_byok(**kwargs):
            captured["tier"] = kwargs["tier_cfg"].name
            from typer import Exit
            raise Exit(code=0)

        monkeypatch.setattr("parallect.cli.research._run_byok", fake_byok)

        result = runner.invoke(app, ["research", "q", "--deep"])
        assert result.exit_code == 0
        assert captured["tier"] == "deep"
        assert "deprecated" in result.stdout.lower()

    def test_explicit_tier_wins_over_deep(self, monkeypatch):
        monkeypatch.delenv("PARALLECT_API_KEY", raising=False)
        captured = {}

        async def fake_byok(**kwargs):
            captured["tier"] = kwargs["tier_cfg"].name
            from typer import Exit
            raise Exit(code=0)

        monkeypatch.setattr("parallect.cli.research._run_byok", fake_byok)

        result = runner.invoke(app, ["research", "q", "--deep", "--tier", "lite"])
        assert result.exit_code == 0
        assert captured["tier"] == "lite"


# ---------------------------------------------------------------------------
# Banner wording
# ---------------------------------------------------------------------------


class TestBanner:
    def test_saas_banner(self, tmp_path: Path, monkeypatch):
        handler = _seq_handler(
            job_states=[{"id": "j_1", "status": "completed"}],
        )
        _MockDriver(handler).patch(monkeypatch)

        result = runner.invoke(
            app,
            [
                "research", "q",
                "--tier", "max",
                "--api-key", "par_live_8906_secret",
                "--api-url", "http://api.test",
                "--output", str(tmp_path / "out.prx"),
                "--poll-interval", "0",
            ],
        )
        assert "Running on Parallect API" in result.stdout
        assert "tier: max" in result.stdout
        assert "par_live_8906" in result.stdout


# ---------------------------------------------------------------------------
# enhance CLI
# ---------------------------------------------------------------------------


class TestEnhanceCli:
    def test_missing_api_key_exits(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("PARALLECT_API_KEY", raising=False)
        # Use a fresh settings load that has no parallect_api_key
        from parallect.config_mod.settings import ParallectSettings

        monkeypatch.setattr(
            ParallectSettings, "load", classmethod(lambda cls: cls())
        )

        src = tmp_path / "b.prx"
        src.write_bytes(b"x")
        result = runner.invoke(app, ["enhance", str(src)])
        assert result.exit_code == 1
        assert "api key" in result.stdout.lower()

    def test_file_not_found(self, monkeypatch):
        result = runner.invoke(
            app, ["enhance", "/nonexistent/nope.prx", "--api-key", "par_k"]
        )
        assert result.exit_code == 1

    def test_enhance_calls_api(self, tmp_path: Path, monkeypatch):
        src = tmp_path / "b.prx"
        src.write_bytes(b"x")

        from parallect.api import EnhanceJob

        async def fake_enhance(bundle_path, api_key, tier="standard", **kwargs):
            return EnhanceJob(
                job_id="j",
                status="completed",
                enhanced_path=tmp_path / "b.enhanced.prx",
            )

        monkeypatch.setattr("parallect.api.enhance_bundle", fake_enhance)
        (tmp_path / "b.enhanced.prx").write_bytes(b"y")

        result = runner.invoke(
            app, ["enhance", str(src), "--api-key", "par_k", "--tier", "premium"]
        )
        assert result.exit_code == 0
        assert "enhancement complete" in result.stdout.lower()


class TestResolveProvidersTimeout:
    """--timeout must flow into each provider's HTTP client, not just the
    orchestrator's asyncio wrapper. A 120s httpx timeout bites before a 600s
    asyncio timeout, silently returning a failed provider result."""

    def _fake_settings(self):
        class _S:
            perplexity_api_key = "pplx-test"
            google_api_key = "g"
            openai_api_key = "o"
            xai_api_key = "x"
            anthropic_api_key = "a"
            providers = ["perplexity", "gemini", "openai", "grok", "anthropic"]
            ollama_host = "http://localhost:11434"
            ollama_default_model = "llama3.2"
            lmstudio_host = "http://localhost:1234"
            lmstudio_default_model = "default"

        return _S()

    def test_timeout_propagates_to_all_providers(self):
        from parallect.cli.research import _resolve_providers

        instances = _resolve_providers(
            providers_str="perplexity,gemini,openai,grok,anthropic",
            local=False,
            settings=self._fake_settings(),
            timeout=450.0,
        )
        assert len(instances) == 5
        for inst in instances:
            assert inst.timeout == 450.0, (
                f"{inst.name} has timeout={inst.timeout}, expected 450.0"
            )

    def test_default_timeout_is_600_when_none(self):
        """Covers the case where the CLI default (120s) does NOT override —
        previously Perplexity's own 120s default cut off sonar-deep-research
        runs that legitimately take 2–5 minutes."""
        from parallect.cli.research import _resolve_providers

        instances = _resolve_providers(
            providers_str="perplexity",
            local=False,
            settings=self._fake_settings(),
            timeout=None,
        )
        assert instances[0].timeout == 600.0

    def test_zero_timeout_falls_back_to_default(self):
        from parallect.cli.research import _resolve_providers

        instances = _resolve_providers(
            providers_str="perplexity",
            local=False,
            settings=self._fake_settings(),
            timeout=0.0,
        )
        assert instances[0].timeout == 600.0
