"""Tests for parallect.api — SaaS client + enhance flow."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from parallect.api import (
    EnhanceJob,
    InsufficientBalanceError,
    JobFailedError,
    JobStatus,
    ParallectAPIClient,
    ParallectAPIError,
    RateLimitError,
    ServerError,
    UnauthorizedError,
    enhance_bundle,
)


def _mock_transport(handler):
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# JobStatus
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_from_flat(self):
        s = JobStatus.from_api({
            "id": "job_1",
            "status": "running",
            "pipelinePhase": "authoring",
            "sourceCount": 47,
            "synthesisMarkdown": "x" * 100,
        })
        assert s.job_id == "job_1"
        assert s.status == "running"
        assert s.pipeline_phase == "authoring"
        assert s.source_count == 47
        assert s.synthesis_chars == 100

    def test_from_nested(self):
        s = JobStatus.from_api({"job": {"id": "j", "status": "completed"}})
        assert s.job_id == "j"
        assert s.status == "completed"


# ---------------------------------------------------------------------------
# submit_thread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSubmitThread:
    async def test_submit_returns_thread_and_job(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/threads"
            assert request.headers["authorization"] == "Bearer par_test"
            body = json.loads(request.content.decode())
            assert body["message"] == "hello"
            assert body["tier"] == "max"
            return httpx.Response(
                200,
                json={"thread": {"id": "t_1"}, "job": {"id": "j_1", "status": "queued"}},
            )

        client = ParallectAPIClient("par_test", transport=_mock_transport(handler))
        result = await client.submit_thread("hello", "max")
        assert result["thread"]["id"] == "t_1"
        assert result["job"]["id"] == "j_1"

    async def test_submit_401(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "bad key"})

        client = ParallectAPIClient("par_bad", transport=_mock_transport(handler))
        with pytest.raises(UnauthorizedError):
            await client.submit_thread("q", "normal")

    async def test_submit_402(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(402, json={"error": "no credits"})

        client = ParallectAPIClient("par_x", transport=_mock_transport(handler))
        with pytest.raises(InsufficientBalanceError):
            await client.submit_thread("q", "normal")

    async def test_submit_500(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        client = ParallectAPIClient("par_x", transport=_mock_transport(handler))
        with pytest.raises(ServerError):
            await client.submit_thread("q", "normal")

    async def test_submit_retries_on_429(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "rl"})
            return httpx.Response(200, json={"thread": {"id": "t"}, "job": {"id": "j"}})

        client = ParallectAPIClient("par_x", transport=_mock_transport(handler))
        result = await client.submit_thread("q", "normal")
        assert result["job"]["id"] == "j"
        assert calls["n"] >= 2

    async def test_submit_429_exhausted(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "rl"})

        client = ParallectAPIClient("par_x", transport=_mock_transport(handler))
        with pytest.raises(RateLimitError):
            await client.submit_thread("q", "normal")

    async def test_submit_includes_providers(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"thread": {"id": "t"}, "job": {"id": "j"}})

        client = ParallectAPIClient("k", transport=_mock_transport(handler))
        await client.submit_thread("q", "normal", providers=["perplexity", "gemini"])
        assert seen["body"]["providers"] == ["perplexity", "gemini"]


# ---------------------------------------------------------------------------
# poll_until_done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPollUntilDone:
    async def test_transitions_to_completed(self):
        states = iter([
            {"id": "j", "status": "queued"},
            {"id": "j", "status": "running", "pipelinePhase": "authoring"},
            {"id": "j", "status": "completed", "synthesisMarkdown": "# done"},
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=next(states))

        client = ParallectAPIClient("k", transport=_mock_transport(handler))
        seen: list = []
        final = await client.poll_until_done(
            "j", poll_interval=0, on_update=lambda s: seen.append(s.status)
        )
        assert final.status == "completed"
        assert seen[-1] == "completed"

    async def test_failed_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "j", "status": "failed", "error": "oops"})

        client = ParallectAPIClient("k", transport=_mock_transport(handler))
        with pytest.raises(JobFailedError):
            await client.poll_until_done("j", poll_interval=0)

    async def test_timeout(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "j", "status": "running"})

        client = ParallectAPIClient("k", transport=_mock_transport(handler))
        from parallect.api import JobTimeoutError

        with pytest.raises(JobTimeoutError):
            await client.poll_until_done("j", poll_interval=0, max_wait_s=0.1)


# ---------------------------------------------------------------------------
# download_bundle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDownloadBundle:
    async def test_200_writes_file(self, tmp_path: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"PKBUNDLE")

        client = ParallectAPIClient("k", transport=_mock_transport(handler))
        dest = tmp_path / "out.prx"
        ok, path = await client.download_bundle("j", dest)
        assert ok is True
        assert path.read_bytes() == b"PKBUNDLE"

    async def test_404_fallback_signal(self, tmp_path: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "not yet"})

        client = ParallectAPIClient("k", transport=_mock_transport(handler))
        dest = tmp_path / "out.prx"
        ok, _ = await client.download_bundle("j", dest)
        assert ok is False
        assert not dest.exists()

    async def test_500_raises(self, tmp_path: Path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        client = ParallectAPIClient("k", transport=_mock_transport(handler))
        with pytest.raises(ServerError):
            await client.download_bundle("j", tmp_path / "out.prx")


# ---------------------------------------------------------------------------
# enhance_bundle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEnhanceBundle:
    async def test_happy_path(self, tmp_path: Path, monkeypatch):
        src = tmp_path / "in.prx"
        src.write_bytes(b"INPUT")

        call_log: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            call_log.append(f"{request.method} {path}")
            if path == "/api/enhance/initiate":
                return httpx.Response(200, json={
                    "job_id": "j1",
                    "upload_url": "https://upload.example.com/x",
                    "storage_key": "k1",
                })
            if path == "/api/enhance/j1/start":
                return httpx.Response(200, json={"ok": True})
            if path == "/api/enhance/j1":
                return httpx.Response(200, json={
                    "complete": True,
                    "download_url": "https://dl.example.com/out",
                })
            if path == "/x":
                return httpx.Response(200)
            if path == "/out":
                return httpx.Response(200, content=b"ENHANCED")
            return httpx.Response(404)

        # Patch httpx.AsyncClient to use the mock transport
        orig = httpx.AsyncClient

        def patched(*args, **kwargs):
            kwargs["transport"] = _mock_transport(handler)
            return orig(*args, **kwargs)

        monkeypatch.setattr("parallect.api.httpx.AsyncClient", patched)

        result = await enhance_bundle(src, "par_k", tier="standard")
        assert isinstance(result, EnhanceJob)
        assert result.status == "completed"
        assert result.enhanced_path is not None
        assert result.enhanced_path.read_bytes() == b"ENHANCED"

    async def test_unauthorized(self, tmp_path: Path, monkeypatch):
        src = tmp_path / "in.prx"
        src.write_bytes(b"x")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "bad"})

        orig = httpx.AsyncClient

        def patched(*args, **kwargs):
            kwargs["transport"] = _mock_transport(handler)
            return orig(*args, **kwargs)

        monkeypatch.setattr("parallect.api.httpx.AsyncClient", patched)

        with pytest.raises(UnauthorizedError):
            await enhance_bundle(src, "par_bad")
