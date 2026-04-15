"""Parallect API client.

Covers the enhance flow (legacy) and the SaaS research flow
(threads + jobs + bundle download).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

PARALLECT_API_URL = "https://api.parallect.ai"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ParallectAPIError(Exception):
    """Base error for Parallect API calls."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class UnauthorizedError(ParallectAPIError):
    """401 — bad or missing API key."""


class InsufficientBalanceError(ParallectAPIError):
    """402 — account has no credits."""


class RateLimitError(ParallectAPIError):
    """429 — hit rate limit."""


class ServerError(ParallectAPIError):
    """5xx server error."""


class JobFailedError(ParallectAPIError):
    """Job reported status=failed."""


class JobTimeoutError(ParallectAPIError):
    """Polling timed out."""


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        body = response.json()
        detail = body.get("error") or body.get("message") or str(body)
    except Exception:
        detail = response.text or f"HTTP {response.status_code}"
    code = response.status_code
    if code == 401:
        raise UnauthorizedError(f"Unauthorized: {detail}", code)
    if code == 402:
        raise InsufficientBalanceError(f"Insufficient balance: {detail}", code)
    if code == 429:
        raise RateLimitError(f"Rate limited: {detail}", code)
    if 500 <= code < 600:
        raise ServerError(f"Server error ({code}): {detail}", code)
    raise ParallectAPIError(f"HTTP {code}: {detail}", code)


# ---------------------------------------------------------------------------
# Enhance (legacy, retained)
# ---------------------------------------------------------------------------


@dataclass
class EnhanceJob:
    """Result of an enhance request."""

    job_id: str
    status: str
    enhanced_path: Path | None = None


async def enhance_bundle(
    bundle_path: Path,
    api_key: str,
    tier: str = "standard",
    api_url: str = PARALLECT_API_URL,
    *,
    max_retries: int = 3,
) -> EnhanceJob:
    """Upload a .prx bundle to Parallect API for premium enhancement."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await _request_with_retry(
            client,
            "POST",
            f"{api_url}/api/enhance/initiate",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"tier": tier},
            max_retries=max_retries,
        )
        _raise_for_status(response)
        job_data = response.json()

        with open(bundle_path, "rb") as f:
            upload_resp = await client.put(job_data["upload_url"], content=f.read())
        _raise_for_status(upload_resp)

        start_resp = await _request_with_retry(
            client,
            "POST",
            f"{api_url}/api/enhance/{job_data['job_id']}/start",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"storage_key": job_data["storage_key"]},
            max_retries=max_retries,
        )
        _raise_for_status(start_resp)

        while True:
            status_response = await _request_with_retry(
                client,
                "GET",
                f"{api_url}/api/enhance/{job_data['job_id']}",
                headers={"Authorization": f"Bearer {api_key}"},
                max_retries=max_retries,
            )
            _raise_for_status(status_response)
            status = status_response.json()

            if status.get("complete"):
                enhanced_path = bundle_path.with_suffix(".enhanced.prx")
                download = await client.get(status["download_url"])
                _raise_for_status(download)
                enhanced_path.write_bytes(download.content)

                return EnhanceJob(
                    job_id=job_data["job_id"],
                    status="completed",
                    enhanced_path=enhanced_path,
                )

            await asyncio.sleep(status.get("retry_after", 5))


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    **kwargs: Any,
) -> httpx.Response:
    """Issue a request, retrying on 429/503 with exponential backoff."""
    last_resp: httpx.Response | None = None
    for attempt in range(max_retries):
        resp = await client.request(method, url, **kwargs)
        if resp.status_code in (429, 503) and attempt < max_retries - 1:
            last_resp = resp
            delay = float(resp.headers.get("retry-after", 2 ** attempt))
            await asyncio.sleep(delay)
            continue
        return resp
    assert last_resp is not None
    return last_resp


# ---------------------------------------------------------------------------
# SaaS research flow
# ---------------------------------------------------------------------------


@dataclass
class JobStatus:
    """Snapshot of a job's state."""

    job_id: str
    status: str
    thread_id: str | None = None
    pipeline_phase: str | None = None
    source_count: int | None = None
    synthesis_chars: int | None = None
    synthesis_markdown: str | None = None
    evidence_graph_json: str | None = None
    total_cost_cents: int | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> JobStatus:
        job = data.get("job") or data
        synth = job.get("synthesisMarkdown")
        ev = job.get("evidenceGraphJson")
        sources = job.get("sourceCount")
        if sources is None and isinstance(ev, str):
            # best-effort: count "id" occurrences
            sources = ev.count('"id"') if ev else None
        return cls(
            job_id=job.get("id") or job.get("jobId") or "",
            status=job.get("status", "unknown"),
            thread_id=job.get("threadId"),
            pipeline_phase=job.get("pipelinePhase"),
            source_count=sources,
            synthesis_chars=len(synth) if isinstance(synth, str) else None,
            synthesis_markdown=synth if isinstance(synth, str) else None,
            evidence_graph_json=ev if isinstance(ev, str) else None,
            total_cost_cents=job.get("totalCostCents"),
            error=job.get("error"),
            raw=job,
        )


class ParallectAPIClient:
    """Thin HTTP client for the hosted Parallect research API."""

    def __init__(
        self,
        api_key: str,
        api_url: str = PARALLECT_API_URL,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.api_url,
            timeout=self._timeout,
            transport=self._transport,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def submit_thread(
        self, message: str, tier: str, *, providers: list[str] | None = None
    ) -> dict[str, Any]:
        """POST /api/v1/threads — returns the parsed JSON body (thread + job)."""
        payload: dict[str, Any] = {"message": message, "tier": tier}
        if providers:
            payload["providers"] = providers
        async with self._client() as client:
            resp = await _request_with_retry(
                client, "POST", "/api/v1/threads", json=payload
            )
            _raise_for_status(resp)
            return resp.json()

    async def get_job(self, job_id: str) -> JobStatus:
        async with self._client() as client:
            resp = await _request_with_retry(
                client, "GET", f"/api/v1/jobs/{job_id}"
            )
            _raise_for_status(resp)
            return JobStatus.from_api(resp.json())

    async def download_bundle(
        self, job_id: str, dest: Path
    ) -> tuple[bool, Path]:
        """Try to download the PRX bundle. Returns (ok, path).

        On 404, returns (False, dest) without writing; caller may fall back
        to a minimal bundle assembled from the job status payload.
        """
        async with self._client() as client:
            resp = await client.get(f"/api/v1/jobs/{job_id}/prx")
            if resp.status_code == 404:
                return False, dest
            _raise_for_status(resp)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True, dest

    async def poll_until_done(
        self,
        job_id: str,
        *,
        poll_interval: float = 15.0,
        max_wait_s: float | None = None,
        on_update: "callable[[JobStatus], None] | None" = None,
    ) -> JobStatus:
        """Poll a job until status is completed/failed or timeout."""
        start = asyncio.get_event_loop().time()
        while True:
            status = await self.get_job(job_id)
            if on_update is not None:
                try:
                    on_update(status)
                except Exception:
                    pass
            if status.status == "completed":
                return status
            if status.status == "failed":
                raise JobFailedError(status.error or "job failed")
            if max_wait_s is not None:
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed > max_wait_s:
                    raise JobTimeoutError(
                        f"job {job_id} did not complete within {max_wait_s}s"
                    )
            await asyncio.sleep(poll_interval)


# Async iterator helper (not required, but useful for tests)
async def iter_poll(
    client: ParallectAPIClient, job_id: str, *, poll_interval: float = 15.0
) -> AsyncIterator[JobStatus]:
    while True:
        status = await client.get_job(job_id)
        yield status
        if status.status in ("completed", "failed"):
            return
        await asyncio.sleep(poll_interval)
