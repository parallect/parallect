"""Parallect API client for the enhance command."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx

PARALLECT_API_URL = "https://api.parallect.ai"


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
) -> EnhanceJob:
    """Upload a .prx bundle to Parallect API for premium enhancement."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. Initiate enhancement
        response = await client.post(
            f"{api_url}/api/enhance/initiate",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"tier": tier},
        )
        response.raise_for_status()
        job_data = response.json()

        # 2. Upload bundle to presigned URL
        with open(bundle_path, "rb") as f:
            await client.put(job_data["upload_url"], content=f.read())

        # 3. Start enhancement
        await client.post(
            f"{api_url}/api/enhance/{job_data['job_id']}/start",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"storage_key": job_data["storage_key"]},
        )

        # 4. Poll for completion
        while True:
            status_response = await client.get(
                f"{api_url}/api/enhance/{job_data['job_id']}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            status_response.raise_for_status()
            status = status_response.json()

            if status.get("complete"):
                # 5. Download enhanced bundle
                enhanced_path = bundle_path.with_suffix(".enhanced.prx")
                download = await client.get(status["download_url"])
                download.raise_for_status()
                enhanced_path.write_bytes(download.content)

                return EnhanceJob(
                    job_id=job_data["job_id"],
                    status="completed",
                    enhanced_path=enhanced_path,
                )

            await asyncio.sleep(status.get("retry_after", 5))
