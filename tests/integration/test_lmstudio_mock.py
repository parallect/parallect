"""Integration: full-path LM Studio mock using an actual TCP socket.

Spins up a tiny async HTTP server in-process (stdlib only), points the
adapter at it, and verifies the whole resolver + adapter path end-to-end.
No aiohttp dependency, no Docker.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytestmark = pytest.mark.integration

from parallect.backends import BackendSpec
from parallect.backends.adapters import (
    call_openai_compat_chat,
    call_openai_compat_embeddings,
)


class _LMStudioMock:
    """Bare-bones stdlib asyncio HTTP server mimicking LM Studio."""

    def __init__(self):
        self.host = "127.0.0.1"
        self.port: int | None = None
        self._server: asyncio.AbstractServer | None = None
        self.received: list[dict] = []

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            method, path, _ = request_line.decode().split(" ", 2)
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().strip().partition(":")
                headers[k.lower()] = v.strip()

            content_length = int(headers.get("content-length", "0"))
            body_bytes = await reader.readexactly(content_length) if content_length else b""
            body = json.loads(body_bytes) if body_bytes else {}
            self.received.append({"method": method, "path": path, "body": body})

            if path.startswith("/v1/chat/completions"):
                msg = body["messages"][-1]["content"]
                payload = {
                    "model": body.get("model", "mock"),
                    "choices": [{"message": {"content": f"echo: {msg}"}}],
                    "usage": {
                        "prompt_tokens": len(msg),
                        "completion_tokens": 5,
                        "total_tokens": len(msg) + 5,
                    },
                }
            elif path.startswith("/v1/embeddings"):
                texts = body["input"]
                if isinstance(texts, str):
                    texts = [texts]
                payload = {
                    "model": body.get("model", "mock-embed"),
                    "data": [
                        {"index": i, "embedding": [float(len(t))] * 4}
                        for i, t in enumerate(texts)
                    ],
                }
            elif path.startswith("/v1/models"):
                payload = {"data": [{"id": "mock"}]}
            else:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            body_out = json.dumps(payload).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(body_out)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + body_out
            )
            await writer.drain()
        finally:
            writer.close()


def _spec(base: str) -> BackendSpec:
    return BackendSpec(
        kind="lmstudio",
        base_url=base,
        api_key="not-needed",
        model="mock",
        api_key_env="LMSTUDIO_API_KEY",
    )


@pytest.mark.asyncio
async def test_chat_against_mock_lmstudio():
    server = _LMStudioMock()
    await server.start()
    try:
        out = await call_openai_compat_chat(_spec(server.base_url), "hello", "sys")
        assert "echo: hello" in out["content"]
        assert out["tokens"]["output"] == 5
        # Verify headers + model were forwarded
        received = server.received[-1]
        assert received["path"].startswith("/v1/chat/completions")
        assert received["body"]["model"] == "mock"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_embeddings_against_mock_lmstudio():
    server = _LMStudioMock()
    await server.start()
    try:
        vectors = await call_openai_compat_embeddings(_spec(server.base_url), ["a", "bb"])
        assert vectors == [[1.0] * 4, [2.0] * 4]
    finally:
        await server.stop()
