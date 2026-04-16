"""PrxhubPlugin tests — mock httpx, assert request shape + response parsing."""

from __future__ import annotations

import httpx

from parallect.plugins.data_sources.prxhub import PrxhubPlugin


def _mock_transport_for(handler):
    return httpx.MockTransport(handler)


def _patch_httpx(monkeypatch, handler):
    orig = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return orig(*args, **kwargs)

    monkeypatch.setattr("parallect.plugins.data_sources.prxhub.httpx.AsyncClient", patched)


class TestPrxhubConfigure:
    async def test_defaults(self):
        p = PrxhubPlugin()
        await p.configure({})
        assert p._api_url == "https://prxhub.com"
        assert p._api_key is None

    async def test_custom_url_and_key(self):
        p = PrxhubPlugin()
        await p.configure({"api_url": "http://hub.local/", "api_key": "secret"})
        # trailing slash is stripped
        assert p._api_url == "http://hub.local"
        assert p._api_key == "secret"


class TestPrxhubSearch:
    async def test_search_parses_bundles_and_claims(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["params"] = dict(request.url.params)
            return httpx.Response(
                200,
                json={
                    "bundles": {
                        "results": [
                            {
                                "id": "b1",
                                "title": "Consensus algorithms",
                                "query": "what are consensus algorithms",
                                "trust_score": 0.87,
                                "divergence": 0.12,
                                "attestation_count": 3,
                                "providers": ["perplexity", "gemini"],
                                "score": 0.95,
                            }
                        ],
                    },
                    "claims": {
                        "results": [
                            {
                                "id": "c9",
                                "bundle_id": "b1",
                                "content": "Paxos requires 2f+1 nodes to tolerate f failures.",
                                "confidence": 0.9,
                                "trust_score": 0.88,
                                "providers": ["anthropic"],
                            }
                        ]
                    },
                },
            )

        _patch_httpx(monkeypatch, handler)

        p = PrxhubPlugin()
        await p.configure({"api_url": "http://hub.test"})
        docs = await p.search("consensus", limit=5)

        # Request shape
        assert captured["path"] == "/api/search"
        assert captured["params"]["q"] == "consensus"
        assert captured["params"]["limit"] == "5"

        # Got both bundle + claim
        assert len(docs) == 2
        bundle = [d for d in docs if d.metadata["kind"] == "bundle"][0]
        claim = [d for d in docs if d.metadata["kind"] == "claim"][0]

        assert bundle.id == "bundle:b1"
        assert "Consensus algorithms" in (bundle.title or "")
        assert bundle.source_url == "http://hub.test/bundles/b1"
        assert bundle.metadata["trust_score"] == 0.87
        assert bundle.metadata["divergence"] == 0.12
        assert bundle.metadata["attestation_count"] == 3
        assert bundle.metadata["providers"] == ["perplexity", "gemini"]

        assert claim.id == "claim:c9"
        assert claim.metadata["confidence"] == 0.9
        assert claim.metadata["bundle_id"] == "b1"
        assert "b1#claim-c9" in (claim.source_url or "")

    async def test_search_sends_auth_header_when_configured(self, monkeypatch):
        captured_headers: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(request.headers)
            return httpx.Response(200, json={"bundles": {"results": []}, "claims": {"results": []}})

        _patch_httpx(monkeypatch, handler)
        p = PrxhubPlugin()
        await p.configure({"api_url": "http://hub.test", "api_key": "sekret"})
        await p.search("q")
        assert captured_headers.get("authorization") == "Bearer sekret"

    async def test_search_empty_response(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={"bundles": {"results": []}, "claims": {"results": []}})

        _patch_httpx(monkeypatch, handler)
        p = PrxhubPlugin()
        await p.configure({})
        docs = await p.search("nothing")
        assert docs == []


class TestPrxhubHealth:
    async def test_health_ok(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/health"
            return httpx.Response(200, json={"status": "ok", "version": "1.2"})

        _patch_httpx(monkeypatch, handler)
        p = PrxhubPlugin()
        await p.configure({"api_url": "http://hub.test"})
        info = await p.health_check()
        assert info["status"] == "ok"
        assert info["http_status"] == 200
        assert info["body"]["version"] == "1.2"

    async def test_health_network_error(self, monkeypatch):
        def handler(request):
            raise httpx.ConnectError("refused")

        _patch_httpx(monkeypatch, handler)
        p = PrxhubPlugin()
        await p.configure({"api_url": "http://hub.test"})
        info = await p.health_check()
        assert info["status"] == "error"
        assert "refused" in info["error"]

    async def test_health_4xx(self, monkeypatch):
        def handler(request):
            return httpx.Response(503, json={"error": "down"})

        _patch_httpx(monkeypatch, handler)
        p = PrxhubPlugin()
        await p.configure({"api_url": "http://hub.test"})
        info = await p.health_check()
        assert info["status"] == "error"
        assert info["http_status"] == 503


class TestPrxhubIndex:
    async def test_index_is_noop(self):
        p = PrxhubPlugin()
        stats = await p.index()
        assert stats.documents_indexed == 0
        assert stats.index_path is None

    async def test_fresh_always_true(self):
        p = PrxhubPlugin()
        assert await p.is_fresh() is True
