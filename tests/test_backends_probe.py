"""Tests for `parallect.backends.probe.probe_local_backends`."""

from __future__ import annotations

from unittest.mock import Mock, patch

import httpx

from parallect.backends.probe import (
    LMSTUDIO_MODELS_URL,
    OLLAMA_TAGS_URL,
    LocalProbeResult,
    probe_local_backends,
)


class _FakeClient:
    """Stand-in for httpx.Client that returns canned responses per URL."""

    def __init__(self, mapping: dict[str, int | Exception]):
        self.mapping = mapping

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url: str):
        outcome = self.mapping.get(url)
        if isinstance(outcome, Exception):
            raise outcome
        resp = Mock()
        resp.status_code = outcome
        return resp


class TestProbeLocalBackends:
    def test_both_reachable(self):
        fake = _FakeClient({LMSTUDIO_MODELS_URL: 200, OLLAMA_TAGS_URL: 200})
        with patch("httpx.Client", return_value=fake):
            result = probe_local_backends()
        assert result.lmstudio_reachable is True
        assert result.ollama_reachable is True
        assert result.any_reachable is True
        assert result.preferred_backend == "lmstudio"

    def test_lmstudio_only(self):
        fake = _FakeClient({
            LMSTUDIO_MODELS_URL: 200,
            OLLAMA_TAGS_URL: httpx.ConnectError("refused"),
        })
        with patch("httpx.Client", return_value=fake):
            result = probe_local_backends()
        assert result.lmstudio_reachable is True
        assert result.ollama_reachable is False
        assert result.preferred_backend == "lmstudio"

    def test_ollama_only(self):
        fake = _FakeClient({
            LMSTUDIO_MODELS_URL: httpx.ConnectError("refused"),
            OLLAMA_TAGS_URL: 200,
        })
        with patch("httpx.Client", return_value=fake):
            result = probe_local_backends()
        assert result.lmstudio_reachable is False
        assert result.ollama_reachable is True
        assert result.preferred_backend == "ollama"

    def test_neither_reachable(self):
        fake = _FakeClient({
            LMSTUDIO_MODELS_URL: httpx.ConnectError("refused"),
            OLLAMA_TAGS_URL: httpx.ConnectError("refused"),
        })
        with patch("httpx.Client", return_value=fake):
            result = probe_local_backends()
        assert result.any_reachable is False
        assert result.preferred_backend is None

    def test_non_200_is_not_reachable(self):
        fake = _FakeClient({
            LMSTUDIO_MODELS_URL: 500,
            OLLAMA_TAGS_URL: 404,
        })
        with patch("httpx.Client", return_value=fake):
            result = probe_local_backends()
        assert result.lmstudio_reachable is False
        assert result.ollama_reachable is False
        assert result.any_reachable is False


class TestLocalProbeResult:
    def test_no_preferred_when_neither(self):
        r = LocalProbeResult(lmstudio_reachable=False, ollama_reachable=False)
        assert r.preferred_backend is None
        assert r.any_reachable is False

    def test_lmstudio_preferred_over_ollama(self):
        r = LocalProbeResult(lmstudio_reachable=True, ollama_reachable=True)
        assert r.preferred_backend == "lmstudio"
