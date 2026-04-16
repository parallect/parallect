"""Tests for response hashing utilities."""

import hashlib

from parallect.providers import ProviderResult
from parallect.providers.hash_response import attach_response_hash, hash_response


class TestHashResponse:
    def test_deterministic(self):
        body = "Hello, world!"
        assert hash_response(body) == hash_response(body)

    def test_sensitivity(self):
        assert hash_response("Hello, world!") != hash_response("Hello, world?")

    def test_correct_sha256(self):
        body = "test body"
        expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert hash_response(body) == expected

    def test_empty_string(self):
        result = hash_response("")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_multibyte_characters(self):
        body = "こんにちは世界 🌍"
        result = hash_response(body)
        expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert result == expected


class TestAttachResponseHash:
    def _make_result(self) -> ProviderResult:
        return ProviderResult(
            provider="test",
            status="completed",
            report_markdown="# Test report",
        )

    def test_adds_all_fields(self):
        result = self._make_result()
        raw_body = '{"choices": [{"message": {"content": "hello"}}]}'
        hashed = attach_response_hash(result, raw_body)
        assert hashed.response_hash is not None
        assert hashed.raw_response_size is not None
        assert hashed.received_at is not None

    def test_hash_is_correct(self):
        result = self._make_result()
        raw_body = "test body"
        hashed = attach_response_hash(result, raw_body)
        expected = hashlib.sha256(raw_body.encode("utf-8")).hexdigest()
        assert hashed.response_hash == expected

    def test_size_is_byte_length(self):
        result = self._make_result()
        raw_body = "こんにちは"  # 5 chars, 15 bytes in UTF-8
        hashed = attach_response_hash(result, raw_body)
        assert hashed.raw_response_size == len(raw_body.encode("utf-8"))

    def test_received_at_is_utc(self):
        result = self._make_result()
        hashed = attach_response_hash(result, "body")
        assert hashed.received_at is not None
        assert hashed.received_at.tzinfo is not None

    def test_preserves_existing_fields(self):
        result = ProviderResult(
            provider="perplexity",
            status="completed",
            report_markdown="# Report",
            cost_usd=1.50,
            model="sonar-deep",
        )
        hashed = attach_response_hash(result, "raw response")
        assert hashed.provider == "perplexity"
        assert hashed.cost_usd == 1.50
        assert hashed.model == "sonar-deep"
        assert hashed.response_hash is not None

    def test_none_fields_before_hashing(self):
        result = self._make_result()
        assert result.response_hash is None
        assert result.raw_response_size is None
        assert result.received_at is None
