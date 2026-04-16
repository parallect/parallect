"""CLI tests for ``parallect plugins`` subcommands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from parallect.cli import app
from parallect.plugins.data_sources import reset_registry

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class TestPluginsList:
    def test_list_default(self):
        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0, result.stdout
        # Rich table output — match by plugin name substrings.
        assert "prxhub" in result.stdout
        assert "filesystem" in result.stdout
        assert "prior-research" in result.stdout

    def test_list_json(self):
        result = runner.invoke(app, ["plugins", "list", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        names = {p["name"] for p in payload}
        assert {"prxhub", "filesystem", "prior-research"} <= names
        # filesystem requires an index; prxhub doesn't.
        idx_map = {p["name"]: p["requires_index"] for p in payload}
        assert idx_map["filesystem"] is True
        assert idx_map["prxhub"] is False
        assert idx_map["prior-research"] is False


class TestPluginsStatus:
    def test_status_unknown_plugin(self):
        result = runner.invoke(app, ["plugins", "status", "does-not-exist"])
        assert result.exit_code == 1
        assert "unknown" in result.stdout.lower()

    def test_status_all_json(self, monkeypatch):
        """Run status for all; mock health_check to avoid network."""
        from parallect.plugins.data_sources import get_registry

        # Force registry population
        get_registry()

        async def fake_health(self):
            return {"status": "ok", "mocked": True}

        # Patch all three plugin classes' health_check
        monkeypatch.setattr(
            "parallect.plugins.data_sources.prxhub.PrxhubPlugin.health_check",
            fake_health,
        )
        monkeypatch.setattr(
            "parallect.plugins.data_sources.filesystem.FilesystemPlugin.health_check",
            fake_health,
        )
        monkeypatch.setattr(
            "parallect.plugins.data_sources.prior_research.PriorResearchCache.health_check",
            fake_health,
        )

        result = runner.invoke(app, ["plugins", "status", "--json"])
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert all(info.get("status") == "ok" for info in payload.values())


class TestPluginsIndex:
    def test_index_unknown_plugin(self):
        result = runner.invoke(app, ["plugins", "index", "does-not-exist"])
        assert result.exit_code == 1
        assert "unknown" in result.stdout.lower()

    def test_index_prxhub_is_noop(self):
        result = runner.invoke(app, ["plugins", "index", "prxhub"])
        assert result.exit_code == 0
        assert "does not require an index" in result.stdout.lower()

    def test_index_filesystem_bubbles_error(self, monkeypatch):
        """filesystem without configure() should error clearly."""
        monkeypatch.setattr(
            "parallect.orchestrator.plugin_sources._extract_plugin_configs",
            lambda _: {},
        )
        result = runner.invoke(app, ["plugins", "index", "filesystem"])
        assert result.exit_code == 1
        assert "failed" in result.stdout.lower() or "not configured" in result.stdout.lower()


class TestPluginsConfig:
    def test_config_unknown(self):
        result = runner.invoke(app, ["plugins", "config", "does-not-exist"])
        assert result.exit_code == 1
