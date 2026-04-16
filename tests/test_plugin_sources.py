"""Tests for the orchestrator's plugin fan-out helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from parallect.plugins.data_sources import (
    Document,
    PluginError,
    register,
    reset_registry,
)
from parallect.orchestrator.plugin_sources import (
    _extract_plugin_configs,
    _select_config,
    run_plugin_sources,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class FakeDS:
    """Minimal in-memory DataSource for fan-out tests."""

    name = "fake"
    display_name = "fake"
    requires_index = False

    def __init__(self, name="fake", docs=None, raise_on_search=None):
        self.name = name
        self.display_name = name
        self.requires_index = False
        self.configured_with: dict | None = None
        self._docs = docs or []
        self._raise = raise_on_search

    async def configure(self, config: dict) -> None:
        self.configured_with = dict(config)

    async def search(self, query: str, *, limit: int = 10):
        if self._raise:
            raise self._raise
        return list(self._docs)

    async def index(self, *, force=False):
        from parallect.plugins.data_sources import IndexStats
        return IndexStats(source_name=self.name, documents_indexed=0, documents_skipped=0, elapsed_seconds=0.0)

    async def is_fresh(self):
        return True

    async def fetch(self, doc_id):
        return None

    async def health_check(self):
        return {"status": "ok"}


class TestUnknownSourceRaises:
    async def test_unknown_source_in_list(self):
        with pytest.raises(PluginError, match="unknown data source"):
            await run_plugin_sources("q", "does-not-exist")


class TestFanOut:
    async def test_single_source_returns_result(self):
        docs = [Document(id="1", content="hello")]
        fake = FakeDS(name="fake", docs=docs)
        register(fake)

        results = await run_plugin_sources("q", "fake")
        assert len(results) == 1
        r = results[0]
        assert r.error is None
        assert r.result is not None
        assert r.result.status == "completed"
        assert r.result.provider == "fake"
        assert "hello" in r.result.report_markdown
        assert len(r.documents) == 1

    async def test_plugin_exception_becomes_error_result(self):
        fake = FakeDS(name="broken", raise_on_search=RuntimeError("boom"))
        register(fake)
        results = await run_plugin_sources("q", "broken")
        assert len(results) == 1
        assert results[0].result is None
        assert "boom" in results[0].error

    async def test_empty_sources_returns_empty(self):
        results = await run_plugin_sources("q", None)
        assert results == []

    async def test_sources_string_empty(self):
        results = await run_plugin_sources("q", "")
        assert results == []

    async def test_configure_receives_config(self, tmp_path: Path, monkeypatch):
        """`_run_one` must call plugin.configure with the selected config."""
        fake = FakeDS(name="prxhub")
        register(fake)

        # Fake a config file.
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[plugins.prxhub]\napi_url = "http://configured"\n'
        )

        def fake_user_dir(_name):
            return str(config_dir)

        monkeypatch.setattr(
            "parallect.orchestrator.plugin_sources.platformdirs.user_config_dir", fake_user_dir
        )

        await run_plugin_sources("q", "prxhub")
        assert fake.configured_with == {"api_url": "http://configured"}


class TestFilesystemRequiresPath:
    async def test_filesystem_no_path_raises(self, monkeypatch):
        """Explicit failure when user asks for filesystem without config."""
        fake = FakeDS(name="filesystem")
        register(fake)

        # Mock away the TOML config so the plugin truly has no path.
        monkeypatch.setattr(
            "parallect.orchestrator.plugin_sources._extract_plugin_configs",
            lambda _: {},
        )
        with pytest.raises(PluginError, match="has no `path` configured"):
            await run_plugin_sources("q", "filesystem")


class TestSelectConfig:
    def test_empty_configs_with_instance_returns_name(self):
        out = _select_config("filesystem", "notes", {})
        assert out == {"name": "notes"}

    def test_filesystem_instance_matches_by_name(self):
        configs = {
            "filesystem": [
                {"name": "notes", "path": "/a"},
                {"name": "code", "path": "/b"},
            ]
        }
        assert _select_config("filesystem", "notes", configs) == {"name": "notes", "path": "/a"}
        assert _select_config("filesystem", "code", configs) == {"name": "code", "path": "/b"}

    def test_filesystem_unknown_instance_raises(self):
        configs = {"filesystem": [{"name": "notes", "path": "/a"}]}
        with pytest.raises(PluginError, match="not found in config"):
            _select_config("filesystem", "nope", configs)

    def test_filesystem_no_instance_picks_first(self):
        configs = {"filesystem": [{"name": "notes", "path": "/a"}, {"name": "code"}]}
        assert _select_config("filesystem", None, configs) == {"name": "notes", "path": "/a"}

    def test_single_instance_plugin(self):
        configs = {"prxhub": [{"api_url": "http://h"}]}
        assert _select_config("prxhub", None, configs) == {"api_url": "http://h"}


class TestExtractConfigs:
    def test_reads_table_and_array(self, tmp_path: Path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text(
            """
[plugins.prxhub]
api_url = "http://hub"

[[plugins.filesystem]]
name = "notes"
path = "/tmp/n"

[[plugins.filesystem]]
name = "code"
path = "/tmp/c"
"""
        )
        monkeypatch.setattr(
            "parallect.orchestrator.plugin_sources.platformdirs.user_config_dir",
            lambda _name: str(cfg_dir),
        )
        out = _extract_plugin_configs(None)
        assert "prxhub" in out
        assert out["prxhub"][0]["api_url"] == "http://hub"
        assert len(out["filesystem"]) == 2
        assert {c["name"] for c in out["filesystem"]} == {"notes", "code"}
