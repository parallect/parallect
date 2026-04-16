"""Protocol + registry tests for the data source plugin system."""

from __future__ import annotations

import pytest

from parallect.plugins.data_sources import (
    DataSource,
    Document,
    IndexStats,
    PluginError,
    SourceSpec,
    get,
    get_registry,
    parse_sources,
    register,
    reset_registry,
)
from parallect.plugins.data_sources.filesystem import FilesystemPlugin
from parallect.plugins.data_sources.prior_research import PriorResearchCache
from parallect.plugins.data_sources.prxhub import PrxhubPlugin


class TestProtocolConformance:
    """Every in-core plugin satisfies the DataSource protocol."""

    def test_prxhub_is_datasource(self):
        assert isinstance(PrxhubPlugin(), DataSource)

    def test_filesystem_is_datasource(self):
        assert isinstance(FilesystemPlugin(), DataSource)

    def test_prior_research_is_datasource(self):
        assert isinstance(PriorResearchCache(), DataSource)

    @pytest.mark.parametrize("plugin_cls", [PrxhubPlugin, FilesystemPlugin, PriorResearchCache])
    def test_plugin_has_name(self, plugin_cls):
        p = plugin_cls()
        assert p.name
        assert p.display_name
        assert isinstance(p.requires_index, bool)


class TestRegistry:
    def setup_method(self):
        reset_registry()

    def teardown_method(self):
        reset_registry()

    def test_incore_plugins_autoloaded(self):
        reg = get_registry()
        assert "prxhub" in reg
        assert "filesystem" in reg
        assert "prior-research" in reg

    def test_get_unknown_plugin_raises(self):
        with pytest.raises(PluginError, match="unknown data source"):
            get("does-not-exist")

    def test_get_known_plugin(self):
        plugin = get("prxhub")
        assert isinstance(plugin, PrxhubPlugin)

    def test_register_overrides_incore(self):
        """A later register() call overrides an in-core plugin."""

        class FakePrxhub(PrxhubPlugin):
            name = "prxhub"
            display_name = "fake-prxhub"

        # Seed the registry first (normal flow)
        get_registry()
        fake = FakePrxhub()
        register(fake)
        assert get("prxhub") is fake

    def test_register_requires_name(self):
        class Nameless:
            name = ""
            display_name = "x"
            requires_index = False

        with pytest.raises(PluginError, match="has no .name"):
            register(Nameless())  # type: ignore[arg-type]

    def test_reset_registry_clears_incore_flag(self):
        get_registry()
        reset_registry()
        # After reset, next get_registry() should re-load in-core plugins.
        reg = get_registry()
        assert "prxhub" in reg


class TestParseSources:
    def test_none_returns_empty(self):
        assert parse_sources(None) == []

    def test_empty_string_returns_empty(self):
        assert parse_sources("") == []

    def test_single_source(self):
        specs = parse_sources("prxhub")
        assert specs == [SourceSpec(name="prxhub", instance=None)]

    def test_multi_source(self):
        specs = parse_sources("prxhub,filesystem,prior-research")
        assert [s.name for s in specs] == ["prxhub", "filesystem", "prior-research"]
        assert all(s.instance is None for s in specs)

    def test_instance_syntax(self):
        specs = parse_sources("filesystem:notes")
        assert specs == [SourceSpec(name="filesystem", instance="notes")]

    def test_mixed(self):
        specs = parse_sources("prxhub,filesystem:notes,filesystem:code")
        assert specs[0] == SourceSpec("prxhub", None)
        assert specs[1] == SourceSpec("filesystem", "notes")
        assert specs[2] == SourceSpec("filesystem", "code")

    def test_whitespace_trimmed(self):
        specs = parse_sources(" prxhub , filesystem:notes ")
        assert specs[0].name == "prxhub"
        assert specs[1].name == "filesystem"
        assert specs[1].instance == "notes"

    def test_empty_tokens_skipped(self):
        specs = parse_sources("prxhub,,filesystem")
        assert len(specs) == 2

    def test_empty_name_with_instance_raises(self):
        with pytest.raises(PluginError, match="invalid --sources token"):
            parse_sources(":notes")

    def test_spec_display(self):
        assert SourceSpec("prxhub", None).display == "prxhub"
        assert SourceSpec("filesystem", "notes").display == "filesystem:notes"


class TestDocumentIndexStats:
    def test_document_defaults(self):
        d = Document(id="x", content="hello")
        assert d.title is None
        assert d.source_url is None
        assert d.metadata == {}
        assert d.score is None

    def test_index_stats_required_fields(self):
        stats = IndexStats(
            source_name="fs",
            documents_indexed=5,
            documents_skipped=2,
            elapsed_seconds=0.1,
        )
        assert stats.index_path is None
