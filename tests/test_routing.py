"""Routing decision: SaaS vs BYOK."""

from parallect.cli.research import (
    TIER_CONFIGS,
    VALID_TIERS,
    decide_route,
    resolve_tier,
    _short_key,
)


class TestDecideRoute:
    def test_flag_key_routes_saas(self):
        d = decide_route(api_key_flag="par_live_abc_def", local_flag=False, env_key=None)
        assert d.mode == "saas"
        assert d.api_key == "par_live_abc_def"

    def test_env_key_routes_saas(self):
        d = decide_route(api_key_flag=None, local_flag=False, env_key="par_live_env")
        assert d.mode == "saas"
        assert d.api_key == "par_live_env"

    def test_local_forces_byok_even_with_env_key(self):
        d = decide_route(api_key_flag=None, local_flag=True, env_key="par_live_env")
        assert d.mode == "byok"
        assert d.api_key is None

    def test_local_forces_byok_even_with_flag_key(self):
        d = decide_route(api_key_flag="par_live_x", local_flag=True, env_key=None)
        assert d.mode == "byok"

    def test_no_key_routes_byok(self):
        d = decide_route(api_key_flag=None, local_flag=False, env_key=None)
        assert d.mode == "byok"

    def test_flag_key_takes_precedence_over_env(self):
        d = decide_route(
            api_key_flag="par_live_flag", local_flag=False, env_key="par_live_env"
        )
        assert d.api_key == "par_live_flag"


class TestResolveTier:
    def test_default_tier(self):
        t = resolve_tier(None)
        assert t.name == "normal"

    def test_all_tiers_valid(self):
        for name in VALID_TIERS:
            t = resolve_tier(name)
            assert t.name == name

    def test_invalid_tier_raises(self):
        import typer
        import pytest

        with pytest.raises(typer.BadParameter):
            resolve_tier("ultra")

    def test_tier_budget_ordering(self):
        assert TIER_CONFIGS["nano"].budget_cap_usd < TIER_CONFIGS["max"].budget_cap_usd
        assert TIER_CONFIGS["deep"].deep is True
        assert TIER_CONFIGS["normal"].deep is False


class TestShortKey:
    def test_standard_key(self):
        assert _short_key("par_live_abc123_secretpart") == "par_live_abc123"

    def test_short_input(self):
        # Short keys collapse to the first segment — never leak the rest.
        assert _short_key("par_test") == "par"

    def test_never_exposes_full_secret(self):
        full = "par_live_abc123_verylongsecretthatshouldntleak"
        assert "verylongsecret" not in _short_key(full)
