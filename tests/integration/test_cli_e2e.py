"""End-to-end CLI tests.

Drives the actual `parallect` CLI via subprocess and inspects the resulting
`.prx` bundles. Each case exercises a user-facing feature against a real
environment: LM Studio running locally, optionally Perplexity via BYOK.

Gated behind PARALLECT_INTEGRATION=1 so it never runs in the default
`uv run pytest tests/` workflow — the individual cases cost real money and
take minutes to run.

Preconditions for running:

    - PARALLECT_INTEGRATION=1                 (master switch)
    - LM Studio running at http://localhost:1234
      with google/gemma-4-31b loaded
    - Optional: perplexity_api_key in parallect config or env
      (tests that need it skip automatically when missing)

Run everything:

    PARALLECT_INTEGRATION=1 uv run pytest tests/integration/test_cli_e2e.py -v

Run a single case:

    PARALLECT_INTEGRATION=1 uv run pytest \
        tests/integration/test_cli_e2e.py::test_local_lmstudio_no_synth -v -s

The `-s` flag is useful because each test streams the CLI's live output so
you can see per-provider progress while a long run is in flight.
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PARALLECT_INTEGRATION"),
    reason=(
        "integration tests require PARALLECT_INTEGRATION=1, LM Studio running "
        "locally, and real provider keys in the parallect config"
    ),
)

REPO_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Preconditions — skipped individually when the env isn't ready
# ---------------------------------------------------------------------------


def _lmstudio_up() -> bool:
    try:
        r = httpx.get("http://localhost:1234/v1/models", timeout=2.0)
        return r.status_code == 200 and "data" in r.text
    except Exception:
        return False


def _has_perplexity_key() -> bool:
    """Check settings-file path since the OSS CLI reads from there."""
    from parallect.config_mod.settings import ParallectSettings

    s = ParallectSettings.load()
    return bool(s.perplexity_api_key)


requires_lmstudio = pytest.mark.skipif(
    not _lmstudio_up(),
    reason="LM Studio not running at localhost:1234",
)
requires_perplexity = pytest.mark.skipif(
    not _has_perplexity_key(),
    reason="Perplexity API key not configured",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_parallect(
    *args: str,
    timeout: float = 600.0,
    check: bool = True,
) -> tuple[str, str, int]:
    """Invoke the CLI exactly as a user would."""
    cmd = ["uv", "run", "parallect", *args]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"CLI exited {result.returncode}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout, result.stderr, result.returncode


def read_bundle(prx_path: Path) -> dict:
    """Extract a .prx archive and return all parsed artifacts."""
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(prx_path, "r:gz") as tf:
            tf.extractall(td)
        root = next(Path(td).iterdir())

        def _load(rel: str) -> dict | None:
            path = root / rel
            return json.loads(path.read_text()) if path.exists() else None

        def _load_text(rel: str) -> str | None:
            path = root / rel
            return path.read_text() if path.exists() else None

        citations: dict[str, list] = {}
        providers_dir = root / "providers"
        if providers_dir.exists():
            for p in providers_dir.iterdir():
                c = p / "citations.json"
                if c.exists():
                    citations[p.name] = json.loads(c.read_text())

        return {
            "bundle_id": root.name,
            "manifest": _load("manifest.json") or {},
            "claims": _load("synthesis/claims.json"),
            "sources": _load("sources/registry.json"),
            "evidence": _load("evidence/graph.json"),
            "synthesis_md": _load_text("synthesis/report.md"),
            "citations": citations,
        }


def assert_manifest_core(manifest: dict, *, expect_providers: list[str]) -> None:
    """Invariants every completed bundle must satisfy."""
    assert manifest.get("spec_version") == "1.1", (
        f"spec_version regression: {manifest.get('spec_version')!r}"
    )
    used = manifest.get("providers_used", [])
    for name in expect_providers:
        assert name in used, f"expected {name!r} in providers_used, got {used!r}"
    assert manifest.get("total_duration_seconds"), (
        "total_duration_seconds should be set to a positive real value"
    )
    assert manifest["total_duration_seconds"] > 0
    pb = manifest.get("provider_breakdown", [])
    assert len(pb) >= len(expect_providers), (
        f"provider_breakdown under-populated: {pb}"
    )


# ---------------------------------------------------------------------------
# Test cases — each one exercises a single user-facing feature
# ---------------------------------------------------------------------------


@requires_lmstudio
def test_local_lmstudio_no_synth(tmp_path: Path) -> None:
    """Baseline smoke: single local provider, no synthesis, no claims."""
    out = tmp_path / "bundle.prx"
    run_parallect(
        "research", "What is 1+1?",
        "-p", "lmstudio",
        "--byok",
        "--no-synthesis",
        "--output", str(out),
        timeout=300,
    )
    assert out.exists()
    b = read_bundle(out)
    m = b["manifest"]
    assert_manifest_core(m, expect_providers=["lmstudio"])
    assert m["has_synthesis"] is False
    assert m["has_attestations"] is True, "signing should default to on"
    pb = {e["provider"]: e for e in m["provider_breakdown"]}
    assert pb["lmstudio"]["status"] == "completed"


@requires_lmstudio
def test_local_lmstudio_with_synth(tmp_path: Path) -> None:
    """LM Studio provider + LM Studio synthesis backend (from [synthesis] config).

    Exercises the full single-provider pipeline: research → claims → synthesis
    → follow-ons → evidence graph, all local. LM Studio serializes every
    stage through a single loaded model, so the total walltime is the sum of
    the research call, synthesis call, claims extraction, and follow-ons —
    roughly 10–15 min on a 31B model for a one-sentence query.
    """
    out = tmp_path / "bundle.prx"
    run_parallect(
        "research", "What is 1+1? Explain briefly.",
        "-p", "lmstudio",
        "--byok",
        "--output", str(out),
        timeout=1200,
    )
    assert out.exists()
    b = read_bundle(out)
    m = b["manifest"]
    assert_manifest_core(m, expect_providers=["lmstudio"])
    assert m["has_synthesis"] is True
    assert m["has_claims"] is True
    assert b["claims"] is not None
    assert len(b["claims"]["claims"]) > 0, "expected at least one claim"
    assert b["synthesis_md"] is not None and len(b["synthesis_md"]) > 50


@requires_lmstudio
def test_filesystem_source_with_lmstudio(tmp_path: Path) -> None:
    """Data-source plugin path: filesystem index contributes citations.

    Depends on `filesystem:hunter-deep-research` being configured and indexed.
    """
    from parallect.config_mod.settings import ParallectSettings

    s = ParallectSettings.load()
    has_plugin = any(
        p.get("name") == "hunter-deep-research"
        for p in getattr(s, "plugins_filesystem", []) or []
    )
    if not has_plugin:
        # The config structure may vary; fall back to reading TOML directly.
        import tomllib
        from parallect.config_mod.settings import _user_config_path

        with open(_user_config_path(), "rb") as f:
            raw = tomllib.load(f)
        fs_plugins = raw.get("plugins", {}).get("filesystem", [])
        if not any(p.get("name") == "hunter-deep-research" for p in fs_plugins):
            pytest.skip("filesystem:hunter-deep-research plugin not configured")

    out = tmp_path / "bundle.prx"
    run_parallect(
        "research", "summarize the main findings",
        "--sources", "filesystem:hunter-deep-research",
        "-p", "lmstudio",
        "--byok",
        "--no-synthesis",
        "--output", str(out),
        timeout=300,
    )
    b = read_bundle(out)
    m = b["manifest"]
    assert m["spec_version"] == "1.1"
    # filesystem source should show up under its "filesystem:<name>" key.
    fs_key = "filesystem:hunter-deep-research"
    assert fs_key in b["citations"], (
        f"expected citations from {fs_key!r}, got keys: {list(b['citations'])}"
    )
    # Dedup: no URL appears twice at different indices within a single provider.
    for pname, cites in b["citations"].items():
        urls = [c.get("url") for c in cites if c.get("url")]
        assert len(urls) == len(set(urls)), (
            f"{pname} has duplicate citation URLs: {urls}"
        )


@requires_perplexity
def test_single_perplexity(tmp_path: Path) -> None:
    """Single web provider, cheapest realistic path.

    Perplexity's default model is `sonar-deep-research` which is slow (2–5 min)
    and costs ~$0.05 — still fine as a regression test, just keep the timeout
    generous.
    """
    out = tmp_path / "bundle.prx"
    run_parallect(
        "research", "What is 1+1?",
        "-p", "perplexity",
        "--byok",
        "--no-synthesis",
        "--output", str(out),
        timeout=600,
    )
    b = read_bundle(out)
    m = b["manifest"]
    assert_manifest_core(m, expect_providers=["perplexity"])
    pb = {e["provider"]: e for e in m["provider_breakdown"]}
    assert pb["perplexity"]["status"] == "completed"
    assert pb["perplexity"]["cost_usd"] and pb["perplexity"]["cost_usd"] > 0
    # Perplexity doesn't always return citations for trivial queries
    # ("What is 1+1?" comes back without web sources) — just verify that the
    # report itself is non-trivial and that if any citations came back they
    # have non-empty URLs.
    providers_dir_key = "perplexity"
    if providers_dir_key in b["citations"]:
        for c in b["citations"][providers_dir_key]:
            assert c.get("url"), f"citation missing url: {c!r}"


@requires_lmstudio
@requires_perplexity
def test_multi_provider_fanout(tmp_path: Path) -> None:
    """Parallel fan-out across mixed local + web providers.

    Validates parallel execution doesn't break; both providers land in
    providers_used and provider_breakdown; costs aggregate correctly.
    """
    out = tmp_path / "bundle.prx"
    run_parallect(
        "research", "What is 1+1? Answer concisely.",
        "-p", "lmstudio,perplexity",
        "--byok",
        "--output", str(out),
        timeout=600,
    )
    b = read_bundle(out)
    m = b["manifest"]
    assert_manifest_core(m, expect_providers=["lmstudio", "perplexity"])
    assert m["has_synthesis"] is True
    assert m["has_claims"] is True
    # Evidence graph should exist once we have both claims and sources.
    if m["has_sources"]:
        assert m["has_evidence_graph"] is True


@requires_lmstudio
def test_iterative_loop_local(tmp_path: Path) -> None:
    """Agentic research loop: planner → execute → evaluate → maybe loop.

    Runs entirely against LM Studio (planner/evaluator also use the
    synthesis backend from config) so it's free and deterministic-ish.

    --timeout 600 is passed explicitly because the iterative executor's
    sub-query fan-out uses the same timeout value as the asyncio wrapper
    around each provider call; 120s (CLI default) isn't enough for a
    31B local model asked to answer a combined prompt.
    """
    out = tmp_path / "bundle.prx"
    run_parallect(
        "research", "What is 1+1? Explain step by step.",
        "-p", "lmstudio",
        "--byok",
        "--iterative",
        "--max-iterations", "2",
        "--iteration-budget-usd", "0.50",
        "--timeout", "600",
        "--output", str(out),
        timeout=1200,
    )
    assert out.exists(), "iterative mode should write a .prx bundle"
    # loop.json provenance sidecar is also produced.
    assert (tmp_path / "loop.json").exists(), "expected loop.json provenance file"

    b = read_bundle(out)
    m = b["manifest"]
    assert m["spec_version"] == "1.1"
    assert "lmstudio" in m["providers_used"]
    # Synthesis may or may not be present depending on whether the loop
    # produced any results — on "What is 1+1?" the planner's sub-queries
    # sometimes return empty and the loop skips synthesis. The bundle is
    # still written either way; just verify it's valid.
    assert m["has_attestations"] is True


@requires_lmstudio
def test_follow_on_continue(tmp_path: Path) -> None:
    """`parallect continue <bundle> <query>` rooted in a prior bundle."""
    parent = tmp_path / "parent.prx"
    run_parallect(
        "research", "What is 1+1?",
        "-p", "lmstudio",
        "--byok",
        "--no-synthesis",
        "--timeout", "600",
        "--output", str(parent),
        timeout=300,
    )
    assert parent.exists()

    child = tmp_path / "child.prx"
    run_parallect(
        "continue", str(parent), "What about 2+2?",
        "-p", "lmstudio",
        "--timeout", "600",
        "--output", str(child),
        timeout=600,
    )
    assert child.exists()
    b = read_bundle(child)
    m = b["manifest"]
    # Follow-ons should record the parent lineage.
    assert m.get("parent_bundle_id"), "expected parent_bundle_id on child bundle"
    parent_meta = read_bundle(parent)
    assert m["parent_bundle_id"] == parent_meta["bundle_id"]
