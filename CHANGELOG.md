# Changelog

## 0.2.0 (2026-04-17)

- Manifest v1.1 alignment: orchestrator emits `spec_version: "1.1"`, `provider_breakdown`, `total_duration`, `has_attestations`
- Per-provider citation deduplication by canonical URL
- `--timeout` flag propagated into each provider's HTTP client (default raised from 120s to 900s to accommodate long-running deep-research providers)
- Provider-level evidence graph linking claims to sources, emitted during synthesis
- Phase-by-phase progress feedback during long research runs
- Iterative mode now writes a bundle on completion (previously only non-iterative mode did)
- `parallect continue` accepts a `--timeout` flag
- Requires `prx-spec>=0.2.0`
- End-to-end CLI integration test suite

## 0.1.0 (2026-04-01)

Initial open source release.

- Multi-provider research orchestration across 8 providers: Perplexity, Gemini, OpenAI, Grok, Anthropic, Ollama, LM Studio, and Local Deep Research
- CLI interface: `parallect research`, `parallect continue`, `parallect enhance`, `parallect config`
- Library API: `from parallect import research`
- Synthesis with claims extraction via Claude Sonnet 4
- .prx bundle output (portable, verifiable research format)
- Plugin system with 4 hook points
- Budget estimation and cost capping
- Ed25519 bundle signing
