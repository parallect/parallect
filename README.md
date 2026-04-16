# parallect

[![PyPI version](https://img.shields.io/pypi/v/parallect.svg)](https://pypi.org/project/parallect/)
[![Python](https://img.shields.io/pypi/pyversions/parallect.svg)](https://pypi.org/project/parallect/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/parallect/parallect/actions/workflows/ci.yml/badge.svg)](https://github.com/parallect/parallect/actions/workflows/ci.yml)

**Multi-provider AI deep research from the command line. Bring your own keys.**

`parallect` fans a research query out to multiple frontier AI providers in parallel — Perplexity, Gemini, OpenAI, Grok, Anthropic, plus local models via Ollama and LM Studio — then synthesizes their outputs into a single report with cross-referenced citations, extracted claims, and consensus/conflict detection.

The result is packaged as a signed [Portable Research eXchange (`.prx`)](https://github.com/parallect/prx-spec) archive — an open format for AI research that keeps cryptographic attribution back to every provider and source, so the research stays sharable and verifiable whether you keep it private or publish it to [prxhub](https://prxhub.com).

```bash
$ parallect research "What are the leading theories on dark matter?"

  Fanning out to: perplexity, gemini, openai, anthropic
  [████████████████] 4/4 providers  4m 32s  $0.18
  Synthesizing with anthropic ...
  Extracting 47 claims from 4 reports ...

  ✓ Saved to research.prx  (127 KB, signed)

  $ parallect research "..."  - dark matter theories
  → 4 providers  · 47 claims  · 23 sources  · 92% consensus
```

## Why?

- **No single provider knows everything.** Different models see different sources, reason differently, and disagree in useful ways. Ask several at once, then see where they agree.
- **Portable, signed output.** Every run saves to an open [`.prx`](https://github.com/parallect/prx-spec) archive — plain text under the hood, verifiable via Ed25519, shareable via [prxhub](https://prxhub.com).
- **BYOK.** Your API keys, your bill, your data. No intermediate service sees your queries.
- **Local-first option.** Run entirely offline against Ollama or LM Studio if you don't want to call frontier APIs.
- **Scriptable.** Stable CLI, structured output, clean exit codes. Pipe it, chain it, call it from your editor.

## Install

From PyPI:

```bash
pip install parallect
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install parallect
```

Requires Python 3.10+.

## Quick start

```bash
# One-time setup — interactive TUI configures providers, keys, and defaults
parallect config

# Run research
parallect research "What are the leading theories on dark matter?"

# Pick specific providers
parallect research "quantum computing progress 2025" -p perplexity,openai,anthropic

# Deep research mode — premium models, higher cost, deeper reasoning
parallect research "CRISPR gene therapy developments" --deep

# Offline — Ollama or LM Studio only
parallect research "explain transformers" --local

# Follow-on research rooted in a prior bundle
parallect continue output.prx "What about practical applications?"
```

## Commands

| Command | Purpose |
|---|---|
| `parallect config` | Interactive TUI for providers, API keys, backends, plugins |
| `parallect research <query>` | Run a new research query |
| `parallect continue <bundle> <query>` | Follow-on research rooted in an existing bundle |
| `parallect enhance <bundle>` | Send a bundle to the hosted Parallect API for claim extraction + evidence graph |
| `parallect jobs <subcommand>` | Manage async SaaS-mode jobs (`status`, `download`) |
| `parallect plugins <subcommand>` | Manage data-source and pipeline plugins (`list`, `status`, `index`, `config`) |

Run `parallect <command> --help` for full flags.

## Providers

| Provider | Default model | Deep model | Env var |
|---|---|---|---|
| Perplexity | `sonar-deep-research` | — | `PARALLECT_PERPLEXITY_API_KEY` |
| OpenAI | `gpt-4o-mini` | `o3-deep-research` | `PARALLECT_OPENAI_API_KEY` |
| Gemini | `gemini-2.5-flash` | `gemini-2.5-pro` | `PARALLECT_GOOGLE_API_KEY` |
| Anthropic | `claude-sonnet-4` | `claude-opus-4` | `PARALLECT_ANTHROPIC_API_KEY` |
| Grok | `grok-3` | `grok-4` | `PARALLECT_XAI_API_KEY` |
| Ollama | `llama3.2` | — | (local, no key) |
| LM Studio | `default` | — | (local, no key) |
| LDR | `llama3.2` | — | (local; `pip install "parallect[ldr]"`) |

Install optional provider SDKs with the `all` extra:

```bash
pip install "parallect[all]"
```

## Configuration

`parallect config` writes to `~/.config/parallect/config.toml` (or the platform equivalent). Precedence, highest to lowest:

1. Explicit CLI flags
2. Environment variables (`PARALLECT_*` prefix)
3. Project-local `parallect.toml`
4. User config
5. Defaults

## Data source plugins

Plugins let parallect search and cite your local content alongside web sources. Built-in plugins:

| Plugin | Purpose |
|---|---|
| `filesystem` | Index a directory of markdown / text / PDF |
| `obsidian` | Index an Obsidian vault, respecting links and graph ranking |
| `prior_research` | Reuse prior `.prx` bundles as a knowledge base |
| `prxhub` | Pull from a [prxhub](https://prxhub.com) collection |

Add one via `parallect config` → "Data source plugins" → "Add", or edit `config.toml` directly.

## Pipeline plugins

External Python packages can hook into the research pipeline at four points:

```python
class ResearchPlugin:
    async def pre_research(self, query: str, providers: list[str]) -> str: ...
    async def post_provider(self, provider: str, result: Any) -> Any: ...
    async def post_synthesis(self, synthesis: Any) -> Any: ...
    async def post_bundle(self, bundle: Any) -> Any: ...
```

Register via the `parallect.plugins` entry point. See [`docs/PLUGINS.md`](docs/PLUGINS.md) for the full protocol.

## Output format

Every run saves to a [`.prx` bundle](https://github.com/parallect/prx-spec) — a gzipped tar archive:

```
research.prx
├── manifest.json          JSON-LD envelope: providers, cost, provenance
├── manifest.jws           Ed25519 signature of manifest hash
├── public-key.jwk         Public key for standalone verification
├── query.md               The research question
├── providers/
│   ├── perplexity/report.md + meta.json
│   ├── gemini/report.md + meta.json
│   └── ...
├── synthesis/
│   ├── report.md          Unified cross-provider synthesis
│   └── claims.json        Extracted atomic claims with provider attribution
├── sources/registry.json  Deduplicated, quality-scored sources
├── evidence/graph.json    Claim-to-source graph
└── provenance/graph.jsonld  W3C PROV-O provenance
```

Every file is human-readable text or JSON — `tar -xzf research.prx` and read in any editor.

Use the [`prx`](https://github.com/parallect/prx) CLI to read, validate, diff, merge, sign, and publish bundles.

## Documentation

- [Quickstart](docs/QUICKSTART.md) — get running in 5 minutes
- [Architecture](docs/ARCHITECTURE.md) — how the orchestrator works
- [Providers](docs/PROVIDERS.md) — write a custom provider adapter
- [Plugins](docs/PLUGINS.md) — hook into the pipeline or add a data source

## Development

```bash
git clone https://github.com/parallect/parallect.git
cd parallect
uv sync --group dev
uv run pytest tests/
uv run ruff check src/ tests/
```

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). For security reports, email `security@parallect.ai`.

## License

MIT — see [LICENSE](LICENSE).

---

Built by [SecureCoders](https://securecoders.com). A hosted managed version is available at [parallect.ai](https://parallect.ai) — same multi-provider pipeline, with billing, team workspaces, and a web dashboard.
