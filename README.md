# parallect

[![PyPI version](https://img.shields.io/pypi/v/parallect.svg)](https://pypi.org/project/parallect/)
[![Python](https://img.shields.io/pypi/pyversions/parallect.svg)](https://pypi.org/project/parallect/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/parallect/parallect/actions/workflows/ci.yml/badge.svg)](https://github.com/parallect/parallect/actions/workflows/ci.yml)

**Multi-provider AI deep research from the command line. Bring your own keys.**

`parallect` fans a research query out to multiple frontier AI providers in parallel (Perplexity, Gemini, OpenAI, Grok, Anthropic, plus local models via Ollama and LM Studio), synthesizes their outputs into a single report with cross-referenced citations, and packages the result as a portable [`.prx`](https://github.com/parallect/prx-spec) bundle.

```bash
$ parallect research "What are the leading theories on dark matter?"
```

## Why?

- **No single provider knows everything.** Different models see different sources, reason differently, and disagree in useful ways. Ask several at once.
- **Portable output.** Results save to an open, signed [`.prx`](https://github.com/parallect/prx-spec) archive — readable in any text editor, publishable to [prxhub](https://prxhub.com), verifiable via Ed25519.
- **BYOK.** Your keys, your bill, your data. No intermediate service.
- **Local-first option.** Run against Ollama or LM Studio for fully offline research.

## Install

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

# Deep research mode (premium models, higher cost)
parallect research "CRISPR gene therapy developments" --deep

# Local-only (Ollama / LM Studio)
parallect research "explain transformers" --local

# Follow-on research — continues from a prior bundle
parallect continue output.prx "What about practical applications?"
```

## Commands

| Command | Purpose |
|---|---|
| `parallect config` | Interactive TUI for providers, API keys, backends, plugins |
| `parallect research <query>` | Run a new research query |
| `parallect continue <bundle> <query>` | Follow-on research rooted in an existing bundle |
| `parallect enhance <bundle>` | Send a bundle to the hosted Parallect API for extra claim extraction + evidence graph |
| `parallect jobs status <id>` | Check a SaaS-mode job |
| `parallect jobs download <id>` | Download a completed SaaS-mode bundle |
| `parallect plugins list` | List installed data-source and pipeline plugins |
| `parallect plugins index <type>[:<name>]` | Index a data source (e.g. a local filesystem) |
| `parallect plugins config <type>[:<name>]` | Configure a plugin instance |
| `parallect plugins status` | Show per-plugin health / freshness |

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
| `filesystem` | Index a directory of markdown/text/PDF |
| `obsidian` | Index an Obsidian vault, respecting links + graph ranking |
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

Register via the `parallect.plugins` entry point. See [`docs/PLUGINS.md`](docs/PLUGINS.md) for the full protocol and examples.

## Output format

Every research run saves to a [`.prx` bundle](https://github.com/parallect/prx-spec) — a gzipped tar archive containing:

- `manifest.json` — JSON-LD envelope with providers, cost, and provenance
- `query.md` — the research question
- `providers/<name>/report.md` — each provider's raw output
- `synthesis.md` — unified cross-provider synthesis
- `claims.json` — extracted atomic claims with provider attribution
- `sources/registry.json` — deduplicated, quality-scored sources
- `evidence/` — claim-to-source graph
- `manifest.jws` + `public-key.jwk` — Ed25519 signature (if a signing key is configured)

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
