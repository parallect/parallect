# Quickstart Guide

Get multi-provider AI research running in under 5 minutes.

## Installation

```bash
pip install parallect-cli
```

With all provider SDKs:

```bash
pip install "parallect-cli[all]"
```

> The legacy `pip install parallect` also still works — it transparently
> pulls in `parallect-cli` and emits a one-time `DeprecationWarning` so
> you know to update your install command.

## Configuration

Set your API keys via environment variables or config file.

### Environment Variables

```bash
export PARALLECT_PERPLEXITY_API_KEY=pplx-...
export PARALLECT_GOOGLE_API_KEY=AIza...
export PARALLECT_OPENAI_API_KEY=sk-...
export PARALLECT_ANTHROPIC_API_KEY=sk-ant-...
export PARALLECT_XAI_API_KEY=xai-...
```

### Config File

Create `~/.config/parallect/config.toml`:

```toml
[providers]
perplexity_api_key = "pplx-..."
google_api_key = "AIza..."
openai_api_key = "sk-..."
anthropic_api_key = "sk-ant-..."
xai_api_key = "xai-..."

[defaults]
providers = ["perplexity", "gemini", "openai"]
synthesize_with = "anthropic"
budget_cap_usd = 2.00

[signing]
auto_sign = true
identity = "your-name"
```

## Your First Research

```bash
# Basic research with default providers
parallect research "What are the latest advances in quantum computing?"

# Specify providers
parallect research "Rust vs Go for systems programming" -p perplexity,gemini,openai

# Deep research mode (premium models)
parallect research "Impact of AI on job markets" --deep

# Save to a specific location
parallect research "Climate change mitigation" -o climate.prx
```

## Viewing Results

```bash
# Open in terminal UI (requires prx)
prx open climate.prx

# Validate the bundle
parallect validate climate.prx

# List your local bundles
parallect list
```

## Local-Only Research

No API keys needed — uses Ollama:

```bash
# Make sure Ollama is running
ollama serve

# Research with local models only
parallect research "Explain transformer architecture" --local
```

## Follow-On Research

```bash
# Continue from a previous bundle
parallect continue climate.prx "What about carbon capture specifically?"
```

## Signing Your Bundles

```bash
# Generate a signing key (one-time)
parallect keys generate

# Research auto-signs when a key exists
parallect research "My query" -o signed.prx

# Skip signing
parallect research "My query" --no-sign

# Verify a bundle's signatures
parallect verify signed.prx
```

## What's in a .prx Bundle?

A `.prx` file is a gzipped tar archive containing:

```
manifest.json          # Bundle metadata, providers, timestamps
query.md               # Original research question
providers/
  perplexity.md        # Each provider's research report
  gemini.md
  openai.md
  citations/           # Per-provider citations (JSON)
synthesis/
  report.md            # Unified synthesis across providers
  claims.json          # Extracted claims with cross-references
sources/
  registry.json        # Deduplicated source registry
attestations/          # Cryptographic signatures
```

## Next Steps

- [Custom Providers Guide](PROVIDERS.md) — Write your own provider adapter
- [Plugin Guide](PLUGINS.md) — Extend parallect with hooks
- [Architecture](ARCHITECTURE.md) — How the orchestrator works
