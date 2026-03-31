# parallect

Open-source multi-provider AI deep research from the command line. Bring your own API keys (BYOK).

Fan out research queries to multiple frontier AI providers (Perplexity, Gemini, OpenAI, Grok, Anthropic) in parallel, then synthesize their outputs into a single unified report with cross-referenced citations, claim extraction, and confidence scoring.

Output is a `.prx` bundle — an open, portable research format defined by [prx-spec](https://github.com/parallect-ai/prx-spec).

## Install

```bash
pip install parallect
```

## Quick Start

```bash
# Configure API keys
parallect config

# Run research
parallect research "What are the leading theories on dark matter?"

# Use specific providers
parallect research "quantum computing progress 2025" -p perplexity,openai,anthropic

# Deep research mode (premium models, higher cost)
parallect research "CRISPR gene therapy developments" --deep

# Local-only (Ollama)
parallect research "explain transformers" --local

# Follow-on research
parallect continue output.prx "What about practical applications?"
```

## Providers

| Provider | Default Model | Deep Model | API Key Env Var |
|----------|--------------|------------|-----------------|
| Perplexity | sonar-deep-research | — | `PARALLECT_PERPLEXITY_API_KEY` |
| OpenAI | gpt-4o-mini | o3-deep-research | `PARALLECT_OPENAI_API_KEY` |
| Gemini | gemini-2.5-flash | gemini-2.5-pro | `PARALLECT_GOOGLE_API_KEY` |
| Anthropic | claude-sonnet-4 | claude-opus-4 | `PARALLECT_ANTHROPIC_API_KEY` |
| Grok | grok-3 | grok-4 | `PARALLECT_XAI_API_KEY` |
| Ollama | llama3.2 | — | (local) |

## Output Format

Research results are saved as `.prx` bundles — portable ZIP archives containing:
- `manifest.json` — metadata, providers used, cost
- `query.md` — the original research query
- `providers/*.md` — individual provider reports
- `synthesis.md` — unified cross-provider synthesis
- `claims.json` — extracted atomic claims with provider attribution

Use the [prx](https://github.com/parallect-ai/prx) CLI to read, validate, merge, and share bundles.

## License

MIT
