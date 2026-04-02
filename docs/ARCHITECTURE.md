# Architecture

## Overview

Parallect's core pipeline: Query → Fan-out → Collect → Synthesize → Bundle.

```
                    ┌─────────────┐
                    │    Query    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ pre_research│ (plugin hook)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Provider 1│ │Provider 2│ │Provider N│  (asyncio.gather)
        └────┬─────┘ └────┬─────┘ └────┬─────┘
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │post_prov │ │post_prov │ │post_prov │  (plugin hooks)
        └────┬─────┘ └────┬─────┘ └────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │  Synthesis  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │post_synth   │ (plugin hook)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Bundle Build│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ post_bundle │ (plugin hook)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Auto-Sign  │ (if key exists)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Write .prx │
                    └─────────────┘
```

## Key Modules

### `orchestrator/parallel.py`

The main `research()` function:

1. Budget estimation and cap check
2. Plugin `pre_research` hook (may modify query)
3. `fan_out()` — parallel `asyncio.gather` with per-provider timeouts
4. Plugin `post_provider` hooks on each result
5. Optional LLM synthesis (or concatenation fallback)
6. Plugin `post_synthesis` hook
7. Bundle assembly (`BundleData` from prx-spec)
8. Plugin `post_bundle` hook
9. Auto-signing (if local key exists and `auto_sign=true`)
10. Write `.prx` archive to disk

### `providers/`

Each provider implements the `AsyncResearchProvider` protocol:

- `perplexity.py` — Sonar Deep Research (direct response)
- `gemini.py` — Gemini Deep Research (async polling via google-genai)
- `openai_dr.py` — OpenAI Deep Research (Responses API polling)
- `grok.py` — Grok (direct chat completion)
- `anthropic.py` — Claude with extended thinking + web search
- `openai_compat.py` — Base class for OpenAI-compatible APIs
- `ollama.py` — Ollama (wraps OpenAI-compat)
- `lmstudio.py` — LM Studio (wraps OpenAI-compat)
- `ldr.py` — Local Deep Research integration
- `registry.py` — Provider discovery and registration

### `synthesis/`

- `llm.py` — LLM-based synthesis using any provider
- `concat.py` — No-LLM fallback: concatenation with headers

### `claims/`

- `extract.py` — LLM structured output for claim extraction

### `plugins/`

- `__init__.py` — `PluginManager` with entry point discovery and hook execution

### `config_mod/`

- `settings.py` — `ParallectSettings` with TOML + env var loading

## Error Handling

- **Partial failure**: Individual provider failures don't halt research. The orchestrator captures errors and continues with successful providers.
- **Budget enforcement**: Pre-flight cost estimation. Research aborted if estimate exceeds cap.
- **Synthesis failure**: Non-fatal. Bundle is still produced without synthesis.
- **Plugin failures**: Logged but don't halt the pipeline.

## Signing Flow

1. After bundle assembly, check `settings.auto_sign` and `--no-sign` flag
2. Load Ed25519 private key from `~/.config/parallect/keys/prx_signing.key`
3. Create a bundle-level `Attestation` (type: `"bundle"`, signer type: `"researcher"`)
4. Sign using `prx_spec.sign_attestation()`
5. Store in `bundle.attestations`
6. Written to `.prx` archive in `attestations/` directory
