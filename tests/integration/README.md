# Integration tests

These tests verify Parallect against real, running backends. They are **not**
shipped to users and are skipped by default.

## Quick start

```bash
# Start LiteLLM (requires Docker + a local Ollama or OpenAI key)
docker compose -f tests/integration/docker-compose.yml up -d

# Wait for it to be healthy
docker compose -f tests/integration/docker-compose.yml ps

# Run
PARALLECT_INTEGRATION=1 LITELLM_URL=http://localhost:14000 \
    uv run pytest tests/integration/ -v

# Tear down
docker compose -f tests/integration/docker-compose.yml down
```

## What's covered

- `test_litellm_backend.py` — full end-to-end roundtrip against the LiteLLM
  container, exercising backend resolution + OpenAI-compatible adapter.
- `test_lmstudio_mock.py` — spins up an aiohttp test server that mimics LM
  Studio's `/v1/chat/completions` + `/v1/models` endpoints. Fully offline.
- `test_research_smoke.py` — end-to-end `research()` smoke test with mocked
  providers but a real LiteLLM container for synthesis.

## Env vars

| Var | Default | Purpose |
| --- | --- | --- |
| `PARALLECT_INTEGRATION` | unset | Must be `1` for integration tests to run |
| `LITELLM_URL` | `http://localhost:14000` | LiteLLM proxy base URL (no `/v1`) |
| `LITELLM_API_KEY` | `sk-parallect-test` | Master key, matches compose config |
| `LITELLM_MODEL` | `test-llama` | Model name configured in `litellm.config.yaml` |
