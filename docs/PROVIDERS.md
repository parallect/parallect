# Custom Provider Guide

Create your own research provider for parallect.

## Provider Protocol

Every provider must implement the `AsyncResearchProvider` protocol:

```python
from parallect.providers import AsyncResearchProvider, ProviderResult

class MyProvider:
    @property
    def name(self) -> str:
        return "my-provider"

    async def research(self, query: str) -> ProviderResult:
        # Your research logic here
        result = await call_my_api(query)
        return ProviderResult(
            provider=self.name,
            status="completed",
            report_markdown=result.text,
            citations=[{"url": c.url, "title": c.title} for c in result.sources],
            model="my-model-v1",
            cost_usd=0.05,
            duration_seconds=10.0,
        )

    def estimate_cost(self, query: str) -> float:
        return 0.05

    def is_available(self) -> bool:
        return bool(os.environ.get("MY_API_KEY"))
```

## ProviderResult Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | `str` | Yes | Provider name |
| `status` | `str` | Yes | `"completed"`, `"partial"`, or `"failed"` |
| `report_markdown` | `str` | Yes | Research report in Markdown |
| `citations` | `list[dict]` | No | List of `{"url": ..., "title": ...}` |
| `model` | `str` | No | Model identifier used |
| `cost_usd` | `float` | No | Estimated cost of the call |
| `duration_seconds` | `float` | No | Wall-clock time |
| `tokens` | `dict` | No | Token usage breakdown |
| `error` | `str` | No | Error message if status is "failed" |

## Wrapping OpenAI-Compatible APIs

If your provider exposes an OpenAI-compatible endpoint, use the built-in base class:

```python
from parallect.providers.openai_compat import OpenAICompatibleProvider

class MyCustomProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: str):
        super().__init__(
            name="my-custom",
            base_url="https://api.example.com/v1",
            model="gpt-custom",
            api_key=api_key,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)
```

This gives you `research()`, `estimate_cost()`, and streaming for free.

## Async Polling Pattern

For providers that start a job and poll for results (like Gemini Deep Research):

```python
class AsyncPollingProvider:
    @property
    def name(self) -> str:
        return "polling-provider"

    async def research(self, query: str) -> ProviderResult:
        # Start the job
        job_id = await self._start_job(query)

        # Poll until complete
        while True:
            status = await self._check_status(job_id)
            if status.done:
                break
            await asyncio.sleep(5)

        # Fetch results
        result = await self._get_result(job_id)
        return ProviderResult(
            provider=self.name,
            status="completed",
            report_markdown=result.text,
        )
```

## Registering Your Provider

### Via CLI

Add to the provider map in your fork, or use config:

```toml
# parallect.toml or ~/.config/parallect/config.toml
[providers]
my-provider = "my_package.providers:MyProvider"
```

### Via Entry Points (for pip-installable packages)

In your package's `pyproject.toml`:

```toml
[project.entry-points."parallect.providers"]
my-provider = "my_package.providers:MyProvider"
```

## Testing Your Provider

```python
import pytest
from my_package import MyProvider

@pytest.mark.asyncio
async def test_my_provider():
    provider = MyProvider(api_key="test-key")
    assert provider.name == "my-provider"
    assert provider.is_available()

    result = await provider.research("test query")
    assert result.status == "completed"
    assert result.report_markdown
```

## Built-In Providers

| Provider | Module | Type | Deep Research |
|----------|--------|------|---------------|
| Perplexity | `providers.perplexity` | Direct | Sonar Deep Research |
| Gemini | `providers.gemini` | Async polling | Gemini Deep Research |
| OpenAI | `providers.openai_dr` | Async polling | Deep Research |
| Grok | `providers.grok` | Direct | grok-3 |
| Anthropic | `providers.anthropic` | Direct | Extended thinking + web search |
| Ollama | `providers.ollama` | OpenAI-compat | Local models |
| LM Studio | `providers.lmstudio` | OpenAI-compat | Local models |
| LDR | `providers.ldr` | Local deep research | Iterative search |
