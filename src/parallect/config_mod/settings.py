"""Configuration management via pydantic-settings + TOML."""

from __future__ import annotations

import sys
from pathlib import Path

import platformdirs
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _user_config_path() -> Path:
    return Path(platformdirs.user_config_dir("parallect")) / "config.toml"


def _project_config_path() -> Path:
    return Path.cwd() / "parallect.toml"


class ParallectSettings(BaseSettings):
    """Settings loaded from TOML config files, env vars, and CLI args.

    Precedence (highest to lowest):
    1. Explicit constructor args
    2. Environment variables (PARALLECT_ prefix)
    3. Project-local parallect.toml
    4. User config ~/.config/parallect/config.toml
    5. Defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="PARALLECT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Provider API keys
    perplexity_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    xai_api_key: str = ""
    anthropic_api_key: str = ""

    # Defaults
    providers: list[str] = Field(default_factory=lambda: ["perplexity", "gemini", "openai"])
    synthesize_with: str = "anthropic"
    budget_cap_usd: float = 2.00
    output_dir: str = "."
    timeout_per_provider: float = 120.0

    # Parallect API (for enhance command)
    parallect_api_key: str = ""

    # Signing
    key_path: str = ""
    auto_sign: bool = True
    identity: str = ""

    # Local providers
    ollama_host: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.2"
    lmstudio_host: str = "http://localhost:1234"
    lmstudio_default_model: str = "default"

    @classmethod
    def load(cls) -> ParallectSettings:
        """Load settings from all sources with proper precedence."""
        config_data: dict = {}

        # Load user config (lowest precedence)
        user_path = _user_config_path()
        if user_path.exists():
            with open(user_path, "rb") as f:
                raw = tomllib.load(f)
                config_data.update(_flatten_toml(raw))

        # Load project config (overrides user config)
        project_path = _project_config_path()
        if project_path.exists():
            with open(project_path, "rb") as f:
                raw = tomllib.load(f)
                config_data.update(_flatten_toml(raw))

        return cls(**config_data)


def _flatten_toml(data: dict, prefix: str = "") -> dict:
    """Flatten nested TOML sections into dot-free keys matching settings fields."""
    result: dict = {}
    key_map = {
        "providers.perplexity_api_key": "perplexity_api_key",
        "providers.google_api_key": "google_api_key",
        "providers.openai_api_key": "openai_api_key",
        "providers.xai_api_key": "xai_api_key",
        "providers.anthropic_api_key": "anthropic_api_key",
        "defaults.providers": "providers",
        "defaults.synthesize_with": "synthesize_with",
        "defaults.budget_cap_usd": "budget_cap_usd",
        "defaults.output_dir": "output_dir",
        "defaults.timeout_per_provider": "timeout_per_provider",
        "parallect_api.api_key": "parallect_api_key",
        "signing.key_path": "key_path",
        "signing.auto_sign": "auto_sign",
        "signing.identity": "identity",
        "local.ollama_host": "ollama_host",
        "local.ollama_default_model": "ollama_default_model",
        "local.lmstudio_host": "lmstudio_host",
        "local.lmstudio_default_model": "lmstudio_default_model",
    }

    for section_key, value in data.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                full_key = f"{section_key}.{inner_key}"
                if full_key in key_map:
                    result[key_map[full_key]] = inner_value
        elif prefix == "" and section_key in ParallectSettings.model_fields:
            result[section_key] = value

    return result
