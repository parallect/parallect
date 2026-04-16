"""Shared config for integration tests.

Integration tests are disabled by default. To run them:

    PARALLECT_INTEGRATION=1 uv run pytest tests/integration/ -v

Most require a live LiteLLM container. Start one with:

    docker compose -f tests/integration/docker-compose.yml up -d
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PARALLECT_INTEGRATION"):
        return
    skip = pytest.mark.skip(reason="set PARALLECT_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
