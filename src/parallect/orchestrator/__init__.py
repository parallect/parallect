"""Orchestrator: parallel fan-out of research queries to multiple providers."""

from parallect.orchestrator.parallel import ProviderOutcome, fan_out, research

__all__ = ["ProviderOutcome", "fan_out", "research"]
