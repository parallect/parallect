"""Synthesis engines: LLM-based and concatenation mode."""

from parallect.synthesis.concat import concatenate
from parallect.synthesis.llm import SynthesisResult, synthesize

__all__ = ["SynthesisResult", "concatenate", "synthesize"]
