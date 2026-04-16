"""Probe local LLM servers to enable local-first config defaults.

During first-time `parallect config` setup we try LM Studio (port 1234) and
Ollama (port 11434). Whichever responds gets used as the default synthesis +
embeddings backend. If neither responds, the user is prompted to configure a
cloud backend.

These probes are deliberately fast (200ms timeout) so `parallect config`
remains snappy even when no local LLM is running.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

LMSTUDIO_MODELS_URL = "http://localhost:1234/v1/models"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


@dataclass(frozen=True)
class LocalProbeResult:
    lmstudio_reachable: bool
    ollama_reachable: bool

    @property
    def any_reachable(self) -> bool:
        return self.lmstudio_reachable or self.ollama_reachable

    @property
    def preferred_backend(self) -> str | None:
        """Pick a preferred local backend.

        Preference: LM Studio > Ollama. LM Studio tends to ship with stronger
        default models (gpt-oss-120b) and has a GUI that makes model install
        easier for first-time users.
        """
        if self.lmstudio_reachable:
            return "lmstudio"
        if self.ollama_reachable:
            return "ollama"
        return None


def _probe_url(url: str, timeout: float) -> bool:
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            return response.status_code == 200
    except Exception:
        return False


def probe_local_backends(timeout: float = 0.2) -> LocalProbeResult:
    """Probe the two conventional local endpoints.

    Both probes are sequential rather than concurrent -- we're only saving
    a total of ~400ms, and running them one at a time keeps error paths
    simple (no asyncio, no thread pools).
    """
    lmstudio = _probe_url(LMSTUDIO_MODELS_URL, timeout)
    ollama = _probe_url(OLLAMA_TAGS_URL, timeout)
    return LocalProbeResult(lmstudio_reachable=lmstudio, ollama_reachable=ollama)
