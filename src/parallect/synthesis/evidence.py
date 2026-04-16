"""Build an evidence graph linking claims to sources.

This is the cheap, provider-level mapping — for every claim, for every
provider that supports the claim, we emit one edge to each source that
same provider cited. It's an over-approximation (not semantic matching
between claim text and source content), but it's what the existing data
can support without any additional LLM calls.

A follow-up can add claim-text → source-snippet matching for tighter
precision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prx_spec.models.evidence import EvidenceGraph
    from prx_spec.models.sources import SourcesRegistry
    from prx_spec.models.synthesis import ClaimsFile


def build_evidence_graph(
    claims_file: "ClaimsFile",
    sources_registry: "SourcesRegistry",
) -> "EvidenceGraph | None":
    """Connect each claim to every source cited by a provider that supports it.

    Returns ``None`` if the inputs don't produce any edges (either there are
    no claims, no sources, or no overlap between supporting providers and
    cited-by provider lists). Otherwise returns an ``EvidenceGraph`` with
    one ``EvidenceEdge`` per (claim × supporting_provider × that_provider's
    cited source).
    """
    from prx_spec.models.evidence import EvidenceEdge, EvidenceGraph

    if not claims_file or not sources_registry:
        return None
    if not claims_file.claims or not sources_registry.sources:
        return None

    # Pre-index sources by provider so each claim lookup is O(sources) once.
    # provider_name -> list of source ids
    sources_by_provider: dict[str, list[str]] = {}
    for src in sources_registry.sources:
        for prov in (src.cited_by_providers or []):
            sources_by_provider.setdefault(prov, []).append(src.id)

    edges: list[EvidenceEdge] = []
    for claim in claims_file.claims:
        for provider in (claim.providers_supporting or []):
            for source_id in sources_by_provider.get(provider, []):
                edge_id = f"ev_{claim.id}_{provider}_{source_id}"
                edges.append(
                    EvidenceEdge(
                        id=edge_id,
                        claim_id=claim.id,
                        source_id=source_id,
                        relation="supports",
                        discovered_by_provider=provider,
                    )
                )

    if not edges:
        return None

    return EvidenceGraph(evidence=edges)
