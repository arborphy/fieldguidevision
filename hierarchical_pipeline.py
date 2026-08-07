"""Pre-curated, evidence-gated plant identification workflow contracts.

This module deliberately does not classify pixels or infer biological truth. It
accepts reviewed results from those systems, decides which explicitly curated
child task is now eligible, and narrows source-vocabulary taxon candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from canonical_adapter import CanonicalRelease


TAXON_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


@dataclass(frozen=True)
class WorkflowNode:
    """A pre-curated task that may only run after its parent is confirmed."""

    node_id: str
    label: str
    kind: str  # taxon | notice | feature
    parent_id: str | None = None
    taxon_rank: str | None = None
    source_feature_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"taxon", "notice", "feature"}:
            raise ValueError("Workflow node kind must be taxon, notice, or feature")
        if self.kind == "taxon" and self.taxon_rank not in TAXON_RANKS:
            raise ValueError("Taxon workflow nodes require a supported taxonomic rank")
        if self.kind == "feature" and not self.source_feature_id:
            raise ValueError("Feature workflow nodes require a source feature ID")
        if self.kind != "feature" and self.source_feature_id:
            raise ValueError("Only feature workflow nodes may carry a source feature ID")


@dataclass(frozen=True)
class WorkflowEvidence:
    """A curator-confirmed result that can unlock a child workflow task."""

    node_id: str
    state: str  # confirmed | rejected | unknown
    notice_proposal_id: str | None = None
    annotation_id: str | None = None

    def __post_init__(self) -> None:
        if self.state not in {"confirmed", "rejected", "unknown"}:
            raise ValueError("Workflow evidence state must be confirmed, rejected, or unknown")


@dataclass(frozen=True)
class FeatureEvidence:
    """One reviewed source-vocabulary feature value tied to an accepted notice."""

    release_id: str
    feature_id: str
    value_id: str
    notice_proposal_id: str
    annotation_id: str | None = None
    confidence: float | None = None
    review_state: str = "confirmed"

    def __post_init__(self) -> None:
        if self.review_state != "confirmed":
            raise ValueError("Only curator-confirmed feature evidence can narrow taxa")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Feature evidence confidence must be normalized 0-1")


@dataclass(frozen=True)
class InstanceRecommendation:
    """A non-binding instance recommendation with complete evidence lineage."""

    taxon_name: str | None
    taxon_rank: str | None
    candidate_taxa: tuple[str, ...]
    release_ids: tuple[str, ...]
    evidence: tuple[FeatureEvidence, ...]
    status: str  # recommended | ambiguous | no_match | no_evidence


class CuratedWorkflow:
    """Validates and traverses an explicit taxon/notice/feature tree."""

    def __init__(self, nodes: Sequence[WorkflowNode]) -> None:
        indexed = {node.node_id: node for node in nodes}
        if len(indexed) != len(nodes):
            raise ValueError("Workflow node IDs must be unique")
        for node in nodes:
            if node.parent_id and node.parent_id not in indexed:
                raise ValueError(f"Workflow parent does not exist: {node.parent_id}")
        self._nodes = indexed
        self._assert_acyclic()

    def next_tasks(self, evidence: Sequence[WorkflowEvidence]) -> tuple[WorkflowNode, ...]:
        """Return unconfirmed roots or children of confirmed parent nodes."""
        states = {item.node_id: item.state for item in evidence}
        unknown_nodes = set(states) - set(self._nodes)
        if unknown_nodes:
            raise ValueError(f"Evidence references unknown workflow nodes: {sorted(unknown_nodes)}")
        eligible = [
            node
            for node in self._nodes.values()
            if node.node_id not in states
            and (node.parent_id is None or states.get(node.parent_id) == "confirmed")
        ]
        return tuple(sorted(eligible, key=lambda node: node.node_id))

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("Workflow hierarchy must not contain a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            parent_id = self._nodes[node_id].parent_id
            if parent_id:
                visit(parent_id)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self._nodes:
            visit(node_id)


def recommend_instance(
    releases: Sequence[CanonicalRelease],
    evidence: Sequence[FeatureEvidence],
    taxon_lineage: Mapping[str, Mapping[str, str]] | None = None,
) -> InstanceRecommendation:
    """Intersect release-scoped taxon claims without merging source assertions.

    Each feature evidence item must resolve to an allowed source value in its
    release. A candidate survives only when it has every reviewed value within
    that release. When more than one release contributes, their candidate sets
    are intersected by taxon name while release-specific evidence remains
    visible in the result.
    """
    if not evidence:
        return InstanceRecommendation(None, None, (), (), (), "no_evidence")

    releases_by_id = {release.vocabulary_release_id: release for release in releases}
    evidence_by_release: dict[str, list[FeatureEvidence]] = {}
    for item in evidence:
        if item.release_id not in releases_by_id:
            raise ValueError(f"Feature evidence references unknown release: {item.release_id}")
        evidence_by_release.setdefault(item.release_id, []).append(item)

    candidate_sets: list[set[str]] = []
    for release_id, release_evidence in sorted(evidence_by_release.items()):
        release = releases_by_id[release_id]
        requested = {(item.feature_id, item.value_id) for item in release_evidence}
        for feature_id, value_id in requested:
            value = release.feature_values.get(value_id)
            if value is None:
                raise ValueError(f"Unknown source value for release {release_id}: {value_id}")
            if value["feature_name"] != feature_id:
                raise ValueError(f"Source value {value_id} does not belong to feature {feature_id}")
        candidates = {
            taxon_name
            for taxon_name in release.taxa
            if requested.issubset({
                (claim["feature_name"], claim["vocab_id"])
                for claim in release.claims
                if claim["taxon_name"] == taxon_name
            })
        }
        candidate_sets.append(candidates)

    candidates = set.intersection(*candidate_sets) if candidate_sets else set()
    candidate_taxa = tuple(sorted(candidates))
    if not candidate_taxa:
        return InstanceRecommendation(
            None, None, (), tuple(sorted(evidence_by_release)), tuple(evidence), "no_match"
        )
    rank, taxon_name = _most_specific_shared_taxon(candidate_taxa, taxon_lineage or {})
    return InstanceRecommendation(
        taxon_name,
        rank,
        candidate_taxa,
        tuple(sorted(evidence_by_release)),
        tuple(evidence),
        "recommended" if len(candidate_taxa) == 1 else "ambiguous",
    )


def _most_specific_shared_taxon(
    candidates: Iterable[str], lineage: Mapping[str, Mapping[str, str]]
) -> tuple[str | None, str | None]:
    candidate_list = tuple(candidates)
    if len(candidate_list) == 1:
        return "species", candidate_list[0]
    for rank in reversed(TAXON_RANKS[:-1]):
        values = {lineage.get(taxon, {}).get(rank) for taxon in candidate_list}
        if len(values) == 1 and None not in values:
            return rank, values.pop()
    return None, None
