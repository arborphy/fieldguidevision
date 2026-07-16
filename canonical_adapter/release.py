"""Read source-distinct canonical DeVo releases for experiment inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_TABLES = (
    "devo_registry",
    "devo_features",
    "devo_feature_values",
    "devo_images",
    "devo_annotations",
    "devo_taxa",
    "devo_taxon_feature_value",
)


@dataclass(frozen=True)
class ExperimentCase:
    """One canonical taxon-feature-value claim with source release identity."""

    vocabulary_name: str
    vocabulary_release_id: str
    canonical_taxon_id: str
    canonical_feature_id: str
    canonical_value_id: str
    source_taxon_id: str | None
    source_value_id: str | None
    assertion_source: str
    source_context_id: str | None


@dataclass(frozen=True)
class CanonicalRelease:
    """Validated release data and indexes needed to build experiment cases."""

    path: Path
    data_hash: str
    contract_version: str
    vocabulary_name: str
    vocabulary_release_id: str
    source_version: str
    source_content_hash: str
    feature_values: dict[str, dict[str, Any]]
    taxa: dict[str, dict[str, Any]]
    claims: tuple[dict[str, Any], ...]

    def experiment_cases_for_taxon(self, canonical_taxon_id: str) -> tuple[ExperimentCase, ...]:
        """Return source-distinct claims for a taxon without deriving new IDs."""
        if canonical_taxon_id not in self.taxa:
            raise KeyError(f"Unknown canonical taxon ID: {canonical_taxon_id}")

        taxon = self.taxa[canonical_taxon_id]
        cases: list[ExperimentCase] = []
        for claim in self.claims:
            if claim["taxon_name"] != canonical_taxon_id:
                continue
            value = self.feature_values[claim["vocab_id"]]
            cases.append(
                ExperimentCase(
                    vocabulary_name=self.vocabulary_name,
                    vocabulary_release_id=self.vocabulary_release_id,
                    canonical_taxon_id=canonical_taxon_id,
                    canonical_feature_id=claim["feature_name"],
                    canonical_value_id=claim["vocab_id"],
                    source_taxon_id=taxon.get("source_taxon_id"),
                    source_value_id=value.get("source_value_id"),
                    assertion_source=claim["assertion_source"],
                    source_context_id=claim.get("source_context_id"),
                )
            )
        return tuple(cases)


def load_canonical_release(path: str | Path) -> CanonicalRelease:
    """Load a canonical DeVo JSON release and preserve its provenance verbatim."""
    release_path = Path(path)
    raw_bytes = release_path.read_bytes()
    data = json.loads(raw_bytes)
    _validate_release_shape(data)

    manifest = data["manifest"]
    source = manifest["source"]
    vocabulary_name = source["devo_name"]
    registry_names = {row["devo_name"] for row in data["devo_registry"]}
    if registry_names != {vocabulary_name}:
        raise ValueError("Release registry must contain exactly the manifest vocabulary")

    feature_values = _index_rows(data["devo_feature_values"], "vocab_id", "feature value")
    taxa = _index_rows(data["devo_taxa"], "taxon_name", "taxon")
    claims = tuple(data["devo_taxon_feature_value"])
    for claim in claims:
        if claim["devo_name"] != vocabulary_name:
            raise ValueError("Claim vocabulary does not match the manifest vocabulary")
        if claim["taxon_name"] not in taxa:
            raise ValueError(f"Claim references unknown canonical taxon: {claim['taxon_name']}")
        value = feature_values.get(claim["vocab_id"])
        if value is None:
            raise ValueError(f"Claim references unknown canonical value: {claim['vocab_id']}")
        if value["feature_name"] != claim["feature_name"]:
            raise ValueError("Claim feature does not match its canonical value")

    return CanonicalRelease(
        path=release_path,
        data_hash=f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}",
        contract_version=manifest["contract_version"],
        vocabulary_name=vocabulary_name,
        vocabulary_release_id=manifest["release_id"],
        source_version=source["version"],
        source_content_hash=source["content_hash"],
        feature_values=feature_values,
        taxa=taxa,
        claims=claims,
    )


def _validate_release_shape(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Canonical release must be a JSON object")
    for table in REQUIRED_TABLES:
        if table not in data or not isinstance(data[table], list):
            raise ValueError(f"Canonical release is missing table: {table}")

    manifest = data.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Canonical release is missing a manifest")
    for key in ("contract_version", "release_id", "source"):
        if key not in manifest:
            raise ValueError(f"Canonical release manifest is missing: {key}")
    if not isinstance(manifest["source"], dict):
        raise ValueError("Canonical release manifest source must be an object")
    for key in ("devo_name", "version", "content_hash"):
        if key not in manifest["source"]:
            raise ValueError(f"Canonical release source is missing: {key}")


def _index_rows(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get(key):
            raise ValueError(f"Canonical {label} row is missing {key}")
        row_key = row[key]
        if row_key in indexed:
            raise ValueError(f"Canonical release has duplicate {label} ID: {row_key}")
        indexed[row_key] = row
    return indexed
