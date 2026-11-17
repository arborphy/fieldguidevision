# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Dirr weak-label join with a hard anatomy gate.

Dirr claims are *taxon-level* assertions. A crop's image evidence may only be
supervised by a claim when the crop's detected anatomy is compatible with the
claim's feature:

===============  ==============================================
feature          eligible crop anatomy
===============  ==============================================
leaf_type        leaf
leaf_shape_margins leaf
leaf_arrangement leaf
inflorescences   flower or inflorescence
leaf_type_conifer needle
===============  ==============================================

This module never derives a box or a label from the image content. It attaches
the claim as a weak label (with provenance) and records every rejected row and
the reason, so the dataset build can report exactly how many claims were used
as image evidence and how many were refused.
"""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    TargetSpec,
    canonical_binomial,
    canonical_binomial_aliases,
    canonical_label,
)

DEFAULT_DB_COMMAND: tuple[str, ...] = (
    "docker",
    "exec",
    "arq-compose-postgres",
    "psql",
    "-U",
    "arq",
    "-d",
    "arq",
)

# The Dirr features this sprint consumes. Used when the caller does not
# narrow the export/join.
DIRR_FEATURES: tuple[str, ...] = (
    "leaf_type",
    "leaf_shape_margins",
    "leaf_arrangement",
    "leaf_type_conifer",
    "inflorescences",
)

_SAFE_IDENT = re.compile(r"^[a-z0-9_]+$")
_SAFE_COLUMN = re.compile(r"^[a-z0-9_-]+$")


@dataclass(frozen=True)
class Claim:
    taxon_name: str
    feature_name: str
    value_label: str
    devo_name: str = "dirr"
    provenance: str | None = None

    @property
    def species_key(self) -> str | None:
        return canonical_binomial(self.taxon_name)


def _claims_from_payload(payload: Any) -> list[Claim]:
    if isinstance(payload, dict):
        raw = payload.get("claims", payload.get("rows", []))
    else:
        raw = payload
    if not isinstance(raw, list):
        raise ValueError("claims JSON must be a list or an object with a 'claims' list")
    claims: list[Claim] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        taxon = item.get("taxon_name") or item.get("scientific_name")
        feature = item.get("feature_name") or item.get("feature")
        value = item.get("value_label") or item.get("value")
        if not taxon or not feature or value is None:
            continue
        claims.append(
            Claim(
                taxon_name=str(taxon),
                feature_name=str(feature),
                value_label=str(value),
                devo_name=str(item.get("devo_name") or item.get("devo") or "dirr"),
                provenance=str(item.get("provenance") or "") or None,
            )
        )
    return claims


def load_claims_json(path: str | Path) -> list[Claim]:
    return _claims_from_payload(json.loads(Path(path).read_text()))


def _run_psql(db_command: Sequence[str], query: str) -> list[dict[str, str]]:
    csv.field_size_limit(64 * 1024 * 1024)
    command = [*db_command, "-c", f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER true)"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return list(csv.DictReader(io.StringIO(completed.stdout)))


def _feature_filter(features: Iterable[str]) -> tuple[list[str], str]:
    wanted = sorted({str(f) for f in features})
    for feature in wanted:
        if not _SAFE_IDENT.match(feature):
            raise ValueError(f"unsafe feature name: {feature!r}")
    if not wanted:
        return [], ""
    return wanted, " AND feature_name IN (" + ", ".join(f"'{f}'" for f in wanted) + ")"


def _wide_feature_columns(
    features: Iterable[str], db_command: Sequence[str] | None = None
) -> list[tuple[str, str, str]]:
    """Return ``(column, feature_name, value_label)`` for devo_taxa wide columns."""

    wanted = set(features)
    if not wanted:
        return []
    rows = _run_psql(
        list(db_command or DEFAULT_DB_COMMAND),
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name='devo_taxa' "
        "AND column_name LIKE '%-%' ORDER BY column_name",
    )
    out: list[tuple[str, str, str]] = []
    for row in rows:
        column = row.get("column_name") or ""
        if not _SAFE_COLUMN.match(column):
            continue
        feature, _, value = column.partition("-")
        if feature in wanted and value:
            out.append((column, feature, value))
    return out


def load_claims_db(
    *,
    db_command: Sequence[str] | None = None,
    devo_name: str = "dirr",
    features: Sequence[str] | None = None,
) -> list[Claim]:
    """Read Dirr claims, preferring ``devo_taxon_claims`` then wide ``devo_taxa``.

    The local app DB currently stores Dirr as the wide ``devo_taxa`` one-hot
    matrix; other devos use ``devo_taxon_claims``. Both shapes are normalized
    to the same :class:`Claim` list. Read-only by construction (COPY ... TO
    STDOUT).
    """

    if not _SAFE_IDENT.match(devo_name):
        raise ValueError(f"unsafe devo name: {devo_name!r}")
    command = list(db_command or DEFAULT_DB_COMMAND)
    wanted, feature_clause = _feature_filter(features if features is not None else DIRR_FEATURES)
    query = (
        "SELECT devo_name, taxon_name, feature_name, value_label, "
        "COALESCE(provenance, '') AS provenance FROM devo_taxon_claims "
        f"WHERE devo_name = '{devo_name}'{feature_clause} ORDER BY taxon_name, feature_name, value_label"
    )
    rows = _run_psql(command, query)
    claims = [
        Claim(
            taxon_name=str(row["taxon_name"]),
            feature_name=str(row["feature_name"]),
            value_label=str(row["value_label"]),
            devo_name=str(row.get("devo_name") or devo_name),
            provenance=str(row.get("provenance") or "") or None,
        )
        for row in rows
        if row.get("taxon_name") and row.get("feature_name") and row.get("value_label") is not None
    ]

    # The local Dirr source is the wide devo_taxa table; other devos use the
    # claims table. Union per feature so a partially populated claims table
    # cannot hide a feature that only exists in the wide table.
    found = {claim.feature_name for claim in claims}
    missing = [feature for feature in wanted if feature not in found]
    if not missing:
        return claims
    columns = _wide_feature_columns(missing, command)
    if not columns:
        return claims
    select_columns = ", ".join(f'"{column}"' for column, _, _ in columns)
    # A single pass over taxa; unpivot in Python to avoid a huge UNION query.
    wide_rows = _run_psql(
        command,
        f"SELECT taxon_name, {select_columns} FROM devo_taxa WHERE devo_name = '{devo_name}'",
    )
    for row in wide_rows:
        taxon = row.get("taxon_name")
        if not taxon:
            continue
        for column, feature, value in columns:
            raw = row.get(column)
            if raw in (None, "", "0", "0.0", "f", "false", "False"):
                continue
            claims.append(
                Claim(
                    taxon_name=str(taxon),
                    feature_name=feature,
                    value_label=value,
                    devo_name=devo_name,
                    provenance="devo_taxa",
                )
            )
    return claims


def load_claims(
    *,
    claims_json: str | Path | None = None,
    db_command: Sequence[str] | None = None,
    devo_name: str = "dirr",
    features: Sequence[str] | None = None,
) -> list[Claim]:
    if claims_json:
        return load_claims_json(claims_json)
    return load_claims_db(db_command=db_command, devo_name=devo_name, features=features)


class ClaimsIndex:
    """Species -> feature -> raw values index."""

    def __init__(self, claims: Iterable[Claim]) -> None:
        self.by_species: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.provenance: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.n_claims = 0
        self.n_skipped = 0
        for claim in claims:
            keys = canonical_binomial_aliases(claim.taxon_name)
            if not keys:
                self.n_skipped += 1
                continue
            for key in keys:
                self.by_species[key][claim.feature_name].add(claim.value_label)
                self.provenance[(key, claim.feature_name)].add(claim.provenance or "dirr")
            self.n_claims += 1

    def values_for_keys(self, species_keys: Iterable[str], features: Iterable[str]) -> list[str]:
        values: list[str] = []
        for key in species_keys:
            for feature in features:
                values.extend(self.by_species.get(key, {}).get(feature, ()))
        return sorted(set(values))

    def values(self, species_key: str, features: Iterable[str]) -> list[str]:
        return self.values_for_keys((species_key,), features)

    def features_for(self, species_key: str) -> dict[str, list[str]]:
        return {feature: sorted(values) for feature, values in self.by_species.get(species_key, {}).items()}


def _join_one(
    row: Mapping[str, Any],
    index: ClaimsIndex,
    target: TargetSpec,
    *,
    require_anatomy_match: bool,
) -> dict[str, Any]:
    out = dict(row)
    anatomy = str(row.get("anatomy_class") or "")
    anatomy_matches = target.anatomy_allows(anatomy)
    anatomy_ok = anatomy_matches if require_anatomy_match else True
    keys = canonical_binomial_aliases(row.get("taxon_name"))
    raw_values = index.values_for_keys(keys, target.claim_features) if keys else []
    labels = sorted(
        {
            label
            for label in (canonical_label(value, target) for value in raw_values)
            if label
        }
    )
    # Any value the claim source asserts for an in-target feature but that
    # falls outside this target makes the row ambiguous. We do not silently
    # keep the one value that happens to be in-target.
    out_of_target = sorted(
        value for value in raw_values if canonical_label(value, target) is None
    )

    if not anatomy:
        reason = "unknown_anatomy_class"
    elif not anatomy_ok:
        reason = f"anatomy_mismatch:{anatomy}"
    elif not keys:
        reason = "no_species_key"
    elif not raw_values:
        reason = "no_claim"
    elif not labels:
        reason = "claim_outside_target"
    elif out_of_target or (len(labels) > 1 and not target.allow_multi_value):
        reason = "ambiguous_claim"
    else:
        reason = "eligible"

    eligible = reason == "eligible"
    out["weak_label_eligible"] = eligible
    out["weak_label_reason"] = reason
    out["claim_feature"] = target.feature_name
    out["claim_values"] = sorted(raw_values)
    out["claim_labels"] = labels
    out["claim_out_of_target_values"] = out_of_target
    if eligible:
        out["target_label"] = labels[0] if len(labels) == 1 else labels
        out["label_source"] = "dirr"
        provenance: set[str] = set()
        for key in keys:
            provenance.update(index.provenance.get((key, target.feature_name), ()))
        out["label_provenance"] = sorted(provenance)[0] if provenance else "dirr"
    else:
        out.pop("target_label", None)
    out["anatomy_matches_claim"] = anatomy_matches
    return out


def join_weak_labels(
    rows: Sequence[Mapping[str, Any]],
    claims: Iterable[Claim] | ClaimsIndex,
    target: TargetSpec,
    *,
    require_anatomy_match: bool = True,
) -> dict[str, Any]:
    """Attach Dirr weak labels to crops, gated by anatomy.

    Returns ``{"rows": [...], "audit": {...}}``. Ineligible rows are returned
    with ``weak_label_eligible=False`` and a reason; they are never silently
    dropped here (the dataset validator decides whether to exclude them).
    """

    index = claims if isinstance(claims, ClaimsIndex) else ClaimsIndex(claims)
    joined = [
        _join_one(row, index, target, require_anatomy_match=require_anatomy_match)
        for row in rows
    ]
    reasons = Counter(str(row["weak_label_reason"]) for row in joined)
    eligible = [row for row in joined if row.get("weak_label_eligible")]
    label_counts = Counter(str(row["target_label"]) for row in eligible)
    anatomy_gate = Counter(
        "eligible" if row.get("anatomy_matches_claim") else "rejected"
        for row in joined
    )
    return {
        "rows": joined,
        "audit": {
            "claims_total": index.n_claims,
            "claims_skipped_no_species": index.n_skipped,
            "rows_total": len(joined),
            "rows_eligible": len(eligible),
            "rows_ineligible": len(joined) - len(eligible),
            "reasons": dict(sorted(reasons.items())),
            "eligible_by_label": dict(sorted(label_counts.items())),
            "anatomy_gate": dict(anatomy_gate),
            "target": target.target_id,
            "feature": target.feature_name,
            "rule": (
                "A Dirr claim is attached only when the crop's anatomy_class is in "
                f"{list(target.anatomy_classes)} and the species has exactly one "
                "in-target value (unless allow_multi_value)."
            ),
        },
    }


def apply_weak_labels(
    rows: Sequence[Mapping[str, Any]],
    claims: Iterable[Claim] | ClaimsIndex,
    target: TargetSpec,
    *,
    require_anatomy_match: bool = True,
) -> list[dict[str, Any]]:
    return join_weak_labels(
        rows, claims, target, require_anatomy_match=require_anatomy_match
    )["rows"]
