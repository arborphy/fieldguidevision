# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Deterministic, isolated split assignment.

Two splitter families are provided and both are observation-disjoint by
construction:

``observation``
    Every crop of an observation lands in the same split. Species may span
    splits (a species with several observations can appear in train and test).
    Use this when the scientific question is generalization to new
    observations/individuals of known species.

``species``
    Every crop of a species lands in the same split. This is strict
    species-disjoint and therefore also observation-disjoint. Use this for the
    headline "can the expert generalize to an unseen species?" number.

Optionally, ``species_then_observation`` is the same as ``species`` (kept as
an explicit alias for callers that want intent-documenting config).

All decisions are SHA-256 based on stable ids, so reruns on the same manifest
produce identical assignments. Missing taxon names are grouped per observation
(a sentinel species key) and reported — they can never leak a shared unknown
bucket across splits.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import SPLITS, canonical_binomial

SPLIT_MODES: tuple[str, ...] = ("observation", "species", "species_then_observation")


def stable_unit_interval(key: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def observation_split(observation_id: str, seed: str, fractions: Mapping[str, float] | None = None) -> str:
    """Deterministic observation-level split for one id."""

    frac = _validate_fractions(fractions)
    value = stable_unit_interval(str(observation_id), seed)
    if value < frac["train"]:
        return "train"
    if value < frac["train"] + frac["val"]:
        return "val"
    return "test"


def _validate_fractions(fractions: Mapping[str, float] | None) -> dict[str, float]:
    frac = dict(fractions or {"train": 0.70, "val": 0.15, "test": 0.15})
    missing = [name for name in SPLITS if name not in frac]
    if missing:
        raise ValueError(f"split fractions missing {missing}")
    total = sum(float(frac[name]) for name in SPLITS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    return {name: float(frac[name]) for name in SPLITS}


def species_key(row: Mapping[str, Any]) -> str:
    """Stable species grouping key.

    Unknown/missing species are keyed to their observation so that every
    unnamed observation is its own group (never a shared global bucket).
    """

    binomial = canonical_binomial(row.get("taxon_name"))
    if binomial:
        return f"sp:{binomial}"
    observation_id = row.get("observation_id")
    if observation_id in (None, ""):
        raise ValueError("row without taxon_name must carry observation_id")
    return f"unknown:{observation_id}"


def assign_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str = "species",
    seed: str = "anatomy4-specialists-v1",
    fractions: Mapping[str, float] | None = None,
    required_classes: Sequence[str] | None = None,
    required_min_per_class: int = 0,
    max_seed_attempts: int = 200,
    prefer_test_unseen: bool = False,
) -> dict[str, Any]:
    """Assign a split to every row.

    Returns a report with the per-row assignments (in input order), the group
    map, and counts. The algorithm allocates whole groups (species or
    observation) greedily in hash order toward the requested *row* fractions:
    a group goes to the split furthest below its target at that moment. This
    keeps class balance reasonably stable while guaranteeing isolation.

    When ``required_classes`` is given (the trainer passes the target's
    classes), seeds are searched deterministically until train and val each
    contain every class. This prevents a tiny flower class from making a
    valid dataset randomly abort on an unlucky hash. If no seed succeeds, the
    report records ``seed_search.satisfied=False`` and callers get a clear
    validation failure rather than a silent empty split.
    """

    if mode not in SPLIT_MODES:
        raise ValueError(f"unknown split mode {mode!r}; expected {SPLIT_MODES}")
    frac = _validate_fractions(fractions)
    grouping = "observation" if mode == "observation" else "species"
    required = tuple(dict.fromkeys(required_classes or ()))
    attempts = max(1, int(max_seed_attempts)) if required else 1
    last: dict[str, Any] | None = None
    for attempt in range(attempts):
        attempt_seed = seed if attempt == 0 else f"{seed}:seed{attempt}"
        report = _assign_once(
            rows,
            mode=mode,
            grouping=grouping,
            seed=attempt_seed,
            frac=frac,
            prefer_test_unseen=prefer_test_unseen,
        )
        last = report
        if not required:
            report["seed_search"] = {
                "attempts": 1,
                "chosen_seed": attempt_seed,
                "required_classes": [],
                "satisfied": True,
            }
            return report
        missing = _missing_required(
            report["class_counts"], required, minimum=required_min_per_class
        )
        if not missing["train"] and not missing["val"]:
            report["seed_search"] = {
                "attempts": attempt + 1,
                "chosen_seed": attempt_seed,
                "required_classes": list(required),
                "required_min_per_class": required_min_per_class,
                "satisfied": True,
                "missing": missing,
            }
            return report
    assert last is not None
    last["seed_search"] = {
        "attempts": attempts,
        "chosen_seed": str(last.get("seed")),
        "required_classes": list(required),
        "required_min_per_class": required_min_per_class,
        "satisfied": False,
        "missing": _missing_required(
            last["class_counts"], required, minimum=required_min_per_class
        ),
    }
    return last


def _missing_required(
    class_counts: Mapping[str, Mapping[str, int]],
    required: Sequence[str],
    *,
    minimum: int = 0,
) -> dict[str, list[str]]:
    threshold = max(1, int(minimum))
    return {
        split: [
            name
            for name in required
            if class_counts.get(split, {}).get(name, 0) < threshold
        ]
        for split in SPLITS
    }


def _assign_once(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    grouping: str,
    seed: str,
    frac: Mapping[str, float],
    prefer_test_unseen: bool = False,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = str(row["observation_id"]) if grouping == "observation" else species_key(row)
        groups[key].append(index)

    # Deterministic group order, then greedy deficit allocation.
    ordered = sorted(groups, key=lambda key: stable_unit_interval(key, f"{seed}:{grouping}"))
    total_rows = max(1, len(rows))
    targets = {name: frac[name] * total_rows for name in SPLITS}
    allocated = {name: 0 for name in SPLITS}
    group_split: dict[str, str] = {}
    assigned: set[str] = set()

    # Optional Stage-1-unseen test preference: fill test from groups whose
    # rows were never in Stage-1 train/val first (whole groups, so isolation is
    # preserved). Remaining groups use the normal deficit allocation and can
    # still fill test if there are not enough unseen groups.
    if prefer_test_unseen:
        unseen = {
            key: all(rows[index].get("in_stage1_train_val") is False for index in indices)
            for key, indices in groups.items()
        }
        cap = targets["test"] * 1.25 + 1.0
        for key in ordered:
            size = len(groups[key])
            if not unseen.get(key):
                continue
            if allocated["test"] >= targets["test"] or allocated["test"] + size > cap:
                continue
            group_split[key] = "test"
            allocated["test"] += size
            assigned.add(key)

    for key in ordered:
        if key in assigned:
            continue
        size = len(groups[key])

        def deficit(name: str) -> float:
            return (targets[name] - allocated[name]) / max(1.0, targets[name])

        chosen = max(SPLITS, key=lambda name: (deficit(name), -SPLITS.index(name)))
        group_split[key] = chosen
        allocated[chosen] += size

    assignments = [""] * len(rows)
    observations: dict[str, set[str]] = defaultdict(set)
    species_per_split: dict[str, set[str]] = defaultdict(set)
    for key, indices in groups.items():
        split = group_split[key]
        for index in indices:
            assignments[index] = split
            row = rows[index]
            observations[split].add(str(row["observation_id"]))
            species_per_split[split].add(species_key(row))

    report = {
        "mode": mode,
        "grouping": grouping,
        "seed": seed,
        "fractions": dict(frac),
        "assignments": assignments,
        "group_split": group_split,
        "counts": {
            split: {
                "rows": assignments.count(split),
                "groups": sum(1 for value in group_split.values() if value == split),
                "species": len(species_per_split[split])
                if grouping == "species"
                else len({species_key(rows[i]) for i in range(len(rows)) if assignments[i] == split}),
                "observations": len(observations[split]),
            }
            for split in SPLITS
        },
        "class_counts": _class_counts(rows, assignments),
        "prefer_test_unseen": prefer_test_unseen,
        "stage1_overlap_counts": _stage1_overlap_counts(rows, assignments),
    }
    report["isolation"] = verify_isolation(
        rows, assignments, require_species_disjoint=(grouping == "species")
    )
    return report


def _stage1_overlap_counts(
    rows: Sequence[Mapping[str, Any]], assignments: Sequence[str]
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for row, split in zip(rows, assignments):
        value = row.get("in_stage1_train_val")
        key = "unknown" if value is None else ("in_train_val" if value else "unseen")
        counts[split][key] += 1
    return {split: dict(sorted(counter.items())) for split, counter in counts.items()}


def _class_counts(rows: Sequence[Mapping[str, Any]], assignments: Sequence[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for row, split in zip(rows, assignments):
        label = row.get("target_label")
        if label is not None:
            counts[split][str(label)] += 1
    return {split: dict(sorted(counter.items())) for split, counter in counts.items()}


def verify_isolation(
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[str],
    *,
    require_species_disjoint: bool,
) -> dict[str, Any]:
    """Hard checks: observation isolation always; species isolation if asked."""

    if len(rows) != len(assignments):
        raise ValueError("rows and assignments length mismatch")
    obs_splits: dict[str, set[str]] = defaultdict(set)
    species_splits: dict[str, set[str]] = defaultdict(set)
    for row, split in zip(rows, assignments):
        obs_splits[str(row["observation_id"])].add(split)
        species_splits[species_key(row)].add(split)
    leaking_observations = {key: sorted(value) for key, value in obs_splits.items() if len(value) > 1}
    leaking_species = {key: sorted(value) for key, value in species_splits.items() if len(value) > 1}
    ok = not leaking_observations and (not require_species_disjoint or not leaking_species)
    return {
        "ok": ok,
        "observation_disjoint": not leaking_observations,
        "species_disjoint": not leaking_species,
        "require_species_disjoint": require_species_disjoint,
        "leaking_observations": dict(sorted(leaking_observations.items())[:20]),
        "leaking_species": dict(sorted(leaking_species.items())[:20]),
        "n_observations": len(obs_splits),
        "n_species": len(species_splits),
    }


def apply_assignments(rows: Sequence[Mapping[str, Any]], assignments: Sequence[str]) -> list[dict[str, Any]]:
    """Return copies with ``observation_split`` set (and ``split`` mirrored)."""

    out: list[dict[str, Any]] = []
    for row, split in zip(rows, assignments):
        updated = dict(row)
        updated["observation_split"] = split
        updated["split"] = split
        out.append(updated)
    return out


@dataclass(frozen=True)
class SplitSummary:
    mode: str
    counts: dict[str, dict[str, int]]
    isolation: dict[str, Any]

    def assert_clean(self) -> None:
        if not self.isolation.get("ok"):
            raise ValueError(f"split isolation failed: {self.isolation}")
