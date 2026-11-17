# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

from collections import Counter

import pytest

from experiments.anatomy4_specialists.splits import (
    assign_splits,
    observation_split,
    species_key,
    verify_isolation,
)


def _rows(n_species: int = 4, obs_per_species: int = 3, crops_per_obs: int = 2):
    rows = []
    for s in range(n_species):
        for o in range(obs_per_species):
            for c in range(crops_per_obs):
                rows.append(
                    {
                        "crop_id": f"crop-{s}-{o}-{c}",
                        "observation_id": f"obs-{s}-{o}",
                        "taxon_name": f"Genus{s} species{s}",
                        "anatomy_class": "leaf",
                        "target_label": "Simple" if c == 0 else "Compound",
                    }
                )
    return rows


def test_species_mode_is_species_and_observation_disjoint() -> None:
    rows = _rows()
    report = assign_splits(rows, mode="species", seed="unit")
    assignments = report["assignments"]
    assert report["isolation"]["ok"]
    assert report["isolation"]["observation_disjoint"]
    assert report["isolation"]["species_disjoint"]

    by_species: dict[str, set[str]] = {}
    by_obs: dict[str, set[str]] = {}
    for row, split in zip(rows, assignments):
        by_species.setdefault(species_key(row), set()).add(split)
        by_obs.setdefault(row["observation_id"], set()).add(split)
    assert all(len(v) == 1 for v in by_species.values())
    assert all(len(v) == 1 for v in by_obs.values())


def test_observation_mode_is_observation_disjoint() -> None:
    rows = _rows(n_species=3, obs_per_species=8, crops_per_obs=1)
    report = assign_splits(rows, mode="observation", seed="unit")
    assert report["isolation"]["ok"]
    assert report["isolation"]["observation_disjoint"]
    assert report["grouping"] == "observation"
    by_obs: dict[str, set[str]] = {}
    for row, split in zip(rows, report["assignments"]):
        by_obs.setdefault(row["observation_id"], set()).add(split)
    assert all(len(v) == 1 for v in by_obs.values())


def test_assignments_are_deterministic() -> None:
    rows = _rows()
    first = assign_splits(rows, mode="species", seed="stable")
    second = assign_splits(rows, mode="species", seed="stable")
    assert first["assignments"] == second["assignments"]
    assert first["group_split"] == second["group_split"]


def test_unknown_taxa_are_grouped_per_observation() -> None:
    rows = [
        {"observation_id": "o1", "crop_id": "c1", "taxon_name": None},
        {"observation_id": "o1", "crop_id": "c2", "taxon_name": ""},
        {"observation_id": "o2", "crop_id": "c3", "taxon_name": "Unknown sp."},
    ]
    assert species_key(rows[0]) == "unknown:o1"
    assert species_key(rows[2]) == "unknown:o2"
    report = assign_splits(rows, mode="species", seed="x")
    assert report["isolation"]["ok"]


def test_verify_isolation_detects_injected_leak() -> None:
    rows = _rows(n_species=1, obs_per_species=2, crops_per_obs=1)
    assignments = ["train", "val"]
    report = verify_isolation(rows, assignments, require_species_disjoint=True)
    assert not report["ok"]
    # Each observation stays whole, but the single species spans train/val.
    assert report["observation_disjoint"]
    assert not report["species_disjoint"]


def test_fraction_validation() -> None:
    with pytest.raises(ValueError):
        assign_splits(_rows(), mode="species", fractions={"train": 0.9, "val": 0.2, "test": 0.1})
    with pytest.raises(ValueError):
        assign_splits(_rows(), mode="nope")


def test_observation_split_is_stable_and_covers_all_splits() -> None:
    values = {observation_split(f"obs-{i}", "seed") for i in range(200)}
    assert values == {"train", "val", "test"}
    assert all(observation_split("obs-1", "seed") == observation_split("obs-1", "seed") for _ in range(5))


def test_class_counts_are_reported_per_split() -> None:
    rows = _rows()
    report = assign_splits(rows, mode="species", seed="unit")
    total = Counter()
    for counts in report["class_counts"].values():
        total.update(counts)
    assert total["Simple"] == 12
    assert total["Compound"] == 12
