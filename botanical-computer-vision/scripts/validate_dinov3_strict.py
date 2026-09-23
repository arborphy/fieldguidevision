"""Validate strict DINOv3 benchmark outputs and individual-level isolation."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


output = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/dinov3_strict_linear_probe")


def read_csv(name: str) -> list[dict[str, str]]:
    with (output / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
manifest = read_csv("split_manifest.csv")
predictions = read_csv("predictions.csv")
per_species = read_csv("per_species.csv")

assert len(manifest) == 1641
assert len({row["image_id"] for row in manifest}) == 1641
assert len({row["ground_truth_species"] for row in manifest}) == 63
assert set(row["split"] for row in manifest) == {"train", "validation", "test"}

groups: dict[str, dict[str, set[str]]] = defaultdict(
    lambda: {"train": set(), "validation": set(), "test": set()}
)
for row in manifest:
    groups[row["ground_truth_species"]][row["split"]].add(row["individual_id"])

for species, split_groups in groups.items():
    assert all(split_groups.values()), f"empty split for {species}"
    assert not (split_groups["train"] & split_groups["validation"]), species
    assert not (split_groups["train"] & split_groups["test"]), species
    assert not (split_groups["validation"] & split_groups["test"]), species

test_ids = {row["image_id"] for row in manifest if row["split"] == "test"}
prediction_ids = {row["image_id"] for row in predictions}
assert prediction_ids == test_ids
assert len(predictions) == int(summary["test_images"])
assert all(row["status"] == "ok" for row in predictions)
assert all(row["split"] == "test" for row in predictions)

top1 = sum(row["correct"].lower() == "true" for row in predictions)
top5 = sum(row["correct_top5"].lower() == "true" for row in predictions)
assert top1 == summary["test_top1_correct"]
assert top5 == summary["test_top5_correct"]
assert abs(top1 / len(predictions) - summary["test_top1_accuracy"]) < 1e-12
assert abs(top5 / len(predictions) - summary["test_top5_accuracy"]) < 1e-12
assert len(per_species) == 63
assert sum(int(row["total"]) for row in per_species) == len(predictions)

print(
    json.dumps(
        {
            "validated": True,
            "images": len(manifest),
            "species": len(groups),
            "test_images": len(predictions),
            "top1_accuracy": summary["test_top1_accuracy"],
            "top5_accuracy": summary["test_top5_accuracy"],
        },
        indent=2,
    )
)
