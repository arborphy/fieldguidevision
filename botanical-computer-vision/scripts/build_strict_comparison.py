"""Build the three-way strict BioCLIP/DINOv3 comparison artifacts."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZERO_DIR = ROOT / "outputs" / "bioclip25_strict_zero_shot"
PROBE_DIR = ROOT / "outputs" / "bioclip25_strict_linear_probe"
DINO_DIR = ROOT / "outputs" / "dinov3_strict_linear_probe"
OUT_DIR = ROOT / "outputs" / "strict_baseline_comparison"
PUBLIC_DIR = ROOT / "public" / "downloads"
APP_DATA = ROOT / "app" / "strict-comparison-data.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def paired_bootstrap(
    first: list[bool], second: list[bool], *, seed: int, iterations: int = 20000
) -> list[float]:
    rng = random.Random(seed)
    count = len(first)
    differences = []
    for _ in range(iterations):
        indices = [rng.randrange(count) for _ in range(count)]
        differences.append(
            sum(int(first[index]) - int(second[index]) for index in indices) / count
        )
    differences.sort()
    return [differences[int(iterations * 0.025)], differences[int(iterations * 0.975)]]


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    smaller = min(first_only, second_only)
    if not discordant:
        return 1.0
    lower_tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * lower_tail)


def pair_summary(
    first_rows: list[dict], second_rows: list[dict], *, seed: int
) -> dict:
    counts = Counter()
    first_vector = []
    second_vector = []
    first_top5_vector = []
    second_top5_vector = []
    for first, second in zip(first_rows, second_rows):
        first_correct = as_bool(first["correct"])
        second_correct = as_bool(second["correct"])
        if first_correct and second_correct:
            counts["both_correct"] += 1
        elif first_correct:
            counts["first_only"] += 1
        elif second_correct:
            counts["second_only"] += 1
        else:
            counts["both_wrong"] += 1
        counts["same_prediction"] += first["predicted_species"] == second["predicted_species"]
        first_vector.append(first_correct)
        second_vector.append(second_correct)
        first_top5_vector.append(as_bool(first["correct_top5"]))
        second_top5_vector.append(as_bool(second["correct_top5"]))
    return {
        "both_correct": counts["both_correct"],
        "first_only": counts["first_only"],
        "second_only": counts["second_only"],
        "both_wrong": counts["both_wrong"],
        "same_prediction": counts["same_prediction"],
        "top1_delta": (sum(first_vector) - sum(second_vector)) / len(first_vector),
        "top5_delta": (sum(first_top5_vector) - sum(second_top5_vector)) / len(first_vector),
        "top1_paired_bootstrap_95_ci": paired_bootstrap(
            first_vector, second_vector, seed=seed
        ),
        "top5_paired_bootstrap_95_ci": paired_bootstrap(
            first_top5_vector, second_top5_vector, seed=seed + 1
        ),
        "exact_mcnemar_p": exact_mcnemar_p(
            counts["first_only"], counts["second_only"]
        ),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(summary: dict, *, zero_shot: bool = False) -> dict:
    prefix = "" if zero_shot else "test_"
    return {
        "model": summary["model"],
        "method": summary["method"],
        "top1_correct": summary[f"{prefix}top1_correct"],
        "top1_accuracy": summary[f"{prefix}top1_accuracy"],
        "top5_correct": summary[f"{prefix}top5_correct"],
        "top5_accuracy": summary[f"{prefix}top5_accuracy"],
        "macro_top1_accuracy": summary[
            "macro_top1_accuracy" if zero_shot else "test_macro_top1_accuracy"
        ],
    }


def main() -> None:
    rows = {
        "zero": read_csv(ZERO_DIR / "predictions.csv"),
        "probe": read_csv(PROBE_DIR / "predictions.csv"),
        "dino": read_csv(DINO_DIR / "predictions.csv"),
    }
    summaries = {
        "zero": read_json(ZERO_DIR / "summary.json"),
        "probe": read_json(PROBE_DIR / "summary.json"),
        "dino": read_json(DINO_DIR / "summary.json"),
    }
    by_id = {
        name: {row["image_id"]: row for row in model_rows}
        for name, model_rows in rows.items()
    }
    expected_ids = set(by_id["zero"])
    if len(expected_ids) != 211 or any(set(model) != expected_ids for model in by_id.values()):
        raise RuntimeError("All three models must contain the same 211 unique test image IDs")
    if any(row["status"] != "ok" for model_rows in rows.values() for row in model_rows):
        raise RuntimeError("At least one prediction has non-ok status")

    ordered_ids = sorted(expected_ids)
    ordered = {name: [by_id[name][image_id] for image_id in ordered_ids] for name in rows}
    for image_id in ordered_ids:
        truths = {by_id[name][image_id]["ground_truth_species"] for name in rows}
        individuals = {by_id[name][image_id]["individual_id"] for name in rows}
        if len(truths) != 1 or len(individuals) != 1:
            raise RuntimeError(f"Metadata mismatch for {image_id}")

    species = defaultdict(Counter)
    organs = defaultdict(Counter)
    paired_rows = []
    for image_id in ordered_ids:
        zero = by_id["zero"][image_id]
        probe = by_id["probe"][image_id]
        dino = by_id["dino"][image_id]
        truth = zero["ground_truth_species"]
        organ = zero["organ_category"]
        values = {
            "zero_top1": as_bool(zero["correct"]),
            "zero_top5": as_bool(zero["correct_top5"]),
            "probe_top1": as_bool(probe["correct"]),
            "probe_top5": as_bool(probe["correct_top5"]),
            "dino_top1": as_bool(dino["correct"]),
            "dino_top5": as_bool(dino["correct_top5"]),
        }
        for bucket in (species[truth], organs[organ]):
            bucket["total"] += 1
            bucket.update({name: int(value) for name, value in values.items()})
        paired_rows.append(
            {
                "image_id": image_id,
                "individual_id": zero["individual_id"],
                "organ_category": organ,
                "ground_truth_species": truth,
                "bioclip_zero_shot_prediction": zero["predicted_species"],
                "bioclip_zero_shot_confidence": zero["confidence"],
                "bioclip_zero_shot_correct": values["zero_top1"],
                "bioclip_zero_shot_top5_correct": values["zero_top5"],
                "bioclip_probe_prediction": probe["predicted_species"],
                "bioclip_probe_confidence": probe["confidence"],
                "bioclip_probe_correct": values["probe_top1"],
                "bioclip_probe_top5_correct": values["probe_top5"],
                "dinov3_probe_prediction": dino["predicted_species"],
                "dinov3_probe_confidence": dino["confidence"],
                "dinov3_probe_correct": values["dino_top1"],
                "dinov3_probe_top5_correct": values["dino_top5"],
            }
        )

    def comparison_rows(source: dict, name_key: str) -> list[dict]:
        output = []
        for name, values in sorted(source.items()):
            total = values["total"]
            output.append(
                {
                    name_key: name,
                    "total": total,
                    "bioclip_zero_top1_accuracy": values["zero_top1"] / total,
                    "bioclip_zero_top5_accuracy": values["zero_top5"] / total,
                    "bioclip_probe_top1_accuracy": values["probe_top1"] / total,
                    "bioclip_probe_top5_accuracy": values["probe_top5"] / total,
                    "dinov3_probe_top1_accuracy": values["dino_top1"] / total,
                    "dinov3_probe_top5_accuracy": values["dino_top5"] / total,
                    "top1_delta_probe_minus_zero":
                        (values["probe_top1"] - values["zero_top1"]) / total,
                    "top1_delta_bioclip_probe_minus_dinov3":
                        (values["probe_top1"] - values["dino_top1"]) / total,
                }
            )
        return output

    species_rows = comparison_rows(species, "species")
    organ_rows = comparison_rows(organs, "organ_category")
    summary = {
        "protocol": {
            "test_images": 211,
            "species": 63,
            "seed": 42,
            "test_groups_per_species": 1,
            "individual_disjoint": True,
            "candidate_set_identical": True,
            "train_images": 1181,
            "validation_images": 249,
        },
        "bioclip_zero_shot": metric_summary(summaries["zero"], zero_shot=True),
        "bioclip_linear_probe": {
            **metric_summary(summaries["probe"]),
            "embedding_dimension": summaries["probe"]["embedding_dimension"],
            "best_C": summaries["probe"]["best_C"],
        },
        "dinov3_linear_probe": {
            **metric_summary(summaries["dino"]),
            "embedding_dimension": summaries["dino"]["embedding_dimension"],
            "best_C": summaries["dino"]["best_C"],
        },
        "pairwise": {
            "bioclip_probe_vs_zero_shot": pair_summary(
                ordered["probe"], ordered["zero"], seed=42
            ),
            "bioclip_probe_vs_dinov3": pair_summary(
                ordered["probe"], ordered["dino"], seed=44
            ),
            "zero_shot_vs_dinov3": pair_summary(
                ordered["zero"], ordered["dino"], seed=46
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "paired_predictions.csv", paired_rows, list(paired_rows[0]))
    write_csv(OUT_DIR / "per_species_comparison.csv", species_rows, list(species_rows[0]))
    write_csv(OUT_DIR / "per_organ_comparison.csv", organ_rows, list(organ_rows[0]))
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    APP_DATA.write_text(
        json.dumps(
            {"summary": summary, "organs": organ_rows, "species": species_rows},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for source, destination in (
        (OUT_DIR / "paired_predictions.csv", PUBLIC_DIR / "strict_paired_predictions.csv"),
        (OUT_DIR / "per_species_comparison.csv", PUBLIC_DIR / "strict_per_species_comparison.csv"),
        (OUT_DIR / "per_organ_comparison.csv", PUBLIC_DIR / "strict_per_organ_comparison.csv"),
        (OUT_DIR / "summary.json", PUBLIC_DIR / "strict_comparison_summary.json"),
        (ZERO_DIR / "predictions.csv", PUBLIC_DIR / "bioclip25_strict_predictions.csv"),
        (PROBE_DIR / "predictions.csv", PUBLIC_DIR / "bioclip25_probe_strict_predictions.csv"),
        (DINO_DIR / "predictions.csv", PUBLIC_DIR / "dinov3_strict_predictions.csv"),
    ):
        shutil.copyfile(source, destination)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
