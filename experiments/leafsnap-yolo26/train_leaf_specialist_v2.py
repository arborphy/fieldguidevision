# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "classification-v2"
OUT_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "leaf-specialist-v2"
V1_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "leaf-specialist"
ENGINE_PATH = HERE / "train_leaf_specialist.py"


def load_engine():
    spec = importlib.util.spec_from_file_location("leaf_specialist_engine_v2", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load training engine: {ENGINE_PATH}")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    engine.INPUT_ROOT = DATA_ROOT
    engine.MANIFEST = DATA_ROOT / "manifest.jsonl"
    engine.COUNTS = DATA_ROOT / "counts.json"
    engine.LABELS = DATA_ROOT / "labels.json"
    engine.OUT_ROOT = OUT_ROOT
    original_preflight = engine.preflight

    def preflight_with_revision(rows: list[dict[str, Any]], labels: dict[str, Any], counts: dict[str, Any]):
        result = original_preflight(rows, labels, counts)
        result["dataset_version"] = counts["dataset_version"]
        result["revision"] = counts["revision"]
        result["mapping_audit"] = counts["mapping_audit"]
        result["counts"] = {
            "total_crops": counts["total_crops"],
            "by_split_class": counts["by_split_class"],
            "by_split_class_domain": counts["by_split_class_domain"],
            "by_split_domain": counts["by_split_domain"],
            "split_species_counts": counts["split_species_counts"],
        }
        return result

    engine.preflight = preflight_with_revision
    original_resolve_pretrained = engine.resolve_pretrained
    v1_pretrained = V1_ROOT / "pretrained" / "yolo26n-cls.pt"
    v1_training = read_json(V1_ROOT / "training_params.json")

    def resolve_pretrained(dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / "yolo26n-cls.pt"
        if not target.exists():
            if not v1_pretrained.is_file():
                raise FileNotFoundError(f"v1 pretrained checkpoint is missing: {v1_pretrained}")
            shutil.copy2(v1_pretrained, target)
        resolved = original_resolve_pretrained(dest_dir)
        expected_sha = v1_training["pretrained_sha256"]
        actual_sha = engine.sha256_file(resolved)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"v2 pretrained weights differ from v1: expected {expected_sha}, got {actual_sha}"
            )
        return resolved

    engine.resolve_pretrained = resolve_pretrained
    return engine


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        groups[row["species"]].append(row)
    for species, rows in groups.items():
        first = rows[0]
        n = len(rows)
        correct = sum(row["actual"] == row["predicted"] for row in rows)
        field_rows = [row for row in rows if row["domain"] == "field"]
        lab_rows = [row for row in rows if row["domain"] == "lab"]
        field_correct = sum(row["actual"] == row["predicted"] for row in field_rows)
        lab_correct = sum(row["actual"] == row["predicted"] for row in lab_rows)
        result[species] = {
            "species": species,
            "class_name": first["actual"],
            "n": n,
            "correct": correct,
            "errors": n - correct,
            "accuracy": correct / n if n else None,
            "field": {
                "n": len(field_rows),
                "correct": field_correct,
                "errors": len(field_rows) - field_correct,
                "accuracy": field_correct / len(field_rows) if field_rows else None,
            },
            "lab": {
                "n": len(lab_rows),
                "correct": lab_correct,
                "errors": len(lab_rows) - lab_correct,
                "accuracy": lab_correct / len(lab_rows) if lab_rows else None,
            },
        }
    return result


def metric_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics["evaluation"]["test"]
    field = metrics["evaluation"]["field"]
    checks = {row["name"]: row for row in metrics["e4"]["checks"]}
    data = metrics["data"]
    return {
        "test_top1": test["accuracy"],
        "test_macro_precision": test["macro"]["precision"],
        "test_macro_recall": test["macro"]["recall"],
        "test_macro_f1": test["macro"]["f1"],
        "test_count": test["n"],
        "test_class_counts": data["by_split_class"]["test"],
        "test_field_count": field["n"],
        "simple_leaf_recall": checks["simple_leaf_recall"]["value"],
        "compound_leaf_recall": checks["compound_leaf_recall"]["value"],
        "compound_leaf_field_recall": checks["compound_leaf_field_recall"]["value"],
        "e4_passed": metrics["e4"]["passed"],
        "training_minutes": metrics["training"].get("wall_clock_minutes"),
        "epochs_run": metrics["training"].get("epochs_run"),
        "test_species": data["species_by_split"]["test"],
        "train_compound_domain_counts": data["by_split_class_domain"]["train"]["compound_leaf"],
    }


def make_comparison(v1_metrics: dict[str, Any], v2_metrics: dict[str, Any]) -> dict[str, Any]:
    v1_species = aggregate_predictions(V1_ROOT / "predictions_test.jsonl")
    v2_species = aggregate_predictions(OUT_ROOT / "predictions_test.jsonl")
    species_union = sorted(set(v1_species) | set(v2_species))
    rows = []
    for species in species_union:
        old = v1_species.get(species)
        new = v2_species.get(species)
        row = {
            "species": species,
            "class_name": (new or old)["class_name"],
            "v1_test": old,
            "v2_test": new,
            "in_both_test_splits": old is not None and new is not None,
        }
        if old is not None and new is not None:
            row["accuracy_delta_v2_minus_v1"] = new["accuracy"] - old["accuracy"]
            row["field_accuracy_delta_v2_minus_v1"] = (
                new["field"]["accuracy"] - old["field"]["accuracy"]
                if new["field"]["accuracy"] is not None and old["field"]["accuracy"] is not None
                else None
            )
            row["error_delta_v2_minus_v1"] = new["errors"] - old["errors"]
            row["field_error_delta_v2_minus_v1"] = new["field"]["errors"] - old["field"]["errors"]
        rows.append(row)

    v1 = metric_snapshot(v1_metrics)
    v2 = metric_snapshot(v2_metrics)
    intersection = sorted(set(v1["test_species"]) & set(v2["test_species"]))
    juglans = next((row for row in rows if row["species"] == "Juglans cinerea"), None)
    juglans_improved = None
    if juglans and juglans["v1_test"] and juglans["v2_test"]:
        old = juglans["v1_test"]
        new = juglans["v2_test"]
        juglans_improved = {
            "overall_accuracy_improved": new["accuracy"] > old["accuracy"],
            "field_accuracy_improved": (
                new["field"]["accuracy"] is not None
                and old["field"]["accuracy"] is not None
                and new["field"]["accuracy"] > old["field"]["accuracy"]
            ),
            "overall_errors_reduced": new["errors"] < old["errors"],
            "field_errors_reduced": new["field"]["errors"] < old["field"]["errors"],
            "v1": old,
            "v2": new,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "v1_metrics": str(V1_ROOT / "metrics.json"),
        "v2_metrics": str(OUT_ROOT / "metrics.json"),
        "comparison_note": "V1 and V2 use species-disjoint test splits and the same deterministic split algorithm, but the expanded 122-species mapping changes the split cardinality and can move some species across test boundaries. Overall aggregate deltas are descriptive; paired per-species rows are directly comparable only where both versions include the species in test. No weak-performing species were removed.",
        "v1": v1,
        "v2": v2,
        "deltas_v2_minus_v1": {
            key: (v2[key] - v1[key]) if isinstance(v1[key], (int, float)) and isinstance(v2[key], (int, float)) else None
            for key in (
                "test_top1",
                "test_macro_f1",
                "simple_leaf_recall",
                "compound_leaf_recall",
                "compound_leaf_field_recall",
            )
        },
        "test_species_overlap": {
            "v1_count": len(v1["test_species"]),
            "v2_count": len(v2["test_species"]),
            "intersection_count": len(intersection),
            "intersection": intersection,
            "v1_only": sorted(set(v1["test_species"]) - set(v2["test_species"])),
            "v2_only": sorted(set(v2["test_species"]) - set(v1["test_species"])),
        },
        "juglans_cinerea": juglans_improved,
        "per_species_test_errors": rows,
    }


def percent(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.2f}%"


def comparison_markdown(comparison: dict[str, Any]) -> str:
    v1 = comparison["v1"]
    v2 = comparison["v2"]
    delta = comparison["deltas_v2_minus_v1"]
    overlap = comparison["test_species_overlap"]
    juglans = comparison["juglans_cinerea"]
    lines = [
        "# V1 vs V2 leaf specialist comparison",
        "",
        comparison["comparison_note"],
        "",
        "## Aggregate held-out metrics",
        "",
        "| metric | v1 | v2 | delta (v2-v1) |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("test_top1", "test top1"),
        ("simple_leaf_recall", "simple recall"),
        ("compound_leaf_recall", "compound recall"),
        ("compound_leaf_field_recall", "compound field recall"),
        ("test_macro_f1", "test macro F1"),
    ):
        lines.append(f"| {label} | {percent(v1[key])} | {percent(v2[key])} | {percent(delta[key])} |")
    lines += [
        "",
        f"Test support: v1 n={v1['test_count']} across {len(v1['test_species'])} species; "
        f"v2 n={v2['test_count']} across {len(v2['test_species'])} species; "
        f"species intersection {overlap['intersection_count']}.",
        f"Compound training domain counts: v1 {json.dumps(v1['train_compound_domain_counts'], sort_keys=True)}; "
        f"v2 {json.dumps(v2['train_compound_domain_counts'], sort_keys=True)}.",
        "",
        "## Juglans cinerea",
        "",
    ]
    if juglans is None:
        lines.append("Juglans cinerea was not present in both test splits; no paired improvement claim is made.")
    else:
        lines += [
            f"- V1: {percent(juglans['v1']['accuracy'])} overall ({juglans['v1']['errors']}/{juglans['v1']['n']} errors); "
            f"field {percent(juglans['v1']['field']['accuracy'])} "
            f"({juglans['v1']['field']['errors']}/{juglans['v1']['field']['n']} field errors).",
            f"- V2: {percent(juglans['v2']['accuracy'])} overall ({juglans['v2']['errors']}/{juglans['v2']['n']} errors); "
            f"field {percent(juglans['v2']['field']['accuracy'])} "
            f"({juglans['v2']['field']['errors']}/{juglans['v2']['field']['n']} field errors).",
            f"- Improved: overall accuracy {juglans['overall_accuracy_improved']}; "
            f"field accuracy {juglans['field_accuracy_improved']}; "
            f"overall errors reduced {juglans['overall_errors_reduced']}; "
            f"field errors reduced {juglans['field_errors_reduced']}.",
            "- The species remains in both test splits; no species was removed based on its error rate.",
        ]
    lines += [
        "",
        "## Per-species test errors",
        "",
        "Errors are wrong test crops; field/lab error counts are shown separately. `—` means the species was not in that version's test split.",
        "",
        "| species | class | v1 overall errors / n | v1 field errors / n | v2 overall errors / n | v2 field errors / n | paired accuracy Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["per_species_test_errors"]:
        old = row["v1_test"]
        new = row["v2_test"]
        old_all = f"{old['errors']}/{old['n']}" if old else "—"
        old_field = f"{old['field']['errors']}/{old['field']['n']}" if old else "—"
        new_all = f"{new['errors']}/{new['n']}" if new else "—"
        new_field = f"{new['field']['errors']}/{new['field']['n']}" if new else "—"
        delta_text = percent(row.get("accuracy_delta_v2_minus_v1"))
        lines.append(
            f"| {row['species']} | {row['class_name']} | {old_all} | {old_field} | "
            f"{new_all} | {new_field} | {delta_text} |"
        )
    lines += ["", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    if not (DATA_ROOT / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"build the v2 dataset first: {DATA_ROOT}")
    if not (V1_ROOT / "metrics.json").is_file() or not (V1_ROOT / "predictions_test.jsonl").is_file():
        raise FileNotFoundError(f"v1 comparison artifacts are missing under {V1_ROOT}")

    engine = load_engine()
    train_args = [
        "--epochs",
        "30",
        "--patience",
        "8",
        "--batch",
        "16",
        "--imgsz",
        "224",
        "--seed",
        "26",
    ]
    if args.skip_train:
        train_args.append("--skip-train")
    if args.skip_export:
        train_args.append("--skip-export")
    if args.cpu:
        train_args.append("--cpu")
    engine.main(train_args)

    metrics_path = OUT_ROOT / "metrics.json"
    metrics = read_json(metrics_path)
    v1_metrics = read_json(V1_ROOT / "metrics.json")
    metrics["experiment"] = "E4 v2 — LeafSnap leaf-type specialist"
    metrics["source"] = str(HERE / "train_leaf_specialist_v2.py")
    metrics["revision"] = {
        "dataset": str(DATA_ROOT),
        "data_report": str(HERE / "LEAF_SPECIALIST_V2_DATA.md"),
        "training_policy": "single v2 proof run, unchanged 30 epoch / patience 8 / seed 26 / batch 16 / imgsz 224 budget",
        "sample_set_change": "use all 122 EDA-audited exact Simple/Compound species; train compound on lab crops only; retain field compound crops in validation and test",
        "e4_thresholds_unchanged": True,
    }
    metrics["caveats"].extend(
        [
            "V2 expands the audited EDA mapping to all unambiguous species whose leaf_type values are exactly Simple or Compound: 122 species (101 simple, 21 compound). Ginkgo biloba and Gymnocladus dioicus remain excluded as mixed; the 20 EDA-unmatched names are not force-matched.",
            "V2 compound training uses LAB crops only, favoring pressed whole-architecture views. FIELD compound crops remain in validation and test; they may still isolate one leaflet and inherit the species-level compound label, so FIELD recall includes a view-validity noise floor.",
            "No species was excluded or downweighted because of validation/test performance. The per-species comparison includes all v1/v2 test species and marks species not in a version's test split as not comparable.",
        ]
    )
    comparison = make_comparison(v1_metrics, metrics)
    metrics["v1_vs_v2"] = comparison
    engine.write_json(metrics_path, metrics)
    engine.write_json(OUT_ROOT / "v1_v2_comparison.json", comparison)
    comparison_path = OUT_ROOT / "V1_V2_COMPARISON.md"
    comparison_path.write_text(comparison_markdown(comparison), encoding="utf-8")
    engine.write_report(metrics, OUT_ROOT)
    with (OUT_ROOT / "LEAF_SPECIALIST.md").open("a", encoding="utf-8") as report:
        report.write("\n\n" + comparison_markdown(comparison))

    checks = metrics["e4"]["checks"]
    result = {
        "e4_passed": metrics["e4"]["passed"],
        "checks": [
            {
                "name": row["name"],
                "value": row["value"],
                "threshold": row["threshold"],
                "pass": row["pass"],
            }
            for row in checks
        ],
        "v2_counts": metrics["data"]["by_split_class"],
        "test_top1": metrics["evaluation"]["test"]["accuracy"],
        "simple_recall": metrics["evaluation"]["test"]["per_class"]["simple_leaf"]["recall"],
        "compound_recall": metrics["evaluation"]["test"]["per_class"]["compound_leaf"]["recall"],
        "compound_field_recall": metrics["evaluation"]["field"]["per_class"]["compound_leaf"]["recall"],
        "juglans_comparison": comparison["juglans_cinerea"],
        "training_minutes": metrics["training"].get("wall_clock_minutes"),
        "model": str(OUT_ROOT / "leaf_specialist.onnx"),
        "checkpoint": str(OUT_ROOT / "best.pt"),
        "report": str(OUT_ROOT / "LEAF_SPECIALIST.md"),
        "comparison_report": str(comparison_path),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
