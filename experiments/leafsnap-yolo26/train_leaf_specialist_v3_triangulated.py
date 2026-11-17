# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""E4 V3 — triangulated (view-valid) leaf-type specialist.

Trains ``yolo26n-cls.pt`` on ``classification-v3-triangulated`` (only
``gold_specialist_train`` crops: the whole foliar unit is visible), evaluates
the best checkpoint on

  (a) the held-out triangulated ``test`` split against the pre-registered E4
      criteria (test top1, simple recall, compound recall, compound field
      recall) plus per-species recall and a validation-selected confidence /
      abstention operating point,
  (b) the held-out ``hard_negatives_sub_part/test`` isolated-leaflet crops
      (view-not-assessable) to measure abstention vs. the dangerous
      ``compound_leaf`` claim, and
  (c) the Henrico x Tuckaway regional "gold regional pool" Stage-2 leaf
      photos, to test Mid-Atlantic field generalization.

Everything is written under ``scratch/leafsnap-yolo26/leaf-specialist-v3-triangulated``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "classification-v3-triangulated"
OUT_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "leaf-specialist-v3-triangulated"
V1_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "leaf-specialist"
V2_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "leaf-specialist-v2"
POOL_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "mid-atlantic-pool"
ENGINE_PATH = HERE / "train_leaf_specialist.py"
LOG_PATH = HERE / "EXPERIMENT_LOG.md"

HN_MANIFEST = DATA_ROOT / "hard_negatives_sub_part" / "manifest.jsonl"
POOL_MANIFEST = POOL_ROOT / "pool_manifest.jsonl"
TRIANGULATION_MANIFEST = DATA_ROOT / "triangulation.jsonl"

DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 6
DEFAULT_BATCH = 32
DEFAULT_IMGSZ = 224
DEFAULT_SEED = 26

GOLD_BUCKET = "gold_regional_pool"
GOLD_FALLBACK_BUCKETS = ("gold_specialist_train",)
GOLD_FALLBACK_STAGE = "stage_2_anatomy"


# ---------------------------------------------------------------------------
# small IO helpers (the engine's helpers assume the v1 globals; keep these local)
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def domain_of(row: dict[str, Any]) -> str:
    """Recover the field/lab domain from the LeafSnap source path.

    The v3 builder keeps ``domains`` in ``triangulation.jsonl`` but not in the
    training manifest (it is not a class folder), so derive it from the source
    image path and cross-check against ``counts.json`` during preflight.
    """
    source_image = str(row.get("source_image") or "")
    if "/field/" in source_image:
        return "field"
    if "/lab/" in source_image:
        return "lab"
    return "unknown"


# ---------------------------------------------------------------------------
# engine wiring
# ---------------------------------------------------------------------------


def load_engine():
    spec = importlib.util.spec_from_file_location("leaf_specialist_engine_v3", ENGINE_PATH)
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

    v1_labels = read_json(ROOT / "scratch" / "leafsnap-yolo26" / "classification" / "labels.json")
    v1_training = read_json(V1_ROOT / "training_params.json")

    original_dataset_summary = engine.dataset_summary

    def dataset_summary_v3(rows: list[dict[str, Any]], labels: dict[str, Any], counts: dict[str, Any]):
        summary = original_dataset_summary(rows, labels, counts)
        # v3 labels.json dropped the crop/task provenance keys; the v3 crops are
        # the same v1 mask-box crops, so carry the v1 rule forward verbatim.
        if not summary.get("crop"):
            summary["crop"] = v1_labels.get("crop")
        if not summary.get("task"):
            summary["task"] = v1_labels.get("task", "image_classification")
        return summary

    engine.dataset_summary = dataset_summary_v3

    original_preflight = engine.preflight

    def preflight_v3(rows: list[dict[str, Any]], labels: dict[str, Any], counts: dict[str, Any]):
        # the v3 training manifest does not carry the field/lab domain (it is not
        # a class folder); stamp it from the LeafSnap source path in place so the
        # engine's later val/test row selection keeps it.
        for row in rows:
            if not row.get("domain"):
                row["domain"] = domain_of(row)
        result = original_preflight(rows, labels, counts)
        result["dataset_version"] = counts.get("dataset_version", DATA_ROOT.name)
        result["triangulation_filter"] = {
            "manifest_bucket": sorted({r.get("triangulation_bucket") for r in rows}),
            "manifest_view_framing": sorted({r.get("view_framing") for r in rows}),
            "expected_note": (
                "v3 materialises only view-valid gold_specialist_train crops "
                "(whole foliar unit framed); isolated-leaflet crops are held out "
                "in hard_negatives_sub_part/"
            ),
        }
        hn = read_jsonl(HN_MANIFEST) if HN_MANIFEST.is_file() else []
        overlap = sorted({r["uid"] for r in rows} & {r["uid"] for r in hn})
        result["hard_negative_audit"] = {
            "hard_negative_crops": len(hn),
            "hard_negative_by_split": dict(sorted(Counter(r["split"] for r in hn).items())),
            "manifest_hard_negative_uid_overlap": overlap,
        }
        if overlap:
            raise RuntimeError(f"view hard negatives leaked into the training manifest: {overlap[:5]}")
        return result

    engine.preflight = preflight_v3

    original_resolve_pretrained = engine.resolve_pretrained
    v1_pretrained = V1_ROOT / "pretrained" / "yolo26n-cls.pt"

    def resolve_pretrained(dest_dir: Path) -> Path:
        import shutil

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
                f"v3 pretrained weights differ from the v1 base: expected {expected_sha}, got {actual_sha}"
            )
        return resolved

    engine.resolve_pretrained = resolve_pretrained
    return engine


# ---------------------------------------------------------------------------
# prediction helpers
# ---------------------------------------------------------------------------


def predict_paths(
    net: Any,
    items: Sequence[dict[str, Any]],
    classes: Sequence[str],
    imgsz: int,
    device: str,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Predict a list of explicit ``(base, image)`` items.

    The engine's ``predict_rows`` binds to the global ``INPUT_ROOT`` and needs a
    ground-truth ``class_name``. The hard negatives and the regional pool have a
    different root and no assessable v3 class, so iterate explicit paths here.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from ultralytics.data.augment import classify_transforms

    missing = [str(item["base"] / item["image"]) for item in items if not (item["base"] / item["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} evaluation images missing, e.g. {missing[:3]}")

    transform = classify_transforms(size=imgsz)
    paths = [item["base"] / item["image"] for item in items]
    loader = DataLoader(
        engine.TransformDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device)
            out = net(images)
            probs = out[0] if isinstance(out, (tuple, list)) else out
            probs = probs.float().cpu().numpy()
            if probs.ndim != 2 or probs.shape[1] != len(classes):
                raise RuntimeError(f"unexpected probability shape {probs.shape} for {len(classes)} classes")
            for local, global_idx in enumerate(indices.tolist()):
                item = items[global_idx]
                p = probs[local]
                ordered = np.sort(p)
                pred_idx = int(np.argmax(p))
                record = {
                    "image": item["image"],
                    "predicted": classes[pred_idx],
                    "predicted_index": pred_idx,
                    "confidence": float(p[pred_idx]),
                    "margin": float(ordered[-1] - ordered[-2]),
                    "probs": [float(x) for x in p],
                }
                record.update({k: v for k, v in item.items() if k not in {"base", "image"}})
                records.append(record)
    return records


def with_actual(records: Sequence[dict[str, Any]], classes: Sequence[str]) -> list[dict[str, Any]]:
    class_index = {name: i for i, name in enumerate(classes)}
    out = []
    for record in records:
        item = dict(record)
        item["actual"] = record["class_name"]
        item["actual_index"] = class_index[record["class_name"]]
        out.append(item)
    return out


def margin_curve(records: Sequence[dict[str, Any]], classes: Sequence[str], thresholds: Sequence[float]) -> list[dict[str, Any]]:
    """Confidence-curve analogue over the top1-top2 margin."""
    class_index = {name: i for i, name in enumerate(classes)}
    y_true = [class_index[r["actual"]] for r in records]
    y_pred = [class_index[r["predicted"]] for r in records]

    def record_margin(record: dict[str, Any]) -> float:
        if "margin" in record:
            return float(record["margin"])
        probs = sorted(float(p) for p in record["probs"])
        return probs[-1] - probs[-2]

    margins = [record_margin(r) for r in records]
    total = len(records)
    out = []
    for t in thresholds:
        covered = [i for i, m in enumerate(margins) if m >= t]
        correct = sum(1 for i in covered if y_true[i] == y_pred[i])
        per_class = {}
        for name in classes:
            cidx = class_index[name]
            predicted = [i for i in covered if y_pred[i] == cidx]
            tp = sum(1 for i in predicted if y_true[i] == cidx)
            support = sum(1 for i in range(total) if y_true[i] == cidx)
            per_class[name] = {
                "predicted_covered": len(predicted),
                "precision_at_coverage": tp / len(predicted) if predicted else None,
                "recall_at_coverage": tp / support if support else None,
                "support": support,
            }
        out.append(
            {
                "threshold": round(float(t), 4),
                "coverage": len(covered) / total if total else None,
                "covered": len(covered),
                "abstained": total - len(covered),
                "correct_covered": correct,
                "accuracy_at_coverage": correct / len(covered) if covered else None,
                "precision_at_coverage": correct / len(covered) if covered else None,
                "per_class": per_class,
                "n": total,
            }
        )
    return out


# ---------------------------------------------------------------------------
# hard negatives (isolated leaflets — leaf type is not assessable)
# ---------------------------------------------------------------------------


def hard_negative_summary(
    records: Sequence[dict[str, Any]],
    conf_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    n = len(records)
    pred_compound = [r for r in records if r["predicted"] == "compound_leaf"]
    pred_simple = [r for r in records if r["predicted"] == "simple_leaf"]
    abstain_conf = [r for r in records if r["confidence"] < conf_threshold]
    abstain_margin = [r for r in records if r["margin"] < margin_threshold]
    abstain_any = [
        r for r in records if r["confidence"] < conf_threshold or r["margin"] < margin_threshold
    ]
    abstain_all = [
        r for r in records if r["confidence"] < conf_threshold and r["margin"] < margin_threshold
    ]
    covered = [r for r in records if r["confidence"] >= conf_threshold]
    covered_compound = [r for r in covered if r["predicted"] == "compound_leaf"]

    def ratio(count: int) -> float | None:
        return count / n if n else None

    return {
        "n": n,
        "species": dict(sorted(Counter(r["species"] for r in records).items())),
        "view_framing": dict(sorted(Counter(r.get("view_framing") for r in records).items())),
        "confidence": {
            "min": min((r["confidence"] for r in records), default=None),
            "median": statistics.median([r["confidence"] for r in records]) if records else None,
            "max": max((r["confidence"] for r in records), default=None),
        },
        "margin": {
            "min": min((r["margin"] for r in records), default=None),
            "median": statistics.median([r["margin"] for r in records]) if records else None,
            "max": max((r["margin"] for r in records), default=None),
        },
        "predicted_compound": len(pred_compound),
        "predicted_simple": len(pred_simple),
        "predicted_compound_fraction": ratio(len(pred_compound)),
        "predicted_simple_fraction": ratio(len(pred_simple)),
        "confidence_threshold": conf_threshold,
        "margin_threshold": margin_threshold,
        "abstained_confidence": len(abstain_conf),
        "abstained_confidence_fraction": ratio(len(abstain_conf)),
        "abstained_margin": len(abstain_margin),
        "abstained_margin_fraction": ratio(len(abstain_margin)),
        "abstained_any": len(abstain_any),
        "abstained_any_fraction": ratio(len(abstain_any)),
        "abstained_both": len(abstain_all),
        "abstained_both_fraction": ratio(len(abstain_all)),
        "covered_at_confidence_threshold": len(covered),
        "covered_at_confidence_fraction": ratio(len(covered)),
        "compound_false_positives_among_covered": len(covered_compound),
        "compound_false_positive_fraction_among_covered": (
            len(covered_compound) / len(covered) if covered else None
        ),
        "species_label_errors_if_leaflet_counted_as_compound": len(pred_simple),
        "interpretation": (
            "hard negatives are view-not-assessable isolated leaflets: a simple_leaf "
            "prediction is the expected safe answer, a compound_leaf prediction is a "
            "false positive of compound evidence. Abstention is the other safe answer."
        ),
        "rows": [
            {
                "image": r["image"],
                "uid": r.get("uid"),
                "species": r["species"],
                "predicted": r["predicted"],
                "confidence": r["confidence"],
                "margin": r["margin"],
                "abstained_confidence": r["confidence"] < conf_threshold,
                "abstained_margin": r["margin"] < margin_threshold,
            }
            for r in records
        ],
    }


# ---------------------------------------------------------------------------
# regional Mid-Atlantic pool
# ---------------------------------------------------------------------------


def select_gold_regional(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    literal = [r for r in rows if r.get("triangulation_bucket") == GOLD_BUCKET]
    if literal:
        selected = literal
        rule = f"literal triangulation_bucket == '{GOLD_BUCKET}'"
    else:
        selected = [
            r
            for r in rows
            if r.get("triangulation_bucket") in GOLD_FALLBACK_BUCKETS
            and r.get("stage") == GOLD_FALLBACK_STAGE
        ]
        rule = (
            f"bucket '{GOLD_BUCKET}' is absent from pool_manifest.jsonl; used the "
            f"gold, view-valid regional cohort (bucket in {list(GOLD_FALLBACK_BUCKETS)} "
            f"AND stage == '{GOLD_FALLBACK_STAGE}')"
        )
    assessable = [r for r in selected if r.get("devo_leaf_type") in {"simple", "compound"}]
    excluded = [r for r in selected if r.get("devo_leaf_type") not in {"simple", "compound"}]
    meta = {
        "selection_rule": rule,
        "selected": len(selected),
        "assessable": len(assessable),
        "excluded_not_assessable": len(excluded),
        "class_counts": dict(sorted(Counter(f"{r['devo_leaf_type']}_leaf" for r in assessable).items())),
        "source_counts": dict(sorted(Counter(str(r.get("source")) for r in assessable).items())),
        "species_counts": dict(sorted(Counter(str(r.get("species")) for r in assessable).items())),
    }
    return assessable, meta


def regional_records(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in items:
        rows.append(
            {
                "base": POOL_ROOT,
                "image": row["image"],
                "class_name": f"{row['devo_leaf_type']}_leaf",
                "species": row.get("species") or "unknown",
                "source": row.get("source") or "unknown",
                "stage": row.get("stage"),
                "triangulation_bucket": row.get("triangulation_bucket"),
                "uid": row.get("uid"),
                "image_width": row.get("image_width"),
                "image_height": row.get("image_height"),
            }
        )
    return rows


def regional_summary(
    records: Sequence[dict[str, Any]],
    classes: Sequence[str],
    conf_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    class_index = {name: i for i, name in enumerate(classes)}
    y_true = [class_index[r["actual"]] for r in records]
    y_pred = [class_index[r["predicted"]] for r in records]
    overall = engine.classification_metrics(y_true, y_pred, classes)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["species"]].append(record)
    species_rows = []
    for species, items in groups.items():
        correct = sum(1 for r in items if r["actual"] == r["predicted"])
        species_rows.append(
            {
                "species": species,
                "class_name": items[0]["actual"],
                "source": items[0]["source"],
                "n": len(items),
                "correct": correct,
                "accuracy": correct / len(items),
                "mean_confidence": statistics.mean(r["confidence"] for r in items),
                "abstained_confidence": sum(1 for r in items if r["confidence"] < conf_threshold),
            }
        )
    species_rows.sort(key=lambda r: (r["accuracy"], -r["n"], r["species"]))

    by_source = {}
    for source in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == source]
        by_source[source] = engine.classification_metrics(
            [class_index[r["actual"]] for r in subset],
            [class_index[r["predicted"]] for r in subset],
            classes,
        )

    at_threshold = engine.metrics_at_threshold(records, classes, conf_threshold)
    covered = [r for r in records if r["confidence"] >= conf_threshold]
    abstained = [r for r in records if r["confidence"] < conf_threshold]
    return {
        "overall": overall,
        "by_source": by_source,
        "per_species": {
            "n_species": len(species_rows),
            "macro_species_accuracy": statistics.mean(r["accuracy"] for r in species_rows) if species_rows else None,
            "rows": species_rows,
        },
        "selective_at_confidence_threshold": {
            "threshold": conf_threshold,
            "coverage": len(covered) / len(records) if records else None,
            "covered": len(covered),
            "abstained": len(abstained),
            "accuracy_at_coverage": (
                sum(1 for r in covered if r["actual"] == r["predicted"]) / len(covered) if covered else None
            ),
            "metrics": at_threshold,
        },
        "margin_threshold": margin_threshold,
        "abstained_margin": sum(1 for r in records if r["margin"] < margin_threshold),
        "mean_confidence": statistics.mean(r["confidence"] for r in records) if records else None,
        "mean_margin": statistics.mean(r["margin"] for r in records) if records else None,
        "rows": [
            {
                "image": r["image"],
                "species": r["species"],
                "source": r["source"],
                "actual": r["actual"],
                "predicted": r["predicted"],
                "confidence": r["confidence"],
                "margin": r["margin"],
            }
            for r in records
        ],
    }


# ---------------------------------------------------------------------------
# report + comparison
# ---------------------------------------------------------------------------


def percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics["evaluation"]["test"]
    field = metrics["evaluation"]["field"]
    checks = {row["name"]: row for row in metrics["e4"]["checks"]}
    return {
        "test_top1": test["accuracy"],
        "test_macro_f1": test["macro"]["f1"],
        "test_n": test["n"],
        "test_species": len(metrics["data"]["species_by_split"]["test"]),
        "simple_leaf_recall": checks["simple_leaf_recall"]["value"],
        "compound_leaf_recall": checks["compound_leaf_recall"]["value"],
        "compound_leaf_field_recall": checks["compound_leaf_field_recall"]["value"],
        "compound_field_support": metrics["e4"]["compound_field_support"],
        "field_top1": field["accuracy"],
        "e4_passed": metrics["e4"]["passed"],
        "epochs_run": metrics["training"].get("epochs_run"),
        "training_minutes": metrics["training"].get("wall_clock_minutes"),
        "compound_train_domain_counts": metrics["data"]["by_split_class_domain"]["train"]["compound_leaf"],
    }


def juglans_snapshot(metrics: dict[str, Any], predictions_path: Path) -> dict[str, Any]:
    rows = []
    if predictions_path.is_file():
        rows = read_jsonl(predictions_path)
    juglans = [r for r in rows if r["species"] == "Juglans cinerea"]
    if not juglans:
        return {"n": 0}
    correct = sum(1 for r in juglans if r["actual"] == r["predicted"])
    field = [r for r in juglans if r["domain"] == "field"]
    lab = [r for r in juglans if r["domain"] == "lab"]
    species_row = next(
        (r for r in metrics["evaluation"]["test"]["per_species"]["rows"] if r["species"] == "Juglans cinerea"),
        None,
    )
    return {
        "n": len(juglans),
        "correct": correct,
        "accuracy": correct / len(juglans),
        "errors": len(juglans) - correct,
        "field": {
            "n": len(field),
            "correct": sum(1 for r in field if r["actual"] == r["predicted"]),
            "errors": sum(1 for r in field if r["actual"] != r["predicted"]),
            "accuracy": sum(1 for r in field if r["actual"] == r["predicted"]) / len(field) if field else None,
        },
        "lab": {
            "n": len(lab),
            "correct": sum(1 for r in lab if r["actual"] == r["predicted"]),
            "errors": sum(1 for r in lab if r["actual"] != r["predicted"]),
            "accuracy": sum(1 for r in lab if r["actual"] == r["predicted"]) / len(lab) if lab else None,
        },
        "mean_confidence": statistics.mean(r["confidence"] for r in juglans),
        "predicted_compound": sum(1 for r in juglans if r["predicted"] == "compound_leaf"),
        "predicted_simple": sum(1 for r in juglans if r["predicted"] == "simple_leaf"),
        "species_row": species_row,
        "comparison_note": (
            "Juglans cinerea V3 test crops are the 54 whole-rachis gold crops. "
            "The 47 isolated-leaflet Juglans crops are held out as view hard "
            "negatives and are NOT part of this recall."
        ),
    }


def v3_markdown(
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    hn: dict[str, Any],
    regional: dict[str, Any],
    regional_meta: dict[str, Any],
    supplementary: dict[str, Any],
    supplementary_meta: dict[str, Any],
) -> str:
    data = metrics["data"]
    training = metrics["training"]
    test = metrics["evaluation"]["test"]
    val = metrics["evaluation"]["val"]
    e4 = metrics["e4"]
    coverage = metrics["evaluation"]["coverage"]
    onnx = metrics["onnx"]
    verification = onnx["verification"]
    v3 = comparison["v3"]
    jug = comparison["juglans_cinerea"]

    lines = [
        "# E4 V3 — Triangulated Leaf Specialist (YOLO26n-cls)",
        "",
        f"Generated {metrics['generated_at']} by `{metrics['source']}`.",
        "",
        "V3 trains on **view-valid** crops only: every training/validation/test crop "
        "frames the whole foliar unit (simple blade or compound rachis). The 70 "
        "LeafSnap crops where a compound species' mask isolates a single leaflet "
        "carry no rachis evidence and are excluded from training and from the E4 "
        "test split; they are evaluated separately as view hard negatives.",
        "",
        "## 1. Headline: V1 vs V2 vs V3",
        "",
        "| metric | V1 | V2 | V3 (triangulated) | Δ V3−V2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("test_top1", "test top1"),
        ("simple_leaf_recall", "simple leaf recall"),
        ("compound_leaf_recall", "compound leaf recall"),
        ("compound_leaf_field_recall", "compound leaf FIELD recall"),
        ("test_macro_f1", "test macro F1"),
    ):
        before = comparison["v2"][key]
        after = v3[key]
        delta = (after - before) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
        lines.append(f"| {label} | {percent(comparison['v1'][key])} | {percent(before)} | {percent(after)} | {percent(delta)} |")
    lines += [
        f"| E4 overall | {'PASS' if comparison['v1']['e4_passed'] else 'FAIL'} | "
        f"{'PASS' if comparison['v2']['e4_passed'] else 'FAIL'} | "
        f"{'PASS' if v3['e4_passed'] else 'FAIL'} | — |",
        f"| test crops | {comparison['v1']['test_n']:,} | {comparison['v2']['test_n']:,} | {v3['test_n']:,} | — |",
        f"| test species | {comparison['v1']['test_species']} | {comparison['v2']['test_species']} | {v3['test_species']} | — |",
        f"| compound train field/lab | "
        f"{comparison['v1']['compound_train_domain_counts']['field']}/"
        f"{comparison['v1']['compound_train_domain_counts']['lab']} | "
        f"{comparison['v2']['compound_train_domain_counts']['field']}/"
        f"{comparison['v2']['compound_train_domain_counts']['lab']} | "
        f"{v3['compound_train_domain_counts']['field']}/{v3['compound_train_domain_counts']['lab']} | — |",
        "",
        "The three test splits are **not identical** (each revision re-materialises "
        "the species-disjoint split), so aggregate deltas are descriptive. The "
        "paired comparison below is restricted to species present in all three test "
        "splits.",
        "",
        f"Juglans cinerea recall changed **{percent(jug['v1']['accuracy'])} (V1, n={jug['v1']['n']}) → "
        f"{percent(jug['v2']['accuracy'])} (V2, n={jug['v2']['n']}) → "
        f"{percent(jug['v3']['accuracy'])} (V3, n={jug['v3']['n']})**.",
        "",
        "## 2. E4 pre-registered criteria (unchanged thresholds)",
        "",
        f"Criteria were registered in `EVALUATION_CRITERIA.md` before any V3 result; "
        f"thresholds are identical to V1/V2.",
        "",
        "| criterion | threshold | observed | result |",
        "|---|---:|---:|---|",
    ]
    for check in e4["checks"]:
        lines.append(
            f"| {check['name']} | {check['comparison']} {check['threshold']:.2f} | "
            f"{num(check['value'])} | {'PASS' if check['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"**E4 overall: {'PASS' if e4['passed'] else 'FAIL'}.** Compound-field support: "
        f"{e4['compound_field_correct']}/{e4['compound_field_support']}.",
        "",
        "## 3. Training configuration",
        "",
        "| item | value |",
        "|---|---|",
        f"| base model | `{training['model_name']}` (sha256 `{training['pretrained_sha256'][:16]}…`, identical to the V1/V2 base) |",
        f"| device | `{training['device']}` |",
        f"| imgsz / batch | {training['imgsz']} / {training['batch']} |",
        f"| epochs run / requested | {training['epochs_run']} / {training['epochs_requested']} (patience {training['patience']}) |",
        f"| seed / workers | {training['seed']} / 0 (deterministic, AMP off) |",
        f"| wall clock | {training['wall_clock_minutes']:.2f} min |",
        f"| mean epoch | {num(training.get('mean_epoch_seconds'), 1)} s |",
        f"| best epoch (val top1) | {training['best_epoch_from_csv']} ({num(training['best_val_top1_from_csv'])}) |",
        f"| early stopped | {num(training['early_stopped'])} |",
        "",
        "## 4. Data provenance and triangulation filter",
        "",
        f"- Input: `{data['input_root']}` (dataset version `{data.get('dataset_version')}`).",
        f"- Label source: {data['label_source']}.",
        f"- Crop rule: {data['crop']['bbox']}; margin {data['crop']['margin_fraction']} "
        f"({data['crop']['margin_rule']}).",
        f"- Manifest bucket(s): {', '.join(data['triangulation_filter']['manifest_bucket'])}; "
        f"view framing: {', '.join(data['triangulation_filter']['manifest_view_framing'])}.",
        f"- Splits are species-disjoint: {num(data['species_split_disjoint'])}.",
        f"- Hard negatives held out: {data['hard_negative_audit']['hard_negative_crops']} crops "
        f"({data['hard_negative_audit']['hard_negative_by_split']}); overlap with the training "
        f"manifest: {data['hard_negative_audit']['manifest_hard_negative_uid_overlap'] or 'none'}.",
        "",
        "| split | simple_leaf | compound_leaf | field | lab | total | species |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        sc = data["by_split_class"][split]
        sd = data["by_split_domain"][split]
        lines.append(
            f"| {split} | {sc['simple_leaf']} | {sc['compound_leaf']} | "
            f"{sd['field']} | {sd['lab']} | {sum(sc.values())} | {len(data['species_by_split'][split])} |"
        )
    lines += [
        "",
        f"Test is {percent(data['by_split_class']['test']['simple_leaf'] / sum(data['by_split_class']['test'].values()))} "
        f"simple; the no-skill majority baseline is {percent(test['majority_baseline'])}.",
        "",
        "## 5. Held-out test results",
        "",
        f"**Top1: {percent(test['accuracy'])} ({test['correct']}/{test['n']}).** "
        f"Macro precision / recall / F1: {num(test['macro']['precision'])} / "
        f"{num(test['macro']['recall'])} / {num(test['macro']['f1'])}.",
        "",
        "| class | support | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in test["confusion_matrix"]["labels"]:
        pc = test["per_class"][name]
        lines.append(
            f"| {name} | {pc['support']} | {num(pc['precision'])} | {num(pc['recall'])} | {num(pc['f1'])} |"
        )
    labels = test["confusion_matrix"]["labels"]
    lines += [
        "",
        "Confusion matrix (rows actual, columns predicted):",
        "",
        "| actual \\ predicted | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for i, name in enumerate(labels):
        row = test["confusion_matrix"]["rows_actual_columns_predicted"][i]
        lines.append("| " + name + " | " + " | ".join(str(x) for x in row) + " |")
    lines += [
        "",
        "| domain | n | top1 | simple recall | compound recall | macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain in ("field", "lab"):
        block = metrics["evaluation"][domain]
        lines.append(
            f"| {domain} | {block['n']} | {percent(block['accuracy'])} | "
            f"{num(block['per_class']['simple_leaf']['recall'])} | "
            f"{num(block['per_class']['compound_leaf']['recall'])} | {num(block['macro']['f1'])} |"
        )
    lines += [
        "",
        "## 6. Juglans cinerea — 54 whole-rachis gold crops",
        "",
        f"- Overall: **{percent(jug['v3']['accuracy'])}** ({jug['v3']['correct']}/{jug['v3']['n']}; "
        f"{jug['v3']['errors']} errors).",
        f"- Field: {percent(jug['v3']['field']['accuracy'])} "
        f"({jug['v3']['field']['correct']}/{jug['v3']['field']['n']}).",
        f"- Lab: {percent(jug['v3']['lab']['accuracy'])} "
        f"({jug['v3']['lab']['correct']}/{jug['v3']['lab']['n']}).",
        f"- Mean confidence: {num(jug['v3']['mean_confidence'])}; predicted compound "
        f"{jug['v3']['predicted_compound']}/{jug['v3']['n']}.",
        "",
        "The V1/V2 error mode — a mask crop isolating one butternut leaflet that looks "
        "simple — was a **view** defect. Those 47 crops are now hard negatives, so "
        "this number measures the model on crops that actually show the rachis.",
        "",
        "| version | n | accuracy | field accuracy | errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for version in ("v1", "v2", "v3"):
        block = jug[version]
        lines.append(
            f"| {version.upper()} | {block['n']} | {percent(block['accuracy'])} | "
            f"{percent(block['field']['accuracy'])} | {block['errors']} |"
        )
    lines += [
        "",
        "## 7. Per-species test accuracy (species-disjoint)",
        "",
        f"{test['per_species']['n_species']} species; macro species accuracy "
        f"{percent(test['per_species']['macro_species_accuracy'])}; "
        f"{test['per_species']['species_at_1_0']} species at 100%, "
        f"{len(test['per_species']['species_below_0_5'])} below 50%.",
        "",
        "| species | class | n | field/lab | accuracy |",
        "|---|---|---:|---|---:|",
    ]
    for row in test["per_species"]["rows"]:
        lines.append(
            f"| {row['species']} | {row['class_name']} | {row['n']} | "
            f"{row['domains']['field']}/{row['domains']['lab']} | {percent(row['accuracy'])} |"
        )
    lines += [
        "",
        "## 8. Validation-selected confidence / abstention operating point",
        "",
        f"Threshold selected on validation only: **{coverage['chosen']['threshold']:.4f}** "
        f"({coverage['selection']['selection_rule']}; target precision "
        f"{num(coverage['selection']['target_precision_used'])}).",
        "",
        "| split | threshold | coverage | abstained | accuracy@coverage | precision@coverage |",
        "|---|---:|---:|---:|---:|---:|",
        f"| val (selection) | {coverage['chosen']['threshold']:.4f} | {percent(coverage['chosen']['coverage'])} | "
        f"{coverage['chosen']['abstained']} | {percent(coverage['chosen']['accuracy_at_coverage'])} | "
        f"{percent(coverage['chosen']['precision_at_coverage'])} |",
        f"| test (applied) | {coverage['test_at_chosen']['threshold']:.4f} | "
        f"{percent(coverage['test_at_chosen']['coverage'])} | {coverage['test_at_chosen']['abstained']} | "
        f"{percent(coverage['test_at_chosen']['accuracy_at_coverage'])} | "
        f"{percent(coverage['test_at_chosen']['precision_at_coverage'])} |",
        "",
        "| class | test recall@coverage | test precision@coverage |",
        "|---|---:|---:|",
    ]
    for name in labels:
        pc = coverage["test_at_chosen"]["per_class"][name]
        lines.append(f"| {name} | {num(pc['recall_at_coverage'])} | {num(pc['precision_at_coverage'])} |")
    lines += [
        "",
        "Descriptive test-only coverage targets (not used to select the threshold):",
        "",
        "| target coverage | threshold | actual coverage | accuracy@coverage |",
        "|---|---:|---:|---:|",
    ]
    for row in coverage["test_at_targets"]:
        lines.append(
            f"| {percent(row['target_coverage'])} | {row['threshold']:.4f} | "
            f"{percent(row['coverage'])} | {percent(row['accuracy_at_coverage'])} |"
        )
    lines += [
        "",
        "## 9. View hard negatives (isolated leaflets, `hard_negatives_sub_part/test`)",
        "",
        f"**n = {hn['n']}** crops ({', '.join(f'{k}: {v}' for k, v in hn['species'].items())}); "
        f"view framing: {', '.join(f'{k}: {v}' for k, v in hn['view_framing'].items())}. "
        f"Leaf type is not assessable in these crops by construction.",
        "",
        "| outcome | count | fraction |",
        "|---|---:|---:|",
        f"| predicted `compound_leaf` (compound false positive) | {hn['predicted_compound']} | {percent(hn['predicted_compound_fraction'])} |",
        f"| predicted `simple_leaf` (safe answer / species-label error) | {hn['predicted_simple']} | {percent(hn['predicted_simple_fraction'])} |",
        f"| abstained by confidence < {hn['confidence_threshold']:.4f} | {hn['abstained_confidence']} | {percent(hn['abstained_confidence_fraction'])} |",
        f"| abstained by margin < {hn['margin_threshold']:.4f} | {hn['abstained_margin']} | {percent(hn['abstained_margin_fraction'])} |",
        f"| abstained by confidence OR margin | {hn['abstained_any']} | {percent(hn['abstained_any_fraction'])} |",
        f"| abstained by confidence AND margin | {hn['abstained_both']} | {percent(hn['abstained_both_fraction'])} |",
        "",
        f"At the validation-selected confidence threshold {hn['confidence_threshold']:.4f}, "
        f"{hn['covered_at_confidence_threshold']}/{hn['n']} leaflet crops are covered; "
        f"{hn['compound_false_positives_among_covered']} of those covered are compound "
        f"({percent(hn['compound_false_positive_fraction_among_covered'])}). "
        f"Confidence {num(hn['confidence']['min'])}–{num(hn['confidence']['max'])} "
        f"(median {num(hn['confidence']['median'])}); margin "
        f"{num(hn['margin']['min'])}–{num(hn['margin']['max'])} "
        f"(median {num(hn['margin']['median'])}).",
        "",
        "The margin threshold is the confidence rule's analogue applied to the "
        "top1−top2 margin on **validation only**; it is diagnostic for the "
        "two-class specialist, not a new E4 criterion.",
        "",
        f"Important: validation is saturated ({percent(val['accuracy'])} top1 at 100% coverage), "
        "so the pre-registered precision/coverage rule selects the lowest available "
        "thresholds and **cannot express abstention on this data**. The observed "
        "outcome is therefore a confident `simple_leaf` answer on every isolated "
        "leaflet (median confidence "
        f"{num(hn['confidence']['median'])}), not abstention. The model does not claim "
        "compound evidence where the crop shows none — which is the failure mode V3 "
        "was built to remove — but a deployment gate should not rely on this "
        "specialist's confidence to detect an isolated-leaflet view.",
        "",
        "## 10. Mid-Atlantic regional generalization (Henrico × Tuckaway)",
        "",
        f"Cohort: **{regional_meta['assessable']}** view-valid Stage-2 regional leaf photos "
        f"({regional_meta['class_counts']}) across {len(regional_meta['species_counts'])} species; "
        f"sources {regional_meta['source_counts']}. Selection rule: {regional_meta['selection_rule']}.",
        "",
        f"**Regional accuracy: {percent(regional['overall']['accuracy'])} "
        f"({regional['overall']['correct']}/{regional['overall']['n']}).** "
        f"Macro F1 {num(regional['overall']['macro']['f1'])}.",
        "",
        "| class | support | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in regional["overall"]["confusion_matrix"]["labels"]:
        pc = regional["overall"]["per_class"][name]
        lines.append(
            f"| {name} | {pc['support']} | {num(pc['precision'])} | {num(pc['recall'])} | {num(pc['f1'])} |"
        )
    rlabels = regional["overall"]["confusion_matrix"]["labels"]
    lines += [
        "",
        "| actual \\ predicted | " + " | ".join(rlabels) + " |",
        "|---|" + "---:|" * len(rlabels),
    ]
    for i, name in enumerate(rlabels):
        row = regional["overall"]["confusion_matrix"]["rows_actual_columns_predicted"][i]
        lines.append("| " + name + " | " + " | ".join(str(x) for x in row) + " |")
    lines += [
        "",
        "| source | n | top1 | simple recall | compound recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for source, block in regional["by_source"].items():
        lines.append(
            f"| {source} | {block['n']} | {percent(block['accuracy'])} | "
            f"{num(block['per_class']['simple_leaf']['recall'])} | "
            f"{num(block['per_class']['compound_leaf']['recall'])} |"
        )
    lines += [
        "",
        f"Selective at the LeafSnap validation threshold {regional['selective_at_confidence_threshold']['threshold']:.4f}: "
        f"coverage {percent(regional['selective_at_confidence_threshold']['coverage'])} "
        f"({regional['selective_at_confidence_threshold']['covered']}/{regional['overall']['n']}), "
        f"abstained {regional['selective_at_confidence_threshold']['abstained']}, "
        f"accuracy@coverage {percent(regional['selective_at_confidence_threshold']['accuracy_at_coverage'])}. "
        f"Margin-abstained: {regional['abstained_margin']}. Mean confidence "
        f"{num(regional['mean_confidence'])}, mean margin {num(regional['mean_margin'])}.",
        "",
        "| species | class | source | n | correct | accuracy | mean confidence | abstained@threshold |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in regional["per_species"]["rows"]:
        lines.append(
            f"| {row['species']} | {row['class_name']} | {row['source']} | {row['n']} | "
            f"{row['correct']} | {percent(row['accuracy'])} | {num(row['mean_confidence'])} | "
            f"{row['abstained_confidence']} |"
        )
    lines += [
        "",
        f"Descriptive supplementary cohort — all Stage-2 species-labelled regional photos "
        f"(gold **and** non-gold framing), n={supplementary_meta['assessable']}, "
        f"classes {supplementary_meta['class_counts']}: accuracy "
        f"{percent(supplementary['overall']['accuracy'])} "
        f"({supplementary['overall']['correct']}/{supplementary['overall']['n']}). "
        f"This includes photos the triangulation marked `view_unresolved` or "
        f"`stage_context`; it is context, not the pre-registered regional test.",
        "",
        "The regional labels are species-level Dirr/GoBotany leaf-type priors applied "
        "to unsegmented iNaturalist photos, not per-photo adjudications. The Stage-2 "
        "gate is zero-shot CLIP triage. Tuckaway contributes walk photos with no taxon "
        "metadata, so the view-valid regional cohort is effectively **Henrico-only** "
        f"({regional_meta['source_counts'].get('tuckaway', 0)} Tuckaway photos).",
        "",
        "## 11. ONNX export and verification",
        "",
        f"- File: `{onnx['path']}` ({onnx['info']['bytes']} bytes, opset 17, static, batch 1, imgsz {training['imgsz']}).",
        f"- Input: `{onnx['info']['inputs'][0]['name']}` {onnx['info']['inputs'][0]['shape']} "
        f"{onnx['info']['inputs'][0]['dtype']}.",
        f"- Output: `{onnx['info']['outputs'][0]['name']}` {onnx['info']['outputs'][0]['shape']} "
        f"{onnx['info']['outputs'][0]['dtype']} (softmax probabilities).",
        f"- Labels/preprocessing: `{onnx['labels_path']}`.",
        f"- Verified on all {verification['test_image_count']} test crops: ONNX top1 "
        f"{percent(verification['top1_accuracy'])}, PyTorch top1 {percent(test['accuracy'])}, "
        f"prediction agreement {percent(verification['pytorch_prediction_agreement'])}, "
        f"max |prob diff| {verification['max_abs_prob_diff_vs_torch']:.2e}.",
        "",
        "Preprocessing must match `classify_transforms(size=224)`: RGB, shortest edge "
        "resized to 224 (BILINEAR), center crop 224×224, scale to [0,1], mean 0 / std 1. "
        "No letterbox, no ImageNet normalization; the output is already softmax. Class "
        f"order is model order: {onnx['labels']['class_order']}.",
        "",
        "## 12. Caveats",
        "",
    ]
    lines += [f"- {caveat}" for caveat in metrics["caveats"]]
    lines += [
        "",
        "## 13. Artifacts",
        "",
        "| artifact | path |",
        "|---|---|",
        f"| best checkpoint | `{training['best_checkpoint']}` |",
        f"| ONNX | `{metrics['onnx']['path']}` |",
        f"| labels | `{metrics['onnx']['labels_path']}` |",
        f"| metrics | `{OUT_ROOT / 'metrics.json'}` |",
        f"| predictions (test) | `{OUT_ROOT / 'predictions_test.jsonl'}` |",
        f"| predictions (hard negatives) | `{OUT_ROOT / 'predictions_hard_negatives_test.jsonl'}` |",
        f"| predictions (regional) | `{OUT_ROOT / 'predictions_regional_gold.jsonl'}` |",
        f"| V1/V2/V3 comparison | `{OUT_ROOT / 'v1_v2_v3_comparison.json'}` |",
        f"| criteria | `{OUT_ROOT / 'EVALUATION_CRITERIA.md'}` |",
        "",
    ]
    return "\n".join(lines)


def build_comparison(metrics: dict[str, Any], juglans: dict[str, Any]) -> dict[str, Any]:
    v1_metrics = read_json(V1_ROOT / "metrics.json")
    v2_metrics = read_json(V2_ROOT / "metrics.json")
    v1 = metric_row(v1_metrics)
    v2 = metric_row(v2_metrics)
    v3 = metric_row(metrics)

    def pair(metrics_block: dict[str, Any], root: Path) -> dict[str, Any]:
        rows = read_jsonl(root / "predictions_test.jsonl")
        jug = [r for r in rows if r["species"] == "Juglans cinerea"]
        correct = sum(1 for r in jug if r["actual"] == r["predicted"])
        field = [r for r in jug if r["domain"] == "field"]
        return {
            "n": len(jug),
            "correct": correct,
            "errors": len(jug) - correct,
            "accuracy": correct / len(jug) if jug else None,
            "field": {
                "n": len(field),
                "correct": sum(1 for r in field if r["actual"] == r["predicted"]),
                "errors": sum(1 for r in field if r["actual"] != r["predicted"]),
                "accuracy": (
                    sum(1 for r in field if r["actual"] == r["predicted"]) / len(field) if field else None
                ),
            },
        }

    juglans_blocks = {
        "v1": pair(v1_metrics, V1_ROOT),
        "v2": pair(v2_metrics, V2_ROOT),
        "v3": {
            "n": juglans["n"],
            "correct": juglans["correct"],
            "errors": juglans["errors"],
            "accuracy": juglans["accuracy"],
            "mean_confidence": juglans["mean_confidence"],
            "predicted_compound": juglans["predicted_compound"],
            "predicted_simple": juglans["predicted_simple"],
            "field": juglans["field"],
            "lab": juglans["lab"],
        },
    }
    juglans_blocks["v3"]["errors"] = juglans_blocks["v3"]["n"] - juglans_blocks["v3"]["correct"]

    species_sets = {
        "v1": set(v1_metrics["data"]["species_by_split"]["test"]),
        "v2": set(v2_metrics["data"]["species_by_split"]["test"]),
        "v3": set(metrics["data"]["species_by_split"]["test"]),
    }
    common = sorted(species_sets["v1"] & species_sets["v2"] & species_sets["v3"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "v1_metrics": str(V1_ROOT / "metrics.json"),
        "v2_metrics": str(V2_ROOT / "metrics.json"),
        "v3_metrics": str(OUT_ROOT / "metrics.json"),
        "note": (
            "V1/V2/V3 use species-disjoint splits produced by the same deterministic "
            "algorithm, but each revision re-materialises the split, so test species "
            "and cardinality differ. Aggregate deltas describe the shipped experiment "
            "versions; paired claims should use the common-species intersection."
        ),
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "deltas_v3_minus_v2": {
            key: (v3[key] - v2[key]) if isinstance(v2[key], (int, float)) and isinstance(v3[key], (int, float)) else None
            for key in (
                "test_top1",
                "test_macro_f1",
                "simple_leaf_recall",
                "compound_leaf_recall",
                "compound_leaf_field_recall",
            )
        },
        "deltas_v3_minus_v1": {
            key: (v3[key] - v1[key]) if isinstance(v1[key], (int, float)) and isinstance(v3[key], (int, float)) else None
            for key in (
                "test_top1",
                "test_macro_f1",
                "simple_leaf_recall",
                "compound_leaf_recall",
                "compound_leaf_field_recall",
            )
        },
        "test_species_overlap": {
            "v1_count": len(species_sets["v1"]),
            "v2_count": len(species_sets["v2"]),
            "v3_count": len(species_sets["v3"]),
            "common_count": len(common),
            "common": common,
            "v3_only": sorted(species_sets["v3"] - species_sets["v1"] - species_sets["v2"]),
        },
        "juglans_cinerea": juglans_blocks,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


engine = load_engine()


def run_extra_evaluations(metrics: dict[str, Any], device: str) -> dict[str, Any]:
    net, classes = engine.load_best_model(OUT_ROOT / "best.pt", device)
    imgsz = int(metrics["training"]["imgsz"])
    conf_threshold = float(metrics["evaluation"]["coverage"]["chosen"]["threshold"])

    val_records = read_jsonl(OUT_ROOT / "predictions_val.jsonl")
    margin_thresholds = [round(0.005 * i, 4) for i in range(1, 201)]
    val_margin_curve = margin_curve(val_records, classes, margin_thresholds)
    margin_selection = engine.choose_threshold(val_margin_curve)
    margin_threshold = float(margin_selection["chosen"]["threshold"])

    # (b) hard negatives
    hn_rows = [r for r in read_jsonl(HN_MANIFEST) if r["split"] == "test"]
    hn_items = [
        {
            "base": DATA_ROOT,
            "image": row["image"],
            "species": row["species"],
            "uid": row.get("uid"),
            "view_framing": row.get("view_framing"),
            "triangulation_bucket": row.get("triangulation_bucket"),
        }
        for row in hn_rows
    ]
    hn_records = predict_paths(net, hn_items, classes, imgsz, device)
    (OUT_ROOT / "predictions_hard_negatives_test.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in hn_records) + "\n", encoding="utf-8"
    )
    hn = hard_negative_summary(hn_records, conf_threshold, margin_threshold)
    hn["margin_selection"] = {
        "rule": margin_selection["selection_rule"],
        "target_precision_used": margin_selection["target_precision_used"],
        "chosen": margin_selection["chosen"],
        "val_curve": val_margin_curve,
    }

    # (c) regional gold pool
    pool_rows = read_jsonl(POOL_MANIFEST)
    gold_rows, regional_meta = select_gold_regional(pool_rows)
    gold_items = regional_records(gold_rows)
    gold_records = with_actual(predict_paths(net, gold_items, classes, imgsz, device), classes)
    (OUT_ROOT / "predictions_regional_gold.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in r.items()}, sort_keys=True) for r in gold_records) + "\n",
        encoding="utf-8",
    )
    regional = regional_summary(gold_records, classes, conf_threshold, margin_threshold)
    regional["cohort"] = regional_meta

    # supplementary: every Stage-2 species-labelled regional photo
    supplementary_rows = [
        r
        for r in pool_rows
        if r.get("stage") == GOLD_FALLBACK_STAGE and r.get("devo_leaf_type") in {"simple", "compound"}
    ]
    supplementary_meta = {
        "selection_rule": "stage == 'stage_2_anatomy' AND devo_leaf_type in {simple, compound} (all buckets)",
        "assessable": len(supplementary_rows),
        "class_counts": dict(sorted(Counter(f"{r['devo_leaf_type']}_leaf" for r in supplementary_rows).items())),
        "bucket_counts": dict(sorted(Counter(str(r.get("triangulation_bucket")) for r in supplementary_rows).items())),
    }
    supplementary_items = regional_records(supplementary_rows)
    supplementary_records = with_actual(predict_paths(net, supplementary_items, classes, imgsz, device), classes)
    supplementary = regional_summary(supplementary_records, classes, conf_threshold, margin_threshold)
    supplementary["cohort"] = supplementary_meta

    return {
        "hard_negatives": hn,
        "regional_gold": regional,
        "regional_stage2_supplementary": supplementary,
        "regional_meta": regional_meta,
        "supplementary_meta": supplementary_meta,
        "margin_threshold": margin_threshold,
        "margin_selection": {
            "rule": margin_selection["selection_rule"],
            "target_precision_used": margin_selection["target_precision_used"],
            "chosen_threshold": margin_threshold,
            "chosen_coverage": margin_selection["chosen"]["coverage"],
            "chosen_precision": margin_selection["chosen"]["precision_at_coverage"],
        },
    }


def log_section(metrics: dict[str, Any], comparison: dict[str, Any], extras: dict[str, Any]) -> str:
    v3 = comparison["v3"]
    jug = comparison["juglans_cinerea"]
    hn = extras["hard_negatives"]
    regional = extras["regional_gold"]
    regional_meta = extras["regional_meta"]
    e4 = metrics["e4"]
    checks = ", ".join(
        f"{c['name']} {c['value']:.4f} ({'PASS' if c['pass'] else 'FAIL'} ≥ {c['threshold']:.2f})"
        for c in e4["checks"]
    )
    return "\n".join(
        [
            "",
            "## 12. V3 Triangulated Leaf Specialist & Henrico × Tuckaway Regional Generalization",
            "",
            f"Run {metrics['generated_at']}; `experiments/leafsnap-yolo26/"
            "train_leaf_specialist_v3_triangulated.py`; output "
            "`scratch/leafsnap-yolo26/leaf-specialist-v3-triangulated/`.",
            "",
            "### 12.1 What V3 changes",
            "",
            "The V1/V2 residual error was a **view** defect, not an architecture defect: "
            "a compound species' mask crop can isolate one leaflet, which looks simple. "
            "V3 trains only on view-valid `gold_specialist_train` crops (5,262 whole-foliar-"
            "unit crops) and holds the 70 isolated-leaflet crops out of training and out of "
            "the E4 test split. E4 thresholds are unchanged (pre-registered).",
            "",
            "This supersedes the leaf-type specialist status recorded in §11: E4 is now "
            "validated on view-valid crops (all four criteria PASS). The §11 organ-gate "
            "runtime decision (V2 five-class presence gate + fixed center crop) is "
            "unchanged and independent.",
            "",
            f"Training: `yolo26n-cls.pt` from the same V1 base checkpoint, imgsz 224, "
            f"batch {metrics['training']['batch']}, epochs {metrics['training']['epochs_requested']}, "
            f"patience {metrics['training']['patience']}, device {metrics['training']['device']}, "
            f"seed {metrics['training']['seed']}, workers 0; "
            f"{metrics['training']['epochs_run']} epochs run in "
            f"{metrics['training']['wall_clock_minutes']:.2f} min.",
            "",
            "### 12.2 E4 result (held-out triangulated test, n="
            f"{v3['test_n']}, {v3['test_species']} species)",
            "",
            f"**E4 {'PASS' if e4['passed'] else 'FAIL'}.** {checks}.",
            "",
            "| version | test top1 | simple recall | compound recall | compound FIELD recall | E4 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        + [
            f"| {version.upper()} | {percent(comparison[version]['test_top1'])} | "
            f"{percent(comparison[version]['simple_leaf_recall'])} | "
            f"{percent(comparison[version]['compound_leaf_recall'])} | "
            f"{percent(comparison[version]['compound_leaf_field_recall'])} | "
            f"{'PASS' if comparison[version]['e4_passed'] else 'FAIL'} |"
            for version in ("v1", "v2", "v3")
        ]
        + [
            "",
            "The three test splits are species-disjoint but re-materialised per revision; "
            f"only {comparison['test_species_overlap']['common_count']} species appear in all "
            "three, so aggregate deltas are descriptive.",
            "",
            f"**Juglans cinerea** (54 whole-rachis gold test crops): "
            f"{percent(jug['v3']['accuracy'])} ({jug['v3']['correct']}/{jug['v3']['n']}), "
            f"field {percent(jug['v3']['field']['accuracy'])} "
            f"({jug['v3']['field']['correct']}/{jug['v3']['field']['n']}). "
            f"V1 {percent(jug['v1']['accuracy'])} (n={jug['v1']['n']}) → "
            f"V2 {percent(jug['v2']['accuracy'])} (n={jug['v2']['n']}) → "
            f"V3 {percent(jug['v3']['accuracy'])} (n={jug['v3']['n']}).",
            "",
            "### 12.3 View hard negatives (isolated leaflets)",
            "",
            f"`hard_negatives_sub_part/test`, n={hn['n']} (all Juglans cinerea). At the "
            f"validation-selected confidence threshold {hn['confidence_threshold']:.4f}: "
            f"{hn['abstained_confidence']}/{hn['n']} abstain "
            f"({percent(hn['abstained_confidence_fraction'])}); margin-based abstention "
            f"({hn['margin_threshold']:.4f}) {hn['abstained_margin']}/{hn['n']}. "
            f"Predictions: {hn['predicted_simple']} simple (safe), "
            f"{hn['predicted_compound']} compound (false compound evidence). "
            f"Among covered crops, {hn['compound_false_positives_among_covered']}/"
            f"{hn['covered_at_confidence_threshold']} are compound.",
            "",
            "Validation is saturated, so the pre-registered rule chooses the lowest "
            "thresholds and cannot abstain here: the outcome is a confident `simple_leaf` "
            "on every isolated leaflet (median confidence "
            f"{num(hn['confidence']['median'])}), not abstention. No compound evidence is "
            "claimed, but a deployment gate should not rely on this confidence to detect "
            "an isolated-leaflet view.",
            "",
            "### 12.4 Mid-Atlantic regional generalization",
            "",
            f"Gold regional cohort: **{regional_meta['assessable']}** view-valid Stage-2 "
            f"regional photos ({regional_meta['class_counts']}) across "
            f"{len(regional_meta['species_counts'])} species from "
            f"{regional_meta['source_counts']}. "
            f"`gold_regional_pool` is not a literal bucket in the current "
            f"`pool_manifest.jsonl`; the cohort is the gold, view-valid Stage-2 set "
            f"(`gold_specialist_train` ∩ `stage_2_anatomy`).",
            "",
            f"**Regional accuracy {percent(regional['overall']['accuracy'])} "
            f"({regional['overall']['correct']}/{regional['overall']['n']})**, macro F1 "
            f"{num(regional['overall']['macro']['f1'])}; compound recall "
            f"{num(regional['overall']['per_class']['compound_leaf']['recall'])}, simple "
            f"recall {num(regional['overall']['per_class']['simple_leaf']['recall'])}. "
            f"At the LeafSnap threshold {regional['selective_at_confidence_threshold']['threshold']:.4f}, "
            f"coverage {percent(regional['selective_at_confidence_threshold']['coverage'])} "
            f"and accuracy@coverage "
            f"{percent(regional['selective_at_confidence_threshold']['accuracy_at_coverage'])}.",
            "",
            f"Tuckaway contributes **0** view-valid labeled leaf photos (its only "
            f"downloaded photo is Stage 3 / context; the 37 walk frames carry no taxon "
            f"metadata), so this is effectively a Henrico generalization test. Regional "
            f"labels are species-level Dirr/GoBotany priors on unsegmented iNaturalist "
            f"photos, and Stage-2 selection is zero-shot CLIP triage (heuristic agreement "
            f"50%) — treat the number as a first field-generalization estimate, not an "
            f"adjudicated benchmark.",
            "",
            "### 12.5 Artifacts",
            "",
            "`scratch/leafsnap-yolo26/leaf-specialist-v3-triangulated/`: `best.pt`, "
            "`leaf_specialist.onnx` (opset 17, static 224, batch 1), "
            "`leaf_specialist.labels.json`, `metrics.json`, `LEAF_SPECIALIST_V3.md`, "
            "`v1_v2_v3_comparison.json`, `predictions_test.jsonl`, "
            "`predictions_hard_negatives_test.jsonl`, `predictions_regional_gold.jsonl`.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-train", action="store_true", help="reuse an existing best.pt/metrics")
    parser.add_argument("--skip-export", action="store_true", help="reuse an existing leaf_specialist.onnx")
    parser.add_argument("--cpu", action="store_true", help="force CPU instead of MPS")
    args = parser.parse_args(argv)

    if not (DATA_ROOT / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"triangulated dataset is missing: {DATA_ROOT}")
    if not (HN_MANIFEST.is_file() and POOL_MANIFEST.is_file()):
        raise FileNotFoundError("hard-negative or mid-atlantic pool manifest is missing")
    if not args.cpu:
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available; pass --cpu to train on CPU")

    train_args = [
        "--epochs",
        str(DEFAULT_EPOCHS),
        "--patience",
        str(DEFAULT_PATIENCE),
        "--batch",
        str(DEFAULT_BATCH),
        "--imgsz",
        str(DEFAULT_IMGSZ),
        "--seed",
        str(DEFAULT_SEED),
    ]
    if args.skip_train:
        train_args.append("--skip-train")
    if args.skip_export:
        train_args.append("--skip-export")
    if args.cpu:
        train_args.append("--cpu")
    engine.main(train_args)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = read_json(OUT_ROOT / "metrics.json")
    metrics["experiment"] = "E4 V3 — Triangulated LeafSnap leaf-type specialist (YOLO26n-cls)"
    metrics["source"] = str(HERE / "train_leaf_specialist_v3_triangulated.py")
    metrics["revision"] = {
        "dataset": str(DATA_ROOT),
        "dataset_version": metrics["data"].get("dataset_version"),
        "training_policy": (
            "single V3 proof run: 20 epochs / patience 6 / batch 32 / imgsz 224 / seed 26 / workers 0"
        ),
        "sample_set_change": (
            "view-valid gold_specialist_train crops only; isolated-leaflet crops "
            "(hard_negatives_sub_part) removed from train/val/test and evaluated separately"
        ),
        "e4_thresholds_unchanged": True,
        "regional_evaluation": str(POOL_MANIFEST),
    }
    metrics["caveats"] = [
        c
        for c in metrics["caveats"]
        if "leafsnap_1735499.jpg" not in c
    ] + [
        "V3 excludes 70 isolated-leaflet crops from training and from the E4 test split; they are view-not-assessable for leaf type. The 54 Juglans cinerea whole-rachis test crops replace the V1/V2 leaflet crops, so Juglans recall is not directly comparable to the earlier revisions.",
        "The V1/V2/V3 test splits are separately materialised species-disjoint splits. Aggregate deltas are descriptive; only species present in all three test splits support a paired claim.",
        "Regional labels are species-level Dirr/GoBotany leaf-type priors applied to unsegmented iNaturalist photos, not per-photo adjudications. The Stage-2 framing gate is zero-shot CLIP triage (heuristic agreement 50%).",
        "The gold regional cohort has 49 photos across 21 species (support 1-5), and Tuckaway contributes no view-valid labeled leaf photo; per-species regional rows are low-support and should not be read as stable species accuracy.",
        "The requested `gold_regional_pool` bucket is absent from `pool_manifest.jsonl`; the evaluator resolves the regional gold cohort as `gold_specialist_train` AND `stage_2_anatomy` and records that rule in metrics.json.",
        "The hard negatives are not scored as an accuracy: leaf type is not assessable in an isolated leaflet. Abstention and the compound-vs-simple answer distribution are the observable outcomes.",
    ]
    metrics["margin_operating_point"] = {
        "definition": "top1 minus top2 softmax probability",
        "selection": "same pre-registered precision/coverage rule applied to validation only",
    }

    device = metrics["training"]["device"]
    extras = run_extra_evaluations(metrics, device)

    metrics["hard_negatives"] = extras["hard_negatives"]
    metrics["regional_generalization"] = extras["regional_gold"]
    metrics["regional_stage2_supplementary"] = extras["regional_stage2_supplementary"]
    metrics["regional_cohort"] = extras["regional_meta"]
    metrics["secondary_cohort"] = extras["supplementary_meta"]
    metrics["margin_selection"] = extras["margin_selection"]

    juglans = juglans_snapshot(metrics, OUT_ROOT / "predictions_test.jsonl")
    metrics["juglans_cinerea"] = juglans

    comparison = build_comparison(metrics, juglans)
    metrics["v1_v2_v3"] = comparison
    engine.write_json(OUT_ROOT / "metrics.json", metrics)
    engine.write_json(OUT_ROOT / "v1_v2_v3_comparison.json", comparison)

    # regenerate the engine's base report with the V3 caveats, then write the V3 report
    engine.write_report(metrics, OUT_ROOT)
    (OUT_ROOT / "LEAF_SPECIALIST_V3.md").write_text(
        v3_markdown(
            metrics,
            comparison,
            extras["hard_negatives"],
            extras["regional_gold"],
            extras["regional_meta"],
            extras["regional_stage2_supplementary"],
            extras["supplementary_meta"],
        ),
        encoding="utf-8",
    )

    log = log_section(metrics, comparison, extras)
    existing = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.is_file() else ""
    marker = "## 12. V3 Triangulated Leaf Specialist"
    if marker in existing:
        head = existing.split(marker, 1)[0].rstrip() + "\n"
        LOG_PATH.write_text(head + log, encoding="utf-8")
    else:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(log)
            if not log.endswith("\n"):
                fh.write("\n")

    result = {
        "e4_passed": metrics["e4"]["passed"],
        "checks": [
            {"name": c["name"], "value": c["value"], "threshold": c["threshold"], "pass": c["pass"]}
            for c in metrics["e4"]["checks"]
        ],
        "v3": {
            "test_top1": comparison["v3"]["test_top1"],
            "simple_leaf_recall": comparison["v3"]["simple_leaf_recall"],
            "compound_leaf_recall": comparison["v3"]["compound_leaf_recall"],
            "compound_leaf_field_recall": comparison["v3"]["compound_leaf_field_recall"],
            "test_macro_f1": comparison["v3"]["test_macro_f1"],
            "test_n": comparison["v3"]["test_n"],
        },
        "juglans_cinerea": {
            "v1": comparison["juglans_cinerea"]["v1"],
            "v2": comparison["juglans_cinerea"]["v2"],
            "v3": comparison["juglans_cinerea"]["v3"],
        },
        "hard_negatives": {
            "n": extras["hard_negatives"]["n"],
            "abstained_confidence": extras["hard_negatives"]["abstained_confidence"],
            "abstained_margin": extras["hard_negatives"]["abstained_margin"],
            "predicted_simple": extras["hard_negatives"]["predicted_simple"],
            "predicted_compound": extras["hard_negatives"]["predicted_compound"],
            "confidence_threshold": extras["hard_negatives"]["confidence_threshold"],
            "margin_threshold": extras["hard_negatives"]["margin_threshold"],
        },
        "regional_gold": {
            "n": extras["regional_gold"]["overall"]["n"],
            "accuracy": extras["regional_gold"]["overall"]["accuracy"],
            "compound_recall": extras["regional_gold"]["overall"]["per_class"]["compound_leaf"]["recall"],
            "simple_recall": extras["regional_gold"]["overall"]["per_class"]["simple_leaf"]["recall"],
            "selection_rule": extras["regional_meta"]["selection_rule"],
            "sources": extras["regional_meta"]["source_counts"],
        },
        "coverage": {
            "threshold": metrics["evaluation"]["coverage"]["chosen"]["threshold"],
            "test_coverage": metrics["evaluation"]["coverage"]["test_at_chosen"]["coverage"],
            "test_accuracy_at_coverage": metrics["evaluation"]["coverage"]["test_at_chosen"]["accuracy_at_coverage"],
        },
        "onnx": {
            "path": metrics["onnx"]["path"],
            "top1": metrics["onnx"]["verification"]["top1_accuracy"],
            "agreement": metrics["onnx"]["verification"]["pytorch_prediction_agreement"],
        },
        "training_minutes": metrics["training"].get("wall_clock_minutes"),
        "output_dir": str(OUT_ROOT),
        "report": str(OUT_ROOT / "LEAF_SPECIALIST_V3.md"),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
