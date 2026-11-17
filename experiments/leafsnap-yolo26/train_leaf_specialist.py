# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""E4 LeafSnap leaf-type specialist: fine-tune YOLO26n-cls (proof run).

Scope
-----
Train the two-class leaf-type specialist (simple_leaf / compound_leaf) that E3
authorized. E3 gated lower-level training on the organ detector; this script
does not touch the organ-gate artifacts.

What this script does
---------------------
1. Preflight the classification dataset in
   ``scratch/leafsnap-yolo26/classification``: folder counts, manifest rows,
   class/species/domain consistency, species-disjoint splits, image headers.
2. Pre-register the E4 criteria in ``EVALUATION_CRITERIA.md`` *before* any
   training results exist.
3. Deterministic fine-tune of ``yolo26n-cls.pt`` on MPS: imgsz 224, batch 16,
   30 epochs, patience 8, seed 26, workers 0. One proof run, no sweep.
4. Evaluate the best checkpoint on the held-out test split:
   * top1, per-class precision/recall/F1, confusion matrix, macro metrics,
   * field-only and lab-only breakdowns,
   * per-species accuracy (species-disjoint split; reveals class/species
     imbalance),
   * selective accuracy/precision at coverage using a confidence threshold
     selected on validation only.
5. Export the best checkpoint to ONNX opset 17, static 224x224, batch 1,
   output [1, 2]; verify the graph on the held-out test crops against PyTorch.
6. Write ``metrics.json`` + ``LEAF_SPECIALIST.md`` with parameters, data
   provenance, class counts, E4 pass/fail, and caveats.

Usage
-----
    .venv/bin/python experiments/leafsnap-yolo26/train_leaf_specialist.py --smoke
    .venv/bin/python experiments/leafsnap-yolo26/train_leaf_specialist.py
    .venv/bin/python experiments/leafsnap-yolo26/train_leaf_specialist.py --skip-train
    .venv/bin/python experiments/leafsnap-yolo26/train_leaf_specialist.py --report-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "classification"
OUT_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "leaf-specialist"
MANIFEST = INPUT_ROOT / "manifest.jsonl"
COUNTS = INPUT_ROOT / "counts.json"
LABELS = INPUT_ROOT / "labels.json"

DEFAULT_IMGSZ = 224
DEFAULT_BATCH = 16
DEFAULT_EPOCHS = 30
DEFAULT_PATIENCE = 8
DEFAULT_SEED = 26
PRETRAINED = "yolo26n-cls.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DOMAINS = ("field", "lab")

E4_MIN_TOP1 = 0.85
E4_MIN_SIMPLE_RECALL = 0.85
E4_MIN_COMPOUND_RECALL = 0.75
E4_MIN_COMPOUND_FIELD_RECALL = 0.70

SELECTIVE_TARGET_PRECISION = 0.99
SELECTIVE_FALLBACK_PRECISION = 0.95
SELECTIVE_MIN_COVERAGE = 0.50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def package_versions() -> dict[str, str]:
    import torch
    import ultralytics

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    try:
        import onnx

        versions["onnx"] = onnx.__version__
    except Exception:
        versions["onnx"] = "missing"
    try:
        import onnxruntime

        versions["onnxruntime"] = onnxruntime.__version__
    except Exception:
        versions["onnxruntime"] = "missing"
    return versions


def load_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    path = MANIFEST if path is None else path
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def folder_counts() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        out[split] = {}
        for class_dir in sorted(p for p in (INPUT_ROOT / split).iterdir() if p.is_dir()):
            out[split][class_dir.name] = sum(
                1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
    return out


def dataset_summary(rows: list[dict[str, Any]], labels: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    classes = sorted({r["class_name"] for r in rows})
    by_split_class: dict[str, dict[str, int]] = {}
    by_split_domain: dict[str, dict[str, int]] = {}
    by_split_class_domain: dict[str, dict[str, dict[str, int]]] = {}
    species_by_split: dict[str, list[str]] = {}
    species_class: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        srows = [r for r in rows if r["split"] == split]
        by_split_class[split] = {c: sum(1 for r in srows if r["class_name"] == c) for c in classes}
        by_split_domain[split] = {d: sum(1 for r in srows if r["domain"] == d) for d in DOMAINS}
        by_split_class_domain[split] = {
            c: {d: sum(1 for r in srows if r["class_name"] == c and r["domain"] == d) for d in DOMAINS}
            for c in classes
        }
        species_by_split[split] = sorted({r["species"] for r in srows})
        for r in srows:
            species_class.setdefault(r["species"], set()).add(r["class_name"])

    split_sets = [set(species_by_split[s]) for s in ("train", "val", "test")]
    disjoint = all(
        split_sets[i].isdisjoint(split_sets[j])
        for i in range(len(split_sets))
        for j in range(i + 1, len(split_sets))
    )
    multi_class_species = {s: sorted(v) for s, v in species_class.items() if len(v) > 1}
    return {
        "input_root": str(INPUT_ROOT),
        "files": {
            "manifest.jsonl": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST)},
            "labels.json": {"path": str(LABELS), "sha256": sha256_file(LABELS)},
            "counts.json": {"path": str(COUNTS), "sha256": sha256_file(COUNTS)},
        },
        "label_source": labels.get("label_source"),
        "crop": labels.get("crop"),
        "task": labels.get("task"),
        "class_names": classes,
        "total_rows": len(rows),
        "by_split_class": by_split_class,
        "by_split_domain": by_split_domain,
        "by_split_class_domain": by_split_class_domain,
        "counts_json": counts,
        "species_by_split": species_by_split,
        "species_split_disjoint": bool(disjoint),
        "species_class": {s: sorted(v)[0] for s, v in sorted(species_class.items())},
        "multi_class_species": multi_class_species,
    }


def preflight(rows: list[dict[str, Any]], labels: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    expected_classes = {"simple_leaf", "compound_leaf"}

    manifest_classes = {r["class_name"] for r in rows}
    if manifest_classes != expected_classes:
        errors.append(f"manifest classes {sorted(manifest_classes)} != expected {sorted(expected_classes)}")
    if set(labels.get("class_names", [])) != expected_classes:
        errors.append(f"labels.json classes {labels.get('class_names')} != expected {sorted(expected_classes)}")
    if any(r.get("class_name") not in expected_classes for r in rows):
        errors.append("manifest contains a class name outside simple_leaf/compound_leaf")
    if any(r.get("domain") not in DOMAINS for r in rows):
        errors.append("manifest contains an unknown domain")
    if any(not r.get("species") for r in rows):
        errors.append("manifest contains an empty species")
    if len({r["image"] for r in rows}) != len(rows):
        errors.append("manifest image paths are not unique")

    fc = folder_counts()
    for split in ("train", "val", "test"):
        for class_name, expected in counts["by_split_class"][split].items():
            actual = fc.get(split, {}).get(class_name, 0)
            if actual != expected:
                errors.append(f"folder count {split}/{class_name}={actual} != counts.json {expected}")
        actual_split = sum(fc.get(split, {}).values())
        manifest_split = sum(1 for r in rows if r["split"] == split)
        if actual_split != manifest_split:
            errors.append(f"folder count {split}={actual_split} != manifest rows {manifest_split}")

    missing = [r["image"] for r in rows if not (INPUT_ROOT / r["image"]).is_file()]
    if missing:
        errors.append(f"{len(missing)} manifest images missing on disk, e.g. {missing[:3]}")

    from PIL import Image

    unreadable: list[str] = []
    for row in rows:
        if row["split"] not in {"val", "test"}:
            continue
        try:
            with Image.open(INPUT_ROOT / row["image"]) as image:
                image.size
        except Exception as exc:
            unreadable.append(f"{row['image']}: {type(exc).__name__}: {exc}")
    if unreadable:
        errors.append(f"{len(unreadable)} val/test images unreadable, e.g. {unreadable[:3]}")

    summary = dataset_summary(rows, labels, counts)
    if not summary["species_split_disjoint"]:
        errors.append("train/val/test species are not disjoint")
    if summary["multi_class_species"]:
        errors.append(f"species mapped to multiple classes: {summary['multi_class_species']}")

    if errors:
        raise RuntimeError("dataset preflight failed:\n- " + "\n- ".join(errors))
    return summary


def criteria_payload() -> dict[str, Any]:
    return {
        "experiment": "E4 — LeafSnap leaf-type specialist",
        "registered_at": utc_now(),
        "primary_metric": "test top1 accuracy (argmax, no abstention)",
        "criteria": [
            {
                "name": "test_top1",
                "description": "top1 accuracy on the held-out test split",
                "threshold": E4_MIN_TOP1,
                "comparison": ">=",
            },
            {
                "name": "simple_leaf_recall",
                "description": "recall for simple_leaf on the held-out test split",
                "threshold": E4_MIN_SIMPLE_RECALL,
                "comparison": ">=",
            },
            {
                "name": "compound_leaf_recall",
                "description": "recall for compound_leaf on the held-out test split",
                "threshold": E4_MIN_COMPOUND_RECALL,
                "comparison": ">=",
            },
            {
                "name": "compound_leaf_field_recall",
                "description": "recall for compound_leaf on FIELD-domain test images only",
                "threshold": E4_MIN_COMPOUND_FIELD_RECALL,
                "comparison": ">=",
            },
        ],
        "pass_rule": "all four criteria must hold",
        "selective_operating_point": {
            "selection_split": "validation only; never tuned on test",
            "target_precision": SELECTIVE_TARGET_PRECISION,
            "fallback_precision": SELECTIVE_FALLBACK_PRECISION,
            "min_coverage": SELECTIVE_MIN_COVERAGE,
            "selection_rule": (
                "Choose the lowest confidence threshold (maximum coverage) whose "
                "validation precision among covered predictions is at least the "
                "target precision and whose coverage is at least min_coverage. "
                "If no threshold meets the target, fall back to the fallback "
                "precision; if neither is met, use the threshold with maximum "
                "validation precision (ties: higher coverage). Apply the chosen "
                "threshold unchanged to test."
            ),
            "definition": (
                "coverage = fraction of images with max softmax >= threshold; "
                "precision_at_coverage = correct / covered; abstained images are "
                "not scored as errors in that selective metric."
            ),
        },
        "reporting_rules": [
            "Overall accuracy cannot substitute for the per-class and field-only criteria.",
            "compound_leaf recall on FIELD images is reported separately and cannot be hidden behind lab or overall accuracy.",
            "Lab-only metrics are reported as context; they are not part of the pass/fail gate.",
            "The threshold is selected on validation only. Test coverage curves are descriptive and are not used for selection.",
            "The E4 gate is evaluated at full coverage (argmax), independent of the selective operating point.",
        ],
        "caveats_registered_before_results": [
            "Labels are exact Dirr leaf_type assertions at the species level, not per-image adjudications.",
            "Mixed or missing leaf_type species were excluded upstream; mixed-species leaf clusters are not represented.",
            "Crops are mask foreground bboxes plus a 10% margin, not raw full-frame field photos.",
            "The task is only simple vs compound leaf type; no margin, arrangement, or other feature claim is made.",
            "Splits are species-disjoint; test performance is on unseen species and may be lower than random-image splits.",
        ],
    }


def write_criteria(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "EVALUATION_CRITERIA.md"
    json_path = out_dir / "evaluation_criteria.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text())
        if not path.exists():
            write_criteria_markdown(payload, path)
        return payload
    payload = criteria_payload()
    write_json(json_path, payload)
    write_criteria_markdown(payload, path)
    return payload


def write_criteria_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# E4 pre-registered criteria — LeafSnap leaf-type specialist",
        "",
        f"Registered {payload['registered_at']} (before training results).",
        "",
        "All four criteria must hold on the held-out test split at argmax "
        "(full coverage, no abstention):",
        "",
        "| # | criterion | threshold |",
        "|---|---|---:|",
    ]
    for i, c in enumerate(payload["criteria"], 1):
        lines.append(f"| {i} | {c['description']} | {c['comparison']} {c['threshold']:.2f} |")
    sel = payload["selective_operating_point"]
    lines += [
        "",
        "## Selective operating point (selected on validation only)",
        "",
        f"- Target precision: {sel['target_precision']:.3f}; fallback: {sel['fallback_precision']:.3f}.",
        f"- Minimum coverage: {sel['min_coverage']:.2f}.",
        f"- Rule: {sel['selection_rule']}",
        f"- Definition: {sel['definition']}",
        "",
        "## Reporting rules",
        "",
    ]
    lines += [f"- {rule}" for rule in payload["reporting_rules"]]
    lines += ["", "## Caveats registered before results", ""]
    lines += [f"- {caveat}" for caveat in payload["caveats_registered_before_results"]]
    path.write_text("\n".join(lines) + "\n")


def resolve_pretrained(dest_dir: Path) -> Path:
    from ultralytics.utils.downloads import attempt_download_asset

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / PRETRAINED
    local_copy = ROOT / "scratch" / "leaf-specialist" / "pretrained" / PRETRAINED
    if not target.exists():
        if local_copy.is_file():
            shutil.copy2(local_copy, target)
        else:
            downloaded = Path(attempt_download_asset(str(target)))
            if downloaded.resolve() != target.resolve():
                shutil.copy2(downloaded, target)
    if not target.is_file() or target.stat().st_size < 1_000_000:
        raise RuntimeError(f"could not resolve pretrained {PRETRAINED} at {target}")
    return target


def ensure_dataset_links(out_dir: Path) -> Path:
    """Symlink the immutable input splits into the run dir for Ultralytics.

    Ultralytics writes ``<root>/train.cache`` and ``<root>/val.cache`` beside
    the split directories. Symlinks keep those caches out of the input tree.
    """
    work = out_dir / "dataset"
    work.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        link = work / split
        target = INPUT_ROOT / split
        if link.is_symlink():
            if link.resolve() != target.resolve():
                link.unlink()
                link.symlink_to(target, target_is_directory=True)
        elif link.exists():
            if not link.is_dir():
                raise RuntimeError(f"{link} exists and is not a directory")
        else:
            link.symlink_to(target, target_is_directory=True)
    return work


def build_smoke_dataset(out_dir: Path, rows: list[dict[str, Any]], per_class: int = 30) -> Path:
    root = out_dir / "smoke-dataset"
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val", "test"):
        for class_name in ("simple_leaf", "compound_leaf"):
            class_rows = [r for r in rows if r["split"] == split and r["class_name"] == class_name]
            class_rows = class_rows[:per_class]
            target_dir = root / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for i, row in enumerate(class_rows):
                src = INPUT_ROOT / row["image"]
                dst = target_dir / f"{i:04d}_{src.name}"
                dst.symlink_to(src)
    return root


def parse_results_csv(path: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    if not path.exists():
        return history
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clean: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                key = key.strip()
                value = (value or "").strip()
                try:
                    clean[key] = float(value)
                except ValueError:
                    clean[key] = value
            history.append(clean)
    return history


def train(args: argparse.Namespace, out_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    weights = resolve_pretrained(out_dir / "pretrained")
    device = "mps" if torch.backends.mps.is_available() and not args.cpu else "cpu"
    data_root = build_smoke_dataset(out_dir, load_manifest()) if args.smoke else ensure_dataset_links(out_dir)
    epochs = 1 if args.smoke else args.epochs
    patience = 1 if args.smoke else args.patience
    project = out_dir / ("smoke-runs" if args.smoke else "runs")
    run_name = "smoke" if args.smoke else "train"

    params: dict[str, Any] = {
        "experiment": "E4 — LeafSnap leaf-type specialist (YOLO26n-cls proof run)",
        "task": "classify",
        "model_name": PRETRAINED,
        "pretrained_checkpoint": str(weights),
        "pretrained_sha256": sha256_file(weights),
        "device": device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs_requested": epochs,
        "patience": patience,
        "seed": args.seed,
        "deterministic": True,
        "workers": 0,
        "amp": False,
        "cache": False,
        "optimizer": "auto (Ultralytics default for fine-tuning)",
        "data": str(data_root),
        "dataset": summary,
        "augmentation": {
            "train": "classify_augmentations (RandomResizedCrop scale=(1-scale,1), hflip, randaugment, erasing)",
            "eval": "classify_transforms: resize shortest edge 224 BILINEAR, center crop 224, ToTensor, Normalize(0,1)",
        },
        "started_at": utc_now(),
        "seed_everything": True,
        "packages": package_versions(),
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "device": device,
            "torch_mps": bool(torch.backends.mps.is_available()),
        },
    }
    write_json(out_dir / "training_params.json", params)

    seed_everything(args.seed)
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    results = model.train(
        data=str(data_root),
        epochs=epochs,
        patience=patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        seed=args.seed,
        deterministic=True,
        amp=False,
        cache=False,
        project=str(project),
        name=run_name,
        exist_ok=True,
        plots=False,
        verbose=True,
        val=True,
        save=True,
        save_period=-1,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    run_dir = Path(getattr(results, "save_dir", getattr(model.trainer, "save_dir", project / run_name)))
    results_csv = run_dir / "results.csv"
    history = parse_results_csv(results_csv)
    best_src = run_dir / "weights" / "best.pt"
    last_src = run_dir / "weights" / "last.pt"
    if results_csv.exists():
        (out_dir / "history.csv").write_text(results_csv.read_text())
    write_json(out_dir / "history.json", history)
    if best_src.exists():
        shutil.copy2(best_src, out_dir / "best.pt")
    if last_src.exists():
        shutil.copy2(last_src, out_dir / "last.pt")

    epoch_seconds = []
    for prev, cur in zip(history, history[1:]):
        try:
            epoch_seconds.append(float(cur["time"]) - float(prev["time"]))
        except (KeyError, TypeError, ValueError):
            pass

    best_row = None
    for row in history:
        value = row.get("metrics/accuracy_top1")
        if isinstance(value, (int, float)):
            if best_row is None or float(value) > float(best_row["metrics/accuracy_top1"]):
                best_row = row

    params.update(
        {
            "completed_at": utc_now(),
            "epochs_run": len(history),
            "wall_clock_seconds": elapsed,
            "wall_clock_minutes": round(elapsed / 60.0, 2),
            "mean_epoch_seconds": round(statistics.mean(epoch_seconds), 1) if epoch_seconds else None,
            "best_checkpoint": str(out_dir / "best.pt") if (out_dir / "best.pt").exists() else None,
            "best_epoch_from_csv": int(float(best_row["epoch"])) if best_row else None,
            "best_val_top1_from_csv": float(best_row["metrics/accuracy_top1"]) if best_row else None,
            "final_epoch_val_top1_from_csv": (
                float(history[-1]["metrics/accuracy_top1"])
                if history and isinstance(history[-1].get("metrics/accuracy_top1"), (int, float))
                else None
            ),
            "ultralytics_run_dir": str(run_dir),
            "early_stopped": len(history) < epochs,
            "early_stopping_metric": "Ultralytics classify fitness = mean(top1, top5)",
        }
    )
    write_json(out_dir / "training_params.json", params)
    return params | {"_elapsed_s": elapsed, "_run_dir": str(run_dir)}


class TransformDataset:
    """Minimal dataset returning (tensor, index) with classify_transforms."""

    def __init__(self, paths: Sequence[Path], transform: Any):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        from PIL import Image

        with Image.open(self.paths[idx]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, idx


def load_best_model(best_pt: Path, device: str) -> tuple[Any, list[str]]:
    from ultralytics import YOLO

    wrapper = YOLO(str(best_pt))
    net = wrapper.model
    net.to(device)
    net.eval()
    names = {int(k): v for k, v in wrapper.names.items()}
    classes = [names[i] for i in sorted(names)]
    return net, classes


def predict_rows(
    net: Any,
    rows: Sequence[dict[str, Any]],
    classes: Sequence[str],
    imgsz: int,
    device: str,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from ultralytics.data.augment import classify_transforms

    transform = classify_transforms(size=imgsz)
    paths = [INPUT_ROOT / row["image"] for row in rows]
    loader = DataLoader(
        TransformDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    class_index = {name: i for i, name in enumerate(classes)}
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device)
            out = net(images)
            if isinstance(out, (tuple, list)):
                probs = out[0]
            else:
                probs = out
            probs = probs.float().cpu().numpy()
            if probs.ndim != 2 or probs.shape[1] != len(classes):
                raise RuntimeError(f"unexpected probability shape {probs.shape} for {len(classes)} classes")
            for local, global_idx in enumerate(indices.tolist()):
                row = rows[global_idx]
                p = probs[local]
                pred_idx = int(np.argmax(p))
                records.append(
                    {
                        "image": row["image"],
                        "actual": row["class_name"],
                        "actual_index": class_index[row["class_name"]],
                        "predicted": classes[pred_idx],
                        "predicted_index": pred_idx,
                        "confidence": float(p[pred_idx]),
                        "probs": [float(x) for x in p],
                        "domain": row["domain"],
                        "species": row["species"],
                    }
                )
    return records


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n: int) -> list[list[int]]:
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[int(t)][int(p)] += 1
    return matrix


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
) -> dict[str, Any]:
    n = len(classes)
    cm = confusion_matrix(y_true, y_pred, n)
    per_class: dict[str, Any] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for i, name in enumerate(classes):
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(cm[r][i] for r in range(n))
        fp = predicted - tp
        fn = support - tp
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        per_class[name] = {
            "support": int(support),
            "predicted": int(predicted),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if precision is not None:
            precisions.append(precision)
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1s.append(f1)
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return {
        "n": int(total),
        "correct": int(correct),
        "accuracy": correct / total if total else None,
        "per_class": per_class,
        "macro": {
            "precision": statistics.mean(precisions) if precisions else None,
            "recall": statistics.mean(recalls) if recalls else None,
            "f1": statistics.mean(f1s) if f1s else None,
            "classes_counted": len(f1s),
        },
        "confusion_matrix": {
            "labels": list(classes),
            "rows_actual_columns_predicted": cm,
        },
    }


def records_to_arrays(records: Sequence[dict[str, Any]], classes: Sequence[str]) -> tuple[list[int], list[int], list[float]]:
    idx = {name: i for i, name in enumerate(classes)}
    y_true = [idx[r["actual"]] for r in records]
    y_pred = [idx[r["predicted"]] for r in records]
    conf = [float(r["confidence"]) for r in records]
    return y_true, y_pred, conf


def subset_metrics(records: Sequence[dict[str, Any]], classes: Sequence[str], mask: Sequence[bool]) -> dict[str, Any]:
    subset = [r for r, keep in zip(records, mask) if keep]
    y_true, y_pred, _ = records_to_arrays(subset, classes)
    out = classification_metrics(y_true, y_pred, classes)
    out["domain_counts"] = {
        d: sum(1 for r in subset if r["domain"] == d) for d in DOMAINS
    }
    return out


def per_species_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["species"], []).append(record)
    rows = []
    for species, items in groups.items():
        correct = sum(1 for r in items if r["actual"] == r["predicted"])
        rows.append(
            {
                "species": species,
                "class_name": items[0]["actual"],
                "n": len(items),
                "correct": correct,
                "accuracy": correct / len(items),
                "domains": {d: sum(1 for r in items if r["domain"] == d) for d in DOMAINS},
            }
        )
    rows.sort(key=lambda r: (r["accuracy"], -r["n"], r["species"]))
    accuracies = [r["accuracy"] for r in rows]
    return {
        "n_species": len(rows),
        "macro_species_accuracy": statistics.mean(accuracies) if accuracies else None,
        "min_species_accuracy": min(accuracies) if accuracies else None,
        "max_species_accuracy": max(accuracies) if accuracies else None,
        "species_with_zero_accuracy": [r["species"] for r in rows if r["accuracy"] == 0.0],
        "species_below_0_5": [r["species"] for r in rows if r["accuracy"] < 0.5],
        "species_at_1_0": sum(1 for r in rows if r["accuracy"] == 1.0),
        "support_min": min((r["n"] for r in rows), default=None),
        "support_max": max((r["n"] for r in rows), default=None),
        "rows": rows,
    }


def coverage_curve(records: Sequence[dict[str, Any]], classes: Sequence[str], thresholds: Sequence[float]) -> list[dict[str, Any]]:
    out = []
    total = len(records)
    class_idx = {name: i for i, name in enumerate(classes)}
    y_true = [class_idx[r["actual"]] for r in records]
    y_pred = [class_idx[r["predicted"]] for r in records]
    conf = [float(r["confidence"]) for r in records]
    for t in thresholds:
        covered_idx = [i for i, c in enumerate(conf) if c >= t]
        covered = len(covered_idx)
        correct = sum(1 for i in covered_idx if y_true[i] == y_pred[i])
        per_class: dict[str, Any] = {}
        for name in classes:
            cidx = class_idx[name]
            predicted = [i for i in covered_idx if y_pred[i] == cidx]
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
                "coverage": covered / total if total else None,
                "covered": covered,
                "abstained": total - covered,
                "correct_covered": correct,
                "accuracy_at_coverage": correct / covered if covered else None,
                "precision_at_coverage": correct / covered if covered else None,
                "per_class": per_class,
            }
        )
    return out


def choose_threshold(val_curve: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def eligible(curve: Sequence[dict[str, Any]], target: float) -> list[dict[str, Any]]:
        return [
            row
            for row in curve
            if row["coverage"] is not None
            and row["coverage"] >= SELECTIVE_MIN_COVERAGE
            and row["precision_at_coverage"] is not None
            and row["precision_at_coverage"] >= target
        ]

    candidates = eligible(val_curve, SELECTIVE_TARGET_PRECISION)
    used_target = SELECTIVE_TARGET_PRECISION
    if not candidates:
        used_target = SELECTIVE_FALLBACK_PRECISION
        candidates = eligible(val_curve, used_target)
    if candidates:
        chosen = min(candidates, key=lambda row: row["threshold"])
        rule = "minimum threshold meeting target precision and minimum coverage"
    else:
        chosen = max(
            val_curve,
            key=lambda row: (
                row["precision_at_coverage"] if row["precision_at_coverage"] is not None else -1.0,
                row["coverage"] if row["coverage"] is not None else -1.0,
            ),
        )
        used_target = None
        rule = "no target met; maximum validation precision (ties: higher coverage)"
    return {
        "chosen": chosen,
        "target_precision_used": used_target,
        "selection_rule": rule,
        "min_coverage": SELECTIVE_MIN_COVERAGE,
    }


def metrics_at_threshold(records: Sequence[dict[str, Any]], classes: Sequence[str], threshold: float) -> dict[str, Any]:
    y_true, y_pred, conf = records_to_arrays(records, classes)
    covered = [i for i, c in enumerate(conf) if c >= threshold]
    yt = [y_true[i] for i in covered]
    yp = [y_pred[i] for i in covered]
    out = classification_metrics(yt, yp, classes)
    out.update(
        {
            "threshold": threshold,
            "coverage": len(covered) / len(records) if records else None,
            "covered": len(covered),
            "abstained": len(records) - len(covered),
            "accuracy_at_coverage": out["accuracy"],
            "precision_at_coverage": out["accuracy"],
        }
    )
    class_idx = {name: i for i, name in enumerate(classes)}
    for name in classes:
        cidx = class_idx[name]
        support = sum(1 for r in records if r["actual"] == name)
        tp = sum(1 for i in covered if y_true[i] == cidx and y_pred[i] == cidx)
        out["per_class"][name]["recall_at_coverage"] = tp / support if support else None
        predicted = sum(1 for i in covered if y_pred[i] == cidx)
        tp_pred = sum(1 for i in covered if y_pred[i] == cidx and y_true[i] == cidx)
        out["per_class"][name]["precision_at_coverage"] = tp_pred / predicted if predicted else None
    return out


def coverage_at_targets(
    records: Sequence[dict[str, Any]], classes: Sequence[str], targets: Sequence[float]
) -> list[dict[str, Any]]:
    import numpy as np

    conf = np.array([float(r["confidence"]) for r in records], dtype=float)
    out = []
    for target in targets:
        if target >= 1.0:
            threshold = float(conf.min())
        else:
            threshold = float(np.quantile(conf, 1.0 - target))
        row = metrics_at_threshold(records, classes, threshold)
        row["target_coverage"] = target
        out.append(row)
    return out


def describe_onnx(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    inputs = []
    for inp in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        inputs.append(
            {
                "name": inp.name,
                "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
                "shape": dims,
            }
        )
    outputs = []
    for out in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        outputs.append(
            {
                "name": out.name,
                "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
                "shape": dims,
            }
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in model.opset_import],
        "ir_version": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "inputs": inputs,
        "outputs": outputs,
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
    }


def export_onnx(best_pt: Path, out_dir: Path, imgsz: int, classes: Sequence[str]) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    produced = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=17,
            dynamic=False,
            simplify=False,
            batch=1,
            device="cpu",
            half=False,
        )
    )
    final = out_dir / "leaf_specialist.onnx"
    if not produced.is_file():
        raise RuntimeError(f"ONNX export did not produce a file (reported {produced})")
    if produced.resolve() != final.resolve():
        shutil.copy2(produced, final)
        if produced.parent == best_pt.parent and produced.suffix == ".onnx":
            produced.unlink(missing_ok=True)

    import onnx

    graph = onnx.load(str(final))
    onnx.checker.check_model(graph)
    info = describe_onnx(final)
    expected_input = [1, 3, imgsz, imgsz]
    if info["inputs"][0]["shape"] != expected_input:
        raise RuntimeError(f"unexpected ONNX input shape {info['inputs'][0]['shape']} != {expected_input}")
    if info["outputs"][0]["shape"] != [1, len(classes)]:
        raise RuntimeError(f"unexpected ONNX output shape {info['outputs'][0]['shape']} != [1, {len(classes)}]")
    opsets = {row["version"] for row in info["opset"] if row["domain"] in {"ai.onnx", ""}}
    if opsets != {17}:
        raise RuntimeError(f"unexpected ONNX opset {info['opset']}")

    preprocessing = {
        "reference": "ultralytics.data.augment.classify_transforms(size=%d)" % imgsz,
        "ultralytics_version": package_versions()["ultralytics"],
        "input_color": "RGB",
        "resize": {"shortest_edge": imgsz, "interpolation": "BILINEAR", "aspect_ratio": "preserved"},
        "center_crop": [imgsz, imgsz],
        "to_tensor": "PIL uint8 [0,255] -> float32 [0,1], divided by 255; NCHW",
        "normalize": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "batch": 1,
        "notes": [
            "No letterbox; torchvision Resize(int) scales the shortest edge and CenterCrop removes the excess.",
            "Images whose shortest edge is below the target are upscaled.",
            "The normalisation is NOT ImageNet mean/std; Ultralytics 8.4 classify_transforms defaults to mean 0 and std 1.",
            "The ONNX graph outputs softmax probabilities (rows sum to 1), not logits.",
        ],
    }
    labels = {
        "dataset": "leafsnap-yolo26 classification (E4 leaf specialist)",
        "task": "image_classification",
        "model": "yolo26n-cls fine-tuned (2 classes)",
        "class_order": list(classes),
        "class_ids": {name: i for i, name in enumerate(classes)},
        "dataset_class_ids": {"compound_leaf": 1, "simple_leaf": 0},
        "class_order_note": (
            "class_order is the model/ImageFolder order (alphabetical: compound_leaf, simple_leaf). "
            "It differs from the input dataset labels.json, where simple_leaf=0 and compound_leaf=1. "
            "Consumers must use this file's class_order for ONNX argmax indices."
        ),
        "nc": len(classes),
        "input": info["inputs"][0],
        "output": {
            **info["outputs"][0],
            "layout": "[batch, classes] softmax probabilities; argmax index maps to class_order",
        },
        "preprocessing": preprocessing,
        "onnx": {
            "opset": info["opset"],
            "ir_version": info["ir_version"],
            "producer": info["producer"],
            "dynamic": False,
            "batch": 1,
            "imgsz": imgsz,
            "file": "leaf_specialist.onnx",
            "sha256": info["sha256"],
            "bytes": info["bytes"],
        },
        "provenance": {
            "training_manifest_sha256": sha256_file(MANIFEST),
            "input_labels_sha256": sha256_file(LABELS),
            "source": "LeafSnap crops; exact Dirr leaf_type labels; species-disjoint splits",
        },
    }
    write_json(out_dir / "leaf_specialist.labels.json", labels)
    return {"info": info, "labels": labels, "preprocessing": preprocessing}


def verify_onnx(
    onnx_path: Path,
    rows: Sequence[dict[str, Any]],
    classes: Sequence[str],
    torch_records: Sequence[dict[str, Any]],
    imgsz: int,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    from ultralytics.data.augment import classify_transforms

    graph = describe_onnx(onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    transform = classify_transforms(size=imgsz)
    class_index = {name: i for i, name in enumerate(classes)}
    by_image = {r["image"]: r for r in torch_records}

    agreement = 0
    correct = 0
    max_diff = 0.0
    mean_diffs: list[float] = []
    confusion = [[0 for _ in classes] for _ in classes]
    rows_out: list[dict[str, Any]] = []
    from PIL import Image

    for row in rows:
        with Image.open(INPUT_ROOT / row["image"]) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        output = session.run([output_meta.name], {input_meta.name: tensor})[0]
        probs = np.asarray(output, dtype=np.float32).reshape(-1)
        pred_idx = int(np.argmax(probs))
        actual_idx = class_index[row["class_name"]]
        correct += pred_idx == actual_idx
        confusion[actual_idx][pred_idx] += 1
        torch_row = by_image[row["image"]]
        agreement += pred_idx == int(torch_row["predicted_index"])
        diff = float(np.max(np.abs(probs - np.asarray(torch_row["probs"], dtype=np.float32))))
        max_diff = max(max_diff, diff)
        mean_diffs.append(diff)
        rows_out.append(
            {
                "image": row["image"],
                "actual": row["class_name"],
                "predicted": classes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "torch_predicted": torch_row["predicted"],
                "prob_sum": float(probs.sum()),
                "max_abs_prob_diff_vs_torch": diff,
            }
        )

    n = len(rows)
    per_class = {}
    for i, name in enumerate(classes):
        support = sum(confusion[i])
        tp = confusion[i][i]
        predicted = sum(confusion[r][i] for r in range(len(classes)))
        per_class[name] = {
            "support": support,
            "recall": tp / support if support else None,
            "precision": tp / predicted if predicted else None,
        }
    payload = {
        "runtime": "onnxruntime",
        "runtime_version": ort.__version__,
        "providers": session.get_providers(),
        "model": str(onnx_path),
        "graph": graph,
        "test_image_count": n,
        "top1_accuracy": correct / n if n else None,
        "pytorch_prediction_agreement": agreement / n if n else None,
        "pytorch_prediction_agreement_count": agreement,
        "max_abs_prob_diff_vs_torch": max_diff,
        "mean_abs_prob_diff_vs_torch": float(np.mean(mean_diffs)) if mean_diffs else None,
        "prob_sum_min": min((r["prob_sum"] for r in rows_out), default=None),
        "prob_sum_max": max((r["prob_sum"] for r in rows_out), default=None),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(classes),
            "rows_actual_columns_predicted": confusion,
        },
        "examples": rows_out[:6],
    }
    write_json(onnx_path.parent / "onnx_verification.json", payload)
    return payload


def e4_result(test: dict[str, Any], test_records: Sequence[dict[str, Any]], classes: Sequence[str]) -> dict[str, Any]:
    simple = test["per_class"]["simple_leaf"]["recall"]
    compound = test["per_class"]["compound_leaf"]["recall"]
    field_records = [r for r in test_records if r["domain"] == "field"]
    y_true, y_pred, _ = records_to_arrays(field_records, classes)
    compound_idx = list(classes).index("compound_leaf")
    compound_support = sum(1 for t in y_true if t == compound_idx)
    compound_correct = sum(1 for t, p in zip(y_true, y_pred) if t == compound_idx and p == compound_idx)
    compound_field_recall = compound_correct / compound_support if compound_support else None

    checks = [
        {
            "name": "test_top1",
            "description": "test top1 accuracy (full coverage)",
            "value": test["accuracy"],
            "threshold": E4_MIN_TOP1,
            "comparison": ">=",
            "pass": test["accuracy"] is not None and test["accuracy"] >= E4_MIN_TOP1,
        },
        {
            "name": "simple_leaf_recall",
            "description": "simple_leaf recall on test",
            "value": simple,
            "threshold": E4_MIN_SIMPLE_RECALL,
            "comparison": ">=",
            "pass": simple is not None and simple >= E4_MIN_SIMPLE_RECALL,
        },
        {
            "name": "compound_leaf_recall",
            "description": "compound_leaf recall on test",
            "value": compound,
            "threshold": E4_MIN_COMPOUND_RECALL,
            "comparison": ">=",
            "pass": compound is not None and compound >= E4_MIN_COMPOUND_RECALL,
        },
        {
            "name": "compound_leaf_field_recall",
            "description": "compound_leaf recall on FIELD test images",
            "value": compound_field_recall,
            "threshold": E4_MIN_COMPOUND_FIELD_RECALL,
            "comparison": ">=",
            "pass": compound_field_recall is not None
            and compound_field_recall >= E4_MIN_COMPOUND_FIELD_RECALL,
            "support": compound_support,
            "correct": compound_correct,
        },
    ]
    return {
        "passed": all(c["pass"] for c in checks),
        "checks": checks,
        "compound_field_support": compound_support,
        "compound_field_correct": compound_correct,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def write_report(metrics: dict[str, Any], out_dir: Path) -> None:
    data = metrics["data"]
    training = metrics["training"]
    test = metrics["evaluation"]["test"]
    val = metrics["evaluation"]["val"]
    e4 = metrics["e4"]
    coverage = metrics["evaluation"]["coverage"]
    onnx_verification = metrics["onnx"]["verification"]

    lines = [
        f"# {metrics['experiment']}",
        "",
        f"Generated {metrics['generated_at']} by `{metrics['source']}`.",
        "",
        "This is one proof run, not a sweep. E3 (organ gate) passed before this "
        "lower-level training was authorized; no organ-gate artifact was modified.",
        "",
        "## E4 criteria (pre-registered before results)",
        "",
        f"Registered {metrics['criteria']['registered_at']} in "
        "`EVALUATION_CRITERIA.md`. All four must hold on the held-out test split "
        "at argmax (full coverage).",
        "",
        "| criterion | threshold | observed | result |",
        "|---|---:|---:|---|",
    ]
    for check in e4["checks"]:
        lines.append(
            f"| {check['description'] if 'description' in check else check['name']} | "
            f"{check['comparison']} {check['threshold']:.2f} | {fmt(check['value'])} | "
            f"{'PASS' if check['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"**E4 overall: {'PASS' if e4['passed'] else 'FAIL'}.** "
        "The compound-field criterion is evaluated on field-domain test images "
        f"only (n={e4['compound_field_support']}); it is not diluted by lab or "
        "overall accuracy.",
        "",
        "## Training",
        "",
        "| item | value |",
        "|---|---|",
        f"| base model | `{training['model_name']}` |",
        f"| device | `{training['device']}` |",
        f"| imgsz / batch | {training['imgsz']} / {training['batch']} |",
        f"| epochs run / requested | {training['epochs_run']} / {training['epochs_requested']} "
        f"(patience {training['patience']}) |",
        f"| seed / workers | {training['seed']} / 0 (deterministic, amp off) |",
        f"| wall clock | {training['wall_clock_minutes']:.2f} min |",
        f"| mean epoch | {fmt(training['mean_epoch_seconds'], 1)} s |",
        f"| early stopped | {fmt(training['early_stopped'])} |",
        f"| best epoch (val top1) | {training['best_epoch_from_csv']} "
        f"({fmt(training['best_val_top1_from_csv'])}) |",
        f"| train crops | {data['by_split_class']['train']['simple_leaf'] + data['by_split_class']['train']['compound_leaf']} |",
        "",
        "Determinism: fixed seed, `deterministic=True`, workers 0, AMP off. "
        "MPS kernels are not bitwise deterministic across runs; the proof run is "
        "seeded and reproducible in configuration, not bit-identical.",
        "",
        "## Data provenance and counts",
        "",
        f"- Input: `{data['input_root']}` (LeafSnap crops, exact Dirr `leaf_type` "
        "Simple/Compound only; mixed/missing excluded upstream).",
        f"- Label source: {data['label_source']}.",
        f"- Crop rule: {data['crop']['bbox']}; margin {data['crop']['margin_fraction']} "
        f"({data['crop']['margin_rule']}).",
        f"- Splits are species-disjoint: {fmt(data['species_split_disjoint'])}.",
        f"- `manifest.jsonl` sha256: `{data['files']['manifest.jsonl']['sha256'][:16]}…`; "
        f"`labels.json` sha256: `{data['files']['labels.json']['sha256'][:16]}…`; "
        f"`counts.json` sha256: `{data['files']['counts.json']['sha256'][:16]}…`.",
        "",
        "| split | simple_leaf | compound_leaf | field | lab | total | species |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        sc = data["by_split_class"][split]
        sd = data["by_split_domain"][split]
        total = sum(sc.values())
        lines.append(
            f"| {split} | {sc['simple_leaf']} | {sc['compound_leaf']} | "
            f"{sd['field']} | {sd['lab']} | {total} | {len(data['species_by_split'][split])} |"
        )
    lines += [
        "",
        "Class imbalance is substantial: test is "
        f"{pct(data['by_split_class']['test']['simple_leaf'] / sum(data['by_split_class']['test'].values()))} "
        "simple. A no-skill majority baseline would score "
        f"{pct(test['majority_baseline'])}. Overall accuracy is therefore not "
        "evidence that compound leaves are handled; compound recall is reported "
        "separately.",
        "",
        "## Held-out test results",
        "",
        f"**Top1: {pct(test['accuracy'])} ({test['correct']}/{test['n']}).**",
        f"Macro precision / recall / F1: {fmt(test['macro']['precision'])} / "
        f"{fmt(test['macro']['recall'])} / {fmt(test['macro']['f1'])}.",
        "",
        "| class | support | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in test["confusion_matrix"]["labels"]:
        pc = test["per_class"][name]
        lines.append(
            f"| {name} | {pc['support']} | {fmt(pc['precision'])} | "
            f"{fmt(pc['recall'])} | {fmt(pc['f1'])} |"
        )
    lines += [
        "",
        "Confusion matrix (rows actual, columns predicted):",
        "",
        "| actual \\ predicted | "
        + " | ".join(test["confusion_matrix"]["labels"])
        + " |",
        "|---|" + "---:|" * len(test["confusion_matrix"]["labels"]),
    ]
    for i, name in enumerate(test["confusion_matrix"]["labels"]):
        row = test["confusion_matrix"]["rows_actual_columns_predicted"][i]
        lines.append("| " + name + " | " + " | ".join(str(x) for x in row) + " |")
    lines += [
        "",
        "### Field-only and lab-only breakdowns",
        "",
        "| domain | n | top1 | simple recall | compound recall | macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        block = metrics["evaluation"][domain]
        lines.append(
            f"| {domain} | {block['n']} | {pct(block['accuracy'])} | "
            f"{fmt(block['per_class']['simple_leaf']['recall'])} | "
            f"{fmt(block['per_class']['compound_leaf']['recall'])} | "
            f"{fmt(block['macro']['f1'])} |"
        )
    lines += [
        "",
        f"Compound-leaf FIELD recall is **{fmt(e4['checks'][3]['value'])}** "
        f"({e4['compound_field_correct']}/{e4['compound_field_support']}), "
        f"the E4 threshold is {E4_MIN_COMPOUND_FIELD_RECALL:.2f}. "
        f"Compound-leaf LAB recall is "
        f"{fmt(metrics['evaluation']['lab']['per_class']['compound_leaf']['recall'])} "
        "and is shown only as context.",
        "",
        f"Overall compound recall is {e4['checks'][2]['value']:.4f} "
        f"({test['per_class']['compound_leaf']['tp']}/{test['per_class']['compound_leaf']['support']}), "
        f"missing the {E4_MIN_COMPOUND_RECALL:.2f} gate by "
        f"{E4_MIN_COMPOUND_RECALL - e4['checks'][2]['value']:.4f}. "
        f"Compound field recall misses {E4_MIN_COMPOUND_FIELD_RECALL:.2f} by "
        f"{E4_MIN_COMPOUND_FIELD_RECALL - e4['checks'][3]['value']:.4f} "
        f"({e4['compound_field_correct']}/{e4['compound_field_support']}; "
        f"{max(0, math.ceil(E4_MIN_COMPOUND_FIELD_RECALL * e4['compound_field_support']) - e4['compound_field_correct'])} more correct field compound crop(s) required).",
        "",
        "### Per-species accuracy (test, species-disjoint)",
        "",
        f"{test['per_species']['n_species']} species; macro species accuracy "
        f"{pct(test['per_species']['macro_species_accuracy'])}, range "
        f"{pct(test['per_species']['min_species_accuracy'])}–"
        f"{pct(test['per_species']['max_species_accuracy'])}; "
        f"{test['per_species']['species_at_1_0']} species at 100%, "
        f"{len(test['per_species']['species_below_0_5'])} below 50%, "
        f"{len(test['per_species']['species_with_zero_accuracy'])} at 0%.",
        "",
        "| species | class | n | field/lab | accuracy |",
        "|---|---|---:|---|---:|",
    ]
    for row in test["per_species"]["rows"]:
        lines.append(
            f"| {row['species']} | {row['class_name']} | {row['n']} | "
            f"{row['domains']['field']}/{row['domains']['lab']} | {pct(row['accuracy'])} |"
        )
    compound_rows = [r for r in test["per_species"]["rows"] if r["class_name"] == "compound_leaf"]
    simple_rows = [r for r in test["per_species"]["rows"] if r["class_name"] == "simple_leaf"]
    compound_names = ", ".join(f"`{r['species']}`" for r in compound_rows)
    lines += [
        "",
        "Compound test species: "
        + ", ".join(f"{r['species']} {pct(r['accuracy'])} (n={r['n']})" for r in compound_rows)
        + ". Simple test species worst rows: "
        + ", ".join(f"{r['species']} {pct(r['accuracy'])} (n={r['n']})" for r in simple_rows[:3])
        + ". The compound shortfall is concentrated in the largest compound test species, "
        "not spread evenly across the class.",
        "",
        "Species rows are sorted worst-first. Species-level labels are Dirr "
        "assertions applied to every crop of that species; a low per-species "
        "accuracy can reflect within-species image variability or a label edge "
        "case, not necessarily a classifier error. Compound test species "
        f"({len(compound_rows)}): {compound_names}; each species contributes its "
        "own support, so class recall should be interpreted with the per-species "
        "errors above.",
        "",
        "## Selective accuracy/precision at coverage",
        "",
        f"The confidence threshold was selected on **validation only** and then "
        f"applied unchanged to test. Chosen threshold **{coverage['chosen']['threshold']:.4f}** "
        f"({coverage['selection']['selection_rule']}; validation target "
        f"{fmt(coverage['selection']['target_precision_used'])}).",
        "",
        "| split | threshold | coverage | abstained | accuracy@coverage | precision@coverage |",
        "|---|---:|---:|---:|---:|---:|",
        f"| val (selection) | {coverage['chosen']['threshold']:.4f} | "
        f"{pct(coverage['chosen']['coverage'])} | {coverage['chosen']['abstained']} | "
        f"{pct(coverage['chosen']['accuracy_at_coverage'])} | "
        f"{pct(coverage['chosen']['precision_at_coverage'])} |",
        f"| test (applied) | {coverage['test_at_chosen']['threshold']:.4f} | "
        f"{pct(coverage['test_at_chosen']['coverage'])} | {coverage['test_at_chosen']['abstained']} | "
        f"{pct(coverage['test_at_chosen']['accuracy_at_coverage'])} | "
        f"{pct(coverage['test_at_chosen']['precision_at_coverage'])} |",
        "",
        f"Validation produced {coverage['chosen']['correct']}/{coverage['chosen']['n']} correct "
        f"at the selected threshold ({pct(coverage['chosen']['coverage'])} coverage). "
        + (
            "Validation was saturated, so the pre-registered rule bottomed out at the "
            "minimum confidence 0.50 and no validation evidence justified abstention. "
            "The test-only coverage targets below are a descriptive runtime tradeoff, "
            "not a selection input."
            if coverage['chosen']['accuracy_at_coverage'] == 1.0 and coverage['chosen']['coverage'] == 1.0
            else "The selected threshold is the lowest one meeting the pre-registered "
            "precision/coverage rule; test coverage targets below are descriptive."
        ),
        "",
        "Per-class selective recall/precision at the test operating point:",
        "",
        "| class | recall@coverage | precision@coverage |",
        "|---|---:|---:|",
    ]
    for name in test["confusion_matrix"]["labels"]:
        pc = coverage["test_at_chosen"]["per_class"][name]
        lines.append(
            f"| {name} | {fmt(pc['recall_at_coverage'])} | "
            f"{fmt(pc['precision_at_coverage'])} |"
        )
    lines += [
        "",
        "Descriptive test-only coverage targets (threshold is the test confidence "
        "quantile; NOT used to select the operating point):",
        "",
        "| target coverage | threshold | actual coverage | accuracy@coverage |",
        "|---|---:|---:|---:|",
    ]
    for row in coverage["test_at_targets"]:
        lines.append(
            f"| {pct(row['target_coverage'])} | {row['threshold']:.4f} | "
            f"{pct(row['coverage'])} | {pct(row['accuracy_at_coverage'])} |"
        )
    lines += [
        "",
        "## ONNX export and verification",
        "",
        f"- File: `{metrics['onnx']['path']}` "
        f"({metrics['onnx']['info']['bytes']} bytes, opset 17, static, batch 1).",
        f"- Input: `{metrics['onnx']['info']['inputs'][0]['name']}` "
        f"{metrics['onnx']['info']['inputs'][0]['shape']} "
        f"{metrics['onnx']['info']['inputs'][0]['dtype']}.",
        f"- Output: `{metrics['onnx']['info']['outputs'][0]['name']}` "
        f"{metrics['onnx']['info']['outputs'][0]['shape']} "
        f"{metrics['onnx']['info']['outputs'][0]['dtype']} (softmax probabilities).",
        f"- Labels/preprocessing: `leaf_specialist.labels.json`.",
        f"- Verified on all {onnx_verification['test_image_count']} test crops with "
        f"onnxruntime (CPU): top1 {pct(onnx_verification['top1_accuracy'])}, "
        f"PyTorch top1 {pct(test['accuracy'])}, prediction agreement "
        f"{pct(onnx_verification['pytorch_prediction_agreement'])}, "
        f"max |prob diff| {onnx_verification['max_abs_prob_diff_vs_torch']:.2e}.",
        f"- Probability row sums: {onnx_verification['prob_sum_min']:.6f}–"
        f"{onnx_verification['prob_sum_max']:.6f}.",
        "",
        "Preprocessing must match `classify_transforms(size=224)`: RGB, resize "
        "the shortest edge to 224 with bilinear interpolation, center-crop "
        "224x224, scale to [0,1] (ToTensor), and normalize with mean 0 / std 1. "
        "There is no letterbox and no ImageNet mean/std. The ONNX output is "
        "already softmax; do not apply softmax again.",
        "",
        "Class order is **model order** (`compound_leaf`, `simple_leaf`), which "
        "differs from the input dataset's labels.json order "
        "(`simple_leaf`=0, `compound_leaf`=1). Use "
        "`leaf_specialist.labels.json` for indices.",
        "",
        "## Caveats",
        "",
    ]
    lines += [f"- {caveat}" for caveat in metrics["caveats"]]
    lines += [
        "",
        "## Paths",
        "",
        f"- Source: `{metrics['source']}`",
        f"- Output dir: `{out_dir}`",
        f"- Best checkpoint: `{training['best_checkpoint']}`",
        f"- ONNX: `{metrics['onnx']['path']}`",
        f"- Labels: `{metrics['onnx']['labels_path']}`",
        f"- Metrics: `{out_dir / 'metrics.json'}`",
        "",
    ]
    (out_dir / "LEAF_SPECIALIST.md").write_text("\n".join(lines))


def majority_baseline(records: Sequence[dict[str, Any]]) -> float:
    counts: dict[str, int] = {}
    for r in records:
        counts[r["actual"]] = counts.get(r["actual"], 0) + 1
    total = len(records)
    return max(counts.values()) / total if total else 0.0


def build_metrics(
    args: argparse.Namespace,
    out_dir: Path,
    criteria: dict[str, Any],
    training: dict[str, Any],
    summary: dict[str, Any],
    val_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    classes: Sequence[str],
    onnx_payload: dict[str, Any],
    onnx_verification: dict[str, Any],
) -> dict[str, Any]:
    training = {k: v for k, v in training.items() if not k.startswith("_")}
    yt_val, yp_val, _ = records_to_arrays(val_records, classes)
    yt_test, yp_test, _ = records_to_arrays(test_records, classes)
    val_metrics = classification_metrics(yt_val, yp_val, classes)
    test_metrics = classification_metrics(yt_test, yp_test, classes)
    test_metrics["majority_baseline"] = majority_baseline(test_records)
    test_metrics["per_species"] = per_species_metrics(test_records)

    field_metrics = subset_metrics(test_records, classes, [r["domain"] == "field" for r in test_records])
    lab_metrics = subset_metrics(test_records, classes, [r["domain"] == "lab" for r in test_records])

    thresholds = [round(x, 2) for x in [0.50 + 0.01 * i for i in range(50)]] + [0.995, 0.999]
    val_curve = coverage_curve(val_records, classes, thresholds)
    test_curve = coverage_curve(test_records, classes, thresholds)
    selection = choose_threshold(val_curve)
    chosen = selection["chosen"]
    test_at_chosen = metrics_at_threshold(test_records, classes, chosen["threshold"])
    val_at_chosen = metrics_at_threshold(val_records, classes, chosen["threshold"])
    coverage = {
        "selection": selection,
        "chosen": val_at_chosen,
        "val_curve": val_curve,
        "test_curve": test_curve,
        "test_at_chosen": test_at_chosen,
        "test_at_targets": coverage_at_targets(test_records, classes, [1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]),
    }

    e4 = e4_result(test_metrics, test_records, classes)
    e4["chosen_scope"] = "full test split at argmax (no abstention)"
    test_compound = test_metrics["per_class"]["compound_leaf"]["support"]
    test_simple = test_metrics["per_class"]["simple_leaf"]["support"]
    compound_field_records = [
        r for r in test_records if r["actual"] == "compound_leaf" and r["domain"] == "field"
    ]
    compound_field_species = sorted({r["species"] for r in compound_field_records})

    metrics = {
        "experiment": "E4 — LeafSnap leaf-type specialist (YOLO26n-cls proof run)",
        "generated_at": utc_now(),
        "source": str(HERE / "train_leaf_specialist.py"),
        "criteria": criteria,
        "data": summary,
        "training": training,
        "evaluation": {
            "val": val_metrics,
            "test": test_metrics,
            "field": field_metrics,
            "lab": lab_metrics,
            "coverage": coverage,
        },
        "e4": e4,
        "onnx": {
            "path": str(out_dir / "leaf_specialist.onnx"),
            "labels_path": str(out_dir / "leaf_specialist.labels.json"),
            "info": onnx_payload["info"],
            "labels": onnx_payload["labels"],
            "preprocessing": onnx_payload["preprocessing"],
            "verification": onnx_verification,
        },
        "caveats": [
            "Labels are exact Dirr leaf_type assertions at the species level, not per-image adjudications. Every crop of a species inherits that species' Dirr leaf_type; within-species image variability or a label edge case can appear as a classifier error.",
            "Mixed or missing leaf_type species were excluded upstream. This dataset does not represent mixed-species leaf clusters or species without an exact Dirr Simple/Compound assertion.",
            "Crops are derived from segmentation masks: the foreground bbox plus a 10% margin. They are tight leaf crops, not raw full-frame field photos; performance may not transfer to unsegmented phone images without the organ gate/segmentation upstream.",
            "The task is only simple vs compound leaf type. The model makes no claim about leaf margin, arrangement, venation, or other features.",
            "Splits are species-disjoint, so test is on unseen species; this is harder than a random-image split and is the intended generalization test.",
            f"The test set is imbalanced ({test_simple:,} simple vs {test_compound:,} compound). Overall top1 is dominated by simple leaves; compound recall and compound field recall are the load-bearing numbers and cannot be replaced by overall accuracy.",
            f"Compound field test coverage is {len(compound_field_records):,} images across {len(compound_field_species)} species ({', '.join(compound_field_species)}), so the compound-field estimate has wide species-level uncertainty.",
            "The confidence threshold is selected on validation only; the test coverage curves at target coverages are descriptive and were not used to choose the operating point.",
            "ONNX is FP32 (opset 17, static 224x224, batch 1). Preprocessing must use Ultralytics classify_transforms semantics: shortest-edge resize, center crop, [0,1] scaling, mean 0 / std 1; no letterbox, no ImageNet normalization.",
            "Model/ImageFolder class order is alphabetical (compound_leaf=0, simple_leaf=1), which differs from the input labels.json order; consumers must use leaf_specialist.labels.json.",
            "MPS training is seeded and configured deterministically but is not guaranteed bitwise reproducible across runs/hardware.",
            "For large pinnate leaves (notably Juglans cinerea), a mask-derived crop can isolate a single leaflet, which looks simple. For example, the held-out crop test/compound_leaf/leafsnap_1735499.jpg visibly contains one simple-looking leaflet, but inherits the compound species label. The species-level Dirr leaf_type label describes the species, not necessarily what is visible in the crop; the low Juglans cinerea accuracy is consistent with this crop-level ambiguity.",
        ],
    }
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cpu", action="store_true", help="force CPU instead of MPS")
    parser.add_argument("--smoke", action="store_true", help="tiny 1-epoch device/format smoke run")
    parser.add_argument("--skip-train", action="store_true", help="reuse an existing best.pt")
    parser.add_argument("--skip-export", action="store_true", help="reuse an existing leaf_specialist.onnx")
    parser.add_argument("--report-only", action="store_true", help="rewrite LEAF_SPECIALIST.md from metrics.json")
    args = parser.parse_args(argv)

    out_dir = (OUT_ROOT / "smoke") if args.smoke else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        metrics = json.loads((out_dir / "metrics.json").read_text())
        write_report(metrics, out_dir)
        print(f"rewrote {out_dir / 'LEAF_SPECIALIST.md'}")
        return 0

    rows = load_manifest()
    labels = json.loads(LABELS.read_text())
    counts = json.loads(COUNTS.read_text())
    summary = preflight(rows, labels, counts)

    criteria = write_criteria(out_dir)
    print(f"preflight OK: {len(rows)} crops; classes={summary['class_names']}; "
          f"species-disjoint={summary['species_split_disjoint']}")

    if args.skip_train:
        params_path = out_dir / "training_params.json"
        if not params_path.exists():
            raise FileNotFoundError(f"{params_path} not found for --skip-train")
        training = json.loads(params_path.read_text())
        best_pt = out_dir / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"{best_pt} not found for --skip-train")
    else:
        training = train(args, out_dir, summary)
        best_pt = out_dir / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"training finished without {best_pt}")

    import torch

    device = "mps" if torch.backends.mps.is_available() and not args.cpu and not args.smoke else "cpu"
    net, classes = load_best_model(best_pt, device)
    print(f"model classes: {classes} (device {device})")

    if args.smoke:
        def take_per_class(split: str, per_class: int = 30) -> list[dict[str, Any]]:
            picked: list[dict[str, Any]] = []
            for class_name in classes:
                picked.extend([r for r in rows if r["split"] == split and r["class_name"] == class_name][:per_class])
            return picked

        val_rows = take_per_class("val")
        test_rows = take_per_class("test")
    else:
        val_rows = [r for r in rows if r["split"] == "val"]
        test_rows = [r for r in rows if r["split"] == "test"]

    print(f"evaluating {len(val_rows)} val / {len(test_rows)} test crops ...")
    val_records = predict_rows(net, val_rows, classes, args.imgsz, device)
    test_records = predict_rows(net, test_rows, classes, args.imgsz, device)
    write_json(out_dir / "predictions_val_summary.json", {
        "n": len(val_records),
        "classes": classes,
        "top1": sum(1 for r in val_records if r["actual"] == r["predicted"]) / len(val_records),
    })
    write_json(out_dir / "predictions_test_summary.json", {
        "n": len(test_records),
        "classes": classes,
        "top1": sum(1 for r in test_records if r["actual"] == r["predicted"]) / len(test_records),
    })
    (out_dir / "predictions_val.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in val_records) + "\n"
    )
    (out_dir / "predictions_test.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in test_records) + "\n"
    )

    if args.skip_export and (out_dir / "leaf_specialist.onnx").exists():
        info = describe_onnx(out_dir / "leaf_specialist.onnx")
        labels_payload = json.loads((out_dir / "leaf_specialist.labels.json").read_text())
        onnx_payload = {
            "info": info,
            "labels": labels_payload,
            "preprocessing": labels_payload["preprocessing"],
        }
    else:
        print("exporting ONNX ...")
        onnx_payload = export_onnx(best_pt, out_dir, args.imgsz, classes)

    print(f"verifying ONNX on {len(test_records)} test crops ...")
    onnx_verification = verify_onnx(
        out_dir / "leaf_specialist.onnx",
        [r for r in test_rows],
        classes,
        test_records,
        args.imgsz,
    )

    metrics = build_metrics(
        args, out_dir, criteria, training, summary, val_records, test_records, classes,
        onnx_payload, onnx_verification,
    )
    write_json(out_dir / "metrics.json", metrics)
    write_report(metrics, out_dir)

    e4 = metrics["e4"]
    print(json.dumps({
        "e4_passed": e4["passed"],
        "checks": [{"name": c["name"], "value": c["value"], "threshold": c["threshold"], "pass": c["pass"]} for c in e4["checks"]],
        "test_top1": metrics["evaluation"]["test"]["accuracy"],
        "confusion_matrix": metrics["evaluation"]["test"]["confusion_matrix"],
        "field": {"n": metrics["evaluation"]["field"]["n"], "top1": metrics["evaluation"]["field"]["accuracy"]},
        "lab": {"n": metrics["evaluation"]["lab"]["n"], "top1": metrics["evaluation"]["lab"]["accuracy"]},
        "coverage": {
            "threshold": metrics["evaluation"]["coverage"]["chosen"]["threshold"],
            "test_coverage": metrics["evaluation"]["coverage"]["test_at_chosen"]["coverage"],
            "test_accuracy_at_coverage": metrics["evaluation"]["coverage"]["test_at_chosen"]["accuracy_at_coverage"],
        },
        "onnx": {
            "path": metrics["onnx"]["path"],
            "top1": onnx_verification["top1_accuracy"],
            "agreement": onnx_verification["pytorch_prediction_agreement"],
        },
        "training_minutes": training.get("wall_clock_minutes"),
        "output_dir": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
