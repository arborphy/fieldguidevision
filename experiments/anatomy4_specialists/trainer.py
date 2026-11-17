# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Conditioned expert trainer (YOLO26n-cls) with CUDA and bounded memory.

This is the implementation, not a training run. It reuses the proven
``experiments/leafsnap-yolo26/train_leaf_specialist.py`` pattern (deterministic
fine-tune of ``yolo26n-cls.pt``, fixed seed, per-class metrics, confusion
matrix, validation-only abstain threshold) and adds the two things the .245
host needs:

* **CUDA support** — ``--device auto`` prefers CUDA, then MPS, then CPU.
* **bounded memory** — a hard per-process VRAM fraction (default 0.30 of the
  RTX 4060 Ti's 8 GB, i.e. ~2.4 GB, leaving room for the ~4.3 GB BioCLIP
  sidecar), fixed batch/imgsz, disk cache, capped workers, AMP on CUDA, and a
  preflight that requires free VRAM >= budget + margin. Nothing here restarts
  or contacts a service.

Run ``train`` only after the dataset validator passes. The trainer refuses to
overwrite a completed run directory and never emits metrics without a
checkpoint.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import TargetSpec, load_run_config
from .device import (
    DEFAULT_MAX_VRAM_FRACTION,
    DEFAULT_MIN_FREE_VRAM_GB,
    bounded_memory_env,
    configure_memory,
)
from .evaluate import evaluate_run

TRAIN_DEFAULTS: dict[str, Any] = {
    "model": "yolo26n-cls.pt",
    "epochs": 30,
    "patience": 8,
    "imgsz": 224,
    "batch": 32,
    "workers": 2,
    "seed": 26,
    "cache": "disk",
    "amp": True,
    "device": "auto",
    "max_vram_fraction": DEFAULT_MAX_VRAM_FRACTION,
    "min_free_vram_gb": DEFAULT_MIN_FREE_VRAM_GB,
    "target_precision": 0.95,
    "min_coverage": 0.30,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "ultralytics", "onnx", "onnxruntime"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "missing"
    return versions


def training_args(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the full training argument dict (pure; no torch import)."""

    raw = config.get("raw", config)
    block = dict(TRAIN_DEFAULTS)
    block.update(raw.get("training", {}))
    block.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if block["batch"] <= 0:
        raise ValueError("batch must be a positive integer (auto-batch is intentionally disabled)")
    if block["imgsz"] < 64:
        raise ValueError("imgsz must be >= 64")
    fraction = float(block["max_vram_fraction"])
    if not 0.0 < fraction <= 1.0:
        raise ValueError("max_vram_fraction must be in (0, 1]")
    cache = block["cache"]
    if isinstance(cache, str):
        lowered = cache.strip().lower()
        cache = False if lowered in {"false", "none", "off", "0"} else (
            True if lowered in {"true", "on", "1"} else lowered
        )
    if cache not in (False, "disk", "ram", True):
        raise ValueError("cache must be false, 'disk', or 'ram'")
    block["cache"] = cache
    return block


def resolve_resume(run_dir: str | Path, resume: bool) -> Path | None:
    """Return the last checkpoint to resume from, or None for a fresh run."""

    if not resume:
        return None
    last = Path(run_dir) / "weights" / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(
            f"--resume requested but {last} does not exist; start a fresh run first"
        )
    return last


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preflight(config: Mapping[str, Any], args: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the built dataset and target contract before spending GPU time."""

    spec: TargetSpec = config["target"]
    raw = config.get("raw", config)
    dataset_root = Path(raw.get("dataset_root") or raw.get("dataset", {}).get("root") or "")
    if not dataset_root:
        raise ValueError("config.dataset_root is required")
    if not (dataset_root / "labels.json").is_file():
        raise FileNotFoundError(f"{dataset_root}/labels.json missing; build the dataset first")
    if not (dataset_root / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"{dataset_root}/manifest.jsonl missing; build the dataset first")
    labels = json.loads((dataset_root / "labels.json").read_text())
    if list(labels.get("classes", [])) != list(spec.classes):
        raise ValueError(
            f"dataset classes {labels.get('classes')} do not match target {spec.classes}"
        )
    if str(labels.get("target", {}).get("target_id")) != spec.target_id:
        raise ValueError("dataset target_id does not match config target_id")
    validation_path = dataset_root / "validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(f"{validation_path} missing; the dataset was not validated")
    validation = json.loads(validation_path.read_text())
    if not validation.get("ok"):
        raise RuntimeError(f"dataset validation did not pass: {validation.get('errors')}")
    dataset_summary: dict[str, Any] = {}
    counts_path = dataset_root / "counts.json"
    if counts_path.is_file():
        try:
            raw_counts = json.loads(counts_path.read_text())
            dataset_summary = {
                "split_mode": raw_counts.get("split_mode"),
                "prefer_test_unseen": raw_counts.get("prefer_test_unseen"),
                "stage1_overlap_counts": raw_counts.get("stage1_overlap_counts"),
            }
        except json.JSONDecodeError:
            dataset_summary = {}
    for split in ("train", "val", "test"):
        if not (dataset_root / split).is_dir():
            raise FileNotFoundError(f"dataset split directory missing: {dataset_root / split}")
    return {
        "dataset_root": str(dataset_root.resolve()),
        "validation": validation,
        "labels": labels,
        "dataset_summary": dataset_summary,
        "training_args": dict(args),
    }


def train(
    config_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one training job, then evaluate it on val+test.

    Refuses to run if the output directory already holds a completed run
    (``run.json``) unless ``resume`` points at the same directory.
    """

    config = load_run_config(config_path)
    spec: TargetSpec = config["target"]
    raw = config.get("raw", config)
    args = training_args(config, overrides)
    root = Path(raw.get("dataset_root") or raw.get("dataset", {}).get("root") or "")
    out = Path(output_dir or raw.get("output_dir") or (root.parent / "runs" / spec.target_id))
    out.mkdir(parents=True, exist_ok=True)
    run_name = "train"
    run_dir = out / run_name
    run_path = out / "run.json"
    if run_path.is_file() and not resume:
        raise RuntimeError(f"refusing duplicate run: {run_path} exists (pass --resume to continue)")
    resume_checkpoint = resolve_resume(run_dir, resume)

    pre = preflight(config, args)
    memory = configure_memory(
        args["device"],
        max_vram_fraction=float(args["max_vram_fraction"]),
        min_free_vram_gb=float(args["min_free_vram_gb"]),
    )
    import ultralytics
    import torch
    from ultralytics import YOLO

    seed_everything(int(args["seed"]))
    device = memory["device"]
    effective = dict(args)
    effective["device"] = device
    if device.startswith("cuda"):
        effective["amp"] = bool(args["amp"])
        effective["workers"] = int(args["workers"])
    else:
        # MPS/CPU were reliable with workers 0 and AMP off in the predecessor.
        effective["amp"] = False
        effective["workers"] = 0

    if run_dir.exists() and not resume:
        raise RuntimeError(f"refusing duplicate training directory: {run_dir}")
    if resume_checkpoint is not None:
        model = YOLO(str(resume_checkpoint))
    else:
        pretrained = Path(str(args["model"]))
        model = YOLO(str(pretrained) if pretrained.is_file() else str(args["model"]))

    prior = json.loads(run_path.read_text()) if resume and run_path.is_file() else None
    params: dict[str, Any] = prior or {
        "schema": "arq.anatomy4_specialists.training_run/v1",
        "target": spec.to_dict(),
        "dataset_root": pre["dataset_root"],
        "output_dir": str(out.resolve()),
        "started_at": utc_now(),
        "requested": dict(args),
        "effective": effective,
        "memory": memory,
        "packages": package_versions(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "seed": int(args["seed"]),
        "dataset_validation_ok": True,
        "dataset_split_mode": pre.get("dataset_summary", {}).get("split_mode"),
        "holdout_prefer_stage1_unseen": pre.get("dataset_summary", {}).get("prefer_test_unseen"),
        "stage1_overlap_counts": pre.get("dataset_summary", {}).get("stage1_overlap_counts"),
    }
    if resume_checkpoint is not None:
        params["resumed_at"] = utc_now()
        params["resumed_from"] = str(resume_checkpoint.resolve())
        params["effective"] = effective
        params["memory"] = memory
    (out / "run.json").write_text(json.dumps(params, indent=2) + "\n")

    started = time.perf_counter()
    model.train(
        data=str(Path(pre["dataset_root"])),
        epochs=int(effective["epochs"]),
        patience=int(effective["patience"]),
        imgsz=int(effective["imgsz"]),
        batch=int(effective["batch"]),
        device=device,
        workers=int(effective["workers"]),
        seed=int(effective["seed"]),
        deterministic=True,
        amp=bool(effective["amp"]),
        cache=effective["cache"],
        project=str(out),
        name=run_name,
        exist_ok=resume,
        resume=resume_checkpoint is not None,
        plots=False,
        val=True,
        save=True,
        save_period=-1,
        verbose=True,
    )
    elapsed = time.perf_counter() - started
    best_source = run_dir / "weights" / "best.pt"
    if not best_source.is_file():
        raise RuntimeError("training finished without best.pt; refusing to report metrics")
    best = out / "best.pt"
    shutil.copy2(best_source, best)
    last_source = run_dir / "weights" / "last.pt"
    if last_source.is_file():
        shutil.copy2(last_source, out / "last.pt")
    history: list[dict[str, str]] = []
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        with results_csv.open(newline="") as handle:
            history = list(csv.DictReader(handle))
        shutil.copy2(results_csv, out / "history.csv")

    params.update(
        {
            "completed_at": utc_now(),
            "wall_clock_seconds": round(elapsed, 2),
            "wall_clock_minutes": round(elapsed / 60.0, 2),
            "epochs_run": len(history) or None,
            "best_checkpoint": str(best.resolve()),
            "checkpoint_sha256": sha256_file(best),
        }
    )
    (out / "run.json").write_text(json.dumps(params, indent=2) + "\n")

    # Release the training graph/allocator cache before evaluation's own
    # VRAM preflight, otherwise the same process's residual allocations count
    # against the required free budget.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = evaluate_run(
        best,
        pre["dataset_root"],
        spec,
        out,
        device=device,
        imgsz=int(effective["imgsz"]),
        batch_size=min(64, int(effective["batch"])),
        target_precision=float(effective["target_precision"]),
        min_coverage=float(effective["min_coverage"]),
        max_vram_fraction=float(effective["max_vram_fraction"]),
        min_free_vram_gb=float(effective["min_free_vram_gb"]),
    )
    _write_report(out, params, metrics)
    return {
        "run": str((out / "run.json").resolve()),
        "metrics": str((out / "metrics.json").resolve()),
        "best": str(best.resolve()),
        "test_accuracy": metrics["test"]["accuracy"],
        "test_macro_f1": metrics["test"]["macro_prf_present_classes"]["f1"],
        "test_abstain_rate": metrics["test"]["abstention"]["unknown_rate"],
    }


def dry_run(config_path: str | Path, *, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Print the resolved plan without importing torch or touching the GPU."""

    config = load_run_config(config_path)
    spec = config["target"]
    raw = config.get("raw", config)
    args = training_args(config, overrides)
    root = Path(raw.get("dataset_root") or raw.get("dataset", {}).get("root") or "")
    return {
        "target": spec.to_dict(),
        "dataset_root": str(root),
        "output_dir": str(raw.get("output_dir") or (root.parent / "runs" / spec.target_id)),
        "training_args": args,
        "env": bounded_memory_env(),
        "would_check": [
            f"{root}/labels.json",
            f"{root}/manifest.jsonl",
            f"{root}/validation.json",
            f"{root}/train|val|test/",
            "CUDA free VRAM >= max_vram_fraction*total + min_free_vram_gb (when device resolves to cuda)",
        ],
        "note": "dry run only; no training and no metrics were produced",
    }


def _write_report(out: Path, params: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    test = metrics["test"]
    abstain = test["abstention"]
    per_class = test["per_class"]
    lines = [
        f"# {params['target']['target_id']} — conditioned expert",
        "",
        "## Training",
        "",
        f"- model: `{params['requested']['model']}`; device: `{params['effective']['device']}`",
        f"- epochs: {params.get('epochs_run')}/{params['effective']['epochs']}; seed: {params['seed']}; imgsz: {params['effective']['imgsz']}; batch: {params['effective']['batch']}",
        f"- cache: `{params['effective']['cache']}`; workers: {params['effective']['workers']}; AMP: {params['effective']['amp']}",
        f"- VRAM cap: {params['effective']['max_vram_fraction']} of device (free before: {params['memory'].get('free_gb_before')} GB)",
        f"- duration: {params.get('wall_clock_minutes')} minutes",
        f"- dataset: `{params['dataset_root']}` (split mode: {params.get('dataset_split_mode')}; prefer Stage-1-unseen test: {params.get('holdout_prefer_stage1_unseen')})",
        "",
        "## Test metrics (measured; no placeholders)",
        "",
        f"- accuracy: {test['accuracy']:.4f}",
        f"- macro precision / recall / F1 (present classes): {test['macro_prf_present_classes']['precision']:.4f} / {test['macro_prf_present_classes']['recall']:.4f} / {test['macro_prf_present_classes']['f1']:.4f}",
        f"- coverage: {abstain['coverage']:.4f}; abstain/unknown rate: {abstain['unknown_rate']:.4f}",
        f"- selective accuracy: {abstain['selective_accuracy']}",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in per_class.items():
        lines.append(
            f"| {name} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |"
        )
    lines += [
        "",
        "## Confusion matrix (rows actual, columns predicted)",
        "",
        "| actual \\ predicted | " + " | ".join(test["confusion_matrix"]["labels"]) + " |",
        "|---|" + "---:|" * len(test["confusion_matrix"]["labels"]),
    ]
    for name, row in zip(test["confusion_matrix"]["labels"], test["confusion_matrix"]["rows_actual_columns_predicted"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    lines += [
        "",
        f"Abstain confidence threshold selected on **val**: {metrics['abstain_selection']['confidence_threshold']} (target_met={metrics['abstain_selection']['target_met']}).",
        "",
        "## Artifacts",
        "",
        f"- checkpoint: `{params['best_checkpoint']}`",
        f"- metrics: `{out / 'metrics.json'}`",
        f"- predictions: `{out / 'predictions_test.jsonl'}`",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
