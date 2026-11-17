# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""YOLO26 detection training with a hard VRAM ceiling.

The .245 box runs BioCLIP (~4.3 GB of the RTX 4060 Ti's 8 GB) plus the OSM and
observation sidecars. A training job must never OOM a co-tenant service, so this
module:

* caps this process with ``torch.cuda.set_per_process_memory_fraction`` when a
  GPU budget is requested (default 3.2 GB);
* refuses to start when free VRAM is below the budget (unless overridden);
* exposes imgsz/batch/epochs/workers/amp as explicit CLI flags;
* has a dry-run mode that resolves everything (model, data, VRAM) without
  training.

No Docker service is restarted. Run inside the anatomy4 GPU image (or the host
venv) with the repository mounted.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_GPU_BUDGET_GB = 3.2
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_EPOCHS = 80
DEFAULT_WORKERS = 4
DEFAULT_SEED = 20261112


def build_train_kwargs(
    *,
    data: str | Path,
    model: str = "yolo26n.pt",
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    workers: int = DEFAULT_WORKERS,
    device: str = "0",
    project: str | Path,
    name: str = "train",
    seed: int = DEFAULT_SEED,
    patience: int = 20,
    amp: bool = True,
    cache: str | bool = False,
    exist_ok: bool = True,
    close_mosaic: int = 10,
    deterministic: bool = True,
    optimizer: str = "auto",
    lr0: float | None = None,
    lrf: float | None = None,
    cos_lr: bool = False,
    warmup_epochs: float = 3.0,
    cls: float = 0.5,
    mosaic: float = 1.0,
) -> dict[str, Any]:
    """Pure argument construction, unit-tested without torch."""

    return {
        "data": str(data),
        "model": str(model),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "device": str(device),
        "project": str(project),
        "name": str(name),
        "seed": int(seed),
        "patience": int(patience),
        "amp": bool(amp),
        "cache": cache,
        "exist_ok": bool(exist_ok),
        "close_mosaic": int(close_mosaic),
        "deterministic": bool(deterministic),
        "optimizer": str(optimizer),
        "cos_lr": bool(cos_lr),
        "warmup_epochs": float(warmup_epochs),
        "cls": float(cls),
        "mosaic": float(mosaic),
        **({"lr0": float(lr0)} if lr0 is not None else {}),
        **({"lrf": float(lrf)} if lrf is not None else {}),
        "plots": True,
        "val": True,
        "save": True,
        "verbose": True,
        "task": "detect",
    }


def validate_train_kwargs(kwargs: dict[str, Any]) -> None:
    if kwargs.get("epochs", 0) < 1:
        raise ValueError("epochs must be >= 1")
    if kwargs.get("imgsz", 0) < 64:
        raise ValueError("imgsz must be >= 64")
    if kwargs.get("batch", 0) < 1:
        raise ValueError("batch must be >= 1")
    if kwargs.get("workers", 0) < 0:
        raise ValueError("workers must be >= 0")
    if kwargs.get("lr0") is not None and kwargs["lr0"] <= 0:
        raise ValueError("lr0 must be > 0")
    if kwargs.get("lrf") is not None and kwargs["lrf"] < 0:
        raise ValueError("lrf must be >= 0")
    if kwargs.get("mosaic", 0) < 0 or kwargs.get("mosaic", 0) > 1:
        raise ValueError("mosaic must be in [0, 1]")
    if kwargs.get("cls", 0) < 0:
        raise ValueError("cls must be >= 0")
    if str(kwargs.get("model", "")).endswith("-seg.pt"):
        raise ValueError(
            "yolo26n-seg is a segmentation checkpoint; anatomy4-v1 trains a "
            "detector (yolo26n.pt)"
        )


def configure_process_env() -> None:
    """Must run before torch initializes CUDA."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))


def resolve_model_path(model: str) -> str:
    """Resolve a bare checkpoint name into the Ultralytics weights cache.

    Ultralytics otherwise downloads ``yolo26n.pt`` into the current working
    directory, polluting the repository. The cache is shared across runs.
    """

    candidate = Path(model).expanduser()
    if candidate.is_file() or candidate.parent != Path("."):
        return str(candidate)
    if "/" in model or os.sep in model:
        return model
    try:
        from ultralytics.utils import SETTINGS
        from ultralytics.utils.downloads import attempt_download_asset
    except Exception:
        return model
    weights_dir = Path(
        SETTINGS.get("weights_dir") or (Path.home() / ".cache" / "ultralytics" / "weights")
    )
    cached = weights_dir / model
    if cached.is_file():
        return str(cached)
    weights_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        os.chdir(weights_dir)
        return attempt_download_asset(model)
    finally:
        os.chdir(cwd)


def resolve_device(requested: str) -> str:
    """``auto`` picks cuda:0, then mps, then cpu."""

    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _gpu_index(device: str) -> int | None:
    if device in {"cpu", "mps"}:
        return None
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    if device.isdigit():
        return int(device)
    return None


def apply_vram_budget(device: str, budget_gb: float | None) -> dict[str, Any]:
    """Cap the process's CUDA allocator; returns a report for the manifest."""

    import torch

    index = _gpu_index(device)
    if index is None or not torch.cuda.is_available():
        return {"applicable": False, "device": device, "budget_gb": budget_gb}
    if budget_gb is None or budget_gb <= 0:
        return {"applicable": False, "device": device, "budget_gb": budget_gb}
    total = torch.cuda.get_device_properties(index).total_memory / (1024**3)
    free, total_free = torch.cuda.mem_get_info(index)
    free_gb = free / (1024**3)
    fraction = min(0.95, budget_gb / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return {
        "applicable": True,
        "device": f"cuda:{index}",
        "gpu_name": torch.cuda.get_device_name(index),
        "total_vram_gb": round(total, 2),
        "free_vram_gb_at_start": round(free_gb, 2),
        "budget_gb": budget_gb,
        "fraction": round(fraction, 4),
        "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def _free_vram_gb(device: str) -> float | None:
    import torch

    index = _gpu_index(device)
    if index is None or not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info(index)
    return free / (1024**3)


def preflight(device: str, budget_gb: float | None, allow_low_vram: bool) -> dict[str, Any]:
    free = _free_vram_gb(device)
    report: dict[str, Any] = {"device": device, "free_vram_gb": free}
    if free is not None and budget_gb:
        report["budget_gb"] = budget_gb
        report["headroom_ok"] = free >= budget_gb * 0.85
        if not report["headroom_ok"] and not allow_low_vram:
            raise RuntimeError(
                f"only {free:.2f} GiB free on {device}, budget is {budget_gb:.2f} GiB "
                "(BioCLIP and the sidecars share this GPU). Lower --imgsz/--batch, "
                "lower --gpu-budget-gb, or pass --allow-low-vram deliberately."
            )
    else:
        report["headroom_ok"] = True
    return report


def _assert_trainable_data(data_path: str) -> None:
    """Refuse a frozen test-only prepared view even when pointed at data.yaml."""

    data = Path(data_path)
    metadata = data.parent / "anatomy4_prepared.json"
    if not metadata.is_file():
        return
    try:
        doc = json.loads(metadata.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if doc.get("test_only"):
        raise ValueError(
            "frozen test-only dataset: training is disabled. Evaluate a "
            "checkpoint with the `evaluate` command instead."
        )


def train(
    *,
    data: str | Path,
    project: str | Path,
    name: str = "anatomy4",
    device: str = "auto",
    gpu_budget_gb: float | None = DEFAULT_GPU_BUDGET_GB,
    allow_low_vram: bool = False,
    dry_run: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    configure_process_env()
    resolved_device = resolve_device(device)
    train_kwargs = build_train_kwargs(
        data=data,
        project=project,
        name=name,
        device=resolved_device,
        **{key: value for key, value in kwargs.items() if value is not None},
    )
    validate_train_kwargs(train_kwargs)
    _assert_trainable_data(train_kwargs["data"])
    preflight_report = preflight(resolved_device, gpu_budget_gb, allow_low_vram)
    budget_report = apply_vram_budget(resolved_device, gpu_budget_gb)

    from ultralytics import YOLO

    resolved_model = resolve_model_path(train_kwargs["model"])
    train_kwargs["model"] = resolved_model
    model = YOLO(resolved_model)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    report: dict[str, Any] = {
        "trainer": "ultralytics",
        "model": train_kwargs["model"],
        "resolved_model": resolved_model,
        "task": "detect",
        "parameters": parameter_count,
        "device": resolved_device,
        "vram": {"preflight": preflight_report, "budget": budget_report},
        "hyperparameters": {
            key: train_kwargs.get(key)
            for key in (
                "data",
                "epochs",
                "imgsz",
                "batch",
                "workers",
                "seed",
                "patience",
                "amp",
                "cache",
                "close_mosaic",
                "deterministic",
                "optimizer",
                "lr0",
                "lrf",
                "cos_lr",
                "warmup_epochs",
                "cls",
                "mosaic",
            )
        },
        "dry_run": dry_run,
    }
    if dry_run:
        report["ok"] = True
        report["save_dir"] = None
        return report

    started = time.time()
    results = model.train(**train_kwargs)
    elapsed = time.time() - started
    save_dir = Path(getattr(results, "save_dir", Path(project) / name)).resolve()
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    self_reported = getattr(results, "results_dict", None)
    report.update(
        {
            "ok": best.is_file(),
            "save_dir": str(save_dir),
            "best_weights": str(best) if best.is_file() else None,
            "last_weights": str(last) if last.is_file() else None,
            "elapsed_seconds": round(elapsed, 2),
            "trainer_self_reported_metrics": (
                _jsonable(self_reported) if isinstance(self_reported, dict) else None
            ),
            "note": (
                "trainer_self_reported_metrics are Ultralytics' own val numbers; "
                "the held-out test metrics in the manifest come from "
                "anatomy4_detector.metrics on saved predictions"
            ),
        }
    )
    if not best.is_file():
        raise RuntimeError(f"training finished but best.pt is missing in {save_dir}")
    return report


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return {str(key): float(item) if isinstance(item, (int, float)) else str(item)
                for key, item in value.items()}
