# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Device selection and bounded-VRAM helpers shared by train and evaluate.

The .245 host runs BioCLIP (~4.3 GB) beside training/evaluation on an 8 GB
RTX 4060 Ti. The rule is explicit: this process may allocate at most
``max_vram_fraction`` of total VRAM, and training/evaluation may only start
when free VRAM is at least that budget **plus** a safety margin.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MAX_VRAM_FRACTION = 0.30
DEFAULT_MIN_FREE_VRAM_GB = 0.5


def bounded_memory_env() -> dict[str, str]:
    """Environment knobs applied before torch import."""

    return {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
    }


def resolve_device(device: str) -> str:
    import torch

    text = str(device).strip()
    if text == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    # Ultralytics accepts "0"/"0,1"; torch modules need "cuda:0".
    if text.isdigit():
        return f"cuda:{text}"
    if text == "cuda" or (text.startswith("cuda:") is False and text.startswith("cuda,")):
        return "cuda:0" if text == "cuda" else f"cuda:{text.split(',')[0].split(':')[-1]}"
    return text


def vram_requirement(total_gb: float, max_vram_fraction: float, min_free_vram_gb: float) -> dict[str, float]:
    """Pure arithmetic used by the preflight and its tests."""

    if not 0.0 < max_vram_fraction <= 1.0:
        raise ValueError("max_vram_fraction must be in (0, 1]")
    if min_free_vram_gb < 0:
        raise ValueError("min_free_vram_gb must be >= 0")
    budget = total_gb * max_vram_fraction
    return {
        "total_gb": total_gb,
        "process_budget_gb": budget,
        "margin_gb": min_free_vram_gb,
        "required_free_gb": budget + min_free_vram_gb,
    }


def configure_memory(
    device: str,
    *,
    max_vram_fraction: float = DEFAULT_MAX_VRAM_FRACTION,
    min_free_vram_gb: float = DEFAULT_MIN_FREE_VRAM_GB,
) -> dict[str, Any]:
    """Cap this process's CUDA allocator and require budget + margin free."""

    for key, value in bounded_memory_env().items():
        os.environ.setdefault(key, value)
    import torch

    resolved = resolve_device(device)
    info: dict[str, Any] = {
        "requested_device": device,
        "device": resolved,
        "max_vram_fraction": max_vram_fraction,
        "min_free_vram_gb": min_free_vram_gb,
        "cuda": torch.cuda.is_available(),
    }
    if not resolved.startswith("cuda"):
        info["note"] = "CUDA not used; VRAM cap not applied"
        return info

    index = int(resolved.split(":")[1]) if ":" in resolved else 0
    torch.cuda.set_device(index)
    torch.cuda.set_per_process_memory_fraction(float(max_vram_fraction), device=index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    requirement = vram_requirement(total_gb, float(max_vram_fraction), float(min_free_vram_gb))
    info.update(
        {
            "gpu": torch.cuda.get_device_name(index),
            "free_gb_before": round(free_gb, 3),
            "total_gb": round(total_gb, 3),
            "process_budget_gb": round(requirement["process_budget_gb"], 3),
            "required_free_gb": round(requirement["required_free_gb"], 3),
        }
    )
    if free_gb < requirement["required_free_gb"]:
        raise RuntimeError(
            f"free VRAM {free_gb:.2f} GB < required {requirement['required_free_gb']:.2f} GB "
            f"(declared fraction {max_vram_fraction:.2f} = {requirement['process_budget_gb']:.2f} GB "
            f"+ {min_free_vram_gb:.2f} GB margin). Free a GPU consumer or lower "
            "--max-vram-fraction; do NOT stop services automatically."
        )
    info["headroom_note"] = (
        "The cap is enforced per process; BioCLIP on .245 is reported at ~4.3 GB, "
        "so the default 0.30 fraction leaves the rest of the card for it."
    )
    return info
