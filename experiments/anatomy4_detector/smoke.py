# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Synthetic end-to-end smoke: tiny dataset -> train -> eval -> export -> manifest.

This exists so the pipeline can be exercised without the Shenandoah materialized
dataset. Every artifact is stamped ``synthetic: true`` and its metrics are
meaningless as a detector claim — they only prove the plumbing is real.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from .contracts import CLASS_NAMES, CONTRACT
from .dataset import prepare_ultralytics, render_audit_markdown, validate_dataset, write_audit
from .evaluate import evaluate_split
from .export import export_onnx, verify_onnx
from .manifest import build_run_manifest, write_manifest
from .render import CLASS_COLORS
from .train import DEFAULT_GPU_BUDGET_GB, train

SMOKE_SPLITS = {"train": 14, "val": 5, "test": 5}
IMAGE_SIZE = 320


def _draw_shape(draw, class_name: str, box: tuple[int, int, int, int]) -> None:
    color = CLASS_COLORS[class_name]
    x0, y0, x1, y1 = box
    if class_name == "leaf":
        draw.ellipse(box, fill=color, outline=(255, 255, 255), width=2)
        draw.line((x0, (y0 + y1) / 2, x1, (y0 + y1) / 2), fill=(20, 90, 40), width=2)
    elif class_name == "needle":
        draw.rounded_rectangle(box, radius=4, fill=color, outline=(255, 255, 255), width=2)
        for offset in range(6, max(7, x1 - x0), 8):
            draw.line((x0 + offset, y0 + 3, x0 + offset, y1 - 3), fill=(220, 235, 255), width=1)
    elif class_name == "flower":
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        radius = min(x1 - x0, y1 - y0) / 2
        for angle in range(0, 360, 45):
            px = cx + math.cos(math.radians(angle)) * radius * 0.62
            py = cy + math.sin(math.radians(angle)) * radius * 0.62
            petal = radius * 0.38
            draw.ellipse((px - petal, py - petal, px + petal, py + petal), fill=color)
        draw.ellipse((cx - radius * 0.28, cy - radius * 0.28,
                      cx + radius * 0.28, cy + radius * 0.28), fill=(255, 240, 120))
    else:  # inflorescence
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for dx, dy in ((-0.24, -0.18), (0.0, -0.28), (0.24, -0.14), (-0.12, 0.18), (0.14, 0.2)):
            px = cx + dx * (x1 - x0)
            py = cy + dy * (y1 - y0)
            r = max(6, (x1 - x0) * 0.15)
            draw.ellipse((px - r, py - r, px + r, py + r), fill=color,
                         outline=(255, 250, 220), width=1)
        draw.line((cx, cy + (y1 - y0) * 0.3, cx, y1), fill=(40, 120, 55), width=3)


def build_synthetic_dataset(root: str | Path, *, seed: int = 7) -> dict[str, Any]:
    """Deterministic tiny dataset with class shapes, multi-box images, negatives."""

    from PIL import Image, ImageDraw

    dataset_root = Path(root).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    entries: list[dict[str, Any]] = []
    index = 0

    for split, count in SMOKE_SPLITS.items():
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for position in range(count):
            negative = position == count - 1  # one explicit negative per split
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (24, 30, 26))
            draw = ImageDraw.Draw(image)
            for _ in range(18):
                shade = rng.randint(20, 70)
                x = rng.randint(0, IMAGE_SIZE - 40)
                y = rng.randint(0, IMAGE_SIZE - 40)
                draw.rectangle(
                    (x, y, x + rng.randint(10, 60), y + rng.randint(10, 60)),
                    fill=(shade // 2, shade, shade // 2 + 8),
                )
            boxes: list[tuple[int, tuple[int, int, int, int]]] = []
            if not negative:
                class_count = 1 + (1 if rng.random() < 0.45 else 0)
                for _ in range(class_count):
                    class_id = rng.randrange(len(CLASS_NAMES))
                    size = rng.randint(64, 130)
                    x0 = rng.randint(8, IMAGE_SIZE - size - 8)
                    y0 = rng.randint(8, IMAGE_SIZE - size - 8)
                    box = (x0, y0, x0 + size, y0 + size)
                    _draw_shape(draw, CLASS_NAMES[class_id], box)
                    boxes.append((class_id, box))
            stem = f"{split}-{position:02d}"
            image_path = dataset_root / "images" / split / f"{stem}.jpg"
            image.save(image_path, quality=92)
            label_lines = [
                f"{class_id} {x0 / IMAGE_SIZE:.6f} {y0 / IMAGE_SIZE:.6f} "
                f"{x1 / IMAGE_SIZE:.6f} {y1 / IMAGE_SIZE:.6f}"
                for class_id, (x0, y0, x1, y1) in boxes
            ]
            (dataset_root / "labels" / split / f"{stem}.txt").write_text(
                ("\n".join(label_lines) + "\n") if label_lines else ""
            )
            # Two train images deliberately share one observation to exercise
            # the isolation check (same split, so this is legal).
            observation = "obs-shared-train" if split == "train" and position < 2 else f"obs-{stem}"
            entries.append(
                {
                    "image": f"images/{split}/{stem}.jpg",
                    "observation_id": observation,
                    "negative": negative,
                }
            )
            index += 1

    (dataset_root / "dataset.json").write_text(
        json.dumps(
            {
                "schema": "arq.anatomy4.dataset/v1",
                "contract": CONTRACT,
                "classes": list(CLASS_NAMES),
                "images": entries,
                "synthetic": True,
                "seed": seed,
            },
            indent=1,
        )
        + "\n"
    )
    return {
        "root": str(dataset_root),
        "images": index,
        "splits": SMOKE_SPLITS,
        "synthetic": True,
    }


def run_smoke(
    out_dir: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    device: str = "auto",
    epochs: int = 2,
    imgsz: int = 160,
    batch: int = 4,
    workers: int = 0,
    gpu_budget_gb: float | None = DEFAULT_GPU_BUDGET_GB,
    dry_run: bool = False,
) -> dict[str, Any]:
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_dir or output / "synthetic-dataset")
    synthetic = build_synthetic_dataset(dataset_root)
    audit = validate_dataset(dataset_root)
    audit_path = write_audit(audit, output / "dataset-audit.json")
    (output / "dataset-audit.md").write_text(render_audit_markdown(audit))
    if audit["blocking"]:
        raise RuntimeError(f"smoke dataset is invalid: {audit['errors']}")

    prepared = prepare_ultralytics(audit, output / "prepared", link_mode="copy")
    training = train(
        data=prepared["data_yaml"],
        project=output / "runs",
        name="smoke",
        device=device,
        gpu_budget_gb=gpu_budget_gb,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        close_mosaic=0,
        patience=max(1, epochs),
        dry_run=dry_run,
    )
    if dry_run or not training.get("ok"):
        manifest = build_run_manifest(
            code_root=Path(__file__).resolve().parents[2],
            dataset_audit=audit,
            dataset_audit_path=audit_path,
            training=training,
            evaluation=None,
            export=None,
            onnx_verification=None,
            prepared=prepared,
            synthetic=True,
            status="dry_run",
        )
        write_manifest(manifest, output / "run-manifest.json")
        return manifest

    evaluation = evaluate_split(
        audit=audit,
        split="test",
        prepared_root=prepared["root"],
        weights=training["best_weights"],
        out_dir=output / "eval-test",
        imgsz=imgsz,
        device=device,
        batch=batch,
        max_overlays=6,
        overlays_per_sheet=6,
    )
    export = export_onnx(training["best_weights"], output / "export" / "anatomy4.onnx")
    verification = verify_onnx(export["onnx"])
    manifest = build_run_manifest(
        code_root=Path(__file__).resolve().parents[2],
        dataset_audit=audit,
        dataset_audit_path=audit_path,
        training=training,
        evaluation=evaluation,
        export=export,
        onnx_verification=verification,
        prepared=prepared,
        synthetic=True,
    )
    write_manifest(manifest, output / "run-manifest.json")
    manifest["_smoke"] = synthetic
    return manifest
