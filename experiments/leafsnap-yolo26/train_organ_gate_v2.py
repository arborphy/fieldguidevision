#!/usr/bin/env python3
# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Train and evaluate the corrective organ-gate v2 detector."""

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1_ROOT = ROOT / "scratch" / "leafsnap-yolo26"
OUT_ROOT = V1_ROOT / "organ-gate-v2"
DATASET_DIR = OUT_ROOT / "dataset"
RUNS_DIR = OUT_ROOT / "runs"
V1_MANIFEST = V1_ROOT / "detector" / "manifest.jsonl"
V1_RESULTS = V1_ROOT / "live-audit" / "results.json"
PRETRAINED = V1_ROOT / "organ-gate" / "pretrained" / "yolo26n.pt"
CLASSES = ["leaf", "bud", "flower", "twig", "branch"]
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}
NONLEAF = ["bud", "flower", "twig", "branch"]
SEED = 260930
IMGSZ = 320
BATCH = 16
EPOCHS = 18
PATIENCE = 18
CONF_FLOOR = 0.001
CONF_THRESHOLDS = [0.12, 0.25, 0.5]
NMS_IOU = 0.45
MAX_DET = 300


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, Any]:
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
    }


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_v1_compatible_test(rows: Sequence[dict[str, Any]]) -> tuple[Path, list[dict[str, Any]]]:
    from shutil import rmtree

    v1_test_uids = {
        row["uid"] for row in load_manifest(V1_MANIFEST) if row["split"] == "test"
    }
    compatible = [
        row
        for row in rows
        if row["split"] == "test"
        and row["uid"] in v1_test_uids
        and all(box["class_name"] in CLASS_IDS for box in row["boxes"])
    ]
    image_dir = DATASET_DIR / "images" / "v1_compatible_test"
    label_dir = DATASET_DIR / "labels" / "v1_compatible_test"
    for path in (image_dir, label_dir):
        if path.exists():
            rmtree(path)
        path.mkdir(parents=True)
    for row in compatible:
        src_img = (DATASET_DIR / row["image"]).resolve()
        src_label = (DATASET_DIR / row["label"]).resolve()
        (image_dir / src_img.name).symlink_to(src_img)
        (label_dir / src_label.name).symlink_to(src_label)
    yaml_path = DATASET_DIR / "v1_compatible_test.yaml"
    yaml_path.write_text(
        f"path: {DATASET_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/v1_compatible_test\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    )
    return yaml_path, compatible


def parse_results_csv(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            clean: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    clean[key.strip()] = float((value or "").strip())
                except ValueError:
                    clean[key.strip()] = (value or "").strip()
            out.append(clean)
    return out


def train_model(data_yaml: Path) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    if not PRETRAINED.is_file() or PRETRAINED.stat().st_size < 1_000_000:
        raise RuntimeError(f"missing pretrained checkpoint: {PRETRAINED}")
    if not (OUT_ROOT / "dataset" / "validation.json").is_file():
        raise RuntimeError("v2 dataset validation report is missing")
    validation = json.loads((OUT_ROOT / "dataset" / "validation.json").read_text())
    if not validation.get("ok"):
        raise RuntimeError("v2 dataset validation failed")

    run_dir = RUNS_DIR / "train_v2"
    if run_dir.exists():
        raise RuntimeError(f"training run already exists; refusing a second run: {run_dir}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    weights_dir = OUT_ROOT / "pretrained"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights = weights_dir / "yolo26n.pt"
    if not weights.exists():
        shutil.copy2(PRETRAINED, weights)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset_fingerprint = {
        "dataset_yaml": str(data_yaml),
        "manifest": str(DATASET_DIR / "manifest.jsonl"),
        "sha256": {
            "dataset.yaml": sha256_file(DATASET_DIR / "dataset.yaml"),
            "manifest.jsonl": sha256_file(DATASET_DIR / "manifest.jsonl"),
            "counts.json": sha256_file(DATASET_DIR / "counts.json"),
            "validation.json": sha256_file(DATASET_DIR / "validation.json"),
        },
        "manifest_rows": len(load_manifest(DATASET_DIR / "manifest.jsonl")),
        "classes": CLASSES,
    }
    params: dict[str, Any] = {
        "experiment": "leafsnap-yolo26 organ gate v2",
        "task": "detect",
        "model": str(weights),
        "model_name": "yolo26n.pt",
        "device": device,
        "mps_available": bool(torch.backends.mps.is_available()),
        "epochs_requested": EPOCHS,
        "patience": PATIENCE,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "seed": SEED,
        "deterministic": True,
        "workers": 0,
        "amp": False,
        "cache": False,
        "optimizer": "auto",
        "data": str(data_yaml),
        "dataset": dataset_fingerprint,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "packages": package_versions(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": device,
        },
        "run_policy": "one fixed-seed fine-tuning run; no smoke or retry run",
    }
    write_json(OUT_ROOT / "training_params.json", params)
    seed_everything(SEED)
    model = YOLO(str(weights))
    started = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=0,
        seed=SEED,
        deterministic=True,
        amp=False,
        cache=False,
        project=str(RUNS_DIR),
        name="train_v2",
        exist_ok=False,
        plots=False,
        verbose=True,
        val=True,
        save=True,
        save_period=-1,
    )
    elapsed = time.perf_counter() - started
    results_csv = run_dir / "results.csv"
    history = parse_results_csv(results_csv) if results_csv.exists() else []
    best_source = run_dir / "weights" / "best.pt"
    last_source = run_dir / "weights" / "last.pt"
    if not best_source.exists():
        raise RuntimeError(f"training did not produce best.pt in {run_dir}")
    shutil.copy2(best_source, OUT_ROOT / "best.pt")
    if last_source.exists():
        shutil.copy2(last_source, OUT_ROOT / "last.pt")
    if results_csv.exists():
        shutil.copy2(results_csv, OUT_ROOT / "history.csv")
    write_json(OUT_ROOT / "history.json", history)

    epoch_seconds = []
    for prev, current in zip(history, history[1:]):
        try:
            epoch_seconds.append(float(current["time"]) - float(prev["time"]))
        except (KeyError, TypeError, ValueError):
            continue
    params.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "epochs_run": len(history),
            "wall_clock_seconds": round(elapsed, 2),
            "wall_clock_minutes": round(elapsed / 60.0, 2),
            "mean_epoch_seconds": round(statistics.mean(epoch_seconds), 1)
            if epoch_seconds
            else None,
            "best_checkpoint": str(OUT_ROOT / "best.pt"),
            "run_dir": str(run_dir),
            "early_stopped": len(history) < EPOCHS,
        }
    )
    write_json(OUT_ROOT / "training_params.json", params)
    return params


def standard_test_metrics(best_pt: Path, compatible_yaml: Path) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(best_pt))
    metrics = model.val(
        data=str(compatible_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        conf=CONF_FLOOR,
        iou=0.7,
        plots=False,
        verbose=False,
        project=str(RUNS_DIR),
        name="standard-val-v1-compatible-test",
        exist_ok=False,
    )
    box = metrics.box
    indices = getattr(box, "ap_class_index", [])
    gt_counts = getattr(metrics, "nt_per_class", None)
    per_class: dict[str, Any] = {}
    for pos, class_id in enumerate(indices):
        class_id = int(class_id)
        per_class[CLASSES[class_id]] = {
            "class_id": class_id,
            "precision": round(float(box.p[pos]), 4),
            "recall": round(float(box.r[pos]), 4),
            "f1": round(float(box.f1[pos]), 4)
            if getattr(box, "f1", None) is not None
            else None,
            "ap50": round(float(box.ap50[pos]), 4),
            "ap50_95": round(float(box.ap[pos]), 4),
            "n_gt": int(gt_counts[class_id]) if gt_counts is not None else None,
        }
    payload = {
        "dataset": "v1-compatible held-out test; shared v1 test UIDs, excluding removed entire_plant",
        "imgsz": IMGSZ,
        "images": len(list((DATASET_DIR / "images" / "v1_compatible_test").glob("*"))),
        "box": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
        },
        "per_class": per_class,
        "note": "P/R are Ultralytics values at its max-F1 point; mAP is threshold-free. BioImages near-full-frame boxes are weak labels, not localization quality.",
        "speed_ms_per_image": {k: round(float(v), 3) for k, v in metrics.speed.items()},
    }
    write_json(OUT_ROOT / "standard_test_metrics.json", payload)
    return payload


def predict_records(
    model_path: Path,
    rows: Sequence[dict[str, Any]],
    split_label: str,
) -> list[dict[str, Any]]:
    import torch
    from ultralytics import YOLO
    sys.path.insert(0, str(HERE))
    import organ_gate_v2_eval as ev

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(model_path))
    paths = [str((DATASET_DIR / row["image"]).resolve()) for row in rows]
    by_path = {str((DATASET_DIR / row["image"]).resolve()): row for row in rows}
    by_name = {Path(row["image"]).name: row for row in rows}
    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    results = model.predict(
        source=paths,
        imgsz=IMGSZ,
        conf=CONF_FLOOR,
        iou=0.7,
        max_det=300,
        batch=BATCH,
        device=device,
        stream=True,
        verbose=False,
        half=False,
    )
    for result in results:
        key = str(Path(result.path).resolve())
        row = by_path.get(key) or by_name.get(Path(result.path).name)
        if row is None:
            raise RuntimeError(f"prediction path did not map to dataset manifest: {result.path}")
        preds = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().tolist()
            conf = boxes.conf.detach().cpu().tolist()
            cls = boxes.cls.detach().cpu().tolist()
            for box, score, class_id in zip(xyxy, conf, cls):
                preds.append(
                    {"class_id": int(class_id), "confidence": float(score), "bbox_xyxy": box}
                )
        records[row["uid"]] = ev.make_record(row, preds)
    ordered = [records[row["uid"]] for row in rows]
    elapsed = time.perf_counter() - started
    path = OUT_ROOT / f"predictions_{split_label}.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))
    write_json(
        OUT_ROOT / f"predictions_{split_label}_summary.json",
        {
            "split": split_label,
            "images": len(ordered),
            "conf_floor": CONF_FLOOR,
            "nms_iou": 0.7,
            "imgsz": IMGSZ,
            "elapsed_seconds": round(elapsed, 2),
            "mean_ms_per_image": round(elapsed * 1000 / max(1, len(ordered)), 2),
            "total_predictions": sum(len(row["preds"]) for row in ordered),
        },
    )
    return ordered


def run_custom_eval(
    model_path: Path,
    rows: Sequence[dict[str, Any]],
    compatible_test_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import organ_gate_v2_eval as ev

    val_rows = [row for row in rows if row["split"] == "val"]
    val_records = predict_records(model_path, val_rows, "val")
    test_records = predict_records(model_path, compatible_test_rows, "v1_compatible_test")
    selection = ev.select_confidence(val_records)
    chosen = float(selection["confidence"])
    result = {
        "validation_confidence_selection": selection,
        "test_at_validation_selected_confidence": ev.evaluate(
            test_records, chosen, label="v1-compatible held-out test at validation-selected threshold"
        ),
        "test_at_fixed_conf_0.25": ev.evaluate(
            test_records, 0.25, label="v1-compatible held-out test at fixed conf 0.25"
        ),
        "test_images": len(test_records),
        "validation_images": len(val_records),
        "selection_rule": "validation-only max macro F1 across classes present; ties to lower confidence",
    }
    write_json(OUT_ROOT / "custom_test_metrics.json", result)
    return result


def topologically_sort_onnx(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    producer: dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        for output in node.output:
            producer[output] = index
    available = {tensor.name for tensor in model.graph.input} | {
        tensor.name for tensor in model.graph.initializer
    }
    pending = set(range(len(model.graph.node)))
    ordered = []
    while pending:
        progressed = False
        for index in sorted(pending):
            node = model.graph.node[index]
            if all(name in available or not name for name in node.input):
                ordered.append(node)
                available.update(node.output)
                pending.remove(index)
                progressed = True
        if not progressed:
            raise RuntimeError("ONNX graph has a cyclic or unresolved dependency")
    reordered = [node.name for node in model.graph.node] != [node.name for node in ordered]
    if reordered:
        del model.graph.node[:]
        model.graph.node.extend(ordered)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    fp16_initializers = sum(
        initializer.data_type == onnx.TensorProto.FLOAT16 for initializer in model.graph.initializer
    )
    return {
        "onnx_checker": "pass",
        "topologically_sorted": bool(reordered),
        "nodes": len(ordered),
        "fp16_initializers": fp16_initializers,
    }


def export_onnx(best_pt: Path) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    from ultralytics import YOLO

    export_path = Path(
        YOLO(str(best_pt)).export(
            format="onnx",
            imgsz=IMGSZ,
            opset=17,
            half=True,
            dynamic=False,
            simplify=False,
            batch=1,
            device="cpu",
        )
    )
    if not export_path.exists():
        raise RuntimeError(f"Ultralytics did not create ONNX: {export_path}")
    target = OUT_ROOT / "organ_gate.onnx"
    shutil.copy2(export_path, target)
    if export_path.resolve() != target.resolve():
        export_path.unlink(missing_ok=True)
    graph_check = topologically_sort_onnx(target)
    model = onnx.load(str(target), load_external_data=False)

    def dims(value: Any) -> list[Any]:
        return [dimension.dim_value or dimension.dim_param for dimension in value.type.tensor_type.shape.dim]

    inputs = [
        {
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
            "shape": dims(value),
        }
        for value in model.graph.input
    ]
    outputs = [
        {
            "name": value.name,
            "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
            "shape": dims(value),
        }
        for value in model.graph.output
    ]
    if inputs[0]["shape"] != [1, 3, IMGSZ, IMGSZ]:
        raise RuntimeError(f"non-static or wrong ONNX input shape: {inputs[0]['shape']}")
    if outputs[0]["shape"] != [1, 4 + len(CLASSES), 2100]:
        raise RuntimeError(f"unexpected ONNX output shape: {outputs[0]['shape']}")
    if graph_check["fp16_initializers"] <= 0:
        raise RuntimeError("FP16 export contains no FP16 initializers")

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
    output = session.run(None, {session.get_inputs()[0].name: dummy})[0]
    if list(output.shape) != [1, 4 + len(CLASSES), 2100] or not np.isfinite(output).all():
        raise RuntimeError(f"ONNX smoke inference failed: shape={output.shape}")

    labels = {
        "dataset": "LeafSnap/BioImages organ gate v2",
        "task": "detect",
        "model": "yolo26n fine-tuned",
        "class_order": {str(index): name for index, name in enumerate(CLASSES)},
        "names": {name: index for index, name in enumerate(CLASSES)},
        "nc": len(CLASSES),
        "input": {
            "name": inputs[0]["name"],
            "shape": inputs[0]["shape"],
            "dtype": inputs[0]["dtype"],
            "layout": "NCHW",
            "preprocessing": "RGB, letterboxed to 320x320 with 114 fill, scaled 0-1",
        },
        "output": {
            "name": outputs[0]["name"],
            "shape": outputs[0]["shape"],
            "dtype": outputs[0]["dtype"],
            "layout": "[batch, 4 + nc, anchors]: cx, cy, w, h, class scores (no objectness)",
            "coordinate_space": "input pixels (320x320 letterboxed image)",
        },
        "onnx": {
            "file": "organ_gate.onnx",
            "opset": [
                {"domain": op.domain or "ai.onnx", "version": op.version}
                for op in model.opset_import
            ],
            "static": True,
            "batch": 1,
            "imgsz": IMGSZ,
            "half": True,
            "fp16_initializers": graph_check["fp16_initializers"],
            "onnx_checker": graph_check["onnx_checker"],
            "output_shape_verified": [1, 9, 2100],
            "smoke_inference": "pass",
        },
    }
    write_json(OUT_ROOT / "organ_gate.labels.json", labels)
    payload = {
        "checkpoint": str(best_pt),
        "onnx": str(target),
        "bytes": target.stat().st_size,
        "inputs": inputs,
        "outputs": outputs,
        "graph_check": graph_check,
        "smoke_output_shape": list(output.shape),
        "labels": str(OUT_ROOT / "organ_gate.labels.json"),
    }
    write_json(OUT_ROOT / "onnx_export.json", payload)
    return payload


def letterbox_app(image: Image.Image) -> tuple[np.ndarray, float, int, int]:
    width, height = image.size
    scale = min(IMGSZ / width, IMGSZ / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    pad_x = (IMGSZ - new_w) // 2
    pad_y = (IMGSZ - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None]), scale, pad_x, pad_y


def classwise_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> list[int]:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size and len(kept) < MAX_DET:
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
        y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
        x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
        y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
        intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_current = max(
            (boxes[current, 2] - boxes[current, 0])
            * (boxes[current, 3] - boxes[current, 1]),
            1e-9,
        )
        area_rest = np.clip(
            (boxes[remaining, 2] - boxes[remaining, 0])
            * (boxes[remaining, 3] - boxes[remaining, 1]),
            0,
            None,
        )
        union = area_current + area_rest - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = remaining[(iou <= NMS_IOU) | (classes[remaining] != classes[current])]
    return kept


def decode_onnx(
    raw: np.ndarray,
    width: int,
    height: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    preds = np.asarray(raw, dtype=np.float32)
    if preds.ndim == 3:
        preds = preds[0]
    n_channels = 4 + len(CLASSES)
    if preds.shape[0] != n_channels and preds.shape[1] == n_channels:
        preds = preds.T
    if preds.shape[0] != n_channels:
        raise ValueError(f"unexpected v2 output shape {preds.shape}")
    boxes = preds[:4].T
    class_scores = preds[4:].T
    classes = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    max_by_class = {
        name: float(class_scores[:, index].max()) for index, name in enumerate(CLASSES)
    }
    selected = scores >= CONF_FLOOR
    boxes = boxes[selected]
    scores = scores[selected]
    classes = classes[selected]
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    keep = classwise_nms(xyxy, scores, classes)
    xyxy, scores, classes = xyxy[keep], scores[keep], classes[keep]
    xyxy = (xyxy - np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)) / max(
        scale, 1e-9
    )
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
    detections = []
    for box, score, class_id in zip(xyxy, scores, classes):
        x1, y1, x2, y2 = [float(v) for v in box]
        detections.append(
            {
                "class": CLASSES[int(class_id)],
                "class_id": int(class_id),
                "confidence": round(float(score), 6),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "area_frac": round(
                    max(0.0, (x2 - x1) * (y2 - y1)) / (width * height), 6
                ),
            }
        )
    detections.sort(key=lambda row: row["confidence"], reverse=True)
    return detections, max_by_class


def audit_groups() -> dict[str, Any]:
    import onnxruntime as ort

    if not V1_RESULTS.is_file():
        raise RuntimeError(f"missing v1 live-audit results: {V1_RESULTS}")
    baseline = json.loads(V1_RESULTS.read_text())
    requested = ["tuckaway_photos", "video_foliage", "video_flowers"]
    baseline_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline["per_image"]:
        if row["group"] in requested:
            baseline_by_group[row["group"]].append(row)
    session = ort.InferenceSession(
        str(OUT_ROOT / "organ_gate.onnx"), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    all_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in requested:
        for old in baseline_by_group[group]:
            path = Path(old["file"])
            if not path.is_file():
                raise RuntimeError(f"external audit input missing: {path}")
            with Image.open(path) as original:
                image = original.convert("RGB")
            width, height = image.size
            tensor, scale, pad_x, pad_y = letterbox_app(image)
            raw = session.run(None, {input_name: tensor})[0]
            detections, max_scores = decode_onnx(raw, width, height, scale, pad_x, pad_y)
            thresholds = {}
            for threshold in CONF_THRESHOLDS:
                kept = [d for d in detections if d["confidence"] >= threshold]
                counts = Counter(d["class"] for d in kept)
                thresholds[f"{threshold:.2f}"] = {
                    "class_counts": dict(counts),
                    "total_boxes": len(kept),
                    "has_leaf": counts.get("leaf", 0) > 0,
                    "leaf_boxes": counts.get("leaf", 0),
                    "has_nonleaf_class": any(counts.get(name, 0) for name in NONLEAF),
                    "nonleaf_boxes": sum(counts.get(name, 0) for name in NONLEAF),
                    "detections": kept,
                }
            all_rows[group].append(
                {
                    "group": group,
                    "file": str(path),
                    "name": path.name,
                    "width": width,
                    "height": height,
                    "max_score_by_class": {
                        key: round(value, 6) for key, value in max_scores.items()
                    },
                    "thresholds": thresholds,
                }
            )

    def summary(group: str, threshold_key: str) -> dict[str, Any]:
        rows = all_rows[group]
        per = [row["thresholds"][threshold_key] for row in rows]
        counts: Counter[str] = Counter()
        for result in per:
            counts.update(result["class_counts"])
        n = len(per)
        leaf_detections = [
            det
            for result in per
            for det in result["detections"]
            if det["class"] == "leaf"
        ]
        leaf_areas = sorted(det["area_frac"] for det in leaf_detections)
        leaf_images_area_ge_05 = sum(
            any(
                det["class"] == "leaf" and det["area_frac"] >= 0.5
                for det in result["detections"]
            )
            for result in per
        )
        leaf_images_area_ge_09 = sum(
            any(
                det["class"] == "leaf" and det["area_frac"] >= 0.9
                for det in result["detections"]
            )
            for result in per
        )
        return {
            "n_images": n,
            "images_with_leaf": sum(result["has_leaf"] for result in per),
            "leaf_box_count": len(leaf_detections),
            "leaf_box_area_frac_median": round(leaf_areas[len(leaf_areas) // 2], 4)
            if leaf_areas
            else None,
            "images_with_leaf_box_area_ge_0.50": leaf_images_area_ge_05,
            "images_with_leaf_box_area_ge_0.90": leaf_images_area_ge_09,
            "leaf_box_area_ge_0.90_image_rate": round(leaf_images_area_ge_09 / n, 4)
            if n
            else None,
            "leaf_detection_rate": round(
                sum(result["has_leaf"] for result in per) / n, 4
            )
            if n
            else None,
            "leaf_boxes": sum(result["leaf_boxes"] for result in per),
            "mean_leaf_boxes_per_image": round(
                sum(result["leaf_boxes"] for result in per) / n, 4
            )
            if n
            else None,
            "images_with_nonleaf_class_candidate": sum(
                result["has_nonleaf_class"] for result in per
            ),
            "nonleaf_class_candidate_boxes": sum(result["nonleaf_boxes"] for result in per),
            "nonleaf_class_candidate_rate": round(
                sum(result["has_nonleaf_class"] for result in per) / n, 4
            )
            if n
            else None,
            "class_boxes": dict(counts),
        }

    v1_by_file = {row["file"]: row for row in baseline["per_image"]}
    comparisons: dict[str, Any] = {}
    for group in requested:
        comparisons[group] = {"n_images": len(all_rows[group]), "thresholds": {}}
        for threshold in CONF_THRESHOLDS:
            key = f"{threshold:.2f}"
            v1_key = "0.5" if threshold == 0.5 else key
            v2_summary = summary(group, key)
            v1_rows = [v1_by_file[row["file"]] for row in all_rows[group]]
            v1_leaf_images = sum(
                row["thresholds"][v1_key]["has_leaf"] for row in v1_rows
            )
            v1_nonleaf_images = sum(
                any(
                    row["thresholds"][v1_key]["class_counts"].get(name, 0) > 0
                    for name in NONLEAF
                )
                for row in v1_rows
            )
            v1_leaf_boxes = sum(row["thresholds"][v1_key]["leaf_count"] for row in v1_rows)
            v1_nonleaf_boxes = sum(
                sum(row["thresholds"][v1_key]["class_counts"].get(name, 0) for name in NONLEAF)
                for row in v1_rows
            )
            n = len(v1_rows)
            comparisons[group]["thresholds"][key] = {
                "v1": {
                    "images_with_leaf": v1_leaf_images,
                    "leaf_detection_rate": round(v1_leaf_images / n, 4) if n else None,
                    "leaf_boxes": v1_leaf_boxes,
                    "images_with_nonleaf_class_candidate": v1_nonleaf_images,
                    "nonleaf_class_candidate_rate": round(v1_nonleaf_images / n, 4)
                    if n
                    else None,
                    "nonleaf_class_candidate_boxes": v1_nonleaf_boxes,
                    "legacy_entire_plant_images": sum(
                        row["thresholds"][v1_key]["class_counts"].get("entire_plant", 0) > 0
                        for row in v1_rows
                    ),
                    "legacy_entire_plant_boxes": sum(
                        row["thresholds"][v1_key]["class_counts"].get("entire_plant", 0)
                        for row in v1_rows
                    ),
                    "class_boxes": {
                        name: sum(
                            row["thresholds"][v1_key]["class_counts"].get(name, 0)
                            for row in v1_rows
                        )
                        for name in ["leaf", *NONLEAF, "entire_plant"]
                    },
                },
                "v2": v2_summary,
                "leaf_rate_delta": round(
                    v2_summary["leaf_detection_rate"] - v1_leaf_images / n, 4
                )
                if n
                else None,
                "nonleaf_candidate_rate_delta": round(
                    v2_summary["nonleaf_class_candidate_rate"] - v1_nonleaf_images / n, 4
                )
                if n
                else None,
            }

    sheet_dir = OUT_ROOT / "external-audit" / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for group, rows in all_rows.items():
        chunk_size = 12 if group == "tuckaway_photos" else 20
        for part, start in enumerate(range(0, len(rows), chunk_size), start=1):
            chunk = rows[start : start + chunk_size]
            cols = 3
            cell_w, cell_h = 360, 310
            sheet = Image.new(
                "RGB",
                (cols * cell_w, math.ceil(len(chunk) / cols) * cell_h),
                (20, 24, 28),
            )
            draw = ImageDraw.Draw(sheet)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except Exception:
                font = ImageFont.load_default()
            for i, row in enumerate(chunk):
                with Image.open(row["file"]) as original:
                    image = original.convert("RGB")
                scale = min((cell_w - 8) / image.width, (cell_h - 34) / image.height)
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                x = (i % cols) * cell_w + (cell_w - image.width) // 2
                y = (i // cols) * cell_h + 2
                sheet.paste(image, (x, y))
                draw = ImageDraw.Draw(sheet)
                sx = image.width / row["width"]
                sy = image.height / row["height"]
                for det in row["thresholds"]["0.12"]["detections"]:
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    color = {
                        "leaf": (60, 230, 120),
                        "bud": (255, 220, 60),
                        "flower": (255, 80, 160),
                        "twig": (180, 140, 80),
                        "branch": (150, 110, 230),
                    }[det["class"]]
                    px = [x + x1 * sx, y + y1 * sy, x + x2 * sx, y + y2 * sy]
                    draw.rectangle(px, outline=color, width=2)
                    text = f"{det['class']} {det['confidence']:.2f}"
                    draw.rectangle([px[0], max(y, px[1] - 13), px[0] + 92, max(y, px[1] - 13) + 13], fill=(0, 0, 0))
                    draw.text((px[0] + 2, max(y, px[1] - 13)), text, fill=color, font=font)
                draw.rectangle(
                    [i % cols * cell_w, (i // cols) * cell_h + cell_h - 28, (i % cols + 1) * cell_w - 1, (i // cols + 1) * cell_h - 1],
                    fill=(0, 0, 0),
                )
                draw.text(
                    (i % cols * cell_w + 4, (i // cols) * cell_h + cell_h - 25),
                    f"v2 .12  {row['name']}",
                    fill=(240, 240, 240),
                    font=font,
                )
            sheet.save(sheet_dir / f"{group}_{part:02d}.jpg", quality=90)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(OUT_ROOT / "organ_gate.onnx"),
        "checkpoint": str(OUT_ROOT / "best.pt"),
        "contract": {
            "preprocess": "RGB letterbox 320x320, fill 114, /255, NCHW float32",
            "output": [1, 4 + len(CLASSES), 2100],
            "classes": CLASSES,
            "nms": f"class-wise IoU {NMS_IOU}, max_det {MAX_DET}",
        },
        "media_counts": {group: len(rows) for group, rows in all_rows.items()},
        "comparison": comparisons,
        "external_ground_truth": None,
        "interpretation": "These external media are unlabeled. Rates mean images with at least one model leaf detection, not recall. Candidate nonleaf classes are counts only, not audited false positives.",
        "per_image": [row for rows in all_rows.values() for row in rows],
        "sheets_dir": str(sheet_dir),
    }
    write_json(OUT_ROOT / "external_audit.json", payload)
    return payload


def build_metrics(
    train_info: dict[str, Any],
    standard_test: dict[str, Any],
    custom_test: dict[str, Any],
    onnx_info: dict[str, Any],
    external: dict[str, Any],
) -> dict[str, Any]:
    selected = custom_test["test_at_validation_selected_confidence"]
    field = selected["sources"]["leafsnap_field_leaf"]
    tuck = external["comparison"]["tuckaway_photos"]["thresholds"]["0.12"]
    v1_tuck = tuck["v1"]
    v2_tuck = tuck["v2"]
    material = (
        v2_tuck["images_with_leaf"] >= 10
        and v2_tuck["leaf_detection_rate"] - v1_tuck["leaf_detection_rate"] >= 0.20
    )
    recall_ok = field["recall"] >= 0.80
    false_positive_guard_assessed = external.get("external_ground_truth") is not None
    passed = recall_ok and material and false_positive_guard_assessed
    tuckaway_area = {
        key: v2_tuck[key]
        for key in (
            "leaf_box_count",
            "leaf_box_area_frac_median",
            "images_with_leaf_box_area_ge_0.50",
            "images_with_leaf_box_area_ge_0.90",
            "leaf_box_area_ge_0.90_image_rate",
        )
    }
    reasons = []
    if not recall_ok:
        reasons.append("LeafSnap field leaf recall is below 0.80")
    if not material:
        reasons.append(
            "Tuckaway .12 detection rate did not reach both >=10/37 images and +20 percentage points"
        )
    if tuckaway_area["images_with_leaf_box_area_ge_0.90"] >= 10:
        reasons.append(
            "most Tuckaway leaf detections are scene-scale boxes covering >=90% of the frame"
        )
    if not false_positive_guard_assessed:
        reasons.append("unlabeled external media cannot establish nonleaf false-positive change")
    payload = {
        "version": 2,
        "classes": CLASSES,
        "training": train_info,
        "test": {
            "v1_compatible_standard": standard_test,
            "v1_compatible_custom": custom_test,
            "field_leaf_gate": {
                "recall": field["recall"],
                "ap50": field["ap50"],
                "n_ground_truth": field["n_gt"],
                "recall_minimum": 0.80,
                "passes_recall": recall_ok,
                "evaluation": "custom greedy same-class IoU>=0.5, threshold selected on v2 validation only",
            },
        },
        "external_audit": {
            "path": str(OUT_ROOT / "external_audit.json"),
            "tuckaway_v1": v1_tuck,
            "tuckaway_v2": v2_tuck,
            "material_improvement_rule": ">=10/37 images with leaf at .12 and >=+0.20 absolute rate over v1",
            "material_improvement": material,
            "tuckaway_v2_leaf_box_area_diagnostic": tuckaway_area,
            "nonleaf_false_positive_increase": "not assessable: existing Tuckaway/video audit contains no human GT labels",
            "external_leaf_rates_are_recall": False,
        },
        "e3_v2": {
            "status": "PASS" if passed else "FAIL_NOT_ESTABLISHED",
            "passes": passed,
            "checks": {
                "leafsnap_field_leaf_recall_ge_0.80": recall_ok,
                "tuckaway_leaf_detection_materially_improves": material,
                "tuckaway_detections_not_scene_scale": tuckaway_area["images_with_leaf_box_area_ge_0.90"] < 10,
                "nonleaf_false_positive_change_assessed": false_positive_guard_assessed,
            },
            "reasons": reasons,
        },
        "model": {
            "checkpoint": str(OUT_ROOT / "best.pt"),
            "onnx": str(OUT_ROOT / "organ_gate.onnx"),
            "labels": str(OUT_ROOT / "organ_gate.labels.json"),
            "onnx_static_output": onnx_info["outputs"][0]["shape"],
            "onnx_precision": "FP16 weights/compute, static batch 1, float32 input/output where exporter uses casts",
        },
    }
    return payload


def write_report(metrics: dict[str, Any]) -> None:
    test = metrics["test"]["v1_compatible_custom"]
    selected = test["test_at_validation_selected_confidence"]
    field = metrics["test"]["field_leaf_gate"]
    external = metrics["external_audit"]
    train = metrics["training"]
    status = metrics["e3_v2"]["status"]
    lines = [
        "# Organ gate v2 training and audit",
        "",
        "## Result",
        "",
        f"**E3-v2: {status}.**",
        "",
        f"Reasons: {'; '.join(metrics['e3_v2']['reasons']) or 'all declared checks passed'}.",
        "",
        "The external Tuckaway and video frames are unlabeled. Their model detection rates",
        "are not recall, and candidate nonleaf-class fires are not counted as false positives.",
        "No field-accuracy claim is made from those media.",
        "",
        "## Training",
        "",
        f"- device: {train['device']} (MPS available: {train['mps_available']})",
        f"- epochs: {train['epochs_run']}/{train['epochs_requested']}; seed {train['seed']}; imgsz {train['imgsz']}; batch {train['batch']}",
        f"- elapsed training: {train['wall_clock_minutes']} min ({train['wall_clock_seconds']} s)",
        f"- run: `{train['run_dir']}`",
        "",
        "## V1-compatible held-out test",
        "",
        f"- images: {selected['n_images']}; validation-selected confidence: {selected['confidence_threshold']}",
        f"- overall at selected confidence: P {selected['overall']['precision']}, R {selected['overall']['recall']}, F1 {selected['overall']['f1']}",
        f"- custom macro AP50: {selected['map50_iou0.5']}; Ultralytics mAP50: {metrics['test']['v1_compatible_standard']['box']['map50']}; mAP50-95: {metrics['test']['v1_compatible_standard']['box']['map50_95']}",
        f"- LeafSnap FIELD leaf: recall **{field['recall']}** (gate >= 0.80), AP50 **{field['ap50']}**, GT boxes {field['n_ground_truth']}",
        "",
        "| class | P @ val threshold | R @ val threshold | F1 @ val threshold | Ultralytics AP50 | n GT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CLASSES:
        row = selected["per_class"][name]
        official = metrics["test"]["v1_compatible_standard"]["per_class"][name]
        lines.append(
            f"| {name} | {row['precision']} | {row['recall']} | {row['f1']} | {official['ap50']} | {row['n_gt']} |"
        )
    lines += [
        "",
        "P/R/F1 use the confidence selected on validation only. AP50 is threshold-free.",
        "BioImages rows use near-full-frame weak boxes; their AP is not localization quality.",
        "",
        "## External live-audit comparison",
        "",
        "Rates are image-level detections (not recall). Nonleaf column means at least one",
        "bud/flower/twig/branch class candidate, not a false-positive label.",
        "",
        "| media | n | conf | v1 leaf images | v1 rate | v2 leaf images | v2 rate | delta | v1 nonleaf candidate rate | v2 nonleaf candidate rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("tuckaway_photos", "video_foliage", "video_flowers"):
        for key in ("0.12", "0.25", "0.50"):
            entry = metrics["external_audit"]
            comparison = json.loads((OUT_ROOT / "external_audit.json").read_text())["comparison"][group]["thresholds"][key]
            lines.append(
                f"| {group} | {comparison['v2']['n_images']} | {key} | "
                f"{comparison['v1']['images_with_leaf']} | {comparison['v1']['leaf_detection_rate']:.3f} | "
                f"{comparison['v2']['images_with_leaf']} | {comparison['v2']['leaf_detection_rate']:.3f} | "
                f"{comparison['leaf_rate_delta']:+.3f} | "
                f"{comparison['v1']['nonleaf_class_candidate_rate']:.3f} | "
                f"{comparison['v2']['nonleaf_class_candidate_rate']:.3f} |"
            )
    lines += [
        "",
        f"Tuckaway .12 material-improvement rule: {external['material_improvement_rule']}. Result: **{external['material_improvement']}**.",
        f"At .12, v2 leaf detections cover >=90% of the frame in {external['tuckaway_v2_leaf_box_area_diagnostic']['images_with_leaf_box_area_ge_0.90']}/{external['tuckaway_v2']['n_images']} Tuckaway photos; median leaf-box area is {external['tuckaway_v2_leaf_box_area_diagnostic']['leaf_box_area_frac_median']:.3f} of the image. This is scene-scale output, not evidence of leaf localization; the raw detection-rate gain is therefore not accepted as a field-accuracy result.",
        f"v1 `entire_plant` detections occur in {external['tuckaway_v1']['legacy_entire_plant_images']}/37 Tuckaway photos at .12; v2 has no such class but its scene-scale leaf boxes warrant the same composition-confound review.",
        f"Nonleaf false-positive change: **{external['nonleaf_false_positive_increase']}**.",
        "",
        "## Artifacts",
        "",
        f"- checkpoint: `{metrics['model']['checkpoint']}`",
        f"- static FP16 ONNX: `{metrics['model']['onnx']}`",
        f"- labels: `{metrics['model']['labels']}`",
        f"- model output: `{metrics['model']['onnx_static_output']}`",
        f"- dataset counts/validation: `{DATASET_DIR / 'counts.json'}`, `{DATASET_DIR / 'validation.json'}`",
        f"- E1 composite contact sheet: `{OUT_ROOT / 'composites_contact_sheet.jpg'}`",
        f"- external annotated sheets: `{OUT_ROOT / 'external-audit' / 'sheets'}`",
        f"- data documentation: `{HERE / 'ORGAN_GATE_V2_DATA.md'}`",
        "",
        "## Limits",
        "",
        "- BioImages annotations are weak near-full-frame boxes.",
        "- The synthetic scenes improve multi-leaf scale/context variety, but they do not",
        "  replace independent, human-boxed field evaluation.",
        "- Tuckaway/video images have no image-level or box-level ground truth in the existing",
        "  live audit. The E3 gate is not established until nonleaf false positives are reviewed",
        "  on labeled external negatives.",
        "",
    ]
    (OUT_ROOT / "ORGAN_GATE_V2.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-existing", action="store_true")
    args = parser.parse_args()
    if not OUT_ROOT.is_dir() or not (DATASET_DIR / "manifest.jsonl").is_file():
        raise SystemExit("build the v2 dataset first")
    rows = load_manifest(DATASET_DIR / "manifest.jsonl")
    compatible_yaml, compatible_rows = build_v1_compatible_test(rows)
    data_yaml = DATASET_DIR / "dataset.yaml"
    best_pt = OUT_ROOT / "best.pt"
    if args.postprocess_existing:
        params_path = OUT_ROOT / "training_params.json"
        if not best_pt.is_file() or not params_path.is_file():
            raise SystemExit("existing v2 checkpoint and training parameters are required")
        train_info = json.loads(params_path.read_text())
    else:
        train_info = train_model(data_yaml)
    standard = standard_test_metrics(best_pt, compatible_yaml)
    custom = run_custom_eval(best_pt, rows, compatible_rows)
    onnx_info = export_onnx(best_pt)
    external = audit_groups()
    metrics = build_metrics(train_info, standard, custom, onnx_info, external)
    write_json(OUT_ROOT / "metrics.json", metrics)
    write_report(metrics)
    print(
        json.dumps(
            {
                "training_minutes": train_info["wall_clock_minutes"],
                "epochs_run": train_info["epochs_run"],
                "field_leaf_recall": metrics["test"]["field_leaf_gate"]["recall"],
                "field_leaf_ap50": metrics["test"]["field_leaf_gate"]["ap50"],
                "e3_v2": metrics["e3_v2"],
                "checkpoint": metrics["model"]["checkpoint"],
                "onnx": metrics["model"]["onnx"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
