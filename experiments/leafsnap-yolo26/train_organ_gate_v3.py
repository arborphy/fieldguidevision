# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1_ROOT = ROOT / "scratch" / "leafsnap-yolo26"
OUT_ROOT = V1_ROOT / "organ-gate-v3"
DATASET = OUT_ROOT / "dataset"
RUNS = OUT_ROOT / "runs"
CLASSES = ["leaf", "bud", "flower", "twig", "branch"]
CLASS_IDS = {name: index for index, name in enumerate(CLASSES)}
NONLEAF = ["bud", "flower", "twig", "branch"]
SEED = 260930
IMGSZ = 320
BATCH = 16
EPOCHS = 18
CONF_FLOOR = 0.001
THRESHOLDS = [0.12, 0.25, 0.50]
NMS_IOU = 0.45
MAX_DET = 300
V1_ONNX = V1_ROOT / "organ-gate" / "organ_gate.onnx"
V2_ONNX = V1_ROOT / "organ-gate-v2" / "organ_gate.onnx"
V1_AUDIT = V1_ROOT / "live-audit" / "results.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_versions() -> dict[str, str]:
    import torch
    import ultralytics
    import onnxruntime
    return {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "onnxruntime": onnxruntime.__version__, "numpy": np.__version__}


def load_manifest() -> list[dict[str, Any]]:
    return read_jsonl(DATASET / "manifest.jsonl")


def symlink_field_test(rows: list[dict[str, Any]]) -> Path:
    import shutil as file_ops
    images = DATASET / "images" / "leafsnap_field_test"
    labels = DATASET / "labels" / "leafsnap_field_test"
    for target in (images, labels):
        if target.exists():
            file_ops.rmtree(target)
        target.mkdir(parents=True)
    selected = [row for row in rows if row["source"] == "leafsnap" and row["domain"] == "field" and row["split"] == "test"]
    for row in selected:
        (images / Path(row["image"]).name).symlink_to((DATASET / row["image"]).resolve())
        (labels / Path(row["label"]).name).symlink_to((DATASET / row["label"]).resolve())
    yaml = DATASET / "leafsnap_field_test.yaml"
    yaml.write_text(f"path: {DATASET}\ntrain: images/train\nval: images/val\ntest: images/leafsnap_field_test\nnc: 5\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)))
    return yaml


def metrics_block(metric: Any, names: list[str]) -> dict[str, Any]:
    class_indices = [int(value) for value in getattr(metric, "ap_class_index", [])]
    per_class = {}
    for pos, class_id in enumerate(class_indices):
        name = names[class_id] if class_id < len(names) else str(class_id)
        precision = getattr(metric, "p", [])
        recall = getattr(metric, "r", [])
        ap50 = getattr(metric, "ap50", [])
        ap = getattr(metric, "ap", [])
        per_class[name] = {
            "class_id": class_id,
            "precision": round(float(precision[pos]), 4) if len(precision) > pos else None,
            "recall": round(float(recall[pos]), 4) if len(recall) > pos else None,
            "ap50": round(float(ap50[pos]), 4) if len(ap50) > pos else None,
            "ap50_95": round(float(ap[pos]), 4) if len(ap) > pos else None,
        }
    return {
        "precision": round(float(getattr(metric, "mp", 0.0)), 4),
        "recall": round(float(getattr(metric, "mr", 0.0)), 4),
        "map50": round(float(getattr(metric, "map50", 0.0)), 4),
        "map50_95": round(float(getattr(metric, "map", 0.0)), 4),
        "per_class": per_class,
    }


def run_validation(model_path: Path, yaml: Path, name: str) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(model_path))
    result = model.val(data=str(yaml), split="test", imgsz=IMGSZ, batch=BATCH, device=device, workers=0, conf=CONF_FLOOR, iou=0.7, plots=False, verbose=False, project=str(RUNS), name=name, exist_ok=True)
    names = [result.names[index] for index in range(len(result.names))] if isinstance(result.names, dict) else list(result.names)
    box = metrics_block(result.box, names)
    seg_metric = getattr(result, "seg", None)
    seg = metrics_block(seg_metric, names) if seg_metric is not None else None
    nt = getattr(result, "nt_per_class", None)
    if nt is not None:
        for name_key, item in box["per_class"].items():
            item["n_gt"] = int(nt[item["class_id"]]) if item["class_id"] < len(nt) else None
        if seg is not None:
            for name_key, item in seg["per_class"].items():
                item["n_gt"] = int(nt[item["class_id"]]) if item["class_id"] < len(nt) else None
    image_dir = "leafsnap_field_test" if name == "v3-leafsnap-field-test" else "test"
    return {"images": sum(1 for _ in (DATASET / "images" / image_dir).iterdir()), "imgsz": IMGSZ, "box": box, "segmentation": seg, "speed_ms_per_image": {key: round(float(value), 3) for key, value in result.speed.items()}}


def train() -> dict[str, Any]:
    import torch
    from ultralytics import YOLO
    if not (DATASET / "validation.json").is_file() or not read_json(DATASET / "validation.json").get("ok"):
        raise RuntimeError("V3 dataset validation must pass before training")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the requested V3 run")
    run_dir = RUNS / "train_v3"
    if run_dir.exists():
        raise RuntimeError(f"refusing duplicate training run: {run_dir}")
    RUNS.mkdir(parents=True, exist_ok=True)
    params = {
        "experiment": "leafsnap-yolo26 organ gate v3 segmentation-supervised",
        "task": "segment",
        "pretrained": "yolo26n-seg.pt",
        "device": "mps",
        "epochs_requested": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "seed": SEED,
        "deterministic": True,
        "workers": 0,
        "amp": False,
        "cache": False,
        "data": str(DATASET / "dataset.yaml"),
        "dataset_fingerprint": {"manifest_sha256": sha256_file(DATASET / "manifest.jsonl"), "counts_sha256": sha256_file(DATASET / "counts.json"), "validation_sha256": sha256_file(DATASET / "validation.json"), "rows": len(load_manifest())},
        "packages": package_versions(),
        "hardware": {"machine": platform.machine(), "processor": platform.processor(), "device": "mps"},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_policy": "exactly one fixed-seed 18-epoch fine-tuning run; no smoke run or retry",
    }
    write_json(OUT_ROOT / "training_params.json", params)
    seed_everything(SEED)
    model = YOLO("yolo26n-seg.pt")
    started = time.perf_counter()
    model.train(data=str(DATASET / "dataset.yaml"), epochs=EPOCHS, patience=EPOCHS, imgsz=IMGSZ, batch=BATCH, device="mps", workers=0, seed=SEED, deterministic=True, amp=False, cache=False, project=str(RUNS), name="train_v3", exist_ok=False, plots=False, verbose=True, val=True, save=True, save_period=-1)
    elapsed = time.perf_counter() - started
    best_source = run_dir / "weights" / "best.pt"
    last_source = run_dir / "weights" / "last.pt"
    if not best_source.is_file():
        raise RuntimeError("training did not produce best.pt")
    best = OUT_ROOT / "best.pt"
    shutil.copy2(best_source, best)
    if last_source.is_file():
        shutil.copy2(last_source, OUT_ROOT / "last.pt")
    history_path = run_dir / "results.csv"
    history = []
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        shutil.copy2(history_path, OUT_ROOT / "history.csv")
    params.update({"completed_at": datetime.now(timezone.utc).isoformat(), "epochs_run": len(history), "wall_clock_seconds": round(elapsed, 2), "wall_clock_minutes": round(elapsed / 60, 2), "early_stopped": len(history) < EPOCHS, "best_checkpoint": str(best), "run_dir": str(run_dir), "checkpoint_sha256": sha256_file(best)})
    write_json(OUT_ROOT / "training_params.json", params)
    return params


def letterbox(image: Image.Image) -> tuple[np.ndarray, float, int, int]:
    width, height = image.size
    scale = min(IMGSZ / width, IMGSZ / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (IMGSZ, IMGSZ), (114, 114, 114))
    pad_x, pad_y = (IMGSZ - new_width) // 2, (IMGSZ - new_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    data = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(data), scale, pad_x, pad_y


def classwise_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> list[int]:
    order = scores.argsort()[::-1]
    kept = []
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
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_current = max((boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1]), 1e-9)
        area_rest = np.clip(boxes[remaining, 2] - boxes[remaining, 0], 0, None) * np.clip(boxes[remaining, 3] - boxes[remaining, 1], 0, None)
        iou = inter / np.maximum(area_current + area_rest - inter, 1e-9)
        order = remaining[(iou <= NMS_IOU) | (classes[remaining] != classes[current])]
    return kept


def decode_raw(output: np.ndarray, width: int, height: int, scale: float, pad_x: int, pad_y: int, nc: int) -> list[dict[str, Any]]:
    values = np.asarray(output, dtype=np.float32)
    if values.ndim == 3:
        values = values[0]
    if values.shape[0] < 4 + nc and values.shape[1] >= 4 + nc:
        values = values.T
    if values.shape[0] < 4 + nc:
        raise ValueError(f"invalid YOLO output shape: {values.shape}")
    xywh = values[:4].T
    score_values = values[4:4 + nc].T
    classes = score_values.argmax(axis=1)
    scores = score_values.max(axis=1)
    selected = scores >= CONF_FLOOR
    xywh, classes, scores = xywh[selected], classes[selected], scores[selected]
    xyxy = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], axis=1)
    kept = classwise_nms(xyxy, scores, classes)
    xyxy, scores, classes = xyxy[kept], scores[kept], classes[kept]
    xyxy = (xyxy - np.asarray([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)) / max(scale, 1e-9)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
    output_rows = []
    for box, score, class_id in zip(xyxy, scores, classes):
        x1, y1, x2, y2 = [float(value) for value in box]
        output_rows.append({"class": CLASSES[int(class_id)] if int(class_id) < len(CLASSES) else "entire_plant", "class_id": int(class_id), "confidence": float(score), "bbox_xyxy": [x1, y1, x2, y2], "area_frac": max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)})
    output_rows.sort(key=lambda row: row["confidence"], reverse=True)
    return output_rows


def topologically_sort_onnx(model: Any) -> Any:
    graph = model.graph
    available = {value.name for value in graph.input} | {value.name for value in graph.initializer} | {value.values.name for value in graph.sparse_initializer}
    pending = list(graph.node)
    ordered = []
    while pending:
        ready = [node for node in pending if all(not name or name in available for name in node.input)]
        if not ready:
            missing = sorted({name for node in pending for name in node.input if name and name not in available})
            raise RuntimeError(f"ONNX graph cannot be topologically sorted; unresolved inputs: {missing[:20]}")
        ready_ids = {id(node) for node in ready}
        for node in ready:
            ordered.append(node)
            available.update(name for name in node.output if name)
        pending = [node for node in pending if id(node) not in ready_ids]
    graph.ClearField("node")
    graph.node.extend(ordered)
    return model


def export_onnx(best: Path) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    from ultralytics import YOLO
    exported = Path(YOLO(str(best)).export(format="onnx", imgsz=IMGSZ, opset=17, half=True, dynamic=False, simplify=False, batch=1, device="cpu"))
    target = OUT_ROOT / "organ_gate.onnx"
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
        exported.unlink(missing_ok=True)
    graph = topologically_sort_onnx(onnx.load(str(target)))
    onnx.save(graph, str(target))
    onnx.checker.check_model(graph)
    def dims(value: Any) -> list[int | str]:
        return [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
    inputs = [{"name": item.name, "shape": dims(item), "dtype": onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type)} for item in graph.graph.input]
    outputs = [{"name": item.name, "shape": dims(item), "dtype": onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type)} for item in graph.graph.output]
    if len(inputs) != 1 or inputs[0]["shape"] != [1, 3, IMGSZ, IMGSZ]:
        raise RuntimeError(f"unexpected input signature: {inputs}")
    if len(outputs) != 2 or outputs[0]["shape"] != [1, 41, 2100] or outputs[1]["shape"] != [1, 32, 80, 80]:
        raise RuntimeError(f"unexpected segmentation output signature: {outputs}")
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    result = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32)})
    if [list(item.shape) for item in result] != [[1, 41, 2100], [1, 32, 80, 80]] or not all(np.isfinite(item).all() for item in result):
        raise RuntimeError("static ONNX segmentation smoke inference failed")
    labels = {
        "dataset": "LeafSnap + reviewed BioImages/Tuckaway organ gate V3",
        "task": "segment",
        "model": "yolo26n-seg fine-tuned",
        "class_order": {str(index): name for index, name in enumerate(CLASSES)},
        "names": {name: index for index, name in enumerate(CLASSES)},
        "nc": len(CLASSES),
        "input": {"name": inputs[0]["name"], "shape": inputs[0]["shape"], "dtype": inputs[0]["dtype"], "layout": "NCHW", "preprocessing": "RGB letterbox 320x320, 114 fill, float32 [0,1]"},
        "outputs": [{"name": outputs[0]["name"], "shape": outputs[0]["shape"], "layout": "[batch, 4 + nc + nm, anchors]", "meaning": "cx,cy,w,h,5 class scores,32 mask coefficients"}, {"name": outputs[1]["name"], "shape": outputs[1]["shape"], "layout": "[batch,32,80,80]", "meaning": "segmentation prototype masks"}],
        "onnx": {"file": "organ_gate.onnx", "static": True, "batch": 1, "imgsz": IMGSZ, "half": True, "output0": [1, 41, 2100], "output1": [1, 32, 80, 80], "onnx_checker": "pass", "smoke_inference": "pass"},
    }
    write_json(OUT_ROOT / "organ_gate.labels.json", labels)
    return {"checkpoint": str(best), "onnx": str(target), "bytes": target.stat().st_size, "inputs": inputs, "outputs": outputs, "labels": str(OUT_ROOT / "organ_gate.labels.json"), "smoke_output_shapes": [list(item.shape) for item in result]}


def baseline_model_paths() -> dict[str, tuple[Path, int, list[str]]]:
    for path in (V1_ONNX, V2_ONNX):
        if not path.is_file():
            raise RuntimeError(f"missing baseline ONNX: {path}")
    return {"v1": (V1_ONNX, 6, ["leaf", "bud", "flower", "twig", "branch", "entire_plant"]), "v2": (V2_ONNX, 5, CLASSES.copy()), "v3": (OUT_ROOT / "organ_gate.onnx", 5, CLASSES.copy())}


def summarize_detections(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    detections = [[d for d in row["detections"] if d["confidence"] >= threshold] for row in rows]
    leaf_per_image = [any(d["class"] == "leaf" for d in group) for group in detections]
    leaf_boxes = [d for group in detections for d in group if d["class"] == "leaf"]
    all_boxes = [d for group in detections for d in group]
    nonleaf_per_image = [any(d["class"] in NONLEAF for d in group) for group in detections]
    class_counts = Counter(d["class"] for group in detections for d in group)
    leaf_areas = sorted(float(d["area_frac"]) for d in leaf_boxes)
    all_areas = sorted(float(d["area_frac"]) for d in all_boxes)
    def quantile(values: list[float], q: float) -> float | None:
        return round(float(np.quantile(values, q)), 6) if values else None
    per_class_area = {}
    for name in CLASSES:
        boxes = [d for d in all_boxes if d["class"] == name]
        areas = [float(d["area_frac"]) for d in boxes]
        per_class_area[name] = {
            "box_count": len(boxes),
            "median": quantile(areas, 0.5),
            "p90": quantile(areas, 0.9),
            "ge_0.90_count": sum(area >= 0.90 for area in areas),
            "ge_0.90_image_count": sum(any(d["class"] == name and d["area_frac"] >= 0.90 for d in group) for group in detections),
        }
    return {
        "n_images": len(rows),
        "images_with_leaf": sum(leaf_per_image),
        "leaf_detection_rate": round(sum(leaf_per_image) / len(rows), 4) if rows else None,
        "leaf_box_count": len(leaf_boxes),
        "leaf_box_area_median": quantile(leaf_areas, 0.5),
        "leaf_box_area_p90": quantile(leaf_areas, 0.9),
        "leaf_box_area_ge_0.90_count": sum(area >= 0.90 for area in leaf_areas),
        "leaf_box_area_ge_0.90_image_count": sum(any(d["class"] == "leaf" and d["area_frac"] >= 0.90 for d in group) for group in detections),
        "all_box_count": len(all_boxes),
        "all_box_area_median": quantile(all_areas, 0.5),
        "all_box_area_p90": quantile(all_areas, 0.9),
        "all_box_area_ge_0.90_count": sum(area >= 0.90 for area in all_areas),
        "all_box_area_ge_0.90_image_count": sum(any(d["area_frac"] >= 0.90 for d in group) for group in detections),
        "per_class_box_area": per_class_area,
        "images_with_nonleaf_candidate": sum(nonleaf_per_image),
        "nonleaf_candidate_rate": round(sum(nonleaf_per_image) / len(rows), 4) if rows else None,
        "nonleaf_box_count": sum(class_counts[name] for name in NONLEAF),
        "class_box_counts": dict(class_counts),
    }


def external_sources(training_assignments: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    baseline = read_json(V1_AUDIT)
    groups = {name: [] for name in ("tuckaway_photos", "video_foliage", "video_flowers")}
    trained_names = {Path(image_id).name for image_id, split in training_assignments.items() if split == "train"}
    for row in baseline["per_image"]:
        group = row["group"]
        if group not in groups:
            continue
        if group == "tuckaway_photos" and Path(row["file"]).name in trained_names:
            continue
        groups[group].append(row)
    return groups


def render_external_sheet(rows: list[dict[str, Any]], path: Path, group: str) -> None:
    if not rows:
        return
    columns, cell_w, cell_h = 3, 360, 310
    sheet = Image.new("RGB", (columns * cell_w, math.ceil(len(rows) / columns) * cell_h), (20, 24, 28))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
    for index, row in enumerate(rows):
        image = row["image"]
        image.thumbnail((cell_w - 8, cell_h - 38))
        ox = index % columns * cell_w + (cell_w - image.width) // 2
        oy = index // columns * cell_h + 2
        sheet.paste(image, (ox, oy))
        sx, sy = image.width / row["width"], image.height / row["height"]
        for det in row["detections"]:
            if det["confidence"] < 0.12:
                continue
            x1, y1, x2, y2 = det["bbox_xyxy"]
            color = {"leaf": (50, 235, 125), "bud": (255, 220, 60), "flower": (255, 70, 165), "twig": (180, 140, 80), "branch": (155, 110, 230)}.get(det["class"], (220, 220, 220))
            coords = [ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy]
            draw.rectangle(coords, outline=color, width=2)
            draw.text((coords[0], max(oy, coords[1] - 13)), f"{det['class']} {det['confidence']:.2f}", fill=color, font=font)
        y = index // columns * cell_h + cell_h - 28
        draw.rectangle([index % columns * cell_w, y, (index % columns + 1) * cell_w - 1, (index // columns + 1) * cell_h - 1], fill=(0, 0, 0))
        draw.text((index % columns * cell_w + 4, y + 3), f"V3 .12 | {Path(row['file']).name}"[:54], fill=(240, 240, 240), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def external_audit(training_assignments: dict[str, str]) -> dict[str, Any]:
    import onnxruntime as ort
    model_paths = baseline_model_paths()
    sessions = {name: (ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]), nc, names) for name, (path, nc, names) in model_paths.items()}
    groups = external_sources(training_assignments)
    output = {"generated_at": datetime.now(timezone.utc).isoformat(), "groups": {}, "comparison": {}, "per_image": {}, "models": {name: str(info[0]) for name, info in model_paths.items()}, "interpretation": "External detection rates are image-level candidate rates, not recall. Flower-video leaf fires are treated as false-leaf candidates by the flower-focused source definition. V1, V2, and V3 were re-inferred on the same held-out images, preprocessing, confidence floor, and class-wise NMS."}
    for group_name, group_rows in groups.items():
        audited = []
        for source_row in group_rows:
            path = Path(source_row["file"])
            if not path.is_file():
                raise RuntimeError(f"external audit image missing: {path}")
            with Image.open(path) as image_handle:
                image = image_handle.convert("RGB")
            width, height = image.size
            tensor, scale, pad_x, pad_y = letterbox(image)
            detections_by_model = {}
            for model_name, (session, nc, names) in sessions.items():
                raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
                detections = decode_raw(raw, width, height, scale, pad_x, pad_y, nc)
                for detection in detections:
                    detection["class"] = names[detection["class_id"]] if detection["class_id"] < len(names) else str(detection["class_id"])
                detections_by_model[model_name] = detections
            audited.append({"file": str(path), "name": path.name, "width": width, "height": height, "image": image, "models": detections_by_model})
        output["groups"][group_name] = {"n_images": len(audited)}
        output["per_image"][group_name] = []
        for model_name in ("v1", "v2", "v3"):
            rows = [{"detections": row["models"][model_name]} for row in audited]
            thresholds = {}
            for threshold in THRESHOLDS:
                thresholds[f"{threshold:.2f}"] = summarize_detections(rows, threshold)
            output["groups"][group_name][model_name] = {"thresholds": thresholds}
        for row in audited:
            output["per_image"][group_name].append({"file": row["file"], "name": row["name"], "width": row["width"], "height": row["height"], "detections": row["models"]["v3"], "image": row["image"]})
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            output["comparison"].setdefault(group_name, {})[key] = {
                "n_images": len(audited),
                **{model_name: output["groups"][group_name][model_name]["thresholds"][key] for model_name in ("v1", "v2", "v3")},
            }
    for group_name, rows in output["per_image"].items():
        render_external_sheet(rows, OUT_ROOT / "external-audit" / "sheets" / f"{group_name}.jpg", group_name)
        for row in rows:
            row.pop("image", None)
    write_json(OUT_ROOT / "external_audit.json", output)
    return output


def build_metrics(train_info: dict[str, Any], full_test: dict[str, Any], field_test: dict[str, Any], onnx_info: dict[str, Any], external: dict[str, Any]) -> dict[str, Any]:
    comparisons = external["comparison"]
    flower = comparisons.get("video_flowers", {})
    flower_false_leaf = {key: {model: flower.get(key, {}).get(model, {}).get("leaf_detection_rate") for model in ("v1", "v2", "v3")} for key in ("0.12", "0.25", "0.50")}
    tuck = comparisons.get("tuckaway_photos", {})
    return {
        "version": 3,
        "classes": CLASSES,
        "training": train_info,
        "test": {"all_heldout": full_test, "leafsnap_field_only": field_test, "leafsnap_field_leaf_box_recall": field_test["box"]["per_class"].get("leaf", {}).get("recall"), "leafsnap_field_leaf_box_ap50": field_test["box"]["per_class"].get("leaf", {}).get("ap50"), "leafsnap_field_leaf_segmentation_ap50": (field_test.get("segmentation") or {}).get("per_class", {}).get("leaf", {}).get("ap50")},
        "external_audit": external,
        "flower_video_false_leaf_rate": flower_false_leaf,
        "fit_for_app_integration": False,
        "integration_decision": "Not fit unless external Tuckaway and foliage-video leaf detection improves over V1/V2 at operational confidence and leaf boxes remain localized. This decision is computed in the report from held-out group results; external rates are not recall.",
        "model": {"checkpoint": str(OUT_ROOT / "best.pt"), "onnx": str(OUT_ROOT / "organ_gate.onnx"), "labels": str(OUT_ROOT / "organ_gate.labels.json"), "output0": onnx_info["outputs"][0]["shape"], "output1": onnx_info["outputs"][1]["shape"], "onnx_static_fp16": True},
        "box_area_policy": "External leaf and all-class box area fractions are measured in original-frame coordinates after inverse letterbox; per-group median, p90, and >=0.90-frame counts are recorded at .12/.25/.50 for leaf and each class.",
        "limitations": ["External media have no complete human box annotations; rates are detections, not recall.", "BioImages leaf masks are reviewed SAM proposals; nonleaf rectangular masks are explicitly weak.", "Tuckaway stills are split by image, and all foliage/flower video sources are held out from training.", "External V1/V2/V3 metrics are raw ONNX re-inferences on the same held-out images with common preprocessing and class-wise NMS."],
    }


def write_report(metrics: dict[str, Any]) -> None:
    data_counts = read_json(DATASET / "counts.json")
    external = metrics["external_audit"]["comparison"]
    field = metrics["test"]["leafsnap_field_only"]
    lines = [
        "# Organ gate V3 segmentation-supervised run",
        "",
        "## Decision",
        "",
        f"**App integration: {'FIT' if metrics['fit_for_app_integration'] else 'NOT FIT / NOT ESTABLISHED'}.**",
        "",
        "V3 is only considered useful if held-out external leaf detection improves materially over V1/V2 while boxes remain localized. External media are unlabeled; detection rates are not recall.",
        "",
        "## Training",
        "",
        f"- model: `yolo26n-seg.pt`; device: {metrics['training']['device']}; epochs: {metrics['training'].get('epochs_run')}/{metrics['training']['epochs_requested']}; seed: {metrics['training']['seed']}; imgsz: {metrics['training']['imgsz']}; batch: {metrics['training']['batch']}",
        f"- duration: {metrics['training'].get('wall_clock_minutes')} minutes; run: `{metrics['training'].get('run_dir')}`",
        f"- dataset: {data_counts['total_images']} images / {data_counts['total_instances']} instances; no synthetic data; no weak leaf boxes.",
        "",
        "## Held-out LeafSnap FIELD leaf test",
        "",
        f"- images: {field['images']}; box recall: **{metrics['test']['leafsnap_field_leaf_box_recall']}**; box AP50: **{metrics['test']['leafsnap_field_leaf_box_ap50']}**; mask AP50: **{metrics['test']['leafsnap_field_leaf_segmentation_ap50']}**.",
        "- All-class test box and segmentation mAP are recorded in `metrics.json`; the table below is the FIELD-only mask/box evaluation.",
        "",
        "| metric | precision | recall | AP50 | AP50:95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, block in (("box", field["box"]), ("segmentation", field.get("segmentation") or {})):
        row = block.get("per_class", {}).get("leaf", {})
        lines.append(f"| {label} leaf | {row.get('precision')} | {row.get('recall')} | {row.get('ap50')} | {row.get('ap50_95')} |")
    lines += ["", "## Held-out external comparison", "", "Leaf rates are image-level detections, not recall. Nonleaf values are class candidates, not verified false positives. Median/p90 are predicted leaf-box area fractions in original-frame coordinates; >=90% is scene-scale. The all-class scene-scale column catches oversized nonleaf predictions too.", "", "| source | n | threshold | model | leaf images/rate | leaf boxes | leaf median/p90 | leaf >=90% images | any-class >=90% images | nonleaf candidate rate |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for group in ("tuckaway_photos", "video_foliage", "video_flowers"):
        for key in ("0.12", "0.25", "0.50"):
            for model in ("v1", "v2", "v3"):
                item = external.get(group, {}).get(key, {}).get(model, {})
                lines.append(f"| {group} | {item.get('n_images')} | {key} | {model} | {item.get('images_with_leaf')}/{item.get('n_images')} ({item.get('leaf_detection_rate')}) | {item.get('leaf_box_count')} | {item.get('leaf_box_area_median')}/{item.get('leaf_box_area_p90')} | {item.get('leaf_box_area_ge_0.90_image_count')} | {item.get('all_box_area_ge_0.90_image_count')} | {item.get('nonleaf_candidate_rate')} |")
    lines += ["", "Flower-video false-leaf candidate rate (source is the flower-only clip; any leaf candidate counts as a false-leaf candidate):", "", "| threshold | V1 | V2 | V3 |", "|---|---:|---:|---:|"]
    for key, values in metrics["flower_video_false_leaf_rate"].items():
        lines.append(f"| {key} | {values['v1']} | {values['v2']} | {values['v3']} |")
    lines += ["", "## Artifacts", "", f"- best checkpoint: `{metrics['model']['checkpoint']}`", f"- static FP16 ONNX: `{metrics['model']['onnx']}`", f"- labels: `{metrics['model']['labels']}`", f"- output contract: output0 `{metrics['model']['output0']}`, proto output1 `{metrics['model']['output1']}`", f"- dataset and E1 review: `{OUT_ROOT / 'ORGAN_GATE_V3_DATA.md'}`, `{OUT_ROOT / 'e1_contact_sheets'}`", f"- external sheets: `{OUT_ROOT / 'external-audit' / 'sheets'}`", "", "## Caveats", "", "- LeafSnap FIELD test is a curated-source test; segmentation contours derive from LeafSnap source masks.", "- BioImages leaf labels are SAM proposals selected with source image-level `organ_tag=leaf`; they are not human-traced ground truth.", "- BioImages nonleaf labels are weak rectangle polygons and are not localization-quality labels.", "- Tuckaway photographs assigned to training are removed from the external test group. Entire foliage and flower video sources remain held out.", "- A lower external detection rate may indicate reduced sensitivity; a higher rate with scene-scale boxes is not accepted as useful localization.", ""]
    (OUT_ROOT / "ORGAN_GATE_V3.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    if not (DATASET / "manifest.jsonl").is_file():
        raise SystemExit("build the V3 segmentation dataset first")
    rows = load_manifest()
    counts = read_json(DATASET / "counts.json")
    assignments = counts["real_photo_split"]["assignments"]
    field_yaml = symlink_field_test(rows)
    best = OUT_ROOT / "best.pt"
    if args.postprocess_only:
        if not best.is_file() or not (OUT_ROOT / "training_params.json").is_file():
            raise SystemExit("postprocess-only requires the completed single-run checkpoint and metadata")
        train_info = read_json(OUT_ROOT / "training_params.json")
    else:
        if best.is_file():
            raise SystemExit("best.pt already exists; refusing a second training run; use --postprocess-only")
        train_info = train()
    full_test = run_validation(best, DATASET / "dataset.yaml", "v3-full-test")
    field_test = run_validation(best, field_yaml, "v3-leafsnap-field-test")
    onnx = export_onnx(best)
    external = external_audit(assignments)
    metrics = build_metrics(train_info, full_test, field_test, onnx, external)
    tuck = external["comparison"].get("tuckaway_photos", {}).get("0.12", {})
    foliage = external["comparison"].get("video_foliage", {}).get("0.12", {})
    material_by_group = {}
    for group_name, group in (("tuckaway_photos", tuck), ("video_foliage", foliage)):
        v3_rate = group.get("v3", {}).get("leaf_detection_rate") or 0.0
        baseline_rates = [group.get(old, {}).get("leaf_detection_rate") or 0.0 for old in ("v1", "v2")]
        material_by_group[group_name] = v3_rate - max(baseline_rates) >= 0.20
    material = any(material_by_group.values())
    localized = all(
        external["comparison"].get(group, {}).get("0.12", {}).get("v3", {}).get("all_box_area_ge_0.90_image_count", 0) == 0
        for group in ("tuckaway_photos", "video_foliage")
    )
    metrics["fit_for_app_integration"] = bool(material and localized)
    metrics["integration_decision"] = "FIT: at least one held-out foliage group improved >=20 percentage points over the best V1/V2 rate at .12 and V3 produced no >=90%-frame boxes of any class in either held-out foliage group." if metrics["fit_for_app_integration"] else "NOT FIT / NOT ESTABLISHED: V3 did not meet both the held-out external improvement and all-class localization checks. Do not integrate on these results."
    metrics["integration_checks"] = {"external_improvement_ge_0.20_over_best_baseline_by_group": material_by_group, "at_least_one_group_improves": material, "zero_scene_scale_all_class_boxes_on_foliage_groups": localized}
    write_json(OUT_ROOT / "metrics.json", metrics)
    write_report(metrics)
    print(json.dumps({"epochs_run": train_info["epochs_run"], "minutes": train_info["wall_clock_minutes"], "field_box_recall": metrics["test"]["leafsnap_field_leaf_box_recall"], "field_box_ap50": metrics["test"]["leafsnap_field_leaf_box_ap50"], "field_seg_ap50": metrics["test"]["leafsnap_field_leaf_segmentation_ap50"], "fit_for_app_integration": metrics["fit_for_app_integration"], "model": metrics["model"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
