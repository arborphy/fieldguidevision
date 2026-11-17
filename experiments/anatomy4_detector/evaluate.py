# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Held-out evaluation: real predictions -> JSONL -> metrics -> overlays.

Ground truth comes from the audited **normalized xyxy** labels; predictions are
saved in both normalized and pixel coordinates. Nothing is synthesized: every
number in the manifest derives from these saved predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import CLASS_NAMES, CONTRACT, xyxy_norm_to_px, xyxy_px_to_norm
from .metrics import (
    PixelBox,
    PixelDetection,
    evaluate,
    render_metrics_markdown,
    tagged_bbox_metrics,
)
from .onnx_eval import run_predictions_onnx, sha256_file as onnx_sha256
from .render import render_contact_sheets, render_overlay

DEFAULT_CONF_FLOOR = 0.001
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_NMS_IOU = 0.45
DEFAULT_MAX_DET = 300


def split_records(
    audit: Mapping[str, Any], split: str, prepared_root: str | Path
) -> list[dict[str, Any]]:
    root = Path(prepared_root)
    records: list[dict[str, Any]] = []
    for entry in audit["images"]:
        if entry["split"] != split:
            continue
        source = Path(audit["dataset_root"]) / entry["image"]
        prepared = root / "images" / split / source.name
        records.append(
            {
                "stem": source.stem,
                "image": str(source),
                "prepared_image": str(prepared if prepared.is_file() else source),
                "width": entry.get("width"),
                "height": entry.get("height"),
                "observation_id": entry.get("observation_id"),
                "negative": entry.get("negative"),
                "boxes": entry.get("boxes", []),
            }
        )
    return records


def ground_truth_pixels(records: Iterable[Mapping[str, Any]]) -> dict[str, list[PixelBox]]:
    truth: dict[str, list[PixelBox]] = {}
    for record in records:
        width = record.get("width")
        height = record.get("height")
        if not width or not height:
            width, height = _image_size(Path(record["image"]))
        if not width or not height:
            raise RuntimeError(f"cannot resolve image size for {record['image']}")
        boxes: list[PixelBox] = []
        for box in record["boxes"]:
            x0, y0, x1, y1 = xyxy_norm_to_px(
                (box["x0"], box["y0"], box["x1"], box["y1"]), width, height
            )
            boxes.append(
                PixelBox(int(box["class_id"]), x0, y0, x1, y1)
            )
        truth[record["stem"]] = boxes
    return truth


def _image_size(path: Path) -> tuple[int, int] | tuple[None, None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.width, image.height
    except OSError:
        return None, None


def _check_model_classes(names: Any) -> None:
    if isinstance(names, dict):
        ordered = [names[index] for index in sorted(names)]
    else:
        ordered = list(names)
    if ordered != list(CLASS_NAMES):
        raise RuntimeError(
            f"checkpoint classes {ordered} do not match {list(CLASS_NAMES)}; "
            "refusing to evaluate a detector against the wrong class order"
        )


def run_predictions(
    *,
    weights: str | Path,
    records: Sequence[Mapping[str, Any]],
    imgsz: int = 640,
    device: str = "auto",
    batch: int = 8,
    conf_floor: float = DEFAULT_CONF_FLOOR,
    nms_iou: float = DEFAULT_NMS_IOU,
    max_det: int = DEFAULT_MAX_DET,
) -> tuple[dict[str, list[PixelDetection]], dict[str, list[dict[str, Any]]]]:
    """Run the detector once per image and return pixel + serializable predictions."""

    from ultralytics import YOLO

    from .train import resolve_device

    model = YOLO(str(weights))
    _check_model_classes(model.names)
    resolved_device = resolve_device(device)
    sources = [record["prepared_image"] for record in records]
    by_path = {str(Path(record["prepared_image"]).resolve()): record for record in records}
    by_stem = {record["stem"]: record for record in records}

    pixels: dict[str, list[PixelDetection]] = {}
    serializable: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()

    def _per_image():
        # One image per predict call. A single batched list source makes
        # Ultralytics buffer/preload the whole list and OOM on the shared 8 GB
        # .245 GPU (BioCLIP holds ~4.3 GB); per-image peaks at ~100 MB.
        for source in sources:
            yield from model.predict(
                source=source,
                imgsz=imgsz,
                conf=conf_floor,
                iou=nms_iou,
                max_det=max_det,
                device=resolved_device,
                batch=1,
                verbose=False,
                save=False,
            )

    results = _per_image()
    for result in results:
        path = Path(result.path).resolve()
        record = by_path.get(str(path))
        if record is None:
            record = by_stem.get(path.stem)
        if record is None:
            raise RuntimeError(f"prediction for unknown image {path}")
        stem = record["stem"]
        seen.add(stem)
        height, width = result.orig_shape
        boxes = result.boxes
        image_pixels: list[PixelDetection] = []
        image_rows: list[dict[str, Any]] = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for (x0, y0, x1, y1), confidence, class_id in zip(
                xyxy, confidences, classes
            ):
                class_id = int(class_id)
                detection = PixelDetection(
                    class_id,
                    float(x0),
                    float(y0),
                    float(x1),
                    float(y1),
                    float(confidence),
                )
                image_pixels.append(detection)
                image_rows.append(
                    {
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "confidence": round(float(confidence), 6),
                        "xyxy_px": [
                            round(float(x0), 3),
                            round(float(y0), 3),
                            round(float(x1), 3),
                            round(float(y1), 3),
                        ],
                        "xyxy_norm": [
                            round(value, 6)
                            for value in xyxy_px_to_norm(
                                detection.xyxy, width, height
                            )
                        ],
                    }
                )
        pixels[stem] = image_pixels
        serializable[stem] = image_rows

    missing = [record["stem"] for record in records if record["stem"] not in seen]
    if missing:
        raise RuntimeError(
            f"predictions missing for {len(missing)} image(s): {missing[:5]}"
        )
    return pixels, serializable


def write_predictions(
    *,
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    split: str,
    out_dir: str | Path,
    weights: str | Path,
) -> tuple[Path, Path]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    truth_path = output / "ground_truth.jsonl"
    with predictions_path.open("w") as handle:
        for record in records:
            for row in predictions.get(record["stem"], []):
                handle.write(
                    json.dumps(
                        {
                            "schema": "arq.anatomy4.prediction/v1",
                            "contract": CONTRACT,
                            "split": split,
                            "image": record["image"],
                            "stem": record["stem"],
                            "observation_id": record.get("observation_id"),
                            "width": record.get("width"),
                            "height": record.get("height"),
                            **row,
                            "weights": str(weights),
                        }
                    )
                    + "\n"
                )
    with truth_path.open("w") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "schema": "arq.anatomy4.ground-truth/v1",
                        "contract": CONTRACT,
                        "split": split,
                        "image": record["image"],
                        "stem": record["stem"],
                        "observation_id": record.get("observation_id"),
                        "negative": record.get("negative"),
                        "width": record.get("width"),
                        "height": record.get("height"),
                        "boxes": record["boxes"],
                    }
                )
                + "\n"
            )
    return predictions_path, truth_path


def evaluate_split(
    *,
    audit: Mapping[str, Any],
    split: str,
    prepared_root: str | Path,
    weights: str | Path | None,
    out_dir: str | Path,
    imgsz: int = 640,
    device: str = "auto",
    batch: int = 8,
    onnx_path: str | Path | None = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = 0.5,
    conf_floor: float = DEFAULT_CONF_FLOOR,
    nms_iou: float = DEFAULT_NMS_IOU,
    max_det: int = DEFAULT_MAX_DET,
    max_overlays: int = 48,
    overlays_per_sheet: int = 12,
    max_contact_images: int = 240,
    conf_thresholds: Sequence[float] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {split!r}")
    if audit.get("blocking"):
        raise RuntimeError(
            "refusing to evaluate an invalid dataset: "
            f"{[item['code'] for item in audit.get('errors', [])]}"
        )
    records = split_records(audit, split, prepared_root)
    if not records:
        raise RuntimeError(f"split {split!r} has no images")

    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    truth = ground_truth_pixels(records)

    if dry_run:
        return {
            "split": split,
            "images": len(records),
            "gt_boxes": sum(len(boxes) for boxes in truth.values()),
            "dry_run": True,
            "weights": str(weights),
        }

    backend = "onnx" if onnx_path is not None else "pytorch"
    if onnx_path is not None:
        pixels, serializable = run_predictions_onnx(
            onnx_path=onnx_path,
            records=records,
            imgsz=imgsz,
            conf_floor=conf_floor,
            nms_iou=nms_iou,
            max_det=max_det,
        )
    else:
        if weights is None:
            raise RuntimeError("weights are required for the PyTorch evaluator")
        pixels, serializable = run_predictions(
            weights=weights,
            records=records,
            imgsz=imgsz,
            device=device,
            batch=batch,
            conf_floor=conf_floor,
            nms_iou=nms_iou,
            max_det=max_det,
        )
    predictions_path, truth_path = write_predictions(
        records=records,
        predictions=serializable,
        split=split,
        out_dir=output,
        weights=onnx_path or weights,
    )
    result = evaluate(
        truth,
        pixels,
        split=split,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_det,
    )
    thresholds = sorted(
        {
            round(float(value), 4)
            for value in ([conf_threshold] + list(conf_thresholds or []))
        }
    )
    sweep: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        if threshold == conf_threshold:
            entry = result
        else:
            entry = evaluate(
                truth,
                pixels,
                split=split,
                conf_threshold=threshold,
                iou_threshold=iou_threshold,
                max_detections=max_det,
            )
        sweep[f"{threshold:.2f}"] = {
            "per_class": entry["per_class"],
            "macro": entry["macro"],
            "micro": entry["micro"],
            "counts": entry["counts"],
            "operating_point": entry["operating_point"],
            "tagged_bbox": tagged_bbox_metrics(entry),
        }
    result["threshold_sweep"] = {
        "thresholds": thresholds,
        "default": conf_threshold,
        "results": sweep,
    }
    result["tagged_bbox"] = tagged_bbox_metrics(result)
    result["leaf_tagged_bbox"] = result["tagged_bbox"]["leaf"]
    result["predictions_jsonl"] = str(predictions_path)
    result["ground_truth_jsonl"] = str(truth_path)
    result["metrics_json"] = str(output / "metrics.json")
    result["weights"] = str(weights) if weights is not None else None
    result["backend"] = backend
    result["onnx"] = (
        {
            "path": str(Path(onnx_path).expanduser().resolve()),
            "sha256": onnx_sha256(onnx_path),
            "decode": "arq-api letterbox 640, class-wise NMS, [1,4+nc,N]",
        }
        if onnx_path is not None
        else None
    )
    result["config"] = {
        "imgsz": imgsz,
        "conf_floor": conf_floor,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "nms_iou": nms_iou,
        "max_det": max_det,
        "batch": batch,
    }
    result["config"]["onnx"] = str(onnx_path) if onnx_path is not None else None
    result["dataset"] = {
        "split_isolation": audit.get("split_isolation"),
        "labels_sha256": audit.get("labels_sha256"),
        "status": audit.get("status"),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    (output / "metrics.md").write_text(render_metrics_markdown(result))

    overlays: list[dict[str, Any]] = []
    for record in records[: max(0, max_overlays)]:
        overlay_path = output / "overlays" / f"{record['stem']}.jpg"
        overlays.append(
            render_overlay(
                record["prepared_image"],
                truth[record["stem"]],
                pixels[record["stem"]],
                overlay_path,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold,
            )
        )
    (output / "overlays" / "index.json").write_text(
        json.dumps(overlays, indent=1) + "\n"
    )

    sheets = render_contact_sheets(
        [
            {
                "image": record["prepared_image"],
                "ground_truth": truth[record["stem"]],
                "detections": pixels[record["stem"]],
            }
            for record in (
                records[:max_contact_images] if max_contact_images else records
            )
        ],
        output / "contact",
        per_sheet=overlays_per_sheet,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
    )
    result["overlays"] = overlays
    result["contact_sheets"] = sheets
    result["config"]["max_contact_images"] = max_contact_images
    (output / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    return result
