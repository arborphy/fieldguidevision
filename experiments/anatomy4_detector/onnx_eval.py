"""Exact arq-api ONNX inference for held-out evaluation.

# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
# Nuance: Third copy of the YOLO26 ONNX letterbox/orient/NMS/decode kernel (alongside
# `arq-api/fieldvision/anatomy_yolo.py` and `arq-api/photo_library/yolo26_organ_bootstrap.py`).
# Note: docstring claims cross-check against `arq-api`, but `test_onnx_eval.py` does not import it.

The production path (`arq-api/fieldvision/anatomy_yolo.py`) serves the exported
YOLO26 ONNX graph, which decodes the **one2many** head. Ultralytics
``model.predict()`` reads the NMS-free **one2one** head. Evaluating the ONNX
graph with the same letterbox/decode/NMS rules is the only honest way to compare
the shipped detector to the training-time head.

This module mirrors the API preprocessing and decoder (input ``[1,3,640,640]``,
output ``[1,4+nc,N]`` normalized to anchors x channels, class-wise NMS) without
importing arq-api. The unit tests compare it against the API implementation when
that worktree is present.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CLASS_NAMES
from .metrics import PixelDetection

INPUT_SIZE = 640
LETTERBOX_FILL = 114
DEFAULT_CONF_FLOOR = 0.001
DEFAULT_IOU = 0.45
DEFAULT_MAX_DET = 300


class OnnxEvalError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT step=yolo-onnx-letterbox role=canonical
def letterbox_rgb(image, imgsz: int = INPUT_SIZE, fill: int = LETTERBOX_FILL):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise OnnxEvalError("image has zero dimensions")
    scale = min(imgsz / float(width), imgsz / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    pad_x = max(0, (imgsz - new_w) // 2)
    pad_y = max(0, (imgsz - new_h) // 2)
    canvas = Image.new("RGB", (imgsz, imgsz), (fill, fill, fill))
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None, ...] / 255.0
    meta = {
        "orig_w": float(width),
        "orig_h": float(height),
        "input_size": float(imgsz),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "scale_x": new_w / float(width),
        "scale_y": new_h / float(height),
        "new_w": float(new_w),
        "new_h": float(new_h),
    }
    return np.ascontiguousarray(tensor), meta


# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT step=yolo-onnx-orient-output role=canonical
def orient_output(raw: Any, class_count: int) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise OnnxEvalError(f"expected batch size 1, got {array.shape}")
        array = array[0]
    if array.ndim != 2:
        raise OnnxEvalError(f"unexpected detection output shape: {array.shape}")
    expected = 4 + class_count
    if array.shape[0] == expected:
        return array.T
    if array.shape[1] == expected:
        return array
    raise OnnxEvalError(
        f"detection output has shape {array.shape}; expected {expected} channels"
    )


# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT step=yolo-onnx-greedy-nms role=canonical
def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        x0 = np.maximum(boxes[index, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[index, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[index, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[index, 3], boxes[rest, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        area_i = max(0.0, float(boxes[index, 2] - boxes[index, 0])) * max(
            0.0, float(boxes[index, 3] - boxes[index, 1])
        )
        area_r = np.clip(boxes[rest, 2] - boxes[rest, 0], 0, None) * np.clip(
            boxes[rest, 3] - boxes[rest, 1], 0, None
        )
        union = area_i + area_r - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT step=yolo-onnx-decode-detections role=canonical
def decode_detections(
    raw_output: Any,
    meta: Mapping[str, float],
    class_names: Sequence[str],
    *,
    threshold: float = DEFAULT_CONF_FLOOR,
    iou_threshold: float = DEFAULT_IOU,
    max_detections: int = DEFAULT_MAX_DET,
) -> list[dict[str, Any]]:
    if not class_names:
        raise OnnxEvalError("class_names must not be empty")
    rows = orient_output(raw_output, len(class_names))
    boxes_cxcywh = rows[:, :4]
    scores_by_class = rows[:, 4 : 4 + len(class_names)]
    if not np.all(np.isfinite(boxes_cxcywh)):
        return []
    best_scores = scores_by_class.max(axis=1)
    best_classes = scores_by_class.argmax(axis=1)
    orig_w = float(meta["orig_w"])
    orig_h = float(meta["orig_h"])
    pad_x = float(meta["pad_x"])
    pad_y = float(meta["pad_y"])
    scale_x = float(meta["scale_x"])
    scale_y = float(meta["scale_y"])
    input_size = float(meta["input_size"])

    detections: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        candidate_idx = np.where(
            (best_classes == class_id) & (best_scores >= threshold)
        )[0]
        if candidate_idx.size == 0:
            continue
        cx, cy, w, h = (boxes_cxcywh[candidate_idx, i] for i in range(4))
        pixel_boxes = np.stack(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1
        )
        pixel_boxes[:, 0::2] = np.clip(pixel_boxes[:, 0::2], 0.0, input_size)
        pixel_boxes[:, 1::2] = np.clip(pixel_boxes[:, 1::2], 0.0, input_size)
        keep_local = nms_indices(
            pixel_boxes, best_scores[candidate_idx], iou_threshold
        )
        for local_index in keep_local:
            pixel_box = pixel_boxes[local_index]
            x0 = (pixel_box[0] - pad_x) / max(scale_x, 1e-9)
            y0 = (pixel_box[1] - pad_y) / max(scale_y, 1e-9)
            x1 = (pixel_box[2] - pad_x) / max(scale_x, 1e-9)
            y1 = (pixel_box[3] - pad_y) / max(scale_y, 1e-9)
            x0 = float(np.clip(x0, 0.0, orig_w))
            y0 = float(np.clip(y0, 0.0, orig_h))
            x1 = float(np.clip(x1, 0.0, orig_w))
            y1 = float(np.clip(y1, 0.0, orig_h))
            if x1 <= x0 or y1 <= y0:
                continue
            nx0, ny0 = x0 / orig_w, y0 / orig_h
            nx1, ny1 = x1 / orig_w, y1 / orig_h
            confidence = float(
                np.clip(best_scores[candidate_idx[local_index]], 0.0, 1.0)
            )
            detections.append(
                {
                    "class": str(class_name),
                    "class_id": int(class_id),
                    "bbox": [
                        round(nx0, 6),
                        round(ny0, 6),
                        round(nx1, 6),
                        round(ny1, 6),
                    ],
                    "bbox_pixels": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                    "bbox_area": round(float((nx1 - nx0) * (ny1 - ny0)), 6),
                    "confidence": round(confidence, 6),
                }
            )
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections[:max_detections]


def load_session(onnx_path: str | Path):
    import onnxruntime as ort

    path = Path(onnx_path)
    if not path.is_file():
        raise OnnxEvalError(f"ONNX model not found: {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def predict_image(
    session,
    image_path: str | Path,
    class_names: Sequence[str] = CLASS_NAMES,
    *,
    imgsz: int = INPUT_SIZE,
    threshold: float = DEFAULT_CONF_FLOOR,
    iou_threshold: float = DEFAULT_IOU,
    max_detections: int = DEFAULT_MAX_DET,
) -> tuple[list[PixelDetection], list[dict[str, Any]]]:
    from PIL import Image

    with Image.open(image_path) as opened:
        tensor, meta = letterbox_rgb(opened, imgsz)
    input_name = session.get_inputs()[0].name
    raw = session.run(None, {input_name: tensor})[0]
    rows = decode_detections(
        raw,
        meta,
        class_names,
        threshold=threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    pixels: list[PixelDetection] = []
    for row in rows:
        x0, y0, x1, y1 = row["bbox_pixels"]
        pixels.append(
            PixelDetection(
                int(row["class_id"]),
                float(x0),
                float(y0),
                float(x1),
                float(y1),
                float(row["confidence"]),
            )
        )
    return pixels, rows


def run_predictions_onnx(
    *,
    onnx_path: str | Path,
    records: Sequence[Mapping[str, Any]],
    imgsz: int = INPUT_SIZE,
    conf_floor: float = DEFAULT_CONF_FLOOR,
    nms_iou: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
) -> tuple[dict[str, list[PixelDetection]], dict[str, list[dict[str, Any]]]]:
    session = load_session(onnx_path)
    pixels: dict[str, list[PixelDetection]] = {}
    serializable: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        image_pixels, rows = predict_image(
            session,
            record["prepared_image"],
            CLASS_NAMES,
            imgsz=imgsz,
            threshold=conf_floor,
            iou_threshold=nms_iou,
            max_detections=max_det,
        )
        pixels[record["stem"]] = image_pixels
        serializable[record["stem"]] = [
            {
                "class_id": row["class_id"],
                "class_name": row["class"],
                "confidence": row["confidence"],
                "xyxy_px": row["bbox_pixels"],
                "xyxy_norm": row["bbox"],
            }
            for row in rows
        ]
    return pixels, serializable
