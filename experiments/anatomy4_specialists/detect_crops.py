# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Stage-1 anatomy4 YOLO -> padded Stage-2 crops (true two-stage cascade).

The Stage-1 detector has exactly four classes: leaf, needle, flower,
inflorescence. This module runs the trained checkpoint over the staged Dirr
image manifest, materializes padded crops for each detection, and writes the
same crop-manifest contract the rest of the pipeline consumes.

No taxon-derived boxes: every crop comes from a Stage-1 detection box.
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ANATOMY_CLASSES,
    CONTRACT,
    CROP_MANIFEST_SCHEMA,
    SPLITS,
    normalize_crop_row,
    read_jsonl,
    write_jsonl,
)
from .crops import crop_id_for, pad_box_pixels
from .splits import observation_split

STAGE1_CLASSES: tuple[str, ...] = ("leaf", "needle", "flower", "inflorescence")
STAGE1_BOX_SOURCE = "stage1_anatomy4_yolo"
STAGE1_REVIEW_VERDICT = "stage1_predicted"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class DetectCropsReport:
    images_total: int = 0
    images_with_detections: int = 0
    images_without_detections: int = 0
    images_without_crops: int = 0
    detections_total: int = 0
    crops_written: int = 0
    skipped_rows: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)
    by_class: Counter[str] = field(default_factory=Counter)
    by_split: Counter[str] = field(default_factory=Counter)
    overlap_crops: Counter[str] = field(default_factory=Counter)
    overlap_images: Counter[str] = field(default_factory=Counter)
    crops: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "arq.anatomy4_specialists.detect_crops/v1",
            "contract": CONTRACT,
            "stage1_classes": list(STAGE1_CLASSES),
            "images_total": self.images_total,
            "images_with_detections": self.images_with_detections,
            "images_without_detections": self.images_without_detections,
            "unlocalized_images": self.images_without_detections,
            "images_without_crops": self.images_without_crops,
            "detections_total": self.detections_total,
            "crops_written": self.crops_written,
            "skipped_rows": self.skipped_rows,
            "skip_reasons": dict(sorted(self.skip_reasons.items())),
            "by_class": dict(sorted(self.by_class.items())),
            "by_split": dict(sorted(self.by_split.items())),
            "stage1_overlap": {
                "crops": dict(sorted(self.overlap_crops.items())),
                "images": dict(sorted(self.overlap_images.items())),
            },
        }


def _license_present(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in {"null", "none"}


def load_dirr_manifest(
    path: str | Path, *, exclude_null_license: bool = False
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    excluded = 0
    for raw in read_jsonl(path):
        row = dict(raw)
        if exclude_null_license and not _license_present(row.get("license")):
            excluded += 1
            continue
        local_path = str(row.get("local_path") or "").strip()
        observation_id = str(row.get("observation_id") or "").strip()
        taxon_name = str(row.get("taxon_name") or "").strip()
        if not local_path or not observation_id or not taxon_name:
            excluded += 1
            continue
        row["local_path"] = local_path
        row["observation_id"] = observation_id
        row["taxon_name"] = taxon_name
        row["photo_id"] = str(row.get("photo_id") or "").strip() or Path(local_path).stem
        rows.append(row)
    return rows, excluded


def _add_token(tokens: set[str], value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    tokens.add(text)
    stem = Path(text).stem
    if stem:
        tokens.add(stem)
        if "__" in stem:
            observation, _, photo = stem.partition("__")
            tokens.add(observation)
            tokens.add(photo)
        if stem.startswith("inat-"):
            tokens.add(stem.split("_")[0])


def _collect_paths(value: Any, tokens: set[str]) -> None:
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_dir():
            for path in candidate.rglob("*"):
                if path.suffix.lower() in IMAGE_EXTS:
                    _add_token(tokens, path)
        elif candidate.is_file():
            _add_token(tokens, candidate)
        elif any(char in value for char in "*?["):
            for path in glob.glob(value, recursive=True):
                _add_token(tokens, path)
        else:
            _add_token(tokens, value)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_paths(item, tokens)
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect_paths(item, tokens)


def load_stage1_overlap(path: str | Path) -> set[str]:
    """Collect train/val tokens from a Stage-1 dataset JSON.

    Accepts Ultralytics-style ``{"train": "...", "val": "..."}``, a ``splits``
    mapping, or ``observations`` records carrying a ``split``. File name stems
    are split on ``__`` so ``inat-1__photo`` contributes the observation and
    photo ids too.
    """

    data = json.loads(Path(path).read_text())
    if isinstance(data, Mapping) and data.get("classes"):
        classes = {str(name) for name in data["classes"]}
        if classes != set(STAGE1_CLASSES):
            raise ValueError(
                f"Stage-1 dataset classes {sorted(classes)} != {list(STAGE1_CLASSES)}"
            )
    tokens: set[str] = set()
    if isinstance(data, Mapping):
        for key in ("train", "val"):
            if key in data:
                _collect_paths(data[key], tokens)
        for key in ("splits", "split", "observations", "images"):
            value = data.get(key)
            if isinstance(value, Mapping):
                for name in ("train", "val"):
                    if name in value:
                        _collect_paths(value[name], tokens)
            elif isinstance(value, list):
                for record in value:
                    if isinstance(record, Mapping):
                        image_path = str(record.get("image") or record.get("path") or record.get("local_path") or "")
                        split = str(record.get("split", "")).lower()
                        if not split and image_path:
                            lowered = image_path.lower().replace("\\", "/")
                            if "/train/" in lowered:
                                split = "train"
                            elif "/val/" in lowered:
                                split = "val"
                        if split in {"train", "val"}:
                            for field_name in ("observation_id", "photo_id", "path", "local_path", "image"):
                                _add_token(tokens, record.get(field_name))
                    elif isinstance(record, str):
                        _add_token(tokens, record)
    elif isinstance(data, list):
        for record in data:
            if isinstance(record, Mapping) and str(record.get("split", "")).lower() in {"train", "val"}:
                for field_name in ("observation_id", "photo_id", "path", "local_path", "image"):
                    _add_token(tokens, record.get(field_name))
    return tokens


def _row_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for field_name in ("observation_id", "photo_id", "local_path", "source_url"):
        _add_token(tokens, row.get(field_name))
    observation = str(row.get("observation_id") or "")
    photo = str(row.get("photo_id") or "")
    if observation and photo:
        _add_token(tokens, f"{observation}__{photo}")
    return tokens


def stage1_overlap_flag(row: Mapping[str, Any], tokens: set[str] | None) -> bool | None:
    """True when Stage-1 trained/validated on this observation, False otherwise."""

    if tokens is None:
        return None
    return bool(_row_tokens(row) & tokens)


def _row_split(row: Mapping[str, Any], seed: str) -> str:
    existing = str(row.get("split") or "")
    if existing in SPLITS:
        return existing
    return observation_split(str(row["observation_id"]), seed)


def crops_from_detections(
    image_path: str | Path,
    detections: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    observation_id: str,
    taxon_name: str,
    photo_id: str,
    image_url: str,
    observation_split_value: str | None = None,
    split_seed: str = "anatomy4-specialists-v1",
    in_stage1_train_val: bool | None = None,
    padding_frac: float = 0.18,
    padding_mode: str = "relative",
    min_crop_px: int = 48,
    max_crops: int = 6,
    jpeg_quality: int = 92,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Materialize one image's detections into padded crops.

    Pure of model/runtime dependencies (PIL only) so the cascade's crop math is
    testable without a GPU or a real checkpoint.
    """

    from PIL import Image

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    skipped: Counter[str] = Counter()
    usable: list[dict[str, Any]] = []
    for detection in detections:
        class_name = str(detection.get("class_name") or "")
        if class_name not in STAGE1_CLASSES:
            skipped["unknown_stage1_class"] += 1
            continue
        try:
            confidence = float(detection.get("confidence"))
            bbox = [float(value) for value in detection.get("bbox") or []]
        except (TypeError, ValueError):
            skipped["invalid_detection"] += 1
            continue
        if len(bbox) != 4:
            skipped["invalid_detection"] += 1
            continue
        x0, y0, x1, y1 = bbox
        x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
        y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
        if x1 <= x0 or y1 <= y0:
            skipped["degenerate_detection"] += 1
            continue
        usable.append(
            {"class_name": class_name, "confidence": confidence, "bbox": [x0, y0, x1, y1]}
        )
    usable.sort(key=lambda item: item["confidence"], reverse=True)
    usable = usable[: max(0, int(max_crops))]

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    width, height = image.size
    split_value = observation_split_value or observation_split(observation_id, split_seed)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for detection in usable:
        bbox = detection["bbox"]
        bbox_px = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        cx0, cy0, cx1, cy1 = pad_box_pixels(
            bbox_px,
            width,
            height,
            padding_frac=padding_frac,
            padding_mode=padding_mode,
            min_crop_px=min_crop_px,
        )
        if cx1 - cx0 < min_crop_px or cy1 - cy0 < min_crop_px:
            skipped["crop_below_min_size"] += 1
            continue
        crop_id = crop_id_for(photo_id, bbox, detection["class_name"])
        if crop_id in seen:
            skipped["duplicate_crop"] += 1
            continue
        crop = image.crop((cx0, cy0, cx1, cy1))
        crop_path = output / "crops" / f"{crop_id}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, quality=jpeg_quality, optimize=True)
        row = normalize_crop_row(
            {
                "schema": CROP_MANIFEST_SCHEMA,
                "contract": CONTRACT,
                "crop_id": crop_id,
                "observation_id": observation_id,
                "taxon_name": taxon_name,
                "photo_id": photo_id,
                "image": image_url,
                "crop_path": str(crop_path),
                "anatomy_class": detection["class_name"],
                "detector_confidence": detection["confidence"],
                "observation_split": split_value,
                "split": split_value,
                "bbox": bbox,
                "crop_bbox_px": [cx0, cy0, cx1, cy1],
                "source_image_path": str(image_path),
                "source_width": width,
                "source_height": height,
                "padding_frac": padding_frac,
                "padding_mode": padding_mode,
                "detector_box_source": STAGE1_BOX_SOURCE,
                "review_verdict": STAGE1_REVIEW_VERDICT,
                "in_stage1_train_val": in_stage1_train_val,
            }
        )
        rows.append(row)
        seen.add(crop_id)
    return rows, skipped


def _detections_from_result(result: Any) -> list[dict[str, Any]]:
    names = getattr(result, "names", None) or {}
    boxes = getattr(result, "boxes", None)
    height, width = (int(value) for value in getattr(result, "orig_shape", (0, 0))[:2])
    if boxes is None or width <= 0 or height <= 0:
        return []
    xyxy = boxes.xyxy.tolist()
    conf = boxes.conf.tolist()
    classes = boxes.cls.tolist()
    detections: list[dict[str, Any]] = []
    for coords, score, cls in zip(xyxy, conf, classes):
        index = int(cls)
        if isinstance(names, Mapping):
            class_name = str(names.get(index, names.get(str(index), "")))
        elif isinstance(names, (list, tuple)) and 0 <= index < len(names):
            class_name = str(names[index])
        else:
            class_name = ""
        detections.append(
            {
                "class_name": class_name,
                "confidence": float(score),
                "bbox": [
                    float(coords[0]) / width,
                    float(coords[1]) / height,
                    float(coords[2]) / width,
                    float(coords[3]) / height,
                ],
            }
        )
    return detections


def detect_crops(
    *,
    weights: str | Path,
    dirr_manifest: str | Path,
    out_dir: str | Path,
    stage1_dataset: str | Path | None = None,
    exclude_null_license: bool = False,
    conf: float = 0.25,
    iou: float = 0.50,
    padding_frac: float = 0.18,
    padding_mode: str = "relative",
    min_crop_px: int = 48,
    max_crops_per_image: int = 6,
    device: str = "0",
    max_vram_fraction: float = 0.35,
    imgsz: int = 640,
    split_seed: str = "anatomy4-specialists-v1",
    jpeg_quality: int = 92,
) -> dict[str, Any]:
    """Run real Stage-1 YOLO detection and write the crop manifest/report."""

    from .device import configure_memory

    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {weights_path}")
    rows, excluded_license = load_dirr_manifest(
        dirr_manifest, exclude_null_license=exclude_null_license
    )
    tokens = load_stage1_overlap(stage1_dataset) if stage1_dataset else None
    memory = configure_memory(
        device,
        max_vram_fraction=max_vram_fraction,
        min_free_vram_gb=0.5,
    )
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    model_names = model.names
    names = (
        [str(model_names[index]) for index in sorted(model_names, key=lambda value: int(value))]
        if isinstance(model_names, Mapping)
        else [str(name) for name in model_names]
    )
    if set(names) != set(STAGE1_CLASSES):
        raise ValueError(
            f"Stage-1 model classes {names} != {list(STAGE1_CLASSES)}; "
            "this command is only for the four-class anatomy4 detector"
        )

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = DetectCropsReport()
    report.images_total = len(rows)
    report.skip_reasons["excluded_null_license"] += excluded_license
    for row in rows:
        image_path = Path(str(row["local_path"]))
        if not image_path.is_file():
            report.images_without_crops += 1
            report.images_without_detections += 1
            report.skip_reasons["missing_local_path"] += 1
            continue
        try:
            results = model.predict(
                source=str(image_path),
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                device=memory["device"],
                max_det=max(1, int(max_crops_per_image) * 2),
                verbose=False,
            )
        except Exception:
            report.images_without_crops += 1
            report.images_without_detections += 1
            report.skip_reasons["stage1_inference_failed"] += 1
            continue
        detections = _detections_from_result(results[0]) if results else []
        report.detections_total += len(detections)
        overlap = stage1_overlap_flag(row, tokens)
        overlap_key = "unknown" if overlap is None else ("in_stage1_train_val" if overlap else "unseen")
        report.overlap_images[overlap_key] += 1
        if not detections:
            report.images_without_detections += 1
            report.images_without_crops += 1
            continue
        report.images_with_detections += 1
        try:
            crops, skipped = crops_from_detections(
                image_path,
                detections,
                output,
                observation_id=str(row["observation_id"]),
                taxon_name=str(row["taxon_name"]),
                photo_id=str(row["photo_id"]),
                image_url=str(row.get("source_url") or row["local_path"]),
                observation_split_value=_row_split(row, split_seed),
                split_seed=split_seed,
                in_stage1_train_val=overlap,
                padding_frac=padding_frac,
                padding_mode=padding_mode,
                min_crop_px=min_crop_px,
                max_crops=max_crops_per_image,
                jpeg_quality=jpeg_quality,
            )
        except Exception:
            report.images_without_crops += 1
            report.skip_reasons["source_image_unreadable"] += 1
            continue
        report.skipped_rows += sum(skipped.values())
        report.skip_reasons.update(skipped)
        if not crops:
            report.images_without_crops += 1
        for crop in crops:
            report.crops.append(crop)
            report.crops_written += 1
            report.by_class[str(crop["anatomy_class"])] += 1
            report.by_split[str(crop["observation_split"])] += 1
            report.overlap_crops[overlap_key] += 1

    write_jsonl(output / "crop_manifest.jsonl", report.crops)
    payload = report.to_dict()
    payload.update(
        {
            "weights": str(weights_path.resolve()),
            "dirr_manifest": str(Path(dirr_manifest).resolve()),
            "stage1_dataset": str(Path(stage1_dataset).resolve()) if stage1_dataset else None,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "padding": padding_frac,
            "padding_mode": padding_mode,
            "min_crop_px": min_crop_px,
            "max_crops_per_image": max_crops_per_image,
            "device": memory["device"],
            "max_vram_fraction": max_vram_fraction,
            "model_class_order": names,
            "excluded_null_license": excluded_license,
            "stage1_overlap_known": tokens is not None,
        }
    )
    (output / "detect_crops_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
