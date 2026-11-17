# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""YOLOE-26 open-vocabulary comparison backend.

Same proposal schema as the Grounding-DINO/SAM runner so contact sheets and
audits are directly comparable. YOLOE emits instance masks with the detection;
there is no separate SAM stage and no rectangle fallback.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import CLASS_IDS, AnatomyConfig, polygon_area, polygon_to_bbox

LOGGER = logging.getLogger("anatomy4.yoloe")

# One phrase per class keeps the open-vocab class index unambiguous.
DEFAULT_YOLOE_CLASSES = {
    "leaf": "leaf",
    "needle": "pine needle",
    "flower": "flower",
    "inflorescence": "flower cluster",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_yoloe(
    manifest: dict[str, Any],
    config: AnatomyConfig,
    out_dir: str | Path,
    *,
    model_name: str = "yoloe-26s-seg.pt",
    imgsz: int = 1024,
    device: str = "0",
    conf_floor: float = 0.03,
    resume: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLOE
    import torch

    output = Path(out_dir)
    items_dir = output / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    images = [image for image in manifest.get("images", []) if image.get("selected_for_audit")]
    if limit is not None:
        images = images[:limit]

    phrases_by_class = {
        name: (config.class_by_name[name].prompts[0] if config.class_by_name[name].prompts else DEFAULT_YOLOE_CLASSES[name])
        for name in config.class_names
    }
    phrases = [phrases_by_class[name] for name in config.class_names]
    model = YOLOE(model_name)
    model.set_classes(phrases, model.get_text_pe(phrases))
    provenance = {
        "contract": config.contract,
        "config_fingerprint": config.fingerprint(),
        "backend": "yoloe",
        "detector": model_name,
        "classes": phrases,
        "imgsz": imgsz,
        "device": device,
        "conf_floor": conf_floor,
        "torch": torch.__version__,
        "created_at": _utc_now(),
    }

    records: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        photo_id = str(image["photo_id"])
        image_path = image.get("download", {}).get("path")
        if not image_path or not Path(image_path).is_file():
            continue
        item_path = items_dir / f"{photo_id}.json"
        if resume and item_path.is_file():
            records.append(json.loads(item_path.read_text()))
            continue
        started = time.perf_counter()
        result = model.predict(
            image_path,
            imgsz=imgsz,
            conf=conf_floor,
            iou=0.5,
            device=device,
            verbose=False,
        )[0]
        height, width = result.orig_shape
        proposals: list[dict[str, Any]] = []
        polygons = result.masks.xy if result.masks is not None else []
        for det_index, (box, score, class_id) in enumerate(
            zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
        ):
            class_name = config.class_names[int(class_id)]
            spec = config.class_by_name[class_name]
            if float(score) < spec.box_threshold:
                continue
            polygon = [
                (float(x) / width, float(y) / height)
                for x, y in (polygons[det_index] if det_index < len(polygons) else [])
            ]
            normalized = (
                max(0.0, min(1.0, box[0] / width)),
                max(0.0, min(1.0, box[1] / height)),
                max(0.0, min(1.0, box[2] / width)),
                max(0.0, min(1.0, box[3] / height)),
            )
            box_area = max(0.0, normalized[2] - normalized[0]) * max(
                0.0, normalized[3] - normalized[1]
            )
            status = "proposed"
            if box_area > config.max_box_area:
                status = "rejected_area:dominates_frame"
            elif box_area < config.min_box_area:
                status = "rejected_area:too_small"
            elif len(polygon) < config.min_polygon_points:
                status = "mask_failed"
            proposals.append(
                {
                    "proposal_id": f"a4y-{photo_id}-{class_name}-{det_index:03d}",
                    "image_id": photo_id,
                    "photo_id": photo_id,
                    "observation_id": str(image.get("observation_id") or ""),
                    "image_path": image_path,
                    "class_name": class_name,
                    "class_id": CLASS_IDS[class_name],
                    "prompt": phrases_by_class[class_name],
                    "detector_label": phrases_by_class[class_name],
                    "box_score": float(score),
                    "mask_iou": None,
                    "detector_bbox": list(normalized),
                    "detector_box_area": box_area,
                    "polygon": [list(point) for point in polygon],
                    "bbox": list(polygon_to_bbox(polygon)) if len(polygon) >= 3 else None,
                    "area_fraction": polygon_area(polygon) if len(polygon) >= 3 else None,
                    "status": status,
                    "created_at": _utc_now(),
                    "model": provenance,
                }
            )
        record = {
            "image_id": photo_id,
            "observation_id": str(image.get("observation_id") or ""),
            "image_path": image_path,
            "download": image.get("download", {}),
            "original_size": [width, height],
            "inference_size": [imgsz, imgsz],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "raw_detections_by_class": {
                config.class_names[int(c)]: sum(
                    1 for value in result.boxes.cls.tolist() if int(value) == int(c)
                )
                for c in set(result.boxes.cls.tolist())
            },
            "proposals": proposals,
            "memory": {
                "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1)
                if torch.cuda.is_available()
                else 0.0,
            },
            "model": provenance,
        }
        item_path.write_text(json.dumps(record, indent=1))
        records.append(record)
        if index % 10 == 0:
            LOGGER.info("[%d/%d] %s proposals=%d", index, len(images), photo_id, len(proposals))

    proposals = [proposal for record in records for proposal in record["proposals"]]
    output_manifest = {
        "schema": "arq.anatomy4.prelabel/v1",
        "contract": config.contract,
        "created_at": _utc_now(),
        "backend": "yoloe",
        "config_fingerprint": config.fingerprint(),
        "provenance": provenance,
        "images": records,
        "proposals": proposals,
        "summary": {
            "images_requested": len(images),
            "images_processed": len(records),
            "proposals": len(proposals),
            "proposals_by_class": {
                name: len([p for p in proposals if p["class_name"] == name])
                for name in config.class_names
            },
            "exportable_by_class": {
                name: len(
                    [
                        p
                        for p in proposals
                        if p["class_name"] == name and p["status"] == "proposed"
                    ]
                )
                for name in config.class_names
            },
            "total_runtime_seconds": round(
                sum(record.get("elapsed_seconds", 0.0) for record in records), 2
            ),
        },
    }
    (output / "proposals.json").write_text(json.dumps(output_manifest, indent=1))
    return output_manifest


def _slug(text: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in text.lower()).strip("-")


def run_yoloe_variants(
    manifest: dict[str, Any],
    config: AnatomyConfig,
    variants: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    model_name: str = "yoloe-26s-seg.pt",
    imgsz: int = 1024,
    device: str = "0",
    conf_floor: float = 0.02,
    bucket: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run one prompt variant at a time and merge the proposals.

    Each variant is a single-class YOLOE pass, so the class index is
    unambiguous and the prompt that produced a box is recorded on the
    proposal. Filtering by ``audit_bucket`` keeps each class's pass on its
    targeted image pool while the output stays one manifest.
    """

    from ultralytics import YOLOE
    import torch

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    images = [image for image in manifest.get("images", []) if image.get("selected_for_audit")]
    if bucket:
        images = [image for image in images if image.get("audit_bucket") == bucket]
    if limit is not None:
        images = images[:limit]

    model = YOLOE(model_name)
    records: dict[str, dict[str, Any]] = {}
    proposals: list[dict[str, Any]] = []
    provenance_runs: list[dict[str, Any]] = []

    for variant in variants:
        class_name = str(variant["class_name"])
        prompt = str(variant["prompt"])
        spec = config.class_by_name[class_name]
        threshold = float(variant.get("box_threshold", spec.box_threshold))
        model.set_classes([prompt], model.get_text_pe([prompt]))
        run_provenance = {
            "class_name": class_name,
            "prompt": prompt,
            "box_threshold": threshold,
            "imgsz": imgsz,
            "conf_floor": conf_floor,
            "detector": model_name,
            "torch": torch.__version__,
            "device": device,
        }
        provenance_runs.append(run_provenance)
        slug = _slug(prompt)
        for image in images:
            photo_id = str(image.get("photo_id") or image.get("image_id"))
            image_path = image.get("download", {}).get("path")
            if not image_path or not Path(image_path).is_file():
                continue
            started = time.perf_counter()
            result = model.predict(
                image_path,
                imgsz=imgsz,
                conf=min(conf_floor, threshold),
                iou=0.5,
                device=device,
                verbose=False,
            )[0]
            height, width = result.orig_shape
            polygons = result.masks.xy if result.masks is not None else []
            record = records.setdefault(
                photo_id,
                {
                    "image_id": photo_id,
                    "photo_id": photo_id,
                    "observation_id": str(image.get("observation_id") or ""),
                    "audit_bucket": image.get("audit_bucket"),
                    "image_path": image_path,
                    "download": image.get("download", {}),
                    "original_size": [width, height],
                    "inference_size": [imgsz, imgsz],
                    "elapsed_seconds": 0.0,
                    "raw_detections_by_class": {},
                    "proposals": [],
                    "memory": {},
                    "model": {"backend": "yoloe", "detector": model_name},
                },
            )
            record["elapsed_seconds"] = round(
                float(record["elapsed_seconds"]) + (time.perf_counter() - started), 3
            )
            for det_index, (box, score) in enumerate(
                zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
            ):
                if float(score) < threshold:
                    continue
                polygon = [
                    (float(x) / width, float(y) / height)
                    for x, y in (polygons[det_index] if det_index < len(polygons) else [])
                ]
                normalized = (
                    max(0.0, min(1.0, box[0] / width)),
                    max(0.0, min(1.0, box[1] / height)),
                    max(0.0, min(1.0, box[2] / width)),
                    max(0.0, min(1.0, box[3] / height)),
                )
                box_area = max(0.0, normalized[2] - normalized[0]) * max(
                    0.0, normalized[3] - normalized[1]
                )
                status = "proposed"
                if box_area > config.max_box_area:
                    status = "rejected_area:dominates_frame"
                elif box_area < config.min_box_area:
                    status = "rejected_area:too_small"
                elif len(polygon) < config.min_polygon_points:
                    status = "mask_failed"
                proposal = {
                    "proposal_id": (
                        f"a4y-{photo_id}-{class_name}-{slug}-{det_index:03d}"
                    ),
                    "image_id": photo_id,
                    "photo_id": photo_id,
                    "observation_id": str(image.get("observation_id") or ""),
                    "image_path": image_path,
                    "audit_bucket": image.get("audit_bucket"),
                    "class_name": class_name,
                    "class_id": CLASS_IDS[class_name],
                    "prompt": prompt,
                    "prompt_variant": prompt,
                    "detector_label": prompt,
                    "box_score": float(score),
                    "mask_iou": None,
                    "detector_bbox": list(normalized),
                    "detector_box_area": box_area,
                    "polygon": [list(point) for point in polygon],
                    "bbox": list(polygon_to_bbox(polygon)) if len(polygon) >= 3 else None,
                    "area_fraction": polygon_area(polygon) if len(polygon) >= 3 else None,
                    "status": status,
                    "created_at": _utc_now(),
                    "model": run_provenance,
                }
                record["proposals"].append(proposal)
                proposals.append(proposal)

    record_list = list(records.values())
    for record in record_list:
        record["memory"] = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1)
            if torch.cuda.is_available()
            else 0.0
        }
    output_manifest = {
        "schema": "arq.anatomy4.prelabel/v1",
        "contract": config.contract,
        "created_at": _utc_now(),
        "backend": "yoloe-variants",
        "config_fingerprint": config.fingerprint(),
        "audit_bucket": bucket,
        "provenance": {"runs": provenance_runs, "contract": config.contract},
        "images": record_list,
        "proposals": proposals,
        "summary": {
            "images_requested": len(images),
            "images_processed": len(record_list),
            "proposals": len(proposals),
            "proposals_by_class": {
                name: len([p for p in proposals if p["class_name"] == name])
                for name in config.class_names
            },
            "proposals_by_prompt": {
                variant["prompt"]: len(
                    [p for p in proposals if p["prompt"] == variant["prompt"]]
                )
                for variant in variants
            },
            "exportable_by_class": {
                name: len(
                    [
                        p
                        for p in proposals
                        if p["class_name"] == name and p["status"] == "proposed"
                    ]
                )
                for name in config.class_names
            },
            "total_runtime_seconds": round(
                sum(record.get("elapsed_seconds", 0.0) for record in record_list), 2
            ),
        },
    }
    (output / "proposals.json").write_text(json.dumps(output_manifest, indent=1))
    return output_manifest
