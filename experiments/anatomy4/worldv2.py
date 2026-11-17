# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""YOLO-World v2 open-vocabulary *box* proposal backend.

Practical alternative to YOLOE-26: same ultralytics runtime, but the text
encoder is a CLIP language tower over LVIS-style noun phrases and the detector
is a reparameterized YOLOv8. No masks, no SAM stage, no rectangle fallback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .altdet import (
    Timer,
    build_output_manifest,
    dedupe_same_class,
    make_box_proposal,
    prompt_class_index,
    prompts_for,
    select_images,
    utc_now,
)
from .contracts import AnatomyConfig

LOGGER = logging.getLogger("anatomy4.worldv2")


def run_worldv2(
    manifest: dict[str, Any],
    config: AnatomyConfig,
    out_dir: str | Path,
    *,
    model_name: str = "yolov8s-worldv2.pt",
    imgsz: int = 1280,
    device: str = "0",
    conf_floor: float = 0.01,
    iou: float = 0.5,
    max_det: int = 100,
    max_per_class: int = 10,
    bucket: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLOWorld
    import torch

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    images = select_images(manifest, bucket=bucket, limit=limit)
    prompts = prompts_for(config)
    flat = prompt_class_index(prompts)
    phrases = [prompt for prompt, _ in flat]

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    model = YOLOWorld(model_name)
    model.set_classes(phrases)
    provenance = {
        "contract": config.contract,
        "config_fingerprint": config.fingerprint(),
        "backend": "worldv2",
        "detector": model_name,
        "prompts": {name: list(values) for name, values in prompts.items()},
        "query_order": phrases,
        "imgsz": imgsz,
        "device": device,
        "conf_floor": conf_floor,
        "nms_iou": iou,
        "max_det": max_det,
        "max_per_class": max_per_class,
        "torch": torch.__version__,
        "created_at": utc_now(),
    }

    records: list[dict[str, Any]] = []
    wall = Timer()
    raw_detections = 0
    for index, image in enumerate(images, start=1):
        photo_id = str(image.get("photo_id") or image.get("image_id"))
        image_path = image.get("download", {}).get("path")
        if not image_path or not Path(image_path).is_file():
            continue
        started = Timer()
        result = model.predict(
            image_path,
            imgsz=imgsz,
            conf=conf_floor,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )[0]
        height, width = result.orig_shape
        candidates: list[dict[str, Any]] = []
        for rank, (box, score, class_id) in enumerate(
            zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
            )
        ):
            class_index = int(class_id)
            if class_index < 0 or class_index >= len(flat):
                continue
            prompt, class_name = flat[class_index]
            if float(score) < config.class_by_name[class_name].box_threshold:
                continue
            raw_detections += 1
            candidates.append(
                make_box_proposal(
                    photo_id=photo_id,
                    observation_id=str(image.get("observation_id") or ""),
                    image_path=image_path,
                    audit_bucket=image.get("audit_bucket"),
                    class_name=class_name,
                    prompt=prompt,
                    score=float(score),
                    box=box,
                    width=width,
                    height=height,
                    provenance=provenance,
                    detector_rank=rank,
                    slug="w",
                    config=config,
                )
            )
        proposals = dedupe_same_class(candidates, iou_threshold=iou, max_per_class=max_per_class)
        record = {
            "image_id": photo_id,
            "photo_id": photo_id,
            "observation_id": str(image.get("observation_id") or ""),
            "audit_bucket": image.get("audit_bucket"),
            "selected_for_audit": True,
            "image_path": image_path,
            "download": image.get("download", {}),
            "original_size": [width, height],
            "inference_size": [imgsz, imgsz],
            "elapsed_seconds": round(started.elapsed, 3),
            "proposals": proposals,
            "memory": {
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if torch.cuda.is_available()
                else 0.0,
            },
            "model": provenance,
        }
        records.append(record)
        if index % 25 == 0:
            LOGGER.info(
                "[%d/%d] %s proposals=%d", index, len(images), photo_id, len(proposals)
            )

    result_manifest = build_output_manifest(
        records,
        config=config,
        backend="worldv2",
        provenance=provenance,
        audit_bucket=bucket,
        total_runtime_seconds=wall.elapsed,
    )
    result_manifest["summary"]["runtime_seconds_per_image"] = round(
        wall.elapsed / max(1, len(records)), 4
    )
    result_manifest["summary"]["raw_detections_above_floor"] = raw_detections
    (output / "proposals.json").write_text(json.dumps(result_manifest, indent=1))
    return result_manifest
