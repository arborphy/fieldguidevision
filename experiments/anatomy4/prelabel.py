# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Grounding-DINO + SAM 2.1 four-class proposal generator.

The detector prompt and thresholds are configured per anatomy4 class. SAM runs
only on NMS survivors; a detection whose mask cannot produce a polygon is
marked ``mask_failed`` and is never exportable as a label (no rectangle
fallback, no taxonomy-derived boxes).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .contracts import (
    CLASS_IDS,
    AnatomyConfig,
    map_detector_label,
    polygon_area,
    polygon_to_bbox,
)

LOGGER = logging.getLogger("anatomy4.prelabel")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _polygon_from_mask(mask, width: int, height: int, epsilon: float):
    import cv2
    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ()
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return ()
    simplified = cv2.approxPolyDP(
        contour, epsilon * cv2.arcLength(contour, True), True
    ).reshape(-1, 2)
    return tuple((float(x) / width, float(y) / height) for x, y in simplified)


class GroundedAnatomy4:
    """Lazily-loaded detector + segmenter with per-class runtime settings."""

    def __init__(self, config: AnatomyConfig, device: str = "cuda") -> None:
        self.config = config
        self.requested_device = device
        self.device = None
        self.last_memory: dict[str, Any] = {}
        self._torch = None
        self._detector_processor = None
        self._detector = None
        self._sam_processor = None
        self._sam = None
        self.provenance: dict[str, Any] = {}

    def load(self) -> None:
        import torch
        import transformers
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        self._torch = torch
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.device = self.requested_device
        LOGGER.info("loading detector=%s segmenter=%s on %s", self.config.detector, self.config.segmenter, self.device)
        self._detector_processor = AutoProcessor.from_pretrained(self.config.detector)
        self._detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.detector
        ).to(self.device)
        self._sam_processor = Sam2Processor.from_pretrained(self.config.segmenter)
        self._sam = Sam2Model.from_pretrained(self.config.segmenter).to(self.device)
        self._detector.eval()
        self._sam.eval()
        detector_commit = getattr(self._detector.config, "_commit_hash", None)
        segmenter_commit = getattr(self._sam.config, "_commit_hash", None)
        self.provenance = {
            "contract": self.config.contract,
            "config_fingerprint": self.config.fingerprint(),
            "detector": self.config.detector,
            "detector_revision": detector_commit,
            "segmenter": self.config.segmenter,
            "segmenter_revision": segmenter_commit,
            "device": self.device,
            "device_name": (
                torch.cuda.get_device_name(0)
                if self.device == "cuda" and torch.cuda.is_available()
                else None
            ),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "dtype": "float32",
            "loaded_at": _utc_now(),
        }
        self._record_memory("after_load")

    def _record_memory(self, phase: str) -> None:
        torch = self._torch
        if torch is None or self.device != "cuda" or not torch.cuda.is_available():
            return
        self.last_memory = {
            "phase": phase,
            "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        }

    def _release(self) -> None:
        if self.config.empty_cache_each_image and self.device == "cuda":
            gc.collect()
            self._torch.cuda.empty_cache()

    def _detect_class(self, image, spec, width: int, height: int):
        torch = self._torch
        prompt_text = ". ".join(spec.prompts) + "."
        inputs = self._detector_processor(
            images=image, text=prompt_text, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self._detector(**inputs)
        processed = self._detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=spec.box_threshold,
            text_threshold=spec.text_threshold,
            target_sizes=[(height, width)],
        )[0]
        boxes = processed.get("boxes")
        scores = processed.get("scores")
        labels = processed.get("text_labels")
        if labels is None:
            labels = processed.get("labels")
        if boxes is None or scores is None or labels is None or len(boxes) == 0:
            return [], 0

        import torchvision

        label_list = [str(label) for label in labels]
        unique = {label: idx for idx, label in enumerate(sorted(set(label_list)))}
        class_idxs = torch.tensor(
            [unique[label] for label in label_list], device=boxes.device
        )
        keep = torchvision.ops.batched_nms(
            boxes, scores, class_idxs, iou_threshold=self.config.nms_iou
        )
        keep_list = keep.cpu().tolist()
        suppressed = len(boxes) - len(keep_list)
        kept_boxes = boxes[keep]
        if len(kept_boxes) == 0:
            return [], suppressed

        sam_inputs = self._sam_processor(
            images=image,
            input_boxes=[kept_boxes.detach().cpu().tolist()],
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            sam_outputs = self._sam(
                **sam_inputs, multimask_output=self.config.multimask_output
            )
        masks = self._sam_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"]
        )[0]
        mask_scores = sam_outputs.iou_scores[0].cpu()

        detections: list[dict[str, Any]] = []
        for position, original_index in enumerate(keep_list):
            box = boxes[original_index].detach().cpu().tolist()
            label = label_list[original_index]
            mapped, mapping_reason = map_detector_label(label, spec.prompts)
            polygon = _polygon_from_mask(
                masks[position][0].numpy(), width, height, self.config.polygon_epsilon
            )
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
            if mapped != spec.name:
                status = f"rejected_label:{mapping_reason}"
            elif box_area > self.config.max_box_area:
                status = "rejected_area:dominates_frame"
            elif box_area < self.config.min_box_area:
                status = "rejected_area:too_small"
            elif len(polygon) < self.config.min_polygon_points:
                status = "mask_failed"
            detections.append(
                {
                    "class_name": spec.name,
                    "class_id": CLASS_IDS[spec.name],
                    "prompt": ". ".join(spec.prompts),
                    "detector_label": label,
                    "box_score": float(scores[original_index]),
                    "mask_iou": float(mask_scores[position, 0]),
                    "detector_bbox": list(normalized),
                    "detector_box_area": box_area,
                    "polygon": [list(point) for point in polygon],
                    "bbox": list(polygon_to_bbox(polygon)) if len(polygon) >= 3 else None,
                    "area_fraction": polygon_area(polygon) if len(polygon) >= 3 else None,
                    "status": status,
                }
            )
        return detections, suppressed

    def run_image(self, image_path: str | Path, image_id: str, observation_id: str) -> dict[str, Any]:
        from PIL import Image

        torch = self._torch
        started = time.perf_counter()
        original = Image.open(image_path).convert("RGB")
        original_width, original_height = original.size
        max_side = self.config.max_image_side
        scale = min(1.0, max_side / max(original_width, original_height))
        if scale < 1.0:
            inference_image = original.resize(
                (round(original_width * scale), round(original_height * scale)),
                Image.Resampling.LANCZOS,
            )
        else:
            inference_image = original
        width, height = inference_image.size

        proposals: list[dict[str, Any]] = []
        raw_counts: dict[str, int] = {}
        suppressed_counts: dict[str, int] = {}
        for spec in self.config.classes:
            detections, suppressed = self._detect_class(inference_image, spec, width, height)
            raw_counts[spec.name] = len(detections)
            suppressed_counts[spec.name] = suppressed
            for index, detection in enumerate(detections):
                detection.update(
                    {
                        "proposal_id": f"a4-{image_id}-{spec.name}-{index:03d}",
                        "image_id": image_id,
                        "photo_id": image_id,
                        "observation_id": observation_id,
                        "image_path": str(image_path),
                        "created_at": _utc_now(),
                        "model": self.provenance,
                    }
                )
                proposals.append(detection)

        self._record_memory(f"image:{image_id}")
        elapsed = time.perf_counter() - started
        return {
            "image_id": image_id,
            "observation_id": observation_id,
            "image_path": str(image_path),
            "download": {},
            "original_size": [original_width, original_height],
            "inference_size": [width, height],
            "scale": scale,
            "elapsed_seconds": round(elapsed, 3),
            "raw_detections_by_class": raw_counts,
            "nms_suppressed_by_class": suppressed_counts,
            "proposals": proposals,
            "memory": self.last_memory,
            "model": self.provenance,
        }


def run_prelabel(
    manifest: dict[str, Any],
    config: AnatomyConfig,
    out_dir: str | Path,
    *,
    device: str = "cuda",
    resume: bool = False,
    limit: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    output = Path(out_dir)
    items_dir = output / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    images = [image for image in manifest.get("images", []) if image.get("selected_for_audit")]
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError("manifest has no selected images")

    model = GroundedAnatomy4(config, device=device)
    model.load()
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, image in enumerate(images, start=1):
        photo_id = str(image["photo_id"])
        image_path = image.get("download", {}).get("path")
        if not image_path or not Path(image_path).is_file():
            missing.append(photo_id)
            continue
        item_path = items_dir / f"{photo_id}.json"
        if resume and item_path.is_file():
            records.append(json.loads(item_path.read_text()))
            if progress:
                LOGGER.info("[%d/%d] %s cached", index, len(images), photo_id)
            continue
        record = model.run_image(
            image_path,
            photo_id,
            str(image.get("observation_id") or ""),
        )
        record["download"] = image.get("download", {})
        item_path.write_text(json.dumps(record, indent=1))
        records.append(record)
        if progress:
            LOGGER.info(
                "[%d/%d] %s proposals=%d %.2fs mem=%s",
                index,
                len(images),
                photo_id,
                len(record["proposals"]),
                record["elapsed_seconds"],
                record.get("memory", {}).get("reserved_mb"),
            )

    proposals = [proposal for record in records for proposal in record["proposals"]]
    output_manifest = {
        "schema": "arq.anatomy4.prelabel/v1",
        "contract": config.contract,
        "created_at": _utc_now(),
        "config_fingerprint": config.fingerprint(),
        "config": {
            "detector": config.detector,
            "segmenter": config.segmenter,
            "max_image_side": config.max_image_side,
            "nms_iou": config.nms_iou,
            "classes": [
                {
                    "id": c.id,
                    "name": c.name,
                    "prompts": list(c.prompts),
                    "box_threshold": c.box_threshold,
                    "text_threshold": c.text_threshold,
                }
                for c in config.classes
            ],
        },
        "provenance": model.provenance,
        "images": records,
        "proposals": proposals,
        "summary": {
            "images_requested": len(images),
            "images_processed": len(records),
            "images_missing_file": missing,
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


def configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("ANATOMY4_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
