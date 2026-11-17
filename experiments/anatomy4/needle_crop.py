# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Crop-level second pass for the needle class.

The whole-image open-vocab pass cannot separate "conifer in the scene" from
"needle-bearing foliage in the box", and whole-tree boxes are useless for
segmentation. This stage takes the detector's needle candidates, crops each box
(with margin), and re-prompts on the crop. A candidate is kept only when the
crop pass confirms conifer foliage.

No rectangle fallback: the crop is an evaluation window only; the proposal's
own polygon is what would be exported, and nothing here writes labels.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SECOND_PROMPTS = ["conifer needles", "pine needles", "spruce branch"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_needle_second_pass(
    manifest: Mapping[str, Any],
    out_dir: str | Path,
    *,
    model_name: str = "yoloe-26s-seg.pt",
    imgsz: int = 1024,
    device: str = "0",
    crop_margin: float = 0.15,
    min_candidate_score: float = 0.05,
    second_prompts: list[str] | None = None,
    conf_floor: float = 0.02,
    limit: int | None = None,
    first_prompt_only: bool = False,
) -> dict[str, Any]:
    from PIL import Image
    from ultralytics import YOLOE

    output = Path(out_dir)
    crops_dir = output / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    prompts = second_prompts or SECOND_PROMPTS
    candidates = [
        proposal
        for proposal in manifest.get("proposals", [])
        if proposal.get("class_name") == "needle"
        and proposal.get("status") == "proposed"
        and float(proposal.get("box_score") or 0) >= min_candidate_score
    ]
    if limit is not None:
        candidates = candidates[:limit]

    model = YOLOE(model_name)
    model.set_classes(prompts, model.get_text_pe(prompts))
    image_paths = {
        str(image.get("photo_id") or image.get("image_id")): image.get("download", {}).get("path")
        for image in manifest.get("images", [])
    }
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for candidate in candidates:
        photo_id = str(candidate.get("photo_id") or candidate.get("image_id"))
        image_path = image_paths.get(photo_id) or candidate.get("image_path")
        if not image_path or not Path(image_path).is_file():
            continue
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        x0, y0, x1, y1 = candidate["detector_bbox"]
        margin_x = (x1 - x0) * crop_margin
        margin_y = (y1 - y0) * crop_margin
        pixel_box = (
            max(0, int((x0 - margin_x) * width)),
            max(0, int((y0 - margin_y) * height)),
            min(width, int((x1 + margin_x) * width)),
            min(height, int((y1 + margin_y) * height)),
        )
        if pixel_box[2] - pixel_box[0] < 24 or pixel_box[3] - pixel_box[1] < 24:
            rows.append(
                {
                    "proposal_id": candidate["proposal_id"],
                    "photo_id": photo_id,
                    "candidate_score": candidate["box_score"],
                    "candidate_prompt": candidate.get("prompt"),
                    "candidate_area": candidate.get("detector_box_area"),
                    "crop_score": 0.0,
                    "crop_prompt": "",
                    "status": "crop_too_small",
                }
            )
            continue
        crop = image.crop(pixel_box)
        crop_path = crops_dir / f"{candidate['proposal_id']}.jpg"
        crop.save(crop_path, quality=92)
        result = model.predict(
            str(crop_path),
            imgsz=imgsz,
            conf=conf_floor,
            iou=0.5,
            device=device,
            verbose=False,
        )[0]
        best_score = 0.0
        best_prompt = ""
        for score, class_id in zip(result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            if first_prompt_only and int(class_id) != 0:
                continue
            if float(score) > best_score:
                best_score = float(score)
                best_prompt = prompts[int(class_id)]
        rows.append(
            {
                "proposal_id": candidate["proposal_id"],
                "photo_id": photo_id,
                "candidate_score": round(float(candidate["box_score"]), 4),
                "candidate_prompt": candidate.get("prompt"),
                "candidate_area": round(float(candidate.get("detector_box_area") or 0), 4),
                "crop_score": round(best_score, 4),
                "crop_prompt": best_prompt,
                "crop_box": pixel_box,
                "status": "scored",
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "needle-second-pass.csv"
    fieldnames = [
        "proposal_id",
        "photo_id",
        "candidate_score",
        "candidate_prompt",
        "candidate_area",
        "crop_score",
        "crop_prompt",
        "status",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "candidates": len(candidates),
        "scored": len([row for row in rows if row.get("status") == "scored"]),
        "crops_written": len(list(crops_dir.glob("*.jpg"))),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "provenance": {
            "detector": model_name,
            "second_prompts": prompts,
            "crop_margin": crop_margin,
            "min_candidate_score": min_candidate_score,
            "imgsz": imgsz,
            "conf_floor": conf_floor,
            "created_at": _utc_now(),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
