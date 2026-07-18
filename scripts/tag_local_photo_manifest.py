#!/usr/bin/env python3
"""Run Grounding-DINO + SAM 2.1 over an immutable local photo manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmentation.grounded_sam2 import GroundedSam2Segmenter  # noqa: E402
from segmentation.pipeline import CaptureFrame, DEFAULT_PROMPTS, build_notice_batch  # noqa: E402


def _frames(manifest: dict, limit: int | None) -> tuple[CaptureFrame, ...]:
    ready = [row for row in manifest["frames"] if row["status"] == "ready"]
    if limit is not None:
        ready = ready[:limit]
    return tuple(CaptureFrame(
        observation_id=row["observation_id"], image_url=row["image_url"],
        local_path=row["local_path"], image_sha256=row["image_sha256"],
        subject_id=f"hypothesis:{row['observation_id']}", captured_at=row.get("captured_at"),
        frame_index=index,
    ) for index, row in enumerate(ready))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--segmenter-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Segmentation batch already exists: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    frames = _frames(manifest, args.limit)
    prompts = tuple(item.strip() for item in args.prompts.split(",") if item.strip())
    segmenter = GroundedSam2Segmenter(detector_model=args.detector_model, segmenter_model=args.segmenter_model, device=args.device)
    batch = build_notice_batch(frames, segmenter, prompts=prompts, top_k=args.top_k, selection_scope="subject-category", source_name=str(args.manifest.resolve()))
    payload = batch.to_dict()
    payload["input_manifest"] = {
        "schema_version": manifest["schema_version"], "source_area": manifest["source_area"],
        "frames": [{"observation_id": frame.observation_id, "image_sha256": frame.image_sha256, "local_path": frame.local_path} for frame in frames],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
