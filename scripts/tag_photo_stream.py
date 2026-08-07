"""Generate ranked plant-organ notice polygons from an observation photo CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmentation.grounded_sam2 import GroundedSam2Segmenter  # noqa: E402
from segmentation.pipeline import DEFAULT_PROMPTS, build_notice_batch, load_inaturalist_frames  # noqa: E402


DEFAULT_SOURCE = ROOT / "observations_pound_ridge" / "ward_pound_ridge_species.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ground plant organs with text, refine with SAM 2.1, and export ranked polygons."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--selection-scope",
        choices=("category", "subject-category"),
        default="category",
        help="Rank for a review gallery globally by organ, or per observed subject and organ.",
    )
    parser.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--segmenter-model", default="facebook/sam2.1-hiera-small")
    args = parser.parse_args()

    prompts = tuple(item.strip() for item in args.prompts.split(",") if item.strip())
    frames = load_inaturalist_frames(args.source, limit=args.limit)
    segmenter = GroundedSam2Segmenter(
        detector_model=args.detector_model,
        segmenter_model=args.segmenter_model,
        device=args.device,
    )
    batch = build_notice_batch(
        frames,
        segmenter,
        prompts=prompts,
        top_k=args.top_k,
        selection_scope=args.selection_scope,
        source_name=str(args.source),
    )
    batch.write(args.output)
    summary = batch.to_dict()["summary"]
    print(
        f"Wrote {summary['proposals']} proposals from {summary['frames']} frames; "
        f"{summary['selected_for_review']} selected for review; "
        f"{summary['failed_frames']} failed frames recorded to {args.output}"
    )


if __name__ == "__main__":
    main()
