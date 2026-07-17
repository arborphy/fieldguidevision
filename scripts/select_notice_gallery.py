"""Create a small review gallery from an immutable notice-proposal batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmentation.pipeline import canonical_organ_label


def _group_key(annotation: dict, scope: str) -> tuple[str, ...]:
    raw_label = annotation.get("model", {}).get("detector_label") or annotation.get("model", {}).get("prompt")
    category = f"category:{canonical_organ_label(raw_label)}" if raw_label else annotation["notice_category_id"]
    if scope == "category":
        return (category,)
    return (annotation["subject_id"], category)


def select_gallery(payload: dict, *, top_k: int, scope: str) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if scope not in {"category", "subject-category"}:
        raise ValueError("scope must be 'category' or 'subject-category'")
    groups: dict[tuple[str, ...], list[dict]] = {}
    for observation in payload["observations"]:
        for annotation in observation.get("annotations", []):
            groups.setdefault(_group_key(annotation, scope), []).append(annotation)
    selected: list[dict] = []
    for key in sorted(groups):
        candidates = sorted(
            groups[key],
            key=lambda item: (-item["quality"]["representative_score"], item["annotation_id"]),
        )
        for rank, annotation in enumerate(candidates[:top_k], start=1):
            selected.append({
                **annotation,
                "gallery_category_id": key[-1],
                "requires_organ_review": key[-1] == "category:ambiguous_organ",
                "gallery_group": list(key),
                "gallery_rank": rank,
            })
    return {
        "schema_version": "arq.notice-review-gallery/v1",
        "source_batch_schema_version": payload["schema_version"],
        "source": payload["source"],
        "model": payload["model"],
        "selection_scope": scope,
        "top_k": top_k,
        "selected_count": len(selected),
        "annotations": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top-ranked notice polygons for review.")
    parser.add_argument("source", type=Path, help="Immutable notice proposal batch JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--scope", choices=("category", "subject-category"), default="category")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Notice review gallery already exists: {args.output}")
    gallery = select_gallery(json.loads(args.source.read_text()), top_k=args.top_k, scope=args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gallery, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {gallery['selected_count']} ranked proposal polygons to {args.output}")


if __name__ == "__main__":
    main()
