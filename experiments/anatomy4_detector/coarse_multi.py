# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Cheap coarse cross-organ tags for the remaining Plantae pool.

Uses the existing YOLOE + OWLv2 cross-organ proposal dump (the same detectors
the director asked to run) and applies the coarse-tag policy:

* leaf — YOLOE ``leaf`` >= 0.24
* flower — OWLv2 ``single flower`` >= 0.24
* inflorescence — OWLv2 ``cluster of flowers`` >= 0.28

Only clean frames are accepted: 1..8 boxes, every box area <= 0.55, and the
low-confidence leaf clutter rule (leaf proposals >= 0.10 may exceed accepted
leaf boxes by at most 3). Images already in curated-multi / curated-needle /
blind-test, or in curated-multi's excluded list, are never re-added.

Output rows use the reviewed-source schema (`labels`, `source="model"`,
`review_state="accepted"`, full detector provenance) so `combine` can consume
them as an extra curated-style source.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import CLASS_IDS, CLASS_NAMES, CONTRACT

COARSE_SCHEMA = "arq.anatomy4.coarse-multi/v1"
DEFAULT_THRESHOLDS = {
    "leaf": 0.24,
    "single flower": 0.24,
    "cluster of flowers": 0.28,
}
MAX_BOXES = 8
MAX_AREA = 0.55
CLUTTER_MARGIN = 3
LOW_CONF_LEAF = 0.10


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"-tc-([0-9A-Za-z]+)-", text) or re.search(r"tc-([0-9A-Za-z]+)", text)
    if match:
        return f"tc-{match.group(1)}"
    stem = Path(text).stem
    match = re.search(r"tc-([0-9A-Za-z]+)", stem)
    return f"tc-{match.group(1)}" if match else None


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if not source.is_file():
        return rows
    for line in source.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _expand(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        text = str(pattern)
        paths.extend(
            [Path(item) for item in sorted(glob.glob(text))]
            if any(ch in text for ch in "*?[")
            else [Path(text)]
        )
    return sorted({path for path in paths if path.is_file()})


def _collect_ids(
    patterns: Sequence[str | Path],
    *,
    excluded_patterns: Sequence[str | Path] = (),
) -> tuple[set[str], set[str]]:
    accepted: set[str] = set()
    excluded: set[str] = set()
    for path in _expand(patterns):
        for row in _read_jsonl(path):
            image_id = _canonical(
                row.get("review_image_id")
                or row.get("image_id")
                or row.get("image_path")
            )
            if image_id:
                accepted.add(image_id)
    for path in _expand(excluded_patterns):
        for row in _read_jsonl(path):
            image_id = _canonical(
                row.get("review_image_id")
                or row.get("image_id")
                or row.get("image_path")
            )
            if image_id:
                excluded.add(image_id)
    return accepted, excluded


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / (area_a + area_b - inter) if area_a + area_b - inter > 0 else 0.0


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return None
    return (x0, y0, x1, y1)


def build_coarse_multi(
    *,
    proposals_path: str | Path,
    manifest_path: str | Path,
    curated_multi_globs: Sequence[str | Path],
    curated_needle_globs: Sequence[str | Path],
    blind_manifest_path: str | Path,
    blind_annotations_path: str | Path,
    out_path: str | Path,
    report_path: str | Path | None = None,
    leaf_threshold: float = DEFAULT_THRESHOLDS["leaf"],
    flower_threshold: float = DEFAULT_THRESHOLDS["single flower"],
    cluster_threshold: float = DEFAULT_THRESHOLDS["cluster of flowers"],
) -> dict[str, Any]:
    proposals_doc = json.loads(Path(proposals_path).read_text())
    proposals = proposals_doc.get("proposals") or []
    if not proposals:
        raise RuntimeError("hybrid proposals contain no proposals[]")
    manifest_rows = _read_jsonl(manifest_path)
    manifest = {
        str(row.get("image_id") or row.get("review_image_id") or ""): row
        for row in manifest_rows
        if row.get("image_id") or row.get("review_image_id")
    }
    curated_multi, curated_multi_excluded = _collect_ids(
        curated_multi_globs,
        excluded_patterns=[
            str(Path(pattern).parent / "excluded.jsonl")
            for pattern in curated_multi_globs
            if any(ch in str(pattern) for ch in "*?[")
        ],
    )
    curated_needle, _ = _collect_ids(curated_needle_globs)
    blind_ids: set[str] = set()
    blind_obs: set[str] = set()
    for path in (blind_manifest_path, blind_annotations_path):
        for row in _read_jsonl(path):
            image_id = _canonical(row.get("review_image_id") or row.get("image_id"))
            if image_id:
                blind_ids.add(image_id)
            if row.get("observation_id"):
                blind_obs.add(str(row["observation_id"]))
    reserved = curated_multi | curated_needle | blind_ids | curated_multi_excluded

    pool = sorted(set(manifest) - reserved)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        image_id = _canonical(proposal.get("image_id") or proposal.get("photo_id"))
        if image_id:
            by_image[image_id].append(proposal)

    excluded_reasons: Counter[str] = Counter()
    excluded_ids: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    boxes_per_class: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    considered = 0

    for image_id in pool:
        proposals_for_image = by_image.get(image_id)
        if not proposals_for_image:
            excluded_reasons["no_proposals"] += 1
            excluded_ids["no_proposals"].append(image_id)
            continue
        considered += 1
        accepted: dict[tuple[int, tuple[float, float, float, float]], dict[str, Any]] = {}
        low_conf_leaf = 0
        for proposal in proposals_for_image:
            class_name = str(proposal.get("class_name") or "")
            prompt = str(proposal.get("prompt") or "").strip().lower()
            score = float(proposal.get("box_score") or 0.0)
            bbox = _bbox(proposal.get("bbox") or proposal.get("detector_bbox"))
            if class_name == "leaf" and score >= LOW_CONF_LEAF:
                low_conf_leaf += 1
            if bbox is None:
                continue
            keep = False
            if class_name == "leaf" and score >= leaf_threshold:
                keep = True
            elif class_name == "flower" and "single flower" in prompt and score >= flower_threshold:
                keep = True
            elif (
                class_name == "inflorescence"
                and "cluster of flowers" in prompt
                and score >= cluster_threshold
            ):
                keep = True
            if keep:
                key = (CLASS_IDS[class_name], tuple(round(v, 6) for v in bbox))
                accepted[key] = {
                    "class": class_name,
                    "bbox": bbox,
                    "confidence": score,
                    "prompt": proposal.get("prompt"),
                    "proposal_id": proposal.get("proposal_id"),
                    "provenance": {
                        "coarse_policy": COARSE_SCHEMA,
                        "detector": (proposal.get("model") or {}).get("detector")
                        or proposal.get("model"),
                        "model": proposal.get("model"),
                        "detector_label": proposal.get("detector_label"),
                        "prompt": proposal.get("prompt"),
                        "box_score": score,
                        "proposal_id": proposal.get("proposal_id"),
                        "thresholds": {
                            "leaf": leaf_threshold,
                            "single_flower": flower_threshold,
                            "cluster_of_flowers": cluster_threshold,
                        },
                    },
                }

        labels = list(accepted.values())
        if not labels:
            excluded_reasons["no_accepted_boxes"] += 1
            excluded_ids["no_accepted_boxes"].append(image_id)
            continue
        if len(labels) > MAX_BOXES:
            excluded_reasons["too_many_boxes"] += 1
            excluded_ids["too_many_boxes"].append(image_id)
            continue
        max_area = max((b[2] - b[0]) * (b[3] - b[1]) for b in (l["bbox"] for l in labels))
        if max_area > MAX_AREA:
            excluded_reasons["box_too_large"] += 1
            excluded_ids["box_too_large"].append(image_id)
            continue
        if low_conf_leaf - len([l for l in labels if l["class"] == "leaf"]) > CLUTTER_MARGIN:
            excluded_reasons["leaf_clutter"] += 1
            excluded_ids["leaf_clutter"].append(image_id)
            continue

        record = manifest.get(image_id) or {}
        observation_id = str(record.get("observation_id") or "")
        if not observation_id:
            excluded_reasons["no_observation_id"] += 1
            excluded_ids["no_observation_id"].append(image_id)
            continue
        rows.append(
            {
                "schema": "arq.anatomy4.coarse-label/v1",
                "contract": CONTRACT,
                "image_id": image_id,
                "review_image_id": image_id,
                "observation_id": observation_id,
                "batch": "coarse-multi",
                "image_path": record.get("image_path") or record.get("local_image_path"),
                "photo_url": record.get("photo_url"),
                "source": "model",
                "review_state": "accepted",
                "labels": [
                    {
                        "class": label["class"],
                        "bbox": [round(value, 6) for value in label["bbox"]],
                        "confidence": round(label["confidence"], 6),
                        "source": "model",
                        "review_state": "accepted",
                        "provenance": label["provenance"],
                    }
                    for label in labels
                ],
                "provenance": {
                    "schema": COARSE_SCHEMA,
                    "policy": "coarse-cross-organ",
                    "reviewer": "auto",
                    "image_id": image_id,
                    "observation_id": observation_id,
                    "low_conf_leaf_proposals": low_conf_leaf,
                    "accepted_leaf_boxes": sum(1 for l in labels if l["class"] == "leaf"),
                    "thresholds": {
                        "leaf": leaf_threshold,
                        "single_flower": flower_threshold,
                        "cluster_of_flowers": cluster_threshold,
                    },
                    "clean_frame": {
                        "max_boxes": MAX_BOXES,
                        "max_area": MAX_AREA,
                        "leaf_clutter_margin": CLUTTER_MARGIN,
                    },
                },
            }
        )
        for label in labels:
            boxes_per_class[label["class"]] += 1
            prompt_counts[str(label["prompt"])] += 1

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    report = {
        "schema": COARSE_SCHEMA,
        "contract": CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "proposals": {"path": str(Path(proposals_path).resolve()), "sha256": _sha256(proposals_path)},
            "manifest": {"path": str(Path(manifest_path).resolve()), "sha256": _sha256(manifest_path)},
            "blind_manifest": str(Path(blind_manifest_path).resolve()),
            "blind_annotations": str(Path(blind_annotations_path).resolve()),
            "curated_multi_globs": [str(p) for p in curated_multi_globs],
            "curated_needle_globs": [str(p) for p in curated_needle_globs],
        },
        "excluded_sets": {
            "curated_multi": len(curated_multi),
            "curated_multi_excluded": len(curated_multi_excluded),
            "curated_needle": len(curated_needle),
            "blind": len(blind_ids),
            "reserved_total": len(reserved),
        },
        "pool": {
            "manifest_images": len(manifest),
            "images_considered": considered,
            "images_with_coarse_labels": len(rows),
            "observations": len({row["observation_id"] for row in rows}),
        },
        "boxes": {
            "total": sum(boxes_per_class.values()),
            "per_class": {name: boxes_per_class.get(name, 0) for name in CLASS_NAMES},
            "per_prompt": dict(prompt_counts),
        },
        "excluded": {
            "reasons": dict(excluded_reasons),
            "sample_ids": {key: values[:10] for key, values in excluded_ids.items()},
        },
        "thresholds": {
            "leaf": leaf_threshold,
            "single_flower": flower_threshold,
            "cluster_of_flowers": cluster_threshold,
        },
        "output": str(target.resolve()),
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=1) + "\n")
    return report
