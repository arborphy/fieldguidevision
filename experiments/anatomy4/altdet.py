# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Shared helpers for alternate open-vocabulary *box* proposal generators.

The anatomy4 contract expects proposals to be auditable and provenance-stamped.
YOLO-World and OWLv2 are box-only detectors (no instance masks), so proposals
from this module are explicitly marked ``proposal_kind = "box"`` and never carry
a fabricated polygon. Nothing here derives a box from taxonomy: every class is a
text prompt, and a proposal's class is the prompt that produced it.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import CLASS_IDS, AnatomyConfig, normalize_phrase

# Only prompts with prior measured support (anatomy4 120 + targeted audits).
# Declared per class so a class can never be produced by another class's text.
ALT_PROMPTS: dict[str, tuple[str, ...]] = {
    "leaf": ("leaf", "leaf blade", "broadleaf"),
    "needle": ("pine needles", "spruce needles", "conifer foliage"),
    "flower": ("single flower", "flower"),
    "inflorescence": ("cluster of flowers", "umbel of flowers", "flower head"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompts_for(config: AnatomyConfig) -> dict[str, list[str]]:
    """Per-class prompt lists, restricted to prompts that map unambiguously."""

    table: dict[str, list[str]] = {}
    for name in config.class_names:
        preferred = ALT_PROMPTS.get(name)
        if preferred:
            table[name] = list(preferred)
        else:
            table[name] = list(config.class_by_name[name].prompts)
    return table


def prompt_class_index(prompts: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Flat ``[(prompt, class_name), ...]`` in the detector's query order."""

    flat: list[tuple[str, str]] = []
    for class_name in ("leaf", "needle", "flower", "inflorescence"):
        for prompt in prompts.get(class_name, []):
            flat.append((prompt, class_name))
    return flat


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_same_class(
    proposals: list[dict[str, Any]],
    iou_threshold: float = 0.6,
    max_per_class: int | None = None,
) -> list[dict[str, Any]]:
    """Greedy per-class NMS across prompt variants.

    A leaf proposed by "leaf" and "leaf blade" is one object, not two; keeping
    both would inflate review counts and hide prompt-level precision. OWLv2
    additionally emits heavily nested boxes for one prompt, so an
    ``max_per_class`` cap is part of the operating point, not a post-hoc filter.
    """

    kept: list[dict[str, Any]] = []
    per_class: dict[str, int] = {}
    for proposal in sorted(proposals, key=lambda p: p["box_score"], reverse=True):
        class_name = proposal["class_name"]
        if max_per_class is not None and per_class.get(class_name, 0) >= max_per_class:
            continue
        same = [
            other
            for other in kept
            if other["class_name"] == class_name
            and _iou(other["detector_bbox"], proposal["detector_bbox"]) >= iou_threshold
        ]
        if same:
            continue
        kept.append(proposal)
        per_class[class_name] = per_class.get(class_name, 0) + 1
    return kept


def make_box_proposal(
    *,
    photo_id: str,
    observation_id: str,
    image_path: str,
    audit_bucket: str | None,
    class_name: str,
    prompt: str,
    score: float,
    box: Sequence[float],
    width: float,
    height: float,
    provenance: dict[str, Any],
    detector_rank: int,
    slug: str,
    config: AnatomyConfig,
) -> dict[str, Any]:
    x0, y0, x1, y1 = (float(v) for v in box)
    normalized = (
        max(0.0, min(1.0, min(x0, x1) / width)),
        max(0.0, min(1.0, min(y0, y1) / height)),
        max(0.0, min(1.0, max(x0, x1) / width)),
        max(0.0, min(1.0, max(y0, y1) / height)),
    )
    box_area = (normalized[2] - normalized[0]) * (normalized[3] - normalized[1])
    status = "proposed"
    if box_area > config.max_box_area:
        status = "rejected_area:dominates_frame"
    elif box_area < config.min_box_area:
        status = "rejected_area:too_small"
    prompt_slug = "".join(
        character if character.isalnum() else "-" for character in prompt.lower()
    ).strip("-")
    return {
        "proposal_id": f"a4{slug}-{photo_id}-{class_name}-{prompt_slug}-{detector_rank:03d}",
        "image_id": photo_id,
        "photo_id": photo_id,
        "observation_id": observation_id,
        "audit_bucket": audit_bucket,
        "image_path": image_path,
        "class_name": class_name,
        "class_id": CLASS_IDS[class_name],
        "prompt": prompt,
        "prompt_variant": prompt,
        "detector_label": prompt,
        "box_score": float(score),
        "mask_iou": None,
        "proposal_kind": "box",
        "detector_bbox": list(normalized),
        "detector_box_area": box_area,
        "polygon": [],
        "bbox": list(normalized),
        "area_fraction": box_area,
        "status": status,
        "created_at": utc_now(),
        "model": provenance,
    }


def select_images(
    manifest: dict[str, Any], *, bucket: str | None, limit: int | None
) -> list[dict[str, Any]]:
    images = [
        image for image in manifest.get("images", []) if image.get("selected_for_audit")
    ]
    if bucket:
        images = [image for image in images if image.get("audit_bucket") == bucket]
    if limit is not None:
        images = images[:limit]
    return images


def build_output_manifest(
    records: Iterable[dict[str, Any]],
    *,
    config: AnatomyConfig,
    backend: str,
    provenance: dict[str, Any],
    audit_bucket: str | None,
    total_runtime_seconds: float,
    prompts: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    record_list = list(records)
    proposals = [proposal for record in record_list for proposal in record["proposals"]]
    return {
        "schema": "arq.anatomy4.prelabel/v1",
        "contract": config.contract,
        "created_at": utc_now(),
        "backend": backend,
        "config_fingerprint": config.fingerprint(),
        "audit_bucket": audit_bucket,
        "prompts": {
            name: list(p)
            for name, p in (prompts or prompts_for(config)).items()
        },
        "provenance": provenance,
        "images": record_list,
        "proposals": proposals,
        "summary": {
            "images_requested": len(record_list),
            "images_processed": len(record_list),
            "images_with_proposals": len(
                {p["photo_id"] for p in proposals if p["status"] == "proposed"}
            ),
            "proposals": len(proposals),
            "proposals_by_class": {
                name: len(
                    [p for p in proposals if p["class_name"] == name and p["status"] == "proposed"]
                )
                for name in config.class_names
            },
            "rejected_by_reason": {
                reason: len([p for p in proposals if p["status"] == reason])
                for reason in sorted({p["status"] for p in proposals if p["status"] != "proposed"})
            },
            "total_runtime_seconds": round(total_runtime_seconds, 2),
        },
    }


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def describe_prompt_mapping(prompts: dict[str, list[str]]) -> dict[str, str]:
    return {
        normalize_phrase(prompt): class_name
        for class_name, values in prompts.items()
        for prompt in values
    }


class Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started
