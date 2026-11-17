# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""YOLO writer that consumes reviewed/accepted proposals only.

Accepted means the human review sheet says ``tp`` (or the proposal carries
``review_state == "accepted"``). Raw proposals are never exported. Splits are
observation-level: every image of an observation lands in the same split.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .contracts import CLASS_IDS, polygon_to_bbox, split_for_observation


def load_review(path: str | Path) -> dict[str, dict[str, str]]:
    reviews: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            proposal_id = (row.get("proposal_id") or "").strip()
            if proposal_id:
                reviews[proposal_id] = row
    return reviews


def is_accepted(proposal: Mapping[str, Any], reviews: Mapping[str, Mapping[str, str]]) -> bool:
    review = reviews.get(str(proposal.get("proposal_id")))
    if review is not None:
        return (review.get("verdict") or "").strip().lower() in {"tp", "accept", "accepted"}
    return str(proposal.get("review_state") or "").lower() == "accepted"


def load_verification(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read verifier verdict rows (``verdicts.jsonl``) keyed by proposal id.

    A later pass wins, mirroring the verifier's own effective-verdict rule.
    """

    verdicts: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        proposal_id = str(row.get("proposal_id") or "")
        if proposal_id:
            verdicts[proposal_id] = row
    return verdicts


def is_verified_accepted(
    proposal: Mapping[str, Any],
    verification: Mapping[str, Mapping[str, Any]],
    *,
    profile: str = "balanced",
) -> bool:
    """Apply the verifier accept policy to one proposal (no review sheet needed)."""

    from .verify import accept_decision

    row = verification.get(str(proposal.get("proposal_id")))
    if row is None:
        return False
    verdict = row.get("verdict") or {}
    return bool(
        accept_decision(
            verdict,
            str(row.get("proposed_class") or proposal.get("class_name")),
            profile=profile,
        )["accept"]
    )


def accepted_proposals(
    manifest: Mapping[str, Any],
    reviews: Mapping[str, Mapping[str, str]],
    verification: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    verification_profile: str = "balanced",
    require_both: bool = False,
) -> list[dict[str, Any]]:
    """Accepted proposals from the review sheet and/or the verifier.

    ``require_both`` is the conservative mode: a proposal must be human-accepted
    **and** verifier-accepted before it can become a label.
    """

    accepted: list[dict[str, Any]] = []
    for proposal in manifest.get("proposals", []):
        if proposal.get("status") != "proposed":
            continue
        if not proposal.get("polygon"):
            continue
        human = is_accepted(proposal, reviews) if reviews else False
        machine = (
            is_verified_accepted(proposal, verification, profile=verification_profile)
            if verification
            else False
        )
        if verification and reviews and require_both:
            keep = human and machine
        elif verification and reviews:
            keep = human or machine
        elif verification:
            keep = machine
        else:
            keep = human
        if keep:
            accepted.append(proposal)
    return accepted


def write_yolo_dataset(
    manifest: Mapping[str, Any],
    out_dir: str | Path,
    *,
    reviews: Mapping[str, Mapping[str, str]] | None = None,
    verification: Mapping[str, Mapping[str, Any]] | None = None,
    verification_profile: str = "balanced",
    require_both: bool = False,
    include_segmentation: bool = True,
    split_seed: str = "anatomy4-v1",
) -> dict[str, Any]:
    output = Path(out_dir)
    reviews = reviews or {}
    proposals = accepted_proposals(
        manifest,
        reviews,
        verification,
        verification_profile=verification_profile,
        require_both=require_both,
    )
    images = {
        str(image.get("photo_id") or image.get("image_id") or ""): image
        for image in manifest.get("images", [])
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        grouped[str(proposal["photo_id"])].append(proposal)

    counts: Counter[str] = Counter()
    observations_by_split: dict[str, set[str]] = defaultdict(set)
    written: list[str] = []
    for photo_id, items in grouped.items():
        observation_id = str(items[0].get("observation_id") or photo_id)
        split = split_for_observation(observation_id, seed=split_seed)
        label_dir = output / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        seg_lines: list[str] = []
        for proposal in items:
            class_id = CLASS_IDS[proposal["class_name"]]
            x0, y0, x1, y1 = polygon_to_bbox(proposal["polygon"])
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            width = x1 - x0
            height = y1 - y0
            if width <= 0 or height <= 0:
                continue
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
            if include_segmentation:
                ring = " ".join(f"{x:.6f} {y:.6f}" for x, y in proposal["polygon"])
                seg_lines.append(f"{class_id} {ring}")
            counts[proposal["class_name"]] += 1
        if not lines:
            continue
        (label_dir / f"{photo_id}.txt").write_text("\n".join(lines) + "\n")
        written.append(str(label_dir / f"{photo_id}.txt"))
        if include_segmentation:
            seg_dir = output / "labels_seg" / split
            seg_dir.mkdir(parents=True, exist_ok=True)
            (seg_dir / f"{photo_id}.txt").write_text("\n".join(seg_lines) + "\n")
        observations_by_split[split].add(observation_id)

    class_names = ["leaf", "needle", "flower", "inflorescence"]
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {json.dumps(class_names)}",
                "",
            ]
        )
    )
    # Copy/symlink only accepted images, preserving the observation-level split.
    for photo_id, items in grouped.items():
        image_record = images.get(photo_id, {})
        observation_id = str(items[0].get("observation_id") or photo_id)
        split = split_for_observation(observation_id, seed=split_seed)
        source = image_record.get("download", {}).get("path")
        if not source or not Path(source).is_file():
            continue
        target_dir = output / "images" / split
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(source).name
        if not target.exists():
            target.write_bytes(Path(source).read_bytes())

    summary = {
        "schema": "arq.anatomy4.yolo/v1",
        "contract": manifest.get("contract"),
        "review_sheet": bool(reviews),
        "verification": bool(verification),
        "verification_profile": verification_profile if verification else None,
        "require_both": bool(require_both),
        "accepted_proposals": len(proposals),
        "accepted_by_class": dict(counts),
        "images_written": len(written),
        "observations_by_split": {
            split: len(values) for split, values in observations_by_split.items()
        },
        "split_seed": split_seed,
        "label_files": written,
        "provenance": manifest.get("provenance"),
    }
    (output / "accepted.json").write_text(json.dumps(summary, indent=1))
    return summary
