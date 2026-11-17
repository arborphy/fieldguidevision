# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Crop builder: reviewed detector boxes -> padded, materialized crops.

Hard rules enforced here (and unit-tested):

* only a **detector** box may become a crop. A taxon-derived box, a
  full-frame fallback, or a proposal without a box is rejected. There is no
  code path that infers a box from ``taxon_name``.
* only proposals a human accepted (``tp`` / ``accept`` / ``accepted``) are
  cropped, unless the caller explicitly opts into unreviewed proposals.
* the crop is padded by a configurable fraction of the detector box and
  clamped to the source image; padding is never allowed to replace the box.
* the output manifest matches the sprint input contract: observation_id,
  taxon_name, image/photo_url, crop_path, anatomy_class, detector confidence,
  and observation split.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    ANATOMY_CLASSES,
    CONTRACT,
    CROP_MANIFEST_SCHEMA,
    normalize_crop_row,
    write_jsonl,
)
from .splits import observation_split

ACCEPTED_VERDICTS = {"tp", "accept", "accepted", "true_positive"}
REJECTED_VERDICTS = {"fp", "reject", "rejected", "false_positive", "unsure", "?"}
TAXON_BOX_SOURCES = {"taxon", "taxonomy", "taxon_derived", "taxon-derived", "label", "claim"}


@dataclass
class CropBuildReport:
    written: int = 0
    skipped: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)
    by_class: Counter[str] = field(default_factory=Counter)
    by_split: Counter[str] = field(default_factory=Counter)
    crops: list[dict[str, Any]] = field(default_factory=list)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "arq.anatomy4_specialists.crop_build/v1",
            "contract": CONTRACT,
            "written": self.written,
            "skipped": self.skipped,
            "skip_reasons": dict(sorted(self.skip_reasons.items())),
            "by_anatomy_class": dict(sorted(self.by_class.items())),
            "by_split": dict(sorted(self.by_split.items())),
        }


def load_reviews(path_or_rows: str | Path | Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """Load a review sheet (CSV path or row sequence) keyed by proposal_id."""

    if isinstance(path_or_rows, (str, Path)):
        rows: list[Mapping[str, Any]] = []
        with Path(path_or_rows).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = list(path_or_rows)
    reviews: dict[str, dict[str, str]] = {}
    for row in rows:
        proposal_id = str(row.get("proposal_id") or "").strip()
        if proposal_id:
            reviews[proposal_id] = {str(k): ("" if v is None else str(v)) for k, v in row.items()}
    return reviews


def _verdict(proposal: Mapping[str, Any], reviews: Mapping[str, Mapping[str, str]]) -> str:
    review = reviews.get(str(proposal.get("proposal_id")))
    if review is not None:
        return str(review.get("verdict") or "").strip().lower()
    return str(proposal.get("review_state") or "").strip().lower()


def is_accepted(
    proposal: Mapping[str, Any],
    reviews: Mapping[str, Mapping[str, str]],
    *,
    allow_unreviewed: bool = False,
) -> tuple[bool, str]:
    verdict = _verdict(proposal, reviews)
    if verdict in ACCEPTED_VERDICTS:
        return True, "accepted"
    if verdict in REJECTED_VERDICTS:
        return False, f"review:{verdict or 'unsure'}"
    if not verdict and allow_unreviewed:
        return True, "unreviewed_allowed"
    return False, "review:missing"


def _detector_bbox(proposal: Mapping[str, Any]) -> tuple[list[float] | None, str]:
    """Return a normalized detector xyxy box or a rejection reason."""

    source = str(proposal.get("box_source") or proposal.get("bbox_source") or "detector").lower()
    if bool(proposal.get("taxon_derived_box")) or source in TAXON_BOX_SOURCES:
        return None, "taxon_derived_box"
    bbox = proposal.get("detector_bbox", proposal.get("bbox"))
    if bbox is None:
        return None, "missing_detector_box"
    if len(bbox) != 4:
        return None, "invalid_detector_box"
    try:
        values = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None, "invalid_detector_box"
    x0, y0, x1, y1 = values
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 <= x0 or y1 <= y0:
        return None, "degenerate_detector_box"
    if x1 <= 0 or y1 <= 0 or x0 >= 1 or y0 >= 1:
        return None, "offscreen_detector_box"
    clipped = [min(max(x0, 0.0), 1.0), min(max(y0, 0.0), 1.0), min(max(x1, 0.0), 1.0), min(max(y1, 0.0), 1.0)]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None, "degenerate_detector_box"
    return clipped, "detector"


def pad_box_pixels(
    bbox_px: Sequence[float],
    width: int,
    height: int,
    *,
    padding_frac: float,
    padding_mode: str = "relative",
    min_crop_px: int = 32,
) -> tuple[int, int, int, int]:
    """Pad a pixel xyxy box and clamp it to the image.

    ``relative`` pads each side by ``padding_frac * box dimension``;
    ``absolute`` pads by ``padding_frac * min(image side)``. Both are clamped.
    """

    if padding_frac < 0:
        raise ValueError("padding_frac must be >= 0")
    x0, y0, x1, y1 = (float(v) for v in bbox_px)
    box_w, box_h = x1 - x0, y1 - y0
    if padding_mode == "relative":
        pad_x, pad_y = padding_frac * box_w, padding_frac * box_h
    elif padding_mode == "absolute":
        base = padding_frac * min(width, height)
        pad_x = pad_y = base
    else:
        raise ValueError(f"unknown padding_mode {padding_mode!r}")
    cx0 = max(0, int(round(x0 - pad_x)))
    cy0 = max(0, int(round(y0 - pad_y)))
    cx1 = min(width, int(round(x1 + pad_x)))
    cy1 = min(height, int(round(y1 + pad_y)))
    if cx1 - cx0 < min_crop_px or cy1 - cy0 < min_crop_px:
        # Grow around the box center until the minimum is met, still clamped.
        cx = (cx0 + cx1) // 2
        cy = (cy0 + cy1) // 2
        half_w = max(min_crop_px, cx1 - cx0) // 2
        half_h = max(min_crop_px, cy1 - cy0) // 2
        cx0, cx1 = max(0, cx - half_w), min(width, cx + half_w)
        cy0, cy1 = max(0, cy - half_h), min(height, cy + half_h)
    return cx0, cy0, cx1, cy1


def crop_id_for(photo_id: str, bbox: Sequence[float], anatomy_class: str) -> str:
    payload = f"{photo_id}|{','.join(f'{v:.6f}' for v in bbox)}|{anatomy_class}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def load_observation_lookup(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path,
) -> dict[str, dict[str, str]]:
    """Load ``{observation_id: taxon_name}`` and ``{photo_id: url}`` maps.

    Accepts the anatomy4 manifest (``observations`` + ``images``), a JSONL
    stream of observation/image records, or an already-built lookup mapping.
    Real prelabel proposal documents do not carry taxon names or photo URLs,
    so this is the join that supplies them.
    """

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".jsonl":
            records: list[Mapping[str, Any]] = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            data: Any = records
        else:
            data = json.loads(path.read_text())
    elif isinstance(source, Mapping):
        if "taxon_by_observation" in source or "url_by_photo" in source:
            return {
                "taxon_by_observation": {
                    str(k): str(v) for k, v in (source.get("taxon_by_observation") or {}).items()
                },
                "url_by_photo": {
                    str(k): str(v) for k, v in (source.get("url_by_photo") or {}).items()
                },
            }
        data = source
    else:
        data = source

    observations: list[Mapping[str, Any]] = []
    images: list[Mapping[str, Any]] = []
    if isinstance(data, Mapping):
        observations = list(data.get("observations") or [])
        images = list(data.get("images") or [])
    elif isinstance(data, list):
        for record in data:
            if not isinstance(record, Mapping):
                continue
            if record.get("photo_id") or record.get("image_id"):
                images.append(record)
            else:
                observations.append(record)
                images.extend(record.get("images") or [])

    taxon_by_observation: dict[str, str] = {}
    url_by_photo: dict[str, str] = {}
    for observation in observations:
        observation_id = str(
            observation.get("observation_id")
            or observation.get("arborphy_id")
            or observation.get("id")
            or ""
        ).strip()
        taxon_name = str(observation.get("taxon_name") or observation.get("scientific_name") or "").strip()
        if observation_id and taxon_name:
            taxon_by_observation[observation_id] = taxon_name
        for image in observation.get("images") or []:
            photo_id = str(image.get("photo_id") or image.get("image_id") or image.get("id") or "").strip()
            url = str(image.get("url") or image.get("photo_url") or "").strip()
            if photo_id and url:
                url_by_photo[photo_id] = url
    for image in images:
        photo_id = str(image.get("photo_id") or image.get("image_id") or image.get("id") or "").strip()
        url = str(image.get("url") or image.get("photo_url") or "").strip()
        if photo_id and url:
            url_by_photo[photo_id] = url
        observation_id = str(image.get("observation_id") or "").strip()
        taxon_name = str(image.get("taxon_name") or "").strip()
        if observation_id and taxon_name:
            taxon_by_observation.setdefault(observation_id, taxon_name)
    return {"taxon_by_observation": taxon_by_observation, "url_by_photo": url_by_photo}


def _image_path_map(manifest: Mapping[str, Any], images_dir: str | Path | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for image in manifest.get("images", []):
        photo_id = str(image.get("photo_id") or image.get("image_id") or "")
        if not photo_id:
            continue
        download = image.get("download") or {}
        candidate = download.get("path") or image.get("path")
        if candidate:
            paths[photo_id] = Path(candidate)
    if images_dir is not None:
        root = Path(images_dir)
        for photo_id, path in list(paths.items()):
            if not path.is_absolute():
                paths[photo_id] = root / path
        # Also allow <images_dir>/<photo_id>.jpg and the downloaded basename.
        for image in manifest.get("images", []):
            photo_id = str(image.get("photo_id") or image.get("image_id") or "")
            if not photo_id or photo_id in paths:
                continue
            for suffix in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = root / f"{photo_id}{suffix}"
                if candidate.is_file():
                    paths[photo_id] = candidate
                    break
    return paths


def build_crops(
    proposals: Mapping[str, Any],
    out_dir: str | Path,
    *,
    reviews: Mapping[str, Mapping[str, str]] | None = None,
    images_dir: str | Path | None = None,
    padding_frac: float = 0.18,
    padding_mode: str = "relative",
    min_crop_px: int = 48,
    min_detector_confidence: float = 0.0,
    jpeg_quality: int = 92,
    allow_unreviewed: bool = False,
    observations: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path | None = None,
    split_seed: str = "anatomy4-specialists-v1",
    split_fractions: Mapping[str, float] | None = None,
    max_crops: int | None = None,
) -> CropBuildReport:
    """Materialize crops from reviewed detector proposals.

    ``proposals`` is the anatomy4 prelabel document (``images`` + ``proposals``)
    or any dict with those keys. Source image paths come from each image's
    ``download.path`` and/or ``images_dir``.
    """

    from PIL import Image

    reviews = reviews or {}
    lookup = load_observation_lookup(observations) if observations is not None else {
        "taxon_by_observation": {},
        "url_by_photo": {},
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_paths = _image_path_map(proposals, images_dir)
    images_by_id = {
        str(image.get("photo_id") or image.get("image_id")): image
        for image in proposals.get("images", [])
    }
    report = CropBuildReport()
    seen_crop_ids: set[str] = set()

    for proposal in proposals.get("proposals", []):
        if report.written >= (max_crops or 10**18):
            break
        if str(proposal.get("status") or "proposed") != "proposed":
            report.skip("proposal_status_not_proposed")
            continue
        accepted, reason = is_accepted(proposal, reviews, allow_unreviewed=allow_unreviewed)
        if not accepted:
            report.skip(reason)
            continue
        anatomy_class = str(proposal.get("class_name") or proposal.get("anatomy_class") or "")
        if anatomy_class not in ANATOMY_CLASSES:
            report.skip("unknown_anatomy_class")
            continue
        bbox, box_reason = _detector_bbox(proposal)
        if bbox is None:
            report.skip(box_reason)
            continue
        try:
            confidence = float(proposal.get("box_score", proposal.get("detector_confidence", 0.0)))
        except (TypeError, ValueError):
            report.skip("invalid_detector_confidence")
            continue
        if confidence < min_detector_confidence:
            report.skip("below_detector_confidence_floor")
            continue
        photo_id = str(proposal.get("photo_id") or "")
        observation_id = str(
            proposal.get("observation_id")
            or images_by_id.get(photo_id, {}).get("observation_id")
            or ""
        ).strip()
        if not observation_id:
            report.skip("missing_observation_id")
            continue
        taxon_name = str(
            proposal.get("taxon_name")
            or images_by_id.get(photo_id, {}).get("taxon_name")
            or lookup["taxon_by_observation"].get(observation_id)
            or ""
        ).strip()
        if not taxon_name:
            report.skip("missing_taxon_name")
            continue
        image_url = str(
            images_by_id.get(photo_id, {}).get("url")
            or lookup["url_by_photo"].get(photo_id)
            or ""
        ).strip()
        if not image_url:
            report.skip("missing_image_url")
            continue
        source_path = image_paths.get(photo_id)
        if not source_path or not Path(source_path).is_file():
            report.skip("source_image_missing")
            continue
        crop_id = crop_id_for(photo_id, bbox, anatomy_class)
        if crop_id in seen_crop_ids:
            report.skip("duplicate_crop")
            continue
        try:
            with Image.open(source_path) as handle:
                image = handle.convert("RGB")
        except Exception:
            report.skip("source_image_unreadable")
            continue
        width, height = image.size
        bbox_px = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        cx0, cy0, cx1, cy1 = pad_box_pixels(
            bbox_px,
            width,
            height,
            padding_frac=padding_frac,
            padding_mode=padding_mode,
            min_crop_px=min_crop_px,
        )
        if cx1 - cx0 < min_crop_px or cy1 - cy0 < min_crop_px:
            report.skip("crop_below_min_size")
            continue
        crop = image.crop((cx0, cy0, cx1, cy1))
        split_dir = output / "crops"
        split_dir.mkdir(parents=True, exist_ok=True)
        crop_path = split_dir / f"{crop_id}.jpg"
        crop.save(crop_path, quality=jpeg_quality, optimize=True)

        image_record = images_by_id.get(photo_id, {})
        observation_split_value = str(
            proposal.get("observation_split")
            or proposal.get("split")
            or observation_split(observation_id or photo_id, split_seed, split_fractions)
        )
        row = normalize_crop_row(
            {
                "schema": CROP_MANIFEST_SCHEMA,
                "contract": CONTRACT,
                "crop_id": crop_id,
                "proposal_id": str(proposal.get("proposal_id") or ""),
                "observation_id": observation_id,
                "taxon_name": taxon_name,
                "photo_id": photo_id,
                "image": image_url,
                "crop_path": str(crop_path),
                "anatomy_class": anatomy_class,
                "detector_confidence": confidence,
                "observation_split": observation_split_value,
                "split": observation_split_value,
                # Normalized detector box (this is the evidence box; padding
                # is recorded separately).
                "bbox": bbox,
                "crop_bbox_px": [cx0, cy0, cx1, cy1],
                "source_image_path": str(source_path),
                "source_width": width,
                "source_height": height,
                "padding_frac": padding_frac,
                "padding_mode": padding_mode,
                "review_verdict": _verdict(proposal, reviews) or "unreviewed_allowed",
                "detector_box_source": "detector",
            }
        )
        report.crops.append(row)
        report.written += 1
        seen_crop_ids.add(crop_id)
        report.by_class[anatomy_class] += 1
        report.by_split[observation_split_value] += 1

    manifest_path = output / "crop_manifest.jsonl"
    write_jsonl(manifest_path, report.crops)
    (output / "crop_build_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


def load_proposals(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
