"""Append-only plant-organ segmentation proposals and representative ranking.

The pipeline deliberately separates noticing from classification. A segmenter
proposes image geometry plus a shared visible-structure category. Source DeVo
feature/value assertions are attached only after a human or classifier reviews
the tagged image region.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence


SCHEMA_VERSION = "arq.notice-proposal-batch/v1"
DEFAULT_PROMPTS = (
    "leaf",
    "flower",
    "fruit",
    "stem",
    "bark",
    "bud",
    "branch",
    "whole plant",
)


def canonical_organ_label(label: str, prompts: Sequence[str] = DEFAULT_PROMPTS) -> str:
    """Map a detector phrase to one controlled organ category without guessing.

    Grounding models may return overlapping phrases such as ``fruit bud``.
    A proposal with exactly one matching controlled prompt is categorized; an
    overlapping phrase remains an explicit review item rather than becoming a
    silently incorrect organ label.
    """
    normalized = " ".join(label.lower().replace("_", " ").split())
    matches = [
        prompt for prompt in prompts
        if " ".join(prompt.lower().replace("_", " ").split()) in normalized
    ]
    if len(matches) == 1:
        return matches[0]
    return "ambiguous organ"


@dataclass(frozen=True)
class CaptureFrame:
    observation_id: str
    image_url: str
    subject_id: str
    species_name: str | None = None
    photo_id: str | None = None
    license: str | None = None
    captured_at: str | None = None
    frame_index: int = 0


@dataclass(frozen=True)
class RawSegment:
    label: str
    polygon: tuple[tuple[float, float], ...]
    detection_confidence: float
    mask_quality: float
    stability_score: float = 0.5
    prompt: str | None = None
    detector_label: str | None = None


@dataclass(frozen=True)
class ProposalQuality:
    representative_score: float
    detection_confidence: float
    mask_quality: float
    stability_score: float
    area_score: float
    border_score: float
    area_fraction: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class NoticeProposal:
    proposal_id: str
    observation_id: str
    subject_id: str
    image_url: str
    photo_id: str | None
    species_name: str | None
    license: str | None
    frame_index: int
    category_concept_id: str
    category_label: str
    polygon: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float]
    quality: ProposalQuality
    model_name: str
    model_version: str
    prompt: str
    detector_label: str | None = None
    review_state: str = "proposed"
    representative_rank: int | None = None
    selected_for_review: bool = False

    def to_app_annotation(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        ring = [[x, y] for x, y in self.polygon]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return {
            "annotation_id": self.proposal_id,
            "observation_id": self.observation_id,
            "image_url": self.image_url,
            "bbox_x0": x0,
            "bbox_y0": y0,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "polygon": ring,
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "notice_category_id": self.category_concept_id,
            "notice_category_label": self.category_label,
            "confidence": self.quality.representative_score,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "prompt": self.prompt,
                "detector_label": self.detector_label,
            },
            "quality": asdict(self.quality),
            "review_state": self.review_state,
            "representative_rank": self.representative_rank,
            "selected_for_review": self.selected_for_review,
            "frame_index": self.frame_index,
            "subject_id": self.subject_id,
        }


@dataclass(frozen=True)
class ProposalFailure:
    observation_id: str
    image_url: str
    model_name: str
    error_type: str
    message: str


@dataclass(frozen=True)
class BatchResult:
    source_name: str
    model_name: str
    model_version: str
    prompts: tuple[str, ...]
    selection_scope: str
    frames: tuple[CaptureFrame, ...]
    proposals: tuple[NoticeProposal, ...]
    failures: tuple[ProposalFailure, ...]
    generated_at: str

    def to_dict(self) -> dict:
        by_observation: dict[str, list[dict]] = {}
        for proposal in self.proposals:
            by_observation.setdefault(proposal.observation_id, []).append(
                proposal.to_app_annotation()
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "source": self.source_name,
            "model": {"name": self.model_name, "version": self.model_version},
            "prompts": list(self.prompts),
            "selection_scope": self.selection_scope,
            "summary": {
                "frames": len(self.frames),
                "proposals": len(self.proposals),
                "selected_for_review": sum(p.selected_for_review for p in self.proposals),
                "failed_frames": len(self.failures),
            },
            "failures": [asdict(failure) for failure in self.failures],
            "observations": [
                {
                    "observation_id": frame.observation_id,
                    "photo_id": frame.photo_id,
                    "image_url": frame.image_url,
                    "subject_id": frame.subject_id,
                    "species_name": frame.species_name,
                    "license": frame.license,
                    "frame_index": frame.frame_index,
                    "annotations": by_observation.get(frame.observation_id, []),
                }
                for frame in self.frames
            ],
        }

    def write(self, path: str | Path) -> None:
        output = Path(path)
        if output.exists():
            raise FileExistsError(f"Segmentation batch already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


class Segmenter(Protocol):
    model_name: str
    model_version: str

    def segment(self, frame: CaptureFrame, prompts: Sequence[str]) -> Sequence[RawSegment]: ...


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _category_id(label: str) -> str:
    normalized = label.lower().strip().replace(" ", "_")
    aliases = {"whole_plant": "plant", "branch": "stem"}
    return f"category:{aliases.get(normalized, normalized)}"


def _bbox(polygon: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(polygon, (*polygon[1:], polygon[0]))
        )
    ) / 2


def _area_score(area: float, preferred_min: float = 0.02, preferred_max: float = 0.65) -> float:
    if area < preferred_min:
        return _clamp(area / preferred_min)
    if area > preferred_max:
        return _clamp((1 - area) / (1 - preferred_max))
    return 1.0


def _quality(segment: RawSegment) -> ProposalQuality:
    area = _polygon_area(segment.polygon)
    x0, y0, x1, y1 = _bbox(segment.polygon)
    border_distance = min(x0, y0, 1 - x1, 1 - y1)
    border_score = _clamp(border_distance / 0.04)
    flags: list[str] = []
    if area < 0.01:
        flags.append("small_region")
    if area > 0.75:
        flags.append("dominates_frame")
    if border_distance <= 0.005:
        flags.append("touches_frame_edge")
    if segment.detection_confidence < 0.3:
        flags.append("low_detection_confidence")
    if segment.mask_quality < 0.5:
        flags.append("low_mask_quality")
    area_score = _area_score(area)
    representative = (
        0.30 * _clamp(segment.detection_confidence)
        + 0.30 * _clamp(segment.mask_quality)
        + 0.15 * _clamp(segment.stability_score)
        + 0.15 * area_score
        + 0.10 * border_score
    )
    return ProposalQuality(
        representative_score=round(representative, 6),
        detection_confidence=_clamp(segment.detection_confidence),
        mask_quality=_clamp(segment.mask_quality),
        stability_score=_clamp(segment.stability_score),
        area_score=round(area_score, 6),
        border_score=round(border_score, 6),
        area_fraction=round(area, 6),
        flags=tuple(flags),
    )


def _proposal_id(frame: CaptureFrame, segment: RawSegment, index: int) -> str:
    payload = "|".join(
        [frame.observation_id, frame.image_url, segment.label, str(index), repr(segment.polygon)]
    )
    return "notice-proposal-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _valid_polygon(polygon: Sequence[tuple[float, float]]) -> bool:
    return len(polygon) >= 3 and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in polygon)


def _rank_top_k(
    proposals: Sequence[NoticeProposal], top_k: int, selection_scope: str
) -> tuple[NoticeProposal, ...]:
    groups: dict[tuple[str, ...], list[NoticeProposal]] = {}
    for proposal in proposals:
        if selection_scope == "category":
            key = (proposal.category_concept_id,)
        elif selection_scope == "subject-category":
            key = (proposal.subject_id, proposal.category_concept_id)
        else:
            raise ValueError("selection_scope must be 'category' or 'subject-category'")
        groups.setdefault(key, []).append(proposal)
    ranked: list[NoticeProposal] = []
    for key in sorted(groups):
        group = sorted(
            groups[key],
            key=lambda item: (-item.quality.representative_score, item.proposal_id),
        )
        ranked.extend(
            replace(item, representative_rank=rank, selected_for_review=rank <= top_k)
            for rank, item in enumerate(group, start=1)
        )
    return tuple(sorted(ranked, key=lambda item: (item.observation_id, item.proposal_id)))


def build_notice_batch(
    frames: Sequence[CaptureFrame],
    segmenter: Segmenter,
    *,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    top_k: int = 5,
    selection_scope: str = "category",
    source_name: str = "fieldguidevision",
) -> BatchResult:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if selection_scope not in {"category", "subject-category"}:
        raise ValueError("selection_scope must be 'category' or 'subject-category'")
    proposals: list[NoticeProposal] = []
    failures: list[ProposalFailure] = []
    for frame in frames:
        try:
            raw_segments = segmenter.segment(frame, prompts)
        except Exception as exc:
            failures.append(
                ProposalFailure(
                    observation_id=frame.observation_id,
                    image_url=frame.image_url,
                    model_name=segmenter.model_name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue
        for index, segment in enumerate(raw_segments):
            if not _valid_polygon(segment.polygon):
                continue
            quality = _quality(segment)
            proposals.append(
                NoticeProposal(
                    proposal_id=_proposal_id(frame, segment, index),
                    observation_id=frame.observation_id,
                    subject_id=frame.subject_id,
                    image_url=frame.image_url,
                    photo_id=frame.photo_id,
                    species_name=frame.species_name,
                    license=frame.license,
                    frame_index=frame.frame_index,
                    category_concept_id=_category_id(segment.label),
                    category_label=segment.label,
                    polygon=segment.polygon,
                    bbox=_bbox(segment.polygon),
                    quality=quality,
                    model_name=segmenter.model_name,
                    model_version=segmenter.model_version,
                    prompt=segment.prompt or segment.label,
                    detector_label=segment.detector_label,
                )
            )
    return BatchResult(
        source_name=source_name,
        model_name=segmenter.model_name,
        model_version=segmenter.model_version,
        prompts=tuple(prompts),
        selection_scope=selection_scope,
        frames=tuple(frames),
        proposals=_rank_top_k(proposals, top_k, selection_scope),
        failures=tuple(failures),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def load_inaturalist_frames(
    path: str | Path,
    *,
    limit: int = 100,
) -> tuple[CaptureFrame, ...]:
    """Load deterministic image frames from the JM-replacement Pound Ridge CSV."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    frames: list[CaptureFrame] = []
    with Path(path).open(newline="") as source:
        for row in csv.DictReader(source):
            image_url = (row.get("image_url") or "").strip()
            observation_id = (row.get("id") or row.get("observation_id") or "").strip()
            if not image_url or not observation_id:
                continue
            species = (row.get("scientific_name") or row.get("species_guess") or "").strip() or None
            frames.append(
                CaptureFrame(
                    observation_id=observation_id,
                    photo_id=(row.get("photo_id") or "").strip() or None,
                    image_url=image_url,
                    subject_id=species or observation_id,
                    species_name=species,
                    license=(row.get("license") or "").strip() or None,
                    captured_at=(row.get("observed_on") or "").strip() or None,
                    frame_index=0,
                )
            )
            if len(frames) >= limit:
                break
    return tuple(frames)
