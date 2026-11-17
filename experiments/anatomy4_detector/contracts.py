# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""anatomy4-v1 contracts: classes, normalized-xyxy labels, geometry, splits.

Everything here is pure Python and unit-tested. No model, DB, or network
dependency. The reviewed box artifact is **normalized xyxy** (x0 y0 x1 y1 in
[0, 1]); Ultralytics' ``cx cy w h`` form exists only in the generated
training view (``dataset.prepare_ultralytics``).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

CONTRACT = "anatomy4-v1"

# YOLO class-id order. This order is the API contract: it is embedded in
# data.yaml, the ONNX metadata, and every metric report.
CLASS_NAMES: tuple[str, ...] = ("leaf", "needle", "flower", "inflorescence")
CLASS_IDS: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}

SPLITS: tuple[str, ...] = ("train", "val", "test")

# A frozen benchmark is test-only by construction. The marker file is written
# next to dataset.json; readers must refuse to treat it as training material.
TEST_ONLY_MARKER = "FROZEN_TEST_ONLY.json"

IMAGE_SUFFIXES: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
)


class ContractError(ValueError):
    """A label, box, or split that violates anatomy4-v1."""


@dataclass(frozen=True)
class Box:
    """Normalized xyxy box."""

    x0: float
    y0: float
    x1: float
    y1: float
    class_id: int

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class Detection:
    """A predicted box with a confidence; normalized xyxy."""

    x0: float
    y0: float
    x1: float
    y1: float
    class_id: int
    confidence: float

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


def parse_label_line(line: str, *, source: str = "<label>") -> Box | None:
    """Parse one normalized-xyxy label line.

    Format: ``class_id x0 y0 x1 y1`` with 5 whitespace-separated values and
    normalized coordinates. Blank lines return ``None`` (negatives are empty
    files, but a stray blank line is not a box).
    """

    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if len(parts) != 5:
        raise ContractError(
            f"{source}: expected 5 fields 'class_id x0 y0 x1 y1', got {len(parts)}"
        )
    try:
        class_raw = parts[0]
        class_id = int(class_raw)
        coords = tuple(float(value) for value in parts[1:])
    except ValueError as exc:
        raise ContractError(f"{source}: non-numeric label field ({exc})") from exc
    if not (0 <= class_id < len(CLASS_NAMES)):
        raise ContractError(
            f"{source}: class id {class_id} outside 0..{len(CLASS_NAMES) - 1}"
        )
    if not all(math.isfinite(value) for value in coords):
        raise ContractError(f"{source}: non-finite coordinate in {coords}")
    x0, y0, x1, y1 = coords
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ContractError(
            f"{source}: box is not normalized xyxy with x0<x1, y0<y1 in [0,1]: "
            f"{coords}"
        )
    return Box(x0=x0, y0=y0, x1=x1, y1=y1, class_id=class_id)


def parse_label_text(text: str, *, source: str = "<label>") -> list[Box]:
    boxes: list[Box] = []
    for number, line in enumerate(text.splitlines(), start=1):
        box = parse_label_line(line, source=f"{source}:{number}")
        if box is not None:
            boxes.append(box)
    return boxes


def format_label_line(box: Box) -> str:
    return (
        f"{box.class_id} {box.x0:.6f} {box.y0:.6f} "
        f"{box.x1:.6f} {box.y1:.6f}"
    )


def xyxy_to_cxcywh(box: Sequence[float]) -> tuple[float, float, float, float]:
    """Normalized xyxy -> normalized YOLO cx cy w h."""

    x0, y0, x1, y1 = (float(value) for value in box)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)


def cxcywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    cx, cy, width, height = (float(value) for value in box)
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def xyxy_norm_to_px(
    box: Sequence[float], width: float, height: float
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in box)
    return (x0 * width, y0 * height, x1 * width, y1 * height)


def xyxy_px_to_norm(
    box: Sequence[float], width: float, height: float
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ContractError("image dimensions must be positive")
    x0, y0, x1, y1 = (float(value) for value in box)
    return (
        min(max(x0 / width, 0.0), 1.0),
        min(max(y0 / height, 0.0), 1.0),
        min(max(x1 / width, 0.0), 1.0),
        min(max(y1 / height, 0.0), 1.0),
    )


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU for two boxes in the same (pixel or normalized) coordinate space."""

    ax0, ay0, ax1, ay1 = (float(value) for value in a)
    bx0, by0, bx1, by1 = (float(value) for value in b)
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def stable_unit_interval(key: str, seed: str = CONTRACT) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def split_for_observation(observation_id: str, seed: str = CONTRACT) -> str:
    """Deterministic observation-level split (70/15/15).

    All images of one observation must land in the same split. The validator
    enforces that against whatever splits were materialized; this helper is
    for producers that want the canonical assignment.
    """

    value = stable_unit_interval(observation_id, seed)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "val"
    return "test"


def observation_leaks(
    split_observations: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    """Observation ids that appear in more than one split."""

    owners: dict[str, list[str]] = {}
    for split, observations in split_observations.items():
        for observation in observations:
            owners.setdefault(str(observation), []).append(split)
    leaks: dict[str, list[str]] = {}
    for observation, splits in owners.items():
        distinct = sorted(set(splits))
        if len(distinct) > 1:
            leaks[observation] = distinct
    return leaks
