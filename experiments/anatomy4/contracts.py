# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Shared contracts for the anatomy4 bootstrap.

Pure functions only: no model, DB, or filesystem dependency beyond reading a
JSON config. Everything here is unit-tested.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CONTRACT = "anatomy4-v1"

# The four approved classes. Order is the YOLO class-id order.
CLASS_NAMES: tuple[str, ...] = ("leaf", "needle", "flower", "inflorescence")
CLASS_IDS: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Prompt variants that map to each class. Detector output is mapped by
# containment of these phrases (normalized), never by taxonomy.
PROMPT_TO_CLASS: dict[str, str] = {
    "leaf": "leaf",
    "leaves": "leaf",
    "leaf blade": "leaf",
    "leaf blades": "leaf",
    "green leaf": "leaf",
    "green leaves": "leaf",
    "needle": "needle",
    "needles": "needle",
    "conifer needle": "needle",
    "conifer needles": "needle",
    "pine needle": "needle",
    "pine needles": "needle",
    "evergreen needle": "needle",
    "evergreen needles": "needle",
    "needle cluster": "needle",
    "needle clusters": "needle",
    "flower": "flower",
    "flowers": "flower",
    "blossom": "flower",
    "blossoms": "flower",
    "single flower": "flower",
    "single flowers": "flower",
    "flower blossom": "flower",
    "flower blossoms": "flower",
    "inflorescence": "inflorescence",
    "inflorescences": "inflorescence",
    "flower cluster": "inflorescence",
    "flower clusters": "inflorescence",
    "flower head": "inflorescence",
    "flower heads": "inflorescence",
    "cluster of flowers": "inflorescence",
    "clusters of flowers": "inflorescence",
    "flower spike": "inflorescence",
    "flower spikes": "inflorescence",
}

_PHOTO_ID_RE = re.compile(r"/photos/(\d+)/")


def normalize_phrase(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())


def map_detector_label(label: str, class_prompts: Sequence[str]) -> tuple[str | None, str]:
    """Map one Grounding-DINO label to an anatomy4 class.

    Returns ``(class_name | None, reason)``. ``None`` means the label is not a
    valid anatomy4 class and the proposal must not be exported. The mapping is
    deliberately conservative: exactly one class may match for a label.
    """

    normalized = normalize_phrase(label)
    # Longest-phrase wins: "flower cluster" maps to inflorescence even though
    # it also contains "flower". Only equal-specificity collisions are
    # ambiguous.
    matches = [
        (len(prompt), PROMPT_TO_CLASS[prompt])
        for prompt in PROMPT_TO_CLASS
        if prompt in normalized
    ]
    allowed = {
        PROMPT_TO_CLASS[prompt]
        for prompt in PROMPT_TO_CLASS
        if any(normalize_phrase(p) == prompt for p in class_prompts)
    }
    if not matches:
        return None, "unmapped"
    longest = max(length for length, _ in matches)
    best = {name for length, name in matches if length == longest}
    if len(best) > 1:
        return None, f"ambiguous:{','.join(sorted(best))}"
    only = best.pop()
    if only in allowed:
        return only, "mapped"
    return None, "unmapped"


def clip_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Clip a normalized xyxy bbox to [0, 1] and enforce ordering."""

    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return (
        min(max(x0, 0.0), 1.0),
        min(max(y0, 0.0), 1.0),
        min(max(x1, 0.0), 1.0),
        min(max(y1, 0.0), 1.0),
    )


def xyxy_to_yolo(
    bbox: Sequence[float], width: float, height: float
) -> tuple[float, float, float, float]:
    """Convert an absolute-pixel xyxy box to YOLO normalized cx cy w h."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    w = (x1 - x0) / width
    h = (y1 - y0) / height
    cx, cy, w, h = (min(max(v, 0.0), 1.0) for v in (cx, cy, w, h))
    return cx, cy, w, h


def polygon_to_bbox(
    polygon: Iterable[Sequence[float]],
) -> tuple[float, float, float, float]:
    """Tight normalized bbox around a polygon; raises on empty input."""

    points = [(float(x), float(y)) for x, y in polygon]
    if not points:
        raise ValueError("cannot compute a bbox for an empty polygon")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return clip_bbox((min(xs), min(ys), max(xs), max(ys)))


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    points = [(float(x), float(y)) for x, y in polygon]
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0])):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def photo_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = _PHOTO_ID_RE.search(url)
    return match.group(1) if match else None


def stable_unit_interval(key: str, seed: str = "") -> float:
    """Deterministic [0, 1) value used for sampling and splits."""

    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def split_for_observation(observation_id: str, seed: str = CONTRACT) -> str:
    """Deterministic observation-level split. All images of an obs agree."""

    value = stable_unit_interval(observation_id, seed)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "val"
    return "test"


@dataclass(frozen=True)
class ClassSpec:
    id: int
    name: str
    prompts: tuple[str, ...]
    box_threshold: float
    text_threshold: float


@dataclass(frozen=True)
class AnatomyConfig:
    contract: str
    classes: tuple[ClassSpec, ...]
    detector: str
    segmenter: str
    max_image_side: int
    nms_iou: float
    polygon_epsilon: float
    multimask_output: bool
    min_polygon_points: int
    empty_cache_each_image: bool = True
    max_box_area: float = 0.92
    min_box_area: float = 0.0004

    @property
    def class_by_name(self) -> dict[str, ClassSpec]:
        return {c.name: c for c in self.classes}

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.classes)

    def prompt_class_table(self) -> dict[str, str]:
        table: dict[str, str] = {}
        for spec in self.classes:
            for prompt in spec.prompts:
                table[normalize_phrase(prompt)] = spec.name
        return table

    def fingerprint(self) -> str:
        payload = {
            "contract": self.contract,
            "classes": [
                {
                    "id": c.id,
                    "name": c.name,
                    "prompts": list(c.prompts),
                    "box_threshold": c.box_threshold,
                    "text_threshold": c.text_threshold,
                }
                for c in self.classes
            ],
            "detector": self.detector,
            "segmenter": self.segmenter,
            "max_image_side": self.max_image_side,
            "nms_iou": self.nms_iou,
            "polygon_epsilon": self.polygon_epsilon,
            "max_box_area": self.max_box_area,
            "min_box_area": self.min_box_area,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]


def load_config(path: str | Path) -> AnatomyConfig:
    data: dict[str, Any] = json.loads(Path(path).read_text())
    classes = tuple(
        ClassSpec(
            id=int(item["id"]),
            name=str(item["name"]),
            prompts=tuple(str(p) for p in item["prompts"]),
            box_threshold=float(item["box_threshold"]),
            text_threshold=float(item["text_threshold"]),
        )
        for item in data["classes"]
    )
    if tuple(c.name for c in classes) != CLASS_NAMES:
        raise ValueError(
            f"config classes must be exactly {CLASS_NAMES}, got "
            f"{tuple(c.name for c in classes)}"
        )
    if [c.id for c in classes] != [CLASS_IDS[c.name] for c in classes]:
        raise ValueError("config class ids do not match the anatomy4 order")
    runtime = data.get("runtime", {})
    models = data.get("models", {})
    return AnatomyConfig(
        contract=str(data.get("contract", CONTRACT)),
        classes=classes,
        detector=str(models["detector"]),
        segmenter=str(models["segmenter"]),
        max_image_side=int(runtime.get("max_image_side", 1536)),
        nms_iou=float(runtime.get("nms_iou", 0.5)),
        polygon_epsilon=float(runtime.get("polygon_epsilon", 0.003)),
        multimask_output=bool(runtime.get("multimask_output", False)),
        min_polygon_points=int(runtime.get("min_polygon_points", 3)),
        empty_cache_each_image=bool(runtime.get("empty_cache_each_image", True)),
        max_box_area=float(runtime.get("max_box_area", 0.92)),
        min_box_area=float(runtime.get("min_box_area", 0.0004)),
    )
