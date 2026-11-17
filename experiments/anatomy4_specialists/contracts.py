# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Shared contracts for the anatomy4 conditioned experts.

Pure data/functions only: importing this module must not import torch, PIL, or
open a database connection.

Two expert families are defined here:

* **leaf expert**  — ``leaf_simple_vs_compound`` (required), plus the
  config-driven ``leaf_margin`` and ``leaf_arrangement`` targets.
* **flower expert** — ``flower_inflorescence_type`` (required).

Every target declares the anatomy classes whose crops may supply image
evidence. A claim from a different anatomy is never eligible supervision.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CONTRACT = "anatomy4-specialists/v1"
CROP_MANIFEST_SCHEMA = "arq.anatomy4_specialists.crop_manifest/v1"
SPLITS: tuple[str, ...] = ("train", "val", "test")

# The anatomy4 detector contract. These are the only crop classes allowed in a
# crop manifest; a target may accept one or more of them.
ANATOMY_CLASSES: tuple[str, ...] = ("leaf", "needle", "flower", "inflorescence")
ANATOMY_IDS: dict[str, int] = {name: idx for idx, name in enumerate(ANATOMY_CLASSES)}

# Canonical organ family used by Dirr weak labels. Dirr ``leaf_*`` features are
# only meaningful on leaf anatomy; ``inflorescences`` requires an inflorescence
# crop. An isolated single-flower crop does NOT carry arrangement evidence.
FEATURE_ANATOMY: dict[str, frozenset[str]] = {
    "leaf_type": frozenset({"leaf"}),
    "leaf_shape_margins": frozenset({"leaf"}),
    "leaf_arrangement": frozenset({"leaf"}),
    "leaf_type_conifer": frozenset({"needle"}),
    "inflorescences": frozenset({"inflorescence"}),
}

_ALIASES: dict[str, str] = {
    # leaf simple/compound
    "simple": "Simple",
    "simple leaf": "Simple",
    "simple_leaf": "Simple",
    "compound": "Compound",
    "compound leaf": "Compound",
    "compound_leaf": "Compound",
    # leaf margin
    "entire": "Entire",
    "serrate": "Serrate",
    "serrated": "Serrate",
    "lobed": "Lobed",
    # leaf arrangement
    "alternate": "Alternate",
    "opposite": "Opposite",
    # inflorescence
    "raceme": "Raceme",
    "corymb": "Corymb",
    "cyme": "Cyme",
    "panicle": "Panicle",
}

_CITATION_RE = re.compile(r"\s+\(?[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)*\)?.*$")


def canonical_binomial(taxon_name: str | None) -> str | None:
    """Return a lower-case ``genus species`` key.

    Dirr rows are species-level. We deliberately do not fuzzy-match: author
    citations are stripped by taking the first two tokens, but a trinomial
    keeps only the binomial (subspecies share the species' claims) and a
    non-species rank has no exact species key beyond its first two tokens.
    """

    if not taxon_name:
        return None
    cleaned = _CITATION_RE.sub("", str(taxon_name).strip())
    # Hybrid markers are written both attached ("×Chitalpa", "xgrandiflora")
    # and detached ("Abelia × grandiflora"); drop detached markers.
    tokens = [
        token
        for token in re.split(r"\s+", cleaned)
        if token and token not in {"×", "x"}
    ]
    if len(tokens) < 2:
        return None
    genus = tokens[0].lower().lstrip("×").strip()
    species = tokens[1].lower().lstrip("×").strip()
    if not genus or not species or species in {"sp.", "sp", "spp.", "spp", "unknown"}:
        return None
    return f"{genus} {species}"


def canonical_binomial_aliases(taxon_name: str | None) -> tuple[str, ...]:
    """Return match keys for a taxon, including the attached ``x`` hybrid form.

    ``Forsythia xintermedia`` and ``Forsythia × intermedia`` must collide. The
    primary key keeps a genuine x-starting epithet (``Salix xanthicola``) so
    the alias is only used as a lookup fallback against actual claims.
    """

    primary = canonical_binomial(taxon_name)
    if not primary:
        return ()
    genus, _, species = primary.partition(" ")
    aliases = [primary]
    if species.startswith("x") and len(species) > 1:
        aliases.append(f"{genus} {species[1:]}")
    return tuple(dict.fromkeys(aliases))


def normalize_value(value: str | None) -> str:
    """Normalize a claim/class value for comparison (keeps the label space open)."""

    if value is None:
        return ""
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def canonical_label(value: str | None, target: "TargetSpec") -> str | None:
    """Map a raw claim/label value onto one of the target's classes.

    Returns ``None`` when the value is outside this target. Unknown values are
    never silently coerced to a nearby class.
    """

    normalized = normalize_value(value)
    if not normalized:
        return None
    by_alias = _ALIASES.get(normalized)
    candidates = {normalize_value(name) for name in target.classes}
    if by_alias and normalize_value(by_alias) in candidates:
        return by_alias
    # Accept the class spelling itself (already-normalized comparison).
    for name in target.classes:
        if normalize_value(name) == normalized:
            return name
    return None


@dataclass(frozen=True)
class TargetSpec:
    """One conditioned classification target."""

    target_id: str
    feature_name: str
    classes: tuple[str, ...]
    anatomy_classes: tuple[str, ...]
    expert: str
    description: str = ""
    dirr_feature_names: tuple[str, ...] = ()
    allow_multi_value: bool = False

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        if len(set(self.classes)) != len(self.classes) or len(self.classes) < 2:
            raise ValueError(f"{self.target_id}: classes must be >=2 and unique")
        unknown = [a for a in self.anatomy_classes if a not in ANATOMY_CLASSES]
        if unknown:
            raise ValueError(f"{self.target_id}: unknown anatomy classes {unknown}")
        if not self.anatomy_classes:
            raise ValueError(f"{self.target_id}: at least one anatomy class is required")

    @property
    def label_to_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.classes)}

    @property
    def index_to_label(self) -> dict[int, str]:
        return {idx: name for idx, name in enumerate(self.classes)}

    @property
    def claim_features(self) -> tuple[str, ...]:
        return self.dirr_feature_names or (self.feature_name,)

    def anatomy_allows(self, anatomy_class: str) -> bool:
        return anatomy_class in self.anatomy_classes

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "feature_name": self.feature_name,
            "dirr_feature_names": list(self.claim_features),
            "classes": list(self.classes),
            "anatomy_classes": list(self.anatomy_classes),
            "expert": self.expert,
            "description": self.description,
            "allow_multi_value": self.allow_multi_value,
        }


# Required pair + the optional, config-driven leaf targets. These are the
# *defaults*; a config file is authoritative for a run.
LEAF_SIMPLE_VS_COMPOUND = TargetSpec(
    target_id="leaf_simple_vs_compound",
    feature_name="leaf_type",
    classes=("Simple", "Compound"),
    anatomy_classes=("leaf",),
    expert="leaf",
    description="Leaf blade simple vs compound (Dirr leaf_type).",
)

FLOWER_INFLORESCENCE_TYPE = TargetSpec(
    target_id="flower_inflorescence_type",
    feature_name="inflorescences",
    classes=("Raceme", "Corymb", "Cyme", "Panicle"),
    anatomy_classes=("inflorescence",),
    expert="flower",
    description="Flower / inflorescence arrangement type (Dirr inflorescences).",
)

LEAF_MARGIN = TargetSpec(
    target_id="leaf_margin",
    feature_name="leaf_shape_margins",
    classes=("Entire", "Serrate", "Lobed"),
    anatomy_classes=("leaf",),
    expert="leaf",
    description="Leaf margin: entire, serrate, or lobed (Dirr leaf_shape_margins).",
)

LEAF_ARRANGEMENT = TargetSpec(
    target_id="leaf_arrangement",
    feature_name="leaf_arrangement",
    classes=("Alternate", "Opposite"),
    anatomy_classes=("leaf",),
    expert="leaf",
    description="Leaf arrangement: alternate or opposite (Dirr leaf_arrangement).",
)

DEFAULT_TARGETS: dict[str, TargetSpec] = {
    target.target_id: target
    for target in (
        LEAF_SIMPLE_VS_COMPOUND,
        FLOWER_INFLORESCENCE_TYPE,
        LEAF_MARGIN,
        LEAF_ARRANGEMENT,
    )
}


def load_target_spec(path: str | Path) -> TargetSpec:
    """Load the target half of a run config (``target`` block or a bare spec)."""

    data = json.loads(Path(path).read_text())
    block = data.get("target", data)
    if "target_id" not in block:
        raise ValueError(f"{path}: missing target.target_id")
    anatomy = block.get("anatomy_classes")
    if anatomy is None:
        anatomy = sorted(FEATURE_ANATOMY.get(str(block["feature_name"]), frozenset()))
    return TargetSpec(
        target_id=str(block["target_id"]),
        feature_name=str(block["feature_name"]),
        classes=tuple(str(v) for v in block["classes"]),
        anatomy_classes=tuple(str(v) for v in anatomy),
        expert=str(block.get("expert", "leaf")),
        description=str(block.get("description", "")),
        dirr_feature_names=tuple(str(v) for v in block.get("dirr_feature_names", ())),
        allow_multi_value=bool(block.get("allow_multi_value", False)),
    )


def load_run_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if "target" not in data:
        raise ValueError(f"{path}: run config must include a 'target' block")
    spec = load_target_spec(path)
    return {"raw": data, "target": spec}


# ---------------------------------------------------------------------------
# Crop manifest contract
# ---------------------------------------------------------------------------
REQUIRED_CROP_FIELDS: tuple[str, ...] = (
    "observation_id",
    "taxon_name",
    "image",
    "crop_path",
    "anatomy_class",
    "detector_confidence",
    "observation_split",
)

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "image": ("image", "photo_url", "source_url"),
    "crop_path": ("crop_path", "crop", "path"),
    "anatomy_class": ("anatomy_class", "detector_class", "class_name"),
    "detector_confidence": ("detector_confidence", "confidence", "box_score", "detector_score"),
    "observation_split": ("observation_split", "split"),
    "photo_id": ("photo_id", "image_id"),
    "taxon_name": ("taxon_name", "species", "scientific_name"),
}


def _first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def normalize_crop_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one manifest row without validating semantic correctness."""

    out = dict(row)
    for canonical, aliases in _FIELD_ALIASES.items():
        value = _first_present(row, aliases)
        if value is not None:
            out[canonical] = value
    if out.get("detector_confidence") is not None:
        try:
            out["detector_confidence"] = float(out["detector_confidence"])
        except (TypeError, ValueError):
            pass
    if isinstance(out.get("bbox"), list):
        out["bbox"] = [float(v) for v in out["bbox"]]
    out.setdefault("schema", CROP_MANIFEST_SCHEMA)
    return out


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
