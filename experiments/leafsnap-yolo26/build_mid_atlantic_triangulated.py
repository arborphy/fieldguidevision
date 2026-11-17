#!/usr/bin/env python3
# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Mid-Atlantic Triangulated Dataset & Staged Encounter Index.

Cross-references the Henrico County (inst-osm-relation-3070728) and Tuckaway
(inst-mtuhbm9y + local walk photos) plant pools with the Dirr / GoBotany DeVo
feature priors, then runs the automatic triangulation & view-framing engine
over (a) the LeafSnap classification crops and (b) the regional pool photos.

Outputs
-------
scratch/leafsnap-yolo26/classification-v3-triangulated/
    train|val|test/{simple_leaf,compound_leaf}/...   gold view-valid crops
    hard_negatives_sub_part/{train,val,test}/...      isolated-leaflet crops
    manifest.jsonl, triangulation.jsonl, labels.json, counts.json
scratch/leafsnap-yolo26/mid-atlantic-pool/
    images/<species>/<uid>.<ext>                      regional cohort
    pool_manifest.jsonl, bench_triangulated_manifest.jsonl,
    triangulation_summary.json
experiments/leafsnap-yolo26/MID_ATLANTIC_TRIANGULATION.md

Usage
-----
    .venv/bin/python experiments/leafsnap-yolo26/build_mid_atlantic_triangulated.py
    ... --offline              # reuse the cached pool, no network
    ... --max-photos 180       # cap the regional cohort
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = ROOT / "scratch" / "leafsnap-yolo26"

CLASSIFICATION = SCRATCH / "classification"
CLASSIFICATION_MANIFEST = CLASSIFICATION / "manifest.jsonl"

OUT_DATASET = SCRATCH / "classification-v3-triangulated"
OUT_POOL = SCRATCH / "mid-atlantic-pool"
POOL_IMAGES = OUT_POOL / "images"

FEATURE_ASSERTIONS = HERE / "feature_assertions.json"
GOBOTANY_PARQUET = ROOT / "gobotany" / "extracted_vocabulary" / "parquet"

TUCKAWAY_PHYS = Path("/Users/valkyrie/Desktop/arborphy product/phys/tuckaway")
TUCKAWAY_QA = Path("/Users/valkyrie/dev/arborphy/fieldguidevision/scratch/tuckaway-qa-20260908")

DB_CONTAINER = "arq-compose-postgres"
DB_NAME = "arq"
DB_USER = "arq"
API_BASE = os.environ.get("ARQ_API_BASE", "http://localhost:8000").rstrip("/")

HENRICO_ID = "inst-osm-relation-3070728"
TUCKAWAY_ID = "inst-mtuhbm9y"

# Curated Mid-Atlantic cohort: the top shared simple & compound woody species
# named by the director, plus the Carya species actually recorded at Henrico.
TARGET_SPECIES: tuple[str, ...] = (
    "Acer rubrum",
    "Liquidambar styraciflua",
    "Quercus alba",
    "Quercus phellos",
    "Quercus rubra",
    "Ilex opaca",
    "Prunus serotina",
    "Toxicodendron radicans",
    "Parthenocissus quinquefolia",
    "Juglans nigra",
    "Fraxinus americana",
    "Robinia pseudoacacia",
    "Albizia julibrissin",
    "Liriodendron tulipifera",
    "Nyssa sylvatica",
    "Viburnum dentatum",
    "Alnus serrulata",
    "Lagerstroemia indica",
    "Pinus taeda",
)
CARYA_SPECIES: tuple[str, ...] = (
    "Carya glabra",
    "Carya tomentosa",
    "Carya ovata",
    "Carya cordiformis",
    "Carya illinoinensis",
    "Carya pallida",
)
GENUS_GROUPS: dict[str, str] = {"Carya": "Carya sp."}

# Hand-curated priors only for taxa absent from BOTH Dirr and GoBotany.
# Kept deliberately small and labelled in provenance.
CURATED_REGIONAL_PRIORS: dict[str, dict[str, Any]] = {
    "Lagerstroemia indica": {
        "leaf_type": "simple",
        "leaf_arrangement": "subopposite",
        "leaf_margins": ["entire"],
        "leaf_lobing": "unlobed",
        "bark_texture": ["exfoliating"],
        "leaf_shape": ["elliptic", "obovate"],
        "source": "curated_regional_flora",
        "note": "Crape myrtle: simple, subopposite/opposite, entire, exfoliating bark.",
    }
}

# ---------------------------------------------------------------------------
# Triangulation thresholds (documented in MID_ATLANTIC_TRIANGULATION.md)
# ---------------------------------------------------------------------------
MIN_COMPONENT_FRAC = 0.005      # sub-contours smaller than 0.5% of fg are noise
DEFECT_DEPTH_FRAC = 0.03        # convexity-defect depth must exceed 3% of max dim
SUBPART_SOLIDITY = 0.78         # > this + single blade => isolated leaflet
SUBPART_MAX_DEFECTS = 1         # "low convexity-defect count"
SUBPART_MIN_ELLIPSE_FILL = 0.40 # reject degenerate slivers, keep serrate leaflets
SIMPLE_MIN_LARGEST_FRAC = 0.55  # a clean simple-leaf crop is one dominant blade
SIMPLE_MIN_FG_FRAC = 0.02       # not a near-empty crop
POOL_MASK_MIN_GREEN = 0.06      # estimated-mask baseline for field photos
POOL_MIN_DOMINANT_FRAC = 0.45   # trust geometry only when one subject dominates

STAGE_LABELS = ("stage_1_anchor", "stage_2_anatomy", "stage_3_area_context")
STAGE_ORGAN_HINTS = {
    "stage_1_anchor": "trunk / bark / root",
    "stage_2_anatomy": "leaf / twig / flower / fruit",
    "stage_3_area_context": "whole plant / habitat",
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def json_line(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=_json_default) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json_line(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return out or "unknown"


def stable_id(*parts: Any, length: int = 16) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:length]


def norm_name(value: str) -> str:
    value = value.lower().replace("-", " ").replace("_", " ")
    value = re.sub(r"\([^)]*\)", " ", value)  # author citations
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def name_variants(value: str) -> set[str]:
    base = norm_name(value)
    variants = {base, base.replace(" ", "")}
    tokens = base.split()
    if tokens:
        variants.add(" ".join(tokens[:2]))
        variants.add("".join(tokens[:2]))
    return variants


def psql(sql: str, timeout: int = 180) -> list[list[str]]:
    cmd = [
        "docker", "exec", DB_CONTAINER, "psql",
        "-U", DB_USER, "-d", DB_NAME, "-t", "-A", "-F", "\t", "-c", sql,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed: {proc.stderr.strip()[:500]}")
    return [line.split("\t") for line in proc.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Leaf priors: Dirr -> GoBotany -> curated fallback
# ---------------------------------------------------------------------------
ARRANGEMENT_MAP = {
    "1 leaf per node": "alternate",
    "2 leaves per node": "opposite",
    "3 leaves per node": "whorled",
    "4 leaves per node": "whorled",
    "alternate": "alternate",
    "opposite": "opposite",
    "whorled": "whorled",
    "subopposite": "subopposite",
    "sub-opposite": "subopposite",
}
MARGIN_MAP = {
    "entire": "entire",
    "lobed": "lobed",
    "serrate": "toothed",
    "dentate": "toothed",
    "doubly-serrate": "toothed",
    "double toothed": "toothed",
    "crenate": "toothed",
    "crenate or crenulate": "toothed",
    "serrate or dentate": "toothed",
    "serrulate or denticulate": "toothed",
    "sinuate": "toothed",
    "undulate": "toothed",
    "toothed": "toothed",
}
BARK_MAP = {
    "exfoliating": "exfoliating",
    "ridged or plated": "ridged",
    "ridged": "ridged",
    "plated": "ridged",
    "close and smooth": "smooth",
    "smooth": "smooth",
}


def canonical_margins(values: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if not values:
        return None
    out = {MARGIN_MAP.get(str(v).strip().lower(), str(v).strip().lower()) for v in values}
    # "lobed + toothed" is common; keep both, they are independent diagnostics.
    return sorted(v for v in out if v)


def canonical_bark(values: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if not values:
        return None
    out = {BARK_MAP.get(str(v).strip().lower(), str(v).strip().lower()) for v in values}
    return sorted(v for v in out if v and v != "na")


def profile_from_dirr(row: dict[str, Any]) -> dict[str, Any]:
    feats = row.get("features") or {}
    leaf_type = (feats.get("leaf_type") or [None])[0]
    leaf_type = str(leaf_type).strip().lower() if leaf_type else None
    conifer = feats.get("leaf_type_conifer") or []
    if leaf_type is None and conifer:
        # Dirr's conifer vocabulary is a separate axis; map to `needle` so the
        # simple/compound triangulation correctly marks it not-assessable.
        leaf_type = "needle"
    arr = (feats.get("leaf_arrangement") or [None])[0]
    arr = ARRANGEMENT_MAP.get(str(arr).strip().lower()) if arr else None
    margins = canonical_margins(feats.get("leaf_shape_margins"))
    lobe_type = feats.get("leaf_shape_lobe_type") or []
    if lobe_type:
        lobing = "lobed"
    elif margins and "lobed" in margins:
        lobing = "lobed"
    elif margins:
        lobing = "unlobed"
    else:
        lobing = None
    shapes = [str(v).strip().lower() for v in (feats.get("leaf_shape_overall") or [])]
    return {
        "leaf_type": leaf_type if leaf_type in {"simple", "compound", "needle"} else None,
        "leaf_arrangement": arr,
        "leaf_margins": margins,
        "leaf_lobing": lobing,
        "bark_texture": None,
        "leaf_shape": shapes or None,
        "source": "dirr",
        "dirr_page": row.get("dirr_book_page"),
        "dirr_species": row.get("dirr_species"),
        "raw": feats,
    }


class PriorResolver:
    """Resolve a species to a canonical DeVo feature prior."""

    def __init__(self) -> None:
        self._dirr: dict[str, dict[str, Any]] = {}
        meta = json.loads(FEATURE_ASSERTIONS.read_text())
        for row in meta["species"]:
            for name in (row.get("dirr_species"), row.get("leafsnap_species")):
                if not name:
                    continue
                for variant in name_variants(name):
                    self._dirr.setdefault(variant, row)
        self._gobotany = None
        try:
            sys.path.insert(0, str(ROOT))
            from segmentation.gobotany_devo_adapter import GoBotanyDevoAdapter  # type: ignore

            if GOBOTANY_PARQUET.exists():
                self._gobotany = GoBotanyDevoAdapter(GOBOTANY_PARQUET)
        except Exception as exc:  # pragma: no cover - optional dependency path
            print(f"[priors] GoBotany adapter unavailable: {exc}", file=sys.stderr)

    # -- GoBotany ---------------------------------------------------------
    def _gobotany_values(self, species: str) -> dict[str, list[str]] | None:
        if self._gobotany is None:
            return None
        rows = self._gobotany.con.execute(
            """
            SELECT c.character_short_name, COALESCE(cvl.display_label, cvl.choice)
            FROM taxon t
            JOIN taxon_character_value tcv ON tcv.taxon_id = t.taxon_id
            JOIN character c
              ON c.character_short_name = tcv.character_short_name
             AND c.pile_slug = tcv.pile_slug
            LEFT JOIN character_value_label cvl
              ON cvl.character_short_name = tcv.character_short_name
             AND cvl.pile_slug = tcv.pile_slug
             AND cvl.value_index = tcv.value_index
            WHERE LOWER(t.scientific_name) = LOWER(?)
              AND c.character_short_name IN (
                'leaf_type_general_wa','leaf_type_specific_wa','leaf_arrangement_wa',
                'leaf_blade_margin_general_wa','leaf_blade_margin_specific_wa',
                'leaf_blade_shape_wa','outer_bark_main_stem_wa','leaf_form_wa',
                'leaf_blade_venation_wa','plant_habit_wa'
              )
            """,
            [species],
        ).fetchall()
        if not rows:
            return None
        out: dict[str, list[str]] = defaultdict(list)
        for char, value in rows:
            if value and str(value).strip().lower() != "na":
                out[char].append(str(value).strip())
        return dict(out)

    def _from_gobotany(self, species: str) -> dict[str, Any] | None:
        values = self._gobotany_values(species)
        if not values:
            return None
        leaf_types = [v.lower() for v in values.get("leaf_type_general_wa", [])]
        leaf_type = None
        if "compound" in leaf_types:
            leaf_type = "compound"
        elif "simple" in leaf_types:
            leaf_type = "simple"
        arrs = [ARRANGEMENT_MAP.get(v.lower()) for v in values.get("leaf_arrangement_wa", [])]
        arrs = sorted({a for a in arrs if a})
        margins = canonical_margins(
            values.get("leaf_blade_margin_general_wa", [])
            + values.get("leaf_blade_margin_specific_wa", [])
        )
        lobing = "lobed" if margins and "lobed" in margins else ("unlobed" if margins else None)
        bark = canonical_bark(values.get("outer_bark_main_stem_wa"))
        shapes = sorted({v.lower() for v in values.get("leaf_blade_shape_wa", [])})
        return {
            "leaf_type": leaf_type,
            "leaf_arrangement": arrs[0] if arrs else None,
            "leaf_margins": margins,
            "leaf_lobing": lobing,
            "bark_texture": bark,
            "leaf_shape": shapes or None,
            "source": "gobotany",
            "gobotany_specific_leaf_type": (values.get("leaf_type_specific_wa") or [None])[0],
            "raw": values,
        }

    def _gobotany_bark_consensus(self, genus: str) -> list[str] | None:
        if self._gobotany is None:
            return None
        rows = self._gobotany.con.execute(
            """
            SELECT COALESCE(cvl.display_label, cvl.choice) AS label
            FROM taxon t
            JOIN taxon_character_value tcv ON tcv.taxon_id = t.taxon_id
            JOIN character c
              ON c.character_short_name = tcv.character_short_name
             AND c.pile_slug = tcv.pile_slug
            LEFT JOIN character_value_label cvl
              ON cvl.character_short_name = tcv.character_short_name
             AND cvl.pile_slug = tcv.pile_slug
             AND cvl.value_index = tcv.value_index
            WHERE LOWER(t.genus) = LOWER(?)
              AND c.character_short_name = 'outer_bark_main_stem_wa'
            """,
            [genus],
        ).fetchall()
        values = Counter(str(r[0]).strip() for r in rows if r and r[0])
        return canonical_bark([v for v, _ in values.most_common(4)])

    def _from_genus_consensus(self, genus: str) -> dict[str, Any] | None:
        rows = [
            row for key, row in self._dirr.items()
            if row.get("dirr_species", "").split(" ")[0].lower() == genus.lower()
        ]
        if not rows:
            return None
        types = Counter()
        arrs = Counter()
        for row in rows:
            prof = profile_from_dirr(row)
            if prof["leaf_type"]:
                types[prof["leaf_type"]] += 1
            if prof["leaf_arrangement"]:
                arrs[prof["leaf_arrangement"]] += 1
        return {
            "leaf_type": types.most_common(1)[0][0] if types else None,
            "leaf_arrangement": arrs.most_common(1)[0][0] if arrs else None,
            "leaf_margins": None,
            "leaf_lobing": None,
            "bark_texture": self._gobotany_bark_consensus(genus),
            "leaf_shape": None,
            "source": "dirr_genus_consensus",
            "genus": genus,
            "n_dirr_species": len(rows),
        }

    def resolve(self, species: str) -> dict[str, Any]:
        for variant in name_variants(species):
            row = self._dirr.get(variant)
            if row:
                prof = profile_from_dirr(row)
                if prof["leaf_type"]:
                    # Dirr is primary; GoBotany only fills fields Dirr leaves
                    # unasserted (notably bark texture, which Dirr lacks).
                    gob = self._from_gobotany(species)
                    supplemented: list[str] = []
                    if gob:
                        for feature in ("leaf_arrangement", "leaf_margins", "leaf_lobing", "bark_texture"):
                            if not prof.get(feature) and gob.get(feature):
                                prof[feature] = gob[feature]
                                supplemented.append(feature)
                    if supplemented:
                        prof["source"] = "dirr+gobotany"
                        prof["supplemented_features"] = supplemented
                    return prof
        gob = self._from_gobotany(species)
        if gob and gob["leaf_type"]:
            return gob
        curated = CURATED_REGIONAL_PRIORS.get(species)
        if curated:
            return dict(curated)
        genus = species.split(" ")[0]
        consensus = self._from_genus_consensus(genus)
        if consensus:
            return consensus
        return {
            "leaf_type": None,
            "leaf_arrangement": None,
            "leaf_margins": None,
            "leaf_lobing": None,
            "bark_texture": None,
            "leaf_shape": None,
            "source": "unresolved",
        }


# ---------------------------------------------------------------------------
# Segmentation / contour triangulation
# ---------------------------------------------------------------------------
def _defect_count(contour: np.ndarray, max_dim: int) -> tuple[int, float]:
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3 or len(contour) < 4:
        return 0, 0.0
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return 0, 0.0
    if defects is None:
        return 0, 0.0
    defects = defects.reshape(-1, defects.shape[-1])
    threshold = DEFECT_DEPTH_FRAC * max(1, max_dim)
    significant = 0
    deepest = 0.0
    for i in range(defects.shape[0]):
        depth = float(defects[i, 3]) / 256.0
        deepest = max(deepest, depth)
        if depth > threshold:
            significant += 1
    return significant, deepest


def measure_mask(binary: np.ndarray) -> dict[str, Any] | None:
    """Structural measurements over a binary foreground mask."""
    binary = (binary > 0).astype(np.uint8)
    total = int(binary.sum())
    if total == 0:
        return None
    height, width = binary.shape[:2]
    max_dim = max(width, height)

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    areas = sorted(
        (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)),
        reverse=True,
    )
    if not areas:
        return None
    significant_components = [a for a in areas if a >= MIN_COMPONENT_FRAC * total]
    largest_fraction = areas[0] / total

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_COMPONENT_FRAC * total]
    if not contours:
        return None
    contours.sort(key=cv2.contourArea, reverse=True)
    main = contours[0]

    hull_all = cv2.convexHull(np.vstack(contours))
    hull_area = float(cv2.contourArea(hull_all))
    solidity = float(total / hull_area) if hull_area > 0 else 0.0

    defects, deepest_defect = _defect_count(main, max_dim)
    perimeter = float(cv2.arcLength(main, True))
    complexity = (perimeter * perimeter / total) if total else 0.0

    ellipse_fill = 0.0
    aspect = 0.0
    if len(main) >= 5:
        (cx, cy), (major, minor), _angle = cv2.fitEllipse(main)
        ellipse_area = math.pi * (major / 2.0) * (minor / 2.0)
        ellipse_fill = float(cv2.contourArea(main) / ellipse_area) if ellipse_area > 0 else 0.0
        aspect = float(max(major, minor) / max(1e-6, min(major, minor)))
    else:
        x, y, w, h = cv2.boundingRect(main)
        cx, cy, w, h = float(x), float(y), float(w), float(h)
        aspect = max(w, h) / max(1e-6, min(w, h))

    x, y, w, h = cv2.boundingRect(main)
    bbox_fill = float(total / max(1, w * h))
    fg_fraction = float(total / max(1, width * height))

    return {
        "mask_width": int(width),
        "mask_height": int(height),
        "foreground_pixels": total,
        "foreground_fraction": round(fg_fraction, 4),
        "n_components": len(significant_components),
        "component_areas": significant_components[:8],
        "largest_component_fraction": round(float(largest_fraction), 4),
        "solidity": round(float(solidity), 4),
        "convexity_defects": int(defects),
        "deepest_defect_px": round(float(deepest_defect), 2),
        "perimeter_complexity": round(float(complexity), 2),
        "ellipse_fill": round(float(ellipse_fill), 4),
        "aspect_ratio": round(float(aspect), 3),
        "bbox_fill": round(float(bbox_fill), 4),
    }


def triangulate_leaf(mask: np.ndarray, prior: dict[str, Any]) -> dict[str, Any]:
    """Apply the view-framing / leaf-type triangulation rules."""
    metrics = measure_mask(mask)
    leaf_type = prior.get("leaf_type")
    if metrics is None:
        return {
            "view_framing": "unmeasured",
            "devo_leaf_type": "not_assessable",
            "triangulation_bucket": "view_unresolved",
            "geometry": None,
            "reason": "empty or unreadable segmentation mask",
        }

    if leaf_type == "compound":
        isolated_leaflet = (
            metrics["solidity"] > SUBPART_SOLIDITY
            and metrics["convexity_defects"] <= SUBPART_MAX_DEFECTS
            and metrics["n_components"] <= 1
            and metrics["ellipse_fill"] >= SUBPART_MIN_ELLIPSE_FILL
            and 0.20 <= metrics["aspect_ratio"] <= 4.0
        )
        if isolated_leaflet:
            return {
                "view_framing": "sub_part",
                "devo_leaf_type": "not_assessable",
                "triangulation_bucket": "view_hard_negative",
                "geometry": metrics,
                "reason": (
                    "compound species but single-blade contour "
                    f"(solidity={metrics['solidity']:.3f} > {SUBPART_SOLIDITY}, "
                    f"defects={metrics['convexity_defects']} <= {SUBPART_MAX_DEFECTS}, "
                    f"components={metrics['n_components']}) - no rachis evidence"
                ),
            }
        return {
            "view_framing": "isolated_whole",
            "devo_leaf_type": "compound",
            "triangulation_bucket": "gold_specialist_train",
            "geometry": metrics,
            "reason": (
                "multi-leaflet / rachis evidence "
                f"(solidity={metrics['solidity']:.3f}, "
                f"defects={metrics['convexity_defects']}, "
                f"components={metrics['n_components']})"
            ),
        }

    if leaf_type == "simple":
        clean = (
            metrics["largest_component_fraction"] >= SIMPLE_MIN_LARGEST_FRAC
            and metrics["foreground_fraction"] >= SIMPLE_MIN_FG_FRAC
            and metrics["n_components"] <= 4
        )
        return {
            "view_framing": "isolated_whole" if clean else "fragmented",
            "devo_leaf_type": "simple",
            "triangulation_bucket": "gold_specialist_train" if clean else "view_unresolved",
            "geometry": metrics,
            "reason": (
                "clean single-blade simple contour"
                if clean
                else "simple species but fragmented / multi-blade crop"
            ),
        }

    if leaf_type == "needle":
        return {
            "view_framing": "isolated_whole",
            "devo_leaf_type": "not_assessable",
            "triangulation_bucket": "not_broadleaf",
            "geometry": metrics,
            "reason": "conifer / needle-leaf species outside the simple-compound axis",
        }

    return {
        "view_framing": "unresolved_prior",
        "devo_leaf_type": "not_assessable",
        "triangulation_bucket": "view_unresolved",
        "geometry": metrics,
        "reason": "no Dirr/GoBotany leaf-type prior",
    }


def leafsnap_mask_for(record: dict[str, Any]) -> np.ndarray | None:
    mask = cv2.imread(record["mask_path"], cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    height, width = mask.shape[:2]
    sx = width / max(1, record["image_width"])
    sy = height / max(1, record["image_height"])
    x1, y1, x2, y2 = record["full_bbox_xyxy"]
    x1, y1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
    x2, y2 = min(width, int(x2 * sx)), min(height, int(y2 * sy))
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return None
    binary = (region > 0).astype(np.uint8)
    return cv2.resize(
        binary,
        (record["crop_width"], record["crop_height"]),
        interpolation=cv2.INTER_NEAREST,
    )


def estimate_photo_mask(bgr: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """HSV green-subject estimate for field photos (no ground-truth mask)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = ((h >= 30) & (h <= 95) & (s >= 35) & (v >= 25)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    total = green.shape[0] * green.shape[1]
    green_fraction = float((green > 0).sum() / total)

    n, _labels, stats, _ = cv2.connectedComponentsWithStats((green > 0).astype(np.uint8), 8)
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True)
    dominant = (areas[0] / max(1, sum(areas))) if areas else 0.0
    info = {
        "green_fraction": round(green_fraction, 4),
        "green_components": len([a for a in areas if a >= 0.002 * total]),
        "dominant_green_fraction": round(float(dominant), 4),
    }
    if green_fraction < POOL_MASK_MIN_GREEN or dominant < POOL_MIN_DOMINANT_FRAC:
        return None, info
    return green, info


def classify_stage(bgr: np.ndarray, mask_info: dict[str, Any]) -> dict[str, Any]:
    """Heuristic Guided Encounter Stage from colour/texture statistics.

    Deliberately auditable: every input statistic is stored alongside the
    label. This is a field-plausible triage, not a trained organ classifier.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total = bgr.shape[0] * bgr.shape[1]

    green = ((h >= 30) & (h <= 95) & (s >= 35) & (v >= 25))
    green_fraction = float(green.sum() / total)

    brown = ((h >= 5) & (h <= 35) & (s >= 25) & (s <= 190) & (v >= 25) & (v <= 190))
    brown_fraction = float(brown.sum() / total)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float((edges > 0).sum() / total)

    saturated_flower = ((s >= 110) & (v >= 70) & ~green & ~brown)
    flower_fraction = float(saturated_flower.sum() / total)

    evidence = {
        "green_fraction": round(green_fraction, 4),
        "brown_fraction": round(brown_fraction, 4),
        "flower_fraction": round(flower_fraction, 4),
        "edge_density": round(edge_density, 4),
        "dominant_green_fraction": mask_info.get("dominant_green_fraction"),
        "green_components": mask_info.get("green_components"),
        "mean_saturation": round(float(s.mean() / 255.0), 4),
        "mean_value": round(float(v.mean() / 255.0), 4),
    }

    bark_scene = brown_fraction >= 0.32 and green_fraction <= 0.20
    sparse_bark = brown_fraction >= 0.18 and green_fraction <= 0.06 and edge_density >= 0.02
    foliage_context = (
        green_fraction >= 0.42
        and (mask_info.get("dominant_green_fraction") or 0.0) < 0.50
    )
    canopy_context = green_fraction >= 0.60 and edge_density >= 0.09

    if bark_scene or sparse_bark:
        stage = "stage_1_anchor"
        reason = "brown bark/trunk dominance with little foliage"
    elif foliage_context or canopy_context:
        stage = "stage_3_area_context"
        reason = "broad foliage coverage split across many green regions / habitat texture"
    else:
        stage = "stage_2_anatomy"
        if flower_fraction >= 0.10:
            reason = "saturated flower/colour subject with a bounded green background"
        elif (mask_info.get("dominant_green_fraction") or 0.0) >= 0.55:
            reason = "one dominant green subject (leaf / twig close-up)"
        else:
            reason = "bounded subject without bark or canopy dominance"
    return {"stage": stage, "stage_reason": reason, "stage_evidence": evidence}



# Zero-shot stage prompts, balanced three-per-stage. The prompt text is part of
# the model contract: changing it changes the staging, so it is recorded in the
# summary. CLIP is only a triage model here; every score is persisted.
CLIP_STAGE_PROMPTS: dict[int, list[str]] = {
    1: [
        "a close-up photo of rough tree bark",
        "a tree trunk filling the frame",
        "the base of a tree with exposed roots",
    ],
    2: [
        "a macro close-up of a single leaf filling the frame",
        "a close-up photo of a twig with leaves",
        "a close-up photo of a flower or fruit",
    ],
    3: [
        "a wide photo of a whole tree standing in a landscape",
        "a wide photo of a whole shrub in a garden",
        "a photo of a forest habitat with many trees",
    ],
}
CLIP_STAGE_NAMES = {
    1: "stage_1_anchor",
    2: "stage_2_anatomy",
    3: "stage_3_area_context",
}


class ClipStageClassifier:
    """Zero-shot Guided Encounter Stage with a transparent heuristic fallback."""

    def __init__(self, enabled: bool = True, model_id: str = "openai/clip-vit-base-patch32") -> None:
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.available = False
        if not enabled:
            return
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor

            self.model = CLIPModel.from_pretrained(model_id)
            self.processor = CLIPProcessor.from_pretrained(model_id)
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            self.model = self.model.to(self.device).eval()
            self.available = True
        except Exception as exc:  # model unavailable: heuristic fallback
            print(f"[stage] CLIP unavailable ({type(exc).__name__}: {exc}); using heuristic", file=sys.stderr)
            self.model = None
            self.processor = None

    def classify(self, images: list[Any]) -> list[dict[str, Any]]:
        if self.available and images:
            try:
                return self._classify_clip(images)
            except Exception as exc:
                print(f"[stage] CLIP inference failed ({type(exc).__name__}: {exc}); heuristic fallback", file=sys.stderr)
        return [
            classify_stage(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), {})
            for image in images
        ]

    def _classify_clip(self, images: list[Any]) -> list[dict[str, Any]]:
        import torch

        labels: list[int] = []
        texts: list[str] = []
        for stage_id, prompts in CLIP_STAGE_PROMPTS.items():
            for prompt in prompts:
                labels.append(stage_id)
                texts.append(prompt)
        out: list[dict[str, Any]] = []
        batch_size = 16
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            inputs = self.processor(text=texts, images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits_per_image
            probs = logits.softmax(dim=1).cpu()
            for row in probs:
                agg = {stage: 0.0 for stage in (1, 2, 3)}
                for j, stage_id in enumerate(labels):
                    agg[stage_id] += float(row[j]) / len(CLIP_STAGE_PROMPTS[stage_id])
                best = max(agg, key=agg.get)
                out.append(
                    {
                        "stage": CLIP_STAGE_NAMES[best],
                        "stage_reason": f"zero-shot CLIP (balanced 3-prompt ensemble) -> {CLIP_STAGE_NAMES[best]}",
                        "stage_scores": {CLIP_STAGE_NAMES[k]: round(v, 4) for k, v in agg.items()},
                        "stage_model": self.model_id,
                    }
                )
        return out

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
@dataclass
class PoolPhoto:
    uid: str
    source: str
    species: str
    taxon_rank: str
    instance_id: str | None
    observed_on: str | None
    url: str | None
    local_path: Path | None
    prior: dict[str, Any]


def sql_in_list(values: tuple[str, ...] | list[str]) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def query_henrico(per_species: int) -> tuple[list[PoolPhoto], dict[str, int]]:
    species = list(TARGET_SPECIES) + list(CARYA_SPECIES)
    sql = f"""
        SELECT o.arborphy_id, o.taxon_name, o.taxon_rank, o.observed_on,
               COALESCE(m.url, o.photo_url) AS url
        FROM observations o
        LEFT JOIN LATERAL (
            SELECT url FROM observation_media
            WHERE observation_id = o.arborphy_id
            ORDER BY position ASC, media_id ASC
            LIMIT 1
        ) m ON TRUE
        WHERE o.deleted_at IS NULL
          AND o.area_instance_id = '{HENRICO_ID}'
          AND o.iconic_taxon_name = 'Plantae'
          AND o.taxon_rank = 'species'
          AND o.taxon_name IN ({sql_in_list(species)})
          AND COALESCE(m.url, o.photo_url) IS NOT NULL
        ORDER BY o.taxon_name ASC, o.observed_on ASC NULLS LAST, o.arborphy_id ASC
    """
    rows = psql(sql)
    counts = Counter(row[1] for row in rows)

    genus_sql = f"""
        SELECT o.arborphy_id, o.taxon_name, o.taxon_rank, o.observed_on,
               COALESCE(m.url, o.photo_url) AS url
        FROM observations o
        LEFT JOIN LATERAL (
            SELECT url FROM observation_media
            WHERE observation_id = o.arborphy_id
            ORDER BY position ASC, media_id ASC
            LIMIT 1
        ) m ON TRUE
        WHERE o.deleted_at IS NULL
          AND o.area_instance_id = '{HENRICO_ID}'
          AND o.iconic_taxon_name = 'Plantae'
          AND o.taxon_rank = 'genus'
          AND o.taxon_name IN ({sql_in_list(list(GENUS_GROUPS))})
          AND COALESCE(m.url, o.photo_url) IS NOT NULL
        ORDER BY o.taxon_name ASC, o.observed_on ASC NULLS LAST, o.arborphy_id ASC
    """
    genus_rows = psql(genus_sql)

    by_species: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        by_species[row[1]].append(row)
    for row in genus_rows:
        by_species[GENUS_GROUPS.get(row[1], row[1])].append(row)
        counts[GENUS_GROUPS.get(row[1], row[1])] += 1

    photos: list[PoolPhoto] = []
    for name, group in sorted(by_species.items()):
        for row in group[:per_species]:
            photos.append(
                PoolPhoto(
                    uid=f"henrico_{row[0]}",
                    source="henrico",
                    species=name,
                    taxon_rank=row[2],
                    instance_id=HENRICO_ID,
                    observed_on=row[3] or None,
                    url=row[4],
                    local_path=None,
                    prior={},
                )
            )
    return photos, dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def query_tuckaway(per_species: int) -> tuple[list[PoolPhoto], dict[str, int]]:
    sql = f"""
        SELECT o.arborphy_id, o.taxon_name, o.taxon_rank, o.observed_on,
               COALESCE(m.url, o.photo_url) AS url
        FROM observations o
        LEFT JOIN LATERAL (
            SELECT url FROM observation_media
            WHERE observation_id = o.arborphy_id
            ORDER BY position ASC, media_id ASC
            LIMIT 1
        ) m ON TRUE
        WHERE o.deleted_at IS NULL
          AND (o.area_instance_id = '{TUCKAWAY_ID}' OR o.instance_id = '{TUCKAWAY_ID}')
          AND o.iconic_taxon_name = 'Plantae'
          AND o.taxon_rank = 'species'
          AND COALESCE(m.url, o.photo_url) IS NOT NULL
        ORDER BY o.taxon_name ASC, o.observed_on ASC NULLS LAST, o.arborphy_id ASC
    """
    rows = psql(sql)
    counts = Counter(row[1] for row in rows)
    photos: list[PoolPhoto] = []
    for row in rows[:per_species]:
        photos.append(
            PoolPhoto(
                uid=f"tuckaway_{row[0]}",
                source="tuckaway",
                species=row[1],
                taxon_rank=row[2],
                instance_id=TUCKAWAY_ID,
                observed_on=row[3] or None,
                url=row[4],
                local_path=None,
                prior={},
            )
        )
    return photos, dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def tuckaway_walk_photos() -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    if TUCKAWAY_QA.exists():
        for input_file in sorted(TUCKAWAY_QA.glob("*/input.json")):
            try:
                payload = json.loads(input_file.read_text())
            except Exception:
                continue
            image = payload.get("image")
            if image and Path(image).exists():
                path = Path(image)
                if path not in seen:
                    seen.add(path)
                    files.append(path)
    if TUCKAWAY_PHYS.exists():
        for path in sorted(TUCKAWAY_PHYS.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg"} and "video" not in path.parts:
                if path not in seen:
                    seen.add(path)
                    files.append(path)
    return files


def upgrade_photo_url(url: str) -> str:
    if url.startswith("/"):
        return API_BASE + url
    for size in ("square", "small", "thumb"):
        url = url.replace(f"/{size}.", "/large.")
    return url


def image_extension(content_type: str, url: str) -> str:
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    lower = url.lower()
    for ext in (".jpeg", ".jpg", ".png", ".webp"):
        if lower.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def download(url: str, destination: Path, timeout: int = 30) -> dict[str, Any]:
    if destination.exists() and destination.stat().st_size > 1024:
        return {"ok": True, "cached": True, "bytes": destination.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "arborphy-leafsnap-dataset/1.0 (+research)"}
    last_error = ""
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                time.sleep(0.6 * (attempt + 1))
                continue
            tmp = destination.with_suffix(destination.suffix + ".part")
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)
            if tmp.stat().st_size <= 1024:
                tmp.unlink(missing_ok=True)
                last_error = "too small"
                continue
            tmp.replace(destination)
            return {
                "ok": True,
                "cached": False,
                "bytes": destination.stat().st_size,
                "content_type": response.headers.get("content-type", ""),
            }
        except Exception as exc:  # network flake: keep going
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.8 * (attempt + 1))
    return {"ok": False, "error": last_error}


def copy_local_resized(source: Path, destination: Path, max_edge: int = 1600) -> dict[str, Any]:
    if destination.exists() and destination.stat().st_size > 1024:
        return {"ok": True, "cached": True, "bytes": destination.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_edge, max_edge), Image.LANCZOS)
            image.save(destination, "JPEG", quality=88, optimize=True)
        return {"ok": True, "cached": False, "bytes": destination.stat().st_size}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def copy_or_link(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "exists"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


# ---------------------------------------------------------------------------
# Chordal Feature Buffer Index
# ---------------------------------------------------------------------------
def _profile_signature(prior: dict[str, Any]) -> dict[str, Any]:
    return {
        "leaf_type": prior.get("leaf_type"),
        "leaf_arrangement": prior.get("leaf_arrangement"),
        "leaf_margins": tuple(prior.get("leaf_margins") or ()) or None,
        "leaf_lobing": prior.get("leaf_lobing"),
        "bark_texture": tuple(prior.get("bark_texture") or ()) or None,
    }


def _matches(candidate: dict[str, Any], target: dict[str, Any], features: list[str]) -> bool:
    """Narrowing match. A feature the target lacks is not a constraint;
    a candidate that lacks the feature cannot be confirmed and drops out."""
    for feature in features:
        tval = target.get(feature)
        if tval is None:
            continue
        cval = candidate.get(feature)
        if cval is None:
            return False
        if feature in {"leaf_margins", "bark_texture"}:
            if not set(cval) & set(tval):
                return False
        elif cval != tval:
            return False
    return True


def chordal_index(
    priors: dict[str, dict[str, Any]],
    stage_counts: dict[str, Counter],
) -> dict[str, Any]:
    signatures = {name: _profile_signature(prior) for name, prior in priors.items()}
    names = sorted(signatures)
    total = len(names)

    trace = []
    active_features: list[str] = []
    for feature in (
        "leaf_type",
        "leaf_arrangement",
        "leaf_margins",
        "leaf_lobing",
        "bark_texture",
    ):
        if not any(signatures[n].get(feature) is not None for n in names):
            continue
        active_features.append(feature)
        # Strict chain: only species carrying every feature so far, so the
        # trace is monotonic and comparable across steps.
        chain = [
            n for n in names
            if all(signatures[n].get(f) is not None for f in active_features)
        ]
        if not chain:
            continue
        counts = []
        for name in chain:
            narrowed = [
                other for other in chain
                if _matches(signatures[other], signatures[name], active_features)
            ]
            counts.append(len(narrowed))
        trace.append(
            {
                "step": len(trace) + 1,
                "feature": feature,
                "species_with_prior": len(chain),
                "mean_candidates_remaining": round(float(np.mean(counts)), 2),
                "median_candidates_remaining": int(np.median(counts)),
                "min_candidates_remaining": int(min(counts)),
                "max_candidates_remaining": int(max(counts)),
            }
        )

    per_species: dict[str, Any] = {}
    for name in names:
        counts = stage_counts.get(name, Counter())
        s1, s2, s3 = counts.get("stage_1_anchor", 0), counts.get("stage_2_anatomy", 0), counts.get("stage_3_area_context", 0)
        # Two anatomy documents = a confident Stage 2 buffer; one is partial
        # credit (the species was seen, but only once).
        anatomy_weight = 0.45 if s2 >= 2 else (0.22 if s2 == 1 else 0.0)
        stage_coverage = (
            0.45 * (1.0 if s1 >= 1 else 0.0)
            + anatomy_weight
            + 0.10 * (1.0 if s3 >= 1 else 0.0)
        )
        enabled = []
        if s1 >= 1:
            enabled.append("bark_texture")
        if s2 >= 1:
            enabled.extend(["leaf_type", "leaf_arrangement", "leaf_margins", "leaf_lobing"])
        enabled = [f for f in enabled if signatures[name].get(f) is not None]

        if not enabled or total <= 1:
            narrowed_names = names
            narrowing_power = 0.0
        else:
            narrowed_names = [
                other for other in names
                if _matches(signatures[other], signatures[name], enabled)
            ]
            narrowing_power = max(0.0, 1.0 - (len(narrowed_names) - 1) / (total - 1))
        cfbi = stage_coverage * narrowing_power
        per_species[name] = {
            "stage_1_documents": s1,
            "stage_2_documents": s2,
            "stage_3_documents": s3,
            "stage_coverage": round(stage_coverage, 3),
            "enabled_features": enabled,
            "narrowed_candidate_count": len(narrowed_names),
            "candidate_pool_size": total,
            "narrowing_power": round(narrowing_power, 3),
            "chordal_feature_buffer_index": round(cfbi, 3),
            "narrowed_species": narrowed_names,
            "signature": {k: (list(v) if isinstance(v, tuple) else v) for k, v in signatures[name].items()},
        }
    return {"candidate_pool_size": total, "pool_narrowing_trace": trace, "per_species": per_species}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_leafsnap_dataset(priors: PriorResolver) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(CLASSIFICATION_MANIFEST)
    triangulated: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    sub_part: list[dict[str, Any]] = []
    buckets: Counter = Counter()
    reasons: Counter = Counter()
    by_species: dict[str, Counter] = defaultdict(Counter)

    for record in records:
        prior = priors.resolve(record["species"])
        mask = leafsnap_mask_for(record)
        result = triangulate_leaf(mask, prior) if mask is not None else {
            "view_framing": "unmeasured",
            "devo_leaf_type": "not_assessable",
            "triangulation_bucket": "view_unresolved",
            "geometry": None,
            "reason": "mask crop outside the source mask bounds",
        }
        row = {
            "uid": record["uid"],
            "dataset": "leafsnap_classification_crop",
            "split": record["split"],
            "species": record["species"],
            "dirr_species": prior.get("dirr_species"),
            "prior_source": prior.get("source"),
            "prior_leaf_type": prior.get("leaf_type"),
            "domains": record.get("domain"),
            "image": record.get("image"),
            "source_image": record.get("source_image"),
            "mask_path": record.get("mask_path"),
            "crop_bbox_xyxy": record.get("crop_bbox_xyxy"),
            "crop_width": record.get("crop_width"),
            "crop_height": record.get("crop_height"),
            **result,
        }
        triangulated.append(row)
        buckets[result["triangulation_bucket"]] += 1
        by_species[record["species"]][result["triangulation_bucket"]] += 1
        if result["triangulation_bucket"] == "gold_specialist_train":
            gold.append(row)
        elif result["triangulation_bucket"] == "view_hard_negative":
            sub_part.append(row)
        else:
            reasons[result["reason"].split(" (")[0]] += 1

    summary = {
        "n_crops": len(records),
        "buckets": dict(buckets),
        "gold_total": len(gold),
        "gold_by_split_class": dict(
            Counter(f"{row['split']}:{row['devo_leaf_type']}" for row in gold)
        ),
        "hard_negative_sub_part": len(sub_part),
        "hard_negative_by_species": dict(
            Counter(row["species"] for row in sub_part).most_common()
        ),
        "unresolved_reasons": dict(reasons.most_common()),
        "by_species": {
            name: dict(counter) for name, counter in sorted(by_species.items())
        },
    }
    return triangulated, gold, summary


def materialize_dataset(
    triangulated: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    **_: Any,
) -> dict[str, Any]:
    if OUT_DATASET.exists():
        shutil.rmtree(OUT_DATASET)
    OUT_DATASET.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    link_modes: Counter = Counter()
    for row in gold:
        class_name = f"{row['devo_leaf_type']}_leaf"
        split = row["split"]
        source = CLASSIFICATION / row["image"]
        filename = f"{row['uid']}_{slug(row['species'])}.jpg"
        destination = OUT_DATASET / split / class_name / filename
        link_modes[copy_or_link(source, destination)] += 1
        manifest_rows.append(
            {
                "uid": row["uid"],
                "image": f"{split}/{class_name}/{filename}",
                "class_id": 0 if class_name == "simple_leaf" else 1,
                "class_name": class_name,
                "split": split,
                "species": row["species"],
                "dirr_species": row.get("dirr_species"),
                "prior_source": row.get("prior_source"),
                "view_framing": row["view_framing"],
                "devo_leaf_type": row["devo_leaf_type"],
                "triangulation_bucket": row["triangulation_bucket"],
                "source": "leafsnap",
                "source_image": row.get("source_image"),
                "mask_path": row.get("mask_path"),
                "crop_bbox_xyxy": row.get("crop_bbox_xyxy"),
                "geometry": row.get("geometry"),
            }
        )

    hard_rows: list[dict[str, Any]] = []
    for row in triangulated:
        if row["triangulation_bucket"] != "view_hard_negative":
            continue
        source = CLASSIFICATION / row["image"]
        filename = f"{row['uid']}_{slug(row['species'])}.jpg"
        destination = OUT_DATASET / "hard_negatives_sub_part" / row["split"] / filename
        link_modes[copy_or_link(source, destination)] += 1
        hard_rows.append(
            {
                "uid": row["uid"],
                "image": f"hard_negatives_sub_part/{row['split']}/{filename}",
                "class_name": "sub_part_leaflet",
                "source_class": "compound_leaf",
                "split": row["split"],
                "species": row["species"],
                "dirr_species": row.get("dirr_species"),
                "prior_source": row.get("prior_source"),
                "view_framing": row["view_framing"],
                "devo_leaf_type": row["devo_leaf_type"],
                "triangulation_bucket": row["triangulation_bucket"],
                "reason": row.get("reason"),
                "geometry": row.get("geometry"),
                "source": "leafsnap",
                "source_image": row.get("source_image"),
                "mask_path": row.get("mask_path"),
                "crop_bbox_xyxy": row.get("crop_bbox_xyxy"),
            }
        )

    write_jsonl(OUT_DATASET / "manifest.jsonl", manifest_rows)
    write_jsonl(OUT_DATASET / "hard_negatives_sub_part" / "manifest.jsonl", hard_rows)
    write_jsonl(OUT_DATASET / "triangulation.jsonl", triangulated)

    counts = {
        "dataset_version": "classification-v3-triangulated",
        "by_split_class": {},
        "by_split_domain": {},
        "hard_negatives_by_split": dict(Counter(row["split"] for row in hard_rows)),
        "link_modes": dict(link_modes),
    }
    for row in manifest_rows:
        split = row["split"]
        counts["by_split_class"].setdefault(split, Counter())
        counts["by_split_class"][split][row["class_name"]] += 1
    counts["by_split_class"] = {
        split: dict(sorted(classes.items())) for split, classes in sorted(counts["by_split_class"].items())
    }
    # domain from the source crops (field/lab), kept for parity with v1/v2
    domain_by_uid = {
        row["uid"]: row.get("domains")
        for row in triangulated
    }
    domain_counter: dict[str, Counter] = defaultdict(Counter)
    for row in manifest_rows:
        domain_counter[row["split"]][str(domain_by_uid.get(row["uid"]))] += 1
    counts["by_split_domain"] = {
        split: dict(sorted(counter.items())) for split, counter in sorted(domain_counter.items())
    }
    write_json(OUT_DATASET / "counts.json", counts)

    labels = {
        "class_names": ["simple_leaf", "compound_leaf"],
        "classes": {"simple_leaf": 0, "compound_leaf": 1},
        "label_source": "Dirr + GoBotany DeVo leaf_type prior, triangulated on segmentation-mask geometry",
        "view_framing": {
            "isolated_whole": "the crop frames the whole foliar unit (simple blade or compound rachis)",
            "sub_part": "the crop frames one leaflet of a compound leaf; leaf type is not assessable",
            "fragmented": "fragmented / multi-blade simple crop",
            "unmeasured": "mask could not be measured",
        },
        "buckets": {
            "gold_specialist_train": "view-valid simple or compound crops",
            "view_hard_negative": "isolated-leaflet compound crops (never deleted; bench hard negatives)",
            "view_unresolved": "fragmentary / unmeasurable crops (not materialised)",
            "not_broadleaf": "conifer/needle-leaf crops outside the simple-compound axis",
        },
        "thresholds": {
            "subpart_solidity": SUBPART_SOLIDITY,
            "subpart_max_defects": SUBPART_MAX_DEFECTS,
            "subpart_min_ellipse_fill": SUBPART_MIN_ELLIPSE_FILL,
            "component_min_fraction": MIN_COMPONENT_FRAC,
            "defect_depth_fraction": DEFECT_DEPTH_FRAC,
        },
        "splits": {
            split: dict(sorted(Counter(row["class_name"] for row in manifest_rows if row["split"] == split).items()))
            for split in sorted({row["split"] for row in manifest_rows})
        },
    }
    write_json(OUT_DATASET / "labels.json", labels)
    return counts


def build_pool(
    priors: PriorResolver,
    henrico_photos: list[PoolPhoto],
    tuckaway_photos: list[PoolPhoto],
    max_photos: int,
    offline: bool,
    workers: int,
    use_clip: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for photo in henrico_photos + tuckaway_photos:
        photo.prior = priors.resolve(photo.species)

    walk_files = tuckaway_walk_photos()
    walk_photos: list[PoolPhoto] = []
    for path in walk_files:
        walk_photos.append(
            PoolPhoto(
                uid=f"tuckaway_walk_{slug(path.stem)}",
                source="tuckaway_walk",
                species="unidentified_tuckaway_walk",
                taxon_rank="unidentified",
                instance_id=TUCKAWAY_ID,
                observed_on=None,
                url=None,
                local_path=path,
                prior={
                    "leaf_type": None,
                    "leaf_arrangement": None,
                    "leaf_margins": None,
                    "leaf_lobing": None,
                    "bark_texture": None,
                    "leaf_shape": None,
                    "source": "unlabeled_walk_photo",
                },
            )
        )

    # Round-robin across species so the cap never starves the long tail.
    # Tuckaway's persisted pool is tiny (one Plantae record), so it is always
    # in the cohort; the Henrico round-robin fills the regional budget.
    candidates: list[PoolPhoto] = list(tuckaway_photos)
    by_species: dict[str, list[PoolPhoto]] = defaultdict(list)
    for photo in henrico_photos:
        by_species[photo.species].append(photo)
    ordered_species = sorted(by_species)
    cursor = 0
    while len(candidates) < max_photos and ordered_species:
        added = False
        for name in ordered_species:
            bucket = by_species[name]
            if cursor < len(bucket) and len(candidates) < max_photos:
                candidates.append(bucket[cursor])
                added = True
        cursor += 1
        if not added:
            break
    # Every local Tuckaway walk frame is free and unlabelled; include them all
    # (capped) so the staged index reflects the physical walk, not a sample.
    walk_budget = min(len(walk_photos), 40)
    candidates.extend(walk_photos[:walk_budget])
    candidates = candidates[: max_photos + 40]

    # -- download / copy ------------------------------------------------
    jobs = []
    for photo in candidates:
        if photo.local_path is not None:
            destination = POOL_IMAGES / slug(photo.species) / f"{photo.uid}.jpg"
            jobs.append((photo, "local", destination))
        else:
            url = upgrade_photo_url(photo.url or "")
            ext = image_extension("", url)
            destination = POOL_IMAGES / slug(photo.species) / f"{photo.uid}{ext}"
            jobs.append((photo, url, destination))

    results: dict[str, dict[str, Any]] = {}
    if offline:
        for photo, target, destination in jobs:
            results[photo.uid] = {
                "ok": destination.exists(),
                "cached": True,
                "bytes": destination.stat().st_size if destination.exists() else 0,
            }
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {}
            for photo, target, destination in jobs:
                if target == "local":
                    futures[pool.submit(copy_local_resized, photo.local_path, destination)] = photo.uid
                else:
                    futures[pool.submit(download, target, destination)] = photo.uid
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    results[uid] = future.result()
                except Exception as exc:
                    results[uid] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # -- decode once, classify stages in batches ------------------------
    decoded: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for photo, target, destination in jobs:
        outcome = results.get(photo.uid, {"ok": False, "error": "not attempted"})
        if not outcome.get("ok") or not destination.exists():
            failures.append(
                {
                    "uid": photo.uid,
                    "species": photo.species,
                    "source": photo.source,
                    "url": photo.url,
                    "error": outcome.get("error", "missing"),
                }
            )
            continue
        try:
            with Image.open(destination) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                width, height = rgb.size
        except Exception as exc:
            failures.append(
                {"uid": photo.uid, "species": photo.species, "source": photo.source, "error": f"decode: {exc}"}
            )
            continue
        decoded.append(
            {
                "photo": photo,
                "destination": destination,
                "width": width,
                "height": height,
                "rgb": rgb,
            }
        )

    stage_model = ClipStageClassifier(enabled=use_clip)
    stages = stage_model.classify([item["rgb"] for item in decoded])

    rows: list[dict[str, Any]] = []
    for item, stage in zip(decoded, stages):
        photo = item["photo"]
        destination = item["destination"]
        width, height = item["width"], item["height"]
        bgr = cv2.cvtColor(np.array(item["rgb"]), cv2.COLOR_RGB2BGR)
        estimated_mask, mask_info = estimate_photo_mask(bgr)
        heuristic = classify_stage(bgr, mask_info)
        prior = photo.prior
        leaf_type = prior.get("leaf_type")

        stage_evidence = dict(stage.get("stage_scores") or {})
        stage_evidence.update({
            "heuristic_stage": heuristic["stage"],
            "heuristic_reason": heuristic["stage_reason"],
            **{f"heuristic_{k}": v for k, v in heuristic["stage_evidence"].items()},
        })

        if stage["stage"] == "stage_2_anatomy" and estimated_mask is not None and leaf_type in {"simple", "compound"}:
            triangulation = triangulate_leaf(estimated_mask, prior)
            triangulation["mask_source"] = "estimated_hsv_green"
        else:
            triangulation = {
                "view_framing": "context_only",
                "devo_leaf_type": "not_assessable" if leaf_type != "simple" else "simple",
                "triangulation_bucket": "stage_context",
                "geometry": None,
                "reason": (
                    f"{stage['stage']} frames the plant rather than an isolated foliar unit"
                    if stage["stage"] != "stage_2_anatomy"
                    else "estimated green mask is not a single bounded subject"
                ),
            }

        rows.append(
            {
                "uid": photo.uid,
                "dataset": "mid_atlantic_pool",
                "split": "regional_pool",
                "source": photo.source,
                "instance_id": photo.instance_id,
                "species": photo.species,
                "taxon_rank": photo.taxon_rank,
                "prior_source": prior.get("source"),
                "prior_leaf_type": leaf_type,
                "observed_on": photo.observed_on,
                "source_url": photo.url,
                "image": str(destination.relative_to(OUT_POOL)),
                "image_width": width,
                "image_height": height,
                "bytes": destination.stat().st_size,
                "stage": stage["stage"],
                "stage_reason": stage["stage_reason"],
                "stage_model": stage.get("stage_model", "heuristic_v1"),
                "stage_scores": stage.get("stage_scores"),
                "stage_evidence": stage_evidence,
                "mask_estimate": mask_info,
                "view_framing": triangulation["view_framing"],
                "devo_leaf_type": triangulation["devo_leaf_type"],
                "triangulation_bucket": triangulation["triangulation_bucket"],
                "geometry": triangulation.get("geometry"),
                "reason": triangulation.get("reason"),
                "mask_source": triangulation.get("mask_source"),
            }
        )

    summary = {
        "requested": len(candidates),
        "downloaded": len(rows),
        "failed": len(failures),
        "failures": failures[:50],
        "stage_model": stage_model.model_id if stage_model.available else "heuristic_v1",
        "stage_model_device": stage_model.device if stage_model.available else None,
        "stage_prompts": {
            CLIP_STAGE_NAMES[k]: v for k, v in CLIP_STAGE_PROMPTS.items()
        } if stage_model.available else None,
        "by_source": dict(Counter(row["source"] for row in rows)),
        "by_species": dict(Counter(row["species"] for row in rows).most_common()),
        "stage_distribution": dict(Counter(row["stage"] for row in rows).most_common()),
        "stage_by_source": {
            source: dict(Counter(row["stage"] for row in rows if row["source"] == source))
            for source in sorted({row["source"] for row in rows})
        },
        "stage_agreement_with_heuristic": round(
            sum(1 for row in rows if row["stage"] == row["stage_evidence"].get("heuristic_stage"))
            / max(1, len(rows)),
            3,
        ),
        "triangulation_buckets": dict(Counter(row["triangulation_bucket"] for row in rows).most_common()),
        "view_framing": dict(Counter(row["view_framing"] for row in rows).most_common()),
    }
    return rows, summary


def write_contact_sheet(rows: list[dict[str, Any]], path: Path, columns: int = 10) -> None:
    """One glanceable sheet of the staged regional cohort (stage + species)."""
    if not rows:
        return
    thumb_w, thumb_h, label_h = 150, 120, 16
    per_sheet = columns * 18
    for page, start in enumerate(range(0, len(rows), per_sheet)):
        chunk = rows[start:start + per_sheet]
        lines = (len(chunk) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_w, lines * (thumb_h + label_h)), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(chunk):
            x = (index % columns) * thumb_w
            y = (index // columns) * (thumb_h + label_h)
            try:
                with Image.open(OUT_POOL / row["image"]) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail((thumb_w, thumb_h))
            except Exception:
                continue
            sheet.paste(thumb, (x, y))
            short = f"{row['stage'][8:10]} {row['species'][:16]}"
            draw.text((x + 2, y + thumb_h + 1), short, fill=(0, 0, 0))
        target = path if page == 0 else path.with_name(f"{path.stem}-{page + 1}{path.suffix}")
        sheet.save(target, quality=85)


def build_markdown(summary: dict[str, Any], dataset_counts: dict[str, Any]) -> str:
    crops = summary.get("leafsnap") or {}
    pool = summary.get("regional_pool") or {}
    chordal = summary.get("chordal") or {}
    henrico = summary.get("henrico") or {}
    tuckaway = summary.get("tuckaway") or {}

    lines: list[str] = []
    lines.append("# Mid-Atlantic Triangulated Dataset & Staged Encounter Index\n")
    lines.append(
        "Generated by `experiments/leafsnap-yolo26/build_mid_atlantic_triangulated.py`.\n"
    )
    lines.append(
        "This run cross-references Henrico County (`inst-osm-relation-3070728`) and "
        "Tuckaway (`inst-mtuhbm9y` + the local walk photos) with the Dirr / GoBotany "
        "DeVo priors, then applies the triangulation & view-framing engine to the "
        "LeafSnap crops and the regional pool.\n"
    )

    lines.append("## 1. Pool inventory\n")
    lines.append(f"- Henrico Plantae species with photos in the curated Mid-Atlantic set: **{len(henrico['species_counts'])}**")
    lines.append(f"- Tuckaway Plantae species with photos: **{len(tuckaway['species_counts'])}** ({', '.join(tuckaway['species_counts']) or 'none'})")
    lines.append(f"- Shared Henrico x Tuckaway species: **{len(summary['shared_species'])}**")
    lines.append(f"- Combined candidate pool: **{len(summary['combined_species'])}** species\n")

    lines.append("| Species | Henrico photos | Prior source | DeVo leaf type | Arrangement | Margins | Bark |")
    lines.append("|---|---:|---|---|---|---|---|")
    for name, count in henrico["species_counts"].items():
        prior = summary["priors"].get(name, {})
        lines.append(
            f"| *{name}* | {count} | {prior.get('source', '?')} | {prior.get('leaf_type') or '-'} | "
            f"{prior.get('leaf_arrangement') or '-'} | {', '.join(prior.get('leaf_margins') or []) or '-'} | "
            f"{', '.join(prior.get('bark_texture') or []) or '-'} |"
        )
    lines.append("")

    lines.append("## 2. LeafSnap crop triangulation\n")
    lines.append(f"- crops measured: **{crops.get('n_crops', 0)}**")
    lines.append(f"- `gold_specialist_train`: **{crops.get('gold_total', 0)}**")
    lines.append(f"- `view_hard_negative` (isolated leaflet of a compound species): **{crops.get('hard_negative_sub_part', 0)}**")
    lines.append(f"- materialised dataset: `scratch/leafsnap-yolo26/classification-v3-triangulated/`")
    lines.append(f"- counts: `{json.dumps(dataset_counts.get('by_split_class', {}))}`\n")

    lines.append("### Hard negatives by species\n")
    lines.append("| Species | Isolated-leaflet crops |")
    lines.append("|---|---:|")
    for name, count in (crops.get("hard_negative_by_species") or {}).items():
        lines.append(f"| *{name}* | {count} |")
    lines.append("")
    lines.append(
        "`Juglans cinerea` dominates the hard-negative set: its field crops often "
        "isolate one leaflet, which carries no rachis evidence and must not be "
        "used to teach `compound_leaf`. Those crops are preserved under "
        "`hard_negatives_sub_part/` and surfaced in `bench_triangulated_manifest.jsonl`.\n"
    )

    lines.append("## 3. Regional pool cohort\n")
    lines.append(f"- requested: **{pool.get('requested', 0)}**, downloaded: **{pool.get('downloaded', 0)}**, failed: **{pool.get('failed', 0)}**")
    lines.append(f"- by source: `{json.dumps(pool.get('by_source', {}))}`")
    lines.append(f"- stage distribution: `{json.dumps(pool.get('stage_distribution', {}))}`")
    lines.append(f"- stage model: `{pool.get('stage_model', 'heuristic_v1')}` (heuristic agreement {pool.get('stage_agreement_with_heuristic', 0):.0%})")
    lines.append(f"- triangulation buckets: `{json.dumps(pool.get('triangulation_buckets', {}))}`\n")

    lines.append("### Species coverage\n")
    lines.append("| Species | Photos | Stage 1 | Stage 2 | Stage 3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, count in (pool.get("by_species") or {}).items():
        stages = (pool.get("stage_by_species_full") or {}).get(name, {})
        lines.append(
            f"| *{name}* | {count} | {stages.get('stage_1_anchor', 0)} | "
            f"{stages.get('stage_2_anatomy', 0)} | {stages.get('stage_3_area_context', 0)} |"
        )
    lines.append("")

    lines.append("## 4. Chordal Feature Buffer Index\n")
    lines.append(
        "The index shows how the Guided Encounter stages narrow the candidate pool: "
        "Stage 1 enables the bark prior, Stage 2 enables the leaf priors "
        "(type, arrangement, margins, lobing). `narrowing_power = 1 - (n-1)/(N-1)`; "
        "`CFBI = stage_coverage x narrowing_power`.\n"
    )
    lines.append("| Species | S1 | S2 | S3 | Enabled features | Candidates left | Narrowing | CFBI |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for name, row in sorted(
        chordal.get("per_species", {}).items(),
        key=lambda kv: -kv[1]["chordal_feature_buffer_index"],
    ):
        lines.append(
            f"| *{name}* | {row['stage_1_documents']} | {row['stage_2_documents']} | "
            f"{row['stage_3_documents']} | {', '.join(row['enabled_features']) or '-'} | "
            f"{row['narrowed_candidate_count']} | {row['narrowing_power']:.2f} | "
            f"{row['chordal_feature_buffer_index']:.2f} |"
        )
    lines.append("")

    lines.append("### Pool narrowing trace (all species)\n")
    lines.append("| Step | Feature | Species with prior | Mean candidates left | Median |")
    lines.append("|---:|---|---:|---:|---:|")
    for step in chordal.get("pool_narrowing_trace", []):
        lines.append(
            f"| {step['step']} | `{step['feature']}` | {step['species_with_prior']} | "
            f"{step['mean_candidates_remaining']} | {step['median_candidates_remaining']} |"
        )
    lines.append("")

    lines.append("## 5. Method and honest limits\n")
    lines.append(
        "- **Leaf priors.** Dirr `leaf_type` wins; `leaf_shape_lobe_type` is folded into "
        "`leaf_shape_lobing`. Dirr has no per-species bark column in this export, so "
        "missing fields (bark texture and, where Dirr is silent, margins/lobing) are "
        "filled from GoBotany and the row is labelled `dirr+gobotany`. Species absent "
        "from both fall back to a single curated regional prior (only *Lagerstroemia "
        "indica* in this cohort). Genus-level *Carya* rows use the Dirr genus consensus "
        "plus GoBotany bark consensus.\n"
        "- **Crop triangulation.** The mask is cropped at the LeafSnap foreground "
        "bbox (scaled into the 700 px mask space) and measured for components, "
        "solidity, significant convexity defects, ellipse fill and aspect ratio. "
        "A compound-species crop with solidity > 0.78, <= 1 significant defect, "
        "one component and ellipse fill >= 0.40 is `sub_part` (isolated leaflet) "
        "and therefore `not_assessable` - never a compound training example.\n"
        "- **Regional photos.** There is no ground-truth segmentation mask for iNat "
        "photos; the mask is an HSV green-subject estimate. Geometry is only trusted "
        "when the Stage 2 anatomy framing applies and one green component dominates "
        "(>= 45% of green area); otherwise the photo is `context_only`.\n"
        "- **Stage labels are zero-shot CLIP triage.** Three balanced prompts per "
        "stage (`openai/clip-vit-base-patch32`) score the frame; the per-stage "
        "mean softmax wins. The colour heuristic (brown dominance -> Stage 1, "
        "broad multi-region foliage -> Stage 3, bounded subject -> Stage 2) runs "
        "in parallel and its label is stored in `stage_evidence`, so the two "
        "independent votes and all raw statistics are auditable. This is a "
        "taxon-agnostic triage, not a trained organ classifier; `--no-clip` "
        "falls back to the heuristic alone.\n"
        "- **Tuckaway walk photos are unlabeled** (37 camera frames, no taxon "
        "metadata). They contribute stage distribution and pool context but are "
        "excluded from species-level counts.\n"
        "- **Network failures are recorded, not fatal.** Failed downloads appear in "
        "`triangulation_summary.json` and the run continues.\n"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-photos", type=int, default=180, help="regional pool cap")
    parser.add_argument("--per-species", type=int, default=10, help="max Henrico/Tuckaway photos per species")
    parser.add_argument("--workers", type=int, default=8, help="download workers")
    parser.add_argument("--offline", action="store_true", help="reuse cached images, no network")
    parser.add_argument("--no-clip", action="store_true", help="use the colour heuristic instead of zero-shot CLIP staging")
    parser.add_argument("--skip-pool", action="store_true", help="only rebuild the LeafSnap dataset")
    parser.add_argument("--skip-dataset", action="store_true", help="only rebuild the regional pool")
    args = parser.parse_args(argv)

    started = time.time()
    print("[1/6] resolving DeVo priors (Dirr -> GoBotany -> curated) ...")
    priors = PriorResolver()

    print("[2/6] querying Henrico + Tuckaway ...")
    henrico_photos, henrico_counts = query_henrico(args.per_species)
    tuckaway_photos, tuckaway_counts = query_tuckaway(args.per_species)
    shared = sorted(set(henrico_counts) & set(tuckaway_counts))
    combined = sorted(set(TARGET_SPECIES) | set(CARYA_SPECIES) | set(GENUS_GROUPS.values()) | set(tuckaway_counts))

    prior_summary = {name: priors.resolve(name) for name in combined}
    unresolved = [name for name, prior in prior_summary.items() if not prior.get("leaf_type")]

    print("[3/6] triangulating LeafSnap classification crops ...")
    if args.skip_dataset:
        triangulated: list[dict[str, Any]] = []
        gold: list[dict[str, Any]] = []
        crop_summary = {"skipped": True}
        dataset_counts = {"skipped": True}
    else:
        triangulated, gold, crop_summary = build_leafsnap_dataset(priors)
        dataset_counts = materialize_dataset(triangulated, gold)
        print(
            f"      gold={crop_summary['gold_total']} "
            f"hard_negatives={crop_summary['hard_negative_sub_part']} "
            f"crops={crop_summary['n_crops']}"
        )

    print("[4/6] building the regional pool ...")
    if args.skip_pool:
        pool_rows: list[dict[str, Any]] = []
        pool_summary = {"skipped": True}
    else:
        pool_rows, pool_summary = build_pool(
            priors, henrico_photos, tuckaway_photos,
            max_photos=args.max_photos, offline=args.offline, workers=args.workers,
            use_clip=not args.no_clip,
        )
        print(
            f"      downloaded={pool_summary['downloaded']} "
            f"failed={pool_summary['failed']} stages={pool_summary['stage_distribution']}"
        )

    print("[5/6] computing the Chordal Feature Buffer Index ...")
    stage_counts: dict[str, Counter] = defaultdict(Counter)
    for row in pool_rows:
        stage_counts[row["species"]][row["stage"]] += 1
    pool_species = sorted(
        (set(henrico_counts) | set(tuckaway_counts)
         | {row["species"] for row in pool_rows})
        - {"unidentified_tuckaway_walk"}
    )
    chordal_priors = {name: prior_summary.get(name) or priors.resolve(name) for name in pool_species}
    chordal = chordal_index(chordal_priors, stage_counts)

    stage_by_species_full = {
        name: dict(stage_counts.get(name, Counter()))
        for name in sorted(set(pool_species) | set(stage_counts))
    }

    print("[6/6] writing manifests and the summary ...")
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 1),
        "inputs": {
            "henrico_instance": HENRICO_ID,
            "tuckaway_instance": TUCKAWAY_ID,
            "tuckaway_walk_root": str(TUCKAWAY_PHYS),
            "tuckaway_qa_root": str(TUCKAWAY_QA),
            "feature_assertions": str(FEATURE_ASSERTIONS),
            "classification_manifest": str(CLASSIFICATION_MANIFEST),
        },
        "henrico": {
            "species_counts": henrico_counts,
            "photos_selected": len(henrico_photos),
        },
        "tuckaway": {
            "species_counts": tuckaway_counts,
            "photos_selected": len(tuckaway_photos),
            "walk_photos_available": len(tuckaway_walk_photos()),
        },
        "shared_species": shared,
        "combined_species": combined,
        "unresolved_priors": unresolved,
        "priors": {
            name: {
                "source": prior.get("source"),
                "leaf_type": prior.get("leaf_type"),
                "leaf_arrangement": prior.get("leaf_arrangement"),
                "leaf_margins": prior.get("leaf_margins"),
                "leaf_lobing": prior.get("leaf_lobing"),
                "bark_texture": prior.get("bark_texture"),
                "leaf_shape": prior.get("leaf_shape"),
            }
            for name, prior in prior_summary.items()
        },
        "leafsnap": crop_summary,
        "classification_v3": dataset_counts,
        "regional_pool": {**pool_summary, "stage_by_species_full": stage_by_species_full},
        "chordal_feature_buffer": chordal,
        "juglans_cinerea": {
            "hard_negatives_sub_part": crop_summary.get("hard_negative_by_species", {}).get("Juglans cinerea", 0)
            if isinstance(crop_summary, dict) else 0,
            "gold_whole_crops": sum(
                1 for row in triangulated
                if row["species"] == "Juglans cinerea"
                and row["triangulation_bucket"] == "gold_specialist_train"
            ),
        },
        "outputs": {
            "dataset": str(OUT_DATASET),
            "pool": str(OUT_POOL),
            "bench_manifest": str(OUT_POOL / "bench_triangulated_manifest.jsonl"),
            "summary": str(OUT_POOL / "triangulation_summary.json"),
            "markdown": str(HERE / "MID_ATLANTIC_TRIANGULATION.md"),
        },
    }

    if not args.skip_pool:
        write_jsonl(OUT_POOL / "pool_manifest.jsonl", pool_rows)
        bench_rows = []
        for row in triangulated:
            if row["triangulation_bucket"] in {"gold_specialist_train", "view_hard_negative"}:
                bench_rows.append(
                    {
                        "uid": row["uid"],
                        "dataset": "leafsnap_classification_crop",
                        "split": row["split"],
                        "species": row["species"],
                        "view_framing": row["view_framing"],
                        "devo_leaf_type": row["devo_leaf_type"],
                        "triangulation_bucket": row["triangulation_bucket"],
                        "bench_role": (
                            "positive" if row["triangulation_bucket"] == "gold_specialist_train"
                            else "hard_negative"
                        ),
                        "geometry": row.get("geometry"),
                    }
                )
        for row in pool_rows:
            bench_rows.append(
                {
                    "uid": row["uid"],
                    "dataset": "mid_atlantic_pool",
                    "split": "regional_pool",
                    "source": row["source"],
                    "species": row["species"],
                    "stage": row["stage"],
                    "view_framing": row["view_framing"],
                    "devo_leaf_type": row["devo_leaf_type"],
                    "triangulation_bucket": row["triangulation_bucket"],
                    "bench_role": (
                        "positive" if row["triangulation_bucket"] == "gold_regional_pool"
                        else "hard_negative" if row["triangulation_bucket"] == "view_hard_negative"
                        else "context"
                    ),
                    "image": row["image"],
                    "geometry": row.get("geometry"),
                }
            )
        write_jsonl(OUT_POOL / "bench_triangulated_manifest.jsonl", bench_rows)
        write_contact_sheet(pool_rows, OUT_POOL / "contact_sheet.jpg")

    write_json(OUT_POOL / "triangulation_summary.json", summary)
    markdown = build_markdown({**summary, "regional_pool": summary["regional_pool"], "leafsnap": crop_summary, "chordal": chordal, "henrico": summary["henrico"], "tuckaway": summary["tuckaway"]}, dataset_counts if isinstance(dataset_counts, dict) else {})
    (HERE / "MID_ATLANTIC_TRIANGULATION.md").write_text(markdown)

    headline = {
        "henrico_species": len(henrico_counts),
        "tuckaway_species": len(tuckaway_counts),
        "shared_species": len(shared),
        "combined_species": len(combined),
        "unresolved_priors": len(unresolved),
        "leafsnap_gold": crop_summary.get("gold_total") if isinstance(crop_summary, dict) else None,
        "leafsnap_hard_negatives": crop_summary.get("hard_negative_sub_part") if isinstance(crop_summary, dict) else None,
        "juglans_cinerea_hard_negatives": summary["juglans_cinerea"]["hard_negatives_sub_part"],
        "pool_photos_downloaded": pool_summary.get("downloaded"),
        "pool_download_failures": pool_summary.get("failed"),
        "stage_distribution": pool_summary.get("stage_distribution"),
        "stage_model": pool_summary.get("stage_model"),
        "runtime_seconds": summary["runtime_seconds"],
    }
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
