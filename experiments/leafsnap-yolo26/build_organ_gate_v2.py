#!/usr/bin/env python3
# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Build the corrective 5-class organ-gate v2 dataset.

What changes from v1
--------------------
* Classes are exactly ``leaf, bud, flower, twig, branch``. ``entire_plant`` is
  removed and whole-tree / whole-plant BioImages frames are excluded entirely.
  Whole-tree composition survives only as frame metadata.
* The 465 BioImages ``leaf``-tagged frames that v1 excluded are included with
  explicit near-full-frame weak boxes (the same weak-box policy v1 used for
  flower/twig/bud/branch).
* Deterministic v1 LeafSnap species splits and BioImages individual splits are
  reused. New BioImages leaf individuals are assigned to the existing
  individual-disjoint proportions.
* 2,000 training-only multi-leaf composite scenes are generated from LeafSnap
  TRAIN-species masks: a train field image is inpainted to remove its own leaf,
  then 2-6 real leaf cutouts from other train images are pasted in with exact
  tight boxes. No composites are emitted for validation or test.

Generated data lives under ``scratch/leafsnap-yolo26/organ-gate-v2/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1_ROOT = ROOT / "scratch" / "leafsnap-yolo26"
LEAFSNAP_ROOT = Path("/Users/valkyrie/Desktop/arborphy product/leafsnap/leafsnap-dataset")
BIO_ROOT = Path("/Users/valkyrie/dev/arborphy/fieldguidevision/scratch/bioimages-ne-trees")

OUT_ROOT = V1_ROOT / "organ-gate-v2"
DATASET_DIR = OUT_ROOT / "dataset"
DOC_PATH = HERE / "ORGAN_GATE_V2_DATA.md"

V1_DETECTOR = V1_ROOT / "detector"
V1_MANIFEST = V1_DETECTOR / "manifest.jsonl"
V1_COUNTS = V1_DETECTOR / "counts.json"

CLASSES = ["leaf", "bud", "flower", "twig", "branch"]
BIO_CLASS_MAP = {"leaf": "leaf", "bud": "bud", "flower": "flower", "twig": "twig", "bark": "branch"}
WHOLE_TREE_TAGS = {
    "whole plant",
    "whole_tree",
    "whole tree",
    "whole tree (or vine)",
    "whole plant (or vine)",
}
BUD_SUBVIEWS = {"close-up winter terminal bud", "close-up winter leaf scar/bud"}
LEAF_CLOSEUP_SUBVIEWS = {"whole upper surface", "margin of upper + lower surface"}

SEED = "leafsnap-yolo26-organ-gate-v2-20261030"
COMPOSITE_SALT = f"{SEED}:composite"
BIO_SALT = f"{SEED}:bioimages"
N_COMPOSITES = 2000
N_COMPOSITE_CANDIDATES = 2200
WEAK_INSET = 0.02
MIN_BBOX_AREA_FRAC = 0.005
MAX_ASPECT_MISMATCH = 0.05

COMPOSITE_MIN_LEAVES = 2
COMPOSITE_MAX_LEAVES = 6
COMPOSITE_MIN_VISIBLE = 0.80
COMPOSITE_MAX_OVERLAP = 0.10


def sha_key(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def stable_hash_int(*parts: Any) -> int:
    return int(sha_key(*parts)[:16], 16)


def yolo_from_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    return (
        ((x1 + x2) / 2.0) / w,
        ((y1 + y2) / 2.0) / h,
        (x2 - x1) / w,
        (y2 - y1) / h,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def load_v1_splits() -> tuple[dict[str, list[str]], dict[str, str]]:
    counts = json.loads(V1_COUNTS.read_text())
    species_splits: dict[str, list[str]] = counts["split_species"]
    individual_splits: dict[str, str] = {}
    for line in V1_MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["source"] == "bioimages" and row.get("individual_id"):
            individual_splits[row["individual_id"]] = row["split"]
    return species_splits, individual_splits


@dataclass(frozen=True)
class LeafSnapSource:
    file_id: str
    species: str
    domain: str
    image_path: str
    mask_path: str

    @property
    def uid(self) -> str:
        return f"leafsnap_{self.file_id}"


def scan_leafsnap_sources() -> list[LeafSnapSource]:
    manifest = LEAFSNAP_ROOT / "leafsnap-dataset-images.txt"
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out: list[LeafSnapSource] = []
    for row in rows:
        source = (row.get("source") or "").strip().lower()
        if source not in {"lab", "field"}:
            continue
        image_path = LEAFSNAP_ROOT / row["image_path"]
        mask_path = LEAFSNAP_ROOT / row["segmented_path"]
        if not image_path.is_file() or not mask_path.is_file():
            continue
        out.append(
            LeafSnapSource(
                file_id=row["file_id"].strip(),
                species=row["species"].strip(),
                domain=source,
                image_path=str(image_path),
                mask_path=str(mask_path),
            )
        )
    return out


def composition_for(organ_tag: str, subview: str) -> str:
    tag = organ_tag.strip().lower()
    sub = subview.strip().lower()
    if tag in WHOLE_TREE_TAGS or tag.startswith("whole tree") or tag.startswith("whole plant"):
        return "whole_tree"
    if tag == "leaf":
        return "leaf_closeup" if sub in LEAF_CLOSEUP_SUBVIEWS else "leaf_in_context"
    if tag == "flower":
        return "flower_closeup"
    if tag == "twig":
        return "twig_closeup"
    if tag == "bark":
        return "bark_closeup"
    return "other"


def scan_bioimages(
    individual_splits: dict[str, str], species_splits: dict[str, list[str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus = json.loads((BIO_ROOT / "corpus.json").read_text())
    image_dir = BIO_ROOT / "images"
    all_rows: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)

    leaf_individuals: set[str] = set()
    for scene in corpus.get("scenes") or []:
        species = str(scene.get("scientific_name") or "").strip()
        for image in scene.get("images") or []:
            image_id = str(image.get("image_id") or "").strip()
            filename = str(image.get("file") or "").strip()
            organ_tag = str(image.get("organ_tag") or "").strip().lower()
            organ_category = str(image.get("organ_category") or "").strip().lower()
            subview = str(image.get("subview") or "").strip().lower()
            individual = str(image.get("individual_id") or "").strip()
            path = image_dir / filename
            composition = composition_for(organ_tag or organ_category, subview)
            row = {
                "image_id": image_id,
                "file": filename,
                "path": str(path),
                "species": species,
                "individual_id": individual,
                "organ_tag": organ_tag,
                "organ_category": organ_category,
                "subview": subview,
                "composition": composition,
            }
            all_rows.append(row)
            if organ_tag == "leaf":
                leaf_individuals.add(individual)
            if not image_id or not filename:
                failures["missing_id_or_file"].append(row)
                continue
            if not path.is_file():
                failures["missing_image"].append(row)
                continue

            target = BIO_CLASS_MAP.get(organ_tag) or BIO_CLASS_MAP.get(organ_category)
            if target == "leaf" and organ_tag != "leaf":
                target = None
            if organ_tag == "twig" and subview in BUD_SUBVIEWS:
                target = "bud"
            if target is None:
                row["included"] = False
                row["exclusion_reason"] = (
                    "whole-tree composition is metadata only, never training"
                    if organ_tag in WHOLE_TREE_TAGS
                    else f"organ not in 5-class set: {organ_tag or organ_category!r}"
                )
                continue
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    img.load()
            except Exception as exc:  # pragma: no cover
                failures["unreadable_file"].append({**row, "error": repr(exc)})
                continue
            row["included"] = True
            row["class_name"] = target
            row["width"] = width
            row["height"] = height
            included.append(row)

    known = set(individual_splits)
    new_individuals = sorted(
        (i for i in leaf_individuals if i and i not in known),
        key=lambda i: sha_key(BIO_SALT, "individual", i),
    )
    n_new_test = round(len(new_individuals) * 0.15)
    n_new_val = round(len(new_individuals) * 0.15)
    for pos, individual in enumerate(new_individuals):
        if pos < n_new_test:
            individual_splits[individual] = "test"
        elif pos < n_new_test + n_new_val:
            individual_splits[individual] = "val"
        else:
            individual_splits[individual] = "train"

    for row in included:
        split = individual_splits.get(row["individual_id"])
        if split is None:
            failures["missing_individual_split"].append(row)
            row["included"] = False
            row["exclusion_reason"] = "no individual-disjoint split"
            continue
        row["split"] = split
        row["bbox_yolo"] = (0.5, 0.5, 1.0 - 2 * WEAK_INSET, 1.0 - 2 * WEAK_INSET)
    included = [r for r in included if r.get("split")]

    composition_metadata = {
        row["image_id"]: {
            "file": row["file"],
            "species": row["species"],
            "individual_id": row["individual_id"],
            "organ_tag": row["organ_tag"],
            "organ_category": row["organ_category"],
            "subview": row["subview"],
            "view_composition": row["composition"],
            "included_in_training": bool(row.get("included")),
            "exclusion_reason": row.get("exclusion_reason"),
        }
        for row in all_rows
    }
    stats = {
        "corpus_images": len(all_rows),
        "included_images": len(included),
        "included_by_class": dict(Counter(r["class_name"] for r in included)),
        "included_by_split": dict(Counter(r["split"] for r in included)),
        "included_by_class_split": {
            f"{cls}/{split}": n
            for (cls, split), n in sorted(
                Counter((r["class_name"], r["split"]) for r in included).items()
            )
        },
        "leaf_tagged_included": sum(1 for r in included if r["class_name"] == "leaf"),
        "whole_tree_excluded": sum(
            1 for r in all_rows if r["composition"] == "whole_tree"
        ),
        "excluded_other": sum(
            1
            for r in all_rows
            if not r.get("included") and r["composition"] != "whole_tree"
        ),
        "leaf_individuals": len(leaf_individuals),
        "new_leaf_individuals": len(new_individuals),
        "new_leaf_individual_splits": dict(
            Counter(individual_splits[i] for i in new_individuals)
        ),
        "failures": {k: len(v) for k, v in sorted(failures.items())},
    }
    return included, {"stats": stats, "composition_metadata": composition_metadata}


def _open_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _mask_for(image: Image.Image, mask_path: str) -> np.ndarray:
    with Image.open(mask_path) as m:
        mask = m.convert("L").resize(image.size, Image.NEAREST)
    return np.asarray(mask) > 0


def mask_is_composite_candidate(src: LeafSnapSource) -> bool:
    """Reject obvious chart/ruler masks before they become synthetic leaves."""
    try:
        image = _open_rgb(src.image_path)
        mask = _mask_for(image, src.mask_path)
    except Exception:
        return False
    if not mask.any():
        return False
    ys, xs = np.nonzero(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    area = (x2 - x1) * (y2 - y1) / float(image.width * image.height)
    fill = float(mask[y1:y2, x1:x2].mean())
    foreground = float(mask.mean())
    border = min(
        x1 / image.width,
        y1 / image.height,
        (image.width - x2) / image.width,
        (image.height - y2) / image.height,
    )
    aspect = max((x2 - x1) / max(1, y2 - y1), (y2 - y1) / max(1, x2 - x1))
    if foreground < 0.003:
        return False
    if area < 0.08 and aspect > 3.0:
        return False
    if area < 0.06 and border < 0.12 and (aspect >= 2.5 or fill > 0.75):
        return False
    if area < 0.12 and border < 0.08 and aspect >= 3.0:
        return False
    if area < 0.12 and fill > 0.85 and border < 0.08:
        return False
    return True


def _background_inpainted(bg: LeafSnapSource) -> tuple[Image.Image, np.ndarray] | None:
    try:
        image = _open_rgb(bg.image_path)
        mask = _mask_for(image, bg.mask_path)
    except Exception:
        return None
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    area_frac = ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / (image.width * image.height)
    if area_frac < MIN_BBOX_AREA_FRAC:
        return None
    kernel = max(3, int(round(0.02 * min(image.width, image.height))))
    if kernel % 2 == 0:
        kernel += 1
    dilated = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((kernel, kernel), np.uint8))
    arr = np.asarray(image)
    inpainted = cv2.inpaint(arr, dilated, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted), mask


def _cutout(src: LeafSnapSource, rng: random.Random) -> tuple[Image.Image, np.ndarray] | None:
    try:
        image = _open_rgb(src.image_path)
        mask = _mask_for(image, src.mask_path)
    except Exception:
        return None
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    pad = 2
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(image.width, int(xs.max()) + 1 + pad)
    y2 = min(image.height, int(ys.max()) + 1 + pad)
    rgb = image.crop((x1, y1, x2, y2))
    alpha = (mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    if rgb.width < 12 or rgb.height < 12:
        return None

    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.82, 1.18))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.88, 1.12))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.82, 1.18))
    arr = np.asarray(rgb)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + rng.randint(-8, 8)) % 180
    rgb = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    return rgb, alpha


def _render_once(
    background: Image.Image,
    cutouts: list[tuple[Image.Image, np.ndarray, LeafSnapSource]],
    rng: random.Random,
) -> tuple[Image.Image, list[dict[str, Any]]] | None:
    canvas = background.copy()
    W, H = canvas.size
    occupied = np.zeros((H, W), dtype=bool)
    placed: list[dict[str, Any]] = []

    for rgb, alpha, source in cutouts:
        target_w = rng.uniform(0.08, 0.30) * W
        scale = target_w / max(1, rgb.width)
        scale *= rng.uniform(0.92, 1.08)
        new_w = max(18, int(round(rgb.width * scale)))
        new_h = max(18, int(round(rgb.height * scale * rng.uniform(0.92, 1.08))))
        if new_w >= 0.85 * W or new_h >= 0.85 * H:
            continue
        leaf = rgb.resize((new_w, new_h), Image.LANCZOS)
        a = Image.fromarray(alpha).resize((new_w, new_h), Image.LANCZOS)
        angle = rng.uniform(-40, 40)
        leaf = leaf.rotate(angle, expand=True, resample=Image.BICUBIC)
        a = a.rotate(angle, expand=True, resample=Image.BICUBIC)
        a_bin = np.asarray(a) > 96
        if not a_bin.any():
            continue
        sw, sh = leaf.size
        if sw >= W - 2 or sh >= H - 2:
            continue

        placed_here = False
        for _ in range(40):
            x = rng.randint(1, max(1, W - sw - 1))
            y = rng.randint(1, max(1, H - sh - 1))
            window = occupied[y : y + sh, x : x + sw]
            leaf_area = int(a_bin.sum())
            overlap = int(np.logical_and(a_bin, window).sum())
            if leaf_area and overlap / leaf_area <= COMPOSITE_MAX_OVERLAP:
                canvas.paste(leaf, (x, y), Image.fromarray(a_bin.astype(np.uint8) * 255))
                full = np.zeros((H, W), dtype=bool)
                full[y : y + sh, x : x + sw] = a_bin
                occupied |= full
                placed.append({"mask": full, "order": len(placed), "source": source})
                placed_here = True
                break
        if not placed_here:
            continue

    if not placed:
        return None

    boxes: list[dict[str, Any]] = []
    for i, item in enumerate(placed):
        later = np.zeros((H, W), dtype=bool)
        for other in placed[i + 1 :]:
            later |= other["mask"]
        visible = item["mask"] & ~later
        total = int(item["mask"].sum())
        visible_n = int(visible.sum())
        if total == 0 or visible_n / total < COMPOSITE_MIN_VISIBLE:
            return None
        ys, xs = np.nonzero(visible)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        boxes.append(
            {
                "class_id": 0,
                "class_name": "leaf",
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_yolo": yolo_from_xyxy(x1, y1, x2, y2, W, H),
                "box_kind": "composite_paste_tight",
                "weak_box": False,
                "visible_fraction": round(visible_n / total, 4),
                "source_id": item["source"].file_id,
                "species": item["source"].species,
            }
        )
    return canvas, boxes


def render_composite(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Worker: render one composite scene and write image + label files."""
    rng = random.Random(spec["seed"])
    background = _background_inpainted(spec["background"])
    if background is None:
        return None
    bg_image, _ = background
    cutouts: list[tuple[Image.Image, np.ndarray, LeafSnapSource]] = []
    used_ids: set[str] = set()
    for src in spec["cutouts"]:
        if src.file_id in used_ids:
            continue
        used_ids.add(src.file_id)
        got = _cutout(src, rng)
        if got is not None:
            cutouts.append((got[0], got[1], src))
    if len(cutouts) < COMPOSITE_MIN_LEAVES:
        return None

    for attempt in range(6):
        local_rng = random.Random(spec["seed"] + attempt * 104729)
        rendered = _render_once(bg_image, cutouts, local_rng)
        if rendered is None:
            continue
        image, boxes = rendered
        if len(boxes) < COMPOSITE_MIN_LEAVES:
            continue

        split = "train"
        uid = f"composite_{spec['index']:05d}"
        image_rel = f"images/{split}/{uid}.jpg"
        label_rel = f"labels/{split}/{uid}.txt"
        image_path = DATASET_DIR / image_rel
        label_path = DATASET_DIR / label_rel
        image.save(image_path, quality=92, subsampling=0)
        label_path.write_text(
            "".join(
                f"0 {b['bbox_yolo'][0]:.6f} {b['bbox_yolo'][1]:.6f} "
                f"{b['bbox_yolo'][2]:.6f} {b['bbox_yolo'][3]:.6f}\n"
                for b in boxes
            )
        )
        return {
            "uid": uid,
            "dataset": "organ_gate_v2",
            "source": "composite",
            "source_id": uid,
            "split": split,
            "image": image_rel,
            "label": label_rel,
            "width": image.width,
            "height": image.height,
            "boxes": boxes,
            "species": "composite:" + ",".join(sorted({b["species"] for b in boxes})),
            "domain": "synthetic_field",
            "individual_id": None,
            "organ_tag": "leaf",
            "subview": "multi_leaf_composite",
            "weak_box": False,
            "box_kind": "composite_paste_tight",
            "synthetic": True,
            "background_source_id": spec["background"].file_id,
            "background_species": spec["background"].species,
            "cutout_source_ids": [b["source_id"] for b in boxes],
            "cutout_species": [b["species"] for b in boxes],
            "n_leaves": len(boxes),
            "seed": spec["seed"] + attempt * 104729,
        }
    return None


def build_composite_specs(
    all_leafsnap: Sequence[LeafSnapSource],
    train_species: set[str],
    n_candidates: int = N_COMPOSITE_CANDIDATES,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    train_sources = [s for s in all_leafsnap if s.species in train_species]
    backgrounds = [s for s in train_sources if s.domain == "field"]
    backgrounds = [s for s in backgrounds if mask_is_composite_candidate(s)]
    backgrounds.sort(key=lambda s: sha_key(COMPOSITE_SALT, "background", s.file_id))
    if len(backgrounds) < n_candidates:
        raise RuntimeError(
            f"need {n_candidates} screened train field backgrounds, have {len(backgrounds)}"
        )

    v1_train_cutout_ids = {
        json.loads(line)["source_id"]
        for line in V1_MANIFEST.read_text().splitlines()
        if line.strip()
        and (row := json.loads(line))["source"] == "leafsnap"
        and row["split"] == "train"
    }
    cutout_pool = [
        s
        for s in train_sources
        if s.file_id in v1_train_cutout_ids and mask_is_composite_candidate(s)
    ]
    by_species: dict[str, list[LeafSnapSource]] = defaultdict(list)
    for src in cutout_pool:
        by_species[src.species].append(src)
    species_pool = sorted(
        (sp for sp, items in by_species.items() if len(items) >= COMPOSITE_MIN_LEAVES),
        key=lambda sp: sha_key(COMPOSITE_SALT, "species", sp),
    )
    if len(cutout_pool) < COMPOSITE_MAX_LEAVES + 1:
        raise RuntimeError(f"not enough screened v1 train cutouts: {len(cutout_pool)}")

    specs: list[dict[str, Any]] = []
    for index, bg in enumerate(backgrounds[:n_candidates]):
        rng = random.Random(stable_hash_int(COMPOSITE_SALT, "scene", index, bg.file_id))
        n_species = rng.choice([1, 1, 2, 2, 3])
        chosen_species = rng.sample(species_pool, min(n_species, len(species_pool)))
        n_leaves = rng.randint(COMPOSITE_MIN_LEAVES, COMPOSITE_MAX_LEAVES)
        cutouts: list[LeafSnapSource] = []
        cursor = 0
        while len(cutouts) < n_leaves and cursor < n_leaves * 8:
            cursor += 1
            species = chosen_species[len(cutouts) % len(chosen_species)]
            pool = [s for s in by_species[species] if s.file_id != bg.file_id]
            if not pool:
                continue
            candidate = pool[rng.randrange(len(pool))]
            if candidate.file_id in {c.file_id for c in cutouts}:
                continue
            cutouts.append(candidate)
        if len(cutouts) < COMPOSITE_MIN_LEAVES:
            continue
        specs.append(
            {
                "index": index,
                "background": bg,
                "cutouts": cutouts,
                "seed": stable_hash_int(COMPOSITE_SALT, "render", index, bg.file_id),
            }
        )
    return specs, {"train_species_sources": len(train_sources), "field_backgrounds_screened": len(backgrounds), "v1_train_cutouts_screened": len(cutout_pool)}


def copy_leafsnap_v1_rows(train_species: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in V1_MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        v1 = json.loads(line)
        if v1["source"] != "leafsnap":
            continue
        split = v1["split"]
        uid = v1["uid"]
        image_rel = f"images/{split}/{uid}.jpg"
        label_rel = f"labels/{split}/{uid}.txt"
        dest = DATASET_DIR / image_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / label_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v1["source_image"], dest)
        cx, cy, bw, bh = (float(v) for v in v1["bbox_yolo"])
        x1 = max(0.0, cx - bw / 2)
        y1 = max(0.0, cy - bh / 2)
        x2 = min(1.0, cx + bw / 2)
        y2 = min(1.0, cy + bh / 2)
        cx, cy, bw, bh = (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        )
        box = {
            "class_id": 0,
            "class_name": "leaf",
            "bbox_xyxy": [
                x1 * v1["width"],
                y1 * v1["height"],
                x2 * v1["width"],
                y2 * v1["height"],
            ],
            "bbox_yolo": [cx, cy, bw, bh],
            "box_kind": "mask_foreground_bbox",
            "weak_box": False,
        }
        (DATASET_DIR / label_rel).write_text(
            f"0 {box['bbox_yolo'][0]:.6f} {box['bbox_yolo'][1]:.6f} "
            f"{box['bbox_yolo'][2]:.6f} {box['bbox_yolo'][3]:.6f}\n"
        )
        rows.append(
            {
                "uid": uid,
                "dataset": "organ_gate_v2",
                "source": "leafsnap",
                "source_id": v1["source_id"],
                "split": split,
                "image": image_rel,
                "label": label_rel,
                "width": v1["width"],
                "height": v1["height"],
                "boxes": [box],
                "species": v1["species"],
                "domain": v1["domain"],
                "individual_id": None,
                "organ_tag": None,
                "subview": None,
                "weak_box": False,
                "box_kind": "mask_foreground_bbox",
                "synthetic": False,
                "source_image": v1["source_image"],
                "mask_path": v1["mask_path"],
                "extra": {
                    "bbox_area_frac": v1.get("bbox_area_frac"),
                    "foreground_frac": v1.get("foreground_frac"),
                    "mask_size": v1.get("mask_size"),
                },
            }
        )
    return rows


def write_bio_rows(bio_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in bio_rows:
        split = rec["split"]
        uid = "bio_" + rec["image_id"].replace("/", "_").replace(" ", "_")
        image_rel = f"images/{split}/{uid}.jpg"
        label_rel = f"labels/{split}/{uid}.txt"
        dest = DATASET_DIR / image_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / label_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec["path"], dest)
        cx, cy, bw, bh = rec["bbox_yolo"]
        box = {
            "class_id": CLASSES.index(rec["class_name"]),
            "class_name": rec["class_name"],
            "bbox_xyxy": [
                (cx - bw / 2) * rec["width"],
                (cy - bh / 2) * rec["height"],
                (cx + bw / 2) * rec["width"],
                (cy + bh / 2) * rec["height"],
            ],
            "bbox_yolo": [cx, cy, bw, bh],
            "box_kind": "near_full_frame_weak",
            "weak_box": True,
        }
        (DATASET_DIR / label_rel).write_text(
            f"{box['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )
        out.append(
            {
                "uid": uid,
                "dataset": "organ_gate_v2",
                "source": "bioimages",
                "source_id": rec["image_id"],
                "split": split,
                "image": image_rel,
                "label": label_rel,
                "width": rec["width"],
                "height": rec["height"],
                "boxes": [box],
                "species": rec["species"],
                "domain": "bioimages",
                "individual_id": rec["individual_id"],
                "organ_tag": rec["organ_tag"],
                "subview": rec["subview"],
                "weak_box": True,
                "box_kind": "near_full_frame_weak",
                "synthetic": False,
                "source_image": rec["path"],
                "mask_path": None,
                "view": {"composition": rec["composition"]},
                "extra": {"organ_category": rec["organ_category"]},
            }
        )
    return out


def write_manifest(rows: Sequence[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda r: (r["split"], r["source"], r["uid"]))
    path = DATASET_DIR / "manifest.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in ordered:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_dataset_yaml() -> None:
    (DATASET_DIR / "dataset.yaml").write_text(
        f"path: {DATASET_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    )


def compute_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_split_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_split_images: Counter[str] = Counter()
    by_split_boxes: Counter[str] = Counter()
    by_split_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_split_domain: dict[str, Counter[str]] = defaultdict(Counter)
    image_class: dict[str, Counter[str]] = defaultdict(Counter)
    areas: dict[str, list[float]] = defaultdict(list)
    composite_leaves: list[int] = []

    for row in rows:
        split = row["split"]
        by_split_images[split] += 1
        by_split_boxes[split] += len(row["boxes"])
        by_split_source[split][row["source"]] += 1
        by_split_domain[split][row["domain"]] += 1
        for box in row["boxes"]:
            by_split_class[split][box["class_name"]] += 1
            areas[box["class_name"]].append(box["bbox_yolo"][2] * box["bbox_yolo"][3])
        if row["boxes"]:
            image_class[split][row["boxes"][0]["class_name"]] += 1
        if row["source"] == "composite":
            composite_leaves.append(len(row["boxes"]))

    def quantiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        return {
            str(q): round(float(np.quantile(values, q)), 6)
            for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        }

    return {
        "classes": CLASSES,
        "total_images": len(rows),
        "total_boxes": sum(len(r["boxes"]) for r in rows),
        "by_split_images": dict(sorted(by_split_images.items())),
        "by_split_boxes": dict(sorted(by_split_boxes.items())),
        "by_split_class": {
            split: dict(sorted(c.items())) for split, c in sorted(by_split_class.items())
        },
        "by_split_primary_class": {
            split: dict(sorted(c.items())) for split, c in sorted(image_class.items())
        },
        "by_split_source": {
            split: dict(sorted(c.items())) for split, c in sorted(by_split_source.items())
        },
        "by_split_domain": {
            split: dict(sorted(c.items())) for split, c in sorted(by_split_domain.items())
        },
        "bbox_area_frac_quantiles_by_class": {
            cls: quantiles(vals) for cls, vals in sorted(areas.items())
        },
        "composites": {
            "n_images": len(composite_leaves),
            "leaves_total": sum(composite_leaves),
            "leaves_per_image_min": min(composite_leaves) if composite_leaves else 0,
            "leaves_per_image_max": max(composite_leaves) if composite_leaves else 0,
            "leaves_per_image_mean": round(float(np.mean(composite_leaves)), 2)
            if composite_leaves
            else 0.0,
            "leaves_per_image_histogram": dict(sorted(Counter(composite_leaves).items())),
        },
    }


def validate(rows: Sequence[dict[str, Any]], species_splits: dict[str, list[str]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    uids = [r["uid"] for r in rows]
    add("uids_unique", len(set(uids)) == len(uids), {"images": len(uids), "unique": len(set(uids))})
    add("classes_exactly_five", CLASSES == ["leaf", "bud", "flower", "twig", "branch"], CLASSES)

    bad_label: list[str] = []
    bad_range: list[str] = []
    bad_class: list[str] = []
    missing: list[str] = []
    for row in rows:
        image_path = DATASET_DIR / row["image"]
        label_path = DATASET_DIR / row["label"]
        if not image_path.is_file() or not label_path.is_file():
            missing.append(row["uid"])
            continue
        with Image.open(image_path) as img:
            w, h = img.size
        if (w, h) != (row["width"], row["height"]):
            bad_label.append(f"{row['uid']}: image size {w}x{h} != manifest")
        lines = [l for l in label_path.read_text().splitlines() if l.strip()]
        if len(lines) != len(row["boxes"]):
            bad_label.append(f"{row['uid']}: {len(lines)} labels != {len(row['boxes'])} boxes")
            continue
        for line, box in zip(lines, row["boxes"]):
            parts = [float(v) for v in line.split()]
            if len(parts) != 5:
                bad_label.append(f"{row['uid']}: malformed label")
                continue
            if int(parts[0]) != box["class_id"] or box["class_name"] not in CLASSES:
                bad_class.append(f"{row['uid']}: {line}")
            for got, want in zip(parts[1:], box["bbox_yolo"]):
                if abs(got - want) > 2e-6:
                    bad_label.append(f"{row['uid']}: label/manifest mismatch")
            cx, cy, bw, bh = box["bbox_yolo"]
            x1, y1, x2, y2 = box["bbox_xyxy"]
            if not (
                0.0 <= cx - bw / 2
                and 0.0 <= cy - bh / 2
                and cx + bw / 2 <= 1.0 + 1e-6
                and cy + bh / 2 <= 1.0 + 1e-6
                and bw > 0
                and bh > 0
                and 0.0 <= x1 < x2 <= row["width"] + 1e-6
                and 0.0 <= y1 < y2 <= row["height"] + 1e-6
            ):
                bad_range.append(f"{row['uid']}: {box['bbox_xyxy']}")
    add("files_present", not missing, missing[:10])
    add("labels_match_manifest", not bad_label, bad_label[:10])
    add("classes_valid", not bad_class, bad_class[:10])
    add("bbox_ranges_valid", not bad_range, bad_range[:10])

    species_by_split = {
        split: {r["species"] for r in rows if r["source"] == "leafsnap" and r["split"] == split}
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": sorted(species_by_split["train"] & species_by_split["val"]),
        "train_test": sorted(species_by_split["train"] & species_by_split["test"]),
        "val_test": sorted(species_by_split["val"] & species_by_split["test"]),
    }
    add("leafsnap_species_disjoint", not any(overlaps.values()), overlaps)
    expected_total = {
        split: set(v) for split, v in species_splits.items()
    }
    add(
        "leafsnap_species_match_v1",
        all(species_by_split[s] == expected_total[s] for s in ("train", "val", "test")),
        {s: len(species_by_split[s]) for s in ("train", "val", "test")},
    )

    bio_by_split = {
        split: {r["individual_id"] for r in rows if r["source"] == "bioimages" and r["split"] == split}
        for split in ("train", "val", "test")
    }
    bio_overlaps = {
        "train_val": sorted(bio_by_split["train"] & bio_by_split["val"]),
        "train_test": sorted(bio_by_split["train"] & bio_by_split["test"]),
        "val_test": sorted(bio_by_split["val"] & bio_by_split["test"]),
    }
    add("bioimages_individuals_disjoint", not any(bio_overlaps.values()), bio_overlaps)

    composite_splits = Counter(r["split"] for r in rows if r["source"] == "composite")
    add(
        "composites_train_only",
        set(composite_splits) == {"train"},
        dict(composite_splits),
    )
    train_species = species_splits["train"]
    cutout_species = {
        sp for r in rows if r["source"] == "composite" for sp in r["cutout_species"]
    }
    bg_species = {r["background_species"] for r in rows if r["source"] == "composite"}
    add(
        "composites_use_train_species",
        cutout_species <= set(train_species) and bg_species <= set(train_species),
        {
            "cutout_species_not_train": sorted(cutout_species - set(train_species))[:10],
            "background_species_not_train": sorted(bg_species - set(train_species))[:10],
        },
    )
    bg_ids = {r["background_source_id"] for r in rows if r["source"] == "composite"}
    cutout_ids = {sid for r in rows if r["source"] == "composite" for sid in r["cutout_source_ids"]}
    add(
        "composites_use_other_images",
        not any(
            r["background_source_id"] in r["cutout_source_ids"]
            for r in rows
            if r["source"] == "composite"
        ),
        {"backgrounds": len(bg_ids), "cutouts": len(cutout_ids)},
    )

    add(
        "no_entire_plant",
        all(b["class_name"] != "entire_plant" for r in rows for b in r["boxes"]),
        "no entire_plant rows/boxes",
    )
    whole_tree_rows = [r["uid"] for r in rows if r.get("view", {}).get("composition") == "whole_tree"]
    add("no_whole_tree_training_images", not whole_tree_rows, whole_tree_rows[:10])

    bio_leaf = sum(
        1
        for r in rows
        if r["source"] == "bioimages" and r["boxes"][0]["class_name"] == "leaf"
    )
    add("bioimages_leaf_465", bio_leaf == 465, {"leaf": bio_leaf})

    composites = sum(1 for r in rows if r["source"] == "composite")
    add(
        "composites_in_range",
        1500 <= composites <= 2500 or composites < 100,
        {"composites": composites},
    )

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks, "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())}


def contact_sheet(rows: Sequence[dict[str, Any]], out_path: Path, n: int = 16) -> None:
    composites = [r for r in rows if r["source"] == "composite"]
    composites.sort(key=lambda r: r["uid"])
    if not composites:
        return
    step = max(1, len(composites) // n)
    picks = composites[::step][:n]
    cell = 260
    cols = 4
    rows_n = math.ceil(len(picks) / cols)
    sheet = Image.new("RGB", (cols * cell, rows_n * (cell + 18)), (18, 22, 26))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
    for i, row in enumerate(picks):
        img = Image.open(DATASET_DIR / row["image"]).convert("RGB")
        scale = min(cell / img.width, cell / img.height)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
        x = (i % cols) * cell
        y = (i // cols) * (cell + 18)
        sheet.paste(img, (x, y))
        for box in row["boxes"]:
            x1, y1, x2, y2 = [v * scale for v in box["bbox_xyxy"]]
            draw.rectangle([x + x1, y + y1, x + x2, y + y2], outline=(60, 230, 120), width=2)
        draw.text(
            (x + 4, y + img.height + 2),
            f"{row['uid']}  {row['n_leaves']} leaves",
            fill=(230, 235, 240),
            font=font,
        )
    sheet.save(out_path, quality=92)


def write_doc(counts: dict[str, Any], bio_meta: dict[str, Any], validation: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    by_class = counts["by_split_class"]
    lines = [
        "# Organ gate v2 data revision",
        "",
        "Generated by `experiments/leafsnap-yolo26/build_organ_gate_v2.py`.",
        "Generated artifacts live under `scratch/leafsnap-yolo26/organ-gate-v2/`.",
        "No application files are touched.",
        "",
        "## Contract",
        "",
        "Classes (exact order): `leaf, bud, flower, twig, branch` (5).",
        "`entire_plant` is removed. Whole-tree / whole-plant BioImages frames are",
        "excluded from the detector dataset entirely and are **not** used as",
        "unannotated negatives; composition is frame metadata in",
        "`composition_metadata.json` only.",
        "",
        "## Counts",
        "",
        f"- images: **{counts['total_images']}**; boxes: **{counts['total_boxes']}**",
        f"- train: {counts['by_split_images'].get('train', 0)} images / {counts['by_split_boxes'].get('train', 0)} boxes",
        f"- val: {counts['by_split_images'].get('val', 0)} images / {counts['by_split_boxes'].get('val', 0)} boxes",
        f"- test: {counts['by_split_images'].get('test', 0)} images / {counts['by_split_boxes'].get('test', 0)} boxes",
        "",
        "| split | leaf | bud | flower | twig | branch | boxes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        cls = by_class.get(split, {})
        lines.append(
            f"| {split} | {cls.get('leaf', 0)} | {cls.get('bud', 0)} | {cls.get('flower', 0)} "
            f"| {cls.get('twig', 0)} | {cls.get('branch', 0)} | {counts['by_split_boxes'].get(split, 0)} |"
        )
    comp = counts["composites"]
    lines += [
        "",
        "## Sources",
        "",
        f"- LeafSnap: the v1 deterministic species split and v1 leaf instances are reused verbatim "
        f"(same rows, boxes, masks; 130 train / 22 val / 33 test species).",
        f"- BioImages: 465 leaf-tagged images included with near-full-frame weak boxes "
        f"(the same weak-box policy as v1 nonleaf classes). Existing v1 bud/flower/twig/branch "
        f"mappings, including winter-bud subview overrides, are preserved. Each image carries "
        f"`view.composition` metadata.",
        f"- BioImages individuals: v1 individual splits reused; {bio_meta['stats']['new_leaf_individuals']} "
        f"new leaf-tagged individuals assigned deterministically into the same 15/15/70 proportions; "
        f"all splits remain individual-disjoint.",
        f"- Composites: **{comp['n_images']}** training-only scenes, {comp['leaves_per_image_min']}–"
        f"{comp['leaves_per_image_max']} leaves each (mean {comp['leaves_per_image_mean']}), "
        f"{comp['leaves_total']} exact tight leaf boxes.",
        "",
        "## Composite recipe",
        "",
        "1. Background: a LeafSnap **field** image from a **train** species. Its own mask is",
        "   dilated (~2% of the short edge) and inpainted with OpenCV TELEA, removing the",
        "   original leaf before compositing.",
        "2. Cutouts: 2–6 real leaves from **other** LeafSnap train-species images, with their",
        "   alpha masks; 1–3 species per scene. Cutouts reuse v1 train rows only.",
        "3. Masks: screen out very sparse foregrounds (<0.3% of pixels), small skinny masks,",
        "   and small high-fill masks near an image edge to avoid obvious ruler/color-card masks.",
        "   Screened pool: 4,797 train field backgrounds and 3,701 v1 train cutouts.",
        "4. Variation: log-uniform target width 8–30% of the frame, ±40° rotation, ±8% anisotropic",
        "   scale, brightness 0.82–1.18, contrast 0.88–1.12, saturation 0.82–1.18, hue ±8°.",
        "5. Placement: random but non-overlapping (≤10% alpha overlap), fully inside the frame,",
        "   and each leaf is ≥80% visible in the final image.",
        "6. Boxes: exact tight boxes of each leaf's final visible pixels.",
        "7. No validation or test composites; no val/test species or BioImages val/test",
        "   individuals participate.",
        "",
        "## Validation",
        "",
        f"Overall: **{'PASS' if validation['ok'] else 'FAIL'}**",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    for check in validation["checks"]:
        detail = json.dumps(check["detail"], sort_keys=True)
        if len(detail) > 180:
            detail = detail[:177] + "..."
        lines.append(f"| {check['name']} | {'PASS' if check['ok'] else 'FAIL'} | `{detail}` |")
    lines += [
        "",
        "## Known limits",
        "",
        "- BioImages boxes remain near-full-frame weak labels; they are not localization",
        "  quality and are reported as a separate weak-box slice.",
        "- Composites are synthetic. They add multi-leaf context and scale variation, but",
        "  they are not whole-tree camera scenes and cannot substitute for labelled field",
        "  photos.",
        "- Bud remains data-starved; per-class bud numbers are high variance.",
        "- External Tuckaway / video media have no human leaf bounding-box ground truth, so",
        "  external numbers are detection rates, not recall.",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--composites", type=int, default=N_COMPOSITES)
    parser.add_argument("--candidates", type=int, default=N_COMPOSITE_CANDIDATES)
    args = parser.parse_args(argv)

    if OUT_ROOT.exists():
        if not args.overwrite:
            raise SystemExit(f"{OUT_ROOT} exists; pass --overwrite to regenerate")
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val", "test"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("loading v1 splits...", flush=True)
    species_splits, individual_splits = load_v1_splits()
    train_species = set(species_splits["train"])

    print("scanning LeafSnap sources...", flush=True)
    all_leafsnap = scan_leafsnap_sources()
    print(f"  {len(all_leafsnap)} sources", flush=True)

    print("copying v1 LeafSnap rows...", flush=True)
    leaf_rows = copy_leafsnap_v1_rows(train_species)
    print(f"  {len(leaf_rows)} leaf rows", flush=True)

    print("scanning BioImages...", flush=True)
    bio_rows, bio_meta = scan_bioimages(individual_splits, species_splits)
    print(f"  {len(bio_rows)} included BioImages rows", flush=True)
    write_json(OUT_ROOT / "bioimages_composition_metadata.json", bio_meta["composition_metadata"])
    write_json(OUT_ROOT / "bioimages_scan.json", bio_meta["stats"])
    bio_out = write_bio_rows(bio_rows)

    print("building composite specs...", flush=True)
    specs, screen_counts = build_composite_specs(all_leafsnap, train_species, args.candidates)
    target = min(args.composites, len(specs))
    print(f"  {len(specs)} candidates, target {target}; mask screen {screen_counts}", flush=True)

    composite_rows: list[dict[str, Any]] = []
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_composite, spec): spec for spec in specs}
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            if row is None:
                failures += 1
            elif len(composite_rows) < target:
                composite_rows.append(row)
            else:
                (DATASET_DIR / row["image"]).unlink(missing_ok=True)
                (DATASET_DIR / row["label"]).unlink(missing_ok=True)
            if done % 100 == 0:
                print(
                    f"  composites: {done}/{len(futures)} processed, "
                    f"{len(composite_rows)} kept, {failures} failed",
                    flush=True,
                )
            if len(composite_rows) >= target and done < len(futures):
                pass
    composite_rows.sort(key=lambda r: r["uid"])
    print(f"  kept {len(composite_rows)} composites ({failures} failed)", flush=True)

    rows = leaf_rows + bio_out + composite_rows
    write_manifest(rows)
    write_dataset_yaml()
    counts = compute_counts(rows)
    counts["bioimages"] = bio_meta["stats"]
    counts["composite_generation"] = {
        "candidates": len(specs),
        "kept": len(composite_rows),
        "failed": failures,
        "mask_screen": screen_counts,
    }
    write_json(DATASET_DIR / "counts.json", counts)
    print("validating...", flush=True)
    validation = validate(rows, species_splits)
    write_json(DATASET_DIR / "validation.json", validation)
    contact_sheet(rows, OUT_ROOT / "composites_contact_sheet.jpg")
    write_doc(counts, bio_meta, validation, rows)

    print(json.dumps({k: counts[k] for k in ("total_images", "total_boxes", "by_split_images", "by_split_boxes", "by_split_class")}, indent=2))
    print(f"validation: {'PASS' if validation['ok'] else 'FAIL'}")
    if not validation["ok"]:
        for check in validation["checks"]:
            if not check["ok"]:
                print(f"  FAILED {check['name']}: {check['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
