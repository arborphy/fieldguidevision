#!/usr/bin/env python3
# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Build reproducible LeafSnap/BioImages datasets for the two-layer leaf pipeline.

Layer A: YOLO26 organ detector (leaf, bud, flower, twig, branch, entire_plant).
Layer B: leaf-type classifier (simple_leaf, compound_leaf) from LeafSnap crops.

The script is deterministic: every split, sample cap, and filename is derived from
a SHA-256 key over stable source identifiers and a fixed salt.  It does not train
or download anything.

Usage:
    .venv/bin/python experiments/leafsnap-yolo26/build_datasets.py
    .venv/bin/python experiments/leafsnap-yolo26/build_datasets.py --overwrite

Generated images/labels live under scratch/leafsnap-yolo26/.  Source and the
generated DATASETS.md live under experiments/leafsnap-yolo26/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmentation.devo_adapter import DirrDevoAdapter  # noqa: E402

DEFAULT_LEAFSNAP_ROOT = Path(
    "/Users/valkyrie/Desktop/arborphy product/leafsnap/leafsnap-dataset"
)
DEFAULT_BIOIMAGES_ROOT = Path(
    "/Users/valkyrie/dev/arborphy/fieldguidevision/scratch/bioimages-ne-trees"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "scratch" / "leafsnap-yolo26"
DEFAULT_DOC = Path(__file__).resolve().parent / "DATASETS.md"

DETECTOR_CLASSES = ["leaf", "bud", "flower", "twig", "branch", "entire_plant"]
CLASSIFIER_CLASSES = ["simple_leaf", "compound_leaf"]

# BioImages organ_tag (and organ_category fallback) -> detector class.
BIOIMAGES_ORGAN_MAP = {
    "bud": "bud",
    "flower": "flower",
    "twig": "twig",
    "bark": "branch",
    "whole plant": "entire_plant",
    "whole_tree": "entire_plant",
    "whole tree": "entire_plant",
    "whole tree (or vine)": "entire_plant",
}
EXCLUDED_BIOIMAGES_ORGANS = [
    "leaf",
    "fruit",
    "cone",
    "needle",
    "seed",
    "stem",
    "unspecified",
]

SEED = "leafsnap-yolo26-v1-20261029"
DETECTOR_SALT = f"{SEED}:detector"
BIOIMAGES_SALT = f"{SEED}:bioimages"
CLASSIFIER_SALT = f"{SEED}:classifier"

# A mask foreground box smaller than 0.5% of the image is treated as a degenerate
# segmentation (specks, failed segmentation) rather than a usable leaf box.
MIN_BBOX_AREA_FRAC = 0.005
MIN_BBOX_SIDE_PX = 12
MAX_ASPECT_MISMATCH = 0.05
CLASSIFIER_MARGIN_FRAC = 0.10
WEAK_CLOSEUP_INSET = 0.02
PROGRESS_EVERY = 2500


@dataclass
class LeafSnapRecord:
    file_id: str
    species: str
    domain: str  # "lab" | "field"
    image_path: Path
    mask_path: Path
    width: int
    height: int
    mask_width: int
    mask_height: int
    bbox_px: tuple[float, float, float, float]
    bbox_yolo: tuple[float, float, float, float]
    bbox_area_frac: float
    foreground_frac: float

    @property
    def uid(self) -> str:
        return f"leafsnap_{self.file_id}"

    @property
    def source_id(self) -> str:
        return self.file_id


@dataclass
class BioImagesRecord:
    image_id: str
    file: str
    species: str
    individual_id: str
    organ_tag: str
    organ_category: str
    subview: str
    image_path: Path
    class_name: str
    box_kind: str
    bbox_yolo: tuple[float, float, float, float]
    split: str = ""

    @property
    def uid(self) -> str:
        return "bio_" + self.image_id.replace("/", "_").replace(" ", "_")

    @property
    def source_id(self) -> str:
        return self.image_id


@dataclass
class SplitRecord:
    uid: str
    source: str
    source_id: str
    split: str
    class_id: int
    class_name: str
    image_rel: str
    label_rel: str
    width: int
    height: int
    bbox_xyxy: tuple[float, float, float, float]
    bbox_yolo: tuple[float, float, float, float]
    species: str
    domain: str
    box_kind: str
    weak_box: bool
    source_image: str
    mask_path: str | None
    individual_id: str | None = None
    organ_tag: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CropRecord:
    uid: str
    source_id: str
    split: str
    class_id: int
    class_name: str
    species: str
    domain: str
    image_rel: str
    source_image: str
    mask_path: str
    width: int
    height: int
    image_width: int
    image_height: int
    full_bbox_xyxy: tuple[int, int, int, int]
    crop_bbox_xyxy: tuple[int, int, int, int]
    margin_fraction: float


def sha_key(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def stable_sort(items: Iterable[Any], *key_parts: Any) -> list[Any]:
    return sorted(items, key=lambda item: sha_key(*key_parts, item))


def yolo_from_xyxy(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    return (
        max(0.0, min(float(w), x1)),
        max(0.0, min(float(h), y1)),
        max(0.0, min(float(w), x2)),
        max(0.0, min(float(h), y2)),
    )


# ---------------------------------------------------------------------------
# LeafSnap scan
# ---------------------------------------------------------------------------


def scan_leafsnap(root: Path) -> tuple[list[LeafSnapRecord], dict[str, Any]]:
    manifest_path = root / "leafsnap-dataset-images.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"LeafSnap manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    required = {"file_id", "image_path", "segmented_path", "species", "source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected LeafSnap manifest columns: {rows[0] if rows else None}")

    failures: list[dict[str, Any]] = defaultdict(list)
    ids = Counter(row["file_id"] for row in rows)
    duplicate_ids = sorted(k for k, v in ids.items() if v > 1)

    records: list[LeafSnapRecord] = []
    for i, row in enumerate(rows, 1):
        file_id = row["file_id"].strip()
        source = (row.get("source") or "").strip().lower()
        if source not in {"lab", "field"}:
            failures["unknown_domain"].append({"file_id": file_id, "source": source})
            continue
        image_path = root / row["image_path"]
        mask_path = root / row["segmented_path"]
        if not image_path.is_file():
            failures["missing_image"].append({"file_id": file_id, "path": str(image_path)})
            continue
        if not mask_path.is_file():
            failures["missing_mask"].append({"file_id": file_id, "path": str(mask_path)})
            continue
        try:
            with Image.open(mask_path) as mask_img:
                mask = np.asarray(mask_img)
                mask_w, mask_h = mask_img.size
            with Image.open(image_path) as image_img:
                width, height = image_img.size
                image_img.load()  # full decode: truncated JBIG/JPEG sources are rejected
        except Exception as exc:  # pragma: no cover - corrupt source file
            failures["unreadable_file"].append(
                {"file_id": file_id, "path": str(image_path), "error": repr(exc)}
            )
            continue

        fg = mask > 0
        fg_count = int(fg.sum())
        if fg_count == 0:
            failures["empty_mask"].append(
                {"file_id": file_id, "source": source, "species": row["species"]}
            )
            continue
        ys, xs = np.nonzero(fg)
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            failures["invalid_mask_bbox"].append({"file_id": file_id})
            continue

        image_aspect = width / height
        mask_aspect = mask_w / mask_h
        aspect_mismatch = abs(image_aspect - mask_aspect) / image_aspect
        if aspect_mismatch > MAX_ASPECT_MISMATCH:
            failures["aspect_mismatch"].append(
                {
                    "file_id": file_id,
                    "image_size": [width, height],
                    "mask_size": [mask_w, mask_h],
                    "relative_mismatch": round(aspect_mismatch, 4),
                }
            )
            continue

        bbox_area_frac = ((x2 - x1) * (y2 - y1)) / (mask_w * mask_h)
        if bbox_area_frac < MIN_BBOX_AREA_FRAC:
            failures["tiny_bbox"].append(
                {
                    "file_id": file_id,
                    "source": source,
                    "species": row["species"],
                    "bbox_area_frac": round(bbox_area_frac, 6),
                    "mask_bbox": [x1, y1, x2, y2],
                }
            )
            continue

        # Normalized mask coordinates map onto the original image.  Mask sizes are
        # 700 px on the long edge in this release, originals up to 800 px.
        sx, sy = width / mask_w, height / mask_h
        bx1, by1, bx2, by2 = clamp_box(x1 * sx, y1 * sy, x2 * sx, y2 * sy, width, height)
        if (bx2 - bx1) < MIN_BBOX_SIDE_PX or (by2 - by1) < MIN_BBOX_SIDE_PX:
            failures["tiny_bbox_px"].append(
                {
                    "file_id": file_id,
                    "bbox_px": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                }
            )
            continue

        records.append(
            LeafSnapRecord(
                file_id=file_id,
                species=row["species"].strip(),
                domain=source,
                image_path=image_path,
                mask_path=mask_path,
                width=width,
                height=height,
                mask_width=mask_w,
                mask_height=mask_h,
                bbox_px=(bx1, by1, bx2, by2),
                bbox_yolo=yolo_from_xyxy(bx1, by1, bx2, by2, width, height),
                bbox_area_frac=bbox_area_frac,
                foreground_frac=fg_count / float(mask_w * mask_h),
            )
        )
        if i % PROGRESS_EVERY == 0:
            print(f"  leafsnap scan: {i}/{len(rows)} rows, {len(records)} valid", flush=True)

    # Every retained source image was fully decoded above; no sampling.
    decode_failures = failures.get("unreadable_file", [])

    stats = {
        "manifest_rows": len(rows),
        "duplicate_file_ids": duplicate_ids,
        "valid_images": len(records),
        "by_domain": dict(Counter(r.domain for r in records)),
        "by_species": len({r.species for r in records}),
        "failures": {k: len(v) for k, v in sorted(failures.items())},
        "failure_examples": {
            k: v[:20] for k, v in sorted(failures.items())
        },
        "decoded_images": len(records),
        "decode_failures": decode_failures,
        "bbox_area_frac_quantiles": {
            str(q): round(float(np.quantile([r.bbox_area_frac for r in records], q)), 6)
            for q in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "masks_near_full_frame": sum(1 for r in records if r.bbox_area_frac > 0.95),
        "mask_sizes": dict(
            Counter(f"{r.mask_width}x{r.mask_height}" for r in records).most_common(10)
        ),
    }
    return records, stats


# ---------------------------------------------------------------------------
# BioImages scan
# ---------------------------------------------------------------------------


def scan_bioimages(corpus_path: Path) -> tuple[list[BioImagesRecord], dict[str, Any]]:
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes") or []
    if not scenes:
        raise ValueError(f"No scenes in BioImages corpus: {corpus_path}")

    image_dir = corpus_path.parent / "images"
    records: list[BioImagesRecord] = []
    issues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    organ_counter: Counter[str] = Counter()
    mapped_counter: Counter[str] = Counter()
    ids: Counter[str] = Counter()

    for scene in scenes:
        species = str(scene.get("scientific_name") or "").strip()
        for image in scene.get("images") or []:
            image_id = str(image.get("image_id") or "").strip()
            file_name = str(image.get("file") or "").strip()
            ids[image_id] += 1
            organ_tag = str(image.get("organ_tag") or "").strip().lower()
            organ_category = str(image.get("organ_category") or "").strip().lower()
            organ_counter[organ_tag] += 1
            subview = str(image.get("subview") or "").strip().lower()
            target = BIOIMAGES_ORGAN_MAP.get(organ_tag) or BIOIMAGES_ORGAN_MAP.get(
                organ_category
            )
            # BioImages stores winter-bud close-ups as twig images. The subview
            # is the actual organ annotation for these 49 frames; winter-overall
            # and petiole-orientation views remain twig examples.
            if organ_tag == "twig" and subview in {
                "close-up winter terminal bud",
                "close-up winter leaf scar/bud",
            }:
                target = "bud"
            if target is None:
                continue
            path = image_dir / file_name
            if not image_id or not file_name:
                issues["missing_id_or_file"].append({"species": species, "image": image})
                continue
            if not path.is_file():
                issues["missing_image"].append(
                    {"image_id": image_id, "file": file_name, "path": str(path)}
                )
                continue
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    img.load()  # full decode; some corpus JPEGs are truncated
            except Exception as exc:  # pragma: no cover - corrupt source file
                issues["unreadable_file"].append(
                    {"image_id": image_id, "file": file_name, "error": repr(exc)}
                )
                continue
            if target == "entire_plant":
                bbox = (0.5, 0.5, 1.0, 1.0)
                box_kind = "full_frame_weak"
            else:
                inset = WEAK_CLOSEUP_INSET
                bbox = (0.5, 0.5, 1.0 - 2 * inset, 1.0 - 2 * inset)
                box_kind = "near_full_frame_weak"
            records.append(
                BioImagesRecord(
                    image_id=image_id,
                    file=file_name,
                    species=species,
                    individual_id=str(image.get("individual_id") or "").strip(),
                    organ_tag=organ_tag,
                    organ_category=organ_category,
                    subview=str(image.get("subview") or ""),
                    image_path=path,
                    class_name=target,
                    box_kind=box_kind,
                    bbox_yolo=bbox,
                )
            )
            mapped_counter[target] += 1

    duplicate_ids = sorted(k for k, v in ids.items() if v > 1)
    missing_individual = sum(1 for r in records if not r.individual_id)
    if missing_individual:
        issues["missing_individual_id"] = [
            {"image_id": r.image_id} for r in records if not r.individual_id
        ]

    # Every selected image was fully decoded above; no sampling.
    decode_failures = issues.get("unreadable_file", [])

    selected_by_class = Counter(r.class_name for r in records)
    stats = {
        "corpus_image_count": sum(organ_counter.values()),
        "organ_tag_counts": dict(sorted(organ_counter.items())),
        "mapped_class_counts": dict(sorted(mapped_counter.items())),
        "selected_images": len(records),
        "selected_species": len({r.species for r in records}),
        "selected_individuals": len({r.individual_id for r in records}),
        "unmapped_organs": {
            k: organ_counter[k]
            for k in EXCLUDED_BIOIMAGES_ORGANS
            if organ_counter.get(k)
        },
        "duplicate_image_ids": duplicate_ids,
        "missing_individual_ids": missing_individual,
        "class_counts": dict(selected_by_class),
        "issues": {k: len(v) for k, v in sorted(issues.items())},
        "issue_examples": {k: v[:10] for k, v in sorted(issues.items())},
        "decoded_images": len(records),
        "decode_failures": decode_failures,
    }
    return records, stats


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


def leafsnap_detector_split(records: Sequence[LeafSnapRecord]) -> dict[str, list[str]]:
    by_species: dict[str, list[LeafSnapRecord]] = defaultdict(list)
    for rec in records:
        by_species[rec.species].append(rec)
    ordered = sorted(by_species, key=lambda s: sha_key(DETECTOR_SALT, "species", s))
    n = len(ordered)
    n_test = round(n * 0.18)
    n_val = round(n * 0.12)

    test_species: list[str] = []
    for species in ordered:
        if len(test_species) >= n_test:
            break
        # The detector test set is field-only for domain generalization.  A
        # species without field images cannot contribute to it.
        if any(r.domain == "field" for r in by_species[species]):
            test_species.append(species)
    test_set = set(test_species)
    remaining = [s for s in ordered if s not in test_set]
    val_species = remaining[:n_val]
    train_species = remaining[n_val:]
    return {
        "train": train_species,
        "val": val_species,
        "test": test_species,
    }


def sample_by_domain(
    records: Sequence[LeafSnapRecord],
    caps: dict[str, int],
    split_name: str,
) -> list[LeafSnapRecord]:
    selected: list[LeafSnapRecord] = []
    for domain in ("lab", "field"):
        pool = stable_sort(
            [r for r in records if r.domain == domain], SEED, "detector", split_name, domain
        )
        cap = caps.get(domain, 0)
        if cap is None:
            selected.extend(pool)
        else:
            selected.extend(pool[:cap])
    return selected


def bioimages_split(records: Sequence[BioImagesRecord]) -> dict[str, list[BioImagesRecord]]:
    individuals = sorted({r.individual_id for r in records if r.individual_id})
    ordered = sorted(individuals, key=lambda i: sha_key(BIOIMAGES_SALT, "individual", i))
    n = len(ordered)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    test = ordered[:n_test]
    val = ordered[n_test : n_test + n_val]
    train = ordered[n_test + n_val :]
    out = {"train": [], "val": [], "test": []}
    for rec in records:
        if rec.individual_id in set(test):
            rec.split = "test"
        elif rec.individual_id in set(val):
            rec.split = "val"
        else:
            rec.split = "train"
        out[rec.split].append(rec)
    return out


def classifier_species_split(species_to_class: dict[str, str]) -> dict[str, list[str]]:
    splits = {"train": [], "val": [], "test": []}
    for class_name in CLASSIFIER_CLASSES:
        species = sorted(
            [s for s, c in species_to_class.items() if c == class_name],
            key=lambda s: sha_key(CLASSIFIER_SALT, "species", s),
        )
        n = len(species)
        n_test = max(2, round(n * 0.20))
        n_val = max(1, round(n * 0.15))
        splits["test"].extend(species[:n_test])
        splits["val"].extend(species[n_test : n_test + n_val])
        splits["train"].extend(species[n_test + n_val :])
    return splits


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def prepare_output(out_root: Path, overwrite: bool) -> tuple[Path, Path]:
    if out_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {out_root}. Re-run with --overwrite to regenerate."
        )
    detector_dir = out_root / "detector"
    classification_dir = out_root / "classification"
    if overwrite:
        for path in (detector_dir, classification_dir):
            if path.exists():
                shutil.rmtree(path)
    out_root.mkdir(parents=True, exist_ok=True)
    return detector_dir, classification_dir


def write_detector(
    detector_dir: Path,
    leafsnap_records: Sequence[LeafSnapRecord],
    leafsnap_splits: dict[str, list[str]],
    bio_records: Sequence[BioImagesRecord],
    bio_splits: dict[str, list[BioImagesRecord]],
) -> tuple[list[SplitRecord], dict[str, Any]]:
    for split in ("train", "val", "test"):
        (detector_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (detector_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    by_species: dict[str, list[LeafSnapRecord]] = defaultdict(list)
    for rec in leafsnap_records:
        by_species[rec.species].append(rec)

    leaf_caps = {
        "train": {"lab": 30, "field": 12},
        "val": {"lab": 25, "field": 15},
        "test": {"lab": 0, "field": 45},
    }

    split_records: list[SplitRecord] = []
    for split in ("train", "val", "test"):
        for species in leafsnap_splits[split]:
            for rec in sample_by_domain(by_species[species], leaf_caps[split], split):
                uid = rec.uid
                image_rel = f"images/{split}/{uid}.jpg"
                label_rel = f"labels/{split}/{uid}.txt"
                shutil.copy2(rec.image_path, detector_dir / image_rel)
                split_records.append(
                    SplitRecord(
                        uid=uid,
                        source="leafsnap",
                        source_id=rec.source_id,
                        split=split,
                        class_id=0,
                        class_name="leaf",
                        image_rel=image_rel,
                        label_rel=label_rel,
                        width=rec.width,
                        height=rec.height,
                        bbox_xyxy=rec.bbox_px,
                        bbox_yolo=rec.bbox_yolo,
                        species=rec.species,
                        domain=rec.domain,
                        box_kind="mask_foreground_bbox",
                        weak_box=False,
                        source_image=str(rec.image_path),
                        mask_path=str(rec.mask_path),
                        extra={
                            "bbox_area_frac": round(rec.bbox_area_frac, 6),
                            "foreground_frac": round(rec.foreground_frac, 6),
                            "mask_size": [rec.mask_width, rec.mask_height],
                        },
                    )
                )

    for split in ("train", "val", "test"):
        for rec in bio_splits[split]:
            uid = rec.uid
            image_rel = f"images/{split}/{uid}.jpg"
            label_rel = f"labels/{split}/{uid}.txt"
            shutil.copy2(rec.image_path, detector_dir / image_rel)
            with Image.open(rec.image_path) as img:
                width, height = img.size
            split_records.append(
                SplitRecord(
                    uid=uid,
                    source="bioimages",
                    source_id=rec.source_id,
                    split=split,
                    class_id=DETECTOR_CLASSES.index(rec.class_name),
                    class_name=rec.class_name,
                    image_rel=image_rel,
                    label_rel=label_rel,
                    width=width,
                    height=height,
                    bbox_xyxy=(
                        (rec.bbox_yolo[0] - rec.bbox_yolo[2] / 2) * width,
                        (rec.bbox_yolo[1] - rec.bbox_yolo[3] / 2) * height,
                        (rec.bbox_yolo[0] + rec.bbox_yolo[2] / 2) * width,
                        (rec.bbox_yolo[1] + rec.bbox_yolo[3] / 2) * height,
                    ),
                    bbox_yolo=rec.bbox_yolo,
                    species=rec.species,
                    domain="bioimages",
                    box_kind=rec.box_kind,
                    weak_box=True,
                    source_image=str(rec.image_path),
                    mask_path=None,
                    individual_id=rec.individual_id,
                    organ_tag=rec.organ_tag,
                    extra={
                        "organ_category": rec.organ_category,
                        "subview": rec.subview,
                    },
                )
            )

    # Write labels and manifest.
    for rec in split_records:
        cx, cy, bw, bh = rec.bbox_yolo
        label_path = detector_dir / rec.label_rel
        label_path.write_text(f"{rec.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    manifest_path = detector_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in sorted(split_records, key=lambda r: (r.split, r.source, r.uid)):
            row = {
                "uid": rec.uid,
                "dataset": "detector",
                "source": rec.source,
                "source_id": rec.source_id,
                "split": rec.split,
                "class_id": rec.class_id,
                "class_name": rec.class_name,
                "image": rec.image_rel,
                "label": rec.label_rel,
                "width": rec.width,
                "height": rec.height,
                "bbox_xyxy": [round(v, 3) for v in rec.bbox_xyxy],
                "bbox_yolo": [round(v, 8) for v in rec.bbox_yolo],
                "species": rec.species,
                "domain": rec.domain,
                "box_kind": rec.box_kind,
                "weak_box": rec.weak_box,
                "source_image": rec.source_image,
                "mask_path": rec.mask_path,
                "individual_id": rec.individual_id,
                "organ_tag": rec.organ_tag,
                **rec.extra,
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    yaml_path = detector_dir / "dataset.yaml"
    yaml_path.write_text(
        "# Generated by experiments/leafsnap-yolo26/build_datasets.py\n"
        f"path: {detector_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 6\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(DETECTOR_CLASSES)),
        encoding="utf-8",
    )

    counts = count_detector(split_records, leafsnap_splits, bio_splits)
    (detector_dir / "counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return split_records, counts


def count_detector(
    records: Sequence[SplitRecord],
    leafsnap_splits: dict[str, list[str]],
    bio_splits: dict[str, list[BioImagesRecord]],
) -> dict[str, Any]:
    by_split_class: dict[str, dict[str, int]] = {}
    by_split_source: dict[str, dict[str, int]] = {}
    by_split_domain: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        cls = Counter(r.class_name for r in records if r.split == split)
        src = Counter(r.source for r in records if r.split == split)
        dom = Counter(r.domain for r in records if r.split == split)
        by_split_class[split] = {name: cls.get(name, 0) for name in DETECTOR_CLASSES}
        by_split_source[split] = dict(sorted(src.items()))
        by_split_domain[split] = dict(sorted(dom.items()))

    species_by_split = {}
    for split in ("train", "val", "test"):
        species_by_split[split] = sorted(
            {r.species for r in records if r.split == split and r.source == "leafsnap"}
        )
    individuals_by_split = {}
    for split in ("train", "val", "test"):
        individuals_by_split[split] = sorted(
            {
                r.individual_id
                for r in records
                if r.split == split and r.individual_id
            }
        )

    return {
        "total_images": len(records),
        "by_split_class": by_split_class,
        "by_split_source": by_split_source,
        "by_split_domain": by_split_domain,
        "split_species_counts": {
            split: len(species) for split, species in species_by_split.items()
        },
        "split_species": species_by_split,
        "bioimages_split_individual_counts": {
            split: len(inds) for split, inds in individuals_by_split.items()
        },
        "leafsnap_source_species_counts": {
            split: len(species) for split, species in leafsnap_splits.items()
        },
        "bioimages_source_individual_counts": {
            split: len({r.individual_id for r in recs})
            for split, recs in bio_splits.items()
        },
    }


def write_classification(
    classification_dir: Path,
    leafsnap_records: Sequence[LeafSnapRecord],
    species_to_class: dict[str, str],
    species_splits: dict[str, list[str]],
) -> tuple[list[CropRecord], dict[str, Any]]:
    for split in ("train", "val", "test"):
        for class_name in CLASSIFIER_CLASSES:
            (classification_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    by_species: dict[str, list[LeafSnapRecord]] = defaultdict(list)
    for rec in leafsnap_records:
        if rec.species in species_to_class:
            by_species[rec.species].append(rec)

    caps = {
        "train": {"lab": 40, "field": 15},
        "val": {"lab": 15, "field": 15},
        "test": {"lab": 10, "field": None},
    }

    crops: list[CropRecord] = []
    for split in ("train", "val", "test"):
        for species in species_splits[split]:
            class_name = species_to_class[species]
            class_id = CLASSIFIER_CLASSES.index(class_name)
            for rec in sample_by_domain(by_species[species], caps[split], split):
                x1, y1, x2, y2 = rec.bbox_px
                bw, bh = x2 - x1, y2 - y1
                mx, my = bw * CLASSIFIER_MARGIN_FRAC, bh * CLASSIFIER_MARGIN_FRAC
                cx1, cy1, cx2, cy2 = clamp_box(
                    x1 - mx, y1 - my, x2 + mx, y2 + my, rec.width, rec.height
                )
                ix1, iy1 = int(math.floor(cx1)), int(math.floor(cy1))
                ix2, iy2 = int(math.ceil(cx2)), int(math.ceil(cy2))
                if ix2 - ix1 < 16 or iy2 - iy1 < 16:
                    continue
                uid = rec.uid
                image_rel = f"{split}/{class_name}/{uid}.jpg"
                try:
                    with Image.open(rec.image_path) as img:
                        crop = img.crop((ix1, iy1, ix2, iy2))
                        crop.convert("RGB").save(
                            classification_dir / image_rel, quality=92, subsampling=0
                        )
                except Exception:  # pragma: no cover - corrupt source file
                    continue
                crops.append(
                    CropRecord(
                        uid=uid,
                        source_id=rec.source_id,
                        split=split,
                        class_id=class_id,
                        class_name=class_name,
                        species=rec.species,
                        domain=rec.domain,
                        image_rel=image_rel,
                        source_image=str(rec.image_path),
                        mask_path=str(rec.mask_path),
                        width=ix2 - ix1,
                        height=iy2 - iy1,
                        image_width=rec.width,
                        image_height=rec.height,
                        full_bbox_xyxy=(
                            int(round(x1)),
                            int(round(y1)),
                            int(round(x2)),
                            int(round(y2)),
                        ),
                        crop_bbox_xyxy=(ix1, iy1, ix2, iy2),
                        margin_fraction=CLASSIFIER_MARGIN_FRAC,
                    )
                )

    manifest_path = classification_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in sorted(crops, key=lambda r: (r.split, r.class_name, r.uid)):
            row = {
                "uid": rec.uid,
                "dataset": "leaf_type_classifier",
                "source": "leafsnap",
                "source_id": rec.source_id,
                "split": rec.split,
                "class_id": rec.class_id,
                "class_name": rec.class_name,
                "image": rec.image_rel,
                "species": rec.species,
                "domain": rec.domain,
                "crop_width": rec.width,
                "crop_height": rec.height,
                "image_width": rec.image_width,
                "image_height": rec.image_height,
                "full_bbox_xyxy": list(rec.full_bbox_xyxy),
                "crop_bbox_xyxy": list(rec.crop_bbox_xyxy),
                "margin_fraction": rec.margin_fraction,
                "source_image": rec.source_image,
                "mask_path": rec.mask_path,
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    labels = {
        "task": "image_classification",
        "classes": {name: i for i, name in enumerate(CLASSIFIER_CLASSES)},
        "class_names": CLASSIFIER_CLASSES,
        "source": "leafsnap",
        "crop": {
            "bbox": "normalized mask foreground bbox scaled to the original image",
            "margin_fraction": CLASSIFIER_MARGIN_FRAC,
            "margin_rule": "10% of bbox width/height on each side, clamped",
        },
        "label_source": "Dirr DeVo leaf_type, exact Simple or Compound only",
        "splits": {
            split: {
                "species": sorted(species_splits[split]),
                "classes": dict(
                    sorted(
                        Counter(species_to_class[s] for s in species_splits[split]).items()
                    )
                ),
            }
            for split in ("train", "val", "test")
        },
    }
    (classification_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    counts = count_classification(crops, species_splits, species_to_class)
    (classification_dir / "counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return crops, counts


def count_classification(
    crops: Sequence[CropRecord],
    species_splits: dict[str, list[str]],
    species_to_class: dict[str, str],
) -> dict[str, Any]:
    by_split_class: dict[str, dict[str, int]] = {}
    by_split_domain: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        cls = Counter(c.class_name for c in crops if c.split == split)
        dom = Counter(c.domain for c in crops if c.split == split)
        by_split_class[split] = {name: cls.get(name, 0) for name in CLASSIFIER_CLASSES}
        by_split_domain[split] = dict(sorted(dom.items()))
    by_split_species = {
        split: sorted({c.species for c in crops if c.split == split})
        for split in ("train", "val", "test")
    }
    return {
        "total_crops": len(crops),
        "by_split_class": by_split_class,
        "by_split_domain": by_split_domain,
        "split_species_counts": {
            split: len(species) for split, species in by_split_species.items()
        },
        "split_species": by_split_species,
        "source_species_counts": {
            split: len(species) for split, species in species_splits.items()
        },
        "source_species_by_class": {
            split: dict(
                sorted(Counter(species_to_class[s] for s in species).items())
            )
            for split, species in species_splits.items()
        },
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def decode_check(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        return {"checked": 0, "failures": []}
    ordered = sorted(paths, key=lambda p: sha_key(SEED, "validate-decode", p.name))
    failures = []
    for path in ordered:
        try:
            with Image.open(path) as img:
                img.load()
        except Exception as exc:  # pragma: no cover - corrupt generated file
            failures.append({"path": str(path), "error": repr(exc)})
    return {
        "checked": len(ordered),
        "failures": failures,
    }


def validate_outputs(
    detector_dir: Path,
    classification_dir: Path,
    detector_records: Sequence[SplitRecord],
    crops: Sequence[CropRecord],
    leafsnap_splits: dict[str, list[str]],
    bio_splits: dict[str, list[BioImagesRecord]],
    classifier_splits: dict[str, list[str]],
) -> dict[str, Any]:
    issues: list[str] = []

    # Detector: unique ids, no split overlap, bbox ranges, labels/images.
    det_ids = Counter(r.uid for r in detector_records)
    issues.extend(f"detector duplicate uid: {uid}" for uid, n in det_ids.items() if n > 1)
    det_split_by_uid: dict[str, set[str]] = defaultdict(set)
    for r in detector_records:
        det_split_by_uid[r.uid].add(r.split)
    issues.extend(
        f"detector uid in multiple splits: {uid}" for uid, s in det_split_by_uid.items() if len(s) > 1
    )
    for r in detector_records:
        x1, y1, x2, y2 = r.bbox_xyxy
        cx, cy, bw, bh = r.bbox_yolo
        if not (0 <= x1 < x2 <= r.width and 0 <= y1 < y2 <= r.height):
            issues.append(f"detector bbox out of bounds: {r.uid}")
        if not (0 < bw <= 1 and 0 < bh <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
            issues.append(f"detector YOLO bbox out of range: {r.uid}")
        image_path = detector_dir / r.image_rel
        label_path = detector_dir / r.label_rel
        if not image_path.is_file() or image_path.stat().st_size == 0:
            issues.append(f"detector missing/empty image: {r.uid}")
        if not label_path.is_file():
            issues.append(f"detector missing label: {r.uid}")
        else:
            parts = label_path.read_text(encoding="utf-8").strip().split()
            if len(parts) != 5:
                issues.append(f"detector malformed label: {r.uid}")
            elif int(parts[0]) != r.class_id:
                issues.append(f"detector label class mismatch: {r.uid}")

    # Split disjointness at the sampling unit.
    leaf_species_by_split = {
        split: set(species) for split, species in leafsnap_splits.items()
    }
    species_overlap = (
        leaf_species_by_split["train"] & leaf_species_by_split["val"]
    ) | (leaf_species_by_split["train"] & leaf_species_by_split["test"]) | (
        leaf_species_by_split["val"] & leaf_species_by_split["test"]
    )
    if species_overlap:
        issues.append(f"LeafSnap species split overlap: {sorted(species_overlap)[:5]}")
    bio_ind_by_split = {
        split: {r.individual_id for r in recs} for split, recs in bio_splits.items()
    }
    individual_overlap = (
        bio_ind_by_split["train"] & bio_ind_by_split["val"]
    ) | (bio_ind_by_split["train"] & bio_ind_by_split["test"]) | (
        bio_ind_by_split["val"] & bio_ind_by_split["test"]
    )
    if individual_overlap:
        issues.append(f"BioImages individual split overlap: {sorted(individual_overlap)[:5]}")

    classifier_species_overlap = (
        set(classifier_splits["train"]) & set(classifier_splits["val"])
    ) | (set(classifier_splits["train"]) & set(classifier_splits["test"])) | (
        set(classifier_splits["val"]) & set(classifier_splits["test"])
    )
    if classifier_species_overlap:
        issues.append(
            f"Classifier species split overlap: {sorted(classifier_species_overlap)[:5]}"
        )

    # Classification: unique ids, files, dimensions, folder/class agreement.
    crop_ids = Counter(c.uid for c in crops)
    issues.extend(f"classifier duplicate uid: {uid}" for uid, n in crop_ids.items() if n > 1)
    crop_split_by_uid: dict[str, set[str]] = defaultdict(set)
    for c in crops:
        crop_split_by_uid[c.uid].add(c.split)
    issues.extend(
        f"classifier uid in multiple splits: {uid}"
        for uid, s in crop_split_by_uid.items()
        if len(s) > 1
    )
    for c in crops:
        path = classification_dir / c.image_rel
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"classifier missing/empty crop: {c.uid}")
            continue
        if path.parent.name != c.class_name or path.parent.parent.name != c.split:
            issues.append(f"classifier folder mismatch: {c.uid}")
        if c.width < 16 or c.height < 16:
            issues.append(f"classifier crop too small: {c.uid}")
        if not (0 <= c.crop_bbox_xyxy[0] < c.crop_bbox_xyxy[2] <= c.image_width):
            issues.append(f"classifier crop bbox x out of bounds: {c.uid}")
        if not (0 <= c.crop_bbox_xyxy[1] < c.crop_bbox_xyxy[3] <= c.image_height):
            issues.append(f"classifier crop bbox y out of bounds: {c.uid}")

    det_images = [p for p in (detector_dir / "images").rglob("*") if p.is_file()]
    cls_images = [p for p in classification_dir.rglob("*.jpg") if p.is_file()]
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues[:100],
        "issue_count": len(issues),
        "detector_images": len(det_images),
        "detector_labels": len(list((detector_dir / "labels").rglob("*.txt"))),
        "detector_decode": decode_check(det_images),
        "classifier_crops": len(cls_images),
        "classifier_decode": decode_check(cls_images),
        "split_checks": {
            "leafsnap_species_disjoint": not species_overlap,
            "bioimages_individuals_disjoint": not individual_overlap,
            "classifier_species_disjoint": not classifier_species_overlap,
            "detector_uid_unique": max(det_ids.values(), default=0) == 1,
            "classifier_uid_unique": max(crop_ids.values(), default=0) == 1,
        },
    }


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


def fmt_counts(by_split_class: dict[str, dict[str, int]], classes: Sequence[str]) -> str:
    lines = ["| split | " + " | ".join(classes) + " | total |"]
    lines.append("|---" * (len(classes) + 2) + "|")
    for split in ("train", "val", "test"):
        counts = by_split_class[split]
        total = sum(counts.values())
        lines.append(
            f"| {split} | "
            + " | ".join(str(counts.get(c, 0)) for c in classes)
            + f" | {total} |"
        )
    totals = {
        c: sum(by_split_class[s].get(c, 0) for s in by_split_class) for c in classes
    }
    lines.append(
        "| **total** | "
        + " | ".join(str(totals[c]) for c in classes)
        + f" | {sum(totals.values())} |"
    )
    return "\n".join(lines)


def write_doc(
    doc_path: Path,
    detector_dir: Path,
    classification_dir: Path,
    detector_counts: dict[str, Any],
    classification_counts: dict[str, Any],
    leafsnap_stats: dict[str, Any],
    bio_stats: dict[str, Any],
    dirr_audit: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    unmatched = "\n".join(f"- `{s}`" for s in dirr_audit["unmatched"])
    mixed = "\n".join(
        f"- `{s}` — {', '.join(values)}" for s, values in dirr_audit["mixed"]
    )
    neither = "\n".join(f"- `{s}`" for s in dirr_audit["neither"])

    text = f"""# LeafSnap + BioImages datasets (YOLO26 two-layer leaf pipeline)

Generated by `experiments/leafsnap-yolo26/build_datasets.py`. The build is
deterministic from fixed SHA-256 salts (`{SEED}`); no network access and no
training are involved. Do not hand-edit generated images, labels, manifests, or
counts.

## Paths

- Detector YAML: `{detector_dir / "dataset.yaml"}`
- Detector root: `{detector_dir}`
- Detector manifest: `{detector_dir / "manifest.jsonl"}`
- Detector counts: `{detector_dir / "counts.json"}`
- Classifier root: `{classification_dir}`
- Classifier labels: `{classification_dir / "labels.json"}`
- Classifier manifest: `{classification_dir / "manifest.jsonl"}`
- Classifier counts: `{classification_dir / "counts.json"}`
- Shared exclusions/issues: `{detector_dir.parent / "exclusions.json"}`
- Shared validation: `{detector_dir.parent / "validation.json"}`

## A. Top-level YOLO26 organ detector (E1)

Classes, in YAML order:

| id | class |
|---:|---|
| 0 | `leaf` |
| 1 | `bud` |
| 2 | `flower` |
| 3 | `twig` |
| 4 | `branch` |
| 5 | `entire_plant` |

### Source mapping

| source | source label | detector class | box |
|---|---|---|---|
| LeafSnap | every valid segmented leaf | `leaf` | normalized mask foreground bbox, scaled to original image |
| BioImages | `organ_tag=bud` OR `twig` with explicit winter-bud close-up subview | `bud` | near-full-frame weak box (YOLO 0.5, 0.5, 0.96, 0.96) |
| BioImages | `organ_tag=flower` | `flower` | near-full-frame weak box (YOLO 0.5, 0.5, 0.96, 0.96) |
| BioImages | `organ_tag=twig` | `twig` | near-full-frame weak box (YOLO 0.5, 0.5, 0.96, 0.96) |
| BioImages | `organ_tag=bark` (woody surface) | `branch` | near-full-frame weak box (YOLO 0.5, 0.5, 0.96, 0.96) |
| BioImages | `organ_tag=whole plant` / `whole_tree` | `entire_plant` | full-frame weak box (YOLO center 0.5, 0.5, size 1.0, 1.0) |
| BioImages | `leaf`, `fruit`, `cone`, `needle`, `seed`, `stem`, `unspecified` | excluded | — |

BioImages has **no boxes**. The close-up box is an explicit weak label, not a
human annotation; entire-plant uses the full frame. Detection metrics on
BioImages therefore measure an upper bound / sanity check, not localization
quality. LeafSnap boxes are segmentation-derived but not human-drawn either:
they inherit segmentation errors, and empty/degenerate masks were dropped.

### Split rules

- LeafSnap: species-disjoint. Species are ordered by SHA-256 and allocated
  18% test / 12% validation / remainder train. The test species contribute
  **field images only** (45 per species cap) so the held-out set measures
  domain generalization; their lab images are intentionally not emitted.
- Train samples 30 lab + 12 field images per species; validation samples 25 lab
  + 15 field per species. Caps are applied after deterministic hashing, so the
  same images are selected on every run.
- BioImages: individual-disjoint. Individuals are ordered by SHA-256 and
  allocated 15% test / 15% validation / remainder train; every selected image
  of an individual follows that individual.

### Counts

{fmt_counts(detector_counts["by_split_class"], DETECTOR_CLASSES)}

By source (images):

| split | LeafSnap | BioImages | total |
|---|---:|---:|---:|
""" + "".join(
        f"| {split} | {detector_counts['by_split_source'][split].get('leafsnap', 0)} | "
        f"{detector_counts['by_split_source'][split].get('bioimages', 0)} | "
        f"{sum(detector_counts['by_split_source'][split].values())} |\n"
        for split in ("train", "val", "test")
    ) + f"""
LeafSnap test species: {detector_counts['leafsnap_source_species_counts']['test']};
validation species: {detector_counts['leafsnap_source_species_counts']['val']};
train species: {detector_counts['leafsnap_source_species_counts']['train']}.
BioImages individuals: {detector_counts['bioimages_source_individual_counts']}.

**Class imbalance is expected**: LeafSnap supplies only `leaf`; BioImages
supplies the other organs. The `bud` examples are recovered from BioImages
`twig` images whose explicit subview is `close-up winter terminal bud` or
`close-up winter leaf scar/bud`; other twig views remain `twig`.

## B. Lower-level leaf-type classifier (E2)

Classes: `simple_leaf` (0), `compound_leaf` (1).

### Dirr join

Species names are joined exactly (case-insensitive) with
`segmentation/devo_adapter.py::DirrDevoAdapter.get_taxon`. A species is included
only when its Dirr `leaf_type` value set is exactly `{{"Simple"}}` or exactly
`{{"Compound"}}`. Missing rows, mixed Simple+Compound rows, and species with
only conifer leaf-type values are excluded and recorded below.

Audit: {dirr_audit['matched_count']} LeafSnap species matched unambiguously
({dirr_audit['class_counts'].get('simple_leaf', 0)} simple,
{dirr_audit['class_counts'].get('compound_leaf', 0)} compound).
{len(dirr_audit['unmatched'])} species had no exact Dirr row,
{len(dirr_audit['mixed'])} were mixed, and {len(dirr_audit['neither'])} had no
Simple/Compound label (mostly conifers).

Unmatched LeafSnap species (excluded — includes apparent synonyms/typos that
were deliberately not fuzzy-matched):

{unmatched}

Mixed-label species (excluded):

{mixed if mixed else '- none'}

No Simple/Compound label (excluded):

{neither}

### Crops

Each crop is the normalized mask foreground bbox scaled to the original image,
expanded by {int(CLASSIFIER_MARGIN_FRAC * 100)}% of the bbox width/height on each side, clamped to the
image, and saved as JPEG quality 92. This mirrors the tight leaf crop produced
by the organ gate. Empty/degenerate masks are excluded by the same validity
rules as the detector.

### Split rules

- Species-disjoint and class-stratified: species are hashed within each class
  and allocated 20% test / 15% validation / remainder train, with at least two
  test and one validation species per class.
- Train caps per species: 40 lab + 15 field. Validation: 15 lab + 15 field.
  Test: all field images plus at most 10 lab images per species (field-heavy).
- No species appears in more than one split.

### Counts

{fmt_counts(classification_counts["by_split_class"], CLASSIFIER_CLASSES)}

By domain (images):

| split | lab | field | total |
|---|---:|---:|---:|
""" + "".join(
        f"| {split} | {classification_counts['by_split_domain'][split].get('lab', 0)} | "
        f"{classification_counts['by_split_domain'][split].get('field', 0)} | "
        f"{sum(classification_counts['by_split_domain'][split].values())} |\n"
        for split in ("train", "val", "test")
    ) + f"""
Species counts: {classification_counts['split_species_counts']}. Source species
(train/val/test): {classification_counts['source_species_counts']}.

The compound class is small (16 species in LeafSnap/Dirr); the stratified split
keeps compound species in every split, but absolute compound counts remain much
lower than simple.

## Shared validation

- Source file existence checked for every LeafSnap image/mask and BioImages image.
- Every retained source image fully decoded (not sampled):
  {leafsnap_stats['decoded_images']} LeafSnap images and
  {bio_stats['decoded_images']} BioImages images; truncated/unreadable files
  are excluded and recorded.
- Mask validation: empty masks, degenerate bboxes, and aspect mismatches are
  recorded and excluded (see data issues below).
- Detector bboxes checked for pixel bounds and YOLO [0,1] ranges; every image
  has a label whose class id matches.
- No duplicate UIDs; no UID across splits.
- LeafSnap species sets, BioImages individual sets, and classifier species sets
  verified disjoint across train/val/test.
- Generated images decode-checked on a deterministic sample.

Validation result: **{validation['status']}** ({validation['issue_count']} issues).
Detector full decode {validation['detector_decode']['checked']} images,
classifier full decode {validation['classifier_decode']['checked']} crops.

## Source data issues (recorded, not silently dropped)

### LeafSnap

- Manifest rows: {leafsnap_stats['manifest_rows']}; valid images:
  {leafsnap_stats['valid_images']} ({leafsnap_stats['by_domain']}).
- Empty segmentation masks: {leafsnap_stats['failures'].get('empty_mask', 0)}
  ({(leafsnap_stats['failures'].get('empty_mask', 0) / max(1, leafsnap_stats['manifest_rows']) * 100):.1f}%).
- Tiny bboxes (<{MIN_BBOX_AREA_FRAC:.1%} of image area): {leafsnap_stats['failures'].get('tiny_bbox', 0)}.
- Tiny bboxes in pixels (<{MIN_BBOX_SIDE_PX} px): {leafsnap_stats['failures'].get('tiny_bbox_px', 0)}.
- Aspect mismatches (>{MAX_ASPECT_MISMATCH:.0%}): {leafsnap_stats['failures'].get('aspect_mismatch', 0)}.
- Missing files: image {leafsnap_stats['failures'].get('missing_image', 0)}, mask
  {leafsnap_stats['failures'].get('missing_mask', 0)}.
- Unreadable/truncated source images: {leafsnap_stats['failures'].get('unreadable_file', 0)}
  (excluded).
- Duplicate file ids: {len(leafsnap_stats['duplicate_file_ids'])}.
- Masks with bbox area >95%: {leafsnap_stats['masks_near_full_frame']} (retained).
- Mask sizes differ from originals; normalized bbox scaling handles this. No
  aspect mismatch was observed in this release.

### BioImages

- Corpus images: {bio_stats['corpus_image_count']}; selected for detector:
  {bio_stats['selected_images']} across {bio_stats['selected_individuals']} individuals
  and {bio_stats['selected_species']} species.
- Organ counts: `{bio_stats['organ_tag_counts']}`.
- `bud`: {bio_stats['mapped_class_counts'].get('bud', 0)} — recovered from explicit
  `twig` bud-close-up subviews; no winter-overall twig view is relabelled.
- Duplicate image ids: {len(bio_stats['duplicate_image_ids'])}.
- Missing individual ids: {bio_stats['missing_individual_ids']}.
- Unreadable/truncated source images: {bio_stats['issues'].get('unreadable_file', 0)}
  (excluded; PIL reports `image file is truncated`).
- Every retained selected file was fully decoded.

## Citation (LeafSnap README)

> Leafsnap (http://leafsnap.com) is an electronic field guide for identifying
> tree species from photos of their leaves. It was jointly created by computer
> scientists from Columbia University and the University of Maryland, and
> botanists from the Smithsonian Institution in Washington, DC.

> Neeraj Kumar, Peter N. Belhumeur, Arijit Biswas, David W. Jacobs, W. John
> Kress, Ida Lopez, João V. B. Soares, "Leafsnap: A Computer Vision System for
> Automatic Plant Species Identification," Proceedings of the 12th European
> Conference on Computer Vision (ECCV), October 2012.

## E1 / E2 checks

- **E1 (organ detector):** LeafSnap boxes are segmentation-derived foreground
  boxes, not human boxes; BioImages boxes are documented weak near-full-frame
  boxes (full frame for entire plants). Evaluate localization on LeafSnap
  separately from BioImages classification/recall. Class imbalance and the small
  bud sample must be reported with any metrics.
- **E2 (leaf-type classifier):** labels come from Dirr, not from image
  appearance. Mixed/no-label species are excluded; exact-name matching avoids
  silent taxonomic merging. Crops use the same mask-bbox + margin geometry as
  the organ gate. Species-disjoint splits prevent taxon leakage; test is
  field-heavy by construction.

## Reproduce

```bash
cd {REPO_ROOT}
.venv/bin/python experiments/leafsnap-yolo26/build_datasets.py --overwrite
```
"""
    doc_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leafsnap-root", type=Path, default=DEFAULT_LEAFSNAP_ROOT)
    parser.add_argument(
        "--bioimages-root", type=Path, default=DEFAULT_BIOIMAGES_ROOT
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out_root = args.out_root.resolve()
    if out_root.name != "leafsnap-yolo26":
        raise SystemExit(
            f"Refusing to write outside the expected scratch directory name: {out_root}"
        )
    detector_dir, classification_dir = prepare_output(out_root, args.overwrite)

    print("Scanning LeafSnap...", flush=True)
    leafsnap_records, leafsnap_stats = scan_leafsnap(args.leafsnap_root)
    print(
        f"  valid LeafSnap images: {len(leafsnap_records)} "
        f"({leafsnap_stats['by_domain']})",
        flush=True,
    )

    print("Scanning BioImages...", flush=True)
    bio_records, bio_stats = scan_bioimages(args.bioimages_root / "corpus.json")
    print(
        f"  selected BioImages images: {len(bio_records)} "
        f"({bio_stats['class_counts']})",
        flush=True,
    )

    print("Splitting...", flush=True)
    leafsnap_splits = leafsnap_detector_split(leafsnap_records)
    bio_splits = bioimages_split(bio_records)

    print("Writing detector dataset...", flush=True)
    detector_records, detector_counts = write_detector(
        detector_dir, leafsnap_records, leafsnap_splits, bio_records, bio_splits
    )

    print("Joining LeafSnap species to Dirr leaf_type...", flush=True)
    adapter = DirrDevoAdapter()
    species = sorted({r.species for r in leafsnap_records})
    species_to_class: dict[str, str] = {}
    unmatched: list[str] = []
    mixed: list[tuple[str, list[str]]] = []
    neither: list[str] = []
    for name in species:
        taxon = adapter.get_taxon(name)
        if taxon is None:
            unmatched.append(name)
            continue
        labels = {
            fv.value_label for fv in taxon.feature_values if fv.feature_name == "leaf_type"
        }
        if labels == {"Simple"}:
            species_to_class[name] = "simple_leaf"
        elif labels == {"Compound"}:
            species_to_class[name] = "compound_leaf"
        elif labels:
            mixed.append((name, sorted(labels)))
        else:
            neither.append(name)
    dirr_audit = {
        "matched_count": len(species_to_class),
        "class_counts": dict(Counter(species_to_class.values())),
        "unmatched": unmatched,
        "mixed": mixed,
        "neither": neither,
        "matched_species": sorted(species_to_class),
    }
    print(
        f"  matched {len(species_to_class)} species "
        f"({dirr_audit['class_counts']}); unmatched {len(unmatched)}, "
        f"mixed {len(mixed)}, no-label {len(neither)}",
        flush=True,
    )

    classifier_splits = classifier_species_split(species_to_class)
    print("Writing classifier crops...", flush=True)
    crops, classification_counts = write_classification(
        classification_dir,
        leafsnap_records,
        species_to_class,
        classifier_splits,
    )

    print("Writing exclusions and validation...", flush=True)
    exclusions = {
        "leafsnap": {
            "failures": leafsnap_stats["failures"],
            "failure_examples": leafsnap_stats["failure_examples"],
            "duplicate_file_ids": leafsnap_stats["duplicate_file_ids"],
        },
        "bioimages": {
            "issues": bio_stats["issues"],
            "issue_examples": bio_stats["issue_examples"],
            "unmapped_organs": bio_stats["unmapped_organs"],
            "duplicate_image_ids": bio_stats["duplicate_image_ids"],
            "missing_individual_ids": bio_stats["missing_individual_ids"],
        },
        "dirr_join": {
            "matched_count": dirr_audit["matched_count"],
            "class_counts": dirr_audit["class_counts"],
            "unmatched": unmatched,
            "mixed": [{"species": s, "leaf_type": v} for s, v in mixed],
            "no_simple_or_compound_label": neither,
        },
    }
    (out_root / "exclusions.json").write_text(
        json.dumps(exclusions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation = validate_outputs(
        detector_dir,
        classification_dir,
        detector_records,
        crops,
        leafsnap_splits,
        bio_splits,
        classifier_splits,
    )
    validation["leafsnap_decode_failures"] = leafsnap_stats["decode_failures"]
    validation["bioimages_decode_failures"] = bio_stats["decode_failures"]
    (out_root / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    write_doc(
        args.doc,
        detector_dir,
        classification_dir,
        detector_counts,
        classification_counts,
        leafsnap_stats,
        bio_stats,
        dirr_audit,
        validation,
    )

    print("\nDetector counts:", json.dumps(detector_counts["by_split_class"], sort_keys=True))
    print(
        "Classifier counts:",
        json.dumps(classification_counts["by_split_class"], sort_keys=True),
    )
    print(f"Detector YAML: {detector_dir / 'dataset.yaml'}")
    print(f"Classifier:    {classification_dir}")
    print(f"Validation:    {validation['status']} ({validation['issue_count']} issues)")
    print(f"Doc:           {args.doc}")
    if validation["status"] != "pass":
        print("Validation issues:", json.dumps(validation["issues"][:10], indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
