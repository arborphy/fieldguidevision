# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1_ROOT = ROOT / "scratch" / "leafsnap-yolo26"
OUT_ROOT = V1_ROOT / "organ-gate-v3"
DATASET = OUT_ROOT / "dataset"
LEAFSNAP_ROOT = Path("/Users/valkyrie/Desktop/arborphy product/leafsnap/leafsnap-dataset")
BIO_ROOT = Path("/Users/valkyrie/dev/arborphy/fieldguidevision/scratch/bioimages-ne-trees")
PROPOSAL_PATH = V1_ROOT / "leaf-box-proposals" / "proposals.jsonl"
V1_MANIFEST = V1_ROOT / "detector" / "manifest.jsonl"
V1_COUNTS = V1_ROOT / "detector" / "counts.json"
V2_MANIFEST = V1_ROOT / "organ-gate-v2" / "dataset" / "manifest.jsonl"
CLASSES = ["leaf", "bud", "flower", "twig", "branch"]
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}
SEED = 260930
LEAF_MASK_MIN_AREA = 0.015
LEAF_MASK_MAX_AREA = 0.48
LEAF_BBOX_MAX_AREA = 0.58
LEAF_MIN_SCORE = 0.30
LEAF_MIN_QUALITY = 0.88
LEAF_MIN_FILL = 0.36
TUCK_MASK_MIN_AREA = 0.008
TUCK_MASK_MAX_AREA = 0.30
TUCK_BBOX_MAX_AREA = 0.42


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def stable_int(*parts: Any) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:16], 16)


def clean_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def norm_polygon(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    out = []
    for x, y in points:
        px = min(1.0, max(0.0, float(x) / width))
        py = min(1.0, max(0.0, float(y) / height))
        out.append([px, py])
    if len(out) > 3 and out[0] == out[-1]:
        out.pop()
    return out


def proposal_polygon(points: list[list[float]]) -> list[list[float]]:
    out = [[min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))] for x, y in points]
    if len(out) > 3 and out[0] == out[-1]:
        out.pop()
    return out


def polygon_bbox(poly: list[list[float]], width: int, height: int) -> list[float]:
    xs = [p[0] * width for p in poly]
    ys = [p[1] * height for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_yolo(xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height, (x2 - x1) / width, (y2 - y1) / height]


def rectangle_polygon(xyxy: list[float], width: int, height: int) -> list[list[float]]:
    x1, y1, x2, y2 = xyxy
    return norm_polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], width, height)


def mask_polygon(mask_path: str, image_size: tuple[int, int]) -> tuple[list[list[float]], dict[str, Any]]:
    width, height = image_size
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"could not read mask {mask_path}")
    if (mask.shape[1], mask.shape[0]) != image_size:
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"empty segmentation mask {mask_path}")
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, 0.0015 * perimeter)
    contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(contour) < 3:
        raise ValueError(f"degenerate segmentation mask {mask_path}")
    points = norm_polygon([[float(x), float(y)] for x, y in contour], width, height)
    return points, {
        "mask_area_fraction": round(float(binary.mean()), 8),
        "contour_area_px": round(float(cv2.contourArea(max(contours, key=cv2.contourArea))), 2),
        "polygon_vertices": len(points),
        "mask_path": mask_path,
    }


def polygon_iou(a: list[list[float]], b: list[list[float]]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if len(aa) < 3 or len(bb) < 3:
        return 0.0
    x1, y1 = np.min(aa, axis=0)
    x2, y2 = np.max(aa, axis=0)
    u1, v1 = np.min(bb, axis=0)
    u2, v2 = np.max(bb, axis=0)
    intersection = max(0.0, min(x2, u2) - max(x1, u1)) * max(0.0, min(y2, v2) - max(y1, v1))
    union = max(0.0, x2 - x1) * max(0.0, y2 - y1) + max(0.0, u2 - u1) * max(0.0, v2 - v1) - intersection
    return intersection / union if union else 0.0


def load_split_sources() -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
    leaf_split: dict[str, str] = {}
    leaf_v1: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(V1_MANIFEST):
        if row["source"] == "leafsnap":
            leaf_split[row["source_id"]] = row["split"]
            leaf_v1[row["source_id"]] = row
    bio_splits: dict[str, str] = {}
    bio_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(V2_MANIFEST):
        if row["source"] == "bioimages":
            bio_splits[row["source_id"]] = row["split"]
            bio_rows[row["source_id"]] = row
    return leaf_split, bio_splits, bio_rows


def bioimages_leaf_rows(bio_splits: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposals = read_jsonl(PROPOSAL_PATH)
    corpus = json.loads((BIO_ROOT / "corpus.json").read_text())
    image_meta: dict[str, dict[str, Any]] = {}
    for scene in corpus.get("scenes", []):
        for image in scene.get("images", []):
            image_id = str(image.get("image_id") or "")
            image_meta[image_id] = {
                "organ_tag": str(image.get("organ_tag") or "").strip().lower(),
                "individual_id": str(image.get("individual_id") or ""),
                "species": str(scene.get("scientific_name") or ""),
                "subview": str(image.get("subview") or ""),
                "filename": str(image.get("file") or ""),
            }
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    for row in proposals:
        if row.get("source") != "bioimages_leaf":
            continue
        image_id = str(row.get("image_id") or "")
        meta = image_meta.get(image_id, {})
        reason = None
        if meta.get("organ_tag") != "leaf":
            reason = "image-level organ_tag is not leaf"
        elif image_id not in bio_splits:
            reason = "missing deterministic BioImages individual split"
        elif not row.get("polygon") or len(row["polygon"]) < 3:
            reason = "proposal has no usable mask polygon"
        elif float(row.get("box_score") or 0) < LEAF_MIN_SCORE:
            reason = "detector score below 0.30"
        elif float(row.get("mask_quality") or 0) < LEAF_MIN_QUALITY:
            reason = "SAM quality below 0.88"
        elif float(row.get("fill_ratio") or 0) < LEAF_MIN_FILL:
            reason = "mask fill below 0.36"
        elif not LEAF_MASK_MIN_AREA <= float(row.get("mask_area_fraction") or 0) <= LEAF_MASK_MAX_AREA:
            reason = "mask fraction outside [0.015, 0.48]"
        elif float(row.get("bbox_area_fraction") or 1) > LEAF_BBOX_MAX_AREA:
            reason = "box area exceeds 0.58 of frame"
        elif len(set((round(float(p[0]), 5), round(float(p[1]), 5)) for p in row["polygon"])) < 3:
            reason = "degenerate polygon"
        if reason:
            decisions.append({"proposal_id": row.get("proposal_id"), "image_id": image_id, "decision": "reject", "reason": reason, "box_score": row.get("box_score"), "mask_quality": row.get("mask_quality"), "mask_area_fraction": row.get("mask_area_fraction"), "bbox_area_fraction": row.get("bbox_area_fraction")})
            continue
        by_image[image_id].append({**row, "meta": meta, "split": bio_splits[image_id]})
    selected = []
    for image_id, candidates in sorted(by_image.items()):
        candidates.sort(key=lambda row: (
            float(row.get("mask_quality") or 0) * 0.35
            + float(row.get("box_score") or 0) * 0.25
            + float(row.get("fill_ratio") or 0) * 0.15
            + min(float(row.get("mask_area_fraction") or 0) / 0.10, 1.0) * 0.25,
            float(row.get("rank_score") or 0),
        ), reverse=True)
        chosen = candidates[0]
        selected.append(chosen)
        decisions.append({
            "proposal_id": chosen["proposal_id"],
            "image_id": image_id,
            "decision": "select",
            "reason": "best qualifying mask per image; score weights SAM quality 0.35, detector 0.25, fill 0.15, mask area capped at 0.10 fraction 0.25; source organ_tag=leaf",
            "selection_rank": 1,
            "split": chosen["split"],
            "box_score": chosen["box_score"],
            "mask_quality": chosen["mask_quality"],
            "mask_area_fraction": chosen["mask_area_fraction"],
            "bbox_area_fraction": chosen["bbox_area_fraction"],
        })
        for rejected in candidates[1:]:
            decisions.append({"proposal_id": rejected["proposal_id"], "image_id": image_id, "decision": "reject", "reason": "lower-ranked qualifying candidate; one instance proposal per image", "box_score": rejected["box_score"], "mask_quality": rejected["mask_quality"], "mask_area_fraction": rejected["mask_area_fraction"], "bbox_area_fraction": rejected["bbox_area_fraction"]})
    return selected, decisions


def real_media_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposals = read_jsonl(PROPOSAL_PATH)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    for row in proposals:
        if row.get("source") != "tuckaway_photos":
            continue
        reason = None
        if not row.get("polygon") or len(row["polygon"]) < 3:
            reason = "missing SAM polygon"
        elif float(row.get("box_score") or 0) < 0.25:
            reason = "detector score below 0.25"
        elif float(row.get("mask_quality") or 0) < 0.90:
            reason = "SAM quality below 0.90"
        elif float(row.get("fill_ratio") or 0) < 0.35:
            reason = "mask fill below 0.35"
        elif not TUCK_MASK_MIN_AREA <= float(row.get("mask_area_fraction") or 0) <= TUCK_MASK_MAX_AREA:
            reason = "mask fraction outside [0.008, 0.30]"
        elif float(row.get("bbox_area_fraction") or 1) > TUCK_BBOX_MAX_AREA:
            reason = "box area exceeds 0.42 of frame; scene/clump rejected"
        if reason:
            decisions.append({"proposal_id": row.get("proposal_id"), "image_id": row.get("image_id"), "source": row.get("source"), "decision": "reject", "reason": reason, "box_score": row.get("box_score"), "mask_quality": row.get("mask_quality"), "mask_area_fraction": row.get("mask_area_fraction"), "bbox_area_fraction": row.get("bbox_area_fraction")})
        else:
            grouped[str(row["image_id"])].append(row)
    selected = []
    for image_id, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda row: (
            float(row.get("mask_quality") or 0) * 0.35
            + float(row.get("box_score") or 0) * 0.25
            + float(row.get("fill_ratio") or 0) * 0.15
            + min(float(row.get("mask_area_fraction") or 0) / 0.08, 1.0) * 0.25,
            float(row.get("rank_score") or 0),
        ), reverse=True)
        chosen: list[dict[str, Any]] = []
        for candidate in candidates:
            if any(polygon_iou(candidate["polygon"], old["polygon"]) >= 0.58 for old in chosen):
                decisions.append({"proposal_id": candidate["proposal_id"], "image_id": image_id, "source": candidate["source"], "decision": "reject", "reason": "overlaps a higher-ranked selected proposal (bbox IoU >= 0.58)"})
                continue
            chosen.append(candidate)
            if len(chosen) == 3:
                break
        for index, candidate in enumerate(chosen, start=1):
            selected.append({**candidate, "selection_rank": index})
            decisions.append({"proposal_id": candidate["proposal_id"], "image_id": image_id, "source": candidate["source"], "decision": "select_train_or_external_holdout", "selection_rank": index, "reason": "reviewed close foliage candidate; source-tagged Tuckaway photo, SAM mask passes score/quality/fill/size rules; one image-level assignment shared by its proposals", "box_score": candidate["box_score"], "mask_quality": candidate["mask_quality"], "mask_area_fraction": candidate["mask_area_fraction"], "bbox_area_fraction": candidate["bbox_area_fraction"]})
        chosen_ids = {item["proposal_id"] for item in chosen}
        for candidate in candidates:
            if candidate["proposal_id"] not in chosen_ids and not any(d.get("proposal_id") == candidate["proposal_id"] for d in decisions[-len(candidates):]):
                decisions.append({"proposal_id": candidate["proposal_id"], "image_id": image_id, "source": candidate["source"], "decision": "reject", "reason": "not in top three non-overlapping qualifying masks", "box_score": candidate["box_score"], "mask_quality": candidate["mask_quality"], "mask_area_fraction": candidate["mask_area_fraction"], "bbox_area_fraction": candidate["bbox_area_fraction"]})
    return selected, decisions


def build_leafsnap_rows(leaf_splits: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_manifest = LEAFSNAP_ROOT / "leafsnap-dataset-images.txt"
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    rows = []
    rows_excluded = []
    for source in source_rows:
        file_id = str(source.get("file_id") or "").strip()
        domain = str(source.get("source") or "").strip().lower()
        if domain not in {"lab", "field"} or file_id not in leaf_splits:
            continue
        image_path = LEAFSNAP_ROOT / source["image_path"]
        mask_path = LEAFSNAP_ROOT / source["segmented_path"]
        if not image_path.is_file() or not mask_path.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        try:
            polygon, mask_info = mask_polygon(str(mask_path), (width, height))
        except ValueError as exc:
            rows_excluded.append({"source_id": file_id, "source_image": str(image_path), "mask_path": str(mask_path), "decision": "reject", "reason": str(exc)})
            continue
        xyxy = polygon_bbox(polygon, width, height)
        split = leaf_splits[file_id]
        uid = f"leafsnap_{file_id}"
        rows.append(make_row(uid, "leafsnap", file_id, str(image_path), split, width, height, source.get("species", ""), domain, None, "leaf", polygon, xyxy, False, "leafsnap_binary_mask", mask_info))
    return rows, rows_excluded


def make_row(uid: str, source: str, source_id: str, source_image: str, split: str, width: int, height: int, species: str, domain: str, individual: str | None, class_name: str, polygon: list[list[float]], xyxy: list[float], weak: bool, box_kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    class_id = CLASS_IDS[class_name]
    box = {
        "class_id": class_id,
        "class_name": class_name,
        "bbox_xyxy": [round(float(v), 3) for v in xyxy],
        "bbox_yolo": [round(float(v), 8) for v in bbox_yolo(xyxy, width, height)],
        "polygon_norm": [[round(float(x), 8), round(float(y), 8)] for x, y in polygon],
        "box_area_fraction": round(max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1]) / (width * height), 8),
        "box_kind": box_kind,
        "weak_box": weak,
    }
    if extra:
        box["mask"] = extra
    return {
        "uid": uid,
        "dataset": "organ_gate_v3",
        "source": source,
        "source_id": source_id,
        "source_image": source_image,
        "split": split,
        "image": f"images/{split}/{uid}.jpg",
        "label": f"labels/{split}/{uid}.txt",
        "width": width,
        "height": height,
        "species": species,
        "domain": domain,
        "individual_id": individual,
        "boxes": [box],
        "weak_box": weak,
        "box_kind": box_kind,
        "synthetic": False,
    }


def build_bioimages_rows(bio_splits: dict[str, str], v2_bio_rows: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    leaves, decisions = bioimages_leaf_rows(bio_splits)
    rows = []
    for proposal in leaves:
        image_id = str(proposal["image_id"])
        image_path = Path(proposal["image_path"])
        if not image_path.is_file():
            decisions.append({"proposal_id": proposal["proposal_id"], "image_id": image_id, "decision": "reject", "reason": "source image missing"})
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        polygon = proposal_polygon(proposal["polygon"])
        xyxy = polygon_bbox(polygon, width, height)
        meta = proposal["meta"]
        split = bio_splits[image_id]
        extra = {key: proposal.get(key) for key in ("proposal_id", "box_score", "mask_quality", "mask_area_fraction", "bbox_area_fraction", "fill_ratio", "rank_score")}
        rows.append(make_row(f"bio_leaf_{clean_id(image_id)}", "bioimages_sam_leaf", image_id, str(image_path), split, width, height, meta.get("species", ""), "bioimages_leaf_context", meta.get("individual_id"), "leaf", polygon, xyxy, False, "sam_mask_polygon", extra))
    for source_id, row in sorted(v2_bio_rows.items()):
        for old_box in row["boxes"]:
            class_name = old_box["class_name"]
            if class_name == "leaf":
                continue
            if class_name not in {"bud", "flower", "twig", "branch"}:
                continue
            source_image = row["source_image"]
            image_path = Path(source_image)
            if not image_path.is_file():
                continue
            width, height = int(row["width"]), int(row["height"])
            cx, cy, bw, bh = [float(v) for v in old_box["bbox_yolo"]]
            xyxy = [(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height]
            polygon = rectangle_polygon(xyxy, width, height)
            uid = f"bio_weak_{clean_id(source_id)}"
            extra = {"weak_box_policy": "v2 near-full-frame rectangular labels retained only for nonleaf classes", "organ_tag": row.get("organ_tag"), "subview": row.get("subview")}
            rows.append(make_row(uid, "bioimages_weak_nonleaf", source_id, source_image, row["split"], width, height, row.get("species", ""), "bioimages_nonleaf", row.get("individual_id"), class_name, polygon, xyxy, True, "weak_rectangular_polygon", extra))
    return rows, decisions


def assign_tuckaway_splits(selected: list[dict[str, Any]]) -> dict[str, str]:
    ids = sorted({str(row["image_id"]) for row in selected}, key=lambda key: stable_int("tuckaway-v3-image-split", key))
    train_count = max(1, int(math.floor(len(ids) * 0.70))) if len(ids) > 1 else len(ids)
    train_ids = set(ids[:train_count])
    return {image_id: "train" if image_id in train_ids else "external_test" for image_id in ids}


def build_tuckaway_rows(selected: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    assignments = assign_tuckaway_splits(selected)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_image[str(row["image_id"])].append(row)
    rows = []
    for image_id, proposals in sorted(by_image.items()):
        split = assignments[image_id]
        if split != "train":
            for proposal in proposals:
                decisions.append({"proposal_id": proposal["proposal_id"], "image_id": image_id, "source": "tuckaway_photos", "decision": "holdout_external_test", "reason": "deterministic image-level split; never copied into training"})
            continue
        image_path = Path(proposals[0]["image_path"])
        with Image.open(image_path) as image:
            width, height = image.size
        image_row = None
        for proposal in proposals:
            polygon = proposal_polygon(proposal["polygon"])
            xyxy = polygon_bbox(polygon, width, height)
            extra = {key: proposal.get(key) for key in ("proposal_id", "box_score", "mask_quality", "mask_area_fraction", "bbox_area_fraction", "fill_ratio", "rank_score")}
            row = make_row(f"tuck_{clean_id(image_id)}", "tuckaway_reviewed", image_id, str(image_path), "train", width, height, "unknown", "real_close_foliage", None, "leaf", polygon, xyxy, False, "reviewed_sam_mask_polygon", extra)
            if image_row is None:
                image_row = row
            else:
                image_row["boxes"].extend(row["boxes"])
        if image_row is not None:
            rows.append(image_row)
    return rows, assignments


def copy_dataset(rows: list[dict[str, Any]]) -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for row in rows:
        source = Path(row["source_image"])
        target = DATASET / row["image"]
        label = DATASET / row["label"]
        target.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        lines = []
        for box in row["boxes"]:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in box["polygon_norm"])
            lines.append(f"{box['class_id']} {coords}")
        label.write_text("\n".join(lines) + "\n")
    yaml = f"path: {DATASET}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 5\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    (DATASET / "dataset.yaml").write_text(yaml)
    ordered = sorted(rows, key=lambda row: (row["split"], row["source"], row["uid"]))
    (DATASET / "manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))


def summarize(rows: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    images = Counter(row["split"] for row in rows)
    source_split: dict[str, Counter[str]] = defaultdict(Counter)
    class_split: dict[str, Counter[str]] = defaultdict(Counter)
    area: dict[str, list[float]] = defaultdict(list)
    weak_by_class = Counter()
    by_source = Counter()
    for row in rows:
        source_split[row["split"]][row["source"]] += 1
        by_source[row["source"]] += 1
        for box in row["boxes"]:
            class_split[row["split"]][box["class_name"]] += 1
            area[box["class_name"]].append(box["box_area_fraction"])
            if box["weak_box"]:
                weak_by_class[box["class_name"]] += 1
    def quantiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        return {str(q): round(float(np.quantile(values, q)), 6) for q in (0, .1, .5, .9, .99, 1)}
    return {
        "version": 3,
        "classes": CLASSES,
        "seed": SEED,
        "total_images": len(rows),
        "total_instances": sum(len(row["boxes"]) for row in rows),
        "by_split_images": dict(images),
        "by_split_source": {split: dict(counts) for split, counts in source_split.items()},
        "by_split_class_instances": {split: dict(counts) for split, counts in class_split.items()},
        "by_source_images": dict(by_source),
        "weak_instances_by_class": dict(weak_by_class),
        "weak_leaf_instances": 0,
        "bbox_area_fraction_quantiles": {name: quantiles(values) for name, values in area.items()},
        "real_photo_split": {"selected_images": len(assignments), "train_images": sum(value == "train" for value in assignments.values()), "external_test_images": sum(value == "external_test" for value in assignments.values()), "assignments": assignments},
        "synthetic_images": 0,
    }


def validate(rows: list[dict[str, Any]], leaf_splits: dict[str, str], bio_splits: dict[str, str], assignments: dict[str, str]) -> dict[str, Any]:
    checks = []
    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
    bad_labels = []
    no_files = []
    bad_poly = []
    bad_class = []
    leaf_weak = []
    real_overlap = []
    used_leaf_species: dict[str, set[str]] = defaultdict(set)
    used_bio_individuals: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        image_path = DATASET / row["image"]
        label_path = DATASET / row["label"]
        if not image_path.is_file() or not label_path.is_file():
            no_files.append(row["uid"])
            continue
        with Image.open(image_path) as image:
            if image.size != (row["width"], row["height"]):
                bad_labels.append(row["uid"] + ": original image dimensions changed")
        lines = [line for line in label_path.read_text().splitlines() if line.strip()]
        if len(lines) != len(row["boxes"]):
            bad_labels.append(row["uid"] + ": label instance count mismatch")
        for line, box in zip(lines, row["boxes"]):
            values = [float(value) for value in line.split()]
            if len(values) < 7 or len(values) % 2 == 0:
                bad_poly.append(row["uid"] + ": malformed polygon label")
                continue
            if int(values[0]) != box["class_id"] or box["class_name"] not in CLASS_IDS:
                bad_class.append(row["uid"])
            points = np.asarray(values[1:], dtype=float).reshape(-1, 2)
            if (points < 0).any() or (points > 1).any() or len(points) < 3:
                bad_poly.append(row["uid"] + ": normalized coordinates outside bounds")
            if box["class_name"] == "leaf" and box["weak_box"]:
                leaf_weak.append(row["uid"])
        if row["source"] == "leafsnap":
            used_leaf_species[row["split"]].add(row["species"])
        if row["source"].startswith("bioimages") and row.get("individual_id"):
            used_bio_individuals[row["split"]].add(row["individual_id"])
        if row["source"] == "tuckaway_reviewed":
            if assignments.get(row["source_id"]) != "train" or row["split"] != "train":
                real_overlap.append(row["uid"])
    species_overlap = {
        "train_val": sorted(used_leaf_species["train"] & used_leaf_species["val"]),
        "train_test": sorted(used_leaf_species["train"] & used_leaf_species["test"]),
        "val_test": sorted(used_leaf_species["val"] & used_leaf_species["test"]),
    }
    individual_overlap = {
        "train_val": sorted(used_bio_individuals["train"] & used_bio_individuals["val"]),
        "train_test": sorted(used_bio_individuals["train"] & used_bio_individuals["test"]),
        "val_test": sorted(used_bio_individuals["val"] & used_bio_individuals["test"]),
    }
    add("exact_classes", CLASSES == ["leaf", "bud", "flower", "twig", "branch"], CLASSES)
    add("all_original_files_present", not no_files, no_files[:10])
    add("segmentation_labels_valid", not bad_poly, bad_poly[:10])
    add("labels_and_classes_valid", not bad_labels and not bad_class, {"labels": bad_labels[:10], "classes": bad_class[:10]})
    add("no_weak_leaf_boxes", not leaf_weak, leaf_weak[:10])
    add("leafsnap_species_disjoint", not any(species_overlap.values()), species_overlap)
    add("bioimages_individuals_disjoint", not any(individual_overlap.values()), individual_overlap)
    add("real_train_rows_only_from_train_assignments", not real_overlap, real_overlap[:10])
    add("no_synthetic_rows", not any(row.get("synthetic") for row in rows), sum(bool(row.get("synthetic")) for row in rows))
    add("no_entire_plant_class", all(box["class_name"] != "entire_plant" for row in rows for box in row["boxes"]), False)
    payload = {"ok": all(check["ok"] for check in checks), "checks": checks}
    write_json(DATASET / "validation.json", payload)
    return payload


def draw_contact_sheet(rows: list[dict[str, Any]], path: Path, title: str, limit: int = 36, seed: int = SEED) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: stable_int(seed, row["uid"]))[:limit]
    columns = 4
    cell_w, cell_h = 360, 300
    sheet = Image.new("RGB", (columns * cell_w, math.ceil(len(rows) / columns) * cell_h), (22, 25, 29))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
    for index, row in enumerate(rows):
        with Image.open(DATASET / row["image"]) as source:
            image = source.convert("RGB")
        scale = min((cell_w - 8) / image.width, (cell_h - 36) / image.height)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        ox = (index % columns) * cell_w + (cell_w - image.width) // 2
        oy = (index // columns) * cell_h + 3
        sheet.paste(image, (ox, oy))
        sx, sy = image.width / row["width"], image.height / row["height"]
        for box in row["boxes"]:
            points = [(ox + p[0] * row["width"] * sx, oy + p[1] * row["height"] * sy) for p in box["polygon_norm"]]
            color = {"leaf": (52, 236, 130), "bud": (255, 220, 60), "flower": (255, 70, 165), "twig": (180, 140, 80), "branch": (160, 110, 230)}[box["class_name"]]
            draw.line(points + [points[0]], fill=color, width=2, joint="curve")
            x1, y1, x2, y2 = box["bbox_xyxy"]
            draw.rectangle([ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy], outline=color, width=1)
        label = f"{row['source']} | {row['split']} | {row['uid']}"
        draw.rectangle([(index % columns) * cell_w, (index // columns) * cell_h + cell_h - 29, (index % columns + 1) * cell_w - 1, (index // columns + 1) * cell_h - 1], fill=(0, 0, 0))
        draw.text(((index % columns) * cell_w + 5, (index // columns) * cell_h + cell_h - 25), label[:58], fill=(245, 245, 245), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def write_data_doc(counts: dict[str, Any], validation: dict[str, Any], decisions: list[dict[str, Any]]) -> None:
    split_lines = []
    for split in ("train", "val", "test"):
        source = counts["by_split_source"].get(split, {})
        classes = counts["by_split_class_instances"].get(split, {})
        split_lines.append(f"| {split} | {counts['by_split_images'].get(split, 0)} | {classes.get('leaf', 0)} | {classes.get('bud', 0)} | {classes.get('flower', 0)} | {classes.get('twig', 0)} | {classes.get('branch', 0)} | {json.dumps(source, sort_keys=True)} |")
    text = "\n".join([
        "# Organ gate V3 dataset",
        "",
        "Generated by `experiments/leafsnap-yolo26/build_organ_gate_v3.py`. Dataset files and E1 contact sheets are under `scratch/leafsnap-yolo26/organ-gate-v3/`. No application files are changed.",
        "",
        "## Contract and counts",
        "",
        "Classes in fixed order: `leaf, bud, flower, twig, branch`. Training labels are YOLO segmentation polygons in normalized original-image coordinates. No synthetic composites, no `entire_plant`, and zero weak leaf rectangles.",
        "",
        f"Total: **{counts['total_images']} images / {counts['total_instances']} instances**. The counts below list per-image instances by class.",
        "",
        "| split | images | leaf | bud | flower | twig | branch | source rows |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        *split_lines,
        "",
        "## Segmentation supervision and real-media review",
        "",
        f"- LeafSnap: binary source masks converted to polygon contours at original pixel dimensions; split names are reused from V1's deterministic species-disjoint manifest. Selected rows: {counts['by_source_images'].get('leafsnap', 0)}; excluded degenerate/empty masks: {counts.get('leafsnap_mask_exclusions', 0)}.",
        f"- BioImages leaves: {counts['by_source_images'].get('bioimages_sam_leaf', 0)} selected real SAM masks; image-level `organ_tag=leaf` is required. Source individuals reuse V2's deterministic individual split. One candidate per image is selected by the documented score: SAM quality 0.35, detector score 0.25, mask fill 0.15, mask area (capped at 0.10 frame fraction) 0.25. Filter: score >= {LEAF_MIN_SCORE}, mask quality >= {LEAF_MIN_QUALITY}, fill >= {LEAF_MIN_FILL}, mask fraction [{LEAF_MASK_MIN_AREA},{LEAF_MASK_MAX_AREA}], box fraction <= {LEAF_BBOX_MAX_AREA}; polygons are mandatory. The size floor rejects tiny fragments; area/box caps reject clumps/full-frame candidates. `review_decisions.jsonl` preserves every proposal ID and selection/rejection reason.",
        f"- BioImages nonleaf: {counts['by_source_images'].get('bioimages_weak_nonleaf', 0)} existing V2 bud/flower/twig/branch rows remain documented weak rectangular polygons; no leaf rows use weak boxes.",
        f"- Tuckaway still photos: {counts['by_source_images'].get('tuckaway_reviewed', 0)} reviewed close-foliage training images, selected from SAM proposals with score >= 0.25, quality >= 0.90, fill >= 0.35, mask area [{TUCK_MASK_MIN_AREA},{TUCK_MASK_MAX_AREA}], bbox area <= {TUCK_BBOX_MAX_AREA}; up to three non-overlapping instances per image. Deterministic image-level assignment is recorded in `counts.json`; external-assigned photos are never in train.",
        "- Foliage video frames are held out by entire video source. Flower-video leaf proposals are excluded completely because flower heads were frequent semantic false positives. No held-out video media enter training.",
        "- V1 LeafSnap split: train/test/val species counts are inherited without reassigning IDs. All source originals are retained at native dimensions; labels normalize by those dimensions.",
        "",
        "## Validation and E1 review",
        "",
        f"Dataset validation: **{'PASS' if validation['ok'] else 'FAIL'}**. See `dataset/validation.json` for checks. `e1_contact_sheets/` contains reviewed polygon overlays for LeafSnap, BioImages SAM leaves, Tuckaway train, and nonleaf weak classes.",
        "",
        "Known limits: BioImages SAM masks are reviewed proposals, not human trace annotations; nonleaf rectangular polygons remain weak. Held-out external sets have no complete ground-truth leaf annotations, so external detection rates are not recall. Small real-media sample sizes are reported without padding.",
        "",
        f"Review decision rows: {len(decisions)}. Synthetic images: {counts['synthetic_images']}. Weak leaf boxes: {counts['weak_leaf_instances']}.",
        "",
    ])
    (OUT_ROOT / "ORGAN_GATE_V3_DATA.md").write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to replace existing output: {OUT_ROOT}; pass --overwrite")
    leaf_splits, bio_splits, v2_bio_rows = load_split_sources()
    leaf_rows, leaf_exclusions = build_leafsnap_rows(leaf_splits)
    bio_rows, bio_decisions = build_bioimages_rows(bio_splits, v2_bio_rows)
    tuck_candidates, tuck_decisions = real_media_candidates()
    tuck_rows, assignments = build_tuckaway_rows(tuck_candidates, tuck_decisions)
    rows = leaf_rows + bio_rows + tuck_rows
    copy_dataset(rows)
    counts = summarize(rows, assignments)
    counts["leafsnap_mask_exclusions"] = len(leaf_exclusions)
    validation = validate(rows, leaf_splits, bio_splits, assignments)
    write_json(DATASET / "counts.json", counts)
    write_json(OUT_ROOT / "review_decisions.json", {"leafsnap_mask_exclusions": leaf_exclusions, "bioimages_leaf": bio_decisions, "tuckaway_photos": tuck_decisions, "flower_video": "excluded: leaf proposals intentionally not admitted; flower heads are false positives"})
    write_json(OUT_ROOT / "external_media_split.json", {"tuckaway_photos": assignments, "video_foliage": "entire source held out; no video_foliage proposals in training", "video_flowers": "entire source held out; all leaf proposals excluded"})
    sheet_root = OUT_ROOT / "e1_contact_sheets"
    draw_contact_sheet([row for row in rows if row["source"] == "leafsnap" and row["domain"] == "field"], sheet_root / "leafsnap_field_masks.jpg", "LeafSnap field binary masks")
    draw_contact_sheet([row for row in rows if row["source"] == "bioimages_sam_leaf"], sheet_root / "bioimages_sam_leaf.jpg", "BioImages SAM leaf candidates")
    draw_contact_sheet([row for row in rows if row["source"] == "tuckaway_reviewed"], sheet_root / "tuckaway_train.jpg", "Reviewed Tuckaway training masks")
    draw_contact_sheet([row for row in rows if row["source"] == "bioimages_weak_nonleaf"], sheet_root / "bioimages_nonleaf_weak.jpg", "Documented weak nonleaf rectangles")
    write_data_doc(counts, validation, bio_decisions + tuck_decisions)
    print(json.dumps({"counts": counts, "validation": validation, "bioimages_leaf_selected": sum(r["source"] == "bioimages_sam_leaf" for r in rows), "tuckaway_train_images": sum(r["source"] == "tuckaway_reviewed" for r in rows), "output": str(OUT_ROOT)}, indent=2))
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
