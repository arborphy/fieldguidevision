# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_datasets as base

DEFAULT_LEAFSNAP_ROOT = Path("/Users/valkyrie/Desktop/arborphy product/leafsnap/leafsnap-dataset")
DEFAULT_MAPPING = HERE / "feature_assertions.json"
DEFAULT_OUT = ROOT / "scratch" / "leafsnap-yolo26" / "classification-v2"
CLASSES = ["simple_leaf", "compound_leaf"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
MIXED_SPECIES = {"Ginkgo biloba", "Gymnocladus dioicus"}
CLASSIFIER_CAPS = {
    "simple_leaf": {
        "train": {"lab": 40, "field": 15},
        "val": {"lab": 15, "field": 15},
        "test": {"lab": 10, "field": None},
    },
    "compound_leaf": {
        "train": {"lab": 40, "field": 0},
        "val": {"lab": 15, "field": 15},
        "test": {"lab": 10, "field": None},
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_audited_labels(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    eligible: dict[str, dict[str, Any]] = {}
    observed_mixed: set[str] = set()
    class_counts: Counter[str] = Counter()
    match_rules: Counter[str] = Counter()
    match_confidence: Counter[str] = Counter()

    for row in audit["species"]:
        name = row["leafsnap_species"]
        values = row.get("features", {}).get("leaf_type", [])
        if values == ["Simple"]:
            class_name = "simple_leaf"
        elif values == ["Compound"]:
            class_name = "compound_leaf"
        else:
            if set(values) == {"Simple", "Compound"}:
                observed_mixed.add(name)
            continue
        if name in eligible:
            raise ValueError(f"duplicate audited mapping for {name}")
        mapping = {
            "leafsnap_species": name,
            "dirr_species": row["dirr_species"],
            "match_rule": row["match_rule"],
            "match_confidence": row["match_confidence"],
            "match_note": row.get("match_note", ""),
            "dirr_book_page": row.get("dirr_book_page"),
            "leaf_type_values": values,
        }
        eligible[name] = {"class_name": class_name, "mapping": mapping}
        class_counts[class_name] += 1
        match_rules[row["match_rule"]] += 1
        match_confidence[row["match_confidence"]] += 1

    expected = {"simple_leaf": 101, "compound_leaf": 21}
    if dict(class_counts) != expected or len(eligible) != 122:
        raise ValueError(f"audited exact-label counts changed: {dict(class_counts)}")
    if observed_mixed != MIXED_SPECIES:
        raise ValueError(f"mixed species changed: {sorted(observed_mixed)}")
    unmatched_names = {row["leafsnap_species"] for row in audit["unmatched"]}
    if len(unmatched_names) != 20 or unmatched_names & set(eligible):
        raise ValueError("unmatched audit is inconsistent or an unmatched species entered the eligible set")

    eligible_non_exact_count = sum(
        row["mapping"]["match_rule"] != "exact" for row in eligible.values()
    )
    metadata = {
        "eligible_species_count": len(eligible),
        "reviewed_non_exact_match_count": len(audit["match_audit"]),
        "eligible_reviewed_non_exact_match_count": eligible_non_exact_count,
        "reviewed_non_exact_without_exact_leaf_type_count": len(audit["match_audit"]) - eligible_non_exact_count,
        "eligible_class_counts": dict(sorted(class_counts.items())),
        "eligible_match_rule_counts": dict(sorted(match_rules.items())),
        "eligible_match_confidence_counts": dict(sorted(match_confidence.items())),
        "mixed_species_excluded": sorted(observed_mixed),
        "unmatched_species_not_force_matched": sorted(unmatched_names),
        "unmatched_count": len(unmatched_names),
        "matched_species_without_exact_simple_or_compound": sum(
            1
            for row in audit["species"]
            if row.get("features", {}).get("leaf_type") not in (["Simple"], ["Compound"])
            and set(row.get("features", {}).get("leaf_type", [])) != {"Simple", "Compound"}
        ),
        "eda_meta": audit["meta"],
        "mapping_sha256": sha256_file(path),
    }
    return eligible, metadata


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite to rebuild it")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            (path / split / class_name).mkdir(parents=True)


def crop_box(record: Any) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = record.bbox_px
    margin = base.CLASSIFIER_MARGIN_FRAC
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    cx1, cy1, cx2, cy2 = base.clamp_box(
        x1 - mx,
        y1 - my,
        x2 + mx,
        y2 + my,
        record.width,
        record.height,
    )
    ix1, iy1 = int(math.floor(cx1)), int(math.floor(cy1))
    ix2, iy2 = int(math.ceil(cx2)), int(math.ceil(cy2))
    if ix2 - ix1 < 16 or iy2 - iy1 < 16:
        raise ValueError("mask crop is smaller than 16 pixels after margin")
    return ix1, iy1, ix2, iy2


def build_dataset(
    leafsnap_root: Path = DEFAULT_LEAFSNAP_ROOT,
    output_root: Path = DEFAULT_OUT,
    mapping_path: Path = DEFAULT_MAPPING,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not leafsnap_root.is_dir():
        raise FileNotFoundError(f"LeafSnap root not found: {leafsnap_root}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"EDA mapping not found: {mapping_path}")
    eligible, mapping_audit = load_audited_labels(mapping_path)
    prepare_output(output_root, overwrite)

    print("Scanning LeafSnap images and masks ...", flush=True)
    records, scan_stats = base.scan_leafsnap(leafsnap_root)
    source_species = {record.species for record in records}
    missing_species = sorted(set(eligible) - source_species)
    if missing_species:
        raise RuntimeError(f"eligible species have no usable source image/mask: {missing_species}")
    if set(eligible) & set(mapping_audit["unmatched_species_not_force_matched"]):
        raise RuntimeError("a no-match EDA species was included")

    species_to_class = {name: row["class_name"] for name, row in eligible.items()}
    splits = base.classifier_species_split(species_to_class)
    split_sets = {name: set(values) for name, values in splits.items()}
    if any(
        split_sets[a] & split_sets[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("species leakage across train/validation/test")
    if set.union(*split_sets.values()) != set(eligible):
        raise RuntimeError("species split omitted or duplicated eligible species")

    by_species: dict[str, list[Any]] = {}
    for record in records:
        if record.species in eligible:
            by_species.setdefault(record.species, []).append(record)

    rows: list[dict[str, Any]] = []
    crop_failures: list[dict[str, str]] = []
    split_class_domain: dict[str, dict[str, dict[str, int]]] = {}
    species_crop_counts: dict[str, dict[str, int]] = {}
    seen_uids: set[str] = set()

    for split in ("train", "val", "test"):
        split_class_domain[split] = {
            class_name: {"field": 0, "lab": 0} for class_name in CLASSES
        }
        for species in splits[split]:
            class_name = species_to_class[species]
            caps = CLASSIFIER_CAPS[class_name][split]
            selected = base.sample_by_domain(by_species[species], caps, split)
            species_crop_counts[f"{split}:{species}"] = {"class_name": class_name, "selected": 0}
            for record in selected:
                uid = record.uid
                if uid in seen_uids:
                    raise RuntimeError(f"duplicate crop id: {uid}")
                seen_uids.add(uid)
                try:
                    bbox = crop_box(record)
                    with Image.open(record.image_path) as image:
                        image.crop(bbox).convert("RGB").save(
                            output_root / split / class_name / f"{uid}.jpg",
                            quality=92,
                            subsampling=0,
                        )
                except Exception as exc:
                    crop_failures.append({"uid": uid, "species": species, "error": repr(exc)})
                    continue
                x1, y1, x2, y2 = bbox
                mapping = eligible[species]["mapping"]
                rows.append(
                    {
                        "uid": uid,
                        "dataset": "leaf_type_classifier_v2",
                        "source": "leafsnap",
                        "source_id": record.file_id,
                        "split": split,
                        "class_id": CLASS_TO_ID[class_name],
                        "class_name": class_name,
                        "image": f"{split}/{class_name}/{uid}.jpg",
                        "species": species,
                        "domain": record.domain,
                        "crop_width": x2 - x1,
                        "crop_height": y2 - y1,
                        "image_width": record.width,
                        "image_height": record.height,
                        "full_bbox_xyxy": [int(round(v)) for v in record.bbox_px],
                        "crop_bbox_xyxy": list(bbox),
                        "margin_fraction": base.CLASSIFIER_MARGIN_FRAC,
                        "source_image": str(record.image_path),
                        "mask_path": str(record.mask_path),
                        "mask_geometry": "normalized mask foreground bbox scaled to source image, expanded by 10% on each side and clamped",
                        "dirr_mapping": mapping,
                        "mapping_artifact": str(mapping_path),
                        "mapping_sha256": mapping_audit["mapping_sha256"],
                    }
                )
                split_class_domain[split][class_name][record.domain] += 1
                species_crop_counts[f"{split}:{species}"]["selected"] += 1

    if not rows:
        raise RuntimeError("no usable classification crops were written")
    retained_species = {row["species"] for row in rows}
    if retained_species != set(eligible):
        raise RuntimeError(
            f"usable crop species differ from all audited eligible species: "
            f"missing={sorted(set(eligible) - retained_species)} "
            f"unexpected={sorted(retained_species - set(eligible))}"
        )

    by_split_class = {}
    by_split_domain = {}
    split_species = {}
    split_species_counts = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        by_split_class[split] = {
            class_name: sum(row["class_name"] == class_name for row in split_rows)
            for class_name in CLASSES
        }
        by_split_domain[split] = {
            domain: sum(row["domain"] == domain for row in split_rows)
            for domain in ("field", "lab")
        }
        split_species[split] = sorted(splits[split])
        split_species_counts[split] = len(splits[split])

    compound_train_field = split_class_domain["train"]["compound_leaf"]["field"]
    if compound_train_field != 0:
        raise RuntimeError(f"compound field training rows found: {compound_train_field}")
    if split_class_domain["train"]["compound_leaf"]["lab"] == 0:
        raise RuntimeError("no lab compound crops available for training")
    if split_class_domain["val"]["compound_leaf"]["field"] == 0:
        raise RuntimeError("no field compound validation crops retained")
    if split_class_domain["test"]["compound_leaf"]["field"] == 0:
        raise RuntimeError("no field compound test crops retained")

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (r["split"], r["class_name"], r["uid"])):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    source_manifest = leafsnap_root / "leafsnap-dataset-images.txt"
    revision = {
        "version": 2,
        "name": "audited-eda-label-map-with-view-valid-compound-training",
        "eligible_species": mapping_audit["eligible_species_count"],
        "eligible_class_counts": mapping_audit["eligible_class_counts"],
        "mapping_artifact": str(mapping_path),
        "mapping_sha256": mapping_audit["mapping_sha256"],
        "leafsnap_source_root": str(leafsnap_root),
        "leafsnap_manifest": str(source_manifest),
        "leafsnap_manifest_sha256": sha256_file(source_manifest),
        "dataset_builder": str(HERE / "build_leaf_specialist_v2.py"),
        "dataset_builder_sha256": sha256_file(HERE / "build_leaf_specialist_v2.py"),
        "crop_source_builder": str(HERE / "build_datasets.py"),
        "crop_source_builder_sha256": sha256_file(HERE / "build_datasets.py"),
        "split_algorithm": "build_datasets.classifier_species_split; class-stratified SHA-256 ordering; 20% test, 15% validation, remainder train; species are indivisible",
        "split_salt": base.CLASSIFIER_SALT,
        "sample_caps": CLASSIFIER_CAPS,
        "compound_training_domain_policy": "compound_leaf training uses LAB only; field compound crops are retained in validation and test",
        "simple_training_domain_policy": "simple_leaf training retains both LAB and FIELD crops",
        "crop_geometry": {
            "bbox": "normalized mask foreground bbox scaled to the original image",
            "margin_fraction": base.CLASSIFIER_MARGIN_FRAC,
            "margin_rule": "10% of bbox width/height on each side, clamped",
            "jpeg_quality": 92,
            "jpeg_subsampling": 0,
        },
    }
    labels = {
        "task": "image_classification",
        "dataset_version": "leaf-specialist-v2",
        "classes": CLASS_TO_ID,
        "class_names": CLASSES,
        "source": "leafsnap",
        "label_source": "Audited experiments/leafsnap-yolo26/feature_assertions.json; feature leaf_type is exactly ['Simple'] or ['Compound']",
        "crop": revision["crop_geometry"],
        "revision": revision,
        "mapping_audit": mapping_audit,
        "splits": {
            split: {
                "species": split_species[split],
                "classes": {
                    class_name: sum(species_to_class[s] == class_name for s in split_species[split])
                    for class_name in CLASSES
                },
            }
            for split in ("train", "val", "test")
        },
    }
    counts = {
        "dataset_version": "leaf-specialist-v2",
        "total_crops": len(rows),
        "by_split_class": by_split_class,
        "by_split_class_domain": split_class_domain,
        "by_split_domain": by_split_domain,
        "split_species_counts": split_species_counts,
        "split_species": split_species,
        "source_species_counts": mapping_audit["eligible_class_counts"],
        "species_crop_counts": species_crop_counts,
        "eligible_species_count": mapping_audit["eligible_species_count"],
        "eligible_species_by_class": mapping_audit["eligible_class_counts"],
        "mapping_audit": mapping_audit,
        "revision": revision,
        "leafsnap_scan": scan_stats,
        "crop_failures": crop_failures,
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output_root / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
    (output_root / "counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")

    report_path = HERE / "LEAF_SPECIALIST_V2_DATA.md"
    write_data_report(report_path, output_root, labels, counts)
    return {"output_root": str(output_root), "labels": labels, "counts": counts, "report": str(report_path)}


def write_data_report(path: Path, dataset_root: Path, labels: dict[str, Any], counts: dict[str, Any]) -> None:
    revision = counts["revision"]
    audit = counts["mapping_audit"]
    rules = audit["eligible_match_rule_counts"]
    lines = [
        "# Leaf specialist v2 dataset revision",
        "",
        "This is the single E4 revision requested after v1 exposed a view/label mismatch in field compound crops. It changes the audited species join and the compound training views only; it does not remove species based on validation or test performance.",
        "",
        "## Audited label mapping",
        "",
        f"- Mapping: `{revision['mapping_artifact']}` (SHA-256 `{revision['mapping_sha256']}`).",
        f"- EDA inputs: tags `{audit['eda_meta']['inputs']['tags_sha256']}`, vocabulary `{audit['eda_meta']['inputs']['vocab_sha256']}`, LeafSnap name manifest `{audit['eda_meta']['inputs']['manifest_sha256']}`.",
        f"- Exact eligible labels: {audit['eligible_species_count']} species — {audit['eligible_class_counts']['simple_leaf']} simple and {audit['eligible_class_counts']['compound_leaf']} compound.",
        f"- Match rules among eligible species: {json.dumps(rules, sort_keys=True)}.",
        f"- Match confidence: {json.dumps(audit['eligible_match_confidence_counts'], sort_keys=True)}.",
        f"- Mixed and excluded: {', '.join(audit['mixed_species_excluded'])} (both assert Simple and Compound).",
        f"- Unmatched and not force-matched: {audit['unmatched_count']} names recorded in `classification-v2/counts.json`; suggested matches in the EDA are not applied.",
        f"- Matched species without exactly one Simple/Compound value remain excluded: {audit['matched_species_without_exact_simple_or_compound']}; conifer-only values are not coerced into a class.",
        f"- This uses the EDA's audited `leafsnap_species` → `dirr_species` mapping: {audit['eligible_reviewed_non_exact_match_count']} of {audit['reviewed_non_exact_match_count']} reviewed non-exact name matches have an exact single Simple/Compound assertion and are included; the rest remain excluded. The exact-name-only builder's 109-species set is superseded; no unsupported name joins are added.",
        "",
        "## Split and sample policy",
        "",
        f"- Split function: `{revision['split_algorithm']}`.",
        f"- Split salt: `{revision['split_salt']}`. Each species is assigned to exactly one split; image/crop rows cannot cross splits.",
        "- Original class-stratified caps are retained: train 40 lab + 15 field; validation 15 + 15; test 10 lab + all available field per species, except for the compound-training domain rule below.",
        "- Simple training uses both lab and field crops. Compound training uses only lab crops (pressed whole-architecture specimens); all field compound crops remain available in validation/test, subject to the existing caps.",
        "- No species is removed due to per-species or aggregate validation/test accuracy. Species with weak performance remain reported in `LEAF_SPECIALIST.md`.",
        "",
        "## Crop geometry and provenance",
        "",
        f"- Source root: `{revision['leafsnap_source_root']}`.",
        f"- LeafSnap manifest SHA-256: `{revision['leafsnap_manifest_sha256']}`.",
        f"- Dataset generator: `{revision['dataset_builder']}` (SHA-256 `{revision['dataset_builder_sha256']}`).",
        f"- Geometry: {revision['crop_geometry']['bbox']}; margin {revision['crop_geometry']['margin_fraction']:.2f} per side, clamped; JPEG quality {revision['crop_geometry']['jpeg_quality']}, subsampling {revision['crop_geometry']['jpeg_subsampling']}.",
        f"- Shared scan implementation: `{revision['crop_source_builder']}` (SHA-256 `{revision['crop_source_builder_sha256']}`), including its image/mask validity filters and normalized-mask-to-source-image scaling.",
        "- Every manifest row stores the source image, mask path, crop/full bounding boxes, exact EDA mapping row fields, mapping artifact path, and mapping hash.",
        "",
        "## Counts",
        "",
        "| split | simple | compound | field | lab | species |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {counts['by_split_class'][split]['simple_leaf']} | "
            f"{counts['by_split_class'][split]['compound_leaf']} | "
            f"{counts['by_split_domain'][split]['field']} | {counts['by_split_domain'][split]['lab']} | "
            f"{counts['split_species_counts'][split]} |"
        )
    lines += [
        "",
        f"Total retained crops: {counts['total_crops']:,}; species: {audit['eligible_species_count']}.",
        f"Compound training FIELD crops: {counts['by_split_class_domain']['train']['compound_leaf']['field']} (must be zero). Compound field validation/test crops: {counts['by_split_class_domain']['val']['compound_leaf']['field']} / {counts['by_split_class_domain']['test']['compound_leaf']['field']}.",
        f"Species are disjoint: {json.dumps(counts['split_species_counts'], sort_keys=True)}; see `labels.json` for the full species list by split.",
        "",
        "## View-validity limit",
        "",
        "The compound training restriction uses pressed lab specimens to favor whole-architecture views. Field crops are intentionally retained in validation and test; they can still show a single leaflet instead of an entire compound leaf. Therefore field compound recall includes a view-validity noise floor: a visually simple-looking leaflet inherits its species-level compound label. This is not a reason to remove Juglans cinerea or any other weak test species.",
        "",
        f"Dataset output: `{dataset_root}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leafsnap-root", type=Path, default=DEFAULT_LEAFSNAP_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = build_dataset(args.leafsnap_root, args.output, args.mapping, args.overwrite)
    counts = result["counts"]
    print(
        json.dumps(
            {
                "output_root": result["output_root"],
                "eligible_species": counts["eligible_species_count"],
                "total_crops": counts["total_crops"],
                "by_split_class": counts["by_split_class"],
                "by_split_class_domain": counts["by_split_class_domain"],
                "split_species_counts": counts["split_species_counts"],
                "report": result["report"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
