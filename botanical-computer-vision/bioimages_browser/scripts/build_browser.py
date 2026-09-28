#!/usr/bin/env python3
"""Build the static BioImages browser from the canonical Drive corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "organ_benchmark"

GROUPS = (
    ("whole-tree", "Whole tree", {"whole tree (or vine)", "whole tree", "whole plant"}),
    ("bark", "Bark", {"bark"}),
    ("twig", "Twig", {"twig", "stem"}),
    ("leaf", "Leaf", {"leaf"}),
    ("flower", "Flower / Inflorescence", {"inflorescence"}),
    ("fruit", "Fruit", {"fruit"}),
    ("cone", "Cone", {"cone"}),
    ("seed", "Seed", {"seed"}),
)
GROUP_ORDER = [slug for slug, _, _ in GROUPS] + ["other"]


def https(value: str | None) -> str:
    return (value or "").replace("http://bioimages.vanderbilt.edu", "https://bioimages.vanderbilt.edu")


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def group_for(category: str) -> tuple[str, str]:
    for slug, label, categories in GROUPS:
        if category in categories:
            return slug, label
    return "other", "Other"


def read_predictions() -> dict[str, dict]:
    """Attach only held-out DINOv3 predictions; never overwrite BioImages labels."""
    merged: dict[str, dict] = defaultdict(dict)
    for task in ("organ_tag", "organ_category"):
        path = BENCHMARK / "predictions" / f"individual_disjoint_{task}_dinov3.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                merged[row["image_id"]][task] = {
                    "label": row["predicted_label"],
                    "confidence": round(float(row["confidence"]), 4),
                    "match": row["top1_correct"].lower() == "true",
                    "protocol": "individual-disjoint held-out test",
                }
    return dict(merged)


def read_model_evaluation() -> dict:
    summary_path = BENCHMARK / "metrics" / "benchmark_summary.csv"
    per_class_path = BENCHMARK / "metrics" / "per_class_metrics.csv"
    if not summary_path.exists():
        return {"available": False, "summary": [], "per_class": []}

    with summary_path.open(newline="") as handle:
        summary = [
            {
                "model": r["model"],
                "model_key": r["model_key"],
                "protocol": r["protocol"],
                "task": r["task"],
                "test_images": int(r["test_images"]),
                "top1": round(float(r["top1_accuracy"]), 4),
                "macro_f1": round(float(r["macro_f1"]), 4),
            }
            for r in csv.DictReader(handle)
            if r["task"] == "organ_tag"
        ]

    per_class = []
    if per_class_path.exists():
        with per_class_path.open(newline="") as handle:
            per_class = [
                {
                    "model": r["model"],
                    "model_key": r["model_key"],
                    "label": r["canonical_label"],
                    "precision": round(float(r["precision"]), 4),
                    "recall": round(float(r["recall"]), 4),
                    "f1": round(float(r["f1"]), 4),
                    "support": int(r["support"]),
                }
                for r in csv.DictReader(handle)
                if r["task"] == "organ_tag" and r["protocol"] == "individual_disjoint"
            ]

    return {
        "available": True,
        "summary": summary,
        "per_class": per_class,
        "benchmark_url": "../organ_benchmark/index.html",
        "confusions": {
            key: f"../organ_benchmark/figures/confusion_individual_disjoint_organ_tag_{key}.png"
            for key in ("bioclip", "dinov3", "efficientnet_b0")
        },
    }


def representatives(images: list[dict], limit: int = 5) -> list[str]:
    chosen: list[str] = []
    seen_species: set[str] = set()
    for image in sorted(images, key=lambda x: stable_key("browser-rep-v1|" + x["id"])):
        if image["species"] in seen_species:
            continue
        chosen.append(image["id"])
        seen_species.add(image["species"])
        if len(chosen) == limit:
            break
    return chosen


def build(corpus_path: Path) -> dict:
    corpus = json.loads(corpus_path.read_text())
    if corpus.get("image_count") != 1899 or corpus.get("scene_count") != 85:
        raise ValueError("Unexpected corpus size; expected 1,899 images and 85 species")

    predictions = read_predictions()
    images: list[dict] = []
    species_rows: list[dict] = []
    seen_ids: set[str] = set()

    for scene in corpus["scenes"]:
        species_image_ids = []
        for source in scene["images"]:
            image_id = source["image_id"]
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image_id: {image_id}")
            seen_ids.add(image_id)
            slug, group_label = group_for(source["organ_category"])
            primary = f'{source["organ_category"]} / {source["subview"]}'
            record = {
                "id": image_id,
                "file": source["file"],
                "species": scene["scientific_name"],
                "common_name": scene.get("vernacular", ""),
                "author": scene.get("author", ""),
                "family": scene.get("family", ""),
                "order": scene.get("order", ""),
                "tsn": scene.get("tsn", ""),
                "species_page": https(scene.get("species_page")),
                "organ_tag": source["organ_tag"],
                "organ_category": source["organ_category"],
                "subview": source["subview"],
                "view_code": source["view_code"],
                "primary_label": primary,
                "group": slug,
                "group_label": group_label,
                "individual_id": source["individual_id"],
                "thumbnail_url": https(source["source_urls"].get("tn")),
                "image_url": https(source["source_urls"].get("gq") or source["source_urls"].get("lq")),
                "source_url": https(source.get("image_uri")),
                "archive_url": source.get("bq_url", ""),
                "captured_at": source.get("captured_at", ""),
                "creator": source.get("creator", ""),
                "credit": source.get("credit", ""),
                "rights": source.get("rights", ""),
                "license": source.get("license", ""),
                "usage_terms": source.get("usage_terms", ""),
                "locality": source.get("locality", ""),
                "county": source.get("county", ""),
                "state_province": source.get("state_province", ""),
                "country_code": source.get("country_code", ""),
                "latitude": source.get("latitude", ""),
                "longitude": source.get("longitude", ""),
                "establishment_means": source.get("establishment_means", ""),
                "title": source.get("title", ""),
                "prediction": predictions.get(image_id),
                "review_flags": [
                    flag
                    for flag, present in (
                        ("unspecified view", "unspecified" in primary.lower()),
                        ("missing creator", not source.get("creator")),
                        ("other organ group", slug == "other"),
                    )
                    if present
                ],
            }
            images.append(record)
            species_image_ids.append(image_id)

        species_rows.append(
            {
                "slug": scene["scene_id"],
                "name": scene["scientific_name"],
                "common_name": scene.get("vernacular", ""),
                "family": scene.get("family", ""),
                "author": scene.get("author", ""),
                "tsn": scene.get("tsn", ""),
                "source_url": https(scene.get("species_page")),
                "image_ids": species_image_ids,
            }
        )

    images.sort(key=lambda x: (x["species"], GROUP_ORDER.index(x["group"]), x["subview"], x["id"]))
    species_rows.sort(key=lambda x: x["name"])
    by_id = {image["id"]: image for image in images}
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_individual: dict[str, list[dict]] = defaultdict(list)
    for image in images:
        by_group[image["group"]].append(image)
        by_individual[image["individual_id"]].append(image)

    organ_groups = []
    for slug, label, _ in GROUPS:
        members = by_group[slug]
        organ_groups.append(
            {
                "slug": slug,
                "label": label,
                "count": len(members),
                "subviews": [
                    {"name": name, "count": count}
                    for name, count in sorted(Counter(i["subview"] for i in members).items())
                ],
                "representatives": representatives(members),
            }
        )
    if by_group.get("other"):
        members = by_group["other"]
        organ_groups.append(
            {
                "slug": "other",
                "label": "Other / Unspecified",
                "count": len(members),
                "subviews": [
                    {"name": name, "count": count}
                    for name, count in sorted(Counter(i["subview"] for i in members).items())
                ],
                "representatives": representatives(members),
            }
        )

    individuals = []
    for individual_id, members in sorted(by_individual.items()):
        individuals.append(
            {
                "id": individual_id,
                "species": members[0]["species"],
                "common_name": members[0]["common_name"],
                "image_ids": [m["id"] for m in members],
                "groups": sorted({m["group"] for m in members}, key=GROUP_ORDER.index),
                "representatives": representatives(members, 3),
            }
        )

    category_counts = Counter(i["organ_category"] for i in images)
    subview_counts = Counter(i["subview"] for i in images)
    exact_counts = Counter(i["primary_label"] for i in images)
    review_images = [i for i in images if i["review_flags"]]
    missing_primary = [i["id"] for i in images if not i["organ_category"] or not i["subview"]]
    stats = {
        "images": len(images),
        "species": len(species_rows),
        "individuals": len(individuals),
        "organ_categories": len(category_counts),
        "view_types": len(subview_counts),
        "exact_primary_labels": len(exact_counts),
        "complete_primary_labels": len(images) - len(missing_primary),
        "prediction_images": sum(bool(i["prediction"]) for i in images),
        "manual_review_images": len(review_images),
    }

    data = {
        "meta": {
            "title": "BioImages Browser",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_schema": corpus.get("schema"),
            "source_generated_at": corpus.get("generated_at"),
            "source_corpus_sha256": "sha256:" + hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "drive_file_id": "19HAirBJL3UBSAuKMV1IPdF8LvJHmKXUB",
        },
        "stats": stats,
        "organ_groups": organ_groups,
        "species": species_rows,
        "individuals": individuals,
        "images": images,
        "taxonomy": {
            "organ_categories": [
                {"name": name, "count": count} for name, count in sorted(category_counts.items())
            ],
            "subviews": [
                {"name": name, "count": count} for name, count in sorted(subview_counts.items())
            ],
            "exact_labels": [
                {"name": name, "count": count} for name, count in sorted(exact_counts.items())
            ],
        },
        "model_evaluation": read_model_evaluation(),
        "review": {
            "image_ids": [i["id"] for i in review_images],
            "missing_primary_label_ids": missing_primary,
            "reason_counts": dict(sorted(Counter(f for i in review_images for f in i["review_flags"]).items())),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "data" / "site-data.js").write_text(
        "window.BIOIMAGES_DATA=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    )
    validation = {
        "valid": not missing_primary and len(images) == 1899 and len(species_rows) == 85,
        "counts": stats,
        "unique_image_ids": len(seen_ids),
        "source_corpus_sha256": data["meta"]["source_corpus_sha256"],
        "missing_primary_label_ids": missing_primary,
        "review_reason_counts": data["review"]["reason_counts"],
    }
    (OUT / "data" / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    manifest_fields = [
        "id", "file", "species", "common_name", "family", "organ_tag", "organ_category",
        "subview", "view_code", "primary_label", "group", "individual_id", "captured_at",
        "creator", "source_url", "thumbnail_url", "image_url", "review_flags",
    ]
    with (OUT / "data" / "image_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for image in images:
            row = {key: image.get(key, "") for key in manifest_fields}
            row["review_flags"] = "; ".join(image["review_flags"])
            writer.writerow(row)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    validation = build(args.corpus)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
