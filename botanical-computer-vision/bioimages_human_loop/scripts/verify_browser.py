#!/usr/bin/env python3
"""Verify coverage and index integrity for the generated static browser."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw = (ROOT / "data" / "site-data.js").read_text()
    data = json.loads(raw.removeprefix("window.BIOIMAGES_DATA=").removesuffix(";\n"))
    images = data["images"]
    image_ids = {image["id"] for image in images}
    assert len(images) == 1899
    assert len(image_ids) == 1899
    assert len(data["species"]) == 85
    assert len(data["individuals"]) == 492
    assert data["stats"]["complete_primary_labels"] == 1899
    assert not data["review"]["missing_primary_label_ids"]

    required = (
        "id", "species", "organ_tag", "organ_category", "subview", "view_code",
        "primary_label", "individual_id", "thumbnail_url", "image_url", "source_url",
        "captured_at", "credit", "license",
    )
    for image in images:
        missing = [field for field in required if not image.get(field)]
        assert not missing, (image["id"], missing)
        assert image["primary_label"] == f'{image["organ_category"]} / {image["subview"]}'
        assert image["thumbnail_url"].startswith("https://")
        assert image["image_url"].startswith("https://")
        assert image["source_url"].startswith("https://")

    species_ids = [image_id for species in data["species"] for image_id in species["image_ids"]]
    individual_ids = [image_id for individual in data["individuals"] for image_id in individual["image_ids"]]
    assert len(species_ids) == 1899 and set(species_ids) == image_ids
    assert len(individual_ids) == 1899 and set(individual_ids) == image_ids
    assert sum(group["count"] for group in data["organ_groups"]) == 1899
    assert data["stats"]["prediction_images"] == 211

    tagging = data.get("tagging_analysis", {})
    assert tagging.get("available") is True
    assert tagging["gemma_images"] == 1899
    assert len(tagging["summary"]) == 3
    assert len(tagging["vocabulary"]) >= 30
    assert {row["model_key"] for row in tagging["summary"]} == {
        "dinov3", "bioclip", "efficientnet_b0",
    }
    assert tagging["method"]["fine_tuning"] is False
    for image in images:
        result = image.get("tagging")
        assert result, image["id"]
        assert result["gemma_tags"], image["id"]
        assert result["gemma_normalized_tags"], image["id"]
        assert set(result["models"]) == {"dinov3", "bioclip", "efficientnet_b0"}
        for prediction in result["models"].values():
            assert prediction["tags"], (image["id"], prediction["model"])
            assert all(0 <= item["confidence"] <= 1 for item in prediction["tags"])

    tagging_root = ROOT / "tagging"
    with (tagging_root / "gemma_reference.csv").open(newline="") as handle:
        gemma_rows = list(csv.DictReader(handle))
    assert len(gemma_rows) == 1899
    assert {row["image_id"] for row in gemma_rows} == image_ids
    assert all(row["model"] != "synthetic-test-only" for row in gemma_rows)
    for model_key in ("dinov3", "bioclip", "efficientnet_b0"):
        with (tagging_root / "predictions" / f"{model_key}.csv").open(newline="") as handle:
            prediction_rows = list(csv.DictReader(handle))
        assert len(prediction_rows) == 1899
        assert {row["image_id"] for row in prediction_rows} == image_ids
    with (tagging_root / "per_image_comparison.csv").open(newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    assert len(comparison_rows) == 1899 * 3
    with (tagging_root / "metrics" / "fold_audit.csv").open(newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
    assert len(fold_rows) == 15
    assert all(row["individual_leakage"].lower() == "false" for row in fold_rows)

    with (ROOT / "data" / "image_manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 1899
    assert {row["id"] for row in manifest} == image_ids

    index = (ROOT / "index.html").read_text()
    for path in ("assets/styles.css", "assets/app.js", "data/site-data.js"):
        assert path in index
        assert (ROOT / path).exists()
    app = (ROOT / "assets" / "app.js").read_text()
    for view in ("species", "organs", "individuals", "images", "image", "models"):
        assert f'view === "{view}"' in app
    for source in data["model_evaluation"].get("confusions", {}).values():
        assert (ROOT / source).is_file(), source

    validation = json.loads((ROOT / "data" / "validation.json").read_text())
    assert validation["valid"] is True
    audit_path = ROOT / "data" / "image_url_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        assert audit["urls_checked"] == 1899
        assert audit["reachable_images"] == 1899
        assert audit["failures"] == 0
    print(json.dumps({
        "valid": True,
        "images": len(images),
        "species": len(data["species"]),
        "individuals": len(data["individuals"]),
        "organ_categories": data["stats"]["organ_categories"],
        "view_types": data["stats"]["view_types"],
        "exact_primary_labels": data["stats"]["exact_primary_labels"],
        "prediction_images": data["stats"]["prediction_images"],
        "manual_review_images": data["stats"]["manual_review_images"],
        "tagging_models": len(tagging["summary"]),
        "gemma_reference_images": tagging["gemma_images"],
        "reachable_thumbnail_urls": audit["reachable_images"] if audit_path.exists() else None,
    }, indent=2))


if __name__ == "__main__":
    main()
