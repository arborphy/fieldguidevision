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
        "reachable_thumbnail_urls": audit["reachable_images"] if audit_path.exists() else None,
    }, indent=2))


if __name__ == "__main__":
    main()
