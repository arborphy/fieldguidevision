#!/usr/bin/env python3
"""Fail-fast integrity checks for the static Human Loop site."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_js(path: Path, prefix: str) -> dict:
    text = path.read_text()
    assert text.startswith(prefix) and text.rstrip().endswith(";"), path
    return json.loads(text[len(prefix):].rstrip().removesuffix(";"))


def main() -> None:
    site = load_js(ROOT / "data" / "site-data.js", "window.BIOIMAGES_DATA=")
    loop = load_js(ROOT / "data" / "human-loop-data.js", "window.BIOIMAGES_HUMAN_LOOP=")
    image_ids = {image["id"] for image in site["images"]}
    assert len(site["images"]) == 1899
    assert loop["generated_from_images"] == 1899
    assert len(loop["modes"]) == 12
    assert loop["vlm_audited_images"] == 169
    assert len(loop["human_set"]) == 100
    assert len({row["image_id"] for row in loop["human_set"]}) == 100
    assert len({row["species"] for row in loop["human_set"]}) == 85
    assert len({row["individual_id"] for row in loop["human_set"]}) == 100
    assert all(row["selection_reason"] and row["image_id"] in image_ids for row in loop["human_set"])
    for mode in loop["modes"]:
        assert mode["image_count"] > 0
        assert 8 <= len(mode["representative_image_ids"]) <= 12
        assert set(mode["representative_image_ids"]) <= image_ids
        assert len(mode["member_image_ids"]) == mode["image_count"]
        assert set(mode["member_image_ids"]) <= image_ids
        assert set(mode["member_models"]) == set(mode["member_image_ids"])
        assert set(mode["model_counts"]) == {"DINOv3", "BioCLIP 2.5", "EfficientNet-B0"}
        assert mode["vlm_reviewed"] > 0
    audit = [json.loads(line) for line in (ROOT / "analysis" / "vlm_error_audit.jsonl").read_text().splitlines() if line]
    assert len(audit) == 169 and len({row["image_id"] for row in audit}) == 169
    assert all(row["status"] == "ok" and row["source"] == "existing_gemma_image_tags" for row in audit)
    with (ROOT / "analysis" / "error_mode_members.csv").open(newline="") as handle:
        members = list(csv.DictReader(handle))
    assert len(members) > 1000
    html = (ROOT / "index.html").read_text()
    app = (ROOT / "assets" / "app.js").read_text()
    assert "data/human-loop-data.js" in html
    assert "Human review" in html and "Full gallery" in html
    assert HUMAN_STORAGE_KEY in app
    print(json.dumps({
        "status": "ok", "images": len(image_ids), "modes": len(loop["modes"]),
        "vlm_audit": len(audit), "human_set": len(loop["human_set"]),
        "species": len({row["species"] for row in loop["human_set"]}),
        "individuals": len({row["individual_id"] for row in loop["human_set"]}),
        "mode_memberships": len(members),
    }, indent=2))


HUMAN_STORAGE_KEY = "bioimages-human-gold-v2"


if __name__ == "__main__":
    main()
