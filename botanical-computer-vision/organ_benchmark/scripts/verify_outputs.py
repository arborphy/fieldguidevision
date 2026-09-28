#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas==2.2.3"]
# ///
"""Independent structural verification for the generated benchmark bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        "README.md",
        "MORNING_REPORT.md",
        "index.html",
        "split_manifests/master_manifest.csv",
        "split_manifests/individual_disjoint.csv",
        "split_manifests/species_disjoint.csv",
        "split_manifests/leakage_assertions.json",
        "metrics/benchmark_summary.csv",
        "metrics/benchmark_summary.json",
        "metrics/per_class_metrics.csv",
        "metrics/cross_species_retrieval.csv",
        "metrics/cross_species_neighbors.csv",
        "metrics/exact_view_support.csv",
        "galleries/nearest_neighbors.html",
        "galleries/errors.html",
        "ambiguity_audit/multi_label_audit.csv",
        "ambiguity_audit/gallery.html",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    assert not missing, f"Missing outputs: {missing}"

    manifest = pd.read_csv(ROOT / "split_manifests/master_manifest.csv")
    assert len(manifest) == 1641
    assert manifest.image_id.nunique() == 1641
    for split_column, unit in (
        ("individual_split", "individual_id"),
        ("species_split", "species"),
    ):
        groups = {
            split: set(manifest.loc[manifest[split_column] == split, unit])
            for split in ("train", "validation", "test")
        }
        assert not groups["train"] & groups["validation"]
        assert not groups["train"] & groups["test"]
        assert not groups["validation"] & groups["test"]

    leakage = json.loads((ROOT / "split_manifests/leakage_assertions.json").read_text())
    assert leakage["individual_disjoint"]["passed"]
    assert leakage["species_disjoint"]["passed"]

    summary = pd.read_csv(ROOT / "metrics/benchmark_summary.csv")
    assert len(summary) == 12
    assert set(summary.task) == {"organ_tag", "organ_category"}
    assert set(summary.protocol) == {"individual_disjoint", "species_disjoint"}
    assert set(summary.model_key) == {"bioclip", "dinov3", "efficientnet_b0"}
    for column in ("top1_accuracy", "top3_recall", "macro_f1", "weighted_f1"):
        assert summary[column].between(0, 1).all()

    retrieval = pd.read_csv(ROOT / "metrics/cross_species_retrieval.csv")
    assert len(retrieval) == 3
    assert (retrieval.queries == 1641).all()
    neighbors = pd.read_csv(ROOT / "metrics/cross_species_neighbors.csv")
    assert len(neighbors) == 3 * 1641 * 10
    assert not (neighbors.query_species == neighbors.neighbor_species).any()
    assert not (neighbors.query_individual_id == neighbors.neighbor_individual_id).any()

    audit = pd.read_csv(ROOT / "ambiguity_audit/multi_label_audit.csv", keep_default_na=False)
    assert len(audit) == 180
    human_columns = [
        "visible_leaf", "visible_twig", "visible_bark", "visible_flower", "visible_fruit",
        "visible_cone", "visible_seed", "visible_whole_plant", "acceptable_labels", "ambiguous", "notes",
    ]
    assert all((audit[column] == "").all() for column in human_columns)

    html_text = (ROOT / "index.html").read_text(encoding="utf-8")
    for section in (
        "What we tested", "Headline results", "Individual-disjoint", "Species-disjoint",
        "Cross-species retrieval", "Per-organ performance", "Confusion matrices",
        "Nearest-neighbor examples", "Error analysis", "Multi-label ambiguity",
        "Exact-view feasibility", "Methods / reproducibility",
    ):
        assert section in html_text, f"Missing report section: {section}"
    assert "BioImages canonical-label" in html_text
    print("All benchmark output checks passed.")


if __name__ == "__main__":
    main()
