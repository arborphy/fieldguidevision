#!/usr/bin/env python3
"""Deterministic integrity checks for the complete analysis page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ASSETS = HERE / "assets"
REPRESENTATIONS = {"bioclip", "dinov3", "efficientnet_b0"}
SCALES = {"distant", "mid-range", "close-up"}


def main() -> None:
    groups = pd.read_csv(DATA / "image_visual_groups.csv")
    separation = pd.read_csv(DATA / "separation_metrics.csv")
    accuracy = pd.read_csv(DATA / "accuracy_by_visual_group.csv")
    clusters = pd.read_csv(DATA / "cluster_visual_composition.csv")
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    html = (HERE / "index.html").read_text(encoding="utf-8")
    payload = (DATA / "analysis-payload.js").read_text(encoding="utf-8")

    assert len(groups) == 1899
    assert groups.image_id.nunique() == 1899
    assert set(groups.shot_scale) == SCALES | {"uncertain"}
    assert groups.shot_scale.value_counts().to_dict() == summary["shot_scale_counts"]
    assert set(groups.leaf_state) == {"leaf-on", "leaf-off", "mixed", "not-visible"}
    assert set(groups.reproductive_visibility) == {"visible", "not-visible"}
    assert set(groups.background_context) == {"environmental", "isolated", "unspecified"}

    baseline = separation[separation.axis.isin(["all", "shot_scale"])]
    assert len(baseline) == len(REPRESENTATIONS) * 4
    assert set(baseline.representation) == REPRESENTATIONS
    assert baseline.species_purity.between(0, 1).all()
    assert baseline.species_nmi.between(0, 1).all()

    scale_accuracy = accuracy[accuracy.axis == "shot_scale"]
    assert len(scale_accuracy) == len(REPRESENTATIONS) * len(SCALES)
    assert set(scale_accuracy.representation) == REPRESENTATIONS
    assert set(scale_accuracy.value) == SCALES
    assert (scale_accuracy.correct <= scale_accuracy.total).all()
    assert set(scale_accuracy.groupby("representation").total.sum()) == {208}

    assert len(clusters) == 60
    assert set(clusters.representation) == REPRESENTATIONS
    assert (clusters.groupby("representation").cluster_id.nunique() == 20).all()

    for marker in [
        "Scale-only cluster explorer",
        "Does species separation improve after controlling scale?",
        "Every cluster, summarized visually",
        "Classification accuracy by shot scale",
        "Second layer · explain residual mixing",
        "Local projection and rotation",
        "Seed-guided neighborhood explorer",
    ]:
        assert marker in html
    assert "window.SCALE_ANALYSIS" in payload
    assert len(payload) > 1_000_000

    for name in [
        "representation_by_scale.png",
        "species_separation_by_scale.png",
        "accuracy_by_scale.png",
        "secondary_explanatory_variables.png",
        "bioclip_cluster_composition.png",
        "dinov3_cluster_composition.png",
        "efficientnet_b0_cluster_composition.png",
    ]:
        path = ASSETS / name
        assert path.exists() and path.stat().st_size > 10_000, name

    print(
        json.dumps(
            {
                "images": len(groups),
                "clusters": len(clusters),
                "scale_accuracy_rows": len(scale_accuracy),
                "original_embeddings_available": summary[
                    "original_embeddings_available"
                ],
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
