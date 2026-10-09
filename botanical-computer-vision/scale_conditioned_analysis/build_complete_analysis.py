#!/usr/bin/env python3
"""Build the complete scale-conditioned representation analysis.

The page follows the research sequence agreed for this project:

1. explain cluster mixing with photographic scale;
2. inspect residual mixing with secondary visual variables;
3. inspect selected mixed clusters with local projections;
4. expand representative seeds through original-space neighbours.

The first two stages can be rebuilt from committed summaries. Local PCA and
Top-100 neighbours require the ignored frozen ``outputs/*/embeddings.npz``
files. The page never substitutes two-dimensional PCA distance for the
original high-dimensional cosine distance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from build_scale_conditioned_analysis import (
    REP_LABELS,
    REPRESENTATIONS,
    SCALE_COLORS,
    SCALE_ORDER,
    build_codebook,
    build_scale_review_sample,
    display_image_url,
    load_and_label,
    make_explanatory_plots,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
ASSETS = HERE / "assets"

MAIN_SCALES = ["distant", "mid-range", "close-up"]
SECONDARY_AXES = {
    "leaf_state": ["leaf-on", "leaf-off"],
    "reproductive_visibility": ["visible", "not-visible"],
    "background_context": ["environmental", "isolated"],
}
AXIS_LABELS = {
    "leaf_state": "Leaf state",
    "reproductive_visibility": "Reproductive structure",
    "background_context": "Background context",
}
GROUP_LABELS = {
    "leaf-on": "leaf-on",
    "leaf-off": "leaf-off",
    "visible": "visible",
    "not-visible": "not visible",
    "environmental": "environmental context",
    "isolated": "isolated subject",
}
REP_COLORS = {
    "bioclip": "#315f4a",
    "dinov3": "#b36a36",
    "efficientnet_b0": "#6c5a8f",
}
CLUSTER_PALETTE = [
    "#315f4a", "#b36a36", "#6c5a8f", "#3c7187", "#9b5166",
    "#7b7833", "#546b9a", "#8d6142", "#487b70", "#805d79",
    "#56723b", "#b17462", "#4e6681", "#856e35", "#467386",
    "#9a5660", "#59734f", "#6f5f95", "#8a6a4a", "#4e7b65",
]

EMBEDDING_PATHS = {
    "bioclip": ROOT / "outputs/bioclip25_strict_embeddings/embeddings.npz",
    "dinov3": ROOT / "outputs/dinov3_strict_embeddings/embeddings.npz",
    "efficientnet_b0": ROOT / "outputs/efficientnet_b0_strict_embeddings/embeddings.npz",
}


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def weighted_purity(frame: pd.DataFrame, label: str = "species") -> float:
    if frame.empty:
        return float("nan")
    majority = frame.groupby("cluster_id")[label].agg(lambda x: x.value_counts().iloc[0]).sum()
    return float(majority / len(frame))


def load_assignments() -> pd.DataFrame:
    primary = pd.read_csv(ROOT / "representation_analysis/data/cluster_assignments.csv")
    primary = primary[primary["k"] == 20].copy()
    efficient = pd.read_csv(
        ROOT / "representation_analysis/data/efficientnet_k20_cluster_assignments.csv"
    )
    result = pd.concat([primary, efficient], ignore_index=True)
    expected = 1641 * len(REPRESENTATIONS)
    if len(result) != expected:
        raise ValueError(f"Expected {expected} k=20 assignments, found {len(result)}")
    return result


def load_projection_points(images: pd.DataFrame, assignments: pd.DataFrame) -> dict:
    metadata = images.set_index("image_id")
    prediction_lookup: dict[str, dict[str, dict[str, object]]] = {
        rep: {} for rep in REPRESENTATIONS
    }
    strict = pd.read_csv(ROOT / "error_analysis/data/unified_predictions_image.csv")
    for row in strict.itertuples():
        prediction_lookup["bioclip"][row.image_id] = {
            "prediction": row.bioclip_prediction,
            "correct": as_bool(row.bioclip_correct),
        }
        prediction_lookup["dinov3"][row.image_id] = {
            "prediction": row.dinov3_prediction,
            "correct": as_bool(row.dinov3_correct),
        }
    efficient = pd.read_csv(
        ROOT / "representation_analysis/data/efficientnet_test_predictions.csv"
    )
    for row in efficient.itertuples():
        prediction_lookup["efficientnet_b0"][row.image_id] = {
            "prediction": row.predicted_species,
            "correct": as_bool(row.correct),
        }

    result: dict[str, dict[str, object]] = {}
    for rep in REPRESENTATIONS:
        with np.load(
            ROOT / f"representation_analysis/data/{rep}_final_projection_coordinates.npz"
        ) as payload:
            ids = payload["image_ids"].astype(str)
            coords = payload["pca"].astype(float)
        assignment = assignments[assignments.representation == rep].set_index("image_id")
        points = []
        for image_id, xy in zip(ids, coords):
            row = metadata.loc[image_id]
            pred = prediction_lookup[rep].get(image_id, {})
            points.append(
                [
                    image_id,
                    round(float(xy[0]), 5),
                    round(float(xy[1]), 5),
                    int(assignment.loc[image_id, "cluster_id"]),
                    row.species,
                    row.organ_category,
                    row.shot_scale,
                    row.leaf_state,
                    row.reproductive_visibility,
                    row.background_context,
                    display_image_url(image_id, row.thumbnail_url),
                    row.source_url,
                    pred.get("prediction", ""),
                    pred.get("correct", None),
                ]
            )
        result[rep] = {
            "points": points,
            "x_extent": [float(coords[:, 0].min()), float(coords[:, 0].max())],
            "y_extent": [float(coords[:, 1].min()), float(coords[:, 1].max())],
        }
    return result


def build_separation_metrics(images: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    joined = assignments.merge(
        images[
            [
                "image_id",
                "species",
                "organ_category",
                "shot_scale",
                "leaf_state",
                "reproductive_visibility",
                "background_context",
            ]
        ],
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for rep, rep_frame in joined.groupby("representation"):
        groups = [("all", "all", rep_frame)]
        groups.extend(
            ("shot_scale", scale, rep_frame[rep_frame.shot_scale == scale])
            for scale in MAIN_SCALES
        )
        for axis, values in SECONDARY_AXES.items():
            for scale in MAIN_SCALES:
                scale_frame = rep_frame[rep_frame.shot_scale == scale]
                for value in values:
                    groups.append(
                        (
                            f"{axis}|{scale}",
                            value,
                            scale_frame[scale_frame[axis] == value],
                        )
                    )
        for axis, value, group in groups:
            if len(group) < 2:
                continue
            rows.append(
                {
                    "representation": rep,
                    "axis": axis,
                    "value": value,
                    "images": len(group),
                    "species": group.species.nunique(),
                    "species_purity": weighted_purity(group, "species"),
                    "species_nmi": normalized_mutual_info_score(
                        group.species, group.cluster_id
                    ),
                    "organ_purity": weighted_purity(group, "organ_category"),
                    "organ_nmi": normalized_mutual_info_score(
                        group.organ_category, group.cluster_id
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "separation_metrics.csv", index=False)
    return result


def load_test_predictions(images: pd.DataFrame) -> pd.DataFrame:
    strict = pd.read_csv(ROOT / "error_analysis/data/unified_predictions_image.csv")
    rows = []
    for item in strict.itertuples():
        for rep in ["bioclip", "dinov3"]:
            rows.append(
                {
                    "image_id": item.image_id,
                    "representation": rep,
                    "truth": item.ground_truth_species,
                    "prediction": getattr(item, f"{rep}_prediction"),
                    "confidence": float(getattr(item, f"{rep}_score")),
                    "correct": as_bool(getattr(item, f"{rep}_correct")),
                }
            )
    efficient = pd.read_csv(
        ROOT / "representation_analysis/data/efficientnet_test_predictions.csv"
    )
    for item in efficient.itertuples():
        rows.append(
            {
                "image_id": item.image_id,
                "representation": "efficientnet_b0",
                "truth": item.ground_truth_species,
                "prediction": item.predicted_species,
                "confidence": float(item.confidence),
                "correct": as_bool(item.correct),
            }
        )
    predictions = pd.DataFrame(rows)
    return predictions.merge(
        images[
            [
                "image_id",
                "species",
                "organ_category",
                "shot_scale",
                "leaf_state",
                "reproductive_visibility",
                "background_context",
                "thumbnail_url",
                "source_url",
            ]
        ],
        on="image_id",
        how="left",
        validate="many_to_one",
    )


def build_accuracy(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    axes = {"shot_scale": MAIN_SCALES, **SECONDARY_AXES}
    for axis, values in axes.items():
        for value in values:
            for rep in REPRESENTATIONS:
                group = predictions[
                    (predictions.representation == rep) & (predictions[axis] == value)
                ]
                if group.empty:
                    continue
                rows.append(
                    {
                        "axis": axis,
                        "value": value,
                        "representation": rep,
                        "correct": int(group.correct.sum()),
                        "total": len(group),
                        "accuracy": float(group.correct.mean()),
                    }
                )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(DATA / "accuracy_by_visual_group.csv", index=False)

    examples: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for rep in REPRESENTATIONS:
        examples[rep] = {}
        for scale in MAIN_SCALES:
            group = predictions[
                (predictions.representation == rep) & (predictions.shot_scale == scale)
            ]
            examples[rep][scale] = {}
            for correct, label in [(True, "correct"), (False, "incorrect")]:
                chosen = (
                    group[group.correct == correct]
                    .sort_values(["confidence", "image_id"], ascending=[False, True])
                    .head(5)
                )
                examples[rep][scale][label] = [
                    {
                        "id": row.image_id,
                        "truth": row.truth,
                        "prediction": row.prediction,
                        "organ": row.organ_category,
                        "scale": row.shot_scale,
                        "image": display_image_url(row.image_id, row.thumbnail_url),
                        "source": row.source_url,
                    }
                    for row in chosen.itertuples()
                ]
    return metrics, examples


def compact_counts(values: pd.Series, limit: int | None = None) -> list[dict]:
    counts = values.value_counts()
    if limit:
        counts = counts.head(limit)
    return [{"label": str(label), "count": int(count)} for label, count in counts.items()]


def build_cluster_summaries(images: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined = assignments.merge(
        images[
            [
                "image_id",
                "species",
                "organ_category",
                "shot_scale",
                "leaf_state",
                "reproductive_visibility",
                "background_context",
                "thumbnail_url",
                "source_url",
            ]
        ],
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    rows = []
    payload: dict[str, list[dict]] = {rep: [] for rep in REPRESENTATIONS}
    for (rep, cluster_id), group in joined.groupby(["representation", "cluster_id"]):
        species_purity = weighted_purity(group, "species")
        organ_purity = weighted_purity(group, "organ_category")
        scales = group.shot_scale.value_counts(normalize=True)
        scale_entropy = float(
            -(scales * np.log(scales)).sum() / math.log(max(2, len(SCALE_ORDER)))
        )
        score = (1 - species_purity) * scale_entropy * math.log1p(len(group))
        rows.append(
            {
                "representation": rep,
                "cluster_id": int(cluster_id),
                "images": len(group),
                "species_count": group.species.nunique(),
                "species_purity": species_purity,
                "organ_purity": organ_purity,
                "scale_entropy": scale_entropy,
                "review_priority": score,
                "species_distribution": json.dumps(
                    {str(k): int(v) for k, v in group.species.value_counts().items()},
                    ensure_ascii=False,
                ),
                "organ_distribution": json.dumps(
                    {str(k): int(v) for k, v in group.organ_category.value_counts().items()},
                    ensure_ascii=False,
                ),
                "scale_distribution": json.dumps(
                    {str(k): int(v) for k, v in group.shot_scale.value_counts().items()},
                    ensure_ascii=False,
                ),
            }
        )
        gallery = []
        for scale in MAIN_SCALES:
            scale_group = group[group.shot_scale == scale].sort_values("distance_to_centroid")
            gallery.append(
                {
                    "scale": scale,
                    "images": [
                        {
                            "id": item.image_id,
                            "species": item.species,
                            "organ": item.organ_category,
                            "image": display_image_url(item.image_id, item.thumbnail_url),
                            "source": item.source_url,
                        }
                        for item in scale_group.head(12).itertuples()
                    ],
                }
            )
        original = [
            {
                "id": item.image_id,
                "species": item.species,
                "organ": item.organ_category,
                "scale": item.shot_scale,
                "image": display_image_url(item.image_id, item.thumbnail_url),
                "source": item.source_url,
            }
            for item in group.sort_values("distance_to_centroid").head(18).itertuples()
        ]
        payload[rep].append(
            {
                "cluster": int(cluster_id),
                "n": len(group),
                "species_count": int(group.species.nunique()),
                "species_purity": round(species_purity, 4),
                "organ_purity": round(organ_purity, 4),
                "scale_entropy": round(scale_entropy, 4),
                "priority": round(score, 4),
                "species": compact_counts(group.species, 8),
                "organs": compact_counts(group.organ_category, 8),
                "scales": compact_counts(group.shot_scale),
                "leaf": compact_counts(group.leaf_state),
                "reproductive": compact_counts(group.reproductive_visibility),
                "background": compact_counts(group.background_context),
                "original": original,
                "by_scale": gallery,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["representation", "cluster_id"])
    summary.to_csv(DATA / "cluster_visual_composition.csv", index=False)
    for rep in REPRESENTATIONS:
        payload[rep].sort(key=lambda x: x["priority"], reverse=True)
    return summary, payload


def load_embeddings() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for rep, path in EMBEDDING_PATHS.items():
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as payload:
            ids = payload["image_ids"].astype(str)
            matrix = payload["embeddings"].astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-3):
            matrix /= np.clip(norms[:, None], 1e-9, None)
        result[rep] = (ids, matrix)
    return result


def build_distance_metrics(
    images: pd.DataFrame, embeddings: dict[str, tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    columns = [
        "representation", "pair_type", "pairs", "mean_cosine_similarity",
        "median_cosine_similarity", "ci_low", "ci_high",
    ]
    rows = []
    metadata = images.set_index("image_id")
    rng = np.random.default_rng(42)
    for rep, (ids, matrix) in embeddings.items():
        species = np.asarray([metadata.loc[i, "species"] for i in ids])
        scale = np.asarray([metadata.loc[i, "shot_scale"] for i in ids])
        left, right = np.triu_indices(len(ids), k=1)
        if len(left) > 300_000:
            sample = rng.choice(len(left), 300_000, replace=False)
            left, right = left[sample], right[sample]
        similarity = np.sum(matrix[left] * matrix[right], axis=1)
        same_species = species[left] == species[right]
        same_scale = scale[left] == scale[right]
        valid_scale = np.isin(scale[left], MAIN_SCALES) & np.isin(scale[right], MAIN_SCALES)
        categories = [
            ("same species · same scale", same_species & same_scale),
            ("same species · different scale", same_species & ~same_scale),
            ("different species · same scale", ~same_species & same_scale),
            ("different species · different scale", ~same_species & ~same_scale),
        ]
        for name, mask in categories:
            values = similarity[mask & valid_scale]
            if values.size == 0:
                continue
            boot = []
            for _ in range(500):
                boot.append(float(rng.choice(values, size=len(values), replace=True).mean()))
            rows.append(
                {
                    "representation": rep,
                    "pair_type": name,
                    "pairs": int(values.size),
                    "mean_cosine_similarity": float(values.mean()),
                    "median_cosine_similarity": float(np.median(values)),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(DATA / "scale_pairwise_similarity.csv", index=False)
    return result


def build_deep_payload(
    images: pd.DataFrame,
    assignments: pd.DataFrame,
    cluster_payload: dict,
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, dict]:
    metadata = images.set_index("image_id")
    assignment_lookup = {
        rep: assignments[assignments.representation == rep].set_index("image_id")
        for rep in REPRESENTATIONS
    }
    local_payload: dict[str, list[dict]] = {rep: [] for rep in REPRESENTATIONS}
    seed_payload: dict[str, list[dict]] = {rep: [] for rep in REPRESENTATIONS}
    for rep, (ids, matrix) in embeddings.items():
        index = {image_id: i for i, image_id in enumerate(ids)}
        selected_clusters = [item["cluster"] for item in cluster_payload[rep][:5]]
        for cluster_id in selected_clusters:
            cluster_ids = [
                image_id
                for image_id in ids
                if int(assignment_lookup[rep].loc[image_id, "cluster_id"]) == cluster_id
            ]
            positions = np.asarray([index[i] for i in cluster_ids])
            local_matrix = matrix[positions]
            n_components = min(6, len(cluster_ids) - 1, local_matrix.shape[1])
            if n_components < 4:
                continue
            pca = PCA(n_components=n_components, random_state=42)
            coords = pca.fit_transform(local_matrix)
            points = []
            for image_id, values in zip(cluster_ids, coords):
                row = metadata.loc[image_id]
                points.append(
                    {
                        "id": image_id,
                        "pc": [round(float(x), 5) for x in values],
                        "species": row.species,
                        "organ": row.organ_category,
                        "scale": row.shot_scale,
                        "leaf": row.leaf_state,
                        "reproductive": row.reproductive_visibility,
                        "background": row.background_context,
                        "image": display_image_url(image_id, row.thumbnail_url),
                        "source": row.source_url,
                    }
                )
            local_payload[rep].append(
                {
                    "cluster": cluster_id,
                    "n": len(points),
                    "explained_variance": [
                        round(float(x), 5) for x in pca.explained_variance_ratio_
                    ],
                    "points": points,
                }
            )

            centroid_order = sorted(
                cluster_ids,
                key=lambda image_id: float(
                    assignment_lookup[rep].loc[image_id, "distance_to_centroid"]
                ),
            )
            for seed_id in centroid_order[:2]:
                seed_position = index[seed_id]
                similarities = matrix @ matrix[seed_position]
                seed_individual = metadata.loc[seed_id, "individual_id"]
                order = np.argsort(-similarities)
                neighbors = []
                for position in order:
                    neighbor_id = ids[position]
                    if neighbor_id == seed_id:
                        continue
                    if metadata.loc[neighbor_id, "individual_id"] == seed_individual:
                        continue
                    row = metadata.loc[neighbor_id]
                    neighbors.append(
                        {
                            "rank": len(neighbors) + 1,
                            "id": neighbor_id,
                            "similarity": round(float(similarities[position]), 5),
                            "species": row.species,
                            "organ": row.organ_category,
                            "scale": row.shot_scale,
                            "background": row.background_context,
                            "image": display_image_url(neighbor_id, row.thumbnail_url),
                            "source": row.source_url,
                        }
                    )
                    if len(neighbors) == 100:
                        break
                seed = metadata.loc[seed_id]
                seed_payload[rep].append(
                    {
                        "id": seed_id,
                        "cluster": cluster_id,
                        "species": seed.species,
                        "organ": seed.organ_category,
                        "scale": seed.shot_scale,
                        "background": seed.background_context,
                        "image": display_image_url(seed_id, seed.thumbnail_url),
                        "source": seed.source_url,
                        "neighbors": neighbors,
                    }
                )
    return local_payload, seed_payload


def chart_separation(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=170)
    labels = ["All", "Distant", "Mid-range", "Close-up"]
    keys = [("all", "all")] + [("shot_scale", x) for x in MAIN_SCALES]
    width = 0.24
    x = np.arange(len(labels))
    for offset, rep in enumerate(REPRESENTATIONS):
        values = []
        nmis = []
        for axis, value in keys:
            row = metrics[
                (metrics.representation == rep)
                & (metrics.axis == axis)
                & (metrics.value == value)
            ].iloc[0]
            values.append(row.species_purity)
            nmis.append(row.species_nmi)
        position = x + (offset - 1) * width
        axes[0].bar(position, values, width, label=REP_LABELS[rep], color=REP_COLORS[rep])
        axes[1].bar(position, nmis, width, label=REP_LABELS[rep], color=REP_COLORS[rep])
    for ax, title, ylabel in [
        (axes[0], "Species purity", "Weighted cluster purity"),
        (axes[1], "Species NMI", "Normalized mutual information"),
    ]:
        ax.set_title(title, loc="left")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e4e8e5", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("Species separation before and after conditioning on shot scale", x=.06, ha="left")
    fig.tight_layout()
    fig.savefig(ASSETS / "species_separation_by_scale.png", bbox_inches="tight")
    plt.close(fig)


def chart_accuracy(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), dpi=170, sharey=True)
    for ax, rep in zip(axes, REPRESENTATIONS):
        frame = metrics[
            (metrics.axis == "shot_scale") & (metrics.representation == rep)
        ].set_index("value")
        values = [frame.loc[scale, "accuracy"] for scale in MAIN_SCALES]
        bars = ax.bar(
            MAIN_SCALES,
            values,
            color=[SCALE_COLORS[scale] for scale in MAIN_SCALES],
            width=.68,
        )
        for bar, scale in zip(bars, MAIN_SCALES):
            row = frame.loc[scale]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(.98, bar.get_height() + .035),
                f"{int(row.correct)}/{int(row.total)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(REP_LABELS[rep], loc="left")
        ax.set_ylim(0, 1.06)
        ax.set_ylabel("Top-1 accuracy" if rep == "bioclip" else "")
        ax.grid(axis="y", color="#e4e8e5", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Species-classification accuracy by shot scale", x=.04, ha="left")
    fig.tight_layout()
    fig.savefig(ASSETS / "accuracy_by_scale.png", bbox_inches="tight")
    plt.close(fig)


def chart_secondary(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), dpi=170, sharey=True)
    for column, rep in enumerate(REPRESENTATIONS):
        for row_index, (axis, values) in enumerate(SECONDARY_AXES.items()):
            ax = axes[row_index, column]
            means = []
            labels = []
            totals = []
            for value in values:
                subsets = metrics[
                    (metrics.representation == rep)
                    & (metrics.axis.str.startswith(axis + "|"))
                    & (metrics.value == value)
                ]
                means.append(float(np.average(subsets.species_purity, weights=subsets.images)))
                totals.append(int(subsets.images.sum()))
                labels.append(GROUP_LABELS[value])
            bars = ax.bar(labels, means, color=["#315f4a", "#b36a36"], width=.62)
            for bar, total in zip(bars, totals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(.98, bar.get_height() + .025),
                    f"n={total}",
                    ha="center",
                    fontsize=8,
                )
            if row_index == 0:
                ax.set_title(REP_LABELS[rep], loc="left")
            if column == 0:
                ax.set_ylabel(f"{AXIS_LABELS[axis]}\nSpecies purity")
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", labelrotation=12)
            ax.grid(axis="y", color="#e4e8e5", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Residual species separation within shot scale, summarized by second-layer variables",
        x=.04,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "secondary_explanatory_variables.png", bbox_inches="tight")
    plt.close(fig)


def chart_cluster_compositions(summary: pd.DataFrame) -> None:
    organ_colors = {
        "leaf": "#315f4a", "bark": "#8a6a4a", "twig": "#6c5a8f",
        "flower": "#9b5166", "inflorescence": "#b36a36", "fruit": "#c58b3c",
        "whole tree (or vine)": "#3c7187", "other": "#b8c0bb",
    }
    scale_order = MAIN_SCALES + ["uncertain"]
    scale_colors = [SCALE_COLORS[name] for name in scale_order]
    rank_colors = ["#315f4a", "#6c856f", "#9eae9f", "#d4dad6"]
    for rep in REPRESENTATIONS:
        frame = summary[summary.representation == rep].sort_values("cluster_id")
        clusters = frame.cluster_id.astype(int).tolist()
        y = np.arange(len(frame))
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 9.5), dpi=170, sharey=True)

        left = np.zeros(len(frame))
        for name, color in zip(scale_order, scale_colors):
            values = np.array([
                json.loads(raw).get(name, 0) / n
                for raw, n in zip(frame.scale_distribution, frame.images)
            ])
            axes[0].barh(y, values, left=left, color=color, label=name)
            left += values
        axes[0].set_title("Shot-scale distribution", loc="left")
        axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")

        left = np.zeros(len(frame))
        for organ, color in organ_colors.items():
            values = []
            for raw, n in zip(frame.organ_distribution, frame.images):
                counts = json.loads(raw)
                if organ == "other":
                    value = sum(v for k, v in counts.items() if k not in organ_colors or k == "other")
                else:
                    value = counts.get(organ, 0)
                values.append(value / n)
            values = np.asarray(values)
            axes[1].barh(y, values, left=left, color=color, label=organ)
            left += values
        axes[1].set_title("Organ distribution", loc="left")
        axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")

        rank_values = [[], [], [], []]
        for raw, n in zip(frame.species_distribution, frame.images):
            counts = sorted(json.loads(raw).values(), reverse=True)
            rank_values[0].append((counts[0] if len(counts) > 0 else 0) / n)
            rank_values[1].append((counts[1] if len(counts) > 1 else 0) / n)
            rank_values[2].append((counts[2] if len(counts) > 2 else 0) / n)
            rank_values[3].append(max(0, n - sum(counts[:3])) / n)
        left = np.zeros(len(frame))
        for values, color, label in zip(
            rank_values, rank_colors, ["top species", "2nd", "3rd", "all others"]
        ):
            values = np.asarray(values)
            axes[2].barh(y, values, left=left, color=color, label=label)
            left += values
        axes[2].set_title("Species distribution by rank", loc="left")
        axes[2].legend(frameon=False, fontsize=8, loc="lower right")

        axes[0].set_yticks(y, [f"cluster {cluster:02d}" for cluster in clusters])
        axes[0].invert_yaxis()
        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_xlabel("Share of cluster")
            ax.grid(axis="x", color="#e4e8e5", linewidth=.6)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{REP_LABELS[rep]} · composition of every k=20 cluster", x=.04, ha="left")
        fig.tight_layout()
        fig.savefig(ASSETS / f"{rep}_cluster_composition.png", bbox_inches="tight")
        plt.close(fig)


def chart_pairwise_distances(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), dpi=170, sharey=True)
    order = [
        "same species · same scale",
        "same species · different scale",
        "different species · same scale",
        "different species · different scale",
    ]
    labels = ["same sp.\nsame scale", "same sp.\ndiff. scale", "diff. sp.\nsame scale", "diff. sp.\ndiff. scale"]
    for ax, rep in zip(axes, REPRESENTATIONS):
        frame = metrics[metrics.representation == rep].set_index("pair_type").loc[order]
        x = np.arange(len(order))
        means = frame.mean_cosine_similarity.to_numpy()
        low = means - frame.ci_low.to_numpy()
        high = frame.ci_high.to_numpy() - means
        ax.errorbar(x, means, yerr=[low, high], fmt="o", color=REP_COLORS[rep], capsize=4)
        ax.set_xticks(x, labels)
        ax.set_title(REP_LABELS[rep], loc="left")
        ax.set_ylabel("Mean cosine similarity" if rep == "bioclip" else "")
        ax.grid(axis="y", color="#e4e8e5", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("How species identity and shot scale affect embedding similarity", x=.04, ha="left")
    fig.tight_layout()
    fig.savefig(ASSETS / "pairwise_similarity_by_species_and_scale.png", bbox_inches="tight")
    plt.close(fig)


def image_card(item: dict, extra: str = "") -> str:
    species = item.get("species", item.get("truth", "Unknown"))
    return (
        f'<a class="thumb" href="{escape(item["source"])}" target="_blank" rel="noreferrer">'
        f'<img loading="lazy" src="{escape(item["image"])}" alt="{escape(species)}">'
        f'<span><b>{escape(species)}</b><i>{escape(item.get("organ", ""))}</i>{extra}</span></a>'
    )


def build_accuracy_examples(examples: dict) -> str:
    sections = []
    for rep in REPRESENTATIONS:
        scales = []
        for scale in MAIN_SCALES:
            correct = "".join(image_card(x) for x in examples[rep][scale]["correct"])
            incorrect = "".join(
                image_card(
                    x,
                    f'<em>truth {escape(x["truth"])} · predicted {escape(x["prediction"])}</em>',
                )
                for x in examples[rep][scale]["incorrect"]
            )
            scales.append(
                f'<section class="example-group"><h4>{escape(scale)}</h4>'
                f'<div class="example-pair"><div><b>Correct</b><div class="thumb-grid compact">{correct}</div></div>'
                f'<div><b>Incorrect</b><div class="thumb-grid compact">{incorrect}</div></div></div></section>'
            )
        sections.append(
            f'<section class="example-panel" data-example-rep="{rep}"><h3>{REP_LABELS[rep]}</h3>{"".join(scales)}</section>'
        )
    return "".join(sections)


def build_html(
    payload: dict,
    metrics: pd.DataFrame,
    accuracy: pd.DataFrame,
    embeddings_ready: bool,
    asset_version: str,
) -> str:
    counts = payload["counts"]
    metrics_table = []
    for rep in REPRESENTATIONS:
        for label, axis, value in [("all", "all", "all")] + [
            (scale, "shot_scale", scale) for scale in MAIN_SCALES
        ]:
            row = metrics[
                (metrics.representation == rep)
                & (metrics.axis == axis)
                & (metrics.value == value)
            ].iloc[0]
            metrics_table.append(
                f'<tr><td>{REP_LABELS[rep]}</td><td>{escape(label)}</td><td>{int(row.images):,}</td>'
                f'<td>{pct(row.species_purity)}</td><td>{row.species_nmi:.3f}</td>'
                f'<td>{pct(row.organ_purity)}</td><td>{row.organ_nmi:.3f}</td></tr>'
            )
    embedding_note = (
        "Original frozen vectors are loaded; local PCA, pairwise cosine analysis and Top-100 neighbours are active."
        if embeddings_ready
        else "Original frozen vectors are not present in this checkout. The page shows committed global projections and the Top-10 legacy neighbours without pretending they are local high-dimensional results."
    )
    separation_findings = []
    accuracy_findings = []
    for rep in REPRESENTATIONS:
        overall = metrics[
            (metrics.representation == rep) & (metrics.axis == "all")
        ].iloc[0]
        mid = metrics[
            (metrics.representation == rep)
            & (metrics.axis == "shot_scale")
            & (metrics.value == "mid-range")
        ].iloc[0]
        distant_accuracy = accuracy[
            (accuracy.representation == rep)
            & (accuracy.axis == "shot_scale")
            & (accuracy.value == "distant")
        ].iloc[0]
        close_accuracy = accuracy[
            (accuracy.representation == rep)
            & (accuracy.axis == "shot_scale")
            & (accuracy.value == "close-up")
        ].iloc[0]
        separation_findings.append(
            f"{REP_LABELS[rep]}: overall purity {pct(overall.species_purity)}; "
            f"mid-range {pct(mid.species_purity)}"
        )
        accuracy_findings.append(
            f"{REP_LABELS[rep]}: distant {int(distant_accuracy.correct)}/{int(distant_accuracy.total)} "
            f"vs close-up {int(close_accuracy.correct)}/{int(close_accuracy.total)}"
        )
    pairwise_section = (
        '<figure class="wide-figure"><img src="assets/pairwise_similarity_by_species_and_scale.png" alt="Pairwise cosine similarity by species and scale"><figcaption>Points are mean original-space cosine similarity; whiskers are bootstrap 95% intervals.</figcaption></figure>'
        if embeddings_ready
        else '<div class="method-note"><strong>Pending vector regeneration.</strong> This comparison requires the original normalized embeddings.</div>'
    )
    deep_state = "ready" if embeddings_ready else "pending"
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scale-conditioned representation analysis · BioImages</title>
<style>
:root{{--ink:#17221d;--muted:#65716a;--line:#d8dfda;--paper:#fff;--soft:#f3f6f4;--green:#315f4a;--orange:#b36a36;--purple:#6c5a8f}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#fbfcfb;color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}a{{color:var(--green)}}.wrap{{max-width:1240px;margin:auto;padding:0 28px}}header{{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line)}}nav{{min-height:58px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}}nav .brand{{font-weight:750;margin-right:auto}}nav a{{font-size:12px;text-decoration:none;color:var(--ink)}}main{{padding-bottom:80px}}.hero{{padding:64px 0 38px}}.eyebrow{{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--green);font-weight:700}}h1,h2,h3,h4{{font-family:Georgia,"Times New Roman",serif;font-weight:400}}h1{{font-size:clamp(2.6rem,6vw,5.6rem);line-height:.98;margin:12px 0 18px;max-width:1050px}}h2{{font-size:clamp(1.8rem,3vw,2.7rem);margin:0 0 9px}}h3{{font-size:1.35rem}}h4{{font-size:1.05rem}}.lede,.section-note{{color:var(--muted);max-width:900px}}.section{{padding:42px 0;border-top:1px solid var(--line)}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:26px}}.stat{{padding:16px;border:1px solid var(--line);background:var(--paper)}}.stat strong,.stat span{{display:block}}.stat strong{{font:30px Georgia,serif}}.stat span{{color:var(--muted)}}.sequence{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:24px}}.sequence div{{padding:16px;background:var(--soft);border-top:3px solid var(--green)}}.sequence b,.sequence span{{display:block}}.sequence span{{font-size:12px;color:var(--muted)}}.method-note{{padding:16px 18px;background:#edf3ef;border-left:4px solid var(--green);margin:18px 0}}.rule-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.rule{{padding:16px;background:var(--paper);border:1px solid var(--line)}}.rule h3{{margin:0 0 6px}}.rule p{{margin:0;color:var(--muted);font-size:13px}}.wide-figure{{margin:22px 0}}.wide-figure img{{display:block;width:100%;background:white;border:1px solid var(--line)}}figcaption{{font-size:12px;color:var(--muted);margin-top:7px}}.plot-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.plot-grid figure{{margin:0}}.plot-grid img{{display:block;width:100%;border:1px solid var(--line);background:white}}.table-wrap{{overflow:auto;border:1px solid var(--line);background:white;margin-top:18px}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}th{{background:var(--soft)}}.controls{{display:flex;gap:14px;align-items:end;flex-wrap:wrap;margin:18px 0}}label{{display:grid;gap:5px;color:var(--muted);font-size:12px}}select,input[type=range],button{{font:inherit}}select{{padding:8px;border:1px solid var(--line);background:white;min-width:190px}}button{{padding:8px 12px;border:1px solid var(--line);background:white;color:var(--ink)}}button.active{{background:var(--ink);color:white}}.small-multiples{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.scatter-panel{{background:white;border:1px solid var(--line);padding:10px}}.scatter-panel h3{{margin:0 0 4px;font-size:18px}}canvas{{display:block;width:100%;aspect-ratio:1.35;background:#fafbfa}}.selected-point{{display:grid;grid-template-columns:150px 1fr;gap:16px;margin-top:14px;padding:14px;background:var(--soft)}}.selected-point img{{width:150px;aspect-ratio:1/1;object-fit:cover}}.selected-point dl{{display:grid;grid-template-columns:max-content 1fr;gap:4px 12px;margin:0}}.selected-point dt{{color:var(--muted)}}.selected-point dd{{margin:0}}.cluster-grid{{display:grid;grid-template-columns:1fr;gap:18px}}.cluster-case{{border:1px solid var(--line);background:white;padding:18px}}.cluster-case header{{position:static;border:0;background:transparent;display:flex;justify-content:space-between;gap:12px}}.cluster-case header h3{{margin:0}}.distribution-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}}.distribution{{font-size:11px}}.dist-row{{display:grid;grid-template-columns:minmax(90px,1.5fr) 2fr 32px;align-items:center;gap:7px;margin:4px 0}}.dist-bar{{height:8px;background:var(--soft)}}.dist-bar i{{display:block;height:100%;background:var(--green)}}.thumb-grid{{display:grid;grid-template-columns:repeat(9,1fr);gap:7px}}.thumb-grid.compact{{grid-template-columns:repeat(5,1fr)}}.thumb{{display:block;color:var(--ink);text-decoration:none;border:1px solid var(--line);overflow:hidden;background:white}}.thumb img{{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;background:var(--soft)}}.thumb span{{display:grid;padding:5px;font-size:9px;line-height:1.3}}.thumb b{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.thumb i,.thumb em{{font-style:normal;color:var(--muted)}}.thumb em{{color:#8d5036}}.scale-row{{margin-top:12px}}.scale-row>strong{{display:block;margin-bottom:5px}}.example-tabs,.model-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}}.example-panel{{display:none}}.example-panel.active{{display:block}}.example-group{{border-top:1px solid var(--line);padding:16px 0}}.example-pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.deep-explorer{{display:grid;grid-template-columns:1fr 250px;gap:16px}}.deep-canvas{{border:1px solid var(--line);background:white;min-height:420px}}.deep-detail{{background:var(--soft);padding:12px}}.deep-detail img{{width:100%;aspect-ratio:1/1;object-fit:cover}}.neighbor-grid{{display:grid;grid-template-columns:repeat(8,1fr);gap:7px}}.neighbor{{background:white;border:1px solid var(--line);padding:5px}}.neighbor.changed{{border-color:#b36a36}}.neighbor img{{display:block;width:100%;aspect-ratio:1/1;object-fit:cover}}.neighbor small,.neighbor b{{display:block;font-size:9px;line-height:1.25}}.neighbor b{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.boundary{{background:#fff1df;border-left:4px solid var(--orange);padding:12px;margin-top:12px}}.hidden{{display:none!important}}footer{{border-top:1px solid var(--line);padding:28px 0 60px;color:var(--muted);font-size:12px}}
@media(max-width:960px){{.stats,.sequence,.rule-grid{{grid-template-columns:repeat(2,1fr)}}.small-multiples,.plot-grid,.distribution-grid{{grid-template-columns:1fr}}.thumb-grid{{grid-template-columns:repeat(6,1fr)}}.neighbor-grid{{grid-template-columns:repeat(5,1fr)}}}}@media(max-width:650px){{.wrap{{padding:0 17px}}nav a{{display:none}}.stats,.sequence,.rule-grid,.example-pair,.deep-explorer,.selected-point{{grid-template-columns:1fr}}.thumb-grid,.thumb-grid.compact,.neighbor-grid{{grid-template-columns:repeat(3,1fr)}}.selected-point img{{width:100%}}}}
</style></head><body>
<header><div class="wrap"><nav><span class="brand">BioImages · representation analysis</span><a href="#scale">Shot scale</a><a href="#separation">Separation</a><a href="#clusters">Clusters</a><a href="#accuracy">Accuracy</a><a href="#secondary">Second layer</a><a href="#projection">Projection</a><a href="#seed">Seed neighbors</a></nav></div></header>
<main class="wrap"><section class="hero"><div class="eyebrow">Scale-conditioned exploration</div><h1>What explains species mixing in botanical image embeddings?</h1><p class="lede">Start with photographic scale. Only when scale does not explain the remaining mixing, inspect leaf state, reproductive structures and background context; then move into local projections and seed-guided neighborhoods.</p><div class="sequence"><div><b>1 · Shot scale</b><span>distant / mid-range / close-up</span></div><div><b>2 · Residual variables</b><span>leaf / reproductive / background</span></div><div><b>3 · Local projection</b><span>top principal directions</span></div><div><b>4 · Seed neighbors</b><span>expand until semantics change</span></div></div><div class="stats"><div class="stat"><strong>{counts['distant']:,}</strong><span>distant</span></div><div class="stat"><strong>{counts['mid-range']:,}</strong><span>mid-range</span></div><div class="stat"><strong>{counts['close-up']:,}</strong><span>close-up</span></div><div class="stat"><strong>{counts['uncertain']:,}</strong><span>uncertain</span></div></div></section>
<section class="section" id="scale"><h2>First layer · photographic scale</h2><p class="section-note">Scale describes what the camera shows, not the biological organ. A leaf may be a close-up, part of a mid-range branch, or a small element in a distant crown.</p><div class="rule-grid"><article class="rule"><h3>Distant</h3><p>Whole organism, tree, shrub or most of the crown is visible.</p></article><article class="rule"><h3>Mid-range</h3><p>A connected branch system, trunk section or several organs is visible.</p></article><article class="rule"><h3>Close-up</h3><p>One organ, surface or fine diagnostic detail dominates the frame.</p></article><article class="rule"><h3>Uncertain</h3><p>Rules conflict or the visible evidence is insufficient.</p></article></div><p><a href="review/">Open the 182-image human scale review →</a></p><figure class="wide-figure"><img src="assets/representation_by_scale.png" alt="BioCLIP, DINOv3 and EfficientNet PCA colored by shot scale"><figcaption>The same 1,641 embedding images, colored only by provisional shot scale.</figcaption></figure></section>
<section class="section" id="small-multiples"><h2>Scale-only cluster explorer</h2><p class="section-note">Each panel keeps the model's original k=20 cluster colors. Select one species to highlight it without assigning 63 competing colors. Click any point to inspect the real image and metadata.</p><div class="controls"><label>Representation<select id="scale-model">{''.join(f'<option value="{r}">{REP_LABELS[r]}</option>' for r in REPRESENTATIONS)}</select></label><label>Highlight species<select id="species-highlight"><option value="">None</option></select></label></div><div class="small-multiples">{''.join(f'<section class="scatter-panel"><h3>{s}</h3><canvas id="scale-canvas-{s}" width="440" height="326" aria-label="{s} images by cluster"></canvas></section>' for s in MAIN_SCALES)}</div><div class="selected-point" id="scale-selected"><div></div><p>Click a point to inspect its image, species, organ, scale and cluster.</p></div></section>
<section class="section" id="separation"><h2>Does species separation improve after controlling scale?</h2><p class="section-note">Purity and NMI are shown for the complete k=20 solution and again inside each shot-scale subset. They are descriptive; sample size and species support differ across scales.</p><figure class="wide-figure"><img src="assets/species_separation_by_scale.png" alt="Species purity and NMI before and after shot-scale conditioning"></figure><div class="method-note"><strong>Initial read:</strong> scale conditioning does not produce a uniform improvement. Mid-range has the clearest increase in all three spaces, close-up improves modestly, and distant remains highly mixed. {' · '.join(separation_findings)}.</div><div class="table-wrap"><table><thead><tr><th>Representation</th><th>Subset</th><th>n</th><th>Species purity</th><th>Species NMI</th><th>Organ purity</th><th>Organ NMI</th></tr></thead><tbody>{''.join(metrics_table)}</tbody></table></div></section>
<section class="section" id="clusters"><h2>Mixed clusters · before and after scale separation</h2><p class="section-note">The original centroid-nearest gallery is followed by distant, mid-range and close-up rows from the same cluster. Distribution bars make species, organ and scale composition visible rather than leaving them as isolated numbers.</p><div class="controls"><label>Representation<select id="cluster-model">{''.join(f'<option value="{r}">{REP_LABELS[r]}</option>' for r in REPRESENTATIONS)}</select></label><label>Mixed cluster<select id="cluster-select"></select></label></div><div id="cluster-case" class="cluster-grid"></div></section>
<section class="section" id="all-clusters"><h2>Every cluster, summarized visually</h2><p class="section-note">Each row is one original k=20 cluster. The same row shows its scale mix, organ mix and ranked species concentration; the interactive case study above provides the species names and real images.</p><div class="plot-grid">{''.join(f'<figure><img src="assets/{rep}_cluster_composition.png" alt="{REP_LABELS[rep]} cluster composition"><figcaption>{REP_LABELS[rep]}</figcaption></figure>' for rep in REPRESENTATIONS)}</div></section>
<section class="section" id="accuracy"><h2>Classification accuracy by shot scale</h2><p class="section-note">Top-1 species accuracy uses the existing strict 211-image individual-disjoint test set. Every bar shows correct / total, and the real correct and incorrect images sit directly below the chart.</p><figure class="wide-figure"><img src="assets/accuracy_by_scale.png" alt="Species classification accuracy by shot scale"></figure><div class="method-note"><strong>Initial read:</strong> distant views are consistently harder than close-up views. {' · '.join(accuracy_findings)}.</div><div class="example-tabs">{''.join(f'<button data-example-button="{r}">{REP_LABELS[r]}</button>' for r in REPRESENTATIONS)}</div>{build_accuracy_examples(payload['accuracy_examples'])}</section>
<section class="section" id="secondary"><h2>Second layer · explain residual mixing</h2><p class="section-note">These comparisons are made within shot scale. They are used only after scale has been separated, so background or leaf state cannot silently stand in for camera distance.</p><div class="rule-grid"><article class="rule"><h3>Leaf-on / leaf-off</h3><p>Tests whether foliage state or winter habit explains the remaining structure.</p></article><article class="rule"><h3>Reproductive visible / not visible</h3><p>Tests whether flower, fruit, cone, seed or inflorescence changes the representation.</p></article><article class="rule"><h3>Environmental / isolated</h3><p>Tests whether the model responds to scene context instead of the plant subject.</p></article><article class="rule"><h3>Organ and morphology</h3><p>Remaining separation is inspected for organ structure and cross-species morphology.</p></article></div><figure class="wide-figure"><img src="assets/secondary_explanatory_variables.png" alt="Species purity by second-layer visual variables"><figcaption>Weighted summaries across the three scale-specific subsets. Exact subgroup rows are preserved in the downloadable CSV.</figcaption></figure><div class="method-note"><strong>Initial read:</strong> leaf-off and isolated-subject groups are more species-concentrated, while reproductive-visible groups also separate better than not-visible groups. The isolated result is exploratory because only 30 embedding images currently receive that label.</div><p><a href="data/separation_metrics.csv">Download exact separation metrics →</a></p></section>
<section class="section" id="distance"><h2>How strongly does shot scale move images in embedding space?</h2>{pairwise_section}</section>
<section class="section" id="projection"><h2>Local projection and rotation</h2><p class="section-note">For the highest-priority mixed clusters, local PCA is recomputed from the original frozen vectors. Switch PC pairs or rotate across the first four principal directions; recolor by species, scale, organ, correctness or background, then click a point to open its image.</p><div class="method-note" data-deep-state="{deep_state}">{escape(embedding_note)}</div><div id="local-controls" class="controls"></div><div id="local-explorer" class="deep-explorer"></div></section>
<section class="section" id="seed"><h2>Seed-guided neighborhood explorer</h2><p class="section-note">Start from a representative image and reveal more neighbours until the visual semantics change. Original-space cosine neighbours exclude the seed's own observation. The slider expands to 100 when source vectors are available.</p><div id="seed-controls" class="controls"></div><div id="seed-explorer"></div></section>
</main><footer><div class="wrap">Generated from BioImages metadata, Gemma visual tags, the frozen k=20 assignments, strict test predictions and—when present—the original normalized frozen vectors. Provisional visual-group labels do not replace BioImages records.</div></footer>
<script src="data/analysis-payload.js?v={asset_version}"></script><script src="assets/complete-analysis.js?v={asset_version}"></script></body></html>'''


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    images = load_and_label()
    assignments = load_assignments()
    make_explanatory_plots(images, assignments)

    image_columns = [
        "image_id", "file", "species", "organ_category", "subview", "primary_label",
        "group", "individual_id", "shot_scale", "shot_scale_confidence",
        "shot_scale_rationale", "source_distant_evidence", "gemma_distant_evidence",
        "source_mid_evidence", "gemma_mid_evidence", "source_close_evidence",
        "gemma_close_evidence", "needs_scale_review", "leaf_state",
        "reproductive_visibility", "background_context",
    ]
    images[image_columns].to_csv(DATA / "image_visual_groups.csv", index=False)
    codebook = build_codebook(images)
    (DATA / "scale_codebook.json").write_text(
        json.dumps(codebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review = build_scale_review_sample(images)
    review.to_csv(DATA / "scale_review_sample.csv", index=False)

    projection = load_projection_points(images, assignments)
    separation = build_separation_metrics(images, assignments)
    predictions = load_test_predictions(images)
    accuracy, examples = build_accuracy(predictions)
    cluster_summary, clusters = build_cluster_summaries(images, assignments)
    embeddings = load_embeddings()
    pairwise = build_distance_metrics(images, embeddings)
    local, seeds = build_deep_payload(images, assignments, clusters, embeddings)

    chart_separation(separation)
    chart_accuracy(accuracy)
    chart_secondary(separation)
    chart_cluster_compositions(cluster_summary)
    chart_pairwise_distances(pairwise)

    payload = {
        "schema": [
            "id", "x", "y", "cluster", "species", "organ", "scale", "leaf",
            "reproductive", "background", "image", "source", "prediction", "correct",
        ],
        "counts": {key: int(value) for key, value in images.shot_scale.value_counts().items()},
        "species": sorted(images.species.unique().tolist()),
        "representations": {rep: REP_LABELS[rep] for rep in REPRESENTATIONS},
        "cluster_colors": CLUSTER_PALETTE,
        "projection": projection,
        "clusters": clusters,
        "accuracy_examples": examples,
        "local": local,
        "seeds": seeds,
        "embeddings_ready": sorted(embeddings),
    }
    payload_js = (
        "window.SCALE_ANALYSIS = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        + ";\n"
    )
    (DATA / "analysis-payload.js").write_text(payload_js, encoding="utf-8")
    asset_version = hashlib.sha256(
        payload_js.encode("utf-8") + (ASSETS / "complete-analysis.js").read_bytes()
    ).hexdigest()[:12]
    (HERE / "index.html").write_text(
        build_html(
            payload,
            separation,
            accuracy,
            len(embeddings) == len(REPRESENTATIONS),
            asset_version,
        ),
        encoding="utf-8",
    )
    summary = {
        "images": len(images),
        "embedding_images": int(assignments.image_id.nunique()),
        "strict_test_images": int(predictions.image_id.nunique()),
        "shot_scale_counts": payload["counts"],
        "representations": REPRESENTATIONS,
        "k": 20,
        "original_embeddings_available": sorted(embeddings),
        "local_projection_clusters": {rep: len(local[rep]) for rep in REPRESENTATIONS},
        "seed_queries": {rep: len(seeds[rep]) for rep in REPRESENTATIONS},
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
