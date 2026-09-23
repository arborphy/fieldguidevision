# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib==3.10.1",
#   "numpy==2.1.1",
#   "pandas==2.2.3",
#   "scikit-learn==1.6.1",
#   "umap-learn==0.5.7",
# ]
# ///
"""Final label-blind robustness and distance analysis for three saved embeddings."""

from __future__ import annotations

import html
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bioimages-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/bioimages-numba")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import hsv_to_rgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


SEED = 42
SEEDS = list(range(42, 52))
K_VALUES = [10, 20, 50]
REPRESENTATIONS = ["bioclip", "dinov3", "efficientnet_b0"]
REPRESENTATION_LABELS = {
    "bioclip": "BioCLIP 2.5",
    "dinov3": "DINOv3",
    "efficientnet_b0": "EfficientNet-B0",
}
ORGAN_MAPPING = {
    "bark": "bark",
    "cone": "cone",
    "fruit": "fruit",
    "inflorescence": "inflorescence",
    "leaf": "leaf",
    "seed": "seed",
    "stem": "other",
    "twig": "twig",
    "unspecified": "other",
    "whole plant": "whole plant",
    "whole tree": "whole plant",
    "whole tree (or vine)": "whole plant",
}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"
PLOT_DIR = HERE / "assets" / "final_plots"
MANIFEST_PATH = ROOT / "outputs" / "dinov3_strict_embeddings" / "split_manifest.csv"
METADATA_PATH = ROOT / "app" / "dashboard-data.json"
EMBEDDING_PATHS = {
    "bioclip": ROOT / "outputs" / "bioclip25_strict_embeddings" / "embeddings.npz",
    "dinov3": ROOT / "outputs" / "dinov3_strict_embeddings" / "embeddings.npz",
    "efficientnet_b0": ROOT / "outputs" / "efficientnet_b0_strict_embeddings" / "embeddings.npz",
}
REFERENCE_ASSIGNMENT_PATHS = {
    "bioclip": DATA_DIR / "cluster_assignments.csv",
    "dinov3": DATA_DIR / "cluster_assignments.csv",
    "efficientnet_b0": DATA_DIR / "efficientnet_k20_cluster_assignments.csv",
}
PROBE_COMPARISON = DATA_DIR / "frozen_probe_comparison.csv"


def load_inputs() -> tuple[dict[str, np.ndarray], list[str], pd.DataFrame]:
    manifest = pd.read_csv(MANIFEST_PATH)
    if len(manifest) != 1641 or manifest.image_id.nunique() != 1641:
        raise ValueError("Unexpected strict manifest")
    embeddings: dict[str, np.ndarray] = {}
    canonical_ids: list[str] | None = None
    for representation, path in EMBEDDING_PATHS.items():
        with np.load(path, allow_pickle=False) as payload:
            ids = payload["image_ids"].astype(str).tolist()
            matrix = payload["embeddings"].astype(np.float32)
        if canonical_ids is None:
            canonical_ids = ids
        elif ids != canonical_ids:
            raise ValueError(f"Image order differs for {representation}")
        if matrix.shape[0] != 1641:
            raise ValueError(f"Unexpected {representation} rows: {matrix.shape}")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=2e-5):
            raise ValueError(f"{representation} embeddings are not L2-normalized")
        embeddings[representation] = matrix
    if canonical_ids is None:
        raise ValueError("No embeddings found")
    if set(canonical_ids) != set(manifest.image_id):
        raise ValueError("Embedding IDs and manifest differ")
    aligned = manifest.set_index("image_id").loc[canonical_ids].reset_index()
    return embeddings, canonical_ids, aligned


def normalize_metadata(manifest: pd.DataFrame) -> pd.DataFrame:
    metadata_records = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["images"]
    metadata = pd.DataFrame(metadata_records)[["id", "organ", "organ_detail", "subview"]]
    normalized = manifest.merge(metadata, left_on="image_id", right_on="id", how="left", validate="one_to_one")
    normalized = normalized.rename(
        columns={
            "organ_category": "original_organ_category",
            "organ": "existing_dashboard_normalized_organ",
            "organ_detail": "original_organ_detail",
            "subview": "original_subview",
        }
    )
    normalized["normalized_organ"] = normalized.original_organ_category.map(ORGAN_MAPPING)
    if normalized.normalized_organ.isna().any():
        missing = sorted(normalized.loc[normalized.normalized_organ.isna(), "original_organ_category"].unique())
        raise ValueError(f"Unmapped organ values: {missing}")
    if not (normalized.normalized_organ == normalized.existing_dashboard_normalized_organ).all():
        mismatch = normalized.loc[
            normalized.normalized_organ != normalized.existing_dashboard_normalized_organ,
            ["original_organ_category", "normalized_organ", "existing_dashboard_normalized_organ"],
        ]
        raise ValueError(f"Normalization differs from dashboard metadata:\n{mismatch.head()}")
    output_columns = [
        "image_id",
        "file",
        "split",
        "individual_id",
        "ground_truth_species",
        "original_organ_category",
        "original_organ_detail",
        "original_subview",
        "normalized_organ",
    ]
    normalized[output_columns].to_csv(DATA_DIR / "organ_view_metadata_normalized.csv", index=False)
    counts = normalized.original_organ_category.value_counts()
    mapping_rows = []
    for original in sorted(ORGAN_MAPPING):
        target = ORGAN_MAPPING[original]
        if original in {"whole tree", "whole tree (or vine)", "whole plant"}:
            rationale = "semantically equivalent whole-plant view"
        elif original in {"stem", "unspecified"}:
            rationale = "non-specific or rare non-primary organ category"
        else:
            rationale = "identity mapping"
        mapping_rows.append(
            {
                "original_organ_category": original,
                "normalized_organ": target,
                "image_count": int(counts.get(original, 0)),
                "rationale": rationale,
            }
        )
    pd.DataFrame(mapping_rows).to_csv(DATA_DIR / "organ_normalization_mapping.csv", index=False)
    return normalized


def purity(labels: np.ndarray, targets: np.ndarray) -> float:
    total = 0
    for cluster_id in np.unique(labels):
        values, counts = np.unique(targets[labels == cluster_id], return_counts=True)
        total += int(counts.max())
    return total / len(labels)


def multiseed_clustering(
    embeddings: dict[str, np.ndarray], metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    species = metadata.ground_truth_species.to_numpy()
    organs = metadata.normalized_organ.to_numpy()
    run_rows = []
    summary_rows = []
    for representation, matrix in embeddings.items():
        for k in K_VALUES:
            run_labels: dict[int, np.ndarray] = {}
            for seed in SEEDS:
                model = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=1,
                    max_iter=500,
                    algorithm="lloyd",
                    random_state=seed,
                )
                labels = model.fit_predict(matrix)
                run_labels[seed] = labels
                run_rows.append(
                    {
                        "representation": representation,
                        "k": k,
                        "seed": seed,
                        "n_init": 1,
                        "species_nmi": float(normalized_mutual_info_score(species, labels)),
                        "organ_nmi": float(normalized_mutual_info_score(organs, labels)),
                        "species_purity": float(purity(labels, species)),
                        "organ_purity": float(purity(labels, organs)),
                        "inertia": float(model.inertia_),
                        "iterations": int(model.n_iter_),
                    }
                )
            pairwise_ari = [
                adjusted_rand_score(run_labels[first], run_labels[second])
                for first, second in combinations(SEEDS, 2)
            ]
            subset = pd.DataFrame(
                [row for row in run_rows if row["representation"] == representation and row["k"] == k]
            )
            row = {"representation": representation, "k": k, "seeds": len(SEEDS), "n_init_per_seed": 1}
            for metric in ["species_nmi", "organ_nmi", "species_purity", "organ_purity"]:
                row[f"{metric}_mean"] = float(subset[metric].mean())
                row[f"{metric}_std"] = float(subset[metric].std(ddof=1))
                row[f"{metric}_min"] = float(subset[metric].min())
                row[f"{metric}_max"] = float(subset[metric].max())
            row.update(
                {
                    "stability_pairwise_ari_mean": float(np.mean(pairwise_ari)),
                    "stability_pairwise_ari_std": float(np.std(pairwise_ari, ddof=1)),
                    "stability_pairwise_ari_min": float(np.min(pairwise_ari)),
                    "stability_pairwise_ari_max": float(np.max(pairwise_ari)),
                    "stability_pairs": len(pairwise_ari),
                }
            )
            summary_rows.append(row)
    runs = pd.DataFrame(run_rows)
    summary = pd.DataFrame(summary_rows)
    runs.to_csv(DATA_DIR / "multiseed_cluster_runs.csv", index=False)
    summary.to_csv(DATA_DIR / "multiseed_cluster_summary.csv", index=False)
    return runs, summary


def load_reference_k20_labels(image_ids: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for representation in REPRESENTATIONS:
        assignments = pd.read_csv(REFERENCE_ASSIGNMENT_PATHS[representation])
        assignments = assignments.loc[
            (assignments.representation == representation) & (assignments.k == 20)
        ].set_index("image_id")
        if set(assignments.index) != set(image_ids):
            raise ValueError(f"Reference k=20 assignment IDs differ for {representation}")
        result[representation] = assignments.loc[image_ids, "cluster_id"].to_numpy(dtype=int)
    return result


def reference_cluster_summaries(
    metadata: pd.DataFrame, labels_by_representation: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for representation, labels in labels_by_representation.items():
        for cluster_id in range(20):
            members = metadata.loc[labels == cluster_id]
            species_counts = members.ground_truth_species.value_counts()
            organ_counts = members.normalized_organ.value_counts()
            rows.append(
                {
                    "representation": representation,
                    "k": 20,
                    "cluster_id": cluster_id,
                    "size_all": len(members),
                    "size_test": int((members.split == "test").sum()),
                    "species_count": int(members.ground_truth_species.nunique()),
                    "dominant_species": species_counts.index[0],
                    "species_purity": float(species_counts.iloc[0] / len(members)),
                    "species_distribution": json.dumps(
                        {str(key): int(value) for key, value in species_counts.items()}, ensure_ascii=False
                    ),
                    "normalized_organ_count": int(members.normalized_organ.nunique()),
                    "dominant_normalized_organ": organ_counts.index[0],
                    "normalized_organ_purity": float(organ_counts.iloc[0] / len(members)),
                    "normalized_organ_distribution": json.dumps(
                        {str(key): int(value) for key, value in organ_counts.items()}, ensure_ascii=False
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(DATA_DIR / "reference_k20_cluster_summary_normalized.csv", index=False)
    return result


def effect_size(same: np.ndarray, different: np.ndarray) -> float:
    pooled = math.sqrt((float(same.var(ddof=1)) + float(different.var(ddof=1))) / 2)
    return float((same.mean() - different.mean()) / pooled) if pooled else math.nan


def describe_similarity(
    representation: str,
    comparison: str,
    same: np.ndarray,
    different: np.ndarray,
) -> dict[str, object]:
    row: dict[str, object] = {
        "representation": representation,
        "comparison": comparison,
        "same_pair_count": len(same),
        "different_pair_count": len(different),
        "same_mean": float(same.mean()),
        "same_std": float(same.std(ddof=1)),
        "different_mean": float(different.mean()),
        "different_std": float(different.std(ddof=1)),
        "mean_difference_same_minus_different": float(same.mean() - different.mean()),
        "cohens_d": effect_size(same, different),
    }
    for name, values in [("same", same), ("different", different)]:
        for quantile in [0.05, 0.25, 0.5, 0.75, 0.95]:
            row[f"{name}_q{int(quantile * 100):02d}"] = float(np.quantile(values, quantile))
    return row


def distance_analysis(
    embeddings: dict[str, np.ndarray], metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(metadata)
    upper_i, upper_j = np.triu_indices(n, k=1)
    cross_observation = (
        metadata.individual_id.to_numpy()[upper_i]
        != metadata.individual_id.to_numpy()[upper_j]
    )
    pair_i = upper_i[cross_observation]
    pair_j = upper_j[cross_observation]
    species = metadata.ground_truth_species.to_numpy()
    organs = metadata.normalized_organ.to_numpy()
    masks = {
        "species": species[pair_i] == species[pair_j],
        "normalized_organ": organs[pair_i] == organs[pair_j],
    }
    summary_rows = []
    values_by_key: dict[tuple[str, str, str], np.ndarray] = {}
    global_min = 1.0
    global_max = -1.0
    for representation, matrix in embeddings.items():
        similarities = (matrix @ matrix.T)[pair_i, pair_j]
        global_min = min(global_min, float(similarities.min()))
        global_max = max(global_max, float(similarities.max()))
        for comparison, same_mask in masks.items():
            same = similarities[same_mask]
            different = similarities[~same_mask]
            values_by_key[(representation, comparison, "same")] = same
            values_by_key[(representation, comparison, "different")] = different
            summary_rows.append(describe_similarity(representation, comparison, same, different))
    low = max(-1.0, global_min - 0.02)
    high = min(1.0, global_max + 0.02)
    bins = np.linspace(low, high, 81)
    hist_rows = []
    for (representation, comparison, relation), values in values_by_key.items():
        density, edges = np.histogram(values, bins=bins, density=True)
        for index, value in enumerate(density):
            hist_rows.append(
                {
                    "representation": representation,
                    "comparison": comparison,
                    "relation": relation,
                    "bin_left": float(edges[index]),
                    "bin_right": float(edges[index + 1]),
                    "bin_center": float((edges[index] + edges[index + 1]) / 2),
                    "density": float(value),
                }
            )
    summary = pd.DataFrame(summary_rows)
    histograms = pd.DataFrame(hist_rows)
    summary.to_csv(DATA_DIR / "distance_analysis_summary.csv", index=False)
    histograms.to_csv(DATA_DIR / "distance_similarity_histograms.csv", index=False)
    plot_distance_distributions(histograms)
    return summary, histograms


def plot_distance_distributions(histograms: pd.DataFrame) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    comparison_labels = {"species": "Species", "normalized_organ": "Normalized organ/view"}
    colors = {"same": "#28664a", "different": "#bd5b43"}
    for row_index, representation in enumerate(REPRESENTATIONS):
        for column_index, comparison in enumerate(["species", "normalized_organ"]):
            axis = axes[row_index, column_index]
            subset = histograms.loc[
                (histograms.representation == representation)
                & (histograms.comparison == comparison)
            ]
            for relation in ["same", "different"]:
                series = subset.loc[subset.relation == relation]
                axis.plot(series.bin_center, series.density, color=colors[relation], linewidth=1.8, label=relation)
                axis.fill_between(series.bin_center, 0, series.density, color=colors[relation], alpha=0.10)
            axis.set_title(f"{REPRESENTATION_LABELS[representation]} · {comparison_labels[comparison]}")
            axis.set_ylabel("Density")
            axis.grid(axis="y", color="#d9dedb", linewidth=0.6, alpha=0.6)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == 0 and column_index == 1:
                axis.legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("Cosine similarity in original normalized embedding space")
    figure.suptitle("Cross-observation pairwise cosine similarity", fontsize=16)
    figure.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_DIR / "distance_similarity_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def categorical_colors(values: np.ndarray, ordered_categories: list[str]) -> np.ndarray:
    hues = np.linspace(0, 1, len(ordered_categories), endpoint=False)
    rgb = hsv_to_rgb(np.column_stack([hues, np.full(len(hues), 0.60), np.full(len(hues), 0.72)]))
    mapping = {category: rgb[index] for index, category in enumerate(ordered_categories)}
    return np.asarray([mapping[str(value)] for value in values])


def plot_projection_pair(
    representation: str,
    pca_coordinates: np.ndarray,
    umap_coordinates: np.ndarray,
    values: np.ndarray,
    categories: list[str],
    coloring: str,
) -> None:
    colors = categorical_colors(values.astype(str), categories)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, (name, coordinates) in zip(axes, [("PCA", pca_coordinates), ("UMAP", umap_coordinates)]):
        axis.scatter(coordinates[:, 0], coordinates[:, 1], s=9, c=colors, alpha=0.72, linewidths=0)
        axis.set_title(f"{name} · colored by {coloring}")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
    legend_bottom = 0.05
    if len(categories) <= 12:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=categorical_colors(np.asarray([category]), categories)[0], markeredgewidth=0, markersize=6, label=category)
            for category in categories
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=min(5, len(categories)),
            frameon=False,
            fontsize=8,
        )
        legend_bottom = 0.17
    figure.suptitle(f"{REPRESENTATION_LABELS[representation]} · 2D visualization", fontsize=15)
    figure.text(0.5, 0.01, "Visualization only — 2D distances are not original embedding distances.", ha="center", fontsize=8, color="#667069")
    figure.tight_layout(rect=(0, legend_bottom, 1, 0.96))
    figure.savefig(PLOT_DIR / f"{representation}_pca_umap_{coloring.replace('/', '_')}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def projection_visualizations(
    embeddings: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    reference_labels: dict[str, np.ndarray],
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    species = metadata.ground_truth_species.to_numpy(dtype=str)
    organs = metadata.normalized_organ.to_numpy(dtype=str)
    species_categories = sorted(np.unique(species).tolist())
    organ_categories = sorted(np.unique(organs).tolist())
    for representation, matrix in embeddings.items():
        pca_coordinates = PCA(n_components=2, random_state=SEED).fit_transform(matrix)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=20,
            min_dist=0.15,
            metric="cosine",
            random_state=SEED,
            transform_seed=SEED,
        )
        umap_coordinates = reducer.fit_transform(matrix)
        np.savez_compressed(
            DATA_DIR / f"{representation}_final_projection_coordinates.npz",
            image_ids=metadata.image_id.to_numpy(dtype=str),
            pca=pca_coordinates.astype(np.float32),
            umap=umap_coordinates.astype(np.float32),
        )
        plot_projection_pair(
            representation,
            pca_coordinates,
            umap_coordinates,
            species,
            species_categories,
            "species",
        )
        plot_projection_pair(
            representation,
            pca_coordinates,
            umap_coordinates,
            organs,
            organ_categories,
            "normalized-organ",
        )
        cluster_values = np.asarray([f"cluster {value:02d}" for value in reference_labels[representation]])
        plot_projection_pair(
            representation,
            pca_coordinates,
            umap_coordinates,
            cluster_values,
            [f"cluster {value:02d}" for value in range(20)],
            "k20-cluster",
        )


def robustness_summary(
    multiseed: pd.DataFrame, distance: pd.DataFrame
) -> dict[str, object]:
    clustering_winners = []
    for k in K_VALUES:
        subset = multiseed.loc[multiseed.k == k]
        for metric in ["species_nmi_mean", "species_purity_mean", "organ_nmi_mean", "organ_purity_mean"]:
            winner = subset.loc[subset[metric].idxmax(), "representation"]
            clustering_winners.append({"k": k, "metric": metric, "winner": winner})
    distance_winners = []
    for comparison in ["species", "normalized_organ"]:
        subset = distance.loc[distance.comparison == comparison]
        for metric in ["mean_difference_same_minus_different", "cohens_d"]:
            winner = subset.loc[subset[metric].idxmax(), "representation"]
            distance_winners.append({"comparison": comparison, "metric": metric, "winner": winner})
    result = {
        "protocol": {
            "k_values": K_VALUES,
            "seeds": SEEDS,
            "kmeans_n_init_per_seed": 1,
            "stability": "mean/std/min/max pairwise ARI over 45 seed pairs",
            "distance_pairs": "all unordered cross-observation image pairs; identical pair masks in all spaces",
        },
        "clustering_metric_winners": clustering_winners,
        "distance_metric_winners": distance_winners,
        "species_clustering_winner_consistent": len(
            {row["winner"] for row in clustering_winners if row["metric"].startswith("species")}
        )
        == 1,
        "organ_clustering_winner_consistent": len(
            {row["winner"] for row in clustering_winners if row["metric"].startswith("organ")}
        )
        == 1,
    }
    (DATA_DIR / "robustness_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def pct(value: float) -> str:
    return f"{value:.1%}"


def mean_std(row: pd.Series, metric: str, digits: int = 3) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} ± {row[f'{metric}_std']:.{digits}f}"


def top_distribution(raw: str, limit: int = 4) -> str:
    values = json.loads(raw)
    return ", ".join(f"{key} {value}" for key, value in list(values.items())[:limit])


def build_report(
    multiseed: pd.DataFrame,
    distance: pd.DataFrame,
    reference_clusters: pd.DataFrame,
    robustness: dict[str, object],
) -> None:
    probe = pd.read_csv(PROBE_COMPARISON)
    bio = multiseed.loc[multiseed.representation == "bioclip"]
    dino = multiseed.loc[multiseed.representation == "dinov3"]
    efficient = multiseed.loc[multiseed.representation == "efficientnet_b0"]

    species_cluster_winners = {
        row["winner"]
        for row in robustness["clustering_metric_winners"]
        if row["metric"].startswith("species")
    }
    organ_cluster_winners = {
        row["winner"]
        for row in robustness["clustering_metric_winners"]
        if row["metric"].startswith("organ")
    }
    species_distance_winners = {
        row["winner"]
        for row in robustness["distance_metric_winners"]
        if row["comparison"] == "species"
    }
    organ_distance_winners = {
        row["winner"]
        for row in robustness["distance_metric_winners"]
        if row["comparison"] == "normalized_organ"
    }
    species_text = (
        "BioCLIP has the strongest species association for every reported clustering metric and k, and for the original-space species distance contrasts."
        if species_cluster_winners == {"bioclip"} and species_distance_winners == {"bioclip"}
        else "Species-association rankings are not fully uniform across clustering and original-space distance metrics; see the tables below."
    )
    organ_winners_by_k = []
    for k in K_VALUES:
        winners = {
            row["winner"]
            for row in robustness["clustering_metric_winners"]
            if row["k"] == k and row["metric"].startswith("organ")
        }
        organ_winners_by_k.append(
            f"k={k}: {', '.join(REPRESENTATION_LABELS[key] for key in sorted(winners))}"
        )
    organ_text = (
        "No single representation leads every organ/view protocol. "
        + "; ".join(organ_winners_by_k)
        + f". Original-space distance contrasts are strongest for {', '.join(REPRESENTATION_LABELS[key] for key in sorted(organ_distance_winners))}."
    )

    comparison_rows = []
    for k in K_VALUES:
        for representation in REPRESENTATIONS:
            row = multiseed.loc[
                (multiseed.representation == representation) & (multiseed.k == k)
            ].iloc[0]
            comparison_rows.append(
                f"<tr><td>{REPRESENTATION_LABELS[representation]}</td><td>{k}</td>"
                f"<td>{mean_std(row, 'species_nmi')}</td><td>{mean_std(row, 'organ_nmi')}</td>"
                f"<td>{mean_std(row, 'species_purity')}</td><td>{mean_std(row, 'organ_purity')}</td>"
                f"<td>{mean_std(row, 'stability_pairwise_ari')}</td></tr>"
            )
    distance_rows = []
    for row in distance.itertuples(index=False):
        distance_rows.append(
            f"<tr><td>{REPRESENTATION_LABELS[row.representation]}</td>"
            f"<td>{'Species' if row.comparison == 'species' else 'Normalized organ/view'}</td>"
            f"<td>{row.same_pair_count:,}</td><td>{row.different_pair_count:,}</td>"
            f"<td>{row.same_mean:.3f}</td><td>{row.different_mean:.3f}</td>"
            f"<td>{row.mean_difference_same_minus_different:.3f}</td><td>{row.cohens_d:.3f}</td></tr>"
        )
    probe_rows = "".join(
        f"<tr><td>{html.escape(row.model)}</td><td>{pct(row.species_top1_accuracy)}</td>"
        f"<td>{pct(row.species_top5_accuracy)}</td><td>{pct(row.genus_top1_accuracy)}</td>"
        f"<td>{pct(row.family_top1_accuracy)}</td></tr>"
        for row in probe.itertuples(index=False)
    )
    projection_blocks = []
    for representation in REPRESENTATIONS:
        projection_blocks.append(
            f"""
            <details><summary>{REPRESENTATION_LABELS[representation]} projections</summary>
              <div class="projection-grid">
                <figure><img loading="lazy" src="assets/final_plots/{representation}_pca_umap_species.png" alt="{representation} PCA and UMAP colored by species"><figcaption>Colored by species</figcaption></figure>
                <figure><img loading="lazy" src="assets/final_plots/{representation}_pca_umap_normalized-organ.png" alt="{representation} PCA and UMAP colored by normalized organ"><figcaption>Colored by normalized organ/view</figcaption></figure>
                <figure><img loading="lazy" src="assets/final_plots/{representation}_pca_umap_k20-cluster.png" alt="{representation} PCA and UMAP colored by reference k20 cluster"><figcaption>Colored by reference k=20 cluster</figcaption></figure>
              </div>
            </details>"""
        )
    atlas_blocks = []
    for representation in REPRESENTATIONS:
        cards = []
        subset = reference_clusters.loc[reference_clusters.representation == representation].sort_values("cluster_id")
        for row in subset.itertuples(index=False):
            cards.append(
                f"""
                <article class="cluster-card">
                  <img loading="lazy" src="assets/cluster_sheets/{representation}/k20/cluster_{row.cluster_id:02d}.jpg" alt="{representation} centroid-nearest images for cluster {row.cluster_id}">
                  <div class="copy"><h3>Cluster {row.cluster_id:02d}</h3>
                  <p>{row.size_all} images · {row.size_test} test · species purity {pct(row.species_purity)} · normalized-organ purity {pct(row.normalized_organ_purity)}</p>
                  <p><b>Species</b> {html.escape(top_distribution(row.species_distribution))}</p>
                  <p><b>Normalized organs</b> {html.escape(top_distribution(row.normalized_organ_distribution))}</p></div>
                </article>"""
            )
        atlas_blocks.append(
            f"<details><summary>{REPRESENTATION_LABELS[representation]} · 20 reference clusters</summary><div class='cluster-grid'>{''.join(cards)}</div></details>"
        )

    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final three-representation analysis</title>
<style>
:root{{--ink:#17221d;--muted:#667069;--line:#d8ded9;--soft:#f3f5f3;--paper:#fff;--accent:#315f4a}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--paper);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}}main{{width:min(1540px,calc(100% - 40px));margin:auto;padding:48px 0 80px}}h1,h2,h3{{font-family:Georgia,"Times New Roman",serif;font-weight:400}}h1{{font-size:clamp(2.2rem,5vw,4.4rem);line-height:1;margin:8px 0 16px}}h2{{font-size:clamp(1.6rem,3vw,2.5rem);margin:0 0 12px}}h3{{margin:0;font-size:1.12rem}}.eyebrow,.small,figcaption{{color:var(--muted);font-size:.82rem}}.eyebrow{{text-transform:uppercase;letter-spacing:.11em;font-weight:650}}header{{max-width:1050px;padding-bottom:30px;border-bottom:1px solid var(--line)}}section{{padding-top:52px}}.comparison{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:24px 0}}.comparison div{{background:var(--paper);padding:18px}}.comparison h3{{margin-bottom:8px}}.note{{border-left:3px solid var(--accent);padding:10px 14px;color:var(--muted);background:var(--soft);max-width:1100px}}.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:.85rem}}th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--muted);font-size:.73rem;text-transform:uppercase;letter-spacing:.05em}}details{{border-top:1px solid var(--line);padding:14px 0}}summary{{cursor:pointer;font-weight:650}}.projection-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:18px}}figure{{margin:0;min-width:0}}figure img,.distance-plot{{width:100%;display:block;border:1px solid var(--line)}}figcaption{{margin-top:6px}}.cluster-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:18px}}.cluster-card{{border:1px solid var(--line);min-width:0}}.cluster-card>img{{width:100%;display:block;border-bottom:1px solid var(--line)}}.copy{{padding:13px 15px 15px}}.copy p{{margin:6px 0}}a{{color:var(--accent)}}footer{{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:.82rem}}@media(max-width:1000px){{.projection-grid{{grid-template-columns:1fr}}}}@media(max-width:900px){{.comparison,.cluster-grid{{grid-template-columns:1fr}}}}@media(max-width:520px){{main{{width:calc(100% - 24px);padding-top:28px}}}}
</style></head><body><main>
<header><div class="eyebrow">BioImages · final representation robustness analysis</div><h1>Species identity and image content</h1><p>Frozen, L2-normalized BioCLIP, DINOv3 and EfficientNet-B0 embeddings are compared without new inference or model training. K-means receives no species or organ labels.</p></header>
<section><h2>Dataset-specific comparison</h2><div class="comparison"><div><h3>Species identity</h3><p>{species_text}</p></div><div><h3>Organ/view</h3><p>{organ_text}</p></div></div><p class="note">These are observations for this 1,641-image BioImages strict split only. They do not establish a general ranking of the models or a causal explanation.</p></section>
<section><h2>Multi-seed clustering robustness</h2><p class="note">Each cell is mean ± sample standard deviation across the 10 prespecified seeds 42–51. Every run uses k-means++ with n_init=1; no run is selected. Stability is pairwise ARI over all 45 seed pairs. Labels are joined only after clustering.</p><div class="table"><table><thead><tr><th>Space</th><th>k</th><th>Species NMI</th><th>Organ NMI</th><th>Species purity</th><th>Organ purity</th><th>Stability ARI</th></tr></thead><tbody>{''.join(comparison_rows)}</tbody></table></div></section>
<section><h2>Original-space cosine similarity</h2><p class="note">All unordered cross-observation image pairs are used. The exact same pair indices and same/different masks are applied to all three embedding spaces. This analysis does not depend on K-means or 2D projection.</p><img class="distance-plot" src="assets/final_plots/distance_similarity_distributions.png" alt="Same and different species and organ cosine similarity distributions"><div class="table"><table><thead><tr><th>Space</th><th>Comparison</th><th>Same pairs</th><th>Different pairs</th><th>Same mean</th><th>Different mean</th><th>Difference</th><th>Cohen's d</th></tr></thead><tbody>{''.join(distance_rows)}</tbody></table></div></section>
<section><h2>PCA and UMAP visualization</h2><p class="note">PCA and UMAP are visualization tools only. 2D distances are not original embedding distances and should not be used as replacement evidence for the cosine analysis above.</p>{''.join(projection_blocks)}</section>
<section><h2>Reference k=20 contact sheets</h2><p>The existing deterministic seed-42, n_init=20 fits are retained only as qualitative reference atlases; they are not chosen from the 10 robustness runs. Organ summaries below use the normalized field. No cluster is automatically named.</p>{''.join(atlas_blocks)}</section>
<section><h2>Frozen linear-probe context</h2><div class="table"><table><thead><tr><th>Representation</th><th>Species Top-1</th><th>Species Top-5</th><th>Genus Top-1</th><th>Family Top-1</th></tr></thead><tbody>{probe_rows}</tbody></table></div></section>
<section><h2>Data exports</h2><p><a href="data/organ_normalization_mapping.csv">organ mapping</a> · <a href="data/organ_view_metadata_normalized.csv">normalized metadata</a> · <a href="data/multiseed_cluster_runs.csv">all clustering runs</a> · <a href="data/multiseed_cluster_summary.csv">clustering summary</a> · <a href="data/distance_analysis_summary.csv">distance summary</a> · <a href="data/distance_similarity_histograms.csv">distance histograms</a> · <a href="data/reference_k20_cluster_summary_normalized.csv">reference k=20 clusters</a> · <a href="data/robustness_check.json">robustness check</a></p></section>
<footer>Strict split: 1,641 images, 63 species, observation-disjoint train/validation/test. No embedding extraction, classifier training, label-guided clustering, seed selection or automatic morphology interpretation was performed in this final analysis.</footer>
</main></body></html>"""
    (HERE / "efficientnet_control.html").write_text(report, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, image_ids, manifest = load_inputs()
    metadata = normalize_metadata(manifest)
    if "--report-only" in sys.argv:
        multiseed_summary = pd.read_csv(DATA_DIR / "multiseed_cluster_summary.csv")
        distance_summary = pd.read_csv(DATA_DIR / "distance_analysis_summary.csv")
        reference_clusters = pd.read_csv(DATA_DIR / "reference_k20_cluster_summary_normalized.csv")
        robustness = json.loads((DATA_DIR / "robustness_check.json").read_text(encoding="utf-8"))
        build_report(multiseed_summary, distance_summary, reference_clusters, robustness)
        print(json.dumps({"report": str(HERE / "efficientnet_control.html")}, indent=2))
        return
    if "--projections-only" in sys.argv:
        reference_labels = load_reference_k20_labels(image_ids)
        projection_visualizations(embeddings, metadata, reference_labels)
        print(json.dumps({"projection_images": len(list(PLOT_DIR.glob("*_pca_umap_*.png")))}, indent=2))
        return
    multiseed_runs, multiseed_summary = multiseed_clustering(embeddings, metadata)
    reference_labels = load_reference_k20_labels(image_ids)
    reference_clusters = reference_cluster_summaries(metadata, reference_labels)
    distance_summary, _ = distance_analysis(embeddings, metadata)
    projection_visualizations(embeddings, metadata, reference_labels)
    robustness = robustness_summary(multiseed_summary, distance_summary)
    build_report(multiseed_summary, distance_summary, reference_clusters, robustness)
    run_summary = {
        "images": len(image_ids),
        "representations": REPRESENTATIONS,
        "k_values": K_VALUES,
        "seeds": SEEDS,
        "cluster_runs": len(multiseed_runs),
        "organ_mapping_rows": len(ORGAN_MAPPING),
        "normalized_organs": sorted(metadata.normalized_organ.unique().tolist()),
        "distance_protocol": "all unordered cross-observation pairs",
        "projection_images": len(list(PLOT_DIR.glob("*_pca_umap_*.png"))),
        "reference_contact_sheets": sum(
            len(list((HERE / "assets" / "cluster_sheets" / representation / "k20").glob("cluster_*.jpg")))
            for representation in REPRESENTATIONS
        ),
    }
    (DATA_DIR / "final_analysis_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
