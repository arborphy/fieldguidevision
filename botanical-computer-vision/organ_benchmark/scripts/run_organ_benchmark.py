#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib==3.10.1",
#   "numpy==2.2.6",
#   "pandas==2.2.3",
#   "scikit-learn==1.7.2",
#   "tabulate==0.9.0",
# ]
# ///
"""Run the BioImages canonical organ/view benchmark from saved embeddings.

The benchmark deliberately treats BioImages' single label as a canonical-label
target, not as exhaustive visual truth. Images remain in Google Drive / at their
source URLs; this script writes only manifests, metrics, figures, and static HTML.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/organ-benchmark-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


SEED = 42
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
BOOTSTRAP_REPLICATES = 2000
MODEL_LABELS = {
    "bioclip": "BioCLIP 2.5",
    "dinov3": "DINOv3",
    "efficientnet_b0": "EfficientNet-B0",
}

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parent
METADATA_PATH = ROOT / "work" / "bioimages_metadata.json"
BASE_MANIFEST_PATH = ROOT / "outputs" / "dinov3_strict_embeddings" / "split_manifest.csv"
EMBEDDING_PATHS = {
    "bioclip": ROOT / "outputs" / "bioclip25_strict_embeddings" / "embeddings.npz",
    "dinov3": ROOT / "outputs" / "dinov3_strict_embeddings" / "embeddings.npz",
    "efficientnet_b0": ROOT / "outputs" / "efficientnet_b0_strict_embeddings" / "embeddings.npz",
}

DIRS = {
    "splits": OUT / "split_manifests",
    "predictions": OUT / "predictions",
    "metrics": OUT / "metrics",
    "figures": OUT / "figures",
    "galleries": OUT / "galleries",
    "audit": OUT / "ambiguity_audit",
}


def stable_int(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def make_dirs() -> None:
    for directory in DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    metadata = pd.DataFrame(json.loads(METADATA_PATH.read_text(encoding="utf-8")))
    base = pd.read_csv(BASE_MANIFEST_PATH)
    required_metadata = {
        "image_id", "scientific_name", "organ_tag", "organ_category", "subview",
        "individual_id", "thumbnail_url", "image_url",
    }
    if missing := required_metadata - set(metadata.columns):
        raise ValueError(f"metadata missing fields: {sorted(missing)}")
    if len(metadata) != 1899 or metadata.image_id.nunique() != 1899:
        raise ValueError("Expected 1,899 unique metadata records")
    if len(base) != 1641 or base.image_id.nunique() != 1641:
        raise ValueError("Expected 1,641 unique rows in the reused embedding manifest")

    joined = base[["image_id", "file", "split"]].merge(
        metadata,
        on="image_id",
        how="left",
        validate="one_to_one",
        suffixes=("_manifest", ""),
    )
    if joined[list(required_metadata - {"image_id"})].isna().any().any():
        raise ValueError("Failed to attach metadata to every embedding row")
    if not (joined.individual_id == base.individual_id).all():
        raise ValueError("individual_id mismatch between metadata and embedding manifest")
    if not (joined.organ_category == base.organ_category).all():
        raise ValueError("organ_category mismatch between metadata and embedding manifest")
    joined = joined.rename(columns={"scientific_name": "species", "split": "individual_split"})

    matrices: dict[str, np.ndarray] = {}
    canonical_ids: list[str] | None = None
    embedding_audit = []
    for model, path in EMBEDDING_PATHS.items():
        with np.load(path, allow_pickle=False) as payload:
            ids = payload["image_ids"].astype(str).tolist()
            matrix = payload["embeddings"].astype(np.float32)
            splits = payload["splits"].astype(str).tolist()
            normalized = bool(payload["normalized"])
            source_model = str(payload["model"])
        if matrix.shape[0] != 1641 or len(ids) != 1641:
            raise ValueError(f"Unexpected embedding shape for {model}: {matrix.shape}")
        if canonical_ids is None:
            canonical_ids = ids
        elif ids != canonical_ids:
            raise ValueError(f"Embedding row order differs for {model}")
        if splits != joined.set_index("image_id").loc[ids, "individual_split"].tolist():
            raise ValueError(f"Stored split names differ for {model}")
        norms = np.linalg.norm(matrix, axis=1)
        if not normalized or not np.allclose(norms, 1.0, atol=2e-5):
            raise ValueError(f"{model} embeddings are not L2 normalized")
        matrices[model] = matrix
        embedding_audit.append(
            {
                "model": model,
                "source_model": source_model,
                "rows": int(matrix.shape[0]),
                "dimensions": int(matrix.shape[1]),
                "normalized": normalized,
                "min_l2_norm": float(norms.min()),
                "max_l2_norm": float(norms.max()),
                "source_path": str(path.relative_to(ROOT)),
            }
        )

    assert canonical_ids is not None
    joined = joined.set_index("image_id").loc[canonical_ids].reset_index()
    if set(joined.image_id) != set(base.image_id):
        raise ValueError("Embedding IDs and manifest IDs differ")
    pd.DataFrame(embedding_audit).to_csv(DIRS["metrics"] / "embedding_audit.csv", index=False)
    return joined, matrices, canonical_ids


def choose_species_split(frame: pd.DataFrame) -> dict[str, str]:
    """Deterministic constrained random search for balanced species groups."""
    species = sorted(frame.species.unique())
    if len(species) != 63:
        raise ValueError(f"Expected 63 species in embedding universe, found {len(species)}")
    labels = sorted(frame.organ_tag.unique())
    counts = pd.crosstab(frame.species, frame.organ_tag).reindex(index=species, columns=labels, fill_value=0)
    totals = counts.sum(axis=0).to_numpy(dtype=float)
    total_n = float(len(frame))
    target_species = {"train": 44, "validation": 9, "test": 10}
    target_fraction = {name: count / len(species) for name, count in target_species.items()}
    rng = np.random.default_rng(SEED)
    best: tuple[float, np.ndarray] | None = None

    for _ in range(50000):
        order = rng.permutation(len(species))
        partitions = {
            "train": order[:44],
            "validation": order[44:53],
            "test": order[53:],
        }
        train_counts = counts.iloc[partitions["train"]].sum(axis=0).to_numpy()
        val_counts = counts.iloc[partitions["validation"]].sum(axis=0).to_numpy()
        test_counts = counts.iloc[partitions["test"]].sum(axis=0).to_numpy()
        # Any label evaluated outside train must have been seen in train.
        if np.any(((val_counts + test_counts) > 0) & (train_counts == 0)):
            continue
        # Every reasonably populated label should be represented in held-out species.
        common = totals >= 15
        if np.any(common & ((val_counts == 0) | (test_counts == 0))):
            continue
        score = 0.0
        for split_name, indices in partitions.items():
            split_counts = counts.iloc[indices].sum(axis=0).to_numpy(dtype=float)
            expected = totals * target_fraction[split_name]
            score += float(np.mean(((split_counts - expected) / np.sqrt(expected + 1.0)) ** 2))
            split_n = float(split_counts.sum())
            score += 2.0 * ((split_n / total_n) - target_fraction[split_name]) ** 2
        if best is None or score < best[0]:
            best = (score, order.copy())

    if best is None:
        raise RuntimeError("Could not construct a species-disjoint split satisfying coverage constraints")
    order = best[1]
    mapping = {}
    for index in order[:44]:
        mapping[species[int(index)]] = "train"
    for index in order[44:53]:
        mapping[species[int(index)]] = "validation"
    for index in order[53:]:
        mapping[species[int(index)]] = "test"
    return mapping


def build_manifests(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    species_mapping = choose_species_split(frame)
    result = frame.copy()
    result["species_split"] = result.species.map(species_mapping)
    result["exact_view"] = result.organ_category.astype(str) + " / " + result.subview.astype(str)

    # Hard leakage assertions for both protocols.
    individual_sets = {
        name: set(result.loc[result.individual_split == name, "individual_id"])
        for name in ("train", "validation", "test")
    }
    species_sets = {
        name: set(result.loc[result.species_split == name, "species"])
        for name in ("train", "validation", "test")
    }
    individual_overlap = {
        "train_validation": sorted(individual_sets["train"] & individual_sets["validation"]),
        "train_test": sorted(individual_sets["train"] & individual_sets["test"]),
        "validation_test": sorted(individual_sets["validation"] & individual_sets["test"]),
    }
    species_overlap = {
        "train_validation": sorted(species_sets["train"] & species_sets["validation"]),
        "train_test": sorted(species_sets["train"] & species_sets["test"]),
        "validation_test": sorted(species_sets["validation"] & species_sets["test"]),
    }
    if any(individual_overlap.values()):
        raise ValueError(f"individual_id leakage: {individual_overlap}")
    if any(species_overlap.values()):
        raise ValueError(f"species leakage: {species_overlap}")

    columns = [
        "image_id", "file", "individual_id", "species", "organ_tag", "organ_category",
        "subview", "exact_view", "individual_split", "species_split", "thumbnail_url", "image_url",
    ]
    result[columns].to_csv(DIRS["splits"] / "master_manifest.csv", index=False)
    result[columns].rename(columns={"individual_split": "split"}).drop(columns="species_split").to_csv(
        DIRS["splits"] / "individual_disjoint.csv", index=False
    )
    result[columns].rename(columns={"species_split": "split"}).drop(columns="individual_split").to_csv(
        DIRS["splits"] / "species_disjoint.csv", index=False
    )
    species_rows = [
        {
            "species": species,
            "split": species_mapping[species],
            "images": int((result.species == species).sum()),
            "individuals": int(result.loc[result.species == species, "individual_id"].nunique()),
        }
        for species in sorted(species_mapping)
    ]
    pd.DataFrame(species_rows).to_csv(DIRS["splits"] / "species_assignments.csv", index=False)

    audit = {
        "seed": SEED,
        "embedding_universe_images": len(result),
        "embedding_universe_species": result.species.nunique(),
        "individual_disjoint": {
            "counts": result.individual_split.value_counts().to_dict(),
            "individual_counts": result.groupby("individual_split").individual_id.nunique().to_dict(),
            "overlap": individual_overlap,
            "passed": not any(individual_overlap.values()),
        },
        "species_disjoint": {
            "counts": result.species_split.value_counts().to_dict(),
            "species_counts": result.groupby("species_split").species.nunique().to_dict(),
            "overlap": species_overlap,
            "passed": not any(species_overlap.values()),
        },
    }
    write_json(DIRS["splits"] / "leakage_assertions.json", audit)
    return result, audit


def top_k_hits(probabilities: np.ndarray, classes: np.ndarray, truths: np.ndarray, k: int) -> np.ndarray:
    k = min(k, probabilities.shape[1])
    top = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]
    return np.array([truth in set(classes[row]) for truth, row in zip(truths, top)], dtype=bool)


def metric_values(truth: np.ndarray, prediction: np.ndarray, top3_hit: np.ndarray) -> dict[str, float]:
    return {
        "top1_accuracy": float(accuracy_score(truth, prediction)),
        "top3_recall": float(np.mean(top3_hit)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
    }


def clustered_bootstrap(predictions: pd.DataFrame, seed: int) -> dict[str, dict[str, float]]:
    groups = sorted(predictions.individual_id.unique())
    rows_by_group = {
        group: predictions.index[predictions.individual_id == group].to_numpy()
        for group in groups
    }
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("top1_accuracy", "top3_recall", "macro_f1", "weighted_f1")}
    for _ in range(BOOTSTRAP_REPLICATES):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([rows_by_group[group] for group in drawn])
        subset = predictions.loc[indices]
        values = metric_values(
            subset.canonical_label.to_numpy(),
            subset.predicted_label.to_numpy(),
            subset.top3_correct.to_numpy(dtype=bool),
        )
        for key, value in values.items():
            samples[key].append(value)
    output = {}
    point = metric_values(
        predictions.canonical_label.to_numpy(),
        predictions.predicted_label.to_numpy(),
        predictions.top3_correct.to_numpy(dtype=bool),
    )
    for key, values in samples.items():
        output[key] = {
            "point": point[key],
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "bootstrap_unit": "individual_id",
            "replicates": BOOTSTRAP_REPLICATES,
        }
    return output


def plot_confusion(truth: np.ndarray, prediction: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    matrix = confusion_matrix(truth, prediction, labels=labels, normalize="true")
    width = max(7.2, 0.72 * len(labels))
    fig, ax = plt.subplots(figsize=(width, width * 0.83))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted canonical label")
    ax.set_ylabel("BioImages canonical label")
    ax.set_title(title)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix[row, col]
            if value >= 0.08:
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.5 else "#1b1b1b")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Recall within true class")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_probe(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    model: str,
    protocol: str,
    label_column: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_column = "individual_split" if protocol == "individual_disjoint" else "species_split"
    train_indices = np.flatnonzero(frame[split_column].to_numpy() == "train")
    val_indices = np.flatnonzero(frame[split_column].to_numpy() == "validation")
    test_indices = np.flatnonzero(frame[split_column].to_numpy() == "test")
    y = frame[label_column].astype(str).to_numpy()
    train_classes = set(y[train_indices])
    val_supported = np.array([index for index in val_indices if y[index] in train_classes], dtype=int)
    unsupported_validation = sorted(set(y[val_indices]) - train_classes)
    unsupported_test = sorted(set(y[test_indices]) - train_classes)
    if unsupported_test:
        raise ValueError(f"{protocol}/{label_column}: test labels absent from train: {unsupported_test}")

    validation_rows = []
    best: tuple[float, float, float] | None = None
    for c_value in C_GRID:
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=SEED,
        )
        classifier.fit(matrix[train_indices], y[train_indices])
        prediction = classifier.predict(matrix[val_supported])
        macro = float(f1_score(y[val_supported], prediction, average="macro", zero_division=0))
        accuracy = float(accuracy_score(y[val_supported], prediction))
        validation_rows.append(
            {
                "model": MODEL_LABELS[model], "protocol": protocol, "task": label_column,
                "C": c_value, "macro_f1": macro, "top1_accuracy": accuracy,
                "validation_images": len(val_supported),
                "unsupported_validation_labels": "; ".join(unsupported_validation),
            }
        )
        candidate = (macro, accuracy, -math.log10(c_value))
        if best is None or candidate > best:
            best = candidate
            best_c = c_value

    fit_indices = np.concatenate([train_indices, val_indices])
    classifier = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=SEED,
    )
    classifier.fit(matrix[fit_indices], y[fit_indices])
    probabilities = classifier.predict_proba(matrix[test_indices])
    classes = classifier.classes_
    order = np.argsort(-probabilities, axis=1)
    predicted = classes[order[:, 0]]
    top3 = classes[order[:, : min(3, len(classes))]]
    truth = y[test_indices]
    top3_hit = np.array([label in row for label, row in zip(truth, top3)], dtype=bool)
    test_frame = frame.iloc[test_indices].reset_index(drop=True)
    prediction_rows = test_frame[
        ["image_id", "file", "individual_id", "species", "organ_tag", "organ_category", "subview", "thumbnail_url", "image_url"]
    ].copy()
    prediction_rows.insert(0, "model", MODEL_LABELS[model])
    prediction_rows.insert(1, "model_key", model)
    prediction_rows.insert(2, "protocol", protocol)
    prediction_rows.insert(3, "task", label_column)
    prediction_rows["canonical_label"] = truth
    prediction_rows["predicted_label"] = predicted
    prediction_rows["confidence"] = probabilities[np.arange(len(probabilities)), order[:, 0]]
    prediction_rows["top3_labels"] = [json.dumps(row.tolist(), ensure_ascii=False) for row in top3]
    prediction_rows["top1_correct"] = predicted == truth
    prediction_rows["top3_correct"] = top3_hit

    values = metric_values(truth, predicted, top3_hit)
    report = classification_report(truth, predicted, labels=sorted(set(truth)), output_dict=True, zero_division=0)
    per_class = pd.DataFrame(
        [
            {
                "model": MODEL_LABELS[model], "model_key": model, "protocol": protocol,
                "task": label_column, "canonical_label": label,
                "precision": report[label]["precision"], "recall": report[label]["recall"],
                "f1": report[label]["f1-score"], "support": int(report[label]["support"]),
            }
            for label in sorted(set(truth))
        ]
    )
    counts = confusion_matrix(truth, predicted, labels=sorted(set(truth)))
    confusion_rows = []
    for i, true_label in enumerate(sorted(set(truth))):
        for j, pred_label in enumerate(sorted(set(truth))):
            confusion_rows.append(
                {
                    "model": MODEL_LABELS[model], "protocol": protocol, "task": label_column,
                    "true_label": true_label, "predicted_label": pred_label, "count": int(counts[i, j]),
                    "row_fraction": float(counts[i, j] / counts[i].sum()) if counts[i].sum() else 0.0,
                }
            )
    pd.DataFrame(confusion_rows).to_csv(
        DIRS["metrics"] / f"confusion_{protocol}_{label_column}_{model}.csv", index=False
    )
    plot_confusion(
        truth,
        predicted,
        sorted(set(truth)),
        f"{MODEL_LABELS[model]} — {protocol.replace('_', ' ')} — {label_column}",
        DIRS["figures"] / f"confusion_{protocol}_{label_column}_{model}.png",
    )

    ci = None
    if protocol == "individual_disjoint":
        ci = clustered_bootstrap(prediction_rows, stable_int(SEED, model, protocol, label_column) % (2**32))
    summary = {
        "model": MODEL_LABELS[model],
        "model_key": model,
        "protocol": protocol,
        "task": label_column,
        "seed": SEED,
        "classifier": "LogisticRegression(class_weight='balanced', solver='lbfgs')",
        "selected_C": best_c,
        "selection_metric": "validation macro F1",
        "refit": "train + validation",
        "train_images": int(len(train_indices)),
        "validation_images": int(len(val_indices)),
        "test_images": int(len(test_indices)),
        "train_individuals": int(frame.iloc[train_indices].individual_id.nunique()),
        "test_individuals": int(frame.iloc[test_indices].individual_id.nunique()),
        "train_species": int(frame.iloc[train_indices].species.nunique()),
        "test_species": int(frame.iloc[test_indices].species.nunique()),
        "classes_fit": classes.tolist(),
        "test_classes": sorted(set(truth)),
        "unsupported_validation_labels_during_selection": unsupported_validation,
        "unsupported_test_labels": unsupported_test,
        **values,
        "individual_id_bootstrap_95_ci": ci,
    }
    return summary, prediction_rows, per_class, pd.DataFrame(validation_rows)


def run_all_probes(
    frame: pd.DataFrame, matrices: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    summaries = []
    per_class_frames = []
    prediction_sets: dict[tuple[str, str, str], pd.DataFrame] = {}
    validation_frames = []
    for task in ("organ_tag", "organ_category"):
        for protocol in ("individual_disjoint", "species_disjoint"):
            for model, matrix in matrices.items():
                print(f"probe: {task} / {protocol} / {model}", flush=True)
                summary, predictions, per_class, validation = run_probe(
                    frame, matrix, model, protocol, task
                )
                summaries.append(summary)
                per_class_frames.append(per_class)
                validation_frames.append(validation)
                prediction_sets[(task, protocol, model)] = predictions
                predictions.to_csv(
                    DIRS["predictions"] / f"{protocol}_{task}_{model}.csv", index=False
                )
                write_json(
                    DIRS["metrics"] / f"{protocol}_{task}_{model}.json", summary
                )

    flattened_summaries = []
    for row in summaries:
        flat = {key: value for key, value in row.items() if key != "individual_id_bootstrap_95_ci"}
        ci = row.get("individual_id_bootstrap_95_ci")
        for metric in ("top1_accuracy", "top3_recall", "macro_f1", "weighted_f1"):
            flat[f"{metric}_ci_lower"] = ci[metric]["lower_95"] if ci else math.nan
            flat[f"{metric}_ci_upper"] = ci[metric]["upper_95"] if ci else math.nan
        flattened_summaries.append(flat)
    summary_frame = pd.DataFrame(flattened_summaries)
    summary_frame.to_csv(DIRS["metrics"] / "benchmark_summary.csv", index=False)
    write_json(DIRS["metrics"] / "benchmark_summary.json", summaries)
    per_class_frame = pd.concat(per_class_frames, ignore_index=True)
    per_class_frame.to_csv(DIRS["metrics"] / "per_class_metrics.csv", index=False)
    pd.concat(validation_frames, ignore_index=True).to_csv(
        DIRS["metrics"] / "validation_grid.csv", index=False
    )
    return summary_frame, per_class_frame, prediction_sets


def run_retrieval(
    frame: pd.DataFrame, matrices: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    neighbor_rows = []
    unrestricted_rows = []
    organ = frame.organ_tag.astype(str).to_numpy()
    species = frame.species.astype(str).to_numpy()
    individuals = frame.individual_id.astype(str).to_numpy()
    for model, matrix in matrices.items():
        print(f"retrieval: {model}", flush=True)
        similarity = matrix @ matrix.T
        np.fill_diagonal(similarity, -np.inf)
        hit_lists = []
        same_species_lists = []
        same_organ_unrestricted_lists = []
        for query_index in range(len(frame)):
            excluded = (individuals == individuals[query_index]) | (species == species[query_index])
            scores = similarity[query_index].copy()
            scores[excluded] = -np.inf
            top10 = np.argpartition(-scores, kth=9)[:10]
            top10 = top10[np.argsort(-scores[top10])]
            hits = organ[top10] == organ[query_index]
            hit_lists.append(hits)
            for rank, neighbor_index in enumerate(top10, start=1):
                neighbor_rows.append(
                    {
                        "model": MODEL_LABELS[model], "model_key": model,
                        "query_image_id": frame.iloc[query_index].image_id,
                        "query_species": species[query_index], "query_individual_id": individuals[query_index],
                        "query_organ_tag": organ[query_index], "rank": rank,
                        "neighbor_image_id": frame.iloc[neighbor_index].image_id,
                        "neighbor_species": species[neighbor_index],
                        "neighbor_individual_id": individuals[neighbor_index],
                        "neighbor_organ_tag": organ[neighbor_index],
                        "organ_match": bool(hits[rank - 1]),
                        "cosine_similarity": float(scores[neighbor_index]),
                    }
                )

            # Secondary diagnostic without the species exclusion, to test identity bias.
            unrestricted_scores = similarity[query_index].copy()
            unrestricted_scores[individuals == individuals[query_index]] = -np.inf
            top_unrestricted = np.argpartition(-unrestricted_scores, kth=9)[:10]
            top_unrestricted = top_unrestricted[np.argsort(-unrestricted_scores[top_unrestricted])]
            same_species_lists.append(species[top_unrestricted] == species[query_index])
            same_organ_unrestricted_lists.append(organ[top_unrestricted] == organ[query_index])

        hits_array = np.asarray(hit_lists, dtype=bool)
        same_species_array = np.asarray(same_species_lists, dtype=bool)
        same_organ_array = np.asarray(same_organ_unrestricted_lists, dtype=bool)
        summary = {
            "model": MODEL_LABELS[model], "model_key": model, "queries": len(frame),
            "exclusions": "same individual_id and same species",
            "organ_p_at_1": float(hits_array[:, :1].mean()),
            "organ_p_at_5": float(hits_array[:, :5].mean()),
            "organ_p_at_10": float(hits_array[:, :10].mean()),
        }
        summaries.append(summary)
        unrestricted_rows.append(
            {
                "model": MODEL_LABELS[model], "model_key": model, "queries": len(frame),
                "exclusions": "same individual_id only",
                "same_species_p_at_1": float(same_species_array[:, :1].mean()),
                "same_species_p_at_5": float(same_species_array[:, :5].mean()),
                "same_species_p_at_10": float(same_species_array[:, :10].mean()),
                "same_organ_p_at_1": float(same_organ_array[:, :1].mean()),
                "same_organ_p_at_5": float(same_organ_array[:, :5].mean()),
                "same_organ_p_at_10": float(same_organ_array[:, :10].mean()),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    neighbors_frame = pd.DataFrame(neighbor_rows)
    unrestricted_frame = pd.DataFrame(unrestricted_rows)
    summary_frame.to_csv(DIRS["metrics"] / "cross_species_retrieval.csv", index=False)
    write_json(DIRS["metrics"] / "cross_species_retrieval.json", summaries)
    neighbors_frame.to_csv(DIRS["metrics"] / "cross_species_neighbors.csv", index=False)
    unrestricted_frame.to_csv(DIRS["metrics"] / "unrestricted_retrieval_diagnostic.csv", index=False)
    return summary_frame, neighbors_frame, unrestricted_frame


def image_card(row: pd.Series, caption: str, css_class: str = "") -> str:
    thumb = html.escape(str(row.thumbnail_url), quote=True)
    full = html.escape(str(row.image_url), quote=True)
    return (
        f'<a class="image-card {css_class}" href="{full}" target="_blank" rel="noopener">'
        f'<img src="{thumb}" loading="lazy" alt="{html.escape(caption, quote=True)}">'
        f'<span>{html.escape(caption)}</span></a>'
    )


def gallery_shell(title: str, intro: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#fff;color:#20241f;line-height:1.45}}
main{{max-width:1180px;margin:0 auto;padding:28px 20px 60px}}h1{{font-size:1.8rem;margin-bottom:.3rem}}p{{color:#4f574d}}
.group{{border-top:1px solid #dfe4dc;padding:24px 0}}.row{{display:flex;gap:10px;overflow-x:auto;padding:8px 0 12px}}
.image-card{{flex:0 0 150px;color:#263324;text-decoration:none;font-size:.78rem}}.image-card img{{width:150px;height:118px;object-fit:cover;background:#eef1ec;display:block;margin-bottom:5px}}
.query img{{outline:4px solid #20241f;outline-offset:-4px}}.match img{{outline:4px solid #42864b;outline-offset:-4px}}.miss img{{outline:4px solid #b44d45;outline-offset:-4px}}
.meta{{font-size:.84rem;color:#5d655a}}nav a{{color:#315f36}}code{{background:#f1f3ef;padding:.1rem .25rem}}</style></head>
<body><main><nav><a href="../index.html">← Main report</a></nav><h1>{html.escape(title)}</h1><p>{html.escape(intro)}</p>{body}</main></body></html>"""


def select_gallery_queries(frame: pd.DataFrame, count: int = 12) -> list[str]:
    test = frame.loc[frame.individual_split == "test"].copy()
    test["stable"] = test.image_id.map(lambda value: stable_int(SEED, "gallery", value))
    chosen = []
    for _, group in test.sort_values("stable").groupby("organ_tag", sort=True):
        chosen.append(group.iloc[0].image_id)
    remainder = test.loc[~test.image_id.isin(chosen)].sort_values("stable")
    chosen.extend(remainder.image_id.head(max(0, count - len(chosen))).tolist())
    return chosen[:count]


def build_retrieval_gallery(frame: pd.DataFrame, neighbors: pd.DataFrame) -> None:
    by_id = frame.set_index("image_id")
    body = []
    for query_id in select_gallery_queries(frame):
        query = by_id.loc[query_id]
        body.append(
            '<section class="group">' + image_card(
                query, f"QUERY · {query.organ_tag} · {query.species} · {query_id}", "query"
            )
        )
        for model in MODEL_LABELS:
            matches = neighbors.loc[
                (neighbors.model_key == model) & (neighbors.query_image_id == query_id)
            ].sort_values("rank").head(5)
            cards = []
            for neighbor in matches.itertuples():
                row = by_id.loc[neighbor.neighbor_image_id]
                css = "match" if neighbor.organ_match else "miss"
                cards.append(
                    image_card(
                        row,
                        f"#{neighbor.rank} · {row.organ_tag} · {row.species} · cos {neighbor.cosine_similarity:.3f}",
                        css,
                    )
                )
            body.append(
                f'<div><strong>{html.escape(MODEL_LABELS[model])}</strong>'
                f'<div class="row">{"".join(cards)}</div></div>'
            )
        body.append("</section>")
    page = gallery_shell(
        "Cross-species nearest neighbors",
        "The same deterministic queries are shown for all representations. Green frames match the query canonical organ_tag; red frames do not. Same-species and same-individual candidates were excluded.",
        "".join(body),
    )
    (DIRS["galleries"] / "nearest_neighbors.html").write_text(page, encoding="utf-8")


def build_error_analysis(
    frame: pd.DataFrame,
    prediction_sets: dict[tuple[str, str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bio = prediction_sets[("organ_tag", "individual_disjoint", "bioclip")].set_index("image_id")
    dino = prediction_sets[("organ_tag", "individual_disjoint", "dinov3")].set_index("image_id")
    eff = prediction_sets[("organ_tag", "individual_disjoint", "efficientnet_b0")].set_index("image_id")
    ids = sorted(set(bio.index) & set(dino.index) & set(eff.index))
    rows = []
    for image_id in ids:
        b, d, e = bio.loc[image_id], dino.loc[image_id], eff.loc[image_id]
        if bool(d.top1_correct) and not bool(b.top1_correct):
            bucket = "DINO correct / BioCLIP wrong"
        elif bool(b.top1_correct) and not bool(d.top1_correct):
            bucket = "BioCLIP correct / DINO wrong"
        elif bool(b.top1_correct) and bool(d.top1_correct):
            bucket = "both correct"
        else:
            bucket = "both wrong"
        rows.append(
            {
                "image_id": image_id, "individual_id": b.individual_id, "species": b.species,
                "canonical_label": b.canonical_label, "bucket": bucket,
                "bioclip_prediction": b.predicted_label, "bioclip_confidence": b.confidence,
                "bioclip_correct": bool(b.top1_correct),
                "dinov3_prediction": d.predicted_label, "dinov3_confidence": d.confidence,
                "dinov3_correct": bool(d.top1_correct),
                "efficientnet_prediction": e.predicted_label, "efficientnet_confidence": e.confidence,
                "efficientnet_correct": bool(e.top1_correct),
                "thumbnail_url": b.thumbnail_url, "image_url": b.image_url,
            }
        )
    paired = pd.DataFrame(rows)
    paired.to_csv(DIRS["metrics"] / "bioclip_dinov3_disagreements.csv", index=False)
    summary = paired.bucket.value_counts().rename_axis("bucket").reset_index(name="images")
    summary["fraction"] = summary.images / len(paired)
    summary.to_csv(DIRS["metrics"] / "error_bucket_summary.csv", index=False)

    body = []
    bucket_order = [
        "DINO correct / BioCLIP wrong", "BioCLIP correct / DINO wrong", "both wrong", "both correct"
    ]
    for bucket in bucket_order:
        subset = paired.loc[paired.bucket == bucket].copy()
        subset["stable"] = subset.image_id.map(lambda value: stable_int(SEED, "error", bucket, value))
        if bucket == "both wrong":
            # Prioritize confident failures, then deterministically break ties.
            subset["error_confidence"] = subset[["bioclip_confidence", "dinov3_confidence"]].max(axis=1)
            subset = subset.sort_values(["error_confidence", "stable"], ascending=[False, True])
        else:
            subset = subset.sort_values("stable")
        subset = subset.head(16)
        cards = []
        for row in subset.itertuples():
            original = pd.Series({"thumbnail_url": row.thumbnail_url, "image_url": row.image_url})
            caption = (
                f"truth {row.canonical_label} · BioCLIP {row.bioclip_prediction} ({row.bioclip_confidence:.2f}) · "
                f"DINO {row.dinov3_prediction} ({row.dinov3_confidence:.2f}) · {row.species}"
            )
            cards.append(image_card(original, caption))
        body.append(
            f'<section class="group"><h2>{html.escape(bucket)} <span class="meta">n={(paired.bucket == bucket).sum()}</span></h2>'
            f'<div class="row">{"".join(cards)}</div></section>'
        )
    page = gallery_shell(
        "Organ-classification disagreements and errors",
        "All predictions come from the individual-disjoint organ_tag benchmark. The both-wrong section is ordered by the larger of the BioCLIP and DINOv3 confidence scores.",
        "".join(body),
    )
    (DIRS["galleries"] / "errors.html").write_text(page, encoding="utf-8")
    return paired, summary


def allocate_stratified_sample(test: pd.DataFrame, target: int) -> pd.DataFrame:
    counts = test.organ_tag.value_counts().sort_index()
    raw = counts / counts.sum() * target
    quota = np.floor(raw).astype(int)
    quota[counts > 0] = np.maximum(quota[counts > 0], 1)
    while quota.sum() < target:
        choices = (raw - quota).sort_values(ascending=False)
        for label in choices.index:
            if quota[label] < counts[label]:
                quota[label] += 1
                break
    while quota.sum() > target:
        choices = (quota - raw).sort_values(ascending=False)
        for label in choices.index:
            if quota[label] > 1:
                quota[label] -= 1
                break
    selected = []
    for label, n_rows in quota.items():
        group = test.loc[test.organ_tag == label].copy()
        group["stable"] = group.image_id.map(lambda value: stable_int(SEED, "ambiguity", value))
        selected.append(group.sort_values("stable").head(int(n_rows)))
    return pd.concat(selected).sort_values(["organ_tag", "stable"]).drop(columns="stable")


def build_ambiguity_audit(frame: pd.DataFrame) -> pd.DataFrame:
    sample = allocate_stratified_sample(frame.loc[frame.individual_split == "test"].copy(), 180)
    audit = pd.DataFrame(
        {
            "image_id": sample.image_id,
            "BioImages canonical label": sample.organ_tag,
            "visible_leaf": "", "visible_twig": "", "visible_bark": "", "visible_flower": "",
            "visible_fruit": "", "visible_cone": "", "visible_seed": "", "visible_whole_plant": "",
            "acceptable_labels": "", "ambiguous": "", "notes": "",
        }
    )
    audit.to_csv(DIRS["audit"] / "multi_label_audit.csv", index=False)
    cards = []
    for row in sample.itertuples():
        caption = f"{row.image_id} · canonical: {row.organ_tag} · {row.species}"
        cards.append(image_card(pd.Series({"thumbnail_url": row.thumbnail_url, "image_url": row.image_url}), caption))
    body = '<div class="row" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));overflow:visible">' + "".join(cards) + "</div>"
    page = gallery_shell(
        "Blank human multi-label ambiguity audit",
        "Deterministic 180-image stratified sample from the individual-disjoint test set. The accompanying CSV contains blank human-only fields; no VLM labels were generated.",
        body,
    )
    (DIRS["audit"] / "gallery.html").write_text(page, encoding="utf-8")
    return audit


def build_support_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = pd.DataFrame(json.loads(METADATA_PATH.read_text(encoding="utf-8")))
    full["exact_view"] = full.organ_category.astype(str) + " / " + full.subview.astype(str)
    support = (
        full.groupby(["organ_category", "subview", "exact_view"], dropna=False)
        .agg(all_1899_images=("image_id", "size"), all_individuals=("individual_id", "nunique"), all_species=("scientific_name", "nunique"))
        .reset_index()
    )
    embedded_counts = frame.exact_view.value_counts()
    individual_test_counts = frame.loc[frame.individual_split == "test", "exact_view"].value_counts()
    species_test_counts = frame.loc[frame.species_split == "test", "exact_view"].value_counts()
    support["embedding_universe_images"] = support.exact_view.map(embedded_counts).fillna(0).astype(int)
    support["individual_test_images"] = support.exact_view.map(individual_test_counts).fillna(0).astype(int)
    support["species_test_images"] = support.exact_view.map(species_test_counts).fillna(0).astype(int)
    support["feasibility"] = np.select(
        [support.all_1899_images >= 50, support.all_1899_images >= 20, support.all_1899_images >= 5],
        ["reasonable", "limited", "exploratory only"],
        default="insufficient",
    )
    support = support.sort_values(["all_1899_images", "exact_view"], ascending=[False, True])
    support.to_csv(DIRS["metrics"] / "exact_view_support.csv", index=False)

    organ_tag_support = (
        full.groupby("organ_tag").agg(all_1899_images=("image_id", "size"), all_individuals=("individual_id", "nunique"), all_species=("scientific_name", "nunique")).reset_index()
    )
    organ_tag_support["embedding_universe_images"] = organ_tag_support.organ_tag.map(frame.organ_tag.value_counts()).fillna(0).astype(int)
    organ_tag_support.to_csv(DIRS["metrics"] / "organ_tag_support.csv", index=False)

    category_support = (
        full.groupby("organ_category").agg(all_1899_images=("image_id", "size"), all_individuals=("individual_id", "nunique"), all_species=("scientific_name", "nunique")).reset_index()
    )
    category_support["embedding_universe_images"] = category_support.organ_category.map(frame.organ_category.value_counts()).fillna(0).astype(int)
    category_support.to_csv(DIRS["metrics"] / "organ_category_support.csv", index=False)
    return support, organ_tag_support, category_support


def table_html(frame: pd.DataFrame, percent_columns: set[str] | None = None, digits: int = 3) -> str:
    percent_columns = percent_columns or set()
    rows = []
    rows.append("<thead><tr>" + "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns) + "</tr></thead>")
    body = []
    for row in frame.itertuples(index=False):
        cells = []
        for column, value in zip(frame.columns, row):
            if pd.isna(value):
                rendered = "—"
            elif column in percent_columns:
                rendered = pct(float(value))
            elif isinstance(value, (float, np.floating)):
                rendered = f"{float(value):.{digits}f}"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("<tbody>" + "".join(body) + "</tbody>")
    return '<div class="table-wrap"><table>' + "".join(rows) + "</table></div>"


def metric_table(summary: pd.DataFrame, protocol: str, task: str = "organ_tag") -> pd.DataFrame:
    subset = summary.loc[(summary.protocol == protocol) & (summary.task == task)].copy()
    subset = subset.set_index("model_key").loc[list(MODEL_LABELS)].reset_index()
    result = subset[["model", "top1_accuracy", "top3_recall", "macro_f1", "weighted_f1", "test_images"]].rename(
        columns={"model": "Model", "top1_accuracy": "Top-1", "top3_recall": "Top-3", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1", "test_images": "Test n"}
    )
    if protocol == "individual_disjoint":
        result.insert(
            2,
            "Top-1 95% CI",
            [f"{100 * lower:.1f}–{100 * upper:.1f}%" for lower, upper in zip(subset.top1_accuracy_ci_lower, subset.top1_accuracy_ci_upper)],
        )
        result.insert(
            5,
            "Macro F1 95% CI",
            [f"{100 * lower:.1f}–{100 * upper:.1f}%" for lower, upper in zip(subset.macro_f1_ci_lower, subset.macro_f1_ci_upper)],
        )
    return result


def build_reports(
    frame: pd.DataFrame,
    leakage: dict[str, object],
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    retrieval: pd.DataFrame,
    unrestricted: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    exact_support: pd.DataFrame,
) -> None:
    individual = metric_table(summary, "individual_disjoint")
    species = metric_table(summary, "species_disjoint")
    category_individual = metric_table(summary, "individual_disjoint", "organ_category")
    category_species = metric_table(summary, "species_disjoint", "organ_category")
    retrieval_table = retrieval[["model", "organ_p_at_1", "organ_p_at_5", "organ_p_at_10"]].rename(
        columns={"model": "Model", "organ_p_at_1": "P@1", "organ_p_at_5": "P@5", "organ_p_at_10": "P@10"}
    )

    species_by_model = summary.loc[(summary.task == "organ_tag") & (summary.protocol == "species_disjoint")].set_index("model_key")
    retrieval_by_model = retrieval.set_index("model_key")
    dino_wins_species = species_by_model.loc["dinov3", "macro_f1"] > species_by_model.loc["bioclip", "macro_f1"]
    dino_wins_retrieval = retrieval_by_model.loc["dinov3", "organ_p_at_5"] > retrieval_by_model.loc["bioclip", "organ_p_at_5"]
    if dino_wins_species and dino_wins_retrieval:
        verdict = (
            "Supported in this benchmark: DINOv3 is stronger than BioCLIP 2.5 on both species-disjoint "
            "canonical organ classification and cross-species organ retrieval, consistent with a more species-invariant morphology representation."
        )
        verdict_short = "The data support the morphology hypothesis."
    elif dino_wins_species or dino_wins_retrieval:
        verdict = (
            "Partially supported: DINOv3 beats BioCLIP 2.5 on one of the two key cross-species tests, but not both; "
            "the evidence is mixed rather than a clean separation between species identity and organ morphology."
        )
        verdict_short = "The data provide mixed support for the morphology hypothesis."
    else:
        verdict = (
            "Not supported in this benchmark: DINOv3 does not beat BioCLIP 2.5 on either species-disjoint canonical organ classification "
            "or cross-species organ retrieval."
        )
        verdict_short = "The data do not support the morphology hypothesis."

    per_organ = per_class.loc[(per_class.task == "organ_tag") & per_class.canonical_label.isin(["leaf", "twig", "bark", "fruit", "flower", "whole plant", "cone", "needle"])].copy()
    per_organ = per_organ[["protocol", "canonical_label", "model", "precision", "recall", "f1", "support"]]
    per_organ = per_organ.sort_values(["protocol", "canonical_label", "model"])

    exact_display = exact_support[["exact_view", "all_1899_images", "all_species", "embedding_universe_images", "individual_test_images", "species_test_images", "feasibility"]].rename(
        columns={"exact_view": "Exact view", "all_1899_images": "All n", "all_species": "Species", "embedding_universe_images": "Embedded n", "individual_test_images": "Individual test n", "species_test_images": "Species test n", "feasibility": "Feasibility"}
    )

    confusion_cards = []
    for protocol in ("individual_disjoint", "species_disjoint"):
        for model in MODEL_LABELS:
            path = f"figures/confusion_{protocol}_organ_tag_{model}.png"
            confusion_cards.append(
                f'<figure><a href="{path}"><img src="{path}" loading="lazy" alt="{html.escape(protocol)} {html.escape(MODEL_LABELS[model])} confusion matrix"></a>'
                f'<figcaption>{html.escape(MODEL_LABELS[model])} · {html.escape(protocol.replace("_", " "))}</figcaption></figure>'
            )

    methods_counts = pd.DataFrame(
        [
            {"Protocol": "Individual-disjoint", **leakage["individual_disjoint"]["counts"], "Held-out unit": "individual_id"},
            {"Protocol": "Species-disjoint", **leakage["species_disjoint"]["counts"], "Held-out unit": "species"},
        ]
    ).rename(columns={"train": "Train", "validation": "Validation", "test": "Test"})

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BioImages Organ & View Benchmark</title><style>
:root{{--ink:#20251f;--muted:#5c6659;--line:#dce3d9;--green:#315f36;--wash:#f5f7f3}}
*{{box-sizing:border-box}}body{{margin:0;background:white;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;line-height:1.5}}
main{{max-width:1180px;margin:0 auto;padding:34px 22px 80px}}header{{border-bottom:2px solid var(--ink);padding-bottom:24px;margin-bottom:32px}}
h1{{font-size:clamp(2rem,5vw,3.6rem);line-height:1.05;margin:.2rem 0 .7rem;letter-spacing:-.035em}}h2{{margin-top:3.2rem;border-top:1px solid var(--line);padding-top:1.2rem}}h3{{margin-top:1.8rem}}
p{{max-width:850px}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:.75rem;color:var(--muted)}}.verdict{{font-size:1.25rem;max-width:1000px;border-left:6px solid var(--green);padding:16px 18px;background:var(--wash)}}
.note{{background:var(--wash);padding:14px 16px;border:1px solid var(--line);max-width:960px}}.results-note{{border-left:4px solid #315f36}}.interpretation{{border-left:4px solid #9b7b35}}
.table-wrap{{overflow-x:auto;margin:12px 0 24px}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{background:var(--wash);font-weight:650;position:sticky;top:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}}figure{{margin:0;border:1px solid var(--line);padding:10px}}figure img{{display:block;width:100%;height:auto}}figcaption{{font-size:.82rem;color:var(--muted);padding-top:7px}}
a{{color:var(--green)}}code{{background:#eff2ed;padding:.1rem .28rem;border-radius:3px}}.downloads a{{display:inline-block;margin:0 14px 8px 0}}
.small{{font-size:.86rem;color:var(--muted)}}
</style></head><body><main>
<header><div class="eyebrow">BioImages · frozen-representation study · seed 42</div><h1>Can a frozen visual representation recognize plant organs across species?</h1>
<p class="verdict"><strong>{html.escape(verdict_short)}</strong> {html.escape(verdict)}</p>
<p class="small">Generated from 1,641 aligned embeddings drawn from a 1,899-image, 85-species BioImages corpus. Labels are BioImages canonical labels—not claims of a unique visual truth.</p></header>

<section id="tested"><h2>1. What we tested</h2><p>BioCLIP 2.5, DINOv3, and ImageNet EfficientNet-B0 were compared as frozen embeddings with the same class-balanced logistic-regression probe, the same C grid, the same seed, and the same splits. No random image split was used.</p>
{table_html(methods_counts)}<div class="note results-note"><strong>Result protocol:</strong> hyperparameters were selected by validation macro F1, then the probe was refit on train + validation. Individual-disjoint confidence intervals resample <code>individual_id</code>, not images.</div></section>

<section id="headline"><h2>2. Headline results</h2><h3>Table 1 · Individual-disjoint organ_tag benchmark</h3>{table_html(individual, {"Top-1", "Top-3", "Macro F1", "Weighted F1"})}
<h3>Table 2 · Species-disjoint organ_tag benchmark</h3>{table_html(species, {"Top-1", "Top-3", "Macro F1", "Weighted F1"})}
<h3>Table 3 · Cross-species retrieval</h3>{table_html(retrieval_table, {"P@1", "P@5", "P@10"})}
<div class="note interpretation"><strong>Interpretation:</strong> {html.escape(verdict)}</div></section>

<section id="individual"><h2>3. Individual-disjoint</h2><p>The same plant individual never appears in more than one split. This is the normal canonical organ-classification benchmark.</p>{table_html(individual, {"Top-1", "Top-3", "Macro F1", "Weighted F1"})}<p class="small">Clustered 95% confidence intervals are stored in each model JSON file under <code>metrics/</code>.</p></section>

<section id="species"><h2>4. Species-disjoint</h2><p>Train, validation, and test species are mutually exclusive. Success therefore requires organ recognition on botanical species not observed during probe fitting.</p>{table_html(species, {"Top-1", "Top-3", "Macro F1", "Weighted F1"})}</section>

<section id="retrieval"><h2>5. Cross-species retrieval</h2><p>For every embedded image, neighbors from the same individual and the same species were removed before computing canonical organ precision.</p>{table_html(retrieval_table, {"P@1", "P@5", "P@10"})}
<p><a href="galleries/nearest_neighbors.html">Open the same-query nearest-neighbor comparison gallery →</a></p>
<div class="note interpretation"><strong>Secondary identity diagnostic:</strong> an unrestricted (same-individual excluded) retrieval table is saved as <code>metrics/unrestricted_retrieval_diagnostic.csv</code>; it measures how often each representation retrieves the same species when allowed.</div></section>

<section id="per-organ"><h2>6. Per-organ performance</h2><p>Table 4 reports canonical-label precision, recall, and F1. Support is the held-out image count for that protocol and is identical across models.</p>{table_html(per_organ, {"precision", "recall", "f1"})}</section>

<section id="confusions"><h2>7. Confusion matrices</h2><p>Rows are normalized within the true BioImages canonical label.</p><div class="grid">{"".join(confusion_cards)}</div></section>

<section id="neighbors"><h2>8. Nearest-neighbor examples</h2><p>The gallery uses a deterministic, organ-diverse subset of the individual-disjoint test queries and shows the top five cross-species neighbors from each representation.</p><p><a href="galleries/nearest_neighbors.html">Open nearest-neighbor gallery →</a></p></section>

<section id="errors"><h2>9. Error analysis</h2>{table_html(bucket_summary.rename(columns={"bucket": "Bucket", "images": "Images", "fraction": "Fraction"}), {"Fraction"})}<p><a href="galleries/errors.html">Open disagreement and high-confidence error gallery →</a></p></section>

<section id="ambiguity"><h2>10. Multi-label ambiguity</h2><p>A photograph may reasonably show several organs while BioImages assigns one canonical label. The benchmark therefore reports <em>BioImages canonical-label accuracy</em>, not unique visual truth.</p><p><a href="ambiguity_audit/gallery.html">Open the blank 180-image human audit gallery →</a> · <a href="ambiguity_audit/multi_label_audit.csv">Download blank audit CSV</a></p><div class="note results-note"><strong>No VLM annotation:</strong> all human judgment fields are intentionally blank.</div></section>

<section id="fine-view"><h2>11. Exact-view feasibility</h2><p>The corpus contains {len(exact_support)} distinct <code>organ_category / subview</code> combinations. Many have too little support for a defensible headline multiclass benchmark, so this release reports feasibility rather than forcing a sparse fine-view score.</p>{table_html(exact_display)}</section>

<section id="category"><h2>12. Original organ_category task</h2><p>This task preserves BioImages' original categories, including the separate whole-tree variants and <code>inflorescence</code>.</p><h3>Individual-disjoint</h3>{table_html(category_individual, {"Top-1", "Top-3", "Macro F1", "Weighted F1"})}<h3>Species-disjoint</h3>{table_html(category_species, {"Top-1", "Top-3", "Macro F1", "Weighted F1"})}</section>

<section id="methods"><h2>13. Methods / reproducibility</h2><p>Run <code>uv run scripts/run_organ_benchmark.py</code> from this directory. The script validates embedding identity/order/normalization, asserts zero leakage, deterministically selects the species split, fits all probes, bootstraps by individual, computes retrieval, and rebuilds every table, figure, gallery, CSV, JSON, and report.</p>
<div class="downloads"><a href="metrics/benchmark_summary.csv">Benchmark CSV</a><a href="metrics/benchmark_summary.json">Benchmark JSON</a><a href="metrics/per_class_metrics.csv">Per-class CSV</a><a href="metrics/cross_species_retrieval.csv">Retrieval CSV</a><a href="split_manifests/leakage_assertions.json">Leakage audit</a><a href="metrics/exact_view_support.csv">Exact-view support</a></div>
<p class="small">Input embeddings: existing strict outputs for BioCLIP 2.5 (1,024-D), DINOv3 (768-D), and EfficientNet-B0 (1,280-D). Dataset images are not copied into this directory or committed; galleries load BioImages source thumbnails.</p></section>
</main></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")

    individual_md = individual.to_markdown(index=False, floatfmt=".3f")
    species_md = species.to_markdown(index=False, floatfmt=".3f")
    retrieval_md = retrieval_table.to_markdown(index=False, floatfmt=".3f")
    report = f"""# Morning report: BioImages organ / view benchmark

## Bottom line

{verdict}

These are **BioImages canonical-label** results. A photograph can contain several reasonable organs even though BioImages supplies one canonical label.

## Headline tables

### Individual-disjoint `organ_tag`

{individual_md}

### Species-disjoint `organ_tag`

{species_md}

### Cross-species organ retrieval

{retrieval_md}

## What was completed

- Reused three aligned, normalized frozen embedding matrices for the same 1,641 images.
- Built deterministic individual-disjoint and species-disjoint manifests and passed hard leakage assertions.
- Used one downstream method everywhere: class-balanced logistic regression, C selected by validation macro F1, then refit on train + validation.
- Evaluated `organ_tag` and original `organ_category` with Top-1, Top-3, macro F1, weighted F1, per-class metrics, support, and confusion matrices.
- Computed individual-disjoint 95% confidence intervals with 2,000 `individual_id` bootstrap replicates.
- Computed cross-species nearest-neighbor organ P@1/P@5/P@10 for every embedded query, excluding same-individual and same-species candidates.
- Built deterministic same-query retrieval galleries, DINO/BioCLIP disagreement and high-confidence error galleries.
- Created a blank, human-only 180-image multi-label audit CSV and gallery. No VLM labels were generated.
- Counted all exact `organ_category / subview` classes across the full 1,899-image metadata corpus and classified their feasibility.

## Important limitations

- The frozen embedding universe contains 1,641 of the 1,899 images: the established strict subset of 63 species with at least three independent individuals. Exact-view support is reported on all 1,899 images, but learned-probe comparisons use the shared 1,641-image universe.
- Rare canonical labels are intrinsically unstable (for example one embedded `stem` image and two `unspecified` images). They remain visible in support files but are not used to oversell fine-grained performance.
- The canonical target is not exhaustive multi-label visual truth. Human audit is the appropriate next step for measuring label ambiguity.

## Next step

Complete the 180-image human ambiguity audit, then report both strict canonical accuracy and an ambiguity-aware acceptable-label score without changing the frozen benchmark split.
"""
    (OUT / "MORNING_REPORT.md").write_text(report, encoding="utf-8")

    readme = """# BioImages organ benchmark

Static, reproducible comparison of BioCLIP 2.5, DINOv3, and EfficientNet-B0 frozen representations for BioImages canonical organ/view recognition.

## Open the report

Open `index.html` directly or publish this directory with GitHub Pages. The galleries use the source BioImages thumbnail URLs; no image dataset is committed.

## Rebuild

```bash
uv run scripts/run_organ_benchmark.py
uv run scripts/verify_outputs.py
```

Inputs are the existing aligned embedding files under `../outputs/*_strict_embeddings/` and `../work/bioimages_metadata.json`. All generated numbers are preserved under `metrics/`, predictions under `predictions/`, and deterministic split definitions under `split_manifests/`.

The primary metric is **BioImages canonical-label accuracy**. It is not presented as unique visual truth; the blank audit in `ambiguity_audit/` exists to capture reasonable multi-label alternatives later by hand.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    write_json(
        DIRS["metrics"] / "report_verdict.json",
        {
            "verdict": verdict,
            "dino_beats_bioclip_species_disjoint_macro_f1": bool(dino_wins_species),
            "dino_beats_bioclip_cross_species_retrieval_p5": bool(dino_wins_retrieval),
        },
    )


def main() -> None:
    make_dirs()
    frame, matrices, _ = load_inputs()
    frame, leakage = build_manifests(frame)
    exact_support, _, _ = build_support_tables(frame)
    summary, per_class, prediction_sets = run_all_probes(frame, matrices)
    retrieval, neighbors, unrestricted = run_retrieval(frame, matrices)
    build_retrieval_gallery(frame, neighbors)
    _, bucket_summary = build_error_analysis(frame, prediction_sets)
    build_ambiguity_audit(frame)
    build_reports(
        frame, leakage, summary, per_class, retrieval, unrestricted,
        bucket_summary, exact_support,
    )
    print(f"Complete: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
