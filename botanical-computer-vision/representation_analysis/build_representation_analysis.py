# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib==3.10.1",
#   "numpy==2.1.1",
#   "pandas==2.2.3",
#   "pillow==11.1.0",
#   "scikit-learn==1.6.1",
#   "scipy==1.15.2",
#   "umap-learn==0.5.7",
# ]
# ///
"""Build an APL-style, label-free representation analysis from saved embeddings."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import os
import random
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bioimages-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/bioimages-numba")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


SEED = 42
K_VALUES = [10, 20, 50]
CENTROID_SHEET_SIZE = 20
NN_COUNT = 10
NN_GALLERY_COUNT = 8
TOP_CONFUSION_PAIRS = 5

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data"
ASSET_DIR = HERE / "assets"
CACHE_DIR = Path("/tmp/bioimages-representation-cache")

EMBEDDING_SOURCES = {
    "bioclip": ROOT / "outputs" / "bioclip25_strict_embeddings" / "embeddings.npz",
    "dinov3": ROOT / "outputs" / "dinov3_strict_embeddings" / "embeddings.npz",
}
MANIFEST = ROOT / "outputs" / "dinov3_strict_embeddings" / "split_manifest.csv"
UNIFIED = ROOT / "error_analysis" / "data" / "unified_predictions_image.csv"
CONFUSIONS = ROOT / "error_analysis" / "taxonomy_analysis" / "within_genus_confusions.csv"
METADATA = ROOT / "app" / "dashboard-data.json"

REPRESENTATION_LABELS = {"bioclip": "BioCLIP", "dinov3": "DINOv3"}
BASELINES = ["bioclip", "dinov3", "gpt_5_4", "gemini_flash_lite"]
BASELINE_LABELS = {
    "bioclip": "BioCLIP",
    "dinov3": "DINOv3",
    "gpt_5_4": "GPT-5.4",
    "gemini_flash_lite": "Gemini",
}

ImageFile.LOAD_TRUNCATED_IMAGES = True


def safe_id(image_id: str) -> str:
    return image_id.replace("/", "__").replace(" ", "_")


def distribution(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def top_distribution(raw: str, limit: int = 4) -> str:
    values = json.loads(raw)
    return ", ".join(f"{key} {value}" for key, value in list(values.items())[:limit])


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.fillna(1.0).to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    for index in range(len(ranked) - 1, -1, -1):
        running = min(running, ranked[index] * len(ranked) / (index + 1))
        adjusted[index] = running
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=p_values.index)


def load_inputs() -> tuple[
    dict[str, np.ndarray],
    list[str],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict],
]:
    manifest = pd.read_csv(MANIFEST)
    unified = pd.read_csv(UNIFIED)
    metadata_records = json.loads(METADATA.read_text(encoding="utf-8"))["images"]
    metadata = {row["id"]: row for row in metadata_records}

    assert len(manifest) == 1641 and manifest.image_id.nunique() == 1641
    assert len(unified) == 211 and unified.image_id.nunique() == 211
    assert set(manifest.image_id).issubset(metadata)

    embeddings: dict[str, np.ndarray] = {}
    canonical_ids: list[str] | None = None
    for key, source in EMBEDDING_SOURCES.items():
        with np.load(source, allow_pickle=False) as payload:
            ids = payload["image_ids"].astype(str).tolist()
            matrix = payload["embeddings"].astype(np.float32)
            splits = payload["splits"].astype(str)
            assert matrix.shape[0] == 1641
            assert int((splits == "test").sum()) == 211
            assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=2e-5)
        if canonical_ids is None:
            canonical_ids = ids
        else:
            assert ids == canonical_ids
        embeddings[key] = matrix

    assert canonical_ids is not None
    assert set(canonical_ids) == set(manifest.image_id)
    manifest = manifest.set_index("image_id").loc[canonical_ids].reset_index()
    assert set(unified.image_id) == set(manifest.loc[manifest.split == "test", "image_id"])

    return embeddings, canonical_ids, manifest, unified, metadata


def fit_clusters(
    embeddings: dict[str, np.ndarray],
    image_ids: list[str],
    manifest: pd.DataFrame,
    unified: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int, int], list[str]], pd.DataFrame]:
    test = unified.set_index("image_id")
    global_bioclip_errors = int((~test.bioclip_correct.astype(bool)).sum())
    global_all_wrong = int((test.correct_count == 0).sum())
    assignments = []
    summaries = []
    representatives: dict[tuple[str, int, int], list[str]] = {}
    cluster_labels: dict[tuple[str, int], np.ndarray] = {}

    for representation, matrix in embeddings.items():
        for k in K_VALUES:
            model = KMeans(
                n_clusters=k,
                random_state=SEED,
                n_init=20,
                max_iter=500,
                algorithm="lloyd",
            )
            labels = model.fit_predict(matrix)
            distances = model.transform(matrix)
            cluster_labels[(representation, k)] = labels

            for index, (image_id, cluster_id) in enumerate(zip(image_ids, labels)):
                assignments.append(
                    {
                        "representation": representation,
                        "k": k,
                        "image_id": image_id,
                        "split": manifest.iloc[index].split,
                        "cluster_id": int(cluster_id),
                        "distance_to_centroid": float(distances[index, cluster_id]),
                    }
                )

            for cluster_id in range(k):
                member_indices = np.flatnonzero(labels == cluster_id)
                ordered = member_indices[np.argsort(distances[member_indices, cluster_id])]
                representatives[(representation, k, cluster_id)] = [
                    image_ids[index] for index in ordered[:CENTROID_SHEET_SIZE]
                ]
                members = manifest.iloc[member_indices]
                species_counts = members.ground_truth_species.value_counts()
                organ_counts = members.organ_category.value_counts()
                member_ids = set(members.image_id)
                test_ids = sorted(member_ids & set(test.index))
                test_members = test.loc[test_ids] if test_ids else pd.DataFrame()

                row = {
                    "representation": representation,
                    "k": k,
                    "cluster_id": cluster_id,
                    "size_all": len(members),
                    "size_test": len(test_ids),
                    "species_count": int(members.ground_truth_species.nunique()),
                    "dominant_species": species_counts.index[0],
                    "dominant_species_count": int(species_counts.iloc[0]),
                    "species_purity": float(species_counts.iloc[0] / len(members)),
                    "species_distribution": json.dumps(distribution(members.ground_truth_species), ensure_ascii=False),
                    "organ_count": int(members.organ_category.nunique()),
                    "dominant_organ": organ_counts.index[0],
                    "dominant_organ_count": int(organ_counts.iloc[0]),
                    "organ_purity": float(organ_counts.iloc[0] / len(members)),
                    "organ_distribution": json.dumps(distribution(members.organ_category), ensure_ascii=False),
                }
                for baseline in BASELINES:
                    if test_ids:
                        accuracy = float(test_members[f"{baseline}_correct"].astype(bool).mean())
                        correct = int(test_members[f"{baseline}_correct"].astype(bool).sum())
                    else:
                        accuracy = math.nan
                        correct = 0
                    row[f"{baseline}_correct"] = correct
                    row[f"{baseline}_accuracy"] = accuracy
                    row[f"{baseline}_error_rate"] = 1.0 - accuracy if test_ids else math.nan

                if test_ids:
                    bio_errors = int((~test_members.bioclip_correct.astype(bool)).sum())
                    all_wrong = int((test_members.correct_count == 0).sum())
                    bio_rate = bio_errors / len(test_ids)
                    all_wrong_rate = all_wrong / len(test_ids)
                    bio_p = float(
                        hypergeom.sf(bio_errors - 1, len(test), global_bioclip_errors, len(test_ids))
                    )
                    all_wrong_p = float(
                        hypergeom.sf(all_wrong - 1, len(test), global_all_wrong, len(test_ids))
                    )
                else:
                    bio_errors = all_wrong = 0
                    bio_rate = all_wrong_rate = math.nan
                    bio_p = all_wrong_p = 1.0
                row.update(
                    {
                        "bioclip_error_count": bio_errors,
                        "bioclip_error_rate": bio_rate,
                        "bioclip_error_enrichment": bio_rate / (global_bioclip_errors / len(test)) if test_ids else math.nan,
                        "bioclip_error_p": bio_p,
                        "all_four_wrong_count": all_wrong,
                        "all_four_wrong_rate": all_wrong_rate,
                        "all_four_wrong_enrichment": all_wrong_rate / (global_all_wrong / len(test)) if test_ids else math.nan,
                        "all_four_wrong_p": all_wrong_p,
                    }
                )
                summaries.append(row)

    cluster_summary = pd.DataFrame(summaries)
    for (_, _), indices in cluster_summary.groupby(["representation", "k"]).groups.items():
        cluster_summary.loc[indices, "bioclip_error_q"] = bh_adjust(
            cluster_summary.loc[indices, "bioclip_error_p"]
        )
        cluster_summary.loc[indices, "all_four_wrong_q"] = bh_adjust(
            cluster_summary.loc[indices, "all_four_wrong_p"]
        )
    cluster_summary["fdr_error_enriched"] = (
        (cluster_summary.size_test >= 3)
        & (
            ((cluster_summary.bioclip_error_q <= 0.10) & (cluster_summary.bioclip_error_enrichment > 1))
            | ((cluster_summary.all_four_wrong_q <= 0.10) & (cluster_summary.all_four_wrong_enrichment > 1))
        )
    )
    cluster_summary["descriptive_error_enriched"] = (
        (cluster_summary.size_test >= 5)
        & (
            (cluster_summary.bioclip_error_enrichment >= 1.5)
            | (cluster_summary.all_four_wrong_enrichment >= 2.0)
        )
    )

    structure_rows = []
    for representation in embeddings:
        for k in K_VALUES:
            labels = cluster_labels[(representation, k)]
            subset = cluster_summary.loc[
                (cluster_summary.representation == representation) & (cluster_summary.k == k)
            ]
            structure_rows.append(
                {
                    "representation": representation,
                    "k": k,
                    "species_nmi": float(normalized_mutual_info_score(manifest.ground_truth_species, labels)),
                    "organ_nmi": float(normalized_mutual_info_score(manifest.organ_category, labels)),
                    "weighted_species_purity": float((subset.species_purity * subset.size_all).sum() / len(manifest)),
                    "weighted_organ_purity": float((subset.organ_purity * subset.size_all).sum() / len(manifest)),
                    "cluster_size_min": int(subset.size_all.min()),
                    "cluster_size_median": float(subset.size_all.median()),
                    "cluster_size_max": int(subset.size_all.max()),
                }
            )
    structure = pd.DataFrame(structure_rows)
    for k in K_VALUES:
        ari = adjusted_rand_score(cluster_labels[("bioclip", k)], cluster_labels[("dinov3", k)])
        structure.loc[structure.k == k, "bioclip_dinov3_ari"] = ari

    return pd.DataFrame(assignments), cluster_summary, representatives, structure


def nearest_neighbors(
    embeddings: dict[str, np.ndarray],
    image_ids: list[str],
    manifest: pd.DataFrame,
    unified: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], list[tuple[str, float]]], dict[str, float]]:
    id_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    meta = manifest.set_index("image_id")
    test_ids = unified.image_id.tolist()
    rows = []
    gallery_neighbors: dict[tuple[str, str], list[tuple[str, float]]] = {}
    top10_sets: dict[tuple[str, str], set[str]] = {}

    for representation, matrix in embeddings.items():
        for query_id in test_ids:
            query_index = id_to_index[query_id]
            similarities = matrix @ matrix[query_index]
            similarities[query_index] = -np.inf
            query_observation = meta.loc[query_id, "individual_id"]

            for scope in ["all", "cross_observation"]:
                scoped = similarities.copy()
                if scope == "cross_observation":
                    same_observation = manifest.individual_id.to_numpy() == query_observation
                    scoped[same_observation] = -np.inf
                candidates = np.argpartition(-scoped, NN_COUNT)[:NN_COUNT]
                candidates = candidates[np.argsort(-scoped[candidates])]
                neighbor_ids = [image_ids[index] for index in candidates]
                if scope == "cross_observation":
                    top10_sets[(representation, query_id)] = set(neighbor_ids)
                    gallery_neighbors[(representation, query_id)] = [
                        (image_ids[index], float(scoped[index]))
                        for index in candidates[:NN_GALLERY_COUNT]
                    ]
                for rank, index in enumerate(candidates, start=1):
                    neighbor_id = image_ids[index]
                    rows.append(
                        {
                            "representation": representation,
                            "query_image_id": query_id,
                            "query_species": meta.loc[query_id, "ground_truth_species"],
                            "query_organ": meta.loc[query_id, "organ_category"],
                            "scope": scope,
                            "rank": rank,
                            "neighbor_image_id": neighbor_id,
                            "neighbor_species": meta.loc[neighbor_id, "ground_truth_species"],
                            "neighbor_organ": meta.loc[neighbor_id, "organ_category"],
                            "neighbor_split": meta.loc[neighbor_id, "split"],
                            "same_species": meta.loc[neighbor_id, "ground_truth_species"] == meta.loc[query_id, "ground_truth_species"],
                            "same_genus": str(meta.loc[neighbor_id, "ground_truth_species"]).split()[0] == str(meta.loc[query_id, "ground_truth_species"]).split()[0],
                            "same_observation": meta.loc[neighbor_id, "individual_id"] == query_observation,
                            "cosine_similarity": float(scoped[index]),
                        }
                    )

    overlaps = []
    for query_id in test_ids:
        first = top10_sets[("bioclip", query_id)]
        second = top10_sets[("dinov3", query_id)]
        overlaps.append(len(first & second) / NN_COUNT)
    comparison = {
        "cross_observation_neighbor_overlap_at_10_mean": float(np.mean(overlaps)),
        "cross_observation_neighbor_overlap_at_10_median": float(np.median(overlaps)),
    }
    return pd.DataFrame(rows), gallery_neighbors, comparison


def mean_off_diagonal_similarity(matrix: np.ndarray, observations: np.ndarray) -> float:
    if len(matrix) < 2:
        return math.nan
    similarities = matrix @ matrix.T
    mask = ~np.eye(len(matrix), dtype=bool)
    mask &= observations[:, None] != observations[None, :]
    values = similarities[mask]
    return float(values.mean()) if len(values) else math.nan


def confusion_pair_analysis(
    embeddings: dict[str, np.ndarray],
    image_ids: list[str],
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str]], dict[tuple[str, str, str], list[str]]]:
    confusion = pd.read_csv(CONFUSIONS).head(TOP_CONFUSION_PAIRS)
    pairs = list(zip(confusion.ground_truth_species, confusion.predicted_species))
    pair_occurrences = {
        (row.ground_truth_species, row.predicted_species): int(row.total_occurrences)
        for row in confusion.itertuples(index=False)
    }
    labels = manifest.ground_truth_species.to_numpy()
    observations = manifest.individual_id.to_numpy()
    id_array = np.asarray(image_ids)
    rows = []
    representatives: dict[tuple[str, str, str], list[str]] = {}

    for representation, matrix in embeddings.items():
        similarity_all = matrix @ matrix.T
        np.fill_diagonal(similarity_all, -np.inf)
        for species_a, species_b in pairs:
            index_a = np.flatnonzero(labels == species_a)
            index_b = np.flatnonzero(labels == species_b)
            matrix_a = matrix[index_a]
            matrix_b = matrix[index_b]
            centroid_a = matrix_a.mean(axis=0)
            centroid_b = matrix_b.mean(axis=0)
            centroid_a /= np.linalg.norm(centroid_a)
            centroid_b /= np.linalg.norm(centroid_b)

            for species, indices, centroid in [
                (species_a, index_a, centroid_a),
                (species_b, index_b, centroid_b),
            ]:
                scores = matrix[indices] @ centroid
                order = indices[np.argsort(-scores)[:12]]
                representatives[(representation, species_a + "→" + species_b, species)] = id_array[order].tolist()

            pair_indices = np.concatenate([index_a, index_b])
            same_species = other_pair = other_species = 0
            for index in pair_indices:
                scores = similarity_all[index].copy()
                scores[observations == observations[index]] = -np.inf
                neighbor = int(np.argmax(scores))
                if labels[neighbor] == labels[index]:
                    same_species += 1
                elif labels[neighbor] in {species_a, species_b}:
                    other_pair += 1
                else:
                    other_species += 1

            cross = matrix_a @ matrix_b.T
            rows.append(
                {
                    "representation": representation,
                    "species_a": species_a,
                    "species_b": species_b,
                    "source_confusion_occurrences": pair_occurrences[(species_a, species_b)],
                    "species_a_images": len(index_a),
                    "species_b_images": len(index_b),
                    "centroid_cosine_similarity": float(centroid_a @ centroid_b),
                    "species_a_within_similarity_cross_observation": mean_off_diagonal_similarity(matrix_a, observations[index_a]),
                    "species_b_within_similarity_cross_observation": mean_off_diagonal_similarity(matrix_b, observations[index_b]),
                    "cross_species_mean_similarity": float(cross.mean()),
                    "cross_observation_nn_same_species_rate": same_species / len(pair_indices),
                    "cross_observation_nn_other_pair_species_rate": other_pair / len(pair_indices),
                    "cross_observation_nn_other_species_rate": other_species / len(pair_indices),
                }
            )
    return pd.DataFrame(rows), pairs, representatives


def download_one(image_id: str, url: str) -> tuple[str, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    extension = ".jpg"
    destination = CACHE_DIR / f"{safe_id(image_id)}{extension}"
    if destination.exists() and destination.stat().st_size > 0:
        try:
            with Image.open(destination) as image:
                image.verify()
            return image_id, "cached"
        except Exception:
            destination.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "BioImages representation analysis/1.0"})
    last_error = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            destination.write_bytes(payload)
            return image_id, "downloaded"
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(2**attempt)
    return image_id, f"failed: {last_error}"


def ensure_images(image_ids: set[str], metadata: dict[str, dict]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(download_one, image_id, metadata[image_id]["image"]): image_id
            for image_id in sorted(image_ids)
        }
        for future in as_completed(futures):
            image_id, status = future.result()
            statuses[image_id] = status
    return statuses


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def load_tile(image_id: str, size: tuple[int, int]) -> Image.Image:
    path = CACHE_DIR / f"{safe_id(image_id)}.jpg"
    canvas = Image.new("RGB", size, "#f1f3f1")
    if not path.exists():
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "image unavailable", fill="#7a817c", font=font(13))
        return canvas
    try:
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            contained = ImageOps.contain(source, size, method=Image.Resampling.LANCZOS)
        canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    except Exception:
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "decode error", fill="#9a3b34", font=font(13))
    return canvas


def make_sheet(
    image_ids: list[str],
    output: Path,
    title: str,
    labels: list[str] | None = None,
    columns: int = 5,
    tile_size: tuple[int, int] = (210, 155),
) -> None:
    labels = labels or [image_id for image_id in image_ids]
    label_height = 36
    header_height = 58
    rows = math.ceil(len(image_ids) / columns)
    width = columns * tile_size[0]
    height = header_height + rows * (tile_size[1] + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 15), title, fill="#17221d", font=font(22, bold=True))
    for index, (image_id, label) in enumerate(zip(image_ids, labels)):
        row, column = divmod(index, columns)
        x = column * tile_size[0]
        y = header_height + row * (tile_size[1] + label_height)
        tile = load_tile(image_id, tile_size)
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + tile_size[0] - 1, y + tile_size[1] + label_height - 1), outline="#d7ddd8")
        draw.text((x + 6, y + tile_size[1] + 5), label[:42], fill="#4f5952", font=font(12))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=86, optimize=True)


def create_visual_assets(
    representatives: dict[tuple[str, int, int], list[str]],
    gallery_neighbors: dict[tuple[str, str], list[tuple[str, float]]],
    pair_representatives: dict[tuple[str, str, str], list[str]],
    pairs: list[tuple[str, str]],
    manifest: pd.DataFrame,
    unified: pd.DataFrame,
    metadata: dict[str, dict],
    embeddings: dict[str, np.ndarray],
    image_ids: list[str],
) -> dict[str, str]:
    manifest_by_id = manifest.set_index("image_id")
    all_wrong_ids = unified.loc[unified.correct_count == 0, "image_id"].tolist()
    needed: set[str] = set()
    for ids in representatives.values():
        needed.update(ids)
    for representation in embeddings:
        for query_id in all_wrong_ids:
            needed.add(query_id)
            needed.update(image_id for image_id, _ in gallery_neighbors[(representation, query_id)])
    for ids in pair_representatives.values():
        needed.update(ids)
    statuses = ensure_images(needed, metadata)

    for (representation, k, cluster_id), ids in representatives.items():
        make_sheet(
            ids,
            ASSET_DIR / "cluster_sheets" / representation / f"k{k}" / f"cluster_{cluster_id:02d}.jpg",
            f"{REPRESENTATION_LABELS[representation]} · k={k} · cluster {cluster_id:02d} · centroid-nearest",
            labels=[image_id for image_id in ids],
        )

    for representation in embeddings:
        for query_id in all_wrong_ids:
            neighbors = gallery_neighbors[(representation, query_id)]
            ids = [query_id] + [image_id for image_id, _ in neighbors]
            labels = [
                f"QUERY · {manifest_by_id.loc[query_id, 'ground_truth_species']} · {query_id}"
            ] + [
                f"NN {rank} · {manifest_by_id.loc[image_id, 'ground_truth_species']} · {similarity:.3f}"
                for rank, (image_id, similarity) in enumerate(neighbors, start=1)
            ]
            make_sheet(
                ids,
                ASSET_DIR / "nn_galleries" / representation / f"{safe_id(query_id)}.jpg",
                f"{REPRESENTATION_LABELS[representation]} · all-four-wrong query · cross-observation neighbors",
                labels=labels,
                columns=3,
                tile_size=(300, 220),
            )

    for representation in embeddings:
        for species_a, species_b in pairs:
            pair_key = species_a + "→" + species_b
            ids_a = pair_representatives[(representation, pair_key, species_a)]
            ids_b = pair_representatives[(representation, pair_key, species_b)]
            ids = ids_a + ids_b
            labels = [f"{species_a} · {image_id}" for image_id in ids_a] + [
                f"{species_b} · {image_id}" for image_id in ids_b
            ]
            make_sheet(
                ids,
                ASSET_DIR / "pair_sheets" / representation / f"{safe_id(species_a)}--{safe_id(species_b)}.jpg",
                f"{REPRESENTATION_LABELS[representation]} · centroid-nearest examples · {species_a} vs {species_b}",
                labels=labels,
                columns=6,
                tile_size=(185, 140),
            )

    create_projection_plots(embeddings, image_ids, manifest, pairs)
    return statuses


def create_projection_plots(
    embeddings: dict[str, np.ndarray],
    image_ids: list[str],
    manifest: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> None:
    labels = manifest.ground_truth_species.to_numpy()
    plot_dir = ASSET_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for representation, matrix in embeddings.items():
        pca_coords = PCA(n_components=2, random_state=SEED).fit_transform(matrix)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=20,
            min_dist=0.15,
            metric="cosine",
            random_state=SEED,
            transform_seed=SEED,
        )
        umap_coords = reducer.fit_transform(matrix)
        np.savez_compressed(
            DATA_DIR / f"{representation}_projection_coordinates.npz",
            image_ids=np.asarray(image_ids),
            pca=pca_coords.astype(np.float32),
            umap=umap_coords.astype(np.float32),
        )

        figure, axes = plt.subplots(len(pairs), 2, figsize=(13, 4.1 * len(pairs)))
        for row, (species_a, species_b) in enumerate(pairs):
            for column, (name, coords) in enumerate([("PCA", pca_coords), ("UMAP", umap_coords)]):
                axis = axes[row, column]
                axis.scatter(coords[:, 0], coords[:, 1], s=5, c="#d9dedb", alpha=0.35, linewidths=0)
                for species, color, marker in [
                    (species_a, "#28664a", "o"),
                    (species_b, "#bd5b43", "^"),
                ]:
                    mask = labels == species
                    axis.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        s=34,
                        c=color,
                        marker=marker,
                        alpha=0.86,
                        linewidths=0.4,
                        edgecolors="white",
                        label=species,
                    )
                axis.set_title(f"{name} · {species_a} vs {species_b}", fontsize=10)
                axis.set_xticks([])
                axis.set_yticks([])
                axis.spines[:].set_visible(False)
                axis.legend(frameon=False, fontsize=8, loc="best")
        figure.suptitle(
            f"{REPRESENTATION_LABELS[representation]} global projections · highlighted confusion pairs",
            fontsize=15,
            y=1.002,
        )
        figure.tight_layout()
        figure.savefig(plot_dir / f"{representation}_confusion_pairs.png", dpi=170, bbox_inches="tight")
        plt.close(figure)


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1%}"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def esc(value: object) -> str:
    return html.escape(str(value))


def cluster_card(row: pd.Series) -> str:
    representation = row.representation
    k = int(row.k)
    cluster_id = int(row.cluster_id)
    accuracies = "".join(
        f"<span><b>{BASELINE_LABELS[baseline]}</b> {pct(row[f'{baseline}_accuracy'])}</span>"
        for baseline in BASELINES
    )
    flags = []
    if bool(row.fdr_error_enriched):
        flags.append("FDR-enriched")
    elif bool(row.descriptive_error_enriched):
        flags.append("descriptively enriched")
    flag_html = f"<span class='flag'>{' · '.join(flags)}</span>" if flags else ""
    return f"""
      <article class="cluster-card">
        <img loading="lazy" src="assets/cluster_sheets/{representation}/k{k}/cluster_{cluster_id:02d}.jpg" alt="Centroid-nearest images for {representation} k={k} cluster {cluster_id}">
        <div class="cluster-copy">
          <h4>Cluster {cluster_id:02d} {flag_html}</h4>
          <p>{int(row.size_all)} images · {int(row.size_test)} test · species purity {pct(row.species_purity)} · organ purity {pct(row.organ_purity)}</p>
          <p><b>Species</b> {esc(top_distribution(row.species_distribution))}</p>
          <p><b>Organs</b> {esc(top_distribution(row.organ_distribution))}</p>
          <div class="metric-line">{accuracies}</div>
          <p class="small">BioCLIP error enrichment {num(row.bioclip_error_enrichment)}× · all-four-wrong enrichment {num(row.all_four_wrong_enrichment)}×</p>
        </div>
      </article>"""


def build_report(
    cluster_summary: pd.DataFrame,
    structure: pd.DataFrame,
    neighbor_comparison: dict[str, float],
    confusion_metrics: pd.DataFrame,
    pairs: list[tuple[str, str]],
    unified: pd.DataFrame,
    download_statuses: dict[str, str],
) -> None:
    structure_rows = []
    for row in structure.itertuples(index=False):
        structure_rows.append(
            f"<tr><td>{REPRESENTATION_LABELS[row.representation]}</td><td>{row.k}</td>"
            f"<td>{row.species_nmi:.3f}</td><td>{row.organ_nmi:.3f}</td>"
            f"<td>{pct(row.weighted_species_purity)}</td><td>{pct(row.weighted_organ_purity)}</td>"
            f"<td>{row.cluster_size_min}/{row.cluster_size_median:.0f}/{row.cluster_size_max}</td>"
            f"<td>{row.bioclip_dinov3_ari:.3f}</td></tr>"
        )

    enriched_sections = []
    for representation in ["bioclip", "dinov3"]:
        for k in K_VALUES:
            subset = cluster_summary.loc[
                (cluster_summary.representation == representation)
                & (cluster_summary.k == k)
                & (cluster_summary.size_test > 0)
            ].copy()
            subset["priority"] = subset[["bioclip_error_enrichment", "all_four_wrong_enrichment"]].max(axis=1)
            selected = subset.sort_values(["fdr_error_enriched", "descriptive_error_enriched", "priority"], ascending=False).head(4)
            enriched_sections.append(
                f"<h3>{REPRESENTATION_LABELS[representation]} · k={k}</h3>"
                + "".join(cluster_card(row) for _, row in selected.iterrows())
            )

    atlas_sections = []
    for representation in ["bioclip", "dinov3"]:
        for k in K_VALUES:
            subset = cluster_summary.loc[
                (cluster_summary.representation == representation) & (cluster_summary.k == k)
            ].sort_values("cluster_id")
            atlas_sections.append(
                f"<details><summary>{REPRESENTATION_LABELS[representation]} · k={k} · {len(subset)} clusters</summary>"
                + "<div class='cluster-grid'>"
                + "".join(cluster_card(row) for _, row in subset.iterrows())
                + "</div></details>"
            )

    all_wrong = unified.loc[unified.correct_count == 0].sort_values(["ground_truth_species", "image_id"])
    nn_rows = []
    for row in all_wrong.itertuples(index=False):
        nn_rows.append(
            f"""
            <article class="nn-case">
              <header><h3><i>{esc(row.ground_truth_species)}</i></h3><p>{esc(row.image_id)} · {esc(row.organ_category)}</p></header>
              <div class="two-up">
                <figure><img loading="lazy" src="assets/nn_galleries/bioclip/{safe_id(row.image_id)}.jpg" alt="BioCLIP nearest neighbors for {esc(row.image_id)}"><figcaption>BioCLIP</figcaption></figure>
                <figure><img loading="lazy" src="assets/nn_galleries/dinov3/{safe_id(row.image_id)}.jpg" alt="DINOv3 nearest neighbors for {esc(row.image_id)}"><figcaption>DINOv3</figcaption></figure>
              </div>
            </article>"""
        )

    pair_sections = []
    for species_a, species_b in pairs:
        rows = confusion_metrics.loc[
            (confusion_metrics.species_a == species_a) & (confusion_metrics.species_b == species_b)
        ].set_index("representation")
        metric_rows = []
        for representation in ["bioclip", "dinov3"]:
            row = rows.loc[representation]
            metric_rows.append(
                f"<tr><td>{REPRESENTATION_LABELS[representation]}</td>"
                f"<td>{row.species_a_images}/{row.species_b_images}</td>"
                f"<td>{row.centroid_cosine_similarity:.3f}</td>"
                f"<td>{row.species_a_within_similarity_cross_observation:.3f}</td>"
                f"<td>{row.species_b_within_similarity_cross_observation:.3f}</td>"
                f"<td>{row.cross_species_mean_similarity:.3f}</td>"
                f"<td>{pct(row.cross_observation_nn_same_species_rate)}</td>"
                f"<td>{pct(row.cross_observation_nn_other_pair_species_rate)}</td></tr>"
            )
        pair_sections.append(
            f"""
            <section class="pair-block">
              <h3><i>{esc(species_a)}</i> → <i>{esc(species_b)}</i></h3>
              <p class="small">Repeated confusion count across the four baselines: {int(rows.iloc[0].source_confusion_occurrences)} model-prediction occurrences.</p>
              <div class="table-wrap"><table><thead><tr><th>Space</th><th>Images A/B</th><th>Centroid cosine</th><th>Within A</th><th>Within B</th><th>Cross-species</th><th>NN same species</th><th>NN other pair species</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
              <div class="two-up">
                <figure><img loading="lazy" src="assets/pair_sheets/bioclip/{safe_id(species_a)}--{safe_id(species_b)}.jpg" alt="BioCLIP representative images for confusion pair"><figcaption>BioCLIP centroid-nearest examples</figcaption></figure>
                <figure><img loading="lazy" src="assets/pair_sheets/dinov3/{safe_id(species_a)}--{safe_id(species_b)}.jpg" alt="DINOv3 representative images for confusion pair"><figcaption>DINOv3 centroid-nearest examples</figcaption></figure>
              </div>
            </section>"""
        )

    download_failures = sum(status.startswith("failed") for status in download_statuses.values())
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BioImages representation analysis</title>
  <style>
    :root {{ --ink:#17221d; --muted:#667069; --line:#d8ded9; --soft:#f3f5f3; --paper:#fff; --accent:#315f4a; --warn:#8c4b2e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.45; }}
    main {{ width:min(1540px,calc(100% - 40px)); margin:0 auto; padding:48px 0 80px; }}
    h1,h2,h3,h4 {{ font-family:Georgia,"Times New Roman",serif; font-weight:400; }}
    h1 {{ font-size:clamp(2.2rem,5vw,4.6rem); line-height:1; margin:8px 0 16px; }}
    h2 {{ font-size:clamp(1.7rem,3vw,2.6rem); margin:0 0 12px; }}
    h3 {{ margin:26px 0 12px; }} h4 {{ margin:0; font-size:1.15rem; }}
    .eyebrow,.small,figcaption {{ color:var(--muted); font-size:.82rem; }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.11em; font-weight:650; }}
    .page-header {{ max-width:940px; padding-bottom:30px; border-bottom:1px solid var(--line); }}
    .protocol {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin-top:24px; }}
    .protocol div {{ background:var(--paper); padding:14px; }} .protocol b {{ display:block; font-size:1.25rem; }}
    section.major {{ padding-top:56px; }}
    .note {{ border-left:3px solid var(--accent); padding:10px 14px; color:var(--muted); background:var(--soft); max-width:960px; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:.86rem; }}
    th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }}
    th {{ color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; }}
    .cluster-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; margin-top:18px; }}
    .cluster-card {{ border:1px solid var(--line); min-width:0; }} .cluster-card>img {{ width:100%; display:block; border-bottom:1px solid var(--line); }}
    .cluster-copy {{ padding:13px 15px 15px; }} .cluster-copy p {{ margin:6px 0; }}
    .metric-line {{ display:flex; flex-wrap:wrap; gap:8px 14px; font-size:.82rem; margin-top:9px; }}
    .flag {{ font:600 .68rem ui-sans-serif,sans-serif; color:var(--warn); text-transform:uppercase; letter-spacing:.06em; }}
    details {{ border-top:1px solid var(--line); padding:14px 0; }} summary {{ cursor:pointer; font-weight:650; }}
    .two-up {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    figure {{ margin:0; min-width:0; }} figure img {{ width:100%; display:block; border:1px solid var(--line); }} figcaption {{ margin-top:6px; }}
    .nn-case,.pair-block {{ padding:24px 0; border-top:1px solid var(--line); }} .nn-case header h3 {{ margin:0; }} .nn-case header p {{ margin:3px 0 14px; color:var(--muted); }}
    .projection {{ margin-top:24px; }} .projection img {{ width:100%; display:block; border:1px solid var(--line); }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .downloads a {{ color:var(--accent); margin-right:16px; }}
    footer {{ margin-top:64px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:.82rem; }}
    @media(max-width:900px) {{ .protocol {{ grid-template-columns:repeat(2,1fr); }} .cluster-grid,.two-up {{ grid-template-columns:1fr; }} }}
    @media(max-width:520px) {{ main {{ width:calc(100% - 24px); padding-top:28px; }} .protocol {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <header class="page-header">
    <div class="eyebrow">BioImages · exploratory representation analysis</div>
    <h1>Visual structure before labels</h1>
    <p>Frozen BioCLIP and DINOv3 embeddings are examined as perceptual spaces. K-means and nearest-neighbor retrieval use no species label. Labels, organs and benchmark outcomes are joined only afterward for analysis.</p>
    <p><a href="efficientnet_control.html">Open the final BioCLIP / DINOv3 / EfficientNet multi-seed representation comparison.</a></p>
    <div class="protocol"><div><b>1,641</b>embedded images</div><div><b>211</b>strict test images</div><div><b>10 / 20 / 50</b>K-means values</div><div><b>{len(download_statuses)-download_failures}/{len(download_statuses)}</b>source images available</div></div>
  </header>

  <section class="major">
    <h2>Representation structure</h2>
    <p class="note">NMI and purity are post-hoc label summaries, not clustering inputs. ARI compares BioCLIP and DINO cluster assignments at the same k. Higher values indicate stronger association or agreement, not necessarily a better representation.</p>
    <div class="table-wrap"><table><thead><tr><th>Space</th><th>k</th><th>Species NMI</th><th>Organ NMI</th><th>Species purity</th><th>Organ purity</th><th>Cluster size min/median/max</th><th>Cross-space ARI</th></tr></thead><tbody>{''.join(structure_rows)}</tbody></table></div>
    <p>Mean cross-observation BioCLIP/DINO neighbor overlap@10: <b>{pct(neighbor_comparison['cross_observation_neighbor_overlap_at_10_mean'])}</b>; median: <b>{pct(neighbor_comparison['cross_observation_neighbor_overlap_at_10_median'])}</b>.</p>
  </section>

  <section class="major">
    <h2>Error-enriched cluster review</h2>
    <p class="note">Shown clusters are ranked within each space and k by BioCLIP-error or all-four-wrong enrichment. “FDR-enriched” uses a one-sided hypergeometric test with BH q≤0.10; “descriptively enriched” means at least 5 test images plus ≥1.5× BioCLIP-error or ≥2× all-four-wrong enrichment. No cluster is automatically named.</p>
    {''.join(enriched_sections)}
  </section>

  <section class="major">
    <h2>Complete cluster atlas</h2>
    <p>Every cluster has a centroid-nearest contact sheet with up to 20 images. Image IDs are shown, but no generated morphology labels are added.</p>
    {''.join(atlas_sections)}
  </section>

  <section class="major">
    <h2>All-four-wrong nearest neighbors</h2>
    <p class="note">Each query is followed by its eight closest images after excluding every image from the same observation. Similarity is cosine similarity in the named frozen embedding space.</p>
    {''.join(nn_rows)}
  </section>

  <section class="major">
    <h2>Repeated within-genus confusion pairs</h2>
    <p class="note">Within-species similarity excludes same-observation image pairs. Nearest-neighbor rates also exclude the query observation. These values describe local geometry; they do not establish a causal explanation for a classification error.</p>
    {''.join(pair_sections)}
    <div class="two-up projection">
      <figure><img loading="lazy" src="assets/plots/bioclip_confusion_pairs.png" alt="BioCLIP PCA and UMAP plots for confusion pairs"><figcaption>BioCLIP global PCA and UMAP; pair species highlighted.</figcaption></figure>
      <figure><img loading="lazy" src="assets/plots/dinov3_confusion_pairs.png" alt="DINOv3 PCA and UMAP plots for confusion pairs"><figcaption>DINOv3 global PCA and UMAP; pair species highlighted.</figcaption></figure>
    </div>
    <p class="small">PCA and UMAP are two-dimensional projections of the full 1,641-image space. Apparent separation or overlap can change under projection and should not be over-interpreted.</p>
  </section>

  <section class="major downloads">
    <h2>Data exports</h2>
    <p><a href="data/cluster_summary.csv">cluster summary</a><a href="data/cluster_assignments.csv">cluster assignments</a><a href="data/nearest_neighbors.csv">nearest neighbors</a><a href="data/confusion_pair_metrics.csv">confusion-pair metrics</a><a href="data/structure_summary.csv">structure summary</a></p>
  </section>

  <footer>Seed {SEED}. K-means uses normalized frozen embeddings with n_init=20. No backbone training, classifier training, automatic cluster naming or LLM morphology annotation is used in this report.</footer>
</main></body></html>"""
    (HERE / "report.html").write_text(report, encoding="utf-8")


def write_readme(neighbor_comparison: dict[str, float], download_statuses: dict[str, str]) -> None:
    failures = {key: value for key, value in download_statuses.items() if value.startswith("failed")}
    content = f"""# BioImages representation analysis

This directory analyzes saved frozen BioCLIP and DINOv3 embeddings. It does not train a model, rerun inference, name clusters, or generate morphology explanations.

## Protocol

- Embedding universe: 1,641 images from the strict individual-disjoint split.
- Test error overlay: the existing 211-image four-baseline prediction table.
- K-means: k=10, 20, 50; labels excluded; random seed {SEED}; n_init=20.
- Similarity: cosine similarity on normalized embeddings.
- Nearest-neighbor galleries exclude all images from the query observation.
- Error enrichment: post-hoc hypergeometric tests with BH correction, plus explicitly labeled descriptive thresholds.
- PCA and UMAP: global 1,641-image projections, used only for visualization.

Mean BioCLIP/DINO cross-observation neighbor overlap@10: {neighbor_comparison['cross_observation_neighbor_overlap_at_10_mean']:.4f}.

Image download failures while building contact sheets: {len(failures)}.

## Outputs

- `report.html`: meeting-ready static report.
- `data/cluster_summary.csv`: cluster composition, purity, organ distribution, four-baseline test accuracy and enrichment statistics.
- `data/cluster_assignments.csv`: every image assignment and centroid distance for both spaces and all k values.
- `data/nearest_neighbors.csv`: all and cross-observation Top-10 neighbors for every test image in both spaces.
- `data/confusion_pair_metrics.csv`: similarity and neighbor metrics for the repeated confusion pairs.
- `data/structure_summary.csv`: post-hoc species/organ association and cross-space cluster agreement.
- `assets/cluster_sheets/`: all 160 centroid-nearest contact sheets.
- `assets/nn_galleries/`: BioCLIP and DINO galleries for all 17 all-four-wrong queries.
- `assets/pair_sheets/` and `assets/plots/`: confusion-pair representative images and global PCA/UMAP panels.

The persisted source vectors are in `outputs/bioclip25_strict_embeddings/embeddings.npz` and `outputs/dinov3_strict_embeddings/embeddings.npz`.
"""
    (HERE / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    embeddings, image_ids, manifest, unified, metadata = load_inputs()
    assignments, cluster_summary, representatives, structure = fit_clusters(
        embeddings, image_ids, manifest, unified
    )
    neighbors, gallery_neighbors, neighbor_comparison = nearest_neighbors(
        embeddings, image_ids, manifest, unified
    )
    confusion_metrics, pairs, pair_representatives = confusion_pair_analysis(
        embeddings, image_ids, manifest
    )

    required_images: set[str] = set()
    for ids in representatives.values():
        required_images.update(ids)
    all_wrong_ids = unified.loc[unified.correct_count == 0, "image_id"].tolist()
    for representation in embeddings:
        for query_id in all_wrong_ids:
            required_images.add(query_id)
            required_images.update(image_id for image_id, _ in gallery_neighbors[(representation, query_id)])
    for ids in pair_representatives.values():
        required_images.update(ids)
    print(f"Need {len(required_images)} unique images for visual assets", flush=True)

    download_statuses = create_visual_assets(
        representatives,
        gallery_neighbors,
        pair_representatives,
        pairs,
        manifest,
        unified,
        metadata,
        embeddings,
        image_ids,
    )

    assignments.to_csv(DATA_DIR / "cluster_assignments.csv", index=False)
    cluster_summary.to_csv(DATA_DIR / "cluster_summary.csv", index=False)
    neighbors.to_csv(DATA_DIR / "nearest_neighbors.csv", index=False)
    confusion_metrics.to_csv(DATA_DIR / "confusion_pair_metrics.csv", index=False)
    structure.to_csv(DATA_DIR / "structure_summary.csv", index=False)
    pd.DataFrame(
        [{"image_id": key, "status": value} for key, value in sorted(download_statuses.items())]
    ).to_csv(DATA_DIR / "image_download_status.csv", index=False)
    (DATA_DIR / "analysis_summary.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "images": len(image_ids),
                "test_images": len(unified),
                "cluster_rows": len(cluster_summary),
                "all_four_wrong_queries": len(all_wrong_ids),
                "confusion_pairs": pairs,
                "neighbor_comparison": neighbor_comparison,
                "download_status": dict(Counter(download_statuses.values())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_report(
        cluster_summary,
        structure,
        neighbor_comparison,
        confusion_metrics,
        pairs,
        unified,
        download_statuses,
    )
    write_readme(neighbor_comparison, download_statuses)

    # Final reconciliations.
    assert len(assignments) == len(image_ids) * len(embeddings) * len(K_VALUES)
    assert len(cluster_summary) == sum(K_VALUES) * len(embeddings)
    assert len(neighbors) == len(unified) * len(embeddings) * 2 * NN_COUNT
    assert len(list((ASSET_DIR / "cluster_sheets").rglob("*.jpg"))) == sum(K_VALUES) * len(embeddings)
    assert len(list((ASSET_DIR / "nn_galleries").rglob("*.jpg"))) == len(all_wrong_ids) * len(embeddings)
    assert len(list((ASSET_DIR / "pair_sheets").rglob("*.jpg"))) == len(pairs) * len(embeddings)
    print(structure.to_string(index=False), flush=True)
    print(json.dumps(neighbor_comparison, indent=2), flush=True)
    print(f"Wrote {HERE / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
