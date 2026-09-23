# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "pandas==2.2.3",
#   "pillow==11.3.0",
#   "scikit-learn==1.7.2",
# ]
# ///
"""Build the frozen EfficientNet-B0 benchmark and k=20 representation control."""

from __future__ import annotations

import csv
import html
import io
import json
import math
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


SEED = 42
K = 20
CENTROID_SHEET_SIZE = 20

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = ROOT / "outputs" / "efficientnet_b0_strict_embeddings"
DINO_OUTPUT = ROOT / "outputs" / "dinov3_strict_embeddings"
BIOCLIP_OUTPUT = ROOT / "outputs" / "bioclip25_strict_embeddings"
TAXONOMY = ROOT / "error_analysis" / "taxonomy_analysis" / "taxonomy_mapping.csv"
UNIFIED = ROOT / "error_analysis" / "data" / "unified_predictions_image.csv"
METADATA = ROOT / "app" / "dashboard-data.json"
DATA_DIR = HERE / "data"
ASSET_DIR = HERE / "assets" / "cluster_sheets" / "efficientnet_b0" / "k20"
CACHE_DIR = Path("/tmp/bioimages-representation-cache")

ImageFile.LOAD_TRUNCATED_IMAGES = True


def safe_id(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def distribution(values: pd.Series) -> str:
    return json.dumps(
        {str(key): int(value) for key, value in values.value_counts().items()},
        ensure_ascii=False,
    )


def top_distribution(raw: str, limit: int = 4) -> str:
    values = json.loads(raw)
    return ", ".join(f"{key} {value}" for key, value in list(values.items())[:limit])


def load_taxonomy(classes: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    taxonomy = pd.read_csv(TAXONOMY).drop_duplicates("species").set_index("species")
    missing = sorted(set(classes) - set(taxonomy.index))
    if missing:
        raise ValueError(f"Missing taxonomy for: {missing}")
    genus = taxonomy.genus.to_dict()
    family = taxonomy.family.to_dict()
    return genus, family


def enrich_test_predictions(
    predictions: pd.DataFrame,
    genus: dict[str, str],
    family: dict[str, str],
) -> pd.DataFrame:
    result = predictions.copy()
    result["ground_truth_genus"] = result.ground_truth_species.map(genus)
    result["predicted_genus"] = result.predicted_species.map(genus)
    result["ground_truth_family"] = result.ground_truth_species.map(family)
    result["predicted_family"] = result.predicted_species.map(family)
    if result[["ground_truth_genus", "predicted_genus", "ground_truth_family", "predicted_family"]].isna().any().any():
        raise ValueError("Unexpected missing taxonomy value")
    result["genus_correct"] = result.ground_truth_genus == result.predicted_genus
    result["family_correct"] = result.ground_truth_family == result.predicted_family
    return result


def train_and_save_all_predictions(
    matrix: np.ndarray,
    image_ids: list[str],
    manifest: pd.DataFrame,
    classes: list[str],
    best_c: float,
    remote_test: pd.DataFrame,
) -> pd.DataFrame:
    class_to_index = {species: index for index, species in enumerate(classes)}
    aligned = manifest.set_index("image_id").loc[image_ids].reset_index()
    y = aligned.ground_truth_species.map(class_to_index).to_numpy()
    train_validation = aligned.split.isin(["train", "validation"]).to_numpy()
    classifier = LogisticRegression(
        C=best_c,
        max_iter=3000,
        solver="lbfgs",
        random_state=SEED,
    )
    classifier.fit(matrix[train_validation], y[train_validation])
    probabilities = classifier.predict_proba(matrix)
    top5 = np.argsort(-probabilities, axis=1)[:, :5]
    rows = []
    for index, (image_id, split_name, truth) in enumerate(
        zip(image_ids, aligned.split, aligned.ground_truth_species)
    ):
        candidates = [
            {"species": classes[int(class_index)], "confidence": float(probabilities[index, class_index])}
            for class_index in top5[index]
        ]
        rows.append(
            {
                "image_id": image_id,
                "split": split_name,
                "predicted_species": candidates[0]["species"],
                "confidence": candidates[0]["confidence"],
                "top5": json.dumps(candidates, ensure_ascii=False),
                "ground_truth_species": truth,
                "correct": candidates[0]["species"] == truth,
                "correct_top5": truth in {candidate["species"] for candidate in candidates},
                "evaluation_role": "strict_test" if split_name == "test" else "fit_or_selection_split_do_not_report_as_test",
            }
        )
    result = pd.DataFrame(rows)
    local_test = result.loc[result.split == "test"].set_index("image_id")
    remote = remote_test.set_index("image_id")
    if set(local_test.index) != set(remote.index):
        raise ValueError("Local and remote test prediction IDs differ")
    if not (local_test.loc[remote.index, "predicted_species"] == remote.predicted_species).all():
        raise ValueError("Locally reproduced test predictions differ from remote output")
    if not np.allclose(
        local_test.loc[remote.index, "confidence"].to_numpy(),
        remote.confidence.to_numpy(),
        atol=2e-7,
    ):
        raise ValueError("Locally reproduced test probabilities differ from remote output")
    result.to_csv(OUTPUT / "all_predictions.csv", index=False)
    return result


def probe_comparison(
    genus: dict[str, str], family: dict[str, str]
) -> pd.DataFrame:
    sources = {
        "bioclip": ("BioCLIP 2.5 frozen + linear probe", BIOCLIP_OUTPUT),
        "dinov3": ("DINOv3 frozen + linear probe", DINO_OUTPUT),
        "efficientnet_b0": ("EfficientNet-B0 ImageNet frozen + linear probe", OUTPUT),
    }
    rows = []
    for key, (label, directory) in sources.items():
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        predictions = enrich_test_predictions(pd.read_csv(directory / "predictions.csv"), genus, family)
        rows.append(
            {
                "representation": key,
                "model": label,
                "embedding_dimension": int(summary["embedding_dimension"]),
                "selected_C": float(summary["best_C"]),
                "species_top1_correct": int(predictions.correct.astype(bool).sum()),
                "species_top1_accuracy": float(predictions.correct.astype(bool).mean()),
                "species_top5_correct": int(predictions.correct_top5.astype(bool).sum()),
                "species_top5_accuracy": float(predictions.correct_top5.astype(bool).mean()),
                "genus_top1_correct": int(predictions.genus_correct.sum()),
                "genus_top1_accuracy": float(predictions.genus_correct.mean()),
                "family_top1_correct": int(predictions.family_correct.sum()),
                "family_top1_accuracy": float(predictions.family_correct.mean()),
                "test_images": len(predictions),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "frozen_probe_comparison.csv", index=False)
    return frame


def fit_k20(
    matrix: np.ndarray,
    image_ids: list[str],
    manifest: pd.DataFrame,
    unified: pd.DataFrame,
    efficientnet_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[str]], np.ndarray]:
    model = KMeans(
        n_clusters=K,
        random_state=SEED,
        n_init=20,
        max_iter=500,
        algorithm="lloyd",
    )
    labels = model.fit_predict(matrix)
    distances = model.transform(matrix)
    aligned = manifest.set_index("image_id").loc[image_ids].reset_index()
    test = unified.set_index("image_id")
    efficientnet_correct = efficientnet_test.set_index("image_id").correct.astype(bool)
    if set(test.index) != set(efficientnet_correct.index):
        raise ValueError("EfficientNet and unified strict test sets differ")
    test = test.copy()
    test["efficientnet_b0_correct"] = efficientnet_correct.loc[test.index]

    assignment_rows = []
    cluster_rows = []
    representatives: dict[int, list[str]] = {}
    baseline_columns = ["bioclip", "dinov3", "gpt_5_4", "gemini_flash_lite", "efficientnet_b0"]
    for index, (image_id, cluster_id) in enumerate(zip(image_ids, labels)):
        assignment_rows.append(
            {
                "representation": "efficientnet_b0",
                "k": K,
                "image_id": image_id,
                "split": aligned.iloc[index].split,
                "cluster_id": int(cluster_id),
                "distance_to_centroid": float(distances[index, cluster_id]),
            }
        )

    for cluster_id in range(K):
        member_indices = np.flatnonzero(labels == cluster_id)
        ordered = member_indices[np.argsort(distances[member_indices, cluster_id])]
        representatives[cluster_id] = [image_ids[index] for index in ordered[:CENTROID_SHEET_SIZE]]
        members = aligned.iloc[member_indices]
        species_counts = members.ground_truth_species.value_counts()
        organ_counts = members.organ_category.value_counts()
        test_ids = sorted(set(members.image_id) & set(test.index))
        test_members = test.loc[test_ids] if test_ids else pd.DataFrame()
        row = {
            "representation": "efficientnet_b0",
            "k": K,
            "cluster_id": cluster_id,
            "size_all": len(members),
            "size_test": len(test_ids),
            "species_count": int(members.ground_truth_species.nunique()),
            "dominant_species": species_counts.index[0],
            "dominant_species_count": int(species_counts.iloc[0]),
            "species_purity": float(species_counts.iloc[0] / len(members)),
            "species_distribution": distribution(members.ground_truth_species),
            "organ_count": int(members.organ_category.nunique()),
            "dominant_organ": organ_counts.index[0],
            "dominant_organ_count": int(organ_counts.iloc[0]),
            "organ_purity": float(organ_counts.iloc[0] / len(members)),
            "organ_distribution": distribution(members.organ_category),
        }
        for baseline in baseline_columns:
            column = f"{baseline}_correct"
            correct = int(test_members[column].astype(bool).sum()) if test_ids else 0
            accuracy = float(test_members[column].astype(bool).mean()) if test_ids else math.nan
            row[column] = correct
            row[f"{baseline}_accuracy"] = accuracy
            row[f"{baseline}_error_rate"] = 1.0 - accuracy if test_ids else math.nan
        cluster_rows.append(row)

    assignments = pd.DataFrame(assignment_rows)
    clusters = pd.DataFrame(cluster_rows)
    assignments.to_csv(DATA_DIR / "efficientnet_k20_cluster_assignments.csv", index=False)
    clusters.to_csv(DATA_DIR / "efficientnet_k20_cluster_summary.csv", index=False)
    return assignments, clusters, representatives, labels


def build_structure_comparison(
    labels: np.ndarray,
    manifest: pd.DataFrame,
    image_ids: list[str],
    clusters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior_structure = pd.read_csv(DATA_DIR / "structure_summary.csv")
    prior_structure = prior_structure.loc[prior_structure.k == K].copy()
    efficientnet_row = {
        "representation": "efficientnet_b0",
        "k": K,
        "species_nmi": float(normalized_mutual_info_score(manifest.set_index("image_id").loc[image_ids].ground_truth_species, labels)),
        "organ_nmi": float(normalized_mutual_info_score(manifest.set_index("image_id").loc[image_ids].organ_category, labels)),
        "weighted_species_purity": float((clusters.species_purity * clusters.size_all).sum() / len(manifest)),
        "weighted_organ_purity": float((clusters.organ_purity * clusters.size_all).sum() / len(manifest)),
        "cluster_size_min": int(clusters.size_all.min()),
        "cluster_size_median": float(clusters.size_all.median()),
        "cluster_size_max": int(clusters.size_all.max()),
    }
    columns = list(efficientnet_row)
    structure = pd.concat([prior_structure[columns], pd.DataFrame([efficientnet_row])], ignore_index=True)

    prior_assignments = pd.read_csv(DATA_DIR / "cluster_assignments.csv")
    prior_assignments = prior_assignments.loc[prior_assignments.k == K]
    label_by_representation: dict[str, np.ndarray] = {}
    for representation in ["bioclip", "dinov3"]:
        series = prior_assignments.loc[prior_assignments.representation == representation].set_index("image_id").cluster_id
        label_by_representation[representation] = series.loc[image_ids].to_numpy()
    label_by_representation["efficientnet_b0"] = labels
    ari_rows = []
    names = list(label_by_representation)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            ari_rows.append(
                {
                    "k": K,
                    "representation_a": first,
                    "representation_b": second,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(label_by_representation[first], label_by_representation[second])
                    ),
                }
            )
    pairwise = pd.DataFrame(ari_rows)
    structure.to_csv(DATA_DIR / "three_representation_k20_summary.csv", index=False)
    pairwise.to_csv(DATA_DIR / "three_representation_k20_pairwise_ari.csv", index=False)
    return structure, pairwise


def download_one(image_id: str, url: str) -> tuple[str, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / f"{safe_id(image_id)}.jpg"
    if destination.exists() and destination.stat().st_size > 0:
        try:
            with Image.open(destination) as image:
                image.verify()
            return image_id, "cached"
        except Exception:
            destination.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "BioImages EfficientNet representation control/1.0"})
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
    try:
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            contained = ImageOps.contain(source, size, method=Image.Resampling.LANCZOS)
        canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    except Exception:
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "image unavailable", fill="#7a817c", font=font(13))
    return canvas


def make_sheet(image_ids: list[str], output: Path, title: str) -> None:
    columns = 5
    tile_size = (210, 155)
    label_height = 36
    header_height = 58
    rows = math.ceil(len(image_ids) / columns)
    sheet = Image.new("RGB", (columns * tile_size[0], header_height + rows * (tile_size[1] + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 15), title, fill="#17221d", font=font(22, bold=True))
    for index, image_id in enumerate(image_ids):
        row, column = divmod(index, columns)
        x = column * tile_size[0]
        y = header_height + row * (tile_size[1] + label_height)
        sheet.paste(load_tile(image_id, tile_size), (x, y))
        draw.rectangle((x, y, x + tile_size[0] - 1, y + tile_size[1] + label_height - 1), outline="#d7ddd8")
        draw.text((x + 6, y + tile_size[1] + 5), image_id[:42], fill="#4f5952", font=font(12))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=86, optimize=True)


def create_contact_sheets(representatives: dict[int, list[str]]) -> dict[str, str]:
    metadata_records = json.loads(METADATA.read_text(encoding="utf-8"))["images"]
    metadata = {row["id"]: row for row in metadata_records}
    needed = {image_id for ids in representatives.values() for image_id in ids}
    statuses = ensure_images(needed, metadata)
    failed = [image_id for image_id, status in statuses.items() if status.startswith("failed")]
    if failed:
        raise RuntimeError(f"Contact-sheet image downloads failed: {failed[:5]}")
    for cluster_id, ids in representatives.items():
        make_sheet(
            ids,
            ASSET_DIR / f"cluster_{cluster_id:02d}.jpg",
            f"EfficientNet-B0 · k=20 · cluster {cluster_id:02d} · centroid-nearest",
        )
    return statuses


def pct(value: float) -> str:
    return f"{value:.1%}"


def build_report(
    comparison: pd.DataFrame,
    clusters: pd.DataFrame,
    structure: pd.DataFrame,
    pairwise: pd.DataFrame,
    validation: pd.DataFrame,
    summary: dict,
    image_statuses: dict[str, str],
) -> None:
    labels = {"bioclip": "BioCLIP 2.5", "dinov3": "DINOv3", "efficientnet_b0": "EfficientNet-B0"}
    benchmark_rows = "".join(
        f"<tr><td>{html.escape(row.model)}</td><td>{row.embedding_dimension}</td><td>{row.selected_C:g}</td>"
        f"<td>{pct(row.species_top1_accuracy)} ({row.species_top1_correct}/211)</td>"
        f"<td>{pct(row.species_top5_accuracy)} ({row.species_top5_correct}/211)</td>"
        f"<td>{pct(row.genus_top1_accuracy)}</td><td>{pct(row.family_top1_accuracy)}</td></tr>"
        for row in comparison.itertuples(index=False)
    )
    structure_rows = "".join(
        f"<tr><td>{labels[row.representation]}</td><td>{row.species_nmi:.3f}</td><td>{row.organ_nmi:.3f}</td>"
        f"<td>{pct(row.weighted_species_purity)}</td><td>{pct(row.weighted_organ_purity)}</td>"
        f"<td>{row.cluster_size_min}/{row.cluster_size_median:.0f}/{row.cluster_size_max}</td></tr>"
        for row in structure.itertuples(index=False)
    )
    ari_rows = "".join(
        f"<tr><td>{labels[row.representation_a]}</td><td>{labels[row.representation_b]}</td>"
        f"<td>{row.adjusted_rand_index:.3f}</td></tr>"
        for row in pairwise.itertuples(index=False)
    )
    validation_rows = "".join(
        f"<tr><td>{row.C:g}</td><td>{pct(row.top1_accuracy)}</td><td>{pct(row.macro_accuracy)}</td>"
        f"<td>{pct(row.top5_accuracy)}</td><td>{row.iterations}</td></tr>"
        for row in validation.itertuples(index=False)
    )
    baseline_labels = {
        "bioclip": "BioCLIP zero-shot",
        "dinov3": "DINO probe",
        "gpt_5_4": "GPT-5.4",
        "gemini_flash_lite": "Gemini",
        "efficientnet_b0": "EfficientNet probe",
    }
    cards = []
    for row in clusters.sort_values("cluster_id").itertuples(index=False):
        metrics = "".join(
            f"<span><b>{label}</b> {('—' if pd.isna(getattr(row, key + '_accuracy')) else pct(getattr(row, key + '_accuracy')))}</span>"
            for key, label in baseline_labels.items()
        )
        cards.append(
            f"""
            <article class="cluster-card">
              <img loading="lazy" src="assets/cluster_sheets/efficientnet_b0/k20/cluster_{row.cluster_id:02d}.jpg" alt="EfficientNet centroid-nearest images for cluster {row.cluster_id}">
              <div class="copy"><h3>Cluster {row.cluster_id:02d}</h3>
              <p>{row.size_all} images · {row.size_test} test · species purity {pct(row.species_purity)} · organ purity {pct(row.organ_purity)}</p>
              <p><b>Species</b> {html.escape(top_distribution(row.species_distribution))}</p>
              <p><b>Organs</b> {html.escape(top_distribution(row.organ_distribution))}</p>
              <div class="metrics">{metrics}</div></div>
            </article>"""
        )
    downloaded = sum(not value.startswith("failed") for value in image_statuses.values())
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EfficientNet-B0 representation control</title>
<style>
:root{{--ink:#17221d;--muted:#667069;--line:#d8ded9;--soft:#f3f5f3;--paper:#fff;--accent:#315f4a}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--paper);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}}main{{width:min(1500px,calc(100% - 40px));margin:auto;padding:48px 0 80px}}h1,h2,h3{{font-family:Georgia,"Times New Roman",serif;font-weight:400}}h1{{font-size:clamp(2.2rem,5vw,4.3rem);line-height:1;margin:8px 0 16px}}h2{{font-size:clamp(1.6rem,3vw,2.5rem);margin:0 0 12px}}h3{{margin:0;font-size:1.12rem}}.eyebrow,.small{{color:var(--muted);font-size:.82rem}}.eyebrow{{text-transform:uppercase;letter-spacing:.11em;font-weight:650}}header{{max-width:1000px;padding-bottom:30px;border-bottom:1px solid var(--line)}}section{{padding-top:52px}}.note{{border-left:3px solid var(--accent);padding:10px 14px;color:var(--muted);background:var(--soft);max-width:1050px}}.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:.86rem}}th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.cluster-card{{border:1px solid var(--line);min-width:0}}.cluster-card>img{{width:100%;display:block;border-bottom:1px solid var(--line)}}.copy{{padding:13px 15px 15px}}.copy p{{margin:6px 0}}.metrics{{display:flex;flex-wrap:wrap;gap:8px 14px;font-size:.82rem;margin-top:9px}}a{{color:var(--accent)}}footer{{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:.82rem}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}@media(max-width:520px){{main{{width:calc(100% - 24px);padding-top:28px}}}}
</style></head><body><main>
<header><div class="eyebrow">BioImages · conventional ImageNet-supervised CNN control</div><h1>EfficientNet-B0 frozen representation</h1><p>The ImageNet-pretrained backbone is frozen. Global-average-pooled 1,280-dimensional penultimate features are L2-normalized, then evaluated with the same observation-disjoint split and multinomial logistic-regression protocol as DINOv3.</p></header>
<section><h2>Strict linear-probe benchmark</h2><p class="note">All three rows use the same 1,181/249/211 image split, the same 63 species and the same validation-selected C grid. Genus and family are post-hoc mappings of the unchanged species prediction.</p><div class="table"><table><thead><tr><th>Frozen representation</th><th>Dim</th><th>Selected C</th><th>Species Top-1</th><th>Species Top-5</th><th>Genus Top-1</th><th>Family Top-1</th></tr></thead><tbody>{benchmark_rows}</tbody></table></div>
<p class="small">EfficientNet cloud run: {summary['total_run_seconds']:.1f}s total; {summary['full_embedding_wall_seconds']:.1f}s full extraction wall time on {html.escape(summary['gpu'])}. Backbone trainable parameters: 0.</p></section>
<section><h2>Validation selection</h2><p class="note">Only logistic-regression regularization C is selected. No EfficientNet variant search and no backbone fine-tuning are performed.</p><div class="table"><table><thead><tr><th>C</th><th>Top-1</th><th>Macro Top-1</th><th>Top-5</th><th>Iterations</th></tr></thead><tbody>{validation_rows}</tbody></table></div></section>
<section><h2>Three representation spaces at k=20</h2><p class="note">K-means receives only normalized embeddings. Species and organ labels are joined afterward. NMI, purity and ARI describe association/agreement; they are not accuracy metrics.</p><div class="table"><table><thead><tr><th>Space</th><th>Species NMI</th><th>Organ NMI</th><th>Species purity</th><th>Organ purity</th><th>Cluster size min/median/max</th></tr></thead><tbody>{structure_rows}</tbody></table></div><div class="table"><table><thead><tr><th>Space A</th><th>Space B</th><th>k=20 ARI</th></tr></thead><tbody>{ari_rows}</tbody></table></div></section>
<section><h2>EfficientNet-B0 k=20 cluster atlas</h2><p>Each contact sheet contains the 20 images nearest its cluster centroid. No cluster names or morphology explanations are generated. {downloaded}/{len(image_statuses)} required source images were available.</p><div class="grid">{''.join(cards)}</div></section>
<section><h2>Data exports</h2><p><a href="data/frozen_probe_comparison.csv">probe comparison</a> · <a href="data/efficientnet_k20_cluster_summary.csv">cluster summary</a> · <a href="data/efficientnet_k20_cluster_assignments.csv">cluster assignments</a> · <a href="data/three_representation_k20_summary.csv">three-space summary</a> · <a href="data/three_representation_k20_pairwise_ari.csv">pairwise ARI</a> · <a href="data/efficientnet_test_predictions.csv">test predictions</a> · <a href="data/efficientnet_all_predictions.csv">all-image predictions</a> · <a href="data/efficientnet_validation_grid.csv">validation grid</a></p></section>
<footer>Seed {SEED}. EfficientNet-B0 uses torchvision IMAGENET1K_V1 preprocessing and weights. Test truth is read only after cloud test probabilities are produced.</footer>
</main></body></html>"""
    (HERE / "efficientnet_probe_control.html").write_text(report, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(OUTPUT / "split_manifest.csv")
    dino_manifest = pd.read_csv(DINO_OUTPUT / "split_manifest.csv")
    if not manifest.equals(dino_manifest):
        raise ValueError("EfficientNet split manifest is not exactly identical to DINOv3")
    if len(manifest) != 1641 or manifest.image_id.nunique() != 1641:
        raise ValueError("Unexpected strict manifest size")
    if manifest.groupby("split").size().to_dict() != {"test": 211, "train": 1181, "validation": 249}:
        raise ValueError("Unexpected split counts")

    with np.load(OUTPUT / "embeddings.npz", allow_pickle=False) as payload:
        image_ids = payload["image_ids"].astype(str).tolist()
        matrix = payload["embeddings"].astype(np.float32)
        splits = payload["splits"].astype(str)
    if matrix.shape != (1641, 1280) or len(set(image_ids)) != 1641:
        raise ValueError(f"Unexpected embedding payload: {matrix.shape}")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=2e-5):
        raise ValueError("EfficientNet embeddings are not normalized")
    if int((splits == "test").sum()) != 211:
        raise ValueError("Embedding split labels do not contain 211 test images")

    classes = sorted(manifest.ground_truth_species.unique())
    genus, family = load_taxonomy(classes)
    remote_test = pd.read_csv(OUTPUT / "predictions.csv")
    hierarchical = enrich_test_predictions(remote_test, genus, family)
    hierarchical.to_csv(OUTPUT / "predictions_hierarchical.csv", index=False)
    hierarchical.to_csv(DATA_DIR / "efficientnet_test_predictions.csv", index=False)
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    all_predictions = train_and_save_all_predictions(
        matrix, image_ids, manifest, classes, float(summary["best_C"]), remote_test
    )
    all_predictions.to_csv(DATA_DIR / "efficientnet_all_predictions.csv", index=False)
    hierarchy_rows = [
        {"taxonomy_level": "species", "correct": int(hierarchical.correct.astype(bool).sum()), "total": len(hierarchical), "accuracy": float(hierarchical.correct.astype(bool).mean())},
        {"taxonomy_level": "genus", "correct": int(hierarchical.genus_correct.sum()), "total": len(hierarchical), "accuracy": float(hierarchical.genus_correct.mean())},
        {"taxonomy_level": "family", "correct": int(hierarchical.family_correct.sum()), "total": len(hierarchical), "accuracy": float(hierarchical.family_correct.mean())},
    ]
    pd.DataFrame(hierarchy_rows).to_csv(OUTPUT / "hierarchical_accuracy.csv", index=False)
    comparison = probe_comparison(genus, family)

    unified = pd.read_csv(UNIFIED)
    assignments, clusters, representatives, labels = fit_k20(
        matrix, image_ids, manifest, unified, hierarchical
    )
    structure, pairwise = build_structure_comparison(labels, manifest, image_ids, clusters)
    statuses = create_contact_sheets(representatives)
    validation = pd.read_csv(OUTPUT / "validation_grid.csv")
    validation.to_csv(DATA_DIR / "efficientnet_validation_grid.csv", index=False)
    build_report(comparison, clusters, structure, pairwise, validation, summary, statuses)

    result = {
        "validated_identical_to_dinov3_manifest": True,
        "images": len(image_ids),
        "embedding_shape": list(matrix.shape),
        "all_image_predictions": len(all_predictions),
        "test_images": len(hierarchical),
        "species_top1_accuracy": float(hierarchical.correct.astype(bool).mean()),
        "species_top5_accuracy": float(hierarchical.correct_top5.astype(bool).mean()),
        "genus_top1_accuracy": float(hierarchical.genus_correct.mean()),
        "family_top1_accuracy": float(hierarchical.family_correct.mean()),
        "selected_C": float(summary["best_C"]),
        "clusters": len(clusters),
        "contact_sheets": len(list(ASSET_DIR.glob("cluster_*.jpg"))),
        "contact_sheet_images": dict(Counter(statuses.values())),
    }
    (OUTPUT / "efficientnet_control_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
