#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6", "scikit-learn==1.7.2"]
# ///
"""Build individual-disjoint Gemma-supervised probes and agreement artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from tagging_vocabulary import TAGS, STRUCTURE_TAGS, VOCABULARY, canonical_tag, weak_tags


SEED = 42
MODELS = (
    ("dinov3", "DINOv3"),
    ("bioclip", "BioCLIP 2.5"),
    ("efficientnet_b0", "EfficientNet-B0"),
)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_corpus(path: Path) -> list[dict]:
    corpus = json.loads(path.read_text())
    rows = []
    for scene in corpus["scenes"]:
        for image in scene["images"]:
            rows.append({
                "image_id": image["image_id"], "file": image["file"],
                "species": scene["scientific_name"], "common_name": scene.get("vernacular", ""),
                "individual_id": image["individual_id"], "organ_tag": image["organ_tag"],
                "organ_category": image["organ_category"], "subview": image["subview"],
                "bioimages_label": f'{image["organ_category"]} / {image["subview"]}',
                "canonical_tag": canonical_tag(image["organ_tag"]),
                "weak_tags": weak_tags(image["organ_tag"], image["organ_category"], image["subview"]),
                "thumbnail_url": image["source_urls"]["tn"].replace("http://", "https://"),
                "image_url": image["source_urls"]["gq"].replace("http://", "https://"),
            })
    rows.sort(key=lambda row: row["image_id"])
    if len(rows) != 1899 or len({row["image_id"] for row in rows}) != 1899:
        raise RuntimeError("Expected 1,899 unique corpus rows")
    return rows


def load_gemma(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row.get("status") == "ok": rows[row["image_id"]] = row
    return rows


def optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if int(y_true.sum()) == 0: return 0.99
    candidates = np.unique(np.concatenate([np.linspace(0.15, 0.85, 29), np.quantile(scores, [0.25, 0.5, 0.75])]))
    best = (0.0, -1.0, 0.5)
    for threshold in candidates:
        pred = scores >= threshold
        tp = int(np.sum(pred & (y_true == 1))); fp = int(np.sum(pred & (y_true == 0))); fn = int(np.sum(~pred & (y_true == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, -abs(float(threshold) - 0.5), float(threshold))
        if candidate > best: best = candidate
    return best[2]


def probe_model(model_key: str, embedding_path: Path, rows: list[dict], gemma: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    loaded = np.load(embedding_path, allow_pickle=False)
    ids = [str(value) for value in loaded["image_ids"]]
    matrix = loaded["embeddings"].astype(np.float32, copy=False)
    if len(ids) != 1899 or matrix.shape[0] != 1899: raise RuntimeError(f"{model_key}: incomplete embeddings")
    feature_by_id = {image_id: matrix[index] for index, image_id in enumerate(ids)}
    x = np.stack([feature_by_id[row["image_id"]] for row in rows])
    # Gemma labels from training individuals are the multi-label supervision.
    # Labels for each held-out individual are used only after prediction, as the
    # reference for evaluation; the image and its labels never enter its model fit.
    y = np.asarray([
        [tag in set(gemma[row["image_id"]]["normalized_tags"]) for tag in TAGS]
        for row in rows
    ], dtype=np.int8)
    primary = np.asarray([row["organ_tag"] for row in rows])
    groups = np.asarray([row["individual_id"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilities = np.zeros((len(rows), len(TAGS)), dtype=np.float32)
    thresholds = np.zeros((len(rows), len(TAGS)), dtype=np.float32)
    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, primary, groups), 1):
        fold_thresholds = np.full(len(TAGS), 0.5, dtype=np.float32)
        for tag_index, tag in enumerate(TAGS):
            train_y = y[train_idx, tag_index]
            if len(np.unique(train_y)) < 2:
                score = np.full(len(test_idx), float(train_y.mean()), dtype=np.float32)
                probabilities[test_idx, tag_index] = score
                fold_thresholds[tag_index] = 0.99
                continue
            classifier = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1200, solver="liblinear", random_state=SEED + fold,
            )
            classifier.fit(x[train_idx], train_y)
            train_scores = classifier.predict_proba(x[train_idx])[:, 1]
            threshold = optimal_threshold(train_y, train_scores)
            probabilities[test_idx, tag_index] = classifier.predict_proba(x[test_idx])[:, 1]
            fold_thresholds[tag_index] = threshold
        thresholds[test_idx] = fold_thresholds
        fold_records.append({
            "model_key": model_key, "fold": fold, "train_images": len(train_idx), "test_images": len(test_idx),
            "train_individuals": len(set(groups[train_idx])), "test_individuals": len(set(groups[test_idx])),
            "individual_leakage": bool(set(groups[train_idx]) & set(groups[test_idx])),
        })

    return probabilities, thresholds, fold_records


def selected_tags(scores: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    selected = [index for index, score in enumerate(scores) if score >= thresholds[index]]
    if not any(TAGS[index] in STRUCTURE_TAGS for index in selected):
        selected.append(max((i for i, tag in enumerate(TAGS) if tag in STRUCTURE_TAGS), key=lambda i: scores[i]))
    selected = sorted(set(selected), key=lambda index: (-float(scores[index]), TAGS[index]))[:10]
    return [{"tag": TAGS[index], "confidence": round(float(scores[index]), 4)} for index in selected]


def set_metrics(predicted: set[str], reference: set[str]) -> dict:
    intersection = len(predicted & reference)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(predicted | reference)
    return {"precision": precision, "recall": recall, "f1": f1, "jaccard": intersection / union if union else 1.0}


def build(corpus: Path, embeddings: Path, gemma_path: Path, output: Path) -> dict:
    rows = load_corpus(corpus); by_id = {row["image_id"]: row for row in rows}
    gemma = load_gemma(gemma_path)
    if len(gemma) != 1899: raise RuntimeError(f"Gemma reference is incomplete: {len(gemma)}/1899")
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True); (output / "metrics").mkdir(exist_ok=True)
    for source_name, destination_name in (("prompt.txt", "gemma_prompt.txt"), ("summary.json", "gemma_run_summary.json")):
        source = gemma_path.parent / source_name
        if source.exists():
            shutil.copy2(source, output / destination_name)

    gemma_rows = []
    with (output / "gemma_reference.jsonl").open("w") as jsonl:
        for row in rows:
            source = gemma[row["image_id"]]
            record = {
                "image_id": row["image_id"],
                "model": source.get("model", "google/gemma-3-27b-it"),
                "free_tags": source["tags"],
                "normalized_tags": source["normalized_tags"],
                "attempts": source.get("attempts", 1),
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            gemma_rows.append({
                "image_id": row["image_id"],
                "model": record["model"],
                "free_tags": json.dumps(record["free_tags"], ensure_ascii=False),
                "normalized_tags": json.dumps(record["normalized_tags"], ensure_ascii=False),
                "attempts": record["attempts"],
            })
    write_csv(output / "gemma_reference.csv", gemma_rows)

    mapping_counter = Counter((row["organ_tag"], row["organ_category"], row["subview"], tuple(row["weak_tags"]), row["canonical_tag"]) for row in rows)
    mapping_rows = [{
        "organ_tag": key[0], "organ_category": key[1], "subview": key[2],
        "bioimages_canonical_comparison_tag": key[4] or "not evaluable",
        "mapped_visible_tags": json.dumps(list(key[3])), "images": count,
    } for key, count in sorted(mapping_counter.items())]
    write_csv(output / "bioimages_vocabulary_mapping.csv", mapping_rows)
    (output / "tag_vocabulary.json").write_text(json.dumps({
        "vocabulary": VOCABULARY,
        "policy": "Visible structures may co-occur. Specific view tags supplement, never replace, structure tags.",
        "mapping_caveat": "Mappings are for BioImages audit and vocabulary provenance; unlisted tags are not asserted absent and do not train the probes.",
    }, indent=2) + "\n")

    bioimages_support = Counter(tag for row in rows for tag in row["weak_tags"])
    gemma_support = Counter(tag for row in rows for tag in gemma[row["image_id"]]["normalized_tags"])
    vocabulary_rows = [{
        **item,
        "bioimages_mapping_support": bioimages_support[item["tag"]],
        "gemma_reference_support": gemma_support[item["tag"]],
    } for item in VOCABULARY]
    write_csv(output / "tag_vocabulary.csv", vocabulary_rows)

    all_predictions: dict[str, dict[str, dict]] = defaultdict(dict)
    fold_records = []
    for model_key, model_name in MODELS:
        probabilities, thresholds, folds = probe_model(model_key, embeddings / f"{model_key}.npz", rows, gemma)
        fold_records.extend(folds)
        prediction_rows = []
        for index, row in enumerate(rows):
            tags = selected_tags(probabilities[index], thresholds[index])
            tag_set = {item["tag"] for item in tags}
            reference = set(gemma[row["image_id"]]["normalized_tags"])
            agreement = set_metrics(tag_set, reference)
            canonical = row["canonical_tag"]
            prediction = {
                "model": model_name, "model_key": model_key, "tags": tags,
                "bioimages_match": canonical in tag_set if canonical else None,
                "top1_exact_match": tags[0]["tag"] == canonical if canonical else None,
                "gemma_precision": round(agreement["precision"], 4),
                "gemma_recall": round(agreement["recall"], 4),
                "gemma_f1": round(agreement["f1"], 4),
                "gemma_jaccard": round(agreement["jaccard"], 4),
            }
            all_predictions[row["image_id"]][model_key] = prediction
            prediction_rows.append({
                "image_id": row["image_id"], "species": row["species"], "individual_id": row["individual_id"],
                "bioimages_label": row["bioimages_label"], "bioimages_canonical_tag": canonical or "",
                "predicted_tags": json.dumps([item["tag"] for item in tags]),
                "tag_confidences": json.dumps({item["tag"]: item["confidence"] for item in tags}),
                "bioimages_match": prediction["bioimages_match"], "top1_exact_match": prediction["top1_exact_match"],
                "gemma_tags": json.dumps(gemma[row["image_id"]]["tags"]),
                "gemma_normalized_tags": json.dumps(gemma[row["image_id"]]["normalized_tags"]),
                "gemma_precision": prediction["gemma_precision"], "gemma_recall": prediction["gemma_recall"],
                "gemma_f1": prediction["gemma_f1"], "gemma_jaccard": prediction["gemma_jaccard"],
            })
        write_csv(output / "predictions" / f"{model_key}.csv", prediction_rows)

    write_csv(output / "metrics" / "fold_audit.csv", fold_records)
    if any(row["individual_leakage"] for row in fold_records): raise RuntimeError("individual_id leakage")

    comparison_rows = []
    disagreement_rows = []
    for row in rows:
        image_id = row["image_id"]; model_predictions = all_predictions[image_id]
        sets = {key: {item["tag"] for item in pred["tags"]} for key, pred in model_predictions.items()}
        pairwise = []
        for left, right in (("dinov3", "bioclip"), ("dinov3", "efficientnet_b0"), ("bioclip", "efficientnet_b0")):
            union = sets[left] | sets[right]
            pairwise.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
        mean_pairwise = sum(pairwise) / len(pairwise)
        top1_values = {key: pred["tags"][0]["tag"] for key, pred in model_predictions.items()}
        model_disagreement = mean_pairwise < 0.5 or len(set(top1_values.values())) > 1
        disagreement_rows.append({
            "image_id": image_id, "species": row["species"], "mean_pairwise_jaccard": round(mean_pairwise, 4),
            "model_disagreement": model_disagreement, **{f"{key}_top1": value for key, value in top1_values.items()},
        })
        for model_key, model_name in MODELS:
            pred = model_predictions[model_key]
            categories = []
            if pred["bioimages_match"] is True and pred["gemma_f1"] >= 0.67: categories.append("good match")
            if pred["bioimages_match"] is False: categories.append("BioImages mismatch")
            if pred["gemma_jaccard"] < 0.33: categories.append("Gemma disagreement")
            if model_disagreement: categories.append("model disagreement")
            comparison_rows.append({
                "image_id": image_id, "species": row["species"], "model": model_name, "model_key": model_key,
                "bioimages_label": row["bioimages_label"], "bioimages_canonical_tag": row["canonical_tag"] or "",
                "predicted_tags": json.dumps([item["tag"] for item in pred["tags"]]),
                "tag_confidences": json.dumps({item["tag"]: item["confidence"] for item in pred["tags"]}),
                "gemma_tags": json.dumps(gemma[image_id]["tags"]),
                "gemma_normalized_tags": json.dumps(gemma[image_id]["normalized_tags"]),
                "bioimages_match": pred["bioimages_match"], "top1_exact_match": pred["top1_exact_match"],
                "gemma_f1": pred["gemma_f1"], "gemma_jaccard": pred["gemma_jaccard"],
                "model_pairwise_jaccard": round(mean_pairwise, 4), "gallery_categories": "; ".join(categories),
                "thumbnail_url": row["thumbnail_url"], "image_url": row["image_url"],
            })
    write_csv(output / "per_image_comparison.csv", comparison_rows)
    write_csv(output / "metrics" / "model_disagreements.csv", disagreement_rows)

    summary_rows = []
    per_tag_rows = []
    for model_key, model_name in MODELS:
        subset = [row for row in comparison_rows if row["model_key"] == model_key]
        evaluable = [row for row in subset if row["bioimages_canonical_tag"]]
        tp_total = fp_total = fn_total = 0
        for row in subset:
            pred = set(json.loads(row["predicted_tags"])); ref = set(json.loads(row["gemma_normalized_tags"]))
            tp_total += len(pred & ref); fp_total += len(pred - ref); fn_total += len(ref - pred)
        micro_precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0
        micro_recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0
        summary_rows.append({
            "model": model_name, "model_key": model_key, "images": len(subset), "bioimages_evaluable_images": len(evaluable),
            "bioimages_match_rate": sum(str(row["bioimages_match"]).lower() == "true" for row in evaluable) / len(evaluable),
            "top1_exact_match": sum(str(row["top1_exact_match"]).lower() == "true" for row in evaluable) / len(evaluable),
            "gemma_micro_precision": micro_precision, "gemma_micro_recall": micro_recall, "gemma_micro_f1": micro_f1,
            "gemma_mean_jaccard": sum(float(row["gemma_jaccard"]) for row in subset) / len(subset),
            "mean_predicted_tags": sum(len(json.loads(row["predicted_tags"])) for row in subset) / len(subset),
        })
        for tag in TAGS:
            canonical_rows = [row for row in subset if row["bioimages_canonical_tag"] == tag]
            pred_positive = [tag in set(json.loads(row["predicted_tags"])) for row in subset]
            gemma_positive = [tag in set(json.loads(row["gemma_normalized_tags"])) for row in subset]
            tp = sum(p and g for p, g in zip(pred_positive, gemma_positive)); fp = sum(p and not g for p, g in zip(pred_positive, gemma_positive)); fn = sum((not p) and g for p, g in zip(pred_positive, gemma_positive))
            precision = tp / (tp + fp) if tp + fp else 0; recall = tp / (tp + fn) if tp + fn else 0
            per_tag_rows.append({
                "model": model_name, "model_key": model_key, "tag": tag,
                "bioimages_support": len(canonical_rows),
                "bioimages_match_rate": sum(tag in set(json.loads(row["predicted_tags"])) for row in canonical_rows) / len(canonical_rows) if canonical_rows else "",
                "gemma_support": sum(gemma_positive), "gemma_precision": precision, "gemma_recall": recall,
                "gemma_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
                "predicted_count": sum(pred_positive),
            })

    write_csv(output / "metrics" / "model_summary.csv", summary_rows)
    write_csv(output / "metrics" / "per_tag_metrics.csv", per_tag_rows)

    gallery_counts = []
    for model_key, model_name in MODELS:
        subset = [row for row in comparison_rows if row["model_key"] == model_key]
        gallery_counts.append({
            "model": model_name,
            "model_key": model_key,
            "good_matches": sum("good match" in row["gallery_categories"] for row in subset),
            "bioimages_mismatches": sum("BioImages mismatch" in row["gallery_categories"] for row in subset),
            "gemma_disagreements": sum("Gemma disagreement" in row["gallery_categories"] for row in subset),
            "model_disagreements": sum("model disagreement" in row["gallery_categories"] for row in subset),
        })
    (output / "metrics" / "gallery_counts.json").write_text(json.dumps(gallery_counts, indent=2) + "\n")

    analysis_images = []
    for row in rows:
        image_id = row["image_id"]
        disagreement = next(item for item in disagreement_rows if item["image_id"] == image_id)
        analysis_images.append({
            "image_id": image_id,
            "bioimages_label": row["bioimages_label"], "bioimages_canonical_tag": row["canonical_tag"],
            "gemma_tags": gemma[image_id]["tags"], "gemma_normalized_tags": gemma[image_id]["normalized_tags"],
            "models": all_predictions[image_id],
            "model_disagreement": disagreement["model_disagreement"],
            "mean_pairwise_jaccard": disagreement["mean_pairwise_jaccard"],
        })
    payload = {
        "method": {
            "name": "5-fold individual-disjoint Gemma-supervised frozen-embedding multi-label linear probe",
            "seed": SEED, "folds": 5, "classifier": "per-tag balanced logistic regression",
            "thresholds": "per-tag thresholds selected on outer-fold training predictions; test fold never used",
            "fine_tuning": False,
            "supervision_note": "Each fold learns only from Gemma normalized tags on training individuals; held-out image labels never enter its fit.",
            "reference_note": "Gemma free tags are displayed; agreement uses held-out Gemma normalized_tags from the same controlled vocabulary. Gemma is a model teacher/reference, not human ground truth.",
        },
        "vocabulary": VOCABULARY, "summary": summary_rows, "per_tag": per_tag_rows,
        "images": analysis_images,
    }
    (output / "analysis_data.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {"models": summary_rows, "images": 1899, "vocabulary_size": len(TAGS), "gemma_images": len(gemma), "individual_leakage": False, "gallery_counts": gallery_counts}
    (output / "metrics" / "model_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "# Full-corpus multi-label tagging report", "",
        "All 1,899 BioImages photographs were tagged by three frozen representations. Predictions are out-of-fold by `individual_id`; each fold learns from Gemma multi-label references only on its training individuals, and no visual backbone was fine-tuned.", "",
        "| Model | BioImages match | Top-1 exact | Gemma micro F1 | Mean Jaccard | Tags/image |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary_rows:
        report.append(
            f'| {item["model"]} | {item["bioimages_match_rate"]:.1%} | {item["top1_exact_match"]:.1%} | '
            f'{item["gemma_micro_f1"]:.1%} | {item["gemma_mean_jaccard"]:.1%} | {item["mean_predicted_tags"]:.2f} |'
        )
    report.extend([
        "", "## Interpretation notes", "",
        "- BioImages match asks whether its one canonical coarse structure appears anywhere in a model's multi-label set; it is not exhaustive visual ground truth.",
        "- Gemma is a model-generated teacher/reference, not human ground truth. Its free tags are preserved and its normalized tags supervise training folds and score only held-out folds.",
        "- BioImages annotations define the vocabulary mapping and independent canonical-match benchmark; they do not supervise the multi-label probes.",
        "- The two BioImages records whose coarse tag is `unspecified` remain in the gallery but are excluded from the BioImages match denominator.",
        "", "## Outputs", "",
        "- `predictions/`: one row per image for each frozen visual model, including every tag confidence",
        "- `gemma_reference.csv` and `.jsonl`: free and normalized Gemma tags for every image",
        "- `gemma_prompt.txt` and `gemma_run_summary.json`: exact prompt and full-run provenance",
        "- `per_image_comparison.csv`: references, agreement scores, and gallery categories",
        "- `metrics/`: summary, tag-level, leakage-audit, disagreement, and gallery-count files",
        "- `analysis_data.json`: compact data embedded into the static browser",
    ])
    (output / "TAGGING_REPORT.md").write_text("\n".join(report) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("scratch/bioimages-ne-trees/corpus.json"))
    parser.add_argument("--embeddings", type=Path, default=Path("botanical-computer-vision/outputs/full_multilabel_embeddings"))
    parser.add_argument("--gemma", type=Path, default=Path("botanical-computer-vision/outputs/gemma_multilabel_reference/predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("botanical-computer-vision/bioimages_browser/tagging"))
    args = parser.parse_args()
    print(json.dumps(build(args.corpus, args.embeddings, args.gemma, args.output), indent=2))


if __name__ == "__main__":
    main()
