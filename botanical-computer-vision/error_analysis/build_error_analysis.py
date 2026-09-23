#!/usr/bin/env python3
"""Build a reproducible, human-first error analysis from existing predictions only."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DOWNLOADS = ROOT / "public" / "downloads"
MODEL_KEYS = ["bioclip", "dinov3", "gpt_5_4", "gemini_flash_lite"]
MODEL_LABELS = {
    "bioclip": "BioCLIP 2.5 zero-shot (strict 63-class)",
    "dinov3": "DINOv3 frozen + linear probe (strict 63-class)",
    "gpt_5_4": "GPT-5.4 direct VLM (85-class)",
    "gemini_flash_lite": "Gemini 2.5 Flash Lite direct VLM (85-class)",
}
MODEL_IDS = {
    "bioclip": "hf-hub:imageomics/bioclip-2.5-vith14",
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "gpt_5_4": "openai/gpt-5.4",
    "gemini_flash_lite": "google/gemini-2.5-flash-lite",
}
MODEL_METHODS = {
    "bioclip": "zero-shot closed-set cosine similarity",
    "dinov3": "frozen CLS embedding + multinomial logistic regression",
    "gpt_5_4": "direct image + 85-species candidate prompt",
    "gemini_flash_lite": "direct image + 85-species candidate prompt",
}
BUCKETS = {
    "all_four_correct": "All four models are Top-1 correct.",
    "all_four_wrong": "All four models are Top-1 wrong.",
    "bioclip_only_correct": "BioCLIP is correct and the other three models are wrong.",
    "bioclip_wrong_vlm_rescue": "BioCLIP is wrong and GPT-5.4 or Gemini is correct.",
    "bioclip_and_dino_wrong": "Both embedding-based baselines are wrong, regardless of VLM outcomes.",
    "embedding_disagreement": "BioCLIP and DINOv3 predict different species.",
    "vlm_embedding_consensus_conflict": "GPT and Gemini agree; BioCLIP and DINO agree; the two groups disagree.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-bucket", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-downloads", action="store_true")
    return parser.parse_args()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    bioclip = pd.read_csv(DOWNLOADS / "bioclip25_strict_predictions.csv")
    dinov3 = pd.read_csv(DOWNLOADS / "dinov3_strict_predictions.csv")
    vlm = pd.read_csv(DOWNLOADS / "openrouter_vlm_predictions.csv")
    metadata_raw = json.loads((ROOT / "app" / "dashboard-data.json").read_text())
    metadata = {row["id"]: row for row in metadata_raw["images"]}

    strict_ids = set(bioclip.image_id)
    vlm = vlm.loc[vlm.image_id.isin(strict_ids)].copy()
    gpt = vlm.loc[vlm.model_key == "gpt_5_4"].copy()
    gemini = vlm.loc[vlm.model_key == "gemini_flash_lite"].copy()

    assert len(bioclip) == len(dinov3) == len(gpt) == len(gemini) == 211
    assert set(bioclip.image_id) == set(dinov3.image_id) == set(gpt.image_id) == set(gemini.image_id)
    assert bioclip.image_id.nunique() == 211
    assert dinov3.individual_id.nunique() == bioclip.individual_id.nunique() == 63
    assert bioclip.ground_truth_species.nunique() == 63
    for frame in (bioclip, dinov3, gpt, gemini):
        assert (frame.status == "ok").all()
        assert set(frame.image_id).issubset(metadata)

    return bioclip, dinov3, vlm, metadata


def model_projection(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    frame = frame.copy()
    keep = [
        "image_id",
        "predicted_species",
        "confidence",
        "top5",
        "correct",
        "correct_top5",
    ]
    projected = frame[keep].rename(
        columns={
            "predicted_species": f"{key}_prediction",
            "confidence": f"{key}_score",
            "top5": f"{key}_top5",
            "correct": f"{key}_correct",
            "correct_top5": f"{key}_top5_correct",
        }
    )
    return projected


def bucket_memberships(row: pd.Series) -> list[str]:
    correct = {key: bool(row[f"{key}_correct"]) for key in MODEL_KEYS}
    prediction = {key: row[f"{key}_prediction"] for key in MODEL_KEYS}
    buckets = []
    if all(correct.values()):
        buckets.append("all_four_correct")
    if not any(correct.values()):
        buckets.append("all_four_wrong")
    if correct["bioclip"] and not any(correct[key] for key in MODEL_KEYS if key != "bioclip"):
        buckets.append("bioclip_only_correct")
    if not correct["bioclip"] and (correct["gpt_5_4"] or correct["gemini_flash_lite"]):
        buckets.append("bioclip_wrong_vlm_rescue")
    if not correct["bioclip"] and not correct["dinov3"]:
        buckets.append("bioclip_and_dino_wrong")
    if prediction["bioclip"] != prediction["dinov3"]:
        buckets.append("embedding_disagreement")
    if (
        prediction["bioclip"] == prediction["dinov3"]
        and prediction["gpt_5_4"] == prediction["gemini_flash_lite"]
        and prediction["bioclip"] != prediction["gpt_5_4"]
    ):
        buckets.append("vlm_embedding_consensus_conflict")
    return buckets


def build_unified_image_table(
    bioclip: pd.DataFrame,
    dinov3: pd.DataFrame,
    vlm: pd.DataFrame,
    metadata: dict[str, dict],
) -> pd.DataFrame:
    base = bioclip[
        ["image_id", "file", "individual_id", "organ_category", "ground_truth_species"]
    ].copy()
    base = base.rename(columns={"individual_id": "observation_id"})
    frames = {
        "bioclip": bioclip,
        "dinov3": dinov3,
        "gpt_5_4": vlm.loc[vlm.model_key == "gpt_5_4"],
        "gemini_flash_lite": vlm.loc[vlm.model_key == "gemini_flash_lite"],
    }
    unified = base
    for key in MODEL_KEYS:
        unified = unified.merge(
            model_projection(frames[key], key),
            on="image_id",
            how="inner",
            validate="one_to_one",
        )

    for column in ["ground_truth_species", "file", "individual_id", "organ_category"]:
        reference = bioclip.set_index("image_id")[column].sort_index()
        for key, frame in frames.items():
            if column in frame:
                candidate = frame.set_index("image_id")[column].sort_index()
                assert reference.index.equals(candidate.index)
                assert (reference == candidate).all(), f"{column} mismatch for {key}"

    unified["vernacular"] = unified.image_id.map(lambda image_id: metadata[image_id]["vernacular"])
    unified["family"] = unified.image_id.map(lambda image_id: metadata[image_id]["family"])
    unified["organ_detail"] = unified.image_id.map(lambda image_id: metadata[image_id]["organ_detail"])
    unified["subview"] = unified.image_id.map(lambda image_id: metadata[image_id]["subview"])
    unified["thumbnail_url"] = unified.image_id.map(lambda image_id: metadata[image_id]["thumbnail"])
    unified["image_url"] = unified.image_id.map(lambda image_id: metadata[image_id]["image"])
    unified["creator"] = unified.image_id.map(lambda image_id: metadata[image_id]["creator"])
    unified["license"] = unified.image_id.map(lambda image_id: metadata[image_id]["license"])
    unified["observation_image_count"] = unified.groupby("observation_id").image_id.transform("size")

    correct_columns = [f"{key}_correct" for key in MODEL_KEYS]
    unified["correct_count"] = unified[correct_columns].sum(axis=1).astype(int)
    unified["correctness_pattern"] = unified.apply(
        lambda row: "".join("1" if row[f"{key}_correct"] else "0" for key in MODEL_KEYS),
        axis=1,
    )
    unified["prediction_unique_count"] = unified.apply(
        lambda row: len({row[f"{key}_prediction"] for key in MODEL_KEYS}), axis=1
    )
    unified["all_predictions_same"] = unified.prediction_unique_count == 1
    unified["embedding_predictions_agree"] = (
        unified.bioclip_prediction == unified.dinov3_prediction
    )
    unified["vlm_predictions_agree"] = (
        unified.gpt_5_4_prediction == unified.gemini_flash_lite_prediction
    )
    unified["bucket_memberships"] = unified.apply(
        lambda row: "|".join(bucket_memberships(row)), axis=1
    )
    return unified.sort_values(["observation_id", "image_id"]).reset_index(drop=True)


def majority_vote(group: pd.DataFrame, model_key: str) -> dict:
    prediction_col = f"{model_key}_prediction"
    score_col = f"{model_key}_score"
    counts = group[prediction_col].value_counts()
    candidates = counts.loc[counts == counts.max()].index.tolist()
    if len(candidates) > 1:
        tie_scores = {
            species: float(group.loc[group[prediction_col] == species, score_col].mean())
            for species in candidates
        }
        winner = sorted(candidates, key=lambda species: (-tie_scores[species], species))[0]
    else:
        winner = candidates[0]
    winner_rows = group.loc[group[prediction_col] == winner]
    correct_col = f"{model_key}_correct"
    return {
        f"{model_key}_observation_prediction": winner,
        f"{model_key}_observation_correct": winner == group.ground_truth_species.iloc[0],
        f"{model_key}_vote_fraction": float(len(winner_rows) / len(group)),
        f"{model_key}_winner_mean_score": float(winner_rows[score_col].mean()),
        f"{model_key}_image_accuracy": float(group[correct_col].mean()),
        f"{model_key}_any_image_correct": bool(group[correct_col].any()),
        f"{model_key}_all_images_correct": bool(group[correct_col].all()),
    }


def build_observation_table(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for observation_id, group in unified.groupby("observation_id", sort=True):
        row = {
            "observation_id": observation_id,
            "ground_truth_species": group.ground_truth_species.iloc[0],
            "vernacular": group.vernacular.iloc[0],
            "family": group.family.iloc[0],
            "image_count": len(group),
            "image_ids": json.dumps(group.image_id.tolist()),
            "organ_categories": json.dumps(sorted(group.organ_category.unique().tolist())),
            "image_urls": json.dumps(group.image_url.tolist()),
        }
        assert group.ground_truth_species.nunique() == 1
        for key in MODEL_KEYS:
            row.update(majority_vote(group, key))
        correct = [bool(row[f"{key}_observation_correct"]) for key in MODEL_KEYS]
        row["observation_correct_count"] = sum(correct)
        row["observation_correctness_pattern"] = "".join("1" if value else "0" for value in correct)
        rows.append(row)
    return pd.DataFrame(rows)


def build_model_summary(unified: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in MODEL_KEYS:
        rows.append(
            {
                "model_key": key,
                "model": MODEL_LABELS[key],
                "model_id": MODEL_IDS[key],
                "method": MODEL_METHODS[key],
                "candidate_species": 63 if key in {"bioclip", "dinov3"} else 85,
                "image_count": len(unified),
                "image_top1_correct": int(unified[f"{key}_correct"].sum()),
                "image_top1_accuracy": float(unified[f"{key}_correct"].mean()),
                "image_top5_correct": int(unified[f"{key}_top5_correct"].sum()),
                "image_top5_accuracy": float(unified[f"{key}_top5_correct"].mean()),
                "observation_count": len(observations),
                "observation_majority_correct": int(observations[f"{key}_observation_correct"].sum()),
                "observation_majority_accuracy": float(observations[f"{key}_observation_correct"].mean()),
                "observations_with_any_correct_image": int(observations[f"{key}_any_image_correct"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_pattern_summary(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pattern, group in unified.groupby("correctness_pattern"):
        row = {
            "correctness_pattern": pattern,
            "bioclip_correct": pattern[0] == "1",
            "dinov3_correct": pattern[1] == "1",
            "gpt_5_4_correct": pattern[2] == "1",
            "gemini_flash_lite_correct": pattern[3] == "1",
            "image_count": len(group),
            "image_fraction": float(len(group) / len(unified)),
            "species_count": int(group.ground_truth_species.nunique()),
            "observation_count": int(group.observation_id.nunique()),
            "example_image_ids": json.dumps(group.image_id.head(5).tolist()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("image_count", ascending=False)


def build_pairwise_overlap(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for first, second in combinations(MODEL_KEYS, 2):
        first_correct = unified[f"{first}_correct"].astype(bool)
        second_correct = unified[f"{second}_correct"].astype(bool)
        both = int((first_correct & second_correct).sum())
        union = int((first_correct | second_correct).sum())
        rows.append(
            {
                "first_model": first,
                "second_model": second,
                "both_correct": both,
                "first_only_correct": int((first_correct & ~second_correct).sum()),
                "second_only_correct": int((~first_correct & second_correct).sum()),
                "both_wrong": int((~first_correct & ~second_correct).sum()),
                "correct_set_jaccard": float(both / union) if union else math.nan,
                "same_top1_prediction": int(
                    (unified[f"{first}_prediction"] == unified[f"{second}_prediction"]).sum()
                ),
                "top1_prediction_agreement": float(
                    (unified[f"{first}_prediction"] == unified[f"{second}_prediction"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_species_accuracy(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for species, group in unified.groupby("ground_truth_species"):
        row = {
            "ground_truth_species": species,
            "image_count": len(group),
            "observation_count": group.observation_id.nunique(),
        }
        for key in MODEL_KEYS:
            row[f"{key}_correct"] = int(group[f"{key}_correct"].sum())
            row[f"{key}_accuracy"] = float(group[f"{key}_correct"].mean())
        row["all_four_wrong"] = int((group.correct_count == 0).sum())
        row["any_model_correct"] = int((group.correct_count > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["all_four_wrong", "image_count"], ascending=[False, False])


def build_organ_accuracy(unified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for organ, group in unified.groupby("organ_category"):
        row = {"organ_category": organ, "image_count": len(group)}
        for key in MODEL_KEYS:
            row[f"{key}_accuracy"] = float(group[f"{key}_correct"].mean())
        row["all_four_wrong_rate"] = float((group.correct_count == 0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("image_count", ascending=False)


def build_confusions(unified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for key in MODEL_KEYS:
        wrong = unified.loc[~unified[f"{key}_correct"]]
        counts = (
            wrong.groupby(["ground_truth_species", f"{key}_prediction"])
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={f"{key}_prediction": "predicted_species"})
        )
        counts["model_key"] = key
        counts["same_genus"] = counts.apply(
            lambda row: row.ground_truth_species.split()[0] == row.predicted_species.split()[0],
            axis=1,
        )
        rows.append(counts)
    confusion = pd.concat(rows, ignore_index=True)
    confusion = confusion.sort_values(
        ["model_key", "count", "ground_truth_species", "predicted_species"],
        ascending=[True, False, True, True],
    )
    shared = (
        confusion.groupby(["ground_truth_species", "predicted_species", "same_genus"])
        .agg(
            model_count=("model_key", "nunique"),
            total_occurrences=("count", "sum"),
            models=("model_key", lambda values: "|".join(sorted(set(values)))),
        )
        .reset_index()
        .sort_values(["model_count", "total_occurrences"], ascending=False)
    )
    return confusion, shared


def export_buckets(unified: pd.DataFrame) -> pd.DataFrame:
    bucket_dir = HERE / "buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bucket, description in BUCKETS.items():
        mask = unified.bucket_memberships.apply(lambda values: bucket in values.split("|"))
        subset = unified.loc[mask].copy()
        subset.to_csv(bucket_dir / f"{bucket}.csv", index=False)
        rows.append(
            {
                "bucket": bucket,
                "description": description,
                "image_count": len(subset),
                "image_fraction": float(len(subset) / len(unified)),
                "observation_count": int(subset.observation_id.nunique()),
                "species_count": int(subset.ground_truth_species.nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("image_count", ascending=False)


def deterministic_bucket_samples(
    unified: pd.DataFrame,
    per_bucket: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for bucket_index, bucket in enumerate(BUCKETS):
        mask = unified.bucket_memberships.apply(lambda values: bucket in values.split("|"))
        subset = unified.loc[mask].copy()
        if subset.empty:
            continue
        rng = random.Random(seed + bucket_index * 1009)
        observation_ids = sorted(subset.observation_id.unique().tolist())
        rng.shuffle(observation_ids)
        chosen = []
        for observation_id in observation_ids:
            candidates = subset.loc[subset.observation_id == observation_id].sort_values("image_id")
            chosen.append(candidates.iloc[rng.randrange(len(candidates))])
            if len(chosen) >= per_bucket:
                break
        if len(chosen) < min(per_bucket, len(subset)):
            used_ids = {row.image_id for row in chosen}
            remaining = subset.loc[~subset.image_id.isin(used_ids)].sort_values("image_id")
            remaining_indices = list(remaining.index)
            rng.shuffle(remaining_indices)
            for index in remaining_indices:
                chosen.append(remaining.loc[index])
                if len(chosen) >= per_bucket:
                    break
        for order, row in enumerate(chosen, start=1):
            record = row.to_dict()
            record["sample_bucket"] = bucket
            record["sample_order"] = order
            rows.append(record)
    columns = ["sample_bucket", "sample_order"] + list(unified.columns)
    return pd.DataFrame(rows)[columns]


def safe_image_name(image_id: str) -> str:
    return image_id.replace("/", "__").replace(" ", "_") + ".jpg"


def download_one(url: str, destination: Path) -> tuple[str, str]:
    if destination.exists() and destination.stat().st_size > 0:
        return "cached", ""
    request = urllib.request.Request(url, headers={"User-Agent": "BioImages error analysis/1.0"})
    last_error = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            if not payload:
                raise ValueError("empty response")
            destination.write_bytes(payload)
            return "downloaded", ""
        except Exception as exc:  # noqa: BLE001 - record per-image download failure.
            last_error = repr(exc)
            time.sleep(2**attempt)
    return "failed", last_error


def download_sample_images(samples: pd.DataFrame) -> pd.DataFrame:
    image_dir = HERE / "samples" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    unique = samples[["image_id", "image_url"]].drop_duplicates("image_id").copy()
    unique["local_sample_path"] = unique.image_id.map(
        lambda image_id: f"samples/images/{safe_image_name(image_id)}"
    )
    statuses = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                download_one,
                row.image_url,
                HERE / row.local_sample_path,
            ): row.image_id
            for row in unique.itertuples(index=False)
        }
        for future in as_completed(futures):
            statuses[futures[future]] = future.result()
    unique["download_status"] = unique.image_id.map(lambda image_id: statuses[image_id][0])
    unique["download_error"] = unique.image_id.map(lambda image_id: statuses[image_id][1])
    return unique


def build_annotation_table(samples: pd.DataFrame) -> pd.DataFrame:
    bucket_map = (
        samples.groupby("image_id").sample_bucket.apply(lambda values: "|".join(sorted(set(values)))).to_dict()
    )
    annotation = samples.drop_duplicates("image_id").copy()
    annotation["selected_for_buckets"] = annotation.image_id.map(bucket_map)
    annotation.insert(0, "annotation_id", [f"EA-{index:04d}" for index in range(1, len(annotation) + 1)])
    for column in [
        "human_identifiable",
        "failure_primary",
        "useful_organ_visible",
        "better_photo_would_help",
        "notes",
    ]:
        annotation[column] = ""
    preferred = [
        "annotation_id",
        "image_id",
        "observation_id",
        "observation_image_count",
        "selected_for_buckets",
        "ground_truth_species",
        "vernacular",
        "family",
        "organ_category",
        "organ_detail",
        "subview",
        "image_url",
        "thumbnail_url",
        "local_sample_path",
        "download_status",
        "bioclip_prediction",
        "bioclip_score",
        "bioclip_correct",
        "dinov3_prediction",
        "dinov3_score",
        "dinov3_correct",
        "gpt_5_4_prediction",
        "gpt_5_4_score",
        "gpt_5_4_correct",
        "gemini_flash_lite_prediction",
        "gemini_flash_lite_score",
        "gemini_flash_lite_correct",
        "correctness_pattern",
        "human_identifiable",
        "failure_primary",
        "useful_organ_visible",
        "better_photo_would_help",
        "notes",
    ]
    return annotation[preferred].sort_values(["selected_for_buckets", "observation_id", "image_id"])


def write_readme(
    model_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    unified: pd.DataFrame,
    observations: pd.DataFrame,
    pairwise: pd.DataFrame,
    species: pd.DataFrame,
    confusion: pd.DataFrame,
    shared_confusion: pd.DataFrame,
    annotation: pd.DataFrame,
) -> None:
    metrics = {row.model_key: row for row in model_summary.itertuples(index=False)}
    bucket_counts = {row.bucket: row.image_count for row in bucket_summary.itertuples(index=False)}
    any_correct = int((unified.correct_count > 0).sum())
    all_wrong = int((unified.correct_count == 0).sum())
    embeddings_wrong_vlm_rescue = int(
        ((~unified.bioclip_correct) & (~unified.dinov3_correct) & (unified.gpt_5_4_correct | unified.gemini_flash_lite_correct)).sum()
    )
    vlms_wrong_embedding_rescue = int(
        ((~unified.gpt_5_4_correct) & (~unified.gemini_flash_lite_correct) & (unified.bioclip_correct | unified.dinov3_correct)).sum()
    )
    hardest = species.sort_values(["all_four_wrong", "image_count"], ascending=False).head(8)
    common_shared = shared_confusion.loc[shared_confusion.model_count >= 2].head(8)
    closest_pair = pairwise.sort_values("correct_set_jaccard", ascending=False).iloc[0]

    confusion_lines = "\n".join(
        f"- *{row.ground_truth_species}* → *{row.predicted_species}*: {row.total_occurrences} occurrences across {row.model_count} models ({row.models})."
        for row in common_shared.itertuples(index=False)
    ) or "- No identical directed error pair appeared in at least two models."
    hard_lines = "\n".join(
        f"- *{row.ground_truth_species}*: {row.all_four_wrong}/{row.image_count} images were missed by all four."
        for row in hardest.itertuples(index=False)
    )
    bucket_lines = "\n".join(
        f"- `{row.bucket}`: {row.image_count} images, {row.observation_count} observations, {row.species_count} species."
        for row in bucket_summary.itertuples(index=False)
    )

    readme = f"""# BioImages four-baseline error analysis

This directory reorganizes existing predictions only. It does not run new inference and does not assign image failure reasons automatically.

## Scope and protocol

- Common analysis set: **{len(unified)} strict test images**, **{len(observations)} observations/individuals**, **{unified.ground_truth_species.nunique()} species**.
- An observation contains 1–{int(unified.observation_image_count.max())} images; both image-level and observation-level tables are retained.
- BioCLIP and DINOv3 use the strict 63-species candidate set. GPT-5.4 and Gemini use the 85-species full candidate set. Accuracy and error overlap are useful, but this is not a perfectly identical candidate-protocol leaderboard.
- Exact models: BioCLIP `{MODEL_IDS['bioclip']}`; DINOv3 `{MODEL_IDS['dinov3']}`; GPT `{MODEL_IDS['gpt_5_4']}`; Gemini `{MODEL_IDS['gemini_flash_lite']}`.
- DINOv3 uses a frozen 768-dimensional CLS embedding and multinomial logistic regression (`C=100`) trained on 1,181 images and selected on 249 validation images. Its 211-image test set is individual/observation-disjoint from training and validation.
- BioCLIP score is a closed-set similarity softmax; DINOv3 score is classifier probability; GPT/Gemini scores are self-reported. Do not compare score magnitudes across model families.
- `observation_predictions.csv` uses majority vote across image-level Top-1 predictions. Ties are broken by mean score among images voting for each tied species.

## Image-level results

| Model | Top-1 | Top-5 | Observation majority Top-1 |
|---|---:|---:|---:|
| BioCLIP zero-shot | {metrics['bioclip'].image_top1_accuracy:.1%} ({metrics['bioclip'].image_top1_correct}/{len(unified)}) | {metrics['bioclip'].image_top5_accuracy:.1%} | {metrics['bioclip'].observation_majority_accuracy:.1%} |
| DINOv3 + linear probe | {metrics['dinov3'].image_top1_accuracy:.1%} ({metrics['dinov3'].image_top1_correct}/{len(unified)}) | {metrics['dinov3'].image_top5_accuracy:.1%} | {metrics['dinov3'].observation_majority_accuracy:.1%} |
| GPT-5.4 | {metrics['gpt_5_4'].image_top1_accuracy:.1%} ({metrics['gpt_5_4'].image_top1_correct}/{len(unified)}) | {metrics['gpt_5_4'].image_top5_accuracy:.1%} | {metrics['gpt_5_4'].observation_majority_accuracy:.1%} |
| Gemini Flash Lite | {metrics['gemini_flash_lite'].image_top1_accuracy:.1%} ({metrics['gemini_flash_lite'].image_top1_correct}/{len(unified)}) | {metrics['gemini_flash_lite'].image_top5_accuracy:.1%} | {metrics['gemini_flash_lite'].observation_majority_accuracy:.1%} |

At least one model is correct on **{any_correct}/{len(unified)} ({any_correct/len(unified):.1%})** images; all four are wrong on **{all_wrong}/{len(unified)} ({all_wrong/len(unified):.1%})**. VLMs rescue {embeddings_wrong_vlm_rescue} images missed by both embedding methods. Embedding methods rescue {vlms_wrong_embedding_rescue} images missed by both VLMs. The most similar correctness sets are `{closest_pair.first_model}` and `{closest_pair.second_model}` (Jaccard {closest_pair.correct_set_jaccard:.2f}).

## Key buckets

{bucket_lines}

The highest-priority human review buckets are:

1. `all_four_wrong`: likely shared visual/data difficulty and the best place to define the failure taxonomy.
2. `bioclip_wrong_vlm_rescue`: tests genuine general-VLM complementarity.
3. `vlm_embedding_consensus_conflict`: strongest representation-family disagreement.
4. `bioclip_only_correct`: shows where biological pretraining contributes unique signal.

## Species concentrated in shared failures

{hard_lines}

## Repeated directed confusion pairs shared by models

{confusion_lines}

See `data/confusion_pairs.csv` for per-model pairs and `data/shared_confusion_pairs.csv` for pairs repeated across models.

## Files

- `error_gallery.html`: self-contained visual gallery for the three priority buckets; all images are embedded.
- `build_error_gallery.py`: reproducible gallery builder using only the existing bucket CSVs and image URLs.
- `data/unified_predictions_image.csv`: one row per image with all four predictions, scores, Top-5 lists, correctness, observation metadata and URLs.
- `data/observation_predictions.csv`: one row per observation with image lists, organ lists, majority-vote predictions and image-level consistency measures.
- `data/model_summary.csv`: image-level and observation-level headline metrics.
- `data/correctness_patterns.csv`: all observed four-bit correctness patterns in model order BioCLIP, DINOv3, GPT, Gemini.
- `data/pairwise_overlap.csv`: pairwise correct-set overlap and prediction agreement.
- `data/per_species_accuracy.csv`: four-model accuracy by species.
- `data/per_organ_accuracy.csv`: four-model accuracy and all-four-wrong rate by organ.
- `data/confusion_pairs.csv`: directed ground-truth → prediction error pairs per model.
- `data/shared_confusion_pairs.csv`: identical directed error pairs shared across models.
- `data/bucket_summary.csv`: size and coverage of each requested bucket.
- `buckets/*.csv`: complete image rows for each (overlapping) bucket.
- `samples/representative_samples.csv`: deterministic random samples, normally at most one image per observation before reuse.
- `samples/images/`: locally downloaded full-resolution sample images.
- `annotation/human_error_annotation.csv`: **{len(annotation)} unique images** ready for manual annotation; requested human fields are blank.
- `annotation/annotation_codebook.csv`: allowed values and definitions for the blank human fields. The failure taxonomy remains intentionally open.
- `summary.json`: machine-readable counts and headline findings.
- `build_error_analysis.py`: reproducible builder. Run `python3 error_analysis/build_error_analysis.py` from the repository root.
"""
    (HERE / "README.md").write_text(readme)


def main() -> None:
    args = parse_args()
    bioclip, dinov3, vlm, metadata = load_inputs()
    unified = build_unified_image_table(bioclip, dinov3, vlm, metadata)
    observations = build_observation_table(unified)
    model_summary = build_model_summary(unified, observations)
    patterns = build_pattern_summary(unified)
    pairwise = build_pairwise_overlap(unified)
    species = build_species_accuracy(unified)
    organs = build_organ_accuracy(unified)
    confusion, shared_confusion = build_confusions(unified)
    bucket_summary = export_buckets(unified)
    samples = deterministic_bucket_samples(unified, args.sample_per_bucket, args.seed)

    data_dir = HERE / "data"
    samples_dir = HERE / "samples"
    annotation_dir = HERE / "annotation"
    for directory in (data_dir, samples_dir, annotation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.skip_downloads:
        downloads = samples[["image_id", "image_url"]].drop_duplicates("image_id").copy()
        downloads["local_sample_path"] = downloads.image_id.map(
            lambda image_id: f"samples/images/{safe_image_name(image_id)}"
        )
        downloads["download_status"] = "not_requested"
        downloads["download_error"] = ""
    else:
        downloads = download_sample_images(samples)
    samples = samples.merge(downloads, on=["image_id", "image_url"], how="left", validate="many_to_one")
    annotation = build_annotation_table(samples)

    unified.to_csv(data_dir / "unified_predictions_image.csv", index=False)
    observations.to_csv(data_dir / "observation_predictions.csv", index=False)
    model_summary.to_csv(data_dir / "model_summary.csv", index=False)
    patterns.to_csv(data_dir / "correctness_patterns.csv", index=False)
    pairwise.to_csv(data_dir / "pairwise_overlap.csv", index=False)
    species.to_csv(data_dir / "per_species_accuracy.csv", index=False)
    organs.to_csv(data_dir / "per_organ_accuracy.csv", index=False)
    confusion.to_csv(data_dir / "confusion_pairs.csv", index=False)
    shared_confusion.to_csv(data_dir / "shared_confusion_pairs.csv", index=False)
    bucket_summary.to_csv(data_dir / "bucket_summary.csv", index=False)
    samples.to_csv(samples_dir / "representative_samples.csv", index=False)
    annotation.to_csv(annotation_dir / "human_error_annotation.csv", index=False)

    codebook = pd.DataFrame(
        [
            {"field": "human_identifiable", "suggested_values": "yes|maybe|no", "definition": "Can a careful human identify the species from this image alone?"},
            {"field": "failure_primary", "suggested_values": "free text for now", "definition": "Primary failure category; intentionally not pre-labeled by a model."},
            {"field": "useful_organ_visible", "suggested_values": "yes|partial|no", "definition": "Is a diagnostically useful plant organ visible?"},
            {"field": "better_photo_would_help", "suggested_values": "yes|maybe|no", "definition": "Would a better angle, scale, focus, or additional view likely help?"},
            {"field": "notes", "suggested_values": "free text", "definition": "Human observations, candidate taxonomy notes, or follow-up questions."},
        ]
    )
    codebook.to_csv(annotation_dir / "annotation_codebook.csv", index=False)

    summary = {
        "seed": args.seed,
        "common_images": len(unified),
        "observations": len(observations),
        "species": int(unified.ground_truth_species.nunique()),
        "protocol_note": "BioCLIP/DINO use 63 candidates; GPT/Gemini use 85 candidates.",
        "model_ids": MODEL_IDS,
        "methods": MODEL_METHODS,
        "dino_split": {
            "train_images": 1181,
            "validation_images": 249,
            "test_images": 211,
            "individual_disjoint": True,
            "embedding_dimension": 768,
            "linear_probe_C": 100.0,
        },
        "models": model_summary.to_dict(orient="records"),
        "buckets": bucket_summary.to_dict(orient="records"),
        "correctness_patterns": patterns.to_dict(orient="records"),
        "any_model_correct_images": int((unified.correct_count > 0).sum()),
        "all_four_wrong_images": int((unified.correct_count == 0).sum()),
        "unique_annotation_images": len(annotation),
        "download_status": downloads.download_status.value_counts().to_dict(),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    write_readme(
        model_summary,
        bucket_summary,
        unified,
        observations,
        pairwise,
        species,
        confusion,
        shared_confusion,
        annotation,
    )

    # Final reconciliation checks.
    assert len(unified) == 211 and unified.image_id.nunique() == 211
    assert len(observations) == 63 and observations.observation_id.nunique() == 63
    assert int(bucket_summary.loc[bucket_summary.bucket == "all_four_wrong", "image_count"].iloc[0]) == int((unified.correct_count == 0).sum())
    assert annotation.image_id.nunique() == len(annotation)
    assert not annotation[["human_identifiable", "failure_primary", "useful_organ_visible", "better_photo_would_help", "notes"]].astype(bool).any().any()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
