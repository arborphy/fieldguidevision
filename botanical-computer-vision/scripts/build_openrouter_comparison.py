#!/usr/bin/env python3
"""Build checked OpenRouter VLM comparison artifacts for the benchmark site."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "bioclip_zero_shot": "BioCLIP 2.5 zero-shot",
    "gemini_flash_lite": "Gemini 2.5 Flash Lite",
    "gpt_5_4": "GPT-5.4",
}


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    """Two-sided exact McNemar p-value without a scipy dependency."""
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def cluster_bootstrap_delta(
    frame: pd.DataFrame,
    first: str,
    second: str,
    metric: str,
    *,
    seed: int = 42,
    draws: int = 10_000,
) -> list[float]:
    """Bootstrap paired deltas by plant individual, retaining within-plant views."""
    wide = frame.pivot(index=["individual_id", "image_id"], columns="model_key", values=metric)
    grouped = {
        individual: values[[first, second]].to_numpy(dtype=float)
        for individual, values in wide.groupby(level="individual_id")
    }
    keys = np.array(list(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=float)
    for index in range(draws):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([grouped[key] for key in sampled], axis=0)
        deltas[index] = rows[:, 0].mean() - rows[:, 1].mean()
    return [float(value) for value in np.quantile(deltas, [0.025, 0.975])]


def model_metrics(frame: pd.DataFrame, model_key: str, run_summary: dict) -> dict:
    model = frame.loc[frame.model_key == model_key].copy()
    latency = model.latency_seconds.astype(float)
    confidences = model.confidence.astype(float)
    metrics = {
        "model": MODEL_LABELS[model_key],
        "model_key": model_key,
        "images": int(len(model)),
        "species": int(model.ground_truth_species.nunique()),
        "successful_images": int((model.status == "ok").sum()),
        "top1_correct": int(model.correct.sum()),
        "top1_accuracy": float(model.correct.mean()),
        "top5_correct": int(model.correct_top5.sum()),
        "top5_accuracy": float(model.correct_top5.mean()),
        "macro_top1_accuracy": float(model.groupby("ground_truth_species").correct.mean().mean()),
        "mean_reported_confidence": float(confidences.mean()),
        "median_latency_seconds": float(latency.median()),
        "p95_latency_seconds": float(latency.quantile(0.95)),
        "mean_latency_seconds": float(latency.mean()),
        "max_latency_seconds": float(latency.max()),
        "retried_images": int((model.attempts > 1).sum()),
        "cached_images": int((model.cached_tokens.fillna(0) > 0).sum()),
        "prompt_tokens": int(model.prompt_tokens.fillna(0).sum()),
        "cached_tokens": int(model.cached_tokens.fillna(0).sum()),
        "completion_tokens": int(model.completion_tokens.fillna(0).sum()),
        "reasoning_tokens": int(model.reasoning_tokens.fillna(0).sum()),
        "actual_cost_usd": float(model.cost_usd.sum()),
        "mean_cost_per_image_usd": float(model.cost_usd.mean()),
    }
    metrics["wall_seconds"] = float(run_summary["models"][model_key]["wall_seconds"])
    return metrics


def comparison(frame: pd.DataFrame, first: str, second: str) -> dict:
    wide = frame.pivot(index="image_id", columns="model_key", values=["correct", "correct_top5"])
    first_top1 = wide[("correct", first)].astype(bool)
    second_top1 = wide[("correct", second)].astype(bool)
    first_top5 = wide[("correct_top5", first)].astype(bool)
    second_top5 = wide[("correct_top5", second)].astype(bool)
    first_only = int((first_top1 & ~second_top1).sum())
    second_only = int((~first_top1 & second_top1).sum())
    return {
        "first": first,
        "second": second,
        "both_correct": int((first_top1 & second_top1).sum()),
        "first_only": first_only,
        "second_only": second_only,
        "both_wrong": int((~first_top1 & ~second_top1).sum()),
        "top1_delta": float(first_top1.mean() - second_top1.mean()),
        "top5_delta": float(first_top5.mean() - second_top5.mean()),
        "top1_cluster_bootstrap_95_ci": cluster_bootstrap_delta(frame, first, second, "correct"),
        "top5_cluster_bootstrap_95_ci": cluster_bootstrap_delta(frame, first, second, "correct_top5", seed=84),
        "exact_mcnemar_p": exact_mcnemar_p(first_only, second_only),
    }


def grouped_comparison(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    result = []
    for group_value, subset in frame.groupby(group, dropna=False):
        row = {group: str(group_value), "total": int(subset.image_id.nunique())}
        for model_key in MODEL_LABELS:
            model = subset.loc[subset.model_key == model_key]
            row[f"{model_key}_top1_accuracy"] = float(model.correct.mean())
            row[f"{model_key}_top5_accuracy"] = float(model.correct_top5.mean())
            row[f"{model_key}_top1_correct"] = int(model.correct.sum())
        result.append(row)
    return pd.DataFrame(result).sort_values(["total", group], ascending=[False, True]).reset_index(drop=True)


def error_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    wrong = frame.loc[~frame.correct].copy()
    return (
        wrong.groupby(["model_key", "ground_truth_species", "predicted_species"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["model_key", "count", "ground_truth_species", "predicted_species"], ascending=[True, False, True, True])
        .reset_index(drop=True)
    )


def confidence_bins(frame: pd.DataFrame) -> list[dict]:
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    labels = ["<50%", "50–59%", "60–69%", "70–79%", "80–89%", "90–100%"]
    rows: list[dict] = []
    for model_key in ["gemini_flash_lite", "gpt_5_4"]:
        model = frame.loc[frame.model_key == model_key].copy()
        model["confidence_bin"] = pd.cut(model.confidence, bins=bins, labels=labels, right=False, include_lowest=True)
        for label in labels:
            subset = model.loc[model.confidence_bin == label]
            if subset.empty:
                continue
            rows.append(
                {
                    "model_key": model_key,
                    "label": label,
                    "total": int(len(subset)),
                    "accuracy": float(subset.correct.mean()),
                    "average_reported_confidence": float(subset.confidence.mean()),
                    "absolute_gap": float(abs(subset.confidence.mean() - subset.correct.mean())),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openrouter", type=Path, default=Path("outputs/openrouter_vlm_full/predictions.csv"))
    parser.add_argument("--run-summary", type=Path, default=Path("outputs/openrouter_vlm_full/summary.json"))
    parser.add_argument("--bioclip", type=Path, default=Path("public/downloads/bioclip25_predictions.csv"))
    parser.add_argument("--site-data", type=Path, default=Path("app/openrouter-comparison-data.json"))
    parser.add_argument("--downloads", type=Path, default=Path("public/downloads"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vlm = pd.read_csv(args.openrouter)
    run_summary = json.loads(args.run_summary.read_text())
    bioclip = pd.read_csv(args.bioclip)

    expected_models = {"gemini_flash_lite", "gpt_5_4"}
    assert set(vlm.model_key) == expected_models
    assert len(vlm) == 3_798
    assert vlm.image_id.nunique() == 1_899
    assert not vlm.duplicated(["model_key", "image_id"]).any()
    assert (vlm.status == "ok").all()
    assert vlm.ground_truth_species.nunique() == 85
    assert len(bioclip) == 1_899 and bioclip.image_id.nunique() == 1_899
    assert set(bioclip.image_id) == set(vlm.image_id)

    metadata = vlm.loc[vlm.model_key == "gemini_flash_lite", ["image_id", "individual_id", "organ_category"]]
    bioclip = bioclip.merge(metadata, on="image_id", how="left", validate="one_to_one")
    bioclip["model_key"] = "bioclip_zero_shot"
    bioclip["correct"] = bioclip.correct.astype(bool)
    bioclip["correct_top5"] = bioclip.correct_top5.astype(bool)
    vlm["correct"] = vlm.correct.astype(bool)
    vlm["correct_top5"] = vlm.correct_top5.astype(bool)

    common = [
        "image_id", "individual_id", "organ_category", "model_key", "ground_truth_species",
        "predicted_species", "confidence", "correct", "correct_top5",
    ]
    combined = pd.concat([bioclip[common], vlm[common]], ignore_index=True)

    species = grouped_comparison(combined, "ground_truth_species")
    organs = grouped_comparison(combined, "organ_category")
    errors = error_pairs(combined)

    summary = {
        "protocol": {
            "images": 1_899,
            "species": 85,
            "individuals": int(vlm.individual_id.nunique()),
            "candidate_set": "85 scientific names with vernacular aliases",
            "ground_truth_hidden": True,
            "ground_truth_isolation": run_summary["ground_truth_isolation"],
            "image_preprocessing": run_summary["image_preprocessing"],
            "concurrency": int(run_summary["concurrency"]),
            "seed": int(run_summary["seed"]),
        },
        "models": {
            "bioclip_zero_shot": {
                "model": MODEL_LABELS["bioclip_zero_shot"],
                "model_key": "bioclip_zero_shot",
                "images": 1_899,
                "species": 85,
                "successful_images": 1_899,
                "top1_correct": int(bioclip.correct.sum()),
                "top1_accuracy": float(bioclip.correct.mean()),
                "top5_correct": int(bioclip.correct_top5.sum()),
                "top5_accuracy": float(bioclip.correct_top5.mean()),
                "macro_top1_accuracy": float(bioclip.groupby("ground_truth_species").correct.mean().mean()),
            },
            "gemini_flash_lite": model_metrics(vlm, "gemini_flash_lite", run_summary),
            "gpt_5_4": model_metrics(vlm, "gpt_5_4", run_summary),
        },
        "pairwise": {
            "gpt_vs_gemini": comparison(combined, "gpt_5_4", "gemini_flash_lite"),
            "gpt_vs_bioclip": comparison(combined, "gpt_5_4", "bioclip_zero_shot"),
            "gemini_vs_bioclip": comparison(combined, "gemini_flash_lite", "bioclip_zero_shot"),
        },
        "run": {
            "total_actual_cost_usd": float(run_summary["total_actual_cost_usd"]),
            "total_run_seconds": float(run_summary["total_run_seconds"]),
            "dataset_download_and_extract_seconds": float(run_summary["dataset_download_and_extract_seconds"]),
        },
        "confidence_note": "VLM confidence is a self-reported score requested in the response schema. It is not a normalized model probability; the five scores need not sum to one.",
    }

    site_data = {
        "summary": summary,
        "organs": organs.to_dict(orient="records"),
        "species": species.to_dict(orient="records"),
        "errors": errors.groupby("model_key").head(15).to_dict(orient="records"),
        "confidence": confidence_bins(vlm),
    }

    args.site_data.parent.mkdir(parents=True, exist_ok=True)
    args.downloads.mkdir(parents=True, exist_ok=True)
    args.site_data.write_text(json.dumps(site_data, indent=2, ensure_ascii=False) + "\n")
    vlm.to_csv(args.downloads / "openrouter_vlm_predictions.csv", index=False)
    species.to_csv(args.downloads / "openrouter_vlm_per_species.csv", index=False)
    organs.to_csv(args.downloads / "openrouter_vlm_per_organ.csv", index=False)
    errors.to_csv(args.downloads / "openrouter_vlm_error_pairs.csv", index=False)
    (args.downloads / "openrouter_vlm_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.downloads / "openrouter_vlm_candidate_species.json").write_text(
        (args.openrouter.parent / "candidate_species.json").read_text()
    )

    # Independent reconciliation against runner output and expected row counts.
    assert summary["models"]["gemini_flash_lite"]["top1_correct"] == run_summary["models"]["gemini_flash_lite"]["top1_correct"]
    assert summary["models"]["gpt_5_4"]["top5_correct"] == run_summary["models"]["gpt_5_4"]["top5_correct"]
    assert len(species) == 85
    assert int(species.total.sum()) == 1_899
    assert int(organs.total.sum()) == 1_899
    assert len(pd.read_csv(args.downloads / "openrouter_vlm_predictions.csv")) == 3_798

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
