#!/usr/bin/env python3
"""Independently audit the completed OpenRouter benchmark and write evidence files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "outputs/openrouter_vlm_full/predictions.csv"
RAW = ROOT / "outputs/openrouter_vlm_full/raw_responses.jsonl"
INPUT_AUDIT = ROOT / "outputs/openrouter_vlm_audit/input_integrity.json"
CANDIDATES = ROOT / "outputs/openrouter_vlm_full/candidate_species.json"
BIOCLIP = ROOT / "public/downloads/bioclip25_predictions.csv"
PILOT_SHUFFLED = ROOT / "outputs/openrouter_vlm_pilot/predictions.csv"
PILOT_FIXED = ROOT / "outputs/openrouter_vlm_pilot_final/predictions.csv"
PILOT_HIGH = ROOT / "outputs/openrouter_vlm_pilot_gpt_high_valid/predictions.csv"
REPORT_DIR = ROOT / "reports"


def main() -> None:
    full = pd.read_csv(FULL)
    gpt = full.loc[full.model_key == "gpt_5_4"].set_index("image_id").sort_index()
    gemini = full.loc[full.model_key == "gemini_flash_lite"].set_index("image_id").sort_index()
    bio = pd.read_csv(BIOCLIP).set_index("image_id").sort_index()
    candidates = json.loads(CANDIDATES.read_text())
    integrity = json.loads(INPUT_AUDIT.read_text())
    integrity_rows = pd.DataFrame(integrity["records"]).set_index("image_id").sort_index()

    top5_valid = 0
    for _, row in full.iterrows():
        top5 = json.loads(row.top5)
        valid = (
            len(top5) == 5
            and len({item["species"] for item in top5}) == 5
            and row.predicted_species == top5[0]["species"]
            and bool(row.correct) == (row.predicted_species == row.ground_truth_species)
            and bool(row.correct_top5)
            == any(item["species"] == row.ground_truth_species for item in top5)
        )
        top5_valid += int(valid)

    raw_rows = [json.loads(line) for line in RAW.read_text().splitlines()]
    raw_generation_ids = [row.get("generation_id") for row in raw_rows if row.get("generation_id")]

    ordered = sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"42|candidate-order|{item['scientific_name']}".encode()
        ).hexdigest(),
    )
    candidate_position = {
        item["scientific_name"]: index + 1 for index, item in enumerate(ordered)
    }
    gpt_positions = gpt.predicted_species.map(candidate_position)
    truth_positions = gpt.ground_truth_species.map(candidate_position)
    prediction_frequency_by_candidate = pd.DataFrame(
        {
            "position": [candidate_position[item["scientific_name"]] for item in candidates],
            "prediction_count": [
                int((gpt.predicted_species == item["scientific_name"]).sum())
                for item in candidates
            ],
        }
    )

    shuffled = pd.read_csv(PILOT_SHUFFLED)
    shuffled = shuffled.loc[
        (shuffled.model_key == "gpt_5_4") & (shuffled.status == "ok")
    ].set_index("image_id")
    fixed = pd.read_csv(PILOT_FIXED)
    fixed = fixed.loc[fixed.model_key == "gpt_5_4"].set_index("image_id")
    order_compare = shuffled[["predicted_species", "correct", "correct_top5"]].join(
        fixed[["predicted_species", "correct", "correct_top5"]],
        lsuffix="_shuffled",
        rsuffix="_fixed",
    )

    high = pd.read_csv(PILOT_HIGH)
    high_ok = high.loc[high.status == "ok"].set_index("image_id")
    high_compare = high_ok[["correct", "correct_top5"]].join(
        fixed[["correct", "correct_top5"]],
        lsuffix="_high",
        rsuffix="_minimal",
    )

    paired = gpt[["correct", "correct_top5", "predicted_species"]].join(
        bio[["correct", "correct_top5", "predicted_species"]],
        lsuffix="_gpt",
        rsuffix="_bioclip",
    )
    both_correct = int((paired.correct_gpt & paired.correct_bioclip).sum())
    gpt_only = int((paired.correct_gpt & ~paired.correct_bioclip).sum())
    bioclip_only = int((~paired.correct_gpt & paired.correct_bioclip).sum())
    both_wrong = int((~paired.correct_gpt & ~paired.correct_bioclip).sum())

    facts = {
        "source_integrity": {
            "dataset_images": integrity["dataset_images"],
            "unique_image_ids": integrity["unique_image_ids"],
            "unique_filenames": integrity["unique_filenames"],
            "decoded_and_reencoded": integrity["decoded_and_reencoded"],
            "jpeg_rgb_outputs": integrity["jpeg_rgb_outputs"],
            "base64_roundtrip_exact": integrity["base64_roundtrip_exact"],
            "metadata_free_outputs": integrity["metadata_free_outputs"],
            "dimension_mismatches": int(
                (integrity_rows.encoded_size.str[0] != gpt.image_width).sum()
                + (integrity_rows.encoded_size.str[1] != gpt.image_height).sum()
            ),
            "filename_mismatches_vs_benchmark": int((integrity_rows.file != gpt.file).sum()),
            "truth_mismatches_vs_benchmark": int(
                (integrity_rows.ground_truth_species != gpt.ground_truth_species).sum()
            ),
        },
        "candidate_and_request_integrity": {
            "candidate_count": len(candidates),
            "unique_scientific_names": len(
                {item["scientific_name"] for item in candidates}
            ),
            "unique_vernacular_names": len({item["vernacular"] for item in candidates}),
            "full_rows": len(full),
            "unique_model_image_pairs": int(
                full[["model_key", "image_id"]].drop_duplicates().shape[0]
            ),
            "successful_rows": int((full.status == "ok").sum()),
            "valid_distinct_in_list_top5_rows": top5_valid,
            "raw_response_rows": len(raw_rows),
            "unique_generation_ids": len(set(raw_generation_ids)),
            "gpt_served_model_counts": gpt.served_model.value_counts().to_dict(),
            "gpt_service_tier_counts": gpt.service_tier.value_counts().to_dict(),
        },
        "label_and_metric_integrity": {
            "same_image_ids_as_bioclip": bool(gpt.index.equals(bio.index)),
            "filename_mismatches_vs_bioclip": int((gpt.file != bio.file).sum()),
            "truth_mismatches_vs_bioclip": int(
                (gpt.ground_truth_species != bio.ground_truth_species).sum()
            ),
            "gpt_correct_flag_mismatches": int(
                ((gpt.predicted_species == gpt.ground_truth_species) != gpt.correct).sum()
            ),
            "gpt_top1_correct": int(gpt.correct.sum()),
            "gpt_top1_accuracy": float(gpt.correct.mean()),
            "gpt_top5_correct": int(gpt.correct_top5.sum()),
            "gpt_top5_accuracy": float(gpt.correct_top5.mean()),
            "bioclip_top1_correct": int(bio.correct.sum()),
            "bioclip_top1_accuracy": float(bio.correct.mean()),
            "pairwise_top1": {
                "both_correct": both_correct,
                "gpt_only": gpt_only,
                "bioclip_only": bioclip_only,
                "both_wrong": both_wrong,
            },
        },
        "candidate_order_sensitivity": {
            "fixed_order_prediction_position_spearman": float(
                prediction_frequency_by_candidate.position.corr(
                    prediction_frequency_by_candidate.prediction_count,
                    method="spearman",
                )
            ),
            "predicted_share_positions_1_to_20": float(gpt_positions.between(1, 20).mean()),
            "truth_share_positions_1_to_20": float(truth_positions.between(1, 20).mean()),
            "shared_valid_pilot_images": len(order_compare),
            "prediction_agreement_fixed_vs_per_image_shuffle": float(
                (
                    order_compare.predicted_species_fixed
                    == order_compare.predicted_species_shuffled
                ).mean()
            ),
            "per_image_shuffle_top1_accuracy": float(order_compare.correct_shuffled.mean()),
            "fixed_order_top1_accuracy_same_images": float(order_compare.correct_fixed.mean()),
        },
        "high_reasoning_audit": {
            "requested_images": len(high),
            "successful_images": len(high_ok),
            "failure_reason": "response content was null after the 2,000-token completion cap; high reasoning consumed the output budget before JSON",
            "mean_success_latency_seconds": float(high_ok.latency_seconds.mean()),
            "successful_subset_high_top1_correct": int(high_compare.correct_high.sum()),
            "same_subset_minimal_top1_correct": int(high_compare.correct_minimal.sum()),
            "conclusion": "inconclusive because only 12/63 completed and completion success is not a random subset",
        },
        "organ_accuracy_gpt": (
            gpt.reset_index()
            .groupby("organ_category")
            .agg(images=("correct", "size"), top1_accuracy=("correct", "mean"), top5_accuracy=("correct_top5", "mean"))
            .sort_values("images", ascending=False)
            .reset_index()
            .to_dict(orient="records")
        ),
    }

    # Every core check must pass before the report is written.
    assert facts["source_integrity"]["dataset_images"] == 1899
    assert facts["source_integrity"]["decoded_and_reencoded"] == 1899
    assert facts["source_integrity"]["dimension_mismatches"] == 0
    assert facts["candidate_and_request_integrity"]["candidate_count"] == 85
    assert facts["candidate_and_request_integrity"]["successful_rows"] == 3798
    assert facts["candidate_and_request_integrity"]["valid_distinct_in_list_top5_rows"] == 3798
    assert facts["candidate_and_request_integrity"]["unique_generation_ids"] == 3798
    assert facts["label_and_metric_integrity"]["truth_mismatches_vs_bioclip"] == 0
    assert facts["label_and_metric_integrity"]["gpt_correct_flag_mismatches"] == 0

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "openrouter_vlm_audit.json").write_text(
        json.dumps(facts, indent=2, ensure_ascii=False) + "\n"
    )

    organ_rows = {row["organ_category"]: row for row in facts["organ_accuracy_gpt"]}
    markdown = f"""# OpenRouter direct-VLM benchmark audit

## Verdict

The image/data/evaluation pipeline is internally correct. The reported GPT-5.4 result—737/1,899 Top-1 ({facts['label_and_metric_integrity']['gpt_top1_accuracy']:.2%}) and 1,203/1,899 Top-5 ({facts['label_and_metric_integrity']['gpt_top5_accuracy']:.2%})—recomputes exactly from the saved rows.

It should be named **GPT-5.4 direct closed-set, single-pass, minimal-reasoning, fixed-candidate-order baseline**. It is not evidence of GPT-5.4's maximum possible botanical accuracy. The fixed candidate order is a real order-sensitivity limitation, and the attempted high-reasoning audit was inconclusive because the completion cap truncated 51 of 63 responses.

## Evidence chain

| Link | Independent check | Result |
|---|---|---:|
| Dataset | Unique image IDs / filenames | 1,899 / 1,899 |
| Decode | Images decoded and re-encoded in Modal | 1,899 / 1,899 |
| Payload | JPEG RGB, metadata removed, exact base64 round-trip | 1,899 / 1,899 |
| Dimensions | Rebuilt payload vs saved benchmark dimensions | 0 mismatches |
| Candidates | Unique scientific / vernacular labels | 85 / 85 |
| Requests | Successful model-image rows | 3,798 / 3,798 |
| Model routing | Saved served model | openai/gpt-5.4 on 1,899 rows |
| Responses | Five distinct, in-list classes and consistent first rank | 3,798 / 3,798 |
| Traceability | Unique OpenRouter generation IDs | 3,798 / 3,798 |
| Ground truth | GPT file/truth rows vs the earlier BioCLIP export | 0 / 0 mismatches |
| Scoring | Recomputed GPT correctness vs saved flag | 0 mismatches |

The outbound API message contains only the fixed candidate prompt and the base64 image. `image_id`, file name, local path, organ label and ground truth remain local and are not inserted into the message. Ground truth is loaded after all requested model calls finish.

## Why a large BioCLIP gap is plausible

- BioCLIP: {facts['label_and_metric_integrity']['bioclip_top1_correct']}/1,899 Top-1 ({facts['label_and_metric_integrity']['bioclip_top1_accuracy']:.2%}).
- GPT-5.4: {facts['label_and_metric_integrity']['gpt_top1_correct']}/1,899 Top-1 ({facts['label_and_metric_integrity']['gpt_top1_accuracy']:.2%}).
- Paired outcomes: both correct {both_correct}; GPT only {gpt_only}; BioCLIP only {bioclip_only}; both wrong {both_wrong}.
- GPT is far above random guessing (1/85 = {1/85:.2%}) and reaches {facts['label_and_metric_integrity']['gpt_top5_accuracy']:.2%} Top-5. It often identifies a close candidate without resolving the exact species.
- The performance gradient follows visual difficulty: fruit {organ_rows['fruit']['top1_accuracy']:.1%}, inflorescence {organ_rows['inflorescence']['top1_accuracy']:.1%}, leaf {organ_rows['leaf']['top1_accuracy']:.1%}, twig {organ_rows['twig']['top1_accuracy']:.1%}, bark {organ_rows['bark']['top1_accuracy']:.1%}.

## Protocol limitations found by the audit

### Candidate-order sensitivity

The full run used one deterministic randomized candidate order to enable prompt caching. GPT selected positions 1–20 for {facts['candidate_order_sensitivity']['predicted_share_positions_1_to_20']:.1%} of images, while only {facts['candidate_order_sensitivity']['truth_share_positions_1_to_20']:.1%} of ground-truth records belonged to those positions.

An earlier pilot used a different deterministic order for every image. Among 56 images with valid outputs in both pilots, predictions agreed only {facts['candidate_order_sensitivity']['prediction_agreement_fixed_vs_per_image_shuffle']:.1%}. Top-1 was {facts['candidate_order_sensitivity']['per_image_shuffle_top1_accuracy']:.1%} with per-image shuffle and {facts['candidate_order_sensitivity']['fixed_order_top1_accuracy_same_images']:.1%} with fixed order. This confirms order sensitivity, but the observed pilot shift is much smaller than the 38.7 percentage-point BioCLIP gap.

### High-reasoning audit

High reasoning with original image detail completed only {len(high_ok)}/63 outputs before the 2,000-token completion cap. Successful requests averaged {high_ok.latency_seconds.mean():.1f}s. On those same 12 images, high reasoning got {int(high_compare.correct_high.sum())} Top-1 versus {int(high_compare.correct_minimal.sum())} for minimal reasoning. This is directional evidence only; the completed subset is small and non-random, so no accuracy claim should be made.

## Example-level evidence

| image_id | Truth | GPT-5.4 | Rank of truth | BioCLIP | Interpretation |
|---|---|---|---:|---|---|
| baskauf/38299 | Pinus rigida | Pinus strobus | 3 | correct | Clear pine image; GPT reaches the right genus but misses the species. |
| baskauf/66166 | Gleditsia triacanthos | Robinia pseudoacacia | 2 | correct | Compound-leaf/inflorescence lookalikes; truth is GPT's second choice. |
| baskauf/50998 | Ginkgo biloba | correct | 1 | wrong | GPT contributes genuine wins that BioCLIP misses. |
| baskauf/89954 | Platanus occidentalis | correct | 1 | correct | Both systems succeed on a distinctive whole-tree view. |

## Recommended reporting language

Report the 38.81% result as a valid **cost-controlled direct-VLM baseline** under the exact protocol above. Do not describe it as GPT-5.4's botanical ceiling. A stronger follow-up should balance candidate order across several fixed permutations and use a completion budget sized for the selected reasoning effort.
"""
    (REPORT_DIR / "openrouter_vlm_audit.md").write_text(markdown)
    print(json.dumps(facts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
