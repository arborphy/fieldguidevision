#!/usr/bin/env python3
"""Re-score existing four-baseline predictions at species, genus, and family levels."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "error_analysis" / "data" / "unified_predictions_image.csv"
METADATA = ROOT / "app" / "dashboard-data.json"

MODELS = ["bioclip", "dinov3", "gpt_5_4", "gemini_flash_lite"]
MODEL_LABELS = {
    "bioclip": "BioCLIP 2.5 zero-shot",
    "dinov3": "DINOv3 frozen + linear probe",
    "gpt_5_4": "GPT-5.4 direct VLM",
    "gemini_flash_lite": "Gemini 2.5 Flash Lite direct VLM",
}
MODEL_CANDIDATES = {
    "bioclip": 63,
    "dinov3": 63,
    "gpt_5_4": 85,
    "gemini_flash_lite": 85,
}


def genus(species: str) -> str:
    parts = str(species).strip().split()
    if len(parts) < 2:
        raise ValueError(f"Expected a binomial species name, got {species!r}")
    return parts[0]


def load_family_mapping() -> dict[str, str]:
    records = json.loads(METADATA.read_text(encoding="utf-8"))["images"]
    observed: dict[str, set[str]] = {}
    for record in records:
        observed.setdefault(record["truth"], set()).add(record["family"])

    conflicts = {species: values for species, values in observed.items() if len(values) != 1}
    if conflicts:
        raise ValueError(f"Conflicting family metadata: {conflicts}")
    mapping = {species: next(iter(values)) for species, values in observed.items()}
    if any(not family for family in mapping.values()):
        raise ValueError("Blank family value in existing metadata")
    return mapping


def build_taxonomy_mapping(source: pd.DataFrame, families: dict[str, str]) -> pd.DataFrame:
    ground_truth = set(source.ground_truth_species)
    predictions = {
        species
        for model in MODELS
        for species in source[f"{model}_prediction"].dropna().astype(str)
    }
    all_species = sorted(ground_truth | predictions)
    missing = sorted(set(all_species) - set(families))
    if missing:
        raise ValueError(f"Family metadata missing for: {missing}")
    return pd.DataFrame(
        [
            {
                "species": species,
                "genus": genus(species),
                "family": families[species],
                "appears_as_ground_truth": species in ground_truth,
                "appears_as_prediction": species in predictions,
            }
            for species in all_species
        ]
    )


def build_image_table(
    source: pd.DataFrame,
    taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    genus_map = taxonomy.set_index("species").genus.to_dict()
    family_map = taxonomy.set_index("species").family.to_dict()

    table = source[
        [
            "image_id",
            "observation_id",
            "file",
            "ground_truth_species",
            "organ_category",
            "organ_detail",
            "subview",
            "image_url",
        ]
    ].copy()
    table["ground_truth_genus"] = table.ground_truth_species.map(genus_map)
    table["ground_truth_family"] = table.ground_truth_species.map(family_map)

    for model in MODELS:
        prediction = source[f"{model}_prediction"]
        table[f"{model}_predicted_species"] = prediction
        table[f"{model}_predicted_genus"] = prediction.map(genus_map)
        table[f"{model}_predicted_family"] = prediction.map(family_map)
        table[f"{model}_score"] = source[f"{model}_score"]
        table[f"{model}_species_correct"] = (
            table[f"{model}_predicted_species"] == table.ground_truth_species
        )
        table[f"{model}_genus_correct"] = (
            table[f"{model}_predicted_genus"] == table.ground_truth_genus
        )
        table[f"{model}_family_correct"] = (
            table[f"{model}_predicted_family"] == table.ground_truth_family
        )

    required = [
        "ground_truth_genus",
        "ground_truth_family",
        *[f"{model}_predicted_genus" for model in MODELS],
        *[f"{model}_predicted_family" for model in MODELS],
    ]
    if table[required].isna().any().any():
        raise ValueError("Unexpected missing taxonomy mapping")
    return table


def build_accuracy(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for level in ["species", "genus", "family"]:
            values = table[f"{model}_{level}_correct"]
            rows.append(
                {
                    "model_key": model,
                    "model": MODEL_LABELS[model],
                    "candidate_species": MODEL_CANDIDATES[model],
                    "taxonomy_level": level,
                    "correct": int(values.sum()),
                    "total": len(table),
                    "accuracy": float(values.mean()),
                }
            )
    return pd.DataFrame(rows)


def build_error_breakdown(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        errors = table.loc[~table[f"{model}_species_correct"]].copy()
        within_genus = errors[f"{model}_genus_correct"]
        wrong_genus = ~within_genus
        correct_family_wrong_genus = wrong_genus & errors[f"{model}_family_correct"]
        wrong_family = ~errors[f"{model}_family_correct"]
        total = len(errors)
        rows.append(
            {
                "model_key": model,
                "model": MODEL_LABELS[model],
                "species_error_count": total,
                "correct_genus_wrong_species": int(within_genus.sum()),
                "correct_genus_wrong_species_fraction": float(within_genus.mean()),
                "wrong_genus": int(wrong_genus.sum()),
                "wrong_genus_fraction": float(wrong_genus.mean()),
                "correct_family_wrong_genus": int(correct_family_wrong_genus.sum()),
                "correct_family_wrong_genus_fraction": float(correct_family_wrong_genus.mean()),
                "wrong_family": int(wrong_family.sum()),
                "wrong_family_fraction": float(wrong_family.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_within_genus_confusions(table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model in MODELS:
        mask = ~table[f"{model}_species_correct"] & table[f"{model}_genus_correct"]
        errors = table.loc[mask].copy()
        for row in errors.itertuples(index=False):
            records.append(
                {
                    "model_key": model,
                    "genus": row.ground_truth_genus,
                    "ground_truth_species": row.ground_truth_species,
                    "predicted_species": getattr(row, f"{model}_predicted_species"),
                    "image_id": row.image_id,
                    "observation_id": row.observation_id,
                    "organ_category": row.organ_category,
                }
            )

    long = pd.DataFrame(records)
    pair_keys = ["genus", "ground_truth_species", "predicted_species"]
    summary = (
        long.groupby(pair_keys)
        .agg(
            total_occurrences=("image_id", "size"),
            unique_image_count=("image_id", "nunique"),
            observation_count=("observation_id", "nunique"),
            model_count=("model_key", "nunique"),
            models=("model_key", lambda values: "|".join(sorted(set(values)))),
            organ_categories=("organ_category", lambda values: "|".join(sorted(set(values)))),
            example_image_ids=("image_id", lambda values: "|".join(sorted(set(values))[:5])),
        )
        .reset_index()
    )
    counts = (
        long.pivot_table(
            index=pair_keys,
            columns="model_key",
            values="image_id",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
    )
    for model in MODELS:
        if model not in counts:
            counts[model] = 0
    counts = counts.rename(columns={model: f"{model}_count" for model in MODELS})
    result = summary.merge(counts, on=pair_keys, how="left", validate="one_to_one")
    ordered = [
        *pair_keys,
        "total_occurrences",
        "unique_image_count",
        "observation_count",
        "model_count",
        "models",
        *[f"{model}_count" for model in MODELS],
        "organ_categories",
        "example_image_ids",
    ]
    return result[ordered].sort_values(
        ["total_occurrences", "model_count", "genus", "ground_truth_species", "predicted_species"],
        ascending=[False, False, True, True, True],
    )


def write_readme(
    taxonomy: pd.DataFrame,
    accuracy: pd.DataFrame,
    breakdown: pd.DataFrame,
    confusions: pd.DataFrame,
) -> None:
    metric = {
        (row.model_key, row.taxonomy_level): row
        for row in accuracy.itertuples(index=False)
    }
    metric_lines = []
    for model in MODELS:
        metric_lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{metric[(model, 'species')].accuracy:.1%} ({metric[(model, 'species')].correct}/211) | "
            f"{metric[(model, 'genus')].accuracy:.1%} ({metric[(model, 'genus')].correct}/211) | "
            f"{metric[(model, 'family')].accuracy:.1%} ({metric[(model, 'family')].correct}/211) |"
        )

    error_lines = []
    for row in breakdown.itertuples(index=False):
        error_lines.append(
            f"| {row.model} | {row.species_error_count} | "
            f"{row.correct_genus_wrong_species} ({row.correct_genus_wrong_species_fraction:.1%}) | "
            f"{row.wrong_genus} ({row.wrong_genus_fraction:.1%}) |"
        )

    common = confusions.head(12)
    confusion_lines = [
        f"- *{row.ground_truth_species}* → *{row.predicted_species}*: "
        f"{row.total_occurrences} prediction occurrences across {row.model_count} model(s) "
        f"({row.unique_image_count} unique image(s))"
        for row in common.itertuples(index=False)
    ]

    content = f"""# Taxonomy-level error analysis

This analysis re-scores the existing 211 aligned predictions only. It does not run inference and does not modify the original species-level benchmark.

## Mapping coverage

- Ground-truth species: 63
- Species appearing as ground truth or prediction: {len(taxonomy)}
- Genus mapping: first token of each validated binomial name
- Family mapping: existing `app/dashboard-data.json` metadata
- Family coverage: {taxonomy.family.notna().sum()}/{len(taxonomy)}, with no missing values or conflicts
- BioCLIP/DINO use 63 candidate species; GPT/Gemini use 85 candidate species. The taxonomy re-scoring preserves those original predictions.

## Top-1 accuracy

| Model | Species | Genus | Family |
|---|---:|---:|---:|
{chr(10).join(metric_lines)}

## Species-level error decomposition

Percentages below use each model's species-level errors as the denominator.

| Model | Species errors | Correct genus, wrong species | Wrong genus |
|---|---:|---:|---:|
{chr(10).join(error_lines)}

## Most frequent within-genus confusions

{chr(10).join(confusion_lines)}

## Output files

- `taxonomy_mapping.csv`: species-to-genus-to-family mapping and where each species appears.
- `taxonomy_predictions_image.csv`: image-level ground truth and four predictions at species, genus and family levels.
- `taxonomy_level_accuracy.csv`: Top-1 accuracy for every model and taxonomy level.
- `species_error_taxonomy_breakdown.csv`: correct-genus/wrong-species, wrong-genus and family-level decomposition.
- `within_genus_confusions.csv`: directed within-genus confusion pairs aggregated across models, with per-model counts.
"""
    (HERE / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    source = pd.read_csv(SOURCE)
    if len(source) != 211 or source.image_id.nunique() != 211:
        raise ValueError("Expected the existing 211-image aligned prediction table")

    families = load_family_mapping()
    taxonomy = build_taxonomy_mapping(source, families)
    table = build_image_table(source, taxonomy)
    accuracy = build_accuracy(table)
    breakdown = build_error_breakdown(table)
    confusions = build_within_genus_confusions(table)

    taxonomy.to_csv(HERE / "taxonomy_mapping.csv", index=False)
    table.to_csv(HERE / "taxonomy_predictions_image.csv", index=False)
    accuracy.to_csv(HERE / "taxonomy_level_accuracy.csv", index=False)
    breakdown.to_csv(HERE / "species_error_taxonomy_breakdown.csv", index=False)
    confusions.to_csv(HERE / "within_genus_confusions.csv", index=False)
    write_readme(taxonomy, accuracy, breakdown, confusions)

    # Reconciliation checks.
    for model in MODELS:
        source_species_correct = source[f"{model}_correct"].astype(bool)
        assert source_species_correct.equals(table[f"{model}_species_correct"])
        assert (table[f"{model}_species_correct"] <= table[f"{model}_genus_correct"]).all()
        assert (table[f"{model}_genus_correct"] <= table[f"{model}_family_correct"]).all()
        row = breakdown.loc[breakdown.model_key == model].iloc[0]
        assert row.correct_genus_wrong_species + row.wrong_genus == row.species_error_count
        assert int((~table[f"{model}_species_correct"]).sum()) == row.species_error_count

    print(accuracy.to_string(index=False))
    print("\nSpecies error decomposition")
    print(breakdown.to_string(index=False))
    print("\nTop within-genus confusions")
    print(confusions.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
