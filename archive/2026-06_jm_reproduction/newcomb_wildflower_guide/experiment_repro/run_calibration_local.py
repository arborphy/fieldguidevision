from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from model_adapter import call_model
from runner_common import (
    OUTPUT_DIR,
    PRIMARY_FEATURES,
    agreement_parts,
    blind_mc_parts,
    build_options,
    existence_parts,
    load_inputs,
    parse_mc,
    parse_ync,
)


MODELS = [m.strip() for m in os.environ.get("EXPERIMENT_MODELS", "mock-local").split(",") if m.strip()]
OUT_FILE = OUTPUT_DIR / "calibration_results_local.csv"


def main() -> None:
    inputs = load_inputs()
    refs = inputs["refs"]
    ref_mat = inputs["ref_mat"]

    primary_refs = refs[
        refs["feature"].isin(PRIMARY_FEATURES) & refs["reference_illustration_path"].fillna("").ne("")
    ].copy()
    options_by_feature = {feature: build_options(refs, ref_mat, feature) for feature in refs["feature"].dropna().unique()}

    done = set()
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE)
        done = set(zip(existing["model"], existing["feature_value"], existing["prompt_type"]))

    for model in MODELS:
        print(f"\n=== {model} ===")
        for _, ref_row in primary_refs.iterrows():
            feature_col = ref_row["feature"]
            true_value = ref_row["feature_value"]
            description = ref_row.get("reference_description", "") or ""
            ref_img = ref_row.get("reference_illustration_path", "") or ""
            options = options_by_feature.get(feature_col, [])

            save_row(done, model, true_value, "existence", feature_col, "YES", call_model(model, existence_parts(feature_col, ref_img, true_value)), parse_ync)
            save_row(done, model, true_value, "agreement", feature_col, "YES", call_model(model, agreement_parts(feature_col, true_value, description, ref_img)), parse_ync)
            save_row(done, model, true_value, "blind_mc", feature_col, true_value, call_model(model, blind_mc_parts(feature_col, options, ref_img)), lambda raw: parse_mc(raw, [o["value"] for o in options]))

    print(f"\nDone. Raw results -> {OUT_FILE}")


def save_row(done: set, model: str, feature_value: str, prompt_type: str, feature_col: str, expected: str, raw: str, parser) -> None:
    key = (model, feature_value, prompt_type)
    if key in done:
        return
    parsed = parser(raw)
    pd.DataFrame([
        {
            "model": model,
            "feature": feature_col,
            "feature_value": feature_value,
            "prompt_type": prompt_type,
            "raw_response": raw,
            "parsed": parsed,
            "expected": expected,
            "correct": parsed == expected,
        }
    ]).to_csv(OUT_FILE, mode="a", header=not OUT_FILE.exists(), index=False)
    done.add(key)


if __name__ == "__main__":
    main()
