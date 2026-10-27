from __future__ import annotations

from datetime import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import threading

import pandas as pd

from model_adapter import call_model
from runner_common import (
    OUTPUT_DIR,
    PRIMARY_FEATURES,
    blind_mc_parts,
    build_options,
    existence_parts,
    get_true_path,
    load_inputs,
    parse_mc,
    parse_ync,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_MODE = os.environ.get("EXPERIMENT_MODEL_MODE", "mock").strip().lower()
MODELS = [m.strip() for m in os.environ.get("EXPERIMENT_MODELS", "mock-local").split(",") if m.strip()]
NUM_WORKERS = int(os.environ.get("EXPERIMENT_NUM_WORKERS", "3"))
FEATURES = [
    f.strip()
    for f in os.environ.get("EXPERIMENT_FEATURES", ",".join(PRIMARY_FEATURES)).split(",")
    if f.strip()
]
SAMPLE_LIMIT = os.environ.get("EXPERIMENT_SAMPLE_LIMIT")
RUN_STARTED = datetime.now().astimezone()
RUN_STARTED_AT = RUN_STARTED.isoformat(timespec="seconds")
TRIAL_ID = os.environ.get("EXPERIMENT_TRIAL_ID", RUN_STARTED.strftime("trial-%Y%m%d-%H%M%S"))
FEATURES_REQUESTED = ",".join(FEATURES)

ARTIFACT_DIR = Path(os.environ.get("EXPERIMENT_ARTIFACT_DIR", REPO_ROOT / "trials" / "artifacts")).expanduser()
if not ARTIFACT_DIR.is_absolute():
    ARTIFACT_DIR = REPO_ROOT / ARTIFACT_DIR


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug[:120] or "none"


def _output_file() -> Path:
    explicit = os.environ.get("EXPERIMENT_OUT_FILE")
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else ARTIFACT_DIR / path

    model_slug = _slug("-".join(MODELS))
    feature_slug = _slug("-".join(FEATURES))
    sample_slug = f"n{SAMPLE_LIMIT}" if SAMPLE_LIMIT else "all"
    timestamp_slug = RUN_STARTED.strftime("%Y%m%d-%H%M%S")
    trial_slug = f"-{_slug(TRIAL_ID)}" if os.environ.get("EXPERIMENT_TRIAL_ID") else ""
    filename = f"stepwise-{timestamp_slug}-{model_slug}-{sample_slug}-{feature_slug}{trial_slug}.csv"
    return ARTIFACT_DIR / filename


OUT_FILE = _output_file()


def main() -> None:
    inputs = load_inputs()
    refs = inputs["refs"]
    ref_mat = inputs["ref_mat"]
    path_table = inputs["kg"]
    sample = inputs["sample"]
    values_by_feature = inputs["values_by_feature"]

    unknown_features = [feature for feature in FEATURES if feature not in PRIMARY_FEATURES]
    if unknown_features:
        raise ValueError(f"Unsupported EXPERIMENT_FEATURES: {', '.join(unknown_features)}")

    if SAMPLE_LIMIT:
        sample = sample.head(int(SAMPLE_LIMIT))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    options_by_feature = {feature: build_options(refs, ref_mat, feature) for feature in refs["feature"].dropna().unique()}

    done = set()
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE)
        done = set(zip(existing["model"], existing["observation_id"].astype(str), existing["feature"]))

    csv_lock = threading.Lock()
    done_lock = threading.Lock()
    error_count = 0

    def process_task(model: str, row: pd.Series, feature_col: str) -> None:
        obs_id = str(row["observation_id"])
        with done_lock:
            if (model, obs_id, feature_col) in done:
                return

        true_path = get_true_path(path_table, row["species_inat"])
        true_value = true_path.get(feature_col)
        if true_value is None:
            return

        p1_prompt_parts = existence_parts(feature_col, row["photo_url"], true_value)
        p1_raw = call_model(model, p1_prompt_parts)
        p1_parsed = parse_ync(p1_raw)
        p1_parse_rule = "parse_ync: startswith YES -> YES; startswith NO -> NO; contains INC -> INCONCLUSIVE; else INCONCLUSIVE"

        values = values_by_feature.get(feature_col, [])
        options = options_by_feature.get(feature_col, [])
        if p1_parsed == "YES":
            p2_prompt_parts = blind_mc_parts(feature_col, options, row["photo_url"])
            p2_raw = call_model(model, p2_prompt_parts)
            p2_parsed = parse_mc(p2_raw, values)
            p2_parse_rule = "parse_mc: exact value match; else substring value match; else contains 'cannot determine' -> INCONCLUSIVE; else leading option number; else INCONCLUSIVE"
        else:
            p2_prompt_parts = []
            p2_raw = ""
            p2_parsed = "INCONCLUSIVE"
            p2_parse_rule = "P2 skipped because P1 was not YES; parsed as INCONCLUSIVE"

        if p2_parsed == true_value:
            feature_correct, committed = True, True
        elif p2_parsed == "INCONCLUSIVE":
            feature_correct, committed = True, False
        else:
            feature_correct, committed = False, True

        with csv_lock:
            pd.DataFrame([
                {
                    "trial_id": TRIAL_ID,
                    "run_started_at": RUN_STARTED_AT,
                    "model_mode": MODEL_MODE,
                    "model": model,
                    "sample_limit": SAMPLE_LIMIT or "",
                    "features_requested": FEATURES_REQUESTED,
                    "observation_id": obs_id,
                    "photo_id": row.get("photo_id", ""),
                    "photo_url": row.get("photo_url", ""),
                    "taxon_id": row.get("taxon_id", ""),
                    "species_inat": row["species_inat"],
                    "newcomb_species_name": row["newcomb_species_name"],
                    "feature": feature_col,
                    "true_value": true_value,
                    "p1_prompt_parts_json": json.dumps(p1_prompt_parts, ensure_ascii=False),
                    "p1_raw": p1_raw,
                    "p1_parsed": p1_parsed,
                    "p1_parse_rule": p1_parse_rule,
                    "p2_prompt_parts_json": json.dumps(p2_prompt_parts, ensure_ascii=False),
                    "p2_raw": p2_raw,
                    "p2_parsed": p2_parsed,
                    "p2_parse_rule": p2_parse_rule,
                    "feature_correct": feature_correct,
                    "committed": committed,
                }
            ]).to_csv(OUT_FILE, mode="a", header=not OUT_FILE.exists(), index=False)
            done.add((model, obs_id, feature_col))

        status = "✓" if feature_correct else "✗"
        print(f"  obs={obs_id} {feature_col} P1={p1_parsed} P2={p2_parsed} [{status}]")

    for model in MODELS:
        print(f"\n=== {model} ===")
        print(f"sample rows: {len(sample)}")
        print(f"features: {', '.join(FEATURES)}")
        print(f"workers: {NUM_WORKERS}")
        print(f"output: {OUT_FILE}")
        tasks = [(model, row, feature_col) for _, row in sample.iterrows() for feature_col in FEATURES]
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_task, model_name, row, feature_col) for model_name, row, feature_col in tasks]
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    error_count += 1
                    print(f"  [ERROR] {exc}")

    if error_count:
        raise SystemExit(f"\nFailed: {error_count} task(s) raised errors.")

    print(f"\nDone. Raw results -> {OUT_FILE}")


if __name__ == "__main__":
    main()
