from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

FEATURE_DISPLAY = {
    "key_flower_type": "flower type",
    "key_plant_type": "plant type",
    "key_leaf_type": "leaf type",
    "key_subgroup_1": "subsidiary feature (subgroup 1)",
    "key_subgroup_2": "subsidiary feature (subgroup 2)",
    "key_subgroup_3": "subsidiary feature (subgroup 3)",
}

PRIMARY_FEATURES = ["key_flower_type", "key_plant_type", "key_leaf_type"]


def load_inputs() -> dict[str, object]:
    refs = pd.read_csv(OUTPUT_DIR / "references.csv")
    fv = pd.read_csv(OUTPUT_DIR / "feature_value_pairs.csv")
    kg = pd.read_csv(OUTPUT_DIR / "newcomb_preprocessed.csv")
    sample = pd.read_csv(OUTPUT_DIR / "sample.csv")
    illustration_paths = json.loads((OUTPUT_DIR / "illustration_paths.json").read_text())

    values_by_feature = {
        feat: fv.loc[fv["feature"] == feat, "value"].dropna().tolist()
        for feat in fv["feature"].dropna().unique()
    }

    ref_mat = {}
    for _, row in refs.iterrows():
        feat = row["feature"]
        val = row["feature_value"]
        illust_path = row.get("reference_illustration_path", "")
        if not isinstance(illust_path, str):
            illust_path = ""
        ref_mat[(feat, val)] = {
            "img_path": None,
            "illust_path": illust_path if illust_path and Path(illust_path).exists() else None,
            "description": row.get("reference_description", "") or "",
        }

    return {
        "refs": refs,
        "fv": fv,
        "kg": kg,
        "sample": sample,
        "illustration_paths": illustration_paths,
        "values_by_feature": values_by_feature,
        "ref_mat": ref_mat,
    }


def get_true_path(path_table: pd.DataFrame, species_inat: str) -> dict[str, str]:
    matches = path_table.loc[path_table["species_inat"] == species_inat]
    if matches.empty:
        return {}
    row = matches.iloc[0]
    return {
        feature: str(row[feature]).strip()
        for feature in FEATURE_DISPLAY
        if feature in row.index and isinstance(row[feature], str) and row[feature].strip()
    }


def existence_parts(feature_col: str, image_path_or_url: str, true_value: str = "") -> list:
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    extra = ""
    if feature_col == "key_leaf_type" and "no apparent" in true_value.lower():
        extra = (
            " Note: the absence of visible leaves is itself a discernible leaf-type feature"
            " — if the stem has no apparent leaves, answer YES."
        )
    return [
        {"image": image_path_or_url},
        (
            f"Your task is to identify if the visual feature '{fname}' is discernible "
            f"in the following image. Answer only: YES, NO, or INCONCLUSIVE. "
            f"Answer YES if and only if the feature is clearly discernible in the image.{extra}"
        ),
    ]


def agreement_parts(feature_col: str, true_value: str, description: str, image_path_or_url: str) -> list:
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    desc_clause = f" — {description}" if description else ""
    return [
        {"image": image_path_or_url},
        (
            f"A botanical expert has classified the '{fname}' of the plant in this image "
            f"as '{true_value}'{desc_clause}.\n"
            f"Is this classification consistent with what you can observe in the image?\n"
            f"Reply with exactly one word: YES, NO, or INCONCLUSIVE."
        ),
    ]


def blind_mc_parts(feature_col: str, options: list[dict], image_path_or_url: str) -> list:
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    parts = [
        {"image": image_path_or_url},
        (
            f"Which of the following options best describes the '{fname}' of the plant in this image?\n"
            f"If the feature is not clearly visible, choose 'Cannot determine from this image'.\n\n"
            f"Options:"
        ),
    ]
    for i, opt in enumerate(options, 1):
        desc = f" — {opt['description']}" if opt["description"] else ""
        parts.append(f"\n{i}. **{opt['value']}**{desc}")
        if opt.get("img_path"):
            parts.append(f"Reference photo for option {i}:")
            parts.append({"image": opt["img_path"]})
        if opt.get("illust_path"):
            parts.append(f"Botanical illustration for option {i}:")
            parts.append({"image": opt["illust_path"]})
    parts.append(f"\n{len(options) + 1}. **Cannot determine from this image**")
    parts.append(
        f"\nRespond with ONLY the exact option text (e.g., \"{options[0]['value']}\") "
        f"or \"Cannot determine from this image\"."
    )
    return parts


def parse_ync(response: str) -> str:
    r = response.upper()
    if r.startswith("YES"):
        return "YES"
    if r.startswith("NO"):
        return "NO"
    if "INC" in r:
        return "INCONCLUSIVE"
    for kw in ("YES", "NO", "INCONCLUSIVE"):
        if kw in r:
            return kw
    return "INCONCLUSIVE"


def parse_mc(response: str, values: list[str]) -> str:
    r = response.strip()
    for value in values:
        if value.lower() == r.lower():
            return value
    for value in values:
        if value.lower() in r.lower():
            return value
    if "cannot determine" in r.lower():
        return "INCONCLUSIVE"
    match = re.match(r"^(\d+)", r)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(values):
            return values[idx]
        if idx == len(values):
            return "INCONCLUSIVE"
    return "INCONCLUSIVE"


def build_options(refs: pd.DataFrame, ref_mat: dict, feature_col: str) -> list[dict]:
    options = []
    for _, row in refs.loc[refs["feature"] == feature_col].iterrows():
        mat = ref_mat.get((feature_col, row["feature_value"]), {})
        options.append(
            {
                "value": row["feature_value"],
                "img_path": mat.get("img_path"),
                "illust_path": mat.get("illust_path"),
                "description": mat.get("description", ""),
            }
        )
    return options
