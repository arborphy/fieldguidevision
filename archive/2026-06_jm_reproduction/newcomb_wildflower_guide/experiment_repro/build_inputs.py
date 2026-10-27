from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent
PROJECT_ROOT = SOURCE_ROOT.parent

PARQUET_DIR = SOURCE_ROOT / "extracted_vocabulary" / "parquet"
OBS_DIR = PROJECT_ROOT / "observations_pound_ridge"
JM_DIR = PROJECT_ROOT.parent / "JM_Assets" / "data"
JM_ILLUSTRATIONS_DIR = JM_DIR / "illustrations"
OUT_DIR = ROOT / "output"


FEATURE_NAME_MAP = {
    "flower_type": "key_flower_type",
    "plant_type": "key_plant_type",
    "leaf_type": "key_leaf_type",
    "subgroup_1": "key_subgroup_1",
    "subgroup_2": "key_subgroup_2",
    "subgroup_3": "key_subgroup_3",
}


ILLUSTRATION_BY_VALUE = {
    "3 Regular Parts": "regular_flowers.png",
    "4 Regular Parts": "regular_flowers.png",
    "5 Regular Parts": "regular_flowers.png",
    "6 Regular Parts": "regular_flowers.png",
    "7 or More Regular Parts": "regular_flowers.png",
    "Irregular Flowers": "irregular_flowers.png",
    "Wildflowers - Alternate Leaves": "alternate_leaves.png",
    "Wildflowers - Opposite or Whorled Leaves": "opposite_leaves.png",
    "Wildflowers - Basal Leaves Only": "basal_leaves.png",
    "Leaves Entire": "leaves_entire.png",
    "Leaves Toothed or Lobed": "leaves_toothed.png",
    "Leaves Divided": "leaves_divided.png",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    feature_values = build_feature_value_pairs(con)
    feature_values.to_csv(OUT_DIR / "feature_value_pairs.csv", index=False)

    preprocessed = build_newcomb_preprocessed(con)
    preprocessed.to_csv(OUT_DIR / "newcomb_preprocessed.csv", index=False)

    references, illustration_paths = build_references_and_paths(feature_values)
    references.to_csv(OUT_DIR / "references.csv", index=False)
    references.to_csv(OUT_DIR / "references_with_paths.csv", index=False)
    (OUT_DIR / "illustration_paths.json").write_text(json.dumps(illustration_paths, indent=2))

    sample = build_sample(con)
    sample.to_csv(OUT_DIR / "sample.csv", index=False)

    summary = {
        "feature_value_pairs_rows": len(feature_values),
        "references_rows": len(references),
        "newcomb_preprocessed_rows": len(preprocessed),
        "sample_rows": len(sample),
        "illustration_matches": sum(1 for p in illustration_paths.values() if p),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def build_feature_value_pairs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    route_feature_value = PARQUET_DIR / "newcomb_route_feature_value.parquet"
    df = con.execute(
        f"""
        SELECT DISTINCT
          feature_name,
          feature_value
        FROM read_parquet('{route_feature_value}')
        WHERE feature_name IS NOT NULL
          AND feature_value IS NOT NULL
          AND feature_value <> ''
        ORDER BY feature_name, feature_value
        """
    ).df()
    df["feature"] = df["feature_name"].map(FEATURE_NAME_MAP)
    df = df[df["feature"].notna()][["feature", "feature_value"]].rename(columns={"feature_value": "value"})
    return df.reset_index(drop=True)


def build_newcomb_preprocessed(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    taxon = PARQUET_DIR / "newcomb_taxon.parquet"
    taxon_route = PARQUET_DIR / "newcomb_taxon_route.parquet"
    key_route = PARQUET_DIR / "newcomb_key_route.parquet"

    base = con.execute(
        f"""
        SELECT
          tr.taxon_id,
          tr.route_id,
          t.scientific_name AS newcomb_species_name,
          t.canonical_name AS suggested_species_id,
          t.canonical_name AS species_inat,
          t.subspecies_inat,
          kr.key_page_range_start,
          kr.key_page_range_end,
          kr.key_group_number,
          tr.species_page,
          kr.key_description,
          t.species_inat_link,
          CAST(tr.warning_species_extraction AS DOUBLE) AS warning_species_extraction,
          CAST(tr.warning_key_extraction AS DOUBLE) AS warning_key_extraction
        FROM read_parquet('{taxon_route}') tr
        JOIN read_parquet('{taxon}') t USING (taxon_id)
        JOIN read_parquet('{key_route}') kr USING (route_id)
        ORDER BY t.scientific_name, tr.route_id
        """
    ).df()

    route_steps = con.execute(
        f"""
        SELECT
          route_id,
          feature_name,
          feature_value
        FROM read_parquet('{PARQUET_DIR / 'newcomb_route_feature_value.parquet'}')
        """
    ).df()
    route_steps["feature"] = route_steps["feature_name"].map(FEATURE_NAME_MAP)
    route_steps = route_steps[route_steps["feature"].notna()][["route_id", "feature", "feature_value"]]

    pivot = route_steps.pivot_table(
        index="route_id",
        columns="feature",
        values="feature_value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    merged = base.merge(pivot, on="route_id", how="left")
    merged["subspecies_inat_link"] = ""

    ordered = [
        "newcomb_species_name",
        "species_inat",
        "subspecies_inat",
        "suggested_species_id",
        "species_page",
        "key_page_range_start",
        "key_page_range_end",
        "key_group_number",
        "key_flower_type",
        "key_plant_type",
        "key_leaf_type",
        "key_subgroup_1",
        "key_subgroup_2",
        "key_subgroup_3",
        "key_description",
        "species_inat_link",
        "subspecies_inat_link",
        "warning_species_extraction",
        "warning_key_extraction",
    ]
    for col in ordered:
        if col not in merged.columns:
            merged[col] = ""
    return merged[ordered].sort_values(["newcomb_species_name", "species_page", "key_group_number"]).reset_index(drop=True)


def build_references_and_paths(feature_values: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    con = duckdb.connect()
    feature_concept = con.execute(
        f"SELECT * FROM read_parquet('{PARQUET_DIR / 'newcomb_feature_concept.parquet'}')"
    ).df()
    prompt_by_feature = {
        FEATURE_NAME_MAP[row.feature_name]: row.feature_prompt
        for row in feature_concept.itertuples()
        if row.feature_name in FEATURE_NAME_MAP
    }

    rows = []
    paths = {}
    for row in feature_values.itertuples(index=False):
        png = ILLUSTRATION_BY_VALUE.get(row.value)
        local_path = JM_ILLUSTRATIONS_DIR / png if png else None
        local_str = str(local_path) if local_path and local_path.exists() else ""
        key = f"{row.feature}:{row.value}"
        paths[key] = local_str
        rows.append(
            {
                "feature_value": row.value,
                "reference_image_link": "",
                "reference_illustration_path": local_str,
                "reference_description": prompt_by_feature.get(row.feature, ""),
                "feature": row.feature,
            }
        )
    refs = pd.DataFrame(rows)
    return refs, paths


def build_sample(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pre = build_newcomb_preprocessed(con)
    obs_path = OBS_DIR / "ward_pound_ridge_species.csv"
    obs = pd.read_csv(obs_path)

    pre_species = pre[["newcomb_species_name", "species_inat"]].drop_duplicates()
    merged = obs.merge(pre_species, left_on="scientific_name", right_on="species_inat", how="inner")

    if "photo_url" not in merged.columns:
        merged["photo_url"] = merged.get("image_url", "")

    if "taxon_id" not in merged.columns:
        merged["taxon_id"] = ""
    if "photo_id" not in merged.columns:
        merged["photo_id"] = ""
    if "observation_id" not in merged.columns:
        merged["observation_id"] = merged.get("id", "")

    out_cols = [
        "newcomb_species_name",
        "species_inat",
        "taxon_id",
        "observation_id",
        "photo_id",
        "photo_url",
    ]
    for col in out_cols:
        if col not in merged.columns:
            merged[col] = ""

    return merged[out_cols].drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    main()
