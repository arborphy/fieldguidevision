"""Build the Pound Ridge external-photo manifest (no image download).

Joins the two Pound Ridge exports to the strict 63-species BioImages set,
deduplicates iNat vs GBIF by observation id, applies quality and licence
filters, and assigns a sealed external test split grouped so that the same
observer/day/species never crosses splits.

Outputs (relative to botanical-computer-vision/):
  data/expansion/pound_ridge_candidates.csv      every eligible observation
  data/expansion/pound_ridge_pilot_manifest.csv  balanced pilot selection
  data/expansion/pound_ridge_coverage.csv        per-species coverage
  data/expansion/pound_ridge_summary.json        counts for the plan

Run from the repository root:
  python botanical-computer-vision/scripts/build_pound_ridge_manifest.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BCV = REPO / "botanical-computer-vision"
OUT = BCV / "data" / "expansion"
INAT = REPO / "observations_pound_ridge" / "ward_pound_ridge_species.csv"
GBIF = REPO / "observations_pound_ridge" / "gbif_inaturalist_observations.csv"
STRICT = BCV / "organ_benchmark" / "split_manifests" / "individual_disjoint.csv"

# Licences that allow research use with attribution. Rows with no licence are
# "all rights reserved" on iNaturalist and are excluded.
OK_LICENSES = {"CC0", "CC-BY", "CC-BY-NC", "CC-BY-SA", "CC-BY-NC-SA"}
PILOT_CAP_PER_SPECIES = 15  # pilot budget, not a statistical target
SEALED_FRACTION = 0.5       # half of each species' groups never used for tuning
SEED = "fgv-pound-ridge-v1"


def season(month: float) -> str:
    if pd.isna(month):
        return "unknown"
    m = int(month)
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer"}.get(m, "autumn")


def stable_rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    strict = pd.read_csv(STRICT)
    strict_species = set(strict.species)
    bioimages_counts = strict.groupby("species").agg(
        bioimages_images=("image_id", "size"), bioimages_individuals=("individual_id", "nunique"))

    # csv module handles multi-line description fields: 11,645 records, not 11,881 lines.
    with INAT.open(newline="", encoding="utf-8") as fh:
        inat = pd.DataFrame(list(csv.DictReader(fh)))
    inat["observation_id"] = inat["id"].astype(int)
    inat["species"] = inat["taxon_species_name"].where(inat["taxon_species_name"] != "", inat["scientific_name"])
    inat["license"] = inat["license"].replace("", pd.NA)

    gbif = pd.read_csv(GBIF, sep="\t", low_memory=False)
    gbif["observation_id"] = pd.to_numeric(gbif["occurrenceID"].str.extract(r"(\d+)$")[0], errors="coerce")
    gbif_only = gbif[~gbif["observation_id"].isin(inat["observation_id"])]

    total = {
        "inat_records": int(len(inat)),
        "gbif_records": int(len(gbif)),
        "gbif_ids_also_in_inat": int(gbif["observation_id"].isin(inat["observation_id"]).sum()),
        "gbif_only_records": int(len(gbif_only)),
        "union_unique_observations": int(len(set(inat["observation_id"]) | set(gbif["observation_id"].dropna().astype(int)))),
    }

    d = inat[inat["species"].isin(strict_species)].copy()
    total["inat_records_in_strict63"] = int(len(d))
    d = d[(d["quality_grade"] == "research") & (d["captive_cultivated"].str.upper() != "TRUE")]
    total["research_wild_in_strict63"] = int(len(d))
    total["excluded_no_licence"] = int(d["license"].isna().sum())
    d = d[d["license"].isin(OK_LICENSES) & (d["image_url"] != "")].copy()
    total["eligible_observations"] = int(len(d))

    d["observed_on"] = pd.to_datetime(d["observed_on"], errors="coerce")
    d["season"] = d["observed_on"].dt.month.map(season)
    # Group = same observer, same day, same species: likely the same plant or visit.
    d["leakage_group"] = (d["user_id"].astype(str) + "|" + d["observed_on"].dt.strftime("%Y-%m-%d").fillna("na")
                          + "|" + d["species"])
    d["photo_url_medium"] = d["image_url"]
    d["photo_url_large"] = d["image_url"].str.replace("/medium.", "/large.", regex=False)
    d["observation_url"] = d["url"].str.replace("http://", "https://", regex=False)
    d["source"] = "inaturalist_export"
    d["organ_labels"] = ""          # to be filled by photo-level human review
    d["review_status"] = "unreviewed"

    # Sealed split: per species, rank groups deterministically, seal ~half.
    groups = d[["species", "leakage_group"]].drop_duplicates().copy()
    groups["rank"] = groups["leakage_group"].map(stable_rank)
    groups = groups.sort_values(["species", "rank"])
    groups["pos"] = groups.groupby("species").cumcount()
    groups["n"] = groups.groupby("species")["leakage_group"].transform("size")
    groups["external_split"] = (groups["pos"] < (groups["n"] * SEALED_FRACTION).round().clip(lower=1)) \
        .map({True: "sealed_test", False: "dev"})
    d = d.merge(groups[["leakage_group", "external_split"]], on="leakage_group", how="left")

    cols = ["observation_id", "species", "common_name", "observed_on", "season", "user_id", "license",
            "quality_grade", "num_identification_agreements", "latitude", "longitude", "leakage_group",
            "external_split", "photo_url_medium", "photo_url_large", "observation_url", "source",
            "organ_labels", "review_status"]
    d = d.sort_values(["species", "observed_on"])
    d[cols].to_csv(OUT / "pound_ridge_candidates.csv", index=False)

    # Pilot: per species, spread across seasons and observers, cap at budget.
    d["rank"] = d["observation_id"].astype(str).map(stable_rank)
    d = d.sort_values(["species", "rank"])
    d["season_pos"] = d.groupby(["species", "season"]).cumcount()
    d["user_pos"] = d.groupby(["species", "user_id"]).cumcount()
    d = d.sort_values(["species", "user_pos", "season_pos", "rank"])
    pilot = d.groupby("species").head(PILOT_CAP_PER_SPECIES)
    pilot[cols].to_csv(OUT / "pound_ridge_pilot_manifest.csv", index=False)

    cov = d.groupby("species").agg(
        eligible_observations=("observation_id", "size"),
        observers=("user_id", "nunique"),
        leakage_groups=("leakage_group", "nunique"),
        seasons=("season", "nunique"),
        sealed_test_obs=("external_split", lambda s: int((s == "sealed_test").sum())),
    )
    cov["pilot_obs"] = pilot.groupby("species").size()
    cov = bioimages_counts.join(cov, how="left").fillna(0)
    for c in cov.columns:
        cov[c] = cov[c].astype(int)
    cov["coverage_tier"] = pd.cut(cov["eligible_observations"], [-1, 0, 4, 9, 19, 10**6],
                                  labels=["none", "1-4", "5-9", "10-19", "20+"])
    cov.sort_values("eligible_observations", ascending=False).to_csv(OUT / "pound_ridge_coverage.csv")

    total.update({
        "strict_species": len(strict_species),
        "species_with_any_eligible": int((cov["eligible_observations"] > 0).sum()),
        "species_with_10plus": int((cov["eligible_observations"] >= 10).sum()),
        "species_with_20plus": int((cov["eligible_observations"] >= 20).sum()),
        "species_missing": sorted(cov.index[cov["eligible_observations"] == 0].tolist()),
        "pilot_observations": int(len(pilot)),
        "pilot_species": int(pilot["species"].nunique()),
        "pilot_sealed_test": int((pilot["external_split"] == "sealed_test").sum()),
        "licence_mix_eligible": d["license"].value_counts().to_dict(),
        "season_mix_eligible": d["season"].value_counts().to_dict(),
        "note": "One photo URL per observation (the export's first photo). Images are not downloaded or committed.",
    })
    (OUT / "pound_ridge_summary.json").write_text(json.dumps(total, indent=2) + "\n")
    print(json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
