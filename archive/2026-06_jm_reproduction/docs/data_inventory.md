# Newcomb Data Inventory

This inventory focuses on the current Newcomb reproduction path for John
Matter's experiment. The real source folder is `newcomb_wildflower_guide/`.

## Current Runner Inputs

These are the four JM-style CSV inputs consumed by the local reproduction
runner in `newcomb_wildflower_guide/experiment_repro/`.

| File | Rows | Role | Key Fields | John Used? | Relationship | Current Understanding |
| --- | ---: | --- | --- | --- | --- | --- |
| `newcomb_wildflower_guide/experiment_repro/output/feature_value_pairs.csv` | 76 | Enumerates valid values for each feature. | `feature`, `value` | Yes, equivalent shape | Built from `newcomb_route_feature_value.parquet`; used by `parse_mc` to normalize model multiple-choice answers. | Clear |
| `newcomb_wildflower_guide/experiment_repro/output/references.csv` | 76 | Gives each feature value a text description and optional illustration path. | `feature_value`, `reference_illustration_path`, `reference_description`, `feature` | Yes, equivalent shape | Built from feature values plus `newcomb_feature_concept.parquet`; illustration paths are bridged to the old JM illustration folder when available. | Clear, but illustration coverage is incomplete |
| `newcomb_wildflower_guide/experiment_repro/output/newcomb_preprocessed.csv` | 781 | Species-to-key-path table. This is the expected answer table for evaluation. | `newcomb_species_name`, `species_inat`, `key_flower_type`, `key_plant_type`, `key_leaf_type`, subgroup fields | Yes, equivalent shape | Built by joining `newcomb_taxon_route.parquet`, `newcomb_taxon.parquet`, `newcomb_key_route.parquet`, and route feature values. | Clear |
| `newcomb_wildflower_guide/experiment_repro/output/sample.csv` | 288 | Observation/photo sample joined to Newcomb species. | `newcomb_species_name`, `species_inat`, `observation_id`, `photo_url` | Yes, equivalent shape | Built from `observations_pound_ridge/ward_pound_ridge_species.csv` joined to Newcomb taxa by scientific name. | Clear |

The runner also reads `illustration_paths.json`, which maps
`feature:value` pairs to available illustration files. It is currently mostly a
bridge to the old JM illustration assets, not a full canonical Newcomb
illustration extraction.

## Current Trial Outputs

Stepwise trial outputs now live in `trials/artifacts/`.

| File Pattern | Role | Notes |
| --- | --- | --- |
| `*.csv` | Per-observation, per-feature trial results. | Includes prompt parts JSON, raw model answers, parsed answers, expected value, `feature_correct`, `committed`, and `trial_id`. |
| `*.metadata.json` | Reproduction metadata for the trial. | Includes command, model, sample limit, features, mode, output file, git commit, Python version, and package manager. |

The current real smoke test artifact is:

- `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.csv`
- `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.metadata.json`

## Newcomb Parquet Tables

The parquet files are the structured extraction output. The current reproduction
pipeline uses a subset of them to recreate John's CSV-shaped inputs.

| Parquet Table | Rows | Role | Key Fields | Used in Current Runner Build? | Current Understanding |
| --- | ---: | --- | --- | --- | --- |
| `newcomb_feature_concept.parquet` | 6 | Defines the feature concepts and prompts. | `feature_name`, `feature_prompt`, `path_level`, `is_subgroup`, `value_count` | Yes, for reference descriptions | Clear |
| `newcomb_key_route.parquet` | 147 | One row per route through the Newcomb key. | `route_id`, key feature columns, page range, description | Yes, for species path table | Clear |
| `newcomb_route_feature_value.parquet` | 595 | Long-form route steps. | `route_id`, `path_k`, `feature_name`, `feature_value` | Yes, for feature values and pivoted species path columns | Clear |
| `newcomb_taxon.parquet` | 423 | Taxon/species table. | `taxon_id`, `scientific_name`, `canonical_name`, `species_inat_link` | Yes, for species names and iNat join keys | Clear |
| `newcomb_taxon_route.parquet` | 781 | Links taxa to key routes. | `taxon_id`, `route_id`, `species_page`, extraction warnings | Yes, for species path table | Clear |
| `newcomb_locator_entry.parquet` | 232 | Locator/key entry table. | `locator_entry_id`, flower/plant/leaf type, group number, description, page | Not currently used by runner | Partially understood; useful for auditing locator extraction |
| `newcomb_locator_entry_subgroup.parquet` | 210 | Subgroup values for locator entries. | `locator_entry_id`, `subgroup_ordinal`, `subgroup_value` | Not currently used by runner | Partially understood |
| `newcomb_locator_decision_edge.parquet` | 647 | Edges in the locator decision graph. | `from_node_id`, `to_node_id`, `edge_type`, labels, group number | Not currently used by runner | Partially understood |
| `newcomb_locator_json_tree.parquet` | 1683 | Flattened JSON tree of locator extraction. | `json_node_id`, `parent_json_node_id`, `node_key`, `node_type`, `fullkey`, `value` | Not currently used by runner | Mostly diagnostic/source-trace table |
| `newcomb_source_meta.parquet` | 1 | Source-level extraction metadata. | `source_title`, `source_version`, `total_entries`, `unique_groups` | Not currently used by runner | Clear |

## Simplification Hypothesis

For John's reproduction, the useful parquet surface can probably be simplified
to about four conceptual tables:

| Simplified Table | Backing Current Tables | Why It Matters |
| --- | --- | --- |
| `feature_concepts` | `newcomb_feature_concept.parquet` | Defines feature names, prompts, and feature levels. |
| `feature_values_by_route` | `newcomb_route_feature_value.parquet` plus `newcomb_key_route.parquet` | Defines allowed feature values and the key route structure. |
| `taxon_routes` | `newcomb_taxon.parquet` plus `newcomb_taxon_route.parquet` | Links species to expected key paths. |
| `references_or_illustrations` | current bridge logic plus future canonical Newcomb illustrations | Provides definitions and visual references for model prompts. |

The locator graph tables look useful for auditing and future UI/path work, but
they are not necessary for the current JM-style smoke test.

