# Data catalog

Every dataset the research uses, where the authoritative copy lives, and what it may
and may not be used for. Images themselves are never committed; BioImages photos load
from `bioimages.vanderbilt.edu`, field photos from iNaturalist URLs.

## Image sets

| ID | What | Size | Authoritative file | Notes |
|---|---|---|---|---|
| `bioimages_full` | BioImages NE trees catalog | 1,899 images · 85 species · 492 individuals · 12 organ categories | `bioimages_browser/data/` (browser payload) | Keep original `organ_tag`, `organ_category`, `subview`, `view_code`. |
| `bioimages_strict` | Subset used for all embedding work | 1,641 images · 63 species · 465 individuals | `organ_benchmark/split_manifests/master_manifest.csv` | Species with enough individuals for a disjoint split. |
| `split_individual` | Individual-disjoint split | 1,181 / 249 / 211 | `organ_benchmark/split_manifests/individual_disjoint.csv` | Default split. Test labels loaded only after predictions. |
| `split_species` | Species-disjoint split (organ transfer) | 1,144 / 238 / 259; 10 test species | `organ_benchmark/split_manifests/species_disjoint.csv` | Organ tasks only. |
| `pound_ridge_candidates` | Eligible external field observations | 573 obs · 44 of 63 species | `data/expansion/pound_ridge_candidates.csv` | Built by `scripts/build_pound_ridge_manifest.py`. Not downloaded. |
| `pound_ridge_pilot` | Pilot selection | 383 obs · 44 species · 200 sealed test | `data/expansion/pound_ridge_pilot_manifest.csv` | `sealed_test` rows: never used for tuning. |

Task-specific denominators differ (NSC 1,180/246/211 after dropping unsupported labels;
AE trained on 1,181). Always write the denominator next to a metric.

## Embeddings (derived, frozen)

| Model | Dim | Authoritative copy | SHA-256 |
|---|---|---|---|
| BioCLIP 2.5 | — | Drive: `outputs/bioclip25_strict_embeddings/embeddings.npz` (4,255,575 B) | record on recovery |
| DINOv3 ViT-B/16 | — | Drive: `outputs/dinov3_strict_embeddings/` | record on recovery |
| EfficientNet-B0 | 1280 | **Git**: `data/embeddings/efficientnet_b0_strict/embeddings.npz` (rescued from a share ZIP) | `b1b6e4f21f8b6ceb3e08f90ac8766e63a9dcce0caa82857068e85dc71679730b` |

Each `.npz` holds `image_ids`, `embeddings` (L2-normalised), `splits`, `model`,
`normalized`. Check that `image_ids` and `splits` match `individual_disjoint.csv` before
use. Never substitute web-page PCA coordinates for raw vectors. Drive folders are listed
in the handoff note (`FieldGuideVision_Project_Handoff_2026-10-27.md`).

## Labels

| Label | Source | Status | Use as |
|---|---|---|---|
| Species | BioImages / iNat research grade | reliable | ground truth |
| Organ (single) | BioImages `organ_category` | reliable but single-label | ground truth for "primary organ" only |
| Organ (multi) | Gemma 40-tag | model output | reference, not truth |
| Human Gold organ multi-label | `bioimages_human_loop` → Supabase | **completion unknown** | truth once exported |
| Scale (distant 350 / mid 158 / close 1,105 / uncertain 28) | BioImages + Gemma | operational | stratifier, not distance truth |
| Human scale audit (182) | `scale_conditioned_analysis/review` → Supabase | **completion unknown** | truth once exported |

## Field observations and knowledge sources (from `main`, not this branch)

| Folder | Content | Note |
|---|---|---|
| `observations_pound_ridge/` | iNat export 11,645 records (README says 11,881 = line count); GBIF mirror 4,225 (4,038 already in iNat) | Union 11,832 observations. One photo URL each. |
| `manual_of_woody_plants/`, `newcomb_wildflower_guide/`, `gobotany/` | Species × trait vocabularies, definitions, illustrations | Species-level knowledge; never a per-photo visibility label. |
