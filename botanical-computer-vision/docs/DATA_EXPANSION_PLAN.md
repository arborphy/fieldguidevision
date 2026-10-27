# Data expansion plan: which photos to get next

Status: 2026-10-27. Numbers come from `scripts/build_pound_ridge_manifest.py`
(output in `data/expansion/`). No images have been downloaded.

## Why expand

All current results come from one collection (BioImages, 1,641 strict images, 465
individuals, 63 species, ~26 images per species). Every structure we find — clusters,
axes, error modes — may be a property of how BioImages photographs plants. The next data
should answer one question: **do the representation results hold on independent field
photos?** BioImages is frozen as the small regression benchmark.

## What we already have (no new download needed to plan)

| Count | Value |
|---|---|
| iNat export records (true CSV records; README's 11,881 is the line count) | 11,645 |
| GBIF mirror records / already in iNat export / GBIF-only | 4,225 / 4,038 / 187 |
| Unique observations in the union | 11,832 |
| iNat records whose species is in the strict 63 | 1,004 |
| … research grade, not captive/cultivated | 675 |
| … with a reusable licence (102 "all rights reserved" dropped) | **573** |
| Species with ≥ 1 / ≥ 10 / ≥ 20 eligible observations | 44 / 18 / 12 |
| Species with none | 19 (list in `pound_ridge_summary.json`) |
| Season mix of the 573 | autumn 286, summer 126, spring 94, winter 67 |
| Licence mix | CC-BY-NC 395, CC0 163, CC-BY 11, CC-BY-SA 4 |

Caveat: the export holds **one photo URL per observation**. Observations often have
several photos (leaf, bark, fruit); fetching them needs the iNaturalist API per
observation id. Pound Ridge alone is too thin for 19 species and for most per-organ
analyses.

## Stage 1 — Pound Ridge pilot (now)

- Manifest: `data/expansion/pound_ridge_pilot_manifest.csv` — **383 observations,
  44 species, ≤ 15 per species**, spread over observers and seasons; 200 rows are in the
  sealed test half.
- Split rule: observations grouped by `observer + date + species` (likely the same plant
  or visit). Half the groups of every species are **`sealed_test`**: never used for
  tuning, thresholds, cluster choice or prompt choice. The rest is `dev`.
- Steps:
  1. Download the `photo_url_large` images to Drive/local storage (not Git); log
     failures; record a SHA-256 per image; near-duplicate check (perceptual hash) against
     BioImages and within the set.
  2. Photo-level human review on the `dev` half first: organ multi-labels with the
     existing Human Review UI tag set (`leaf, twig, bark, flower, fruit, cone, seed,
     whole plant, other`) and scale (distant / mid / close / uncertain). The species
     label comes from iNat research grade; organ visibility does **not** come from the
     species knowledge tables.
  3. Embed with the same three frozen models and the same preprocessing.
- First uses:
  - BioCLIP zero-shot and the BioImages-trained linear probes on the sealed test →
    domain-shift accuracy, reported separately from BioImages numbers, 44-species
    candidate set (and the 63-candidate set as a second row).
  - Project into BioImages PCA d* / clusters → do the organ/scale enrichments
    (MEETING_EXPERIMENTS §2b) reappear?
- Done when: deduplicated manifest with licence and attribution, failure list, coverage
  table after review, and one sealed-test result table.

## Stage 2 — targeted iNaturalist pull (after Stage 1 works)

Goal: every strict species ≥ 20 independent observations, with several organs.

- Source: iNaturalist API, research grade, wild, licence in {CC0, CC-BY, CC-BY-NC,
  CC-BY-SA}, place = New York + New England + adjacent states (same flora as BioImages
  and Pound Ridge), all photos per observation.
- Priority order:
  1. The 19 species with zero Pound Ridge photos (list in the summary JSON).
  2. The 26 species with < 10 observations.
  3. Under-covered organs and seasons for species in the current error analysis
     (winter twig/bark; flowers in spring; whole-tree distant views).
- Budget: about 20–30 observations × 63 species ≈ 1,500 observations, ~3–5k photos.
  Write the expected effective count, storage and licence summary before running it;
  stop when the per-species target is met, not when the API runs out.
- Keep observation id, photo id, licence, attribution, observer, date, coordinates.
  The observation is the leakage unit; all photos of one observation stay in one split.

## What not to do

- Do not add species outside the 63 into closed-set accuracy. Unseen species is a
  separate open-set task.
- Do not add GBIF-only rows without checking the observation id is not already present.
- Do not commit images or third-party photos to Git; commit manifests only.
- Do not claim independence from model pre-training: iNaturalist photos may be in
  BioCLIP's training data (TreeOfLife-10M includes iNat). Report this as a caveat and
  prefer recent observations (e.g. 2025–2026) for the sealed test as a partial check.
