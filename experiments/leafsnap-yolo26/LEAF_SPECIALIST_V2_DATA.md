# Leaf specialist v2 dataset revision

This is the single E4 revision requested after v1 exposed a view/label mismatch in field compound crops. It changes the audited species join and the compound training views only; it does not remove species based on validation or test performance.

## Audited label mapping

- Mapping: `/Users/valkyrie/dev/fieldguidevision-yolo26-3d/experiments/leafsnap-yolo26/feature_assertions.json` (SHA-256 `43efad068d6afc6dcb70efbc8f61fddf21cd1693c7913853b36643dd62c8f8d4`).
- EDA inputs: tags `b650afac6d4f4da2def30d5db4f985d538af1ca4d64ff605959258f5024e665b`, vocabulary `a69144b9eaf3fd955b14897b99aeb0f3876dd84a6a9ac5c36cdab99f7a4786ed`, LeafSnap name manifest `d2bcc3878a77d691011c233b5743e8d7502e668b08d83cef450a584ff2672845`.
- Exact eligible labels: 122 species — 101 simple and 21 compound.
- Match rules among eligible species: {"curated_synonym": 5, "exact": 109, "hybrid_name_form": 3, "orthographic_variant": 2, "single_character_typo": 3}.
- Match confidence: {"high": 118, "medium": 4}.
- Mixed and excluded: Ginkgo biloba, Gymnocladus dioicus (both assert Simple and Compound).
- Unmatched and not force-matched: 20 names recorded in `classification-v2/counts.json`; suggested matches in the EDA are not applied.
- Matched species without exactly one Simple/Compound value remain excluded: 41; conifer-only values are not coerced into a class.
- This uses the EDA's audited `leafsnap_species` → `dirr_species` mapping: 13 of 14 reviewed non-exact name matches have an exact single Simple/Compound assertion and are included; the rest remain excluded. The exact-name-only builder's 109-species set is superseded; no unsupported name joins are added.

## Split and sample policy

- Split function: `build_datasets.classifier_species_split; class-stratified SHA-256 ordering; 20% test, 15% validation, remainder train; species are indivisible`.
- Split salt: `leafsnap-yolo26-v1-20261029:classifier`. Each species is assigned to exactly one split; image/crop rows cannot cross splits.
- Original class-stratified caps are retained: train 40 lab + 15 field; validation 15 + 15; test 10 lab + all available field per species, except for the compound-training domain rule below.
- Simple training uses both lab and field crops. Compound training uses only lab crops (pressed whole-architecture specimens); all field compound crops remain available in validation/test, subject to the existing caps.
- No species is removed due to per-species or aggregate validation/test accuracy. Species with weak performance remain reported in `LEAF_SPECIALIST.md`.

## Crop geometry and provenance

- Source root: `/Users/valkyrie/Desktop/arborphy product/leafsnap/leafsnap-dataset`.
- LeafSnap manifest SHA-256: `d2bcc3878a77d691011c233b5743e8d7502e668b08d83cef450a584ff2672845`.
- Dataset generator: `/Users/valkyrie/dev/fieldguidevision-yolo26-3d/experiments/leafsnap-yolo26/build_leaf_specialist_v2.py` (SHA-256 `cd6c1dcb27887eaea4681d03c7e0e583bae1dbed72305a1f36e1b4867939a494`).
- Geometry: normalized mask foreground bbox scaled to the original image; margin 0.10 per side, clamped; JPEG quality 92, subsampling 0.
- Shared scan implementation: `/Users/valkyrie/dev/fieldguidevision-yolo26-3d/experiments/leafsnap-yolo26/build_datasets.py` (SHA-256 `56cccb46a95b63ecd68fdc470f692a2206ed143e07dfc426c0bbceb9e8b29f0a`), including its image/mask validity filters and normalized-mask-to-source-image scaling.
- Every manifest row stores the source image, mask path, crop/full bounding boxes, exact EDA mapping row fields, mapping artifact path, and mapping hash.

## Counts

| split | simple | compound | field | lab | species |
|---|---:|---:|---:|---:|---:|
| train | 3505 | 537 | 983 | 3059 | 80 |
| val | 450 | 90 | 270 | 270 | 18 |
| test | 1221 | 254 | 1235 | 240 | 24 |

Total retained crops: 6,057; species: 122.
Compound training FIELD crops: 0 (must be zero). Compound field validation/test crops: 45 / 214.
Species are disjoint: {"test": 24, "train": 80, "val": 18}; see `labels.json` for the full species list by split.

## View-validity limit

The compound training restriction uses pressed lab specimens to favor whole-architecture views. Field crops are intentionally retained in validation and test; they can still show a single leaflet instead of an entire compound leaf. Therefore field compound recall includes a view-validity noise floor: a visually simple-looking leaflet inherits its species-level compound label. This is not a reason to remove Juglans cinerea or any other weak test species.

Dataset output: `/Users/valkyrie/dev/fieldguidevision-yolo26-3d/scratch/leafsnap-yolo26/classification-v2`.
