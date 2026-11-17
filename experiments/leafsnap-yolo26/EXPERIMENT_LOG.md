# Experiment log — 2026-10-29

All times are local (EDT, UTC−4); reports record UTC. Paths are relative to
`/Users/valkyrie/dev/fieldguidevision-yolo26-3d` unless absolute.

## 1. Pipeline specification intake

The pipeline architecture was supplied as `Field Vision Pipeline.pdf` plus a
CSV export. The PDF is image-only (`pdftotext` yields ~1 byte / no words), so
the **CSV is the authoritative specification**:

`/Users/valkyrie/Downloads/Field Vision Pipeline - Experiment architecture (widescreen).csv`
(2,410 bytes, 44 rows; PDF is 354,921 bytes, MD5 `bc11fdd9…`).

The CSV defines one shared evaluator with four variants: E1 label quality,
E2 training-input quality (marked TODO), E3 top-level performance, E4
lower-level performance. Top-level must pass before lower-level training
starts. This log records what was actually executed; no E2 artifact exists.

## 2. LeafSnap + BioImages inventory (~13:00)

Sources:

- LeafSnap: `/Users/valkyrie/Desktop/arborphy product/leafsnap/leafsnap-dataset`
- BioImages: `/Users/valkyrie/dev/arborphy/fieldguidevision/scratch/bioimages-ne-trees`

LeafSnap manifest: **30,866 rows / 185 species** (23,147 lab, 7,719 field).
After mask validation: **26,323 usable images** (18,728 lab, 7,595 field);
**4,352 empty masks (14.1%)**, 190 tiny boxes (<0.5% image area), 1 tiny box
(<12 px), 0 missing files, 0 unreadable, 0 duplicate ids. Seven masks cover
>95% of the image and were retained.

BioImages: 1,899 corpus images; **1,106 selected** for the detector across
**403 individuals / 85 species**. Organ counts: bark 197, cone 38, flower 312,
fruit 245, leaf 465, needle 27, seed 12, stem 1, twig 240, unspecified 2,
whole plant 360. Three files were unreadable/truncated and excluded.

Detector dataset (E1/E3 input): **8,560 images** — train 6,171 / val 1,040 /
test 1,349. LeafSnap train/val/test = 5,388 / 880 / 1,186 (33 test species,
field-only); BioImages = 783 / 160 / 163. Class counts (train/val/test):
leaf 5,388/880/1,186; bud 31/10/8; flower 221/40/49; twig 136/29/26;
branch 152/23/22; entire_plant 243/58/58. Species-disjoint (LeafSnap) and
individual-disjoint (BioImages). Validation passed with 0 structural issues;
all retained source images and generated crops were fully decoded.

Deterministic build: salt `leafsnap-yolo26-v1-20261029`; no network, no
training. Outputs under `scratch/leafsnap-yolo26/`.

## 3. EDA headline numbers (~12:45)

Training-free join of LeafSnap to the Dirr vocabulary (`EDA.md`,
`eda_results.json`, `feature_assertions.json`).

**Join.** 165/185 LeafSnap species matched (89.2%): 151 exact + 14 audited
orthographic/synonym/typo matches. 20 unmatched (not force-joined), 0
ambiguous. Six matched species have no asserted Dirr values.

**Top single features** (species-balanced probe IG, one observed value):

| rank | feature | coverage | probe IG | probe NMI | guide IG (upper bound) | probe median candidates |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `leaf_surface_top` | 63.6% | 2.114 | 0.491 | 3.268 | 45.0 |
| 2 | `leaf_shape_overall` | 77.0% | 2.083 | 0.459 | 4.342 | 52.0 |
| 3 | `leaf_shape_margins` | 64.2% | 2.006 | 0.495 | 2.890 | 43.0 |
| 4 | `inflorescences` | 37.0% | 1.902 | 0.496 | 2.310 | 12.0 |
| 5 | `leaf_shape_base` | 56.4% | 1.890 | 0.489 | 2.492 | 36.0 |
| 6 | `leaf_shape_apex` | 57.0% | 1.489 | 0.446 | 1.612 | 80.0 |

The top five are within 11% of each other (2.11–1.89 bits): **no single Dirr
feature is a species key**.

**Best same-view pair** (both readable from one leaf-obverse crop):
`leaf_surface_top` × `leaf_shape_overall` — probe joint IG **3.895 bits**,
synergy **1.781 bits**, n = 96 shared species.

**Best non-local pair** (requires fusing separate observations):
`leaf_shape_overall` (leaf obverse) × `inflorescences` (flower) — probe joint
IG **3.702 bits**, synergy **1.619 bits**, n = 54.

**`leaf_type` coverage and conflicts.** Coverage 124/165 (75.2%): Simple 101,
Compound 21, mixed 2, unlabelled 41 (mostly conifers). It is a coarse splitter,
not a discriminator: probe IG 1.304 bits (exact-set upper bound 1.388); one
observed value leaves 103 (Simple) or 23 (Compound) candidate species; unique
probe resolution 0.0%. Two species assert both Simple and Compound —
`Ginkgo biloba` and `Gymnocladus dioicus` — and must be excluded or adjudicated.

### EDA fusion warning (carried forward)

Non-local joint features are real information only **after fusing separate
observations of the same plant** (leaf obverse, leaf reverse, whole plant,
flower). A same-view pair can share one crop; a non-local pair cannot. Never
assume a non-local pair is simultaneously visible in one image, and never
train a single-crop model to emit both sides of one pair. The high chordal
numbers for leaf × flower / leaf × habit are **not** single-image model targets.

## 4. Bud subview correction (~13:00, in dataset build)

BioImages has no bud-only organ tag. The detector's 49 `bud` images were
recovered **only** from `twig` images whose explicit subview is
`close-up winter terminal bud` or `close-up winter leaf scar/bud`. No
winter-overall twig view was relabelled as bud; other twig views remain
`twig`. Result: bud is data-starved (train 31 / val 10 / test 8), so its
per-class metrics are noisy and one miss moves test recall by ~12 points.
`entire_plant` uses a full-frame weak box; all BioImages boxes are weak
near-full-frame labels, not human localization.

## 5. E3 organ gate — PASS (~13:43)

Pre-registered gate (both must hold at the library default `conf=0.25`):
LeafSnap field leaf recall ≥ 0.80 and leaf mAP50 ≥ 0.80 on the held-out test
split. Threshold selection on validation only; non-leaf recall reported but
never able to offset a leaf failure.

Training: `yolo26n.pt`, MPS, 320 px, batch 16, 20/20 epochs, seed 0, 24.64 min,
6,171 train images.

| metric | observed | threshold | result |
|---|---:|---:|---|
| leaf mAP50 (test, threshold-free) | **0.9935** | ≥ 0.80 | PASS |
| LeafSnap field leaf recall @ conf 0.25 | **0.9916** | ≥ 0.80 | PASS |
| box mAP50 (all classes) | 0.9107 | — | context |
| leaf precision / recall (Ultralytics max-F1) | 0.962 / 0.992 | — | context |
| non-leaf macro recall @ conf 0.25 | 0.8426 | — | context, not gating |
| leaf recall @ val-selected conf 0.5 | 0.9882 | — | secondary |

Leaf mAP50-95 is 0.867 vs mAP50 0.994: boxes are close but not pixel-tight;
the gate is specified at IoU 0.5. ONNX export (opset 17, static 320×320,
batch 1, FP16) verifies at leaf mAP50 0.9937 vs PyTorch 0.9935.

Caveats recorded: LeafSnap field images are curated single-leaf photographs —
this proves the gate finds *those* leaves, not leaves in cluttered scenes.
BioImages weak-box macro recall 0.8272 is a sanity check, not localization.

## 6. E4 V1 leaf specialist — FAIL (~14:01)

Pre-registered criteria (all four required, argmax/full coverage) registered
before training in `EVALUATION_CRITERIA.md`.

| criterion | threshold | observed | result |
|---|---:|---:|---|
| test top1 | ≥ 0.85 | **0.9358** | PASS |
| simple_leaf recall | ≥ 0.85 | **0.9843** | PASS |
| compound_leaf recall | ≥ 0.75 | **0.6996** | FAIL |
| compound_leaf field recall | ≥ 0.70 | **0.6995** (135/193) | FAIL |

Training: `yolo26n-cls.pt`, 224 px, batch 16, 13/30 epochs (early stop),
seed 26, 3.83 min, 3,767 crops. Test: 1,309 crops / 22 species, 82.96% simple.
Macro P/R/F1 0.9214 / 0.8419 / 0.8750. Confusion: 67 compound→simple and
17 simple→compound. Field-only top1 93.11%; lab-only 95.91%. The compound
shortfall is concentrated in `Juglans cinerea` (44.55%, n=101), not spread
across the class — mask-derived crops can isolate a single leaflet that looks
simple but inherits the species-level compound label.

## 7. E4 V2 revision — FAIL, did not fix the mismatch (~14:16–14:23)

Revision intent: use the EDA-audited name mapping (122 unambiguous species:
101 simple, 21 compound, vs V1's 109) and restrict **compound training** to
lab crops, favouring pressed whole-architecture specimens. Simple training and
all validation/test crops were unchanged by that domain rule.

V2 training: 4,042 crops, 17/30 epochs, 4.98 min. Test: 1,475 crops / 24
species.

| criterion | threshold | V2 observed | V1 observed | result |
|---|---:|---:|---:|---|
| test top1 | ≥ 0.85 | 0.9424 | 0.9358 | PASS |
| simple_leaf recall | ≥ 0.85 | 0.9959 | 0.9843 | PASS |
| compound_leaf recall | ≥ 0.75 | **0.6850** | 0.6996 | FAIL |
| compound_leaf field recall | ≥ 0.70 | **0.6729** (144/214) | 0.6995 | FAIL |

Compound precision rose (0.9721) while recall fell: the model became more
conservative on compound and did not recover leaflets. The gap to the field
gate widened from 0.0005 to 0.0271. Aggregate top1 improved solely because
simple recall improved; that cannot substitute for the compound criteria.

### Why V2 did not fix the view/label mismatch

1. **The defect is in the label/test semantics, not the training mix.** Every
   crop inherits its species' single Dirr `leaf_type` assertion. A mask-derived
   field crop can show one leaflet of a compound leaf; that image is genuinely
   leaflet-like while its label says "compound". V2 removed compound field
   crops from *training* but left them in validation and test (45 / 214), so
   the noise floor stayed in exactly the split that determines the gate.
2. **No per-image view or "whole compound / isolated leaflet" label was
   added.** The revision changed which species joined and which training
   domain was sampled; it did not adjudicate what each crop actually shows.
3. **Split membership changed with the wider mapping** (22 → 24 test species;
   compound field support 193 → 214), so aggregates moved for reasons
   unrelated to the intended fix. On the paired species, `Juglans cinerea`
   stayed at ~44% (44.55% → 43.56%; field 56/91 → 57/91 errors) and
   `Albizia julibrissin` got worse (71.05% → 57.89%), while `Koelreuteria
   paniculata` slipped 100% → 95.24%.
4. **Dirr's assertion describes the species, not the image.** Restricting
   training views cannot make a species-level label true for a crop that
   isolates a leaflet.

## 8. Decision

- **E3 is green** and may gate lower-level work.
- **E4 remains FAIL.** Neither V1 nor V2 is validated for tagging or for any
  user-facing leaf-type claim.
- **V1 is the only model carried forward, and only for proposed field
  telemetry** (shadow mode): it had the higher compound and compound-field
  recall and misses the field gate by one crop, while V2's targeted revision
  made the load-bearing numbers worse. Use its ONNX artifact with
  `leaf_specialist.labels.json` (model order `compound_leaf`, `simple_leaf`;
  preprocessing: shortest-edge 224, center crop, [0,1], mean 0/std 1, no
  letterbox, output already softmax).
- Field data collection is **descriptive**, not a validation claim. See
  `FIELD_TEST.md`; next data requirements are in `NEXT_DATA_CURATION.md`.

## 9. Caveats and open gaps

- E1 has no standalone evaluator artifact; its evidence is the dataset
  validation plus the audited Dirr join. E2 is unimplemented (the supplied
  chain marks it TODO).
- Labels are species-level Dirr assertions, not per-image adjudications;
  mixed/no-label species are excluded upstream (Ginkgo, Gymnocladus +
  conifers), so the dataset does not represent mixed-species clusters.
- Crops are tight mask boxes + 10% margin, not raw phone frames. Performance
  may not transfer through the organ gate to unsegmented images.
- Compound estimates rest on few species (V1: 3 test species; V2: 4), with
  wide species-level uncertainty.
- Proof-run budgets (20 and ≤30 epochs, no sweep) mean a FAIL is evidence
  about this budget/data, not a proof about the architecture.
- The adapter defect noted in EDA (`DirrDevoAdapter` drops 49 assertions
  across 4 of 97 tag columns) was avoided here by using the column-accurate
  EDA mapping; the adapter should be fixed before general pipeline use.

## 10. Artifacts (evidence)

| artifact | sha256 / note |
|---|---|
| Detector `dataset.yaml` | `51fb3c0e61680169…` |
| Detector `manifest.jsonl` | `0e1c4ae087ceb682…` |
| V1 classification `manifest.jsonl` | `a3fe4b59731bc5c5…` |
| V2 classification `manifest.jsonl` | `a8cccc810dc33a84…` |
| V2 `labels.json` / `counts.json` | `cc478d0212e98c89…` / `2e473d2c49dd1f02…` |
| `feature_assertions.json` | `43efad068d6afc6d…` |
| E3 metrics | `scratch/leafsnap-yolo26/organ-gate/metrics.json` |
| E4 V1 metrics | `scratch/leafsnap-yolo26/leaf-specialist/metrics.json` |
| E4 V2 metrics | `scratch/leafsnap-yolo26/leaf-specialist-v2/metrics.json` |

## 11. Live organ-gate failure and correction (2026-10-30)

The director's field test contradicted the original E3 result: the app detected
non-leaf organs but almost no leaves. A same-model audit confirmed that this was
not an ONNX/preprocessing error: PyTorch and ONNX chose the same class on all
153 audited images, but V1 produced leaf detections on only 2/37 Tuckaway walk
photos at 0.12, 1/37 at 0.25 and 0/37 at 0.50, with no foliage-video detections.
The original E3 PASS therefore means **curated LeafSnap single-leaf detection**,
not field leaf-in-context detection.

Three corrective datasets were evaluated:

1. **V2, five-class presence model.** Removed `entire_plant` and whole-tree
   training rows, added BioImages leaf photos and training-only multi-leaf
   composites. External leaf firing improved to 27/37 Tuckaway photos and 9/9
   foliage frames, but 25/37 leaf boxes covered at least 90% of the frame. The
   synthetic composites also proved mostly bright pressed-leaf-style scenes.
   V2 was rejected as an automatic crop detector but retained as a leaf-presence
   cue.
2. **Grounded SAM2 proposal review.** 522 real images produced 459 filter-passing
   proposals. Review found useful close-up masks mixed with merged vegetation,
   tree crowns and flower-head false positives. Proposals remain a review queue,
   not ground truth.
3. **V3, mask-supervised YOLO26n-seg.** Removed synthetic images and weak leaf
   boxes and trained with real masks. LeafSnap field box recall was 0.9615 and
   mask AP50 0.9894, but external Tuckaway leaf detection fell to 10.34%,
   foliage video to 0%, and flower-video false leaf detections rose to 18.18%.
   V3 was not integrated.

Final runtime decision: use **V2 only as a five-class organ-presence gate**
(`leaf,bud,flower,twig,branch`). Scene-scale leaf detections display a center
reticle and the specialist receives the fixed center crop; localized boxes may
still be used when their area is at most 70% of the frame. The app records
`crop_mode`, `leaf_box_area`, effective crop and `view_condition`
(`localized_leaf` or `foliage_context`). This is explicit view conditioning, not
a claim that V2 solved leaf localization. `entire_plant` is removed because it
is a view/composition label rather than an organ.

Evidence: `scratch/leafsnap-yolo26/live-audit/AUDIT.md`,
`organ-gate-v2/ORGAN_GATE_V2.md`, `leaf-box-proposals/REVIEW.md`, and
`organ-gate-v3/ORGAN_GATE_V3.md`. The leaf-type specialist remains V1 and E4 NOT
VALIDATED.

## 12. V3 Triangulated Leaf Specialist & Henrico × Tuckaway Regional Generalization

Run 2026-11-03T23:07:05.631910+00:00; `experiments/leafsnap-yolo26/train_leaf_specialist_v3_triangulated.py`; output `scratch/leafsnap-yolo26/leaf-specialist-v3-triangulated/`.

### 12.1 What V3 changes

The V1/V2 residual error was a **view** defect, not an architecture defect: a compound species' mask crop can isolate one leaflet, which looks simple. V3 trains only on view-valid `gold_specialist_train` crops (5,262 whole-foliar-unit crops) and holds the 70 isolated-leaflet crops out of training and out of the E4 test split. E4 thresholds are unchanged (pre-registered).

This supersedes the leaf-type specialist status recorded in §11: E4 is now validated on view-valid crops (all four criteria PASS). The §11 organ-gate runtime decision (V2 five-class presence gate + fixed center crop) is unchanged and independent.

Training: `yolo26n-cls.pt` from the same V1 base checkpoint, imgsz 224, batch 32, epochs 20, patience 6, device mps, seed 26, workers 0; 11 epochs run in 2.51 min.

### 12.2 E4 result (held-out triangulated test, n=1237, 22 species)

**E4 PASS.** test_top1 0.9563 (PASS ≥ 0.85), simple_leaf_recall 0.9633 (PASS ≥ 0.85), compound_leaf_recall 0.9138 (PASS ≥ 0.75), compound_leaf_field_recall 0.9247 (PASS ≥ 0.70).

| version | test top1 | simple recall | compound recall | compound FIELD recall | E4 |
|---|---:|---:|---:|---:|---|
| V1 | 93.58% | 98.43% | 69.96% | 69.95% | FAIL |
| V2 | 94.24% | 99.59% | 68.50% | 67.29% | FAIL |
| V3 | 95.63% | 96.33% | 91.38% | 92.47% | PASS |

The three test splits are species-disjoint but re-materialised per revision; only 21 species appear in all three, so aggregate deltas are descriptive.

**Juglans cinerea** (54 whole-rachis gold test crops): 81.48% (44/54), field 79.55% (35/44). V1 44.55% (n=101) → V2 43.56% (n=101) → V3 81.48% (n=54).

### 12.3 View hard negatives (isolated leaflets)

`hard_negatives_sub_part/test`, n=47 (all Juglans cinerea). At the validation-selected confidence threshold 0.5000: 0/47 abstain (0.00%); margin-based abstention (0.0050) 0/47. Predictions: 47 simple (safe), 0 compound (false compound evidence). Among covered crops, 0/47 are compound.

Validation is saturated, so the pre-registered rule chooses the lowest thresholds and cannot abstain here: the outcome is a confident `simple_leaf` on every isolated leaflet (median confidence 0.9991), not abstention. No compound evidence is claimed, but a deployment gate should not rely on this confidence to detect an isolated-leaflet view.

### 12.4 Mid-Atlantic regional generalization

Gold regional cohort: **49** view-valid Stage-2 regional photos ({'compound_leaf': 33, 'simple_leaf': 16}) across 21 species from {'henrico': 49}. `gold_regional_pool` is not a literal bucket in the current `pool_manifest.jsonl`; the cohort is the gold, view-valid Stage-2 set (`gold_specialist_train` ∩ `stage_2_anatomy`).

**Regional accuracy 40.82% (20/49)**, macro F1 0.3824; compound recall 0.1515, simple recall 0.9375. At the LeafSnap threshold 0.5000, coverage 100.00% and accuracy@coverage 40.82%.

Tuckaway contributes **0** view-valid labeled leaf photos (its only downloaded photo is Stage 3 / context; the 37 walk frames carry no taxon metadata), so this is effectively a Henrico generalization test. Regional labels are species-level Dirr/GoBotany priors on unsegmented iNaturalist photos, and Stage-2 selection is zero-shot CLIP triage (heuristic agreement 50%) — treat the number as a first field-generalization estimate, not an adjudicated benchmark.

### 12.5 Artifacts

`scratch/leafsnap-yolo26/leaf-specialist-v3-triangulated/`: `best.pt`, `leaf_specialist.onnx` (opset 17, static 224, batch 1), `leaf_specialist.labels.json`, `metrics.json`, `LEAF_SPECIALIST_V3.md`, `v1_v2_v3_comparison.json`, `predictions_test.jsonl`, `predictions_hard_negatives_test.jsonl`, `predictions_regional_gold.jsonl`.
