# Anatomy4 conditioned leaf & flower experts

Sprint: **anatomy4-specialists**. This package implements the conditioned
expert trainers and their data tooling. It does **not** run training and does
**not** contact the app DB except through the read-only claim-join command.

Status: **implementation complete, no training run performed.** No metric in
this repository is a training result; the only numbers recorded are the
Shenandoah *support* counts used to choose targets, below.

## Required pair (and optional leaf targets)

| expert | target id | classes | eligible crop anatomy |
|---|---|---|---|
| leaf (required) | `leaf_simple_vs_compound` | Simple, Compound | `leaf` |
| flower (required) | `flower_inflorescence_type` | Raceme, Corymb, Cyme, Panicle | `inflorescence` |
| leaf (config-driven) | `leaf_margin` | Entire, Serrate, Lobed | `leaf` |
| leaf (config-driven) | `leaf_arrangement` | Alternate, Opposite | `leaf` |

The trainer is generic: each target is one run driven by its config. Start
with the required pair; add the leaf-margin/arrangement runs only after the
simple-vs-compound expert is validated.

## Measured Shenandoah support (target choice input, not a model metric)

Read-only query against the local app DB
(`docker exec arq-compose-postgres psql -U arq -d arq`) on 2026-11-12:
Shenandoah area `inst-osm-relation-5548542`, plant species with photos joined
to the Dirr `devo_taxa` wide table on the lower-case binomial.

| target | class | species | observations |
|---|---|---:|---:|
| leaf | Simple | 70 | 545 |
| leaf | Compound | 18 | 151 |
| flower | Raceme | 7 | 77 |
| flower | Corymb | 7 | 74 |
| flower | Cyme | 8 | 76 |
| flower | Panicle | 11 | 55 |
| leaf margin | Entire | 29 | 231 |
| leaf margin | Serrate | 32 | 224 |
| leaf margin | Lobed | 16 | 122 |
| arrangement | Alternate | 65 | 534 |
| arrangement | Opposite | 26 | 183 |

3,431 species have observations in the area; 107 join to Dirr exactly
(798 observations). These are **claim support**, not crop counts and not
accuracy. The join excludes species with multiple in-target values and any
in-feature value outside the target, so real crop support will be lower and
must be measured with `support` after crops exist. Dirr-wide, after that
ambiguity gate, eligible species are: leaf simple/compound 753 (641/112),
inflorescence type 149 of 265 species with a claim (Panicle 51, Corymb 27,
Raceme 30, Cyme 41), margin 426, arrangement 753. The flower source is
noticeably lossy because Dirr also uses head/umbel/spike/catkin/spadix,
which are outside this sprint's four-class target.

## Input contract

The crop manifest is JSONL, one row per crop:

```json
{
  "observation_id": "inat-383987290",
  "taxon_name": "Quercus rubra",
  "image": "https://.../large.jpg",
  "crop_path": "/srv/fast/.../crops/abc123.jpg",
  "anatomy_class": "leaf",
  "detector_confidence": 0.82,
  "observation_split": "train",
  "crop_id": "abc123",
  "bbox": [0.21, 0.18, 0.64, 0.71],
  "detector_box_source": "detector"
}
```

Accepted aliases: `photo_url`/`source_url` for `image`; `confidence`/
`box_score` for `detector_confidence`; `split` for `observation_split`;
`image_id` for `photo_id`. `target_label` is added by the Dirr join.

## Pipeline

### 1. Build crops from **reviewed detector boxes**

`crops.build_crops` consumes the anatomy4 prelabel document (`images` +
`proposals`) and a review CSV (`proposal_id,verdict`). It:

- accepts only `tp`/`accept`/`accepted` proposals (or `--allow-unreviewed`,
  which is explicit and recorded);
- joins taxon names and photo URLs from `--observations` (the anatomy4
  manifest or a JSONL export). Real prelabel proposal documents do not carry
  `taxon_name` or an image URL, so this input is required in practice and rows
  lacking either are skipped with a recorded reason;
- requires a detector box (`detector_bbox`/`bbox`);
- **rejects** `box_source=taxon`, `taxon_derived_box=true`, missing boxes, and
  off-screen/degenerate boxes. There is no codepath that builds a box from
  `taxon_name`;
- pads each side by `padding_frac` of the box dimension (default 0.18),
  clamps to the image, and grows to `min_crop_px`;
- writes `crop_manifest.jsonl` + `crop_build_report.json`.

### 2. Join Dirr weak labels with a hard anatomy gate

`dirr_join.join_weak_labels` reads claims either from a pre-exported JSON or
read-only from the app DB. DB loading prefers `devo_taxon_claims` and falls
back to the wide `devo_taxa` one-hot table (the local Dirr shape).

**A claim is never image evidence unless the crop's anatomy matches the
feature**: `leaf_*` features require `leaf` crops; `inflorescences` requires an
`inflorescence` crop. An isolated `flower` crop does not carry arrangement
evidence. Rows are returned with `weak_label_eligible`, `weak_label_reason`,
`claim_values`, and (only when eligible) `target_label` and
`label_source="dirr"`. If any claim value for an in-target feature falls
outside the target, the row is `ambiguous_claim` even when another value is
in-target. Mixed simple+compound and multi-value inflorescence claims are also
excluded. Taxonomy is exact-species (binomial); author citations are stripped.
`Forsythia × intermedia` and `Forsythia xintermedia` match through an attached-
`x` alias, but no other fuzzy matching is done.

DB loading unions `devo_taxon_claims` with the wide `devo_taxa` one-hot table
per requested feature, so a partially populated claims table cannot hide a
feature that only exists in the wide table.

### 3. Validate and split

`dataset.validate_dataset` is a hard gate: required fields, anatomy classes,
split values, confidence, duplicate ids/paths, file existence (optional),
label-in-target, **no label on a mismatched anatomy**, class support in
train/val, and split isolation.

`split.assign_splits` provides:

- `observation` — all crops of an observation in one split (species may span);
- `species` / `species_then_observation` — whole species in one split, which
  is also observation-disjoint.

Both are deterministic SHA-256 allocations and both write an isolation report.
When the trainer supplies the target classes, `assign_splits` searches seeds
deterministically until train and val contain every class at the requested
`--min-per-class` count (200 attempts by default) instead of letting an
unlucky hash abort a thin flower class. If no seed can satisfy the count,
validation reports the missing split/class instead of silently training.
`build-dataset` recomputes splits by default (`--use-manifest-splits` only when
the manifest was produced by the same splitter — pass `--split-mode
observation` for observation-level manifests, `species` for species-level).

### 4. Train (conditioned expert)

`trainer.train` fine-tunes `yolo26n-cls.pt` with the LeafSnap-YOLO26 pattern
(imgsz 224, fixed seed, deterministic, per-class metrics, confusion matrix,
validation-only abstain threshold) plus:

- **CUDA device support** (`--device auto` → cuda, then mps, then cpu);
- **bounded memory** for the .245 RTX 4060 Ti 8 GB:
  - `max_vram_fraction` default **0.30** → ~2.4 GB process cap, leaving room
    for the ~4.3 GB BioCLIP sidecar;
  - preflight requires **free VRAM >= process budget + margin**
    (`min_free_vram_gb` default 0.5 → 2.9 GB free for the defaults). It
    refuses rather than racing BioCLIP; lower the fraction or free a consumer
    deliberately, never by stopping a service automatically;
  - the same cap and preflight apply to standalone `evaluate`;
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
  - fixed batch (32 leaf / 24 flower) and imgsz 224, no auto-batch;
  - disk cache by default, capped workers (2), AMP on CUDA;
  - a hard refusal if the dataset validator has not passed.
- `--resume` is real: it requires `runs/train/weights/last.pt` and loads that
  checkpoint with Ultralytics `resume=True`; it never silently starts fresh.
- no service restart, no DB write, no deployment.

### 5. Evaluate, abstain, contact sheets

`evaluate.evaluate_run` writes metrics **only** from a real checkpoint and
real predictions:

- top-1 and top-k accuracy;
- macro precision/recall/F1 over present classes (and an all-class secondary);
- per-class precision/recall/F1 and support;
- confusion matrix (rows actual, columns predicted);
- abstain/unknown: confidence **and** margin thresholds selected on
  validation only, coverage, selective accuracy/macro-F1, unknown rate, and a
  confusion matrix including `abstain` as its own predicted column;
- **both recall definitions** are reported: all-row recall (abstentions count
  as false negatives) and covered-only recall (abstained rows excluded).
- class order is fixed: Ultralytics sorts class directories alphabetically, so
  `model.names` is loaded, the probability columns are reordered into the
  config's `target.classes`, the reorder is recorded in `metrics.json`
  (`class_order`), and a class-set mismatch is a hard error.
- probabilities are not softmaxed twice: Ultralytics classification output is
  detected as probabilities (rows sum to 1) and passed through; raw logits are
  softmaxed exactly once.

`contact_sheet.render_contact_sheets` renders GT crops per split × label, with
taxon, anatomy, detector confidence, and label in the caption. Pass
`--dataset <built dataset root>` (not the pre-split crops) so the sheet uses
the actual trained `manifest.jsonl` split; fonts use DejaVu or Pillow's bundled
default (no macOS-only path).

## Exact commands for the later run (once detector crops exist)

Claim export happens on the **Mac** (local app DB); the join runs remotely on
** .245** from the exported JSON. `join-labels` and `support` now require
exactly one explicit source: `--db` or `--claims-json`.

```bash
# === On the Mac (local DB) =================================================
cd /Users/valkyrie/dev/fieldguidevision-anatomy4-specialists
.venv/bin/python -m experiments.anatomy4_specialists.cli export-claims \
  --features leaf_type leaf_shape_margins leaf_arrangement inflorescences \
  --out /tmp/dirr-claims.json
scp /tmp/dirr-claims.json valkyrie@10.0.0.245:/srv/fast/anatomy4-specialists/dirr-claims.json

# === On .245 ================================================================
cd /srv/fast/fieldguidevision-anatomy4-specialists   # repo checkout on .245
V=.venv/bin/python

# 0) Build crops from reviewed detector proposals; --observations supplies
#    taxon names + photo URLs that the proposals themselves do not carry.
$V -m experiments.anatomy4_specialists.cli build-crops \
  --proposals /srv/fast/anatomy4-shenandoah/prelabel/proposals.json \
  --observations /srv/fast/anatomy4-shenandoah/manifest.json \
  --review /srv/fast/anatomy4-shenandoah/overlays/review.csv \
  --out /srv/fast/anatomy4-specialists/crops \
  --padding 0.18 --min-crop-px 48

# 1) Join Dirr claims from the exported JSON (no DB access on .245).
$V -m experiments.anatomy4_specialists.cli join-labels \
  --crops /srv/fast/anatomy4-specialists/crops/crop_manifest.jsonl \
  --target-config experiments/anatomy4_specialists/configs/leaf_simple_vs_compound.json \
  --claims-json /srv/fast/anatomy4-specialists/dirr-claims.json \
  --out /srv/fast/anatomy4-specialists/leaf_simple_vs_compound/crops_labeled.jsonl

$V -m experiments.anatomy4_specialists.cli join-labels \
  --crops /srv/fast/anatomy4-specialists/crops/crop_manifest.jsonl \
  --target-config experiments/anatomy4_specialists/configs/flower_inflorescence_type.json \
  --claims-json /srv/fast/anatomy4-specialists/dirr-claims.json \
  --out /srv/fast/anatomy4-specialists/flower_inflorescence_type/crops_labeled.jsonl

# 2) Measure actual crop support (record it; no metric claims).
$V -m experiments.anatomy4_specialists.cli support \
  --crops /srv/fast/anatomy4-specialists/leaf_simple_vs_compound/crops_labeled.jsonl \
  --target-config experiments/anatomy4_specialists/configs/leaf_simple_vs_compound.json \
  --claims-json /srv/fast/anatomy4-specialists/dirr-claims.json \
  --out /srv/fast/anatomy4-specialists/leaf_simple_vs_compound/support.json

# 3) Validate/build the classification tree, then render sheets from the
#    built dataset (use its manifest split, not the pre-split crops).
$V -m experiments.anatomy4_specialists.cli build-dataset \
  --crops /srv/fast/anatomy4-specialists/leaf_simple_vs_compound/crops_labeled.jsonl \
  --target-config experiments/anatomy4_specialists/configs/leaf_simple_vs_compound.json \
  --out /srv/fast/anatomy4-specialists/leaf_simple_vs_compound/dataset \
  --split-mode species --min-per-class 2

$V -m experiments.anatomy4_specialists.cli contact-sheet \
  --dataset /srv/fast/anatomy4-specialists/leaf_simple_vs_compound/dataset \
  --target-config experiments/anatomy4_specialists/configs/leaf_simple_vs_compound.json \
  --out /srv/fast/anatomy4-specialists/leaf_simple_vs_compound/contact_sheets

# 4) Dry-run, then the single bounded run (defaults: 0.30 of 8 GB + 0.5 GB).
$V -m experiments.anatomy4_specialists.cli train \
  --config experiments/anatomy4_specialists/configs/leaf_simple_vs_compound.json \
  --dry-run

$V -m experiments.anatomy4_specialists.cli train \
  --config experiments/anatomy4_specialists/configs/leaf_simple_vs_compound.json \
  --device cuda --batch 32

# Optional real resume (requires runs/train/weights/last.pt):
#   ... train --config ... --resume

# Flower expert: same flow with the flower config (batch 24).
$V -m experiments.anatomy4_specialists.cli build-dataset \
  --crops /srv/fast/anatomy4-specialists/flower_inflorescence_type/crops_labeled.jsonl \
  --target-config experiments/anatomy4_specialists/configs/flower_inflorescence_type.json \
  --out /srv/fast/anatomy4-specialists/flower_inflorescence_type/dataset \
  --split-mode species --min-per-class 2

$V -m experiments.anatomy4_specialists.cli train \
  --config experiments/anatomy4_specialists/configs/flower_inflorescence_type.json \
  --device cuda --batch 24
```

Install the pinned training environment with the `train` extra
(`ultralytics==8.4.190`, Torch, NumPy, Pillow, pytest):

```bash
.venv/bin/pip install -e ".[train]"
```

Optional leaf runs swap in `configs/leaf_margin.json` or
`configs/leaf_arrangement.json`. Observation-disjoint evaluation uses
`--split-mode observation`; `--use-manifest-splits` must match the splitter
that produced the manifest.

## Two-stage cascade on `.245` (Stage-1 YOLO -> Stage-2 experts)

`detect-crops` runs the real four-class Stage-1 `anatomy4` YOLO checkpoint over
the staged Dirr manifest, materializes padded crops, and writes
`crop_manifest.jsonl` + `detect_crops_report.json`. Every row is marked
`detector_box_source="stage1_anatomy4_yolo"` and
`review_verdict="stage1_predicted"`; the validator accepts both. No taxon is
used to derive a box.

The `.245` container uses the already-built `anatomy4-detector:0.1` image and
CDI GPU passthrough (`--device nvidia.com/gpu=all`; `--gpus all` fails on this
Docker setup). BioCLIP is resident at ~4.3 GB, so detection uses
`--max-vram-fraction 0.35` and training uses the config's `0.30`.

```bash
# === On .245, after Stage-1 training finishes ==============================
cd /srv/fast/anatomy4-specialists
STAGE1_BEST=$(find /srv/fast/anatomy4-specialists /srv/fast/anatomy4-shenandoah \
  -name best.pt -path '*anatomy4*' | head -1)
STAGE1_DATASET=/srv/fast/anatomy4-shenandoah/datasets/anatomy4-train-mixed-v1/dataset.json
CLAIMS=/srv/fast/anatomy4-specialists/dirr-claims.json
MANIFEST=/srv/fast/anatomy4-specialists/dirr-images-manifest.jsonl
ROOT=/srv/fast/anatomy4-specialists
RUN='docker run --rm --device nvidia.com/gpu=all   -v /srv/fast/anatomy4-specialists/repo:/workspace   -v /srv/fast/anatomy4-specialists:/srv/fast/anatomy4-specialists   -w /workspace anatomy4-detector:0.1'

# 1) Stage-1 detection once; crops are shared by all Stage-2 targets.
$RUN python -m experiments.anatomy4_specialists.cli detect-crops \
  --weights "$STAGE1_BEST" \
  --dirr-manifest "$MANIFEST" \
  --stage1-dataset "$STAGE1_DATASET" \
  --exclude-null-license \
  --conf 0.25 --iou 0.50 --padding 0.18 --min-crop-px 48 \
  --max-crops-per-image 6 --device 0 --max-vram-fraction 0.35 \
  --out "$ROOT/stage1-crops"

# 2) Weak-label join per target.
for TARGET in leaf_simple_vs_compound leaf_margin flower_inflorescence_type; do
  $RUN python -m experiments.anatomy4_specialists.cli join-labels \
    --crops "$ROOT/stage1-crops/crop_manifest.jsonl" \
    --target-config "experiments/anatomy4_specialists/configs/$TARGET.json" \
    --claims-json "$CLAIMS" \
    --out "$ROOT/$TARGET/crops_labeled.jsonl"
done

# 3) Species-disjoint 70/15/15 dataset, preferring Stage-1-unseen test groups.
#    Leaf targets require 2/class; the thin flower target uses 1/class.
for TARGET in leaf_simple_vs_compound leaf_margin; do
  $RUN python -m experiments.anatomy4_specialists.cli build-dataset \
    --crops "$ROOT/$TARGET/crops_labeled.jsonl" \
    --target-config "experiments/anatomy4_specialists/configs/$TARGET.json" \
    --out "$ROOT/$TARGET/dataset" --split-mode species \
    --holdout-prefer-stage1-unseen --min-per-class 2
done
$RUN python -m experiments.anatomy4_specialists.cli build-dataset \
  --crops "$ROOT/flower_inflorescence_type/crops_labeled.jsonl" \
  --target-config experiments/anatomy4_specialists/configs/flower_inflorescence_type.json \
  --out "$ROOT/flower_inflorescence_type/dataset" --split-mode species \
  --holdout-prefer-stage1-unseen --min-per-class 1

# 4) Train + evaluate the held-out test split (evaluate also writes val).
for TARGET in leaf_simple_vs_compound leaf_margin flower_inflorescence_type; do
  $RUN python -m experiments.anatomy4_specialists.cli train \
    --config "experiments/anatomy4_specialists/configs/$TARGET.json" --device 0
  $RUN python -m experiments.anatomy4_specialists.cli evaluate \
    --config "experiments/anatomy4_specialists/configs/$TARGET.json" \
    --out "$ROOT/$TARGET/runs/eval-test" --split test --device 0
done
```

Observation-disjoint variants use
`--split-mode observation` (and a distinct `--out`, `--output-dir` for train).
`evaluate --split test` still selects the abstain threshold on val and reports
both split metrics in `metrics.json`; `split_mode`, `prefer_test_unseen`, and
`stage1_overlap_counts` are copied from the built dataset.

If Stage-1 weights are not yet at a discoverable path, set `STAGE1_BEST`
explicitly. `detect_crops_report.json` records the model class order, per-class
counts, unlocalized images, and Stage-1 overlap flags.

## Tests

```bash
# From the repo root, with the project venv.
.venv/bin/python -m pytest experiments/anatomy4_specialists/tests -q
```

Current result: **94 passed**. Coverage: split isolation and class-aware seed
search, weak-label anatomy filtering/ambiguity/hybrid aliases, metric math
(P/R/F1, macro, confusion, top-k, abstain threshold, both recall definitions),
crop padding/box-source rejection/observation lookup, dataset
validation/build, probability/class-order handling, VRAM arithmetic and
evaluate cap, resume resolution, explicit claim-source CLI, union claims
fallback, dataset-manifest contact sheets, and trainer guards.

## Deliberate non-goals / known limits

- No training run, no checkpoint, no metric. The README's support table is
  claim support, not performance.
- Crops come from the detector; taxon-derived or rectangle-fallback boxes are
  rejected by design. A human review sheet is required.
- Dirr weak labels are taxon-level, exact-species only. Mixed/multi-value
  claims, and any in-feature value outside the target, are excluded rather
  than resolved.
- The attached-`x` hybrid alias makes `Forsythia xintermedia` and
  `Forsythia × intermedia` collide, but could also alias a genuine
  x-starting epithet (e.g. `Salix xanthicola` → `Salix anthicola`) when such a
  claim exists. Treat hybrid joins as a data-quality review item.
- The flower expert's label space is imbalanced and thin, and now requires
  **inflorescence** crops (isolated flower crops are rejected). Expect
  species-level support to collapse when multi-value claims are excluded.
  Measure `support` before deciding the run is viable; if the class count is
  tiny, use `--min-per-class 1` deliberately rather than lowering it silently.
- A single leaf close-up may not show arrangement; the arrangement config is
  config-driven and should be reviewed with contact sheets first.
- No ONNX export is included in this sprint. The trainer writes the
  checkpoint + measured metrics; export can follow the LeafSnap pattern.
- No app DB writes, no service restart, no deployment, no commit.
