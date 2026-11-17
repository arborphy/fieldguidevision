# /bench image-dataset evolution — LeafSnap/BioImages

Status: design for director review, 2026-11-03. No implementation authorized by this document.

## 1. Current deployed evidence

Two different model/data bundles must remain separate.

### Organ-presence gate (deployed V2)

- Root: `scratch/leafsnap-yolo26/organ-gate-v2/`
- Manifest: `dataset/manifest.jsonl`
- 10,666 images / 16,465 boxes
- train 8,278 / val 1,035 / test 1,353
- Five classes: `leaf,bud,flower,twig,branch`; no `entire_plant`
- Train includes 2,000 synthetic composites (never in val/test)
- 1,212 BioImages images have weak near-full-frame boxes
- Two valid evaluation views:
  - full V2 test: 1,353 images
  - v1-compatible test: 1,291 images (symlinked subset, no added BioImages leaf)
- Validation-selected confidence: 0.45; standard operating confidence: 0.25
- E3 V2 is `FAIL_NOT_ESTABLISHED` for field localization; deployment is presence + view conditioning only

### Leaf-type specialist (deployed V1)

- Root: `scratch/leafsnap-yolo26/classification/`
- Manifest: `manifest.jsonl`
- 5,556 crops
- train 3,767 / val 480 / test 1,309
- 109 species, species-disjoint: 71 train / 16 val / 22 test
- Labels: species-level Dirr `Simple|Compound`, not per-image view-adjudicated truth
- Test: field-heavy (1,089/1,309) and compound-poor (223/1,309)
- E4 FAIL: top1 93.58%, simple recall 98.43%, compound recall 69.96%, compound field recall 69.95%

### External field evidence

- 37 Tuckaway stills + 9 foliage + 11 flower frames
- `external_ground_truth: null`
- Detection rates are not recall and nonleaf candidates are not audited false positives
- Current telemetry stores category, model metadata, gate/lock events and 96 px specialist crops, not full-frame golden images

## 2. Why current Golden Sets cannot hold this

`cheap_look_golden_items` references one image-bearing cheap-look `call_id` and stores expected taxon/organs/confidence/notes. It has no:

- train/val/test membership or split-group contract
- dataset/model version or manifest/split hash
- bbox/polygon/crop geometry
- synthetic/weak/algorithmic/human annotation provenance
- per-view labels (`whole_compound`, `isolated_leaflet`, rachis visibility)
- prediction records, confusion matrices, AP, or metric scope
- bulk manifest import/export, item edit, clone, diff, or immutable version freeze

Do not overload this table. Existing cheap-look Golden Sets remain unchanged.

## 3. Product model: Image Sets beside Golden Sets

Add an **Image sets** surface to `/bench`, separate from the existing Golden Sets tab.

Top-level objects:

```text
Dataset
  id, task, name, description

DatasetVersion (immutable)
  id, dataset_id, version, parent_version_id
  manifest_sha256, split_sha256, class_schema_hash
  source_manifest_uri, builder_revision, split_salt
  status: draft | frozen | retired
  created_by, created_at

ImageAsset (content-addressed)
  asset_id, content_sha256, uri, width, height
  source, source_id, species, individual_id, domain
  license, attribution

Annotation (revisioned)
  annotation_id, asset_id, revision
  task: detection | classification | view_quality
  geometry: bbox | polygon | crop | whole_image
  labels + view metadata
  provenance: human | leafsnap_mask | dirr_species | sam_proposal | synthetic | bioimages_weak
  review_state: proposed | accepted | rejected | adjudicated | weak
  source_vocabulary, annotator, notes

SplitMembership
  dataset_version_id, asset_id, split: train | val | test | external_audit
  split_group_id (species/individual/source-video as appropriate)

EvaluationRun
  id, model_id, model_version, dataset_version_id, evaluation_view
  split, prediction_manifest_uri, prediction_sha256
  evaluator_revision, metric_profile, operating_point_selection
  metrics_json, status

FieldSession
  id, telemetry_package_uri, app/model hashes
  review_state, linked_evaluation_run_id (optional)
```

Artifacts (images, masks, JSONL predictions, metrics) live in content-addressed file/GCS storage. Postgres stores searchable metadata and pointers; do not inline thousands of images or large prediction blobs.

## 4. Evidence classes must remain visible

Every item and metric slice carries one of:

- **Adjudicated golden** — human-reviewed image/view/geometry
- **Curated algorithmic** — LeafSnap masks or Dirr species labels, with limitations
- **Weak** — BioImages near-full-frame boxes or unreviewed SAM proposals
- **Synthetic** — composites, training only
- **External unlabeled** — Tuckaway/field telemetry, not accuracy evidence until reviewed

A single green `test` badge is not sufficient. Weak/synthetic/unlabeled items must never silently become golden.

## 5. Read-only Slice 0 — backsolve current splits

Implement this first. No retraining or annotation editing.

### Import

1. Import V2 `dataset/manifest.jsonl` as `organ-gate-v2` version 1.
2. Register both evaluation views: `full-test` (1,353) and `v1-compatible-test` (1,291). Preserve symlink targets or deduplicate by content hash while retaining both view definitions.
3. Import V1 `classification/manifest.jsonl` as `leaf-type-v1` version 1.
4. Import `predictions_val.jsonl`, `predictions_test.jsonl`, metrics, evaluation criteria and model label contracts as EvaluationRuns.
5. Import Tuckaway/video as `external-audit`, explicitly unlabeled.

### Dataset list

Columns: name, version, task, items, class count, split counts, evidence badges, manifest hash, evaluation status.

### Dataset detail

- Split tabs: Train / Val / Test / External audit
- Filters: class, source, domain, species, individual, synthetic, weak, view, review state
- Image grid with bbox/crop overlay and model prediction toggle
- Item inspector: image, geometry, manifest record, source provenance, split group, prediction/confidence and correctness
- Stats panel: class/source/domain/view counts and split-integrity checks

### Metric views

- Explicit scopes: training history (val), validation-selected operating point, held-out test, external audit
- Per-class P/R/F1/AP and confusion matrix where applicable
- Source/domain/species/view slices
- Model comparison only on the same dataset version and evaluation view
- Link every aggregate to the underlying prediction/image rows

### Slice 0 acceptance checks

- V2 counts are exactly 8,278 / 1,035 / 1,353; both test views are visible
- V1 counts are exactly 3,767 / 480 / 1,309
- 2,000 composites appear only in train and carry `synthetic`
- 1,212 weak BioImages rows are visibly weak and excluded from localization-quality claims
- Filtering V1 test to `Juglans cinerea` shows 101 crops and their predictions/confusion
- `Juglans` examples exposing isolated leaflets are viewable beside whole compound leaves
- Tuckaway items say `external_ground_truth: null`; no recall/FPR is displayed
- E4 remains failed; no UI aggregate suggests the specialist is validated

## 6. Slice 1 — evaluation workbench

- Run an exported ONNX/PyTorch model against a frozen evaluation view
- Store raw predictions before thresholding so operating points can be replayed
- Compare models on identical items, including paired per-image deltas
- Regression gates reuse pre-registered criteria; val selects thresholds, test does not tune them
- Detection evaluator generalizes the existing V1 one-box and V2 multi-box forks into a class-configurable evaluator
- Classification metrics include per-class/domain/species/view and coverage/abstention

## 7. Slice 2 — image-set iteration

- Review proposed boxes/labels and accept/reject/adjudicate
- Add per-frame view labels: `whole_simple`, `whole_compound`, `isolated_leaflet`, `compound_fragment`, `not_assessable`, rachis visibility and pair/individual IDs
- Clone a frozen version to `draft+1`; never mutate a split used by an earlier evaluation
- Diff versions: added, removed, relabelled, geometry changed, split moved, review-state changed
- Freeze creates new manifest/split hashes and a reproducible release report
- Leakage checks: content hash, species, individual, burst/video sequence and pair IDs

## 8. Slice 3 — field telemetry to golden evidence

- Import `arq.telemetry/v1` packages and associate app/model hashes and test category
- Preserve raw gate/lock dynamics separately from classification accuracy
- Future mobile option: opt-in full-frame capture for items selected for annotation; 96 px thumbnails are review hints, not golden images
- Annotate external media, then promote accepted items to a new golden dataset version
- Report bench and field metrics side by side without mixing their denominators

## 9. API shape (indicative)

```text
POST /api/bench/datasets/import
GET  /api/bench/datasets
GET  /api/bench/datasets/{id}/versions/{version}
GET  /api/bench/datasets/{id}/versions/{version}/items?split=&class=&species=&view=
GET  /api/bench/datasets/{id}/versions/{version}/stats
POST /api/bench/datasets/{id}/versions/{version}/freeze
POST /api/bench/datasets/{id}/versions/{version}/clone
GET  /api/bench/datasets/{id}/versions/{a}/diff/{b}
POST /api/bench/evaluations/import
POST /api/bench/evaluations/run
GET  /api/bench/evaluations/{id}
POST /api/bench/field-sessions/import
```

Use a new router such as `routers/bench_datasets.py`; do not depend on the currently unmerged fieldvision Runs backend.

## 10. Immediate director decision

Recommended sequence:

1. **Slice 0 now** — faithful read-only import/view of the deployed V2 and V1 manifests, predictions and metrics.
2. **Slice 1 next** — repeatable model comparison on frozen evaluation views.
3. **Slice 2 after that** — view adjudication and immutable dataset revisions, prioritized on compound whole-leaf vs isolated-leaflet evidence.
4. **Slice 3 continuously** — field sessions remain the external reality check and become golden only after review.

This gives a useful bench before another model run while preserving the field test as an independent measure of runtime behavior.
