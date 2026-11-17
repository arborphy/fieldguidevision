# anatomy4_detector — four-class organ detector pipeline

Reusable, auditable pipeline for the Shenandoah `anatomy4-v1` sprint:

| class id | class |
| --- | --- |
| 0 | `leaf` |
| 1 | `needle` |
| 2 | `flower` |
| 3 | `inflorescence` |

It materializes the audit export (`proposals.json` + `review.csv` + images)
into a validated dataset, trains `yolo26n.pt` (DETECT, not `-seg`), runs real
held-out evaluation on the **test** split, renders GT-vs-prediction audits,
exports ONNX under the arq-api contract, and writes one run manifest.

No full training was run against the real dataset by this package (see
**Blockers** for the remaining review-coverage limits).

## Dataset contract

The reviewed label artifact is **normalized xyxy**:

```
<class_id> <x0> <y0> <x1> <y1>      # all coordinates in [0, 1], x0 < x1, y0 < y1
```

Required layout:

```
<root>/
  dataset.json                  # recommended; authoritative when present
  images/<train|val|test>/<stem>.<jpg|png|...>
  labels/<train|val|test>/<stem>.txt
```

`dataset.json`:

```json
{
  "schema": "arq.anatomy4.dataset/v1",
  "contract": "anatomy4-v1",
  "classes": ["leaf", "needle", "flower", "inflorescence"],
  "images": [
    {"image": "images/train/123.jpg", "observation_id": "obs-123", "negative": false},
    {"image": "images/train/456.jpg", "observation_id": "obs-456", "negative": true}
  ]
}
```

Rules enforced by `validate`:

* class ids are exactly `0..3` in contract order;
* **every image has a label file**, including negatives (an explicit, empty
  file). A missing file for a negative is an error, not an implicit negative;
* boxes are normalized xyxy inside the frame;
* `dataset.json` paths/split/negative flags agree with disk;
* **no `observation_id` appears in more than one split**;
* `split_map.json` may substitute for `dataset.json` observation ids
  (`{"<image relpath or stem>": "<observation id>"}`);
* when observation ids are absent the audit says
  `split_isolation: "unverifiable"` instead of pretending it is clean;
* per-class support is counted, never estimated; AP for a class with zero test
  support is `null` and excluded from the macro average.

The Ultralytics `cx cy w h` form exists only in the generated `prepared/`
training view; the reviewed artifact stays xyxy.

## Materialize the audit export

`materialize` converts the DINO/YOLOE artifacts into the contract above.

Targeted audit output (the merged per-stratum run, **recommended**):

```bash
$P materialize \
  --proposals /srv/fast/anatomy4-shenandoah/targeted/proposals.json \
  --review /srv/fast/anatomy4-shenandoah/targeted/review.csv \
  --manifest /srv/fast/anatomy4-shenandoah/targeted/manifest.downloaded.json \
  --images-dir /srv/fast/anatomy4-shenandoah/targeted/images \
  --background-review /srv/fast/anatomy4-shenandoah/targeted/review-background-filled.csv \
  --out /srv/fast/anatomy4-shenandoah/dataset-anatomy4-targeted
```

The earlier 55-accepted export still works (no background review exists for it):

```bash
$P materialize \
  --proposals /srv/fast/anatomy4-shenandoah/prelabel-yoloe2/proposals.json \
  --review /srv/fast/anatomy4-shenandoah/v2-review.csv \
  --manifest /srv/fast/anatomy4-shenandoah/manifest.downloaded.json \
  --images-dir /srv/fast/anatomy4-shenandoah/images \
  --out /srv/fast/anatomy4-shenandoah/dataset-anatomy4-v1
```

Semantics (all enforced and unit-tested):

* proposal ids join to `review.csv`; only `verdict = tp` (or
  `accept`/`accepted`) becomes truth. `fp`, `unsure`, and **unreviewed**
  proposals never become boxes;
* boxes are the proposal's normalized **xyxy** `bbox` (falling back to
  `detector_bbox`, then the tight polygon bbox). Values are copied exactly
  once; a `cx cy w h` box is rejected, never reinterpreted;
* `provenance.jsonl` keeps one row per accepted box: proposal id, confidence,
  detector label/prompt, `model` (detector id, config fingerprint), mask IoU,
  created-at, and the review verdict/notes/sheet row;
* an image becomes an explicit negative (empty label file) only when **all**
  its proposals were reviewed and **all** were rejected
  (`--negative-policy reviewed-rejected`, the default). Images with no
  proposals are never assumed negative; they need `--negative-image`/`--negative-file`
  (explicit auditor selection) or the optional image-level background review
  (`--background-review`, `review-background.csv`: `empty_correct` = explicit
  negative, `missed_organ` = never a negative). `--negative-policy
  explicit|none` are available;
* exact duplicate accepted boxes on one image (same class + same xyxy, common
  when merged prompt passes detect the same organ) are collapsed to the
  highest-score proposal and recorded in `materialization.json`
  (`--keep-duplicate-boxes` disables). Boxes are not NMS'd otherwise;
* with `--classes leaf,needle`, only those boxes are written and reviewed
  negatives are retained only for images that actually reviewed a selected
  class;
* splits are observation-safe (`split_for_observation`, 70/15/15); the adapter
  refuses to finish if validation finds leakage;
* output: `dataset.json`, `images/`, `labels/` (explicit empty negatives),
  `provenance.jsonl`, `materialization.json` (input SHA-256s, policies, counts,
  partial-labeling warnings), `data.yaml`, `dataset-audit.{json,md}`.

Real results (run on .245):

```text
targeted/ (171 reviewed proposals, 20 empty_correct background verdicts)
  images 76 (55 positives, 21 negatives), boxes 70 (2 exact duplicates removed)
  leaf 33, needle 9, flower 14, inflorescence 14
  train 54 img / 51 boxes, val 14 / 9, test 8 / 10
  split_isolation clean; status ok; test support leaf4 needle1 flower2 inflor3
  partial labeling: 55 labeled images also carry 975 unreviewed proposals

v2/yoloe2 55-accepted export
  images 42 (33 positives, 9 reviewed negatives), boxes 55
  leaf 21, needle 4, flower 14, inflorescence 16
  train 31 img / 46 boxes, val 3 / 2, test 8 / 7
  split_isolation clean; status ok
  partial labeling: 25 labeled images also carry 111 unreviewed proposals
```

Partial annotation is honest in the manifest, not hidden: unreviewed proposals
on labeled images are unlabeled and therefore treated as background by the
trainer. `--require-fully-reviewed` emits the stricter subset instead (drops
those images; not the default, because the reviewed sample is score-stratified,
not exhaustive).

## Combine mixed reviewed sources (train/val)

`combine` builds the real Shenandoah train/val dataset from the reviewed source
streams. It refuses to run until the three expected source families exist
(unless `--allow-missing-sources` is passed for a deliberate partial build):

```bash
$P combine \
  --out /srv/fast/anatomy4-shenandoah/datasets/anatomy4-train-mixed-v1 \
  --assisted /srv/fast/anatomy4-shenandoah/train-candidates/verified/assisted-labels.jsonl \
  --curated-needle "/srv/fast/anatomy4-shenandoah/train-candidates/curated-needle/batch-*/annotations.jsonl" \
  --negative-review "/srv/fast/anatomy4-shenandoah/train-candidates/negative-review/batch-*.jsonl" \
  --images-root /srv/fast/anatomy4-shenandoah/train-candidates \
  --candidate-manifest /srv/fast/anatomy4-shenandoah/train-candidates/manifest.jsonl \
  --blind-manifest /srv/fast/anatomy4-shenandoah/blind-test/merged/manifest.jsonl \
  --blind-annotations /srv/fast/anatomy4-shenandoah/blind-test/merged/annotations.jsonl \
  --seed anatomy4-train-v1 --train-ratio 0.85 \
  --overlays 8
```

(Inside the trainer container, replace `/srv/fast/anatomy4-shenandoah` with
`/out`; add `--review-queue <path>` for later manual-review outputs.)

Semantics:

* only accepted/confirmed records and labels contribute truth. `needs_review`,
  `proposed`, `rejected` and unknown review states are written to
  `quarantine/{needs_review,unreviewed}.jsonl`, never silently dropped;
* negative-review batches are mixed by design: `negative=true` rows become
  explicit empty-label negatives, while `negative=false` rows are reviewed
  exclusions (positives/uncertain/license), counted as
  `excluded_non_negative`, written to `excluded_non_negative.jsonl`, and
  skipped cleanly — they are not invalid records and not quarantine errors;
* a negative record for an image that has an accepted positive is invalid: the
  positive wins and the negative goes to `quarantine/conflicts.jsonl`
  (`resolved: positive_wins`);
* identical `class + bbox` (6 dp) boxes from multiple sources collapse to one
  box; `provenance.jsonl` keeps every source (file, row, policy/model,
  confidence, reviewer provenance) and `merge-report.json` counts the merges;
* canonical image identity is the candidate manifest id, so review-local ids
  such as `cn2-tc-…` merge correctly with `tc-…` records from other sources;
* all photos of one observation share a deterministic split (default 85/15,
  `split_for` hash over the observation id); train/val must both be non-empty;
* the frozen blind benchmark is a hard denylist on observation id, photo id,
  photo URL and sha256; any overlap aborts before writing. If the denylist is
  unavailable the combine refuses unless explicitly told to skip the check;
* class-specific provenance is configurable, e.g. needle only from curator
  batches: `--class-source needle=curated-needle`; extra sources:
  `--source name=path`; `--no-negatives` for a positives-only experiment;
* `--curated-needle` and `--curated` are repeatable and accept comma-separated
  JSONL files/globs, so curated flower/leaf/inflorescence batches can be added
  alongside curated-needle under separate source names.

Output: `dataset.json`, `images|labels/{train,val}/`, `data.yaml`,
`provenance.jsonl`, `quarantine/`, `merge-report.{json,md}`,
`dataset-audit.{json,md}`, `overlays/` — the same layout `validate`/`prepare`/
`pipeline` consume.

Production combine (run on .245 with all three source families ready):

```text
628 images (426 positive, 202 negative), 1,209 boxes, 605 observations
leaf 409 · needle 565 · flower 0 · inflorescence 235
train 555 images / 1,092 boxes (181 negatives) · val 73 / 117 (21 negatives)
per source: assisted 580 boxes (358 images) · curated-needle 629 (84)
            negative-review 480 rows -> 186 negatives, 294 excluded_non_negative
dedupe 0 · conflicts 0 · needs_review 0 · unreviewed 0 · invalid 0
blind denylist 1,935 entries (233 obs / 240 photos / 240 urls / 240 sha256), 0 matches
split isolation clean; audit status ok; GT overlays written
```

The earlier schema smoke on the two curated batches (42 images / 344 boxes)
validated the row shapes before the other sources landed.

## Frozen blind benchmark (test-only eval truth)

The curator-reviewed blind benchmark is imported directly from
`annotations.jsonl` + `manifest.jsonl` (no proposal/review CSV):

```bash
$P import-blind \
  --annotations /srv/fast/anatomy4-shenandoah/blind-test/merged/annotations.jsonl \
  --manifest /srv/fast/anatomy4-shenandoah/blind-test/merged/manifest.jsonl \
  --images-root /srv/fast/anatomy4-shenandoah/blind-test \
  --out /srv/fast/anatomy4-shenandoah/blind-test-import/anatomy4-blind-test \
  --overlays 8
```

What it guarantees:

* boxes are copied exactly as normalized xyxy (no proposal conversion); the
  frozen record's `frozen_annotations_sha256` / `frozen_manifest_sha256` and
  merged counts are verified before writing;
* 240 rows / 962 boxes / 46 negatives / 2 `needs_review` rows are the frozen
  source; **238 images / 948 boxes** are included by default, and the two
  `needs_review` rows (14 boxes) are preserved in
  `excluded/needs_review.jsonl` with their flags, notes, and source metadata.
  `--include-needs-review` includes them while keeping `needs_review: true`;
* negatives get explicit empty label files (46 of them);
* only `images/test` and `labels/test` are written. `dataset.json` declares
  `test_only: true`, `FROZEN_TEST_ONLY.json` records `allow_train: false`, and
  `prepare_ultralytics` / the `train` and `pipeline` commands refuse the dataset
  (evaluation passes `allow_test_only=True` internally);
* observation ids are preserved; the 7 multi-photo observation groups stay in
  test (no train/val exists to leak into);
* outputs: `dataset-audit.{json,md}`, `blind-benchmark.json`,
  `blind-benchmark-audit.md`, and GT-only overlays under `overlays/` for
  coordinate QA.

Real import (run on .245): 238 images / 948 boxes (leaf 545, needle 71,
flower 76, inflorescence 256), 46 negatives, 2 excluded needs_review (8 leaf +
6 flower); 948/948 label lines match the source boxes within 1e-6; 46/46
negative label files are empty; `dataset-audit` is `ok` with split isolation
`clean`.

### Evaluate a future checkpoint on the blind test

`evaluate` auto-builds the test-only prepared view (no `--prepared`, no
train/val):

```bash
$P evaluate \
  --dataset /srv/fast/anatomy4-shenandoah/blind-test-import/anatomy4-blind-test \
  --weights /srv/fast/anatomy4-shenandoah/runs/anatomy4-full/train/weights/best.pt \
  --split test \
  --out /srv/fast/anatomy4-shenandoah/blind-test-import/eval-anatomy4-blind \
  --imgsz 640 --batch 8 --device 0 --conf 0.25 --iou 0.5
```

## Commands

Run from the repository root. `P` is the Python that has
`ultralytics==8.4.190`, `torch`, `onnxruntime`, `onnx`, `pillow`, `numpy`
(the `.245` image below provides all of them).

```bash
P="python -m experiments.anatomy4_detector.cli"

# 0. Materialize the audit export (see the section above).
$P materialize --proposals <targeted/proposals.json> --review <targeted/review.csv> \
  --manifest <targeted/manifest.downloaded.json> --images-dir <targeted/images> \
  --background-review <targeted/review-background-filled.csv> --out <dataset>

# 1. Audit the materialized dataset (exit 2 when blocking).
$P validate \
  --dataset /data/anatomy4-shenandoah/dataset \
  --json-out runs/anatomy4/dataset-audit.json \
  --markdown-out runs/anatomy4/dataset-audit.md

# 2. Resolve training without touching the GPU.
$P train \
  --data /data/anatomy4-shenandoah/dataset \
  --prepare-out runs/anatomy4/prepared \
  --project runs/anatomy4 --name train \
  --epochs 80 --imgsz 640 --batch 8 --workers 4 \
  --device 0 --gpu-budget-gb 3.2 --dry-run

# 3. Evaluate a checkpoint on the held-out test split.
$P evaluate \
  --dataset /data/anatomy4-shenandoah/dataset \
  --prepared runs/anatomy4/prepared \
  --weights runs/anatomy4/train/weights/best.pt \
  --split test --out runs/anatomy4/eval-test \
  --imgsz 640 --batch 8 --device 0 \
  --conf 0.25 --iou 0.5 --max-overlays 48

# 4. Export + verify ONNX (arq-api contract).
$P export \
  --weights runs/anatomy4/train/weights/best.pt \
  --out runs/anatomy4/export/anatomy4.onnx

# 5. One-shot pipeline: validate -> prepare -> train -> eval(test) -> export -> manifest.
# Train+val dataset + separate frozen test benchmark (recommended).
$P pipeline \
  --dataset /srv/fast/anatomy4-shenandoah/datasets/anatomy4-train-mixed-v1 \
  --test-dataset /srv/fast/anatomy4-shenandoah/blind-test-import/anatomy4-blind-test \
  --out /srv/fast/anatomy4-shenandoah/runs/anatomy4-full \
  --epochs 80 --imgsz 640 --batch 8 --workers 4 \
  --device 0 --gpu-budget-gb 3.2 \
  --tune-conf-on-val \
  --conf-thresholds 0.10,0.25,0.50
```

With `--test-dataset`:

* `--dataset` is validated with `require_test=False, require_val=True`;
* `--test-dataset` is validated with `require_test=True` and must be test-only;
* observation id, photo id, and sha256 sets are checked for **zero overlap**
  (hard error before training);
* training uses train/val from `--dataset`; `val` is always evaluated on
  `--dataset`, and `test` on `--test-dataset` (auto-prepared with
  `allow_test_only=True`);
* one prediction pass per split yields a confidence-threshold sweep
  (`--conf-thresholds`, 0.25 always included). `--tune-conf-on-val` sweeps
  `--tune-thresholds` on val, selects the best macro-F1 threshold, and
  evaluates test at that threshold and 0.25;
* `metrics.json` gains `threshold_sweep`, `selected_threshold`, `tuning`,
  `tagged_bbox` (all classes over reviewed GT boxes) and top-level
  `leaf_tagged_bbox` (P/R/F1/AP50/AP50-95/support); the run manifest records the
  same under `evaluation` plus `evaluation_val`.

`pipeline --dry-run` validates both datasets (and their isolation), prepares,
resolves the model/VRAM, and writes a `dry_run` manifest without training.

### The later full run on .245 (exact)

The image below is already built on .245 (`anatomy4-detector:0.1`, derived from
`anatomy4-audit:0.1`; CUDA verified through CDI). It does **not** restart or
touch BioCLIP, the OSM sidecar, or the observations sidecar.

```bash
# On the Mac: sync this worktree (no .git in the copy; manifest git = unavailable).
rsync -az -e "ssh -i ~/.ssh/osm245_ed25519 -o IdentitiesOnly=yes" \
  --exclude .git --exclude .venv --exclude __pycache__ \
  ./ valkyrie@10.0.0.245:/srv/fast/anatomy4-shenandoah/build/repo/

# On .245:
cd /srv/fast/anatomy4-shenandoah/build/repo
docker run --rm --device=nvidia.com/gpu=all --network host \
  --user "$(id -u):$(id -g)" \
  -v /srv/fast/anatomy4-shenandoah/build/repo:/workspace \
  -v /srv/fast/anatomy4-shenandoah:/out \
  anatomy4-detector:0.1 \
  python -m experiments.anatomy4_detector.cli pipeline \
    --dataset /out/datasets/anatomy4-train-mixed-v1 \
    --test-dataset /out/blind-test-import/anatomy4-blind-test \
    --out /out/runs/anatomy4-full \
    --epochs 80 --imgsz 640 --batch 8 --workers 4 \
    --device 0 --gpu-budget-gb 3.2 \
    --tune-conf-on-val \
    --conf-thresholds 0.10,0.25,0.50
```

If the remote copy should carry a git revision, commit it in place before the
run (`git init && git add . && git commit -m "anatomy4-detector snapshot"`);
the manifest records whatever `git rev-parse HEAD` returns.

### VRAM plan for BioCLIP coexistence

Measured on .245 before this package: 8,188 MiB total, **4,281 MiB in use**
(BioCLIP), **3.32 GiB free**. `train` defaults to `--gpu-budget-gb 3.2`, calls
`torch.cuda.set_per_process_memory_fraction(0.4197)` and refuses to start when
free VRAM is below 85% of the budget (`--allow-low-vram` overrides). If the
first real run OOMs at 640/batch 8, drop to `--batch 4 --imgsz 512` (same
budget). No service restarts are required or performed.

## Export contract (arq-api)

`export.build_export_kwargs()` is exact and unit-tested:

```python
{"format": "onnx", "imgsz": 640, "batch": 1, "dynamic": False,
 "opset": 17, "simplify": False, "device": "cpu", "half": False}
```

**`nms` is never passed.** In Ultralytics 8.4.190 `nms=False` produces an
end-to-end graph with output `[1, 300, 6]`, which arq-api cannot decode.
`assert_export_kwargs` rejects `nms`, `end2end`, `int8`, and `quantize`.

`verify_onnx` runs onnxruntime on a real zero input and asserts:

* input `[1, 3, 640, 640]`, `tensor(float)` (FP32);
* output `[1, 8, 8400]` (4 box channels + 4 classes), FP32, finite;
* not `[1, 300, 6]`;
* embedded metadata `names` order equals `leaf, needle, flower, inflorescence`;
* `task` metadata is `detect`.

Verified on the real export from the .245 smoke run: `[1,3,640,640]` →
`[1,8,8400]`, fp32, class order match, `end2end: false`.

## Evaluation

`evaluate` predicts every test image (conf floor 0.001, NMS IoU 0.45,
max 300), writes `predictions.jsonl` + `ground_truth.jsonl`, then computes:

* per class: support, TP/FP/FN, precision, recall, F1 at `--conf`/`--iou`,
  AP50, AP50-95 (COCO 101-point), and the full AP-vs-IoU table;
* macro (precision/F1 over classes where defined, recall/AP over classes with
  support) and micro scores;
* a confusion matrix with class-agnostic IoU matching (wrong class = a
  confusion, unmatched prediction = background column, missed GT = background
  row);
* overlays (GT in class color; predictions green TP / red FP; missed GT dashed
  orange) and per-class contact sheets.

AP50 is always the IoU-0.5 column; `--iou` changes only the operating point
(P/R/F1/confusion/overlay colors). Images with no GT (negatives) contribute
false positives. Images are measured in pixel space, so IoU is aspect-correct.

## Manifest

`run-manifest.json` (`arq.anatomy4.detector-run/v1`) contains:

* `git` (revision/branch/dirty when available) and package versions;
* full dataset audit (splits, per-class support, isolation, label SHA-256,
  errors/warnings);
* training: model, device, parameters, hyperparameters, VRAM preflight/budget,
  elapsed time, trainer self-reported numbers (clearly separated from the
  held-out metrics);
* held-out `evaluation.split` metrics (default `test`) and JSONL paths;
* export kwargs, ONNX SHA-256, and ORT verification;
* `artifacts[]` with `kind`, absolute `path`, `bytes`, `sha256`;
* `previews[]` (overlays + contact sheets);
* `api_registration`: contract, class order, ONNX path/SHA, shapes, precision,
  condensed test metrics, `synthetic` flag.

## Production-head evaluation (ONNX one2many)

Ultralytics `model.predict()` reads the NMS-free one2one head; arq-api serves
the exported ONNX graph with the one2many head. Evaluate the shipped graph with
the same letterbox/decode rules:

```bash
$P evaluate \
  --dataset /srv/fast/anatomy4-shenandoah/blind-test-import/anatomy4-blind-test \
  --eval-onnx /srv/fast/anatomy4-shenandoah/runs/anatomy4-yolo26s-v4-adamw/export/anatomy4.onnx \
  --split test --out runs/eval-blind-onnx \
  --conf 0.25 --conf-thresholds 0.10,0.25,0.50
```

`--eval-onnx` uses 640 letterbox (fill 114), class-wise NMS (IoU 0.45), and
`[1,4+nc,N]` decoding identical to `arq-api/fieldvision/anatomy_yolo.py`.
`pipeline` also evaluates the export automatically (`eval-test-onnx`) and records
`evaluation_onnx` in the run manifest; `publish` prefers those metrics.

## Fine-tuning and needle oversampling

`train`/`pipeline` accept `--optimizer`, `--lr0`, `--lrf`, `--cos-lr`,
`--warmup-epochs`, `--cls`, `--mosaic`, `--close-mosaic`, plus
`--oversample-class needle --oversample-factor 3`. Oversampling writes only
train repeats (`tc-…__rep1.jpg`, `__rep2`) and their labels; val/test are never
modified. AdamW at lr0 6e-4 avoids the `optimizer=auto` MuSGD lr0=0.01
catastrophic-forgetting regime observed at epochs 12–14 on 508 images.

### Measured v3 vs v4 (blind-test, ONNX production head)

| run | recipe | best epoch | val mAP50 | blind mAP50 (PyTorch) | blind mAP50 (ONNX) | needle max conf (PyTorch / ONNX) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| yolo26n v3 | MuSGD auto lr0=0.01, 1× | 12 | 0.2527 | 0.1008 | 0.1061 | 0.1665 / 0.1692 |
| yolo26s v3 | MuSGD auto lr0=0.01, 1× | 14 | 0.2561 | 0.0634 | 0.0682 | 0.0950 / 0.1076 |
| yolo26s v4 | AdamW lr0=6e-4, cosine, needle×3 | 15 | 0.2486 | 0.0961 | **0.1030** | **0.2558 / 0.2072** |
| yolo26n v4 | AdamW lr0=6e-4, cosine, needle×3 | 21 | 0.2700 | 0.0906 | 0.0937 | **0.3279 / 0.3774** |

AdamW lifts needle confidence ~1.5–2× and raises the peak (n: 0.274 at epoch 21
vs 0.276 at 12; s: 0.244 at 15 vs 0.249 at 14), but does not remove late
degradation: a forced full 80-epoch yolo26n rerun still peaked at epoch 21
(0.2737) and ended at 0.2174. The ONNX one2many head is consistently a small
improvement over the PyTorch one2one head on the same checkpoint (+0.003–0.009
mAP50), not a different detector. It does not solve blind needle recall
(support 71; 0–1 TP at 0.25).

## Cheap coarse cross-organ tags

`coarse-multi` filters the existing YOLOE/OWLv2 cross-organ proposal dump for
the remaining Plantae pool:

```bash
$P coarse-multi \
  --proposals /srv/fast/anatomy4-shenandoah/train-candidates/hybrid-proposals.json \
  --manifest /srv/fast/anatomy4-shenandoah/train-candidates/manifest.jsonl \
  --out /srv/fast/anatomy4-shenandoah/train-candidates/coarse-multi/auto-plantae.jsonl
```

Policy: leaf ≥0.24, OWLv2 `single flower` ≥0.24, `cluster of flowers` ≥0.28;
1–8 boxes, max box area ≤0.55, leaf-clutter margin ≤3; curated-multi,
curated-needle, blind-test, and curated-multi excluded ids are never re-added.

Real result on .245: 2,400 manifest images, 504 reserved (376 curated-multi,
44 curated-multi excluded, 84 curated-needle), 1,203 pooled images with
proposals → **147 clean coarse-labeled images / 215 boxes** (leaf 32, flower
118, inflorescence 65); 1,042 images had no box above threshold, 693 had no
proposals, 6 oversized, 7 leaf-clutter, 1 >8 boxes.

## Publishing to arq-api

`publish` registers a completed run through the anatomy4 API seam
(`arq-api/routers/fieldvision_anatomy.py`; routes under `/api`). It reads the
run manifest plus the materialized dataset and performs the full registration.

```bash
export ARQ_BASE=https://dev.arborphy.ai        # or http://localhost:8000
export ARQ_API_KEY=...                          # X-API-Key client gate
export ARQ_USER_TOKEN=...                       # or GOOGLE_ID_TOKEN (Bearer)

$P publish \
  --manifest /srv/fast/anatomy4-shenandoah/runs/anatomy4-full/run-manifest.json \
  --dataset /srv/fast/anatomy4-shenandoah/dataset-anatomy4-targeted \
  --base "$ARQ_BASE" \
  --name "anatomy4 yolo26n shenandoah" \
  --batch-size 200 \
  --report-out /srv/fast/anatomy4-shenandoah/runs/anatomy4-full/publish-report.json
```

Add `--test-dataset` when the run used a separate frozen test benchmark:

```bash
$P publish \
  --manifest /srv/fast/anatomy4-shenandoah/runs/anatomy4-full/run-manifest.json \
  --dataset /srv/fast/anatomy4-shenandoah/datasets/anatomy4-train-mixed-v1 \
  --test-dataset /srv/fast/anatomy4-shenandoah/blind-test-import/anatomy4-blind-test \
  --base "$ARQ_BASE" \
  --name "anatomy4 yolo26n shenandoah" \
  --batch-size 200
```

With `--test-dataset`, train+val items come from `--dataset` and test items
(accepted positives + explicit reviewed negatives) come from the frozen test
dataset with explicit `split=test`; the two datasets must be disjoint and
`--dataset` may not contain test items. `metrics.test_count` is the sum of
per-class tagged test-box support, which is exactly what
`AnatomyLabPanel.tsx` reconciles against the registered dataset's test items
(`derivedSupport` total), and the publisher hard-fails when the test dataset's
label count differs from the manifest's test_count. `eval_dataset_id` is the
registered dataset id.

`--dry-run` builds and prints the exact plan (dataset payload summary, item
counts and batch sizes, test item counts, model, artifacts with bytes/sha256,
metrics summary, inference image) **without sending a request**. Auth presence is reported as
booleans; token/key values are never printed, logged, or written to the report.

Operations (in order, all idempotent on re-run):

1. `GET /fieldvision/anatomy/datasets/{id}` then `POST …/datasets` when absent —
   dataset manifest includes the run manifest hash, dataset audit summary,
   materialization counts/partial-labeling note, trainer/hyperparameters, export
   verification, and git revision; provenance carries the audit input SHA-256s.
2. `POST …/datasets/{id}/items` in batches (≤500; default 200) with
   `item_key=img:<photo id>` so multi-photo observations cannot collide:
   * accepted positives — normalized xyxy labels, `review_state=accepted`,
     `source=human`, `confidence` from the proposal, `model_id` = detector, and
     per-label provenance (proposal id, prompt, box score, review notes/row);
   * accepted negatives — empty labels, `negative=true`,
     `review_state=accepted`, `source=human`, negative basis in provenance;
   * pending items — images with unreviewed proposals are staged
     (`review_state=proposed`, empty labels) so curators can finish them.
     Rejected proposals never become truth and cannot be stored as empty
     "rejected" items by the API contract, so they are excluded (counted in the
     dry-run plan).
   Every item carries the materialized split, so API and trainer splits agree.
3. `POST /fieldvision/anatomy/models` (reused when the id exists) with contract,
   class order, dataset id, artifact metadata, and trainer/export provenance.
4. `POST …/models/{id}/artifacts` for `anatomy4.onnx` (≤16 MiB, sha checked
   against the manifest), `best.pt` (≤16 MiB), `metrics.json` (≤2 MiB),
   preview images (≤3 MiB each, default 8) and `preview_detections.json`
   (≤2 MiB: overlay TP/FP/FN index, contact sheets, bounded predictions sample).
5. `POST …/models/{id}/metrics` — `test_count`, per-class precision/recall/
   AP50/AP50-95, and a 4×4 class confusion submatrix (the full 5×5 matrix with
   background stays in `metrics.json`; the API requires NxN). Undefined metrics
   for zero-support classes abort by default; `--allow-zero-support` records
   0.0 with an explicit provenance note.
6. `POST …/models/{id}/activate` (employee-gated; ONNX preflight runs server-side).
7. `POST /fieldvision/anatomy/infer` with a real test-split image and
   `model_id`, `threshold`, `iou_threshold`. The smoke passes only when the
   response reports `source=onnx`, the expected `model_id`/class order, valid
   dimensions, and normalized xyxy detections.

`--skip-items` is for a dataset already registered (or a curator-owned dataset):
it skips step 2 entirely. Missing ONNX/best.pt/metrics/previews fail locally
before any upload; 401/403/422/413 produce a specific, secret-scrubbed message.
Network calls use httpx with an explicit connect/read timeout and bounded
retries (429/5xx/transport, exponential backoff with `Retry-After`).

Verified plans (no requests sent):

```text
targeted dataset (real audit export)
  374 items = 55 accepted positives + 21 accepted negatives + 298 pending
  99 rejected proposals excluded from truth; batches @200 -> [200, 174]
  first positive: img:100967348, split train, class leaf
  first negative: img:101807843

synthetic smoke manifest (plumbing only, --allow-synthetic)
  items 24 = 21 positives + 3 negatives (one negative per split)
  artifacts onnx 9,809,211 B · best.pt 5,414,725 B · metrics 6,019 B
            8 previews · preview_detections.json 91,614 B
  infer test image: .../images/test/test-00.jpg (12,264 B)
```

The targeted materialization's `proposals.json` is ~150 MB, so building the
pending queue parses it once (several seconds, ~1–2 GB RSS); use `--no-pending`
when the audit queue is already imported.

Requirements: this API seam must be deployed on the target base, and the user
token must belong to an arborphy.ai account (employee) for model writes,
activation, and inference. Nothing was published to dev by this work.

## Tests and validation evidence

```bash
python -m pytest experiments/anatomy4_detector/tests -q   # 159 passed
```

Coverage: label parsing/xyxy normalization, IoU, split determinism, observation
leakage, explicit negatives, missing/orphan labels, class-order mismatches,
prepare conversion (`cx cy w h`, empty negative files), AP/precision/recall/
confusion behavior (including zero-support → `null`), export kwargs (no `nms`),
ONNX shape/dtype/metadata verification via onnxruntime, train kwargs and the
`-seg` guard, manifest hashing/registration, the synthetic dataset, and the
materializer (real-shaped proposals/review fixtures, tp-only truth, unreviewed
exclusion, reviewed/explicit/background negatives, class-scoped negatives, box
parity, exact dedupe, conflicting/unknown review rows, missing images,
multi-photo observations, observation split safety), and the API publisher
(payload assembly, bounded batching, accepted negatives/pending items, metrics
and confusion mapping, artifact limits, 401/403/422 handling, retries,
no-secret logging, `--skip-items`, `--dry-run`, infer-response validation, and
parity against the real arq-api `anatomy_contract` validators), and the frozen
blind import (exact box/negative conversion, needs_review exclusion/opt-in,
freeze SHA verification, test-only marker + training refusal, multi-photo
observation preservation, GT overlays) and the mixed-source combine
(multi-source merge, canonical image identity, dedupe with provenance merging,
negative/positive conflict quarantine, needs_review/unreviewed exclusion,
class-specific source selection, deterministic train/val splits, blind
denylist enforcement, per-source/per-class reports).

Synthetic end-to-end (plumbing only; metrics are meaningless and the manifest is
stamped `synthetic: true`):

```bash
$P smoke --out /tmp/a4smoke --device cpu --dry-run          # validator only
$P smoke --out /tmp/a4smoke --device mps --epochs 2 --imgsz 160 --batch 4 --workers 0
```

Real `.245` CUDA smoke (executed, complete):

```text
device 0, RTX 4060 Ti, torch 2.14.0+cu130
preflight free 3.32 GiB / budget 3.2 GiB; allocator fraction 0.4197
train 2 epochs @320, eval test, export, ORT verify ok
output [1,8,8400] fp32, classes leaf,needle,flower,inflorescence, end2end false
```

## Blockers / before the full run

1. **Review coverage, not format.** The adapter now produces the contract
   dataset from both real exports. What limits the full run is review breadth:
   the targeted run reviewed 171 of 3,351 proposals, so 55 labeled images also
   carry 975 unreviewed proposals (recorded in `materialization.json`), and the
   test split has 10 boxes (`needle` test support 1). A larger, per-image
   exhaustive review is the next audit milestone; until then test AP is thin
   evidence.
2. **Negatives are only claimed where reviewed.** The targeted negatives are 20
   `empty_correct` background verdicts plus 1 fully rejected image. Images the
   audit saw nothing in are not assumed negative; select them explicitly with
   `--negative-image`/`--negative-file` (or a background review) if a human has
   checked them.
3. **Git revision on .245.** An rsynced copy has no `.git`; the manifest reports
   `git.available: false`. Commit the remote snapshot (or clone the branch)
   before the full run if a revision stamp is required.

## Known limits

* One author/trainer; no resume orchestration or hyperparameter sweep.
* No iNat/DB access; the package consumes an already-materialized dataset.
* Confusion matrix is class-agnostic by design (class confusion is visible);
  P/R/AP use class-aware matching.
* Overlays/contact sheets are capped (`--max-overlays`, 12 per sheet) and
  center-crop captions on very long stems.
* Unreviewed proposals on labeled images are background (partial annotation);
  use `--require-fully-reviewed` for a smaller, stricter dataset.
* Dedupe is exact class+xyxy only; near-duplicate boxes from merged prompt
  passes are kept and counted by `validate` as distinct boxes.
* The blind benchmark's two `needs_review` rows are excluded from default
  evaluation; run with `--include-needs-review` only if the director accepts
  the flagged ambiguity.
* No TTA, no multi-scale validation, no segmentation; `yolo26n-seg` is refused.
* Bare checkpoint names resolve into the Ultralytics weights cache
  (`SETTINGS["weights_dir"]`, normally `~/.cache/ultralytics/weights`) so runs
  never drop `yolo26n.pt` into the repository.
