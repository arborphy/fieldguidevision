# LeafSnap + BioImages — hierarchical field-vision pipeline (E1→E4)

Experiment notes for the two-layer leaf pipeline: a top-level organ gate
(E3) feeding a lower-level simple-vs-compound leaf specialist (E4).

**Objective.** Establish, with pre-registered evaluators, whether a YOLO26
organ detector is good enough to gate lower-level feature training, and
whether a leaf-type specialist trained on LeafSnap crops clears its
compound-leaf/field gate. This is a feasibility proof run, not a product
integration or a species classifier.

**Current status (2026-10-29).** E3 organ gate: **PASS**. E4 leaf specialist:
**FAIL** in both revisions (V1 and V2). V1 is retained only as a *proposed
field-telemetry* model; it is **not validated tagging** and must not drive
user-facing claims. See `EXPERIMENT_LOG.md`.

## Supplied evaluator chain

The architecture was supplied as a PDF plus a CSV export. The PDF is
image-only (`pdftotext` yields ~1 byte / no words), so the **CSV is
authoritative**:

- CSV ([open](file:///Users/valkyrie/Downloads/Field%20Vision%20Pipeline%20-%20Experiment%20architecture%20%28widescreen%29.csv)): `/Users/valkyrie/Downloads/Field Vision Pipeline - Experiment architecture (widescreen).csv`
- PDF ([open](file:///Users/valkyrie/Downloads/Field%20Vision%20Pipeline.pdf), reference only): `/Users/valkyrie/Downloads/Field Vision Pipeline.pdf`
- Core contract from the CSV: `evaluate(input, level, criteria, threshold)` —
  one evaluator, four variants; the top level must pass before lower-level
  training starts.

| stage | evaluator input | gate in this experiment | evidence | status |
|---|---|---|---|---|
| **E1** | generated labels | are the organ labels good enough? | dataset validation + audited Dirr join (`DATASETS.md`, `EDA.md`) — no separate E1 artifact | partially evidenced |
| **E2** | top-level training input | is the training data good enough? | planned in the chain as TODO; not implemented as a distinct evaluator | not run |
| **E3** | Model A on held-out test | leaf recall + leaf mAP50 | `scratch/leafsnap-yolo26/organ-gate/ORGAN_GATE.md` | **PASS** |
| **E4** | Models B1..Bn + hierarchy | leaf-type accuracy/recall | `scratch/leafsnap-yolo26/leaf-specialist*/LEAF_SPECIALIST.md` | **FAIL** |

## Dataset and model paths

| artifact | path |
|---|---|
| LeafSnap source | `/Users/valkyrie/Desktop/arborphy product/leafsnap/leafsnap-dataset` |
| BioImages source | `/Users/valkyrie/dev/arborphy/fieldguidevision/scratch/bioimages-ne-trees` |
| Detector dataset | `scratch/leafsnap-yolo26/detector` (`dataset.yaml`, `manifest.jsonl`, `counts.json`) |
| Classifier V1 dataset | `scratch/leafsnap-yolo26/classification` |
| Classifier V2 dataset | `scratch/leafsnap-yolo26/classification-v2` |
| E3 checkpoint / ONNX | `scratch/leafsnap-yolo26/organ-gate/best.pt`, `organ_gate.onnx` (FP16, 320×320) |
| E4 V1 checkpoint / ONNX | `scratch/leafsnap-yolo26/leaf-specialist/best.pt`, `leaf_specialist.onnx` (FP32, 224×224) |
| E4 V2 checkpoint / ONNX | `scratch/leafsnap-yolo26/leaf-specialist-v2/best.pt`, `leaf_specialist.onnx` |
| E3 report | `scratch/leafsnap-yolo26/organ-gate/ORGAN_GATE.md` |
| E4 reports | `scratch/leafsnap-yolo26/leaf-specialist/LEAF_SPECIALIST.md`, `.../leaf-specialist-v2/LEAF_SPECIALIST.md` + `V1_V2_COMPARISON.md` |
| Dataset audit / counts | `DATASETS.md`, `scratch/leafsnap-yolo26/{detector,classification,classification-v2}/*.json` |
| EDA | `EDA.md`, `eda_results.json`, `feature_assertions.json` |
| V2 data revision | `LEAF_SPECIALIST_V2_DATA.md` |

All paths are relative to `/Users/valkyrie/dev/fieldguidevision-yolo26-3d`
unless absolute.

## Reproduce

Run from the repository root. `--overwrite` deletes and rebuilds generated
datasets; it does not touch the source corpora.

```bash
cd /Users/valkyrie/dev/fieldguidevision-yolo26-3d

# EDA (training-free; reads LeafSnap manifest + Dirr parquets; optional GoBotany)
.venv/bin/python experiments/leafsnap-yolo26/eda.py
.venv/bin/python experiments/leafsnap-yolo26/test_eda.py

# Datasets: E1/E2 inputs (deterministic; no network, no training)
.venv/bin/python experiments/leafsnap-yolo26/build_datasets.py --overwrite
.venv/bin/python experiments/leafsnap-yolo26/evaluator.py   # self-checks

# E3 organ gate (YOLO26n, MPS, 20 epochs; evaluates and exports ONNX)
.venv/bin/python experiments/leafsnap-yolo26/train_organ_gate.py
.venv/bin/python experiments/leafsnap-yolo26/test_organ_gate.py

# E4 V1 leaf specialist (YOLO26n-cls, 224 px, species-disjoint)
.venv/bin/python experiments/leafsnap-yolo26/train_leaf_specialist.py

# E4 V2: audited 122-species mapping, compound training = lab crops only
.venv/bin/python experiments/leafsnap-yolo26/build_leaf_specialist_v2.py --overwrite
.venv/bin/python experiments/leafsnap-yolo26/train_leaf_specialist_v2.py
```

Re-render an E4 report without retraining:
`.venv/bin/python experiments/leafsnap-yolo26/train_leaf_specialist.py --report-only`
(V1); V2 has no report-only flag, so use `--skip-train --skip-export` (this
appends the V1/V2 comparison section again).

Training uses the Ultralytics pretrained checkpoints (`yolo26n.pt`,
`yolo26n-cls.pt`); a first run may download them. MPS runs are seeded and
configured deterministically but not bitwise reproducible.

## Results at a glance

- **E3 PASS** — leaf mAP50 0.9935 (≥ 0.80); LeafSnap field leaf recall 0.9916
  at conf 0.25 (≥ 0.80). Non-leaf macro recall 0.8426 (reported, not gating).
- **E4 V1 FAIL** — top1 93.58%; simple recall 0.9843 (PASS); compound recall
  0.6996 (< 0.75); compound field recall 0.6995 (< 0.70; 135/193).
- **E4 V2 FAIL** — top1 94.24%; simple recall 0.9959; compound recall 0.6850;
  compound field recall 0.6729 (144/214). Improving overall accuracy did not
  fix the compound/view failure.
- The failure is a **label/view mismatch**: species-level Dirr `leaf_type`
  labels are inherited by crops that can show a single leaflet. V2 removed
  field compound crops from *training* only; validation/test still contain
  them, so the noise floor remains — and Juglans cinerea stayed near 44%.

## Documents

- `EXPERIMENT_LOG.md` — dated, chronological record and decisions.
- `FIELD_TEST.md` — director-led ArborQuest offline APK protocol (telemetry
  collection only; E4 is not validated).
- `NEXT_DATA_CURATION.md` — exact next data requirements and minimum floors.
- `EDA.md` — LeafSnap × Dirr training-free analysis (feature/pair information).
- `DATASETS.md` — deterministic dataset build, mapping and counts.
- `LEAF_SPECIALIST_V2_DATA.md` — V2 label/split revision.

## Known limits

- Only `leaf` vs non-leaf organs, and only simple vs compound leaf type, are
  modelled. No species, margin, arrangement, venation, or bark claim.
- LeafSnap boxes and crops are segmentation-derived, not human annotations;
  BioImages boxes are weak full-frame/near-full-frame labels.
- The organ gate was measured on curated single-leaf photographs; it is not
  evidence for cluttered-scene detection.
- E1 has no standalone evaluator artifact and E2 is unimplemented; both are
  documented as gaps rather than inferred passes.
- No mobile/APK validation has been performed; field conclusions are pending
  the protocol in `FIELD_TEST.md`.
