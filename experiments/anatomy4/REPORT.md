# Anatomy4 Shenandoah bootstrap — 120-image detector audit

Date: 2026-11-12. Area: `inst-osm-relation-5548542` (Shenandoah National Park).
All work ran on `.245` GPU (RTX 4060 Ti 8 GB) against read-only DBs. No app DB
writes, no service restarts, no commits.

## 1. What was built

Pipeline package: `experiments/anatomy4/` (contracts, manifest, download,
prelabel, render, labels, audit, yoloe, cli, configs, tests, Dockerfiles).
Deterministic, resumable, provenance-stamped; raw proposals can never become
YOLO labels without human acceptance.

- `build-manifest` — read-only stratified sample from the app DB (observation
  photos joined through the area polygon), license/attribution retained,
  `square.` iNat URLs upgraded to `large.`, SHA256 download stamping.
- `download` — vault-first, source-URL fallback, per-image dimensions.
- `prelabel` — Grounding-DINO-tiny + SAM2.1-hiera-tiny, per-class prompt sets
  and thresholds, NMS, polygon + bbox, area guards, resume-safe item files.
- `prelabel-yoloe` — YOLOE-26s-seg open-vocab comparison backend.
- `render` — per-image overlays, per-class contact sheets, score-stratified
  `review-sample.csv` with tile indices aligned to sheet captions.
- `labels` — accepted-only YOLO export (detection + segmentation), splits are
  observation-level.
- `audit` — proposal/empty/multi-organ rates, score distributions, reviewed
  precision, markdown + JSON report.

Tests: `.venv/bin/python -m pytest experiments/anatomy4/tests -q` → **7 passed**.

Two bugs found and fixed during the audit pass (both from the image record key
being `image_id` in the downloaded manifest, `photo_id` in the sample manifest):
`audit.py` empty-image detection and `labels.py` image-copy lookup now fall
back across both keys.

## 2. Dataset

| item | value |
| --- | ---: |
| observations in area | 74,104 |
| observations with photos | 16,466 |
| plant observations with photos | 7,656 |
| sampled observations | 120 (86 Plantae / 34 other) |
| audited images | 120 (one cover per observation) |
| total photos on those observations | 175 |
| bytes downloaded | 58.85 MB |
| unique SHA256 | 120 |
| download source | 120 source-url, 0 vault |

Strata: spring 23, summer 22, fall 17, winter 14, needle-candidate 10, other
kingdom/class 2 each, other:fallback 10.

`review-sample.csv` is 24 proposals per class (needle capped by availability:
14 in YOLOE v1, 24 in baseline/v2), score-stratified.

## 3. Runs compared (same 120 images)

| run | backend | proposals | per image | wall clock | notes |
| --- | --- | ---: | ---: | ---: | --- |
| `anatomy4-v1` | DINO-tiny + SAM2.1 | 3,276 proposed (+376 rejected by guard) | 27.3 | 169.4 s (1.41 s/img) | whole-frame boxes on most photos |
| `anatomy4-v1b-targeted` | DINO-tiny + SAM2.1 | 1,394 | 11.6 | 163.2 s (1.36 s/img) | needle still unusable |
| `anatomy4-v1-yoloe` | YOLOE-26s-seg | 287 | 2.4 | 3.8 s | leaf/needle/inflorescence good |
| `anatomy4-v1-yoloe2` | YOLOE-26s-seg | 298 | 2.5 | 4.06 s | best overall |

Baseline guard rejections: `rejected_area:dominates_frame` 296,
`rejected_label:unmapped` 69, `rejected_area:too_small` 11.

Provenance: `IDEA-Research/grounding-dino-tiny@a2bb814`,
`facebook/sam2.1-hiera-tiny@de431c4`, `yoloe-26s-seg.pt`, torch 2.14.0+cu130,
transformers 4.57.6, float32, CUDA 13.0.
DINO peak observed GPU reservation ~2.3–2.6 GB with one transient
expandable-segments allocation warning (no OOM). YOLOE memory was not
instrumented; it loads in seconds and completed all 120 images in 4.06 s.

### Global label-density behavior

| run | images with ≥1 proposal | empty | multi-organ images |
| --- | ---: | ---: | ---: |
| baseline DINO | 120 / 120 | 0.0% | 119 / 120 (99.2%) |
| YOLOE v2 | 89 / 120 | 25.8% | 32 / 120 (26.7%) |

The baseline "labels everything" profile is the main reason it is not suitable
as a bulk prelabeler even though its boxes can be locally correct.

## 4. Manually inspected precision (24/class, single reviewer)

| class | baseline DINO | YOLOE v1* | YOLOE v2 |
| --- | ---: | ---: | ---: |
| leaf | 16/24 = 66.7% | 24/24 = 100% | 21/24 = 87.5% |
| needle | 0/24 = 0.0% | 3/14 = 21.4% | 4/24 = 16.7% |
| flower | 7/24 = 29.2% | 7/24 = 29.2% | 14/22 = 63.6% |
| inflorescence | 7/24 = 29.2% | 13/24 = 54.2% | 16/24 = 66.7% |
| overall | 30/96 = 31.2% | — | 55/94 = 58.5% |

\* YOLOE v1 counts were recorded during the sheet inspection before the review
CSVs were written; v1 has 14 needle proposals so its sample is 14, and it was
not re-run through `audit`. Treat as inspection counts, not audited output.

### Score-threshold behavior (YOLOE v2)

| class | ≥0.10 | ≥0.15 | ≥0.20 | ≥0.25 | ≥0.30 |
| --- | --- | --- | --- | --- | --- |
| leaf | 94% (18) | 91% (11) | 88% (8) | 83% (6) | 100% (4) |
| needle | 17% (24) | 21% (14) | 25% (8) | 50% (4) | 50% (2) |
| flower | 64% (22) | 75% (12) | 88% (8) | 100% (7) | 100% (4) |
| inflorescence | 67% (24) | 71% (14) | 70% (10) | 86% (7) | 100% (5) |

(precision over the reviewed sample; n = reviewed proposals above threshold)

## 5. What the boxes look like

Good:
- YOLOE leaf boxes are tight on a single leaf/blade or a small leaf group
  (e.g. `177370620` shiny leaf, `99108633` single basswood leaf, `723188026`
  compound leaf). Leaf precision holds to low scores.
- YOLOE flower≈single bloom is well separated from clusters once the phrase is
  `single flower` (v2): the purple bellflower, the white aster in hand, the
  thistle head, the yellow composite head.
- Inflorescence: rounded hydrangea-like heads, white umbels, grass panicles,
  thistle heads are caught with usable polygons above 0.25.
- SAM2.1 polygons on the baseline are often accurate *given* a correct box
  (mask IoU p50 high); the failure is box selection, not segmentation.

Bad / systematic:
- **needle**: the class is being asked of wide shots where a conifer is
  background (tree, branch, forest floor). Even true conifer photos get a box
  around a branch tip or cone, not needles. Precision stays ≤50% at any
  threshold. Needs either a dedicated "conifer foliage" class or close-up
  needle training data; not resolvable by prompt tuning (three prompt sets
  tried).
- **flower vs inflorescence** is the other axis of confusion in the baseline
  and in YOLOE v1 ("flower" catches whole heads). `single flower` /
  `cluster of flowers` fixes most of it in v2 but not all (8 flower FPs, 8
  inflorescence FPs).
- Baseline whole-frame boxes survived the `max_box_area=0.92` guard on 296
  detections and still produced 99.2% multi-organ images — the guard removes
  only the extreme cases.
- Fern fronds, moss, and grass are recurrent leaf/needle false positives; they
  are a vocabulary problem (the label set has no fern/moss/grass class).

## 6. Recommended production settings

Use **YOLOE-26s-seg** (`yoloe-26s-seg.pt`, imgsz 1024, conf floor 0.03, NMS-free
YOLOE output) with phrases:

```
leaf                -> "leaf"
needle              -> "pine needles"
flower              -> "single flower"
inflorescence       -> "cluster of flowers"
```

and accept at:

| class | threshold | expected precision | yield on this 120-image sample |
| --- | ---: | ---: | ---: |
| leaf | 0.20 | ~88% | 50 |
| flower | 0.25 | 100% | 7 |
| inflorescence | 0.25 | ~86% | 19 |
| needle | 0.25 | ~50% (review-only) | 7 |

That is ~83 accepted proposals per 120 images after review, versus 55 from the
current full-sample pass. Keep DINO+SAM2.1 only as an optional polygon refiner
for accepted YOLOE boxes; do not use it as the proposal generator at scale.

For a bootstrapping run, route needle and all 0.20–0.25 proposals to human
review; auto-accept leaf ≥0.25, flower ≥0.25, inflorescence ≥0.30.

## 7. Export proof

Accepted-only export (v2 review, 55 accepted of 94 reviewed):

```
/srv/fast/anatomy4-shenandoah/yolo-yoloe2-review/
  accepted.json   data.yaml   labels/{train,val,test}   labels_seg/{train,val,test}   images/{train,val,test}
```

33 images / 33 label files, observation-level splits train 25 / val 2 / test 6,
`accepted_by_class = {leaf 21, needle 4, flower 14, inflorescence 16}`.
Local copy: `experiments/anatomy4/artifacts/yolo-sample/` (manifest + labels;
images stay on `.245`).

## 8. Commands (reproduce)

Host `.245`, images already built: `anatomy4-audit:0.1`, `anatomy4-yoloe:0.1`.

```bash
cd /home/valkyrie/anatomy4-build
docker run --rm --network host -u 1002:1002 \
  -v /home/valkyrie/anatomy4-build/repo:/workspace \
  -v /srv/fast/anatomy4-shenandoah:/out \
  -w /workspace anatomy4-audit:0.1 \
  python -m experiments.anatomy4.cli audit \
    --manifest /out/prelabel-yoloe2/proposals.json \
    --review /out/v2-review.csv --out /out/audit-yoloe2
```

`prelabel` runs need `--device=nvidia.com/gpu=all` and the same mounts plus
`-v /srv/fast/anatomy4-shenandoah/models:/models`. Full command set is in
`experiments/anatomy4/README.md`.

## 9. Known limits

- Precision only. There is no exhaustive ground truth, so recall is unknown;
  24 proposals per class is a thin sample and a single reviewer (no
  inter-annotator agreement).
- `labels` writes `data.yaml` with the absolute container path used at export
  time (`path: /out/...`); rewrite it when moving the dataset.
- YOLOE polygons are its own seg head, not SAM2.1. Polygon quality was visually
  acceptable but not scored against masks.
- One area, one year of seasons; needle class is underrepresented in the
  population (10 needle candidates / 120) and in the review sample.
- No trained model was produced. This is a bootstrap audit, not a training run.

## 10. Next steps

1. Run YOLOE-26s-seg at the recommended thresholds over a 1–2k image slice of
   Shenandoah to measure how the operating points hold (and estimate review
   load per 1k images).
2. Add fern/moss/grass as explicit negative or separate classes, or add a
   "not an organ" rejection class; they are the dominant leaf/needle FPs.
3. Treat needle as its own data-collection task (close-up conifer foliage);
   do not spend more prompt-tuning effort on it.
4. Then build the first reviewed training set from accepted labels and compare
   a fine-tuned detector against the YOLOE bootstrap on the same review sheets.