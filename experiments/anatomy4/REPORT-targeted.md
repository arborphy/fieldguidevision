# Anatomy4 targeted Shenandoah audit — class-specific sampling and prompt variants

Date: 2026-11-12 (follow-up to the 120-image broad audit).
Area: `inst-osm-relation-5548542` (Shenandoah National Park). Read-only DBs,
`.245` RTX 4060 Ti, no service restarts, no commits.

Decision up front: **DINO+SAM is not scaled** (failed the quality gate). YOLOE-26
is the proposal engine. This pass asks whether targeted sampling and prompt
variants make needle, flower, and inflorescence good enough to auto-label.

## 1. Sample (391 images)

Taxonomy is used ONLY to pick candidates; no proposal inherits a taxon label.

| bucket | selection rule | observations | images |
| --- | --- | ---: | ---: |
| needle | Plantae, family Pinaceae/Cupressaceae or a conifer genus; up to 2 photos per obs | 60 | 83 |
| flower | Plantae, observed Mar–Aug, non-conifer | 140 | 140 |
| leaf | Plantae, non-conifer (broadleaf pool) | 120 | 120 |
| background | half non-plant, half winter/unknown Plantae | 48 | 48 |

Area conifer pool is small: 87 observations with photos total, so the needle
bucket is 69% of the entire usable conifer population. One image
(`473273240`) failed both vault and iNat download; 390 images were processed.

Manifest: `artifacts/targeted/manifest.json` (the downloaded copy carries
paths/SHA256).

## 2. Runs

Eight YOLOE-26s-seg variant passes (each prompt is its own single-class run so
the prompt that produced a box is unambiguous), 1024 px, conf floor 0.02–0.05,
merged into one manifest:

| bucket | class | prompts tried |
| --- | --- | --- |
| needle | needle | pine needles · spruce needles · conifer foliage · needle-bearing branch |
| leaf | leaf | leaf · leaf blade · broadleaf |
| flower | flower | single flower · flower with petals · flower head |
| flower | inflorescence | cluster of flowers · flower spike · umbel of flowers |
| background | all four | same prompts, to measure false positives on negatives |

3,331 proposals over 390 images, 31.6 s total inference (~0.08 s/image,
model load excluded). 369/390 images had ≥1 proposal (5.4% empty), 36.4%
multi-organ. Provenance is stamped per run (model, prompt, threshold, imgsz,
conf floor, torch); no rectangle fallback.

## 3. Reviewed precision (image Read tool, contact sheets)

Reviewed: **39 needle / 44 leaf / 44 flower / 44 inflorescence** proposals,
score-stratified per prompt, plus 8 designed-negative proposals forced into
every class sheet.

| class | tp | fp | precision | own-pool precision* |
| --- | ---: | ---: | ---: | ---: |
| leaf | 34 | 10 | 77.3% | 79.2% (19/24) |
| needle | 10 | 29 | 25.6% | 39.1% (9/23) |
| flower | 14 | 30 | 31.8% | 35.5% (11/31) |
| inflorescence | 14 | 30 | 31.8% | 40.0% (12/30) |

\* own-pool = only the class's targeted bucket; the full number includes the
designed negatives (mushrooms, moss, trunks, animals), which is the honest
number for arbitrary image streams.

### Per-prompt precision (own pool)

| class | prompt | tp/fp | precision | usable / 1000 |
| --- | --- | ---: | ---: | ---: |
| leaf | leaf | 5/1 | 83.3% | 833 |
| leaf | broadleaf | 9/2 | 81.8% | 818 |
| leaf | leaf blade | 5/2 | 71.4% | 714 |
| needle | spruce needles | 4/2 | 66.7% | 667 |
| needle | pine needles | 3/5 | 37.5% | 375 |
| needle | conifer foliage | 2/5 | 28.6% | 286 |
| needle | needle-bearing branch | 0/2 | 0% | 0 |
| flower | single flower | 7/3 | 70.0% | 700 |
| flower | flower head | 3/8 | 27.3% | 273 |
| flower | flower with petals | 1/9 | 10.0% | 100 |
| inflorescence | cluster of flowers | 6/5 | 54.5% | 545 |
| inflorescence | umbel of flowers | 4/4 | 50.0% | 500 |
| inflorescence | flower spike | 2/9 | 18.2% | 182 |

Pooled per-prompt (all buckets): leaf 92.3%, broadleaf 75%, leaf blade 68.4%;
needle spruce 44.4%, pine 28.6%, conifer foliage 16.7%, needle-bearing branch
0%; flower single 61.5%, head 23.1%, petals 16.7%; inflorescence cluster
53.8%, umbel 26.3%, spike 16.7%.

### Score thresholds

| class | ≥0.10 | ≥0.20 | ≥0.30 | ≥0.40 |
| --- | --- | --- | --- | --- |
| leaf (all) | 73.7% (19) | 62.5% (8) | 60.0% (5) | 50.0% (4) |
| leaf (own pool) | 100% (2–4) | 100% (1–2) | 100% (3) | — |
| needle | 40.0% (15) | 75.0% (4) | 100% (2) | — |
| flower | 47.6% (21) | 70.0% (10) | 85.7% (7) | 83.3% (6) |
| inflorescence | 25.0% (24) | 23.1% (13) | 16.7% (6) | 0% (4) |

Two calibration facts matter more than the curves:

1. **Leaf precision does not improve with score on arbitrary images** because
   mushroom caps and moss score 0.88/0.88 on the `leaf blade` prompt. On the
   plant pool it is 100% above 0.30, so auto-accept must be gated to
   plant-observation streams, not arbitrary uploads.
2. **Inflorescence is inverted**: the highest-scoring boxes are the worst
   (0.67/0.64/0.59 on ground foliage, a single aster, a single flower). Prompt
   "cluster of flowers" is the only one usable at mid scores.

## 4. Empty and background review (recall / abstention)

53 images with no proposals in their bucket were inspected box-blind:

| bucket | inspected | no target organ | organ visible (missed) |
| --- | ---: | ---: | ---: |
| background | 32 | 15 | 17 |
| leaf | 12 | 0 | 12 |
| flower | 5 | 5 | 0 |
| needle | 4 | 0 | 4 |

Abstention accuracy: **37.7%** (20/53). The worst cells are decisive:

- All 4 needle-bucket images with zero proposals clearly show conifer foliage
  (two are close-ups, one is needles in hand). YOLOE at 0.05 found nothing.
- All 12 leaf-bucket empties have visible leaves; several are dominated by
  flowers, but leaves are present in every one.
- The "background" bucket was half winter/unknown Plantae, which still have
  evergreen or persistent leaves — so it is not a clean negative set. Real
  negatives should be non-plant only (fungi/animals/ground/trunks).

So: precision on proposals is measurable; recall is not, and abstention is not
trustworthy enough to treat a missing box as "no organ".

## 5. Needle second pass (crop-level filter)

Two-stage test over the 376 needle candidates (9.85 s; crops at
`/srv/fast/anatomy4-shenandoah/targeted/needle-pass2`):

1. **Crop + second YOLOE prompt** (`conifer needles`, `pine needles`,
   `spruce branch`) on a 15%-margin crop. Best: 66.7% at crop score ≥0.40
   (n=3); 50% at ≥0.30 (n=8). No setting reaches 85%.
2. **Crop + CLIP ViT-B/32 conifer-vs-confuser classifier** (4 positive,
   7 negative prompts). Best 45.0% at ≥0.7; the scores saturate at 0.999 for
   almost every foliage crop.
3. **Crop + contrastive YOLOE class set** (conifer vs mushroom/broadleaf/
   forest floor/trunk/moss/flower, keep class 0). Worse: 29.2% at the lowest
   floor, and confidences collapse (0 proposals above 0.10).

Box resolution is the second failure: of the 10 correct needle proposals,
7 have box area >0.60 (a close-up filling the frame) and only 3 are both
correct and tight enough to be a useful object box — **3 useful needle boxes
out of 39 reviewed (7.7%)**.

Conclusion: no targeted whole-image setting and no crop filter reaches the
85% bar. Needle stays review-only.

## 6. Production review bands

Auto-accept means "no human look before the label enters the dataset". Bands
below are the evidence-supported maximum; `n` is the reviewed support.

| class | prompt | auto-accept | review band | abstain/reject | evidence |
| --- | --- | --- | --- | --- | --- |
| leaf | `leaf` | ≥0.30 (plant-observation streams only) | 0.10–0.30 | <0.10, or any non-plant image | own-pool ≥0.30 3/3 here + 4/4 in the broad audit |
| flower | `single flower` | ≥0.30 | 0.10–0.30 | <0.10 | 85.7% at ≥0.30 in BOTH audits (6/7 here, 6/7 broad) |
| inflorescence | `cluster of flowers` | none | ≥0.10 (all) | <0.10 | 54.5% at best; highest scores are false positives |
| needle | `spruce needles` (candidates only) | none | all candidates, crops | — | 66.7% best; crop filters 29–67% |

Operational rules:

- Run leaf/flower once per image; run inflorescence/needle only for review.
- Never auto-accept leaf on an image whose observation is not Plantae — the
  mushroom/moss false positives score above 0.85.
- A missing needle/inflorescence box is not evidence of absence; do not use
  detector abstention to prune those taxa.
- Review bands should show the crop, not the whole scene, for needle
  (`needle-pass2/crops`), and whole-image context for flower/inflorescence
  because the class boundary is the judgment.

Estimated usable yield per 1,000 candidates (precision × 1000): leaf `leaf`
833, broadleaf 818, leaf blade 714; flower single 700; inflorescence cluster
545; needle spruce 667 (but only 3/39 useful boxes).

At thresholds over the full 390-image pool: leaf ≥0.30 → 109 proposals
(279/1000 images), flower ≥0.30 → 23 (59/1000), inflorescence ≥0.30 → 20
(51/1000), needle ≥0.30 → 9 (23/1000).

## 7. Next dataset size (defensible)

From one 390-image mixed sample:

- ~109 leaf proposals at ≥0.30; ~100 usable after spot checks.
- ~23 flower proposals at ≥0.30; ~20 usable (single flower only).
- ~20 inflorescence proposals at ≥0.30; treat all as reviewed (≈8–11 usable).
- ~9 needle proposals at ≥0.30; ~3 useful boxes (review-only).

So **~130 auto-accepted + ~50 reviewed labels per 390 images** (≈330–460 per
1,000 mixed images), before any manual middle-band work. A **2,000-image**
targeted collection (needle 30%, flower 30%, leaf 30%, non-plant negatives 10%)
is therefore the largest set that can be labeled defensibly without training:
≈650–900 reviewed labels, with needle contributing ~50 review-only boxes.
Scaling to 10k images is not defensible until a needle-specific capture or
model path exists.

## 8. Commands (reproduce)

```bash
# targeted manifest (host, read-only DB)
.venv/bin/python -m experiments.anatomy4.cli build-manifest --mode targeted \
  --needle-observations 60 --photos-per-needle-observation 2 \
  --leaf-observations 120 --flower-observations 140 \
  --background-observations 48 \
  --out experiments/anatomy4/artifacts/targeted/manifest.json

# download (on .245, anatomy4-audit:0.1)
docker run --rm --network host -u 1002:1002 \
  -v /home/valkyrie/anatomy4-build/repo:/workspace \
  -v /srv/fast/anatomy4-shenandoah:/out \
  -v /home/valkyrie/obs-media-vault/photos:/vault:ro -w /workspace anatomy4-audit:0.1 \
  python -m experiments.anatomy4.cli download \
    --manifest /out/targeted/manifest.json --images-dir /out/targeted/images \
    --vault-root /vault --vault-base-url http://10.0.0.245:55434/photos \
    --out-manifest /out/targeted/manifest.downloaded.json

# one prompt-variant pass per class/bucket (anatomy4-yoloe:0.1, GPU)
python -m experiments.anatomy4.cli prelabel-yoloe-variants \
  --manifest /out/targeted/manifest.downloaded.json \
  --config experiments/anatomy4/configs/anatomy4-targeted.json \
  --variants experiments/anatomy4/configs/targeted-needle.json \
  --out /out/targeted/prelabel-needle --bucket needle

# merge, render, audit (anatomy4-audit:0.1)
python -m experiments.anatomy4.cli merge-runs --inputs /out/targeted/prelabel-*/proposals.json \
  --out /out/targeted/proposals.json
python -m experiments.anatomy4.cli render --manifest /out/targeted/proposals.json \
  --out /out/targeted/overlays --sample-per-class 36
python -m experiments.anatomy4.cli audit --manifest /out/targeted/proposals.json \
  --review /out/targeted/review.csv \
  --background-review /out/targeted/review-background-filled.csv \
  --out /out/targeted/audit

# needle crop second pass + CLIP filter (anatomy4-yoloe:0.1, GPU)
python -m experiments.anatomy4.cli needle-second-pass \
  --manifest /out/targeted/proposals.json --out /out/targeted/needle-pass2
python -m experiments.anatomy4.cli needle-clip-filter \
  --second-pass /out/targeted/needle-pass2 --out /out/targeted/needle-clip
```

## 9. Known limits

- 32–44 reviewed proposals per class, one reviewer, no inter-annotator
  agreement. Precision only; recall unmeasured except through the empty-image
  sample.
- The conifer population is small (87 observations), so needle estimates come
  from 60 observations and 83 images.
- Sampling is bucket-selected by taxonomy, so the class pools are not the
  natural prior; the yield-per-1000 numbers are conditional on that sampling.
- CLIP is ViT-B/32 (the only CLIP in the GPU image); BioCLIP was not run to
  avoid competing with the live sidecar on the 8 GB card.
- No training. No labels were written from this pass except the review CSVs.

## 10. Files

- `artifacts/targeted/audit/{audit.json,audit.md}`
- `artifacts/targeted/review/review.csv` (171 verdicts), `review-background.csv` (53)
- `artifacts/targeted/review/*.jpg` (12 class sheets)
- `artifacts/targeted/needle-pass2/{yoloe-pass.csv,clip-filter.csv,contrast-pass.csv}`
- `artifacts/targeted/manifest.downloaded.json`
- On `.245`: `/srv/fast/anatomy4-shenandoah/targeted/` (images, overlays,
  proposals, crops)
- New code: `manifest.py` targeted sampling, `yoloe.py` variant runner,
  `render.py` prompt-stratified sheets + empty sheets, `audit.py` per-prompt
  and empty-review metrics, `needle_crop.py`, `needle_clip_filter.py`, CLI
  `merge-runs`, `needle-second-pass`, `needle-clip-filter`.