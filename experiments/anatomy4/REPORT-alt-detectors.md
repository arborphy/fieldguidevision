# Anatomy4 — alternate open-vocab detectors (YOLO-World v2 vs OWLv2)

Package: `anatomy4-alt-detectors`
Contract: `anatomy4-v1` (four visual classes: `leaf`, `needle`, `flower`,
`inflorescence`)
Status: complete (development data only — **no final test set was touched**)
Branch/worktree: `anatomy4-alt-detector` (isolated worktree; nothing committed)

## Question

The approved detector (`yoloe-26s-variants`) is usable for `leaf` but weak on
`needle` (25.6 % reviewed precision, 7.7 % correct-and-tight) and on
`flower` / `inflorescence` (~32 %). Before paying for manual bbox labeling,
test at most two alternate open-vocabulary detectors on the **already reviewed**
audit sets and decide:

1. Is either a better proposal generator for needle and flower/inflorescence?
2. Is a per-class hybrid worth the extra runtime?
3. What are reproducible operating settings for producing proposals?

Candidates: **YOLO-World v2** (`yolov8x-worldv2`, Ultralytics) and **OWLv2**
(`google/owlv2-base-patch16-ensemble`, Transformers). Grounding-DINO + SAM was
already rejected in `REPORT.md` and was not re-run.

## Method

- Same manifests as the approved runs: targeted set `391 selected / 390
  downloaded` (flower 140, leaf 120, needle 83, background 48) and the broad
  120-image sample (`/out/manifest.downloaded.json`, no audit buckets).
- Same class prompts as the YOLOE variants run (`ALT_PROMPTS`), one detection
  pass per image, per-class NMS (IoU 0.5), `max_per_class` 10, area guards
  (`dominates_frame` > 0.95, `too_small` < 0.0005), score floor 0.02.
- Review protocol identical to the approved runs: large-tile per-class contact
  sheets at 9 tiles per sheet, plus needle detail sheets and background
  (negative) sheets; verdicts are `tp` / `fp` with a note. Background images in
  the "background" bucket are **not** clean negatives — several are winter
  Plantae / dormant-plant photos — so a `tp` there is recorded as such.
- Everything is measured on proposals a human actually reviewed. No metric is
  inferred from taxonomy, and box-only proposals score exactly like masked ones.

### Class prompt sets

| class | prompts |
| --- | --- |
| leaf | `leaf`, `leaf blade`, `broadleaf` |
| needle | `pine needles`, `spruce needles`, `conifer foliage` |
| flower | `single flower`, `flower` |
| inflorescence | `cluster of flowers`, `umbel of flowers`, `flower head` |

## Runtime and memory (RTX 4060 Ti 8 GB, BioCLIP resident)

| detector | images | wall | s/img | peak VRAM | proposals |
| --- | --- | --- | --- | --- | --- |
| OWLv2 targeted | 390 | 97.7 s | 0.25 | 948.7 MB | 6,861 |
| OWLv2 broad | 120 | 28.8 s | 0.24 | 948.7 MB | 1,777 |
| YOLO-World v2 targeted | 390 | 13.4 s | 0.034 | 257.2 MB | 956 |
| YOLO-World v2 broad | 120 | 3.1 s | 0.026 | 257.1 MB | 244 |

OWLv2 is ~7× slower per image than the current YOLOE run (0.037 s/img) and
~50× slower than YOLO-World, but it is still only ~100 s for 390 images. VRAM
stays under 1 GB, so both fit comfortably alongside BioCLIP.

## Headline comparison — targeted set (reviewed)

Legacy audit convention (ferns not counted as `leaf`; Asteraceae heads counted
under `flower` when boxed as a single head). Percentages are tp / decided.

| detector | leaf | needle | flower | inflorescence | correct+tight | overall |
| --- | --- | --- | --- | --- | --- | --- |
| `yoloe-26s-variants` (approved) | 77.3 % (44) | 25.6 % (39) | 31.8 % (44) | 31.8 % (44) | 34.5 % | 42.1 % |
| **`owlv2-base`** | 45.8 % (24) | **33.3 % (36)** | **78.1 % (32)** | **53.1 % (32)** | **50.8 %** | **52.4 %** |
| `yolo-world-v2` | 75.0 % (4) | — (0 proposals) | 41.7 % (24) | 29.2 % (24) | 38.5 % | 38.5 % |

Matched-N (top-N reviewed proposals by score, so different sample sizes compare
on the same proposal count):

| detector | leaf @12 | needle @12 | flower @12 | inflorescence @12 | tight heavy? |
| --- | --- | --- | --- | --- | --- |
| yoloe-26s-variants | 58.3 % | 41.7 % | 66.7 % | 25.0 % | no |
| owlv2-base | 58.3 % | 41.7 % | **83.3 %** | **66.7 %** | no |
| yolo-world-v2 | n/a (4) | n/a (0) | 41.7 % | 29.2 % | no |

At 24 reviewed proposals: OWLv2 flower 79.2 % vs YOLOE 45.8 %, inflorescence
54.2 % vs 25.0 %, needle 41.7 % vs 29.2 %, leaf 45.8 % vs 66.7 %.

### Supplementary broad-set check (OWLv2, n = 9 per class)

The broad 120-image manifest carries no `audit_bucket`, so the class sheets
sampled 9 proposals per class (an independent, unbiased-by-bucket draw):
`artifacts/alt-review/compare-broad.json`.

| class | precision | tight | median box area | ≥0.10 sweep |
| --- | --- | --- | --- | --- |
| leaf | 66.7 % (6/9) | 6/9 | 0.019 | 5/7 |
| needle | 33.3 % (3/9) | 3/9 | 0.108 | 2/5 |
| flower | 77.8 % (7/9) | 7/9 | 0.016 | 5/7 |
| inflorescence | 55.6 % (5/9) | 5/9 | 0.016 | 4/5 |
| **overall** | **58.3 %** | **55.6 %** | — | — |

Broad-set numbers reproduce the targeted ranking closely (flower ~78 %,
inflorescence ~55 %, needle ~33 %), which is the strongest evidence that the
class ordering is not an artefact of the targeted sampling.

YOLO-World broad was not reviewed beyond the targeted sample: it produced **0
needle proposals in 510 images** and 5 leaf proposals total, which disqualifies
it before review effort is justified.

## Per-class findings

### Needle — the reason OWLv2 was worth testing

OWLv2 finds needles where YOLOE cannot, and its boxes are small rather than
whole-tree (median box area 0.055 vs 0.404 for YOLOE). That is the difference
between "a box the curator can accept" and "a box the curator must re-draw".

| detector | reviewed | precision | correct+tight | median area | ≥0.10 |
| --- | --- | --- | --- | --- | --- |
| yoloe-26s-variants | 39 | 25.6 % | 3 (7.7 %) | 0.404 | 6/15 |
| owlv2-base | 36 | 33.3 % | 12 (33.3 %) | 0.055 | 8/18 |
| yolo-world-v2 | 0 | — | — | — | — |

Prompt breakdown matters more than the class score:

| prompt | OWLv2 precision (n) | YOLOE precision (n) |
| --- | --- | --- |
| `conifer foliage` | 41.2 % (17) | 16.7 % (12) |
| `spruce needles` | **45.5 % (11)** | 44.4 % (9) |
| `pine needles` | **0 % (8)** | 28.6 % (14) |
| `needle-bearing branch` | not used | 0 % (4) |

`pine needles` is a trap prompt for OWLv2 — every one of its 8 reviewed
proposals was a false positive (grass, moss, litter, twig). `conifer foliage`
and `spruce needles` carry the signal. Recommended needle prompt set for OWLv2:
`conifer foliage`, `spruce needles`, `pine branch` (drop `pine needles`; the
broad sample reproduced the same split: conifer 1/3, spruce 2/2, pine 0/4).

The recurring OWLv2 needle false positives are moss cushions, clubmoss /
ground-pine, grass blades and leaf litter — the same confusions that dominate
YOLOE, but they sit at lower scores and tighter boxes, so thresholding removes
more of them.

### Flower — clear OWLv2 win

| detector | reviewed | precision | correct+tight | median area |
| --- | --- | --- | --- | --- |
| yoloe-26s-variants | 44 | 31.8 % | 14 (31.8 %) | 0.017 |
| owlv2-base | 32 | **78.1 %** | 25 (78.1 %) | 0.023 |
| yolo-world-v2 | 24 | 41.7 % | 10 (41.7 %) | 0.372 |

OWLv2's `single flower` prompt is doing almost all of the work (78.1 % over 32
reviewed). The `flower` prompt is the weaker of the two on the targeted set but
is retained because YOLO-World showed the opposite ordering — it is cheap to
keep both and let NMS merge them.

### Inflorescence — OWLv2 matches the best YOLOE prompt, with tighter boxes

| detector | reviewed | precision | correct+tight | median area |
| --- | --- | --- | --- | --- |
| yoloe-26s-variants | 44 | 31.8 % | 11 (25 %) | 0.188 |
| owlv2-base | 32 | **53.1 %** | 15 (46.9 %) | 0.028 |
| yolo-world-v2 | 24 | 29.2 % | 7 (29.2 %) | 0.403 |

OWLv2's `cluster of flowers` is the best single inflorescence prompt
(83.3 % over 6), `umbel of flowers` 48.0 % over 25. YOLOE's best inflorescence
prompt in isolation was `cluster of flowers` (53.8 %) but the merged YOLOE run
dilutes it with `flower spike` (16.7 %) and `umbel of flowers` (26.3 %).

Known failure mode: OWLv2 will happily box a *partial* region of a large white
flower spike, and will box a single Asteraceae head as `inflorescence`. Both
are acceptable under the current contract ("one box per distinct bloom or
cluster"), but the partial-spike boxes need a curator decision at label time.

### Leaf — OWLv2 is worse and floods

OWLv2 leaf precision 45.8 % (24 reviewed) vs YOLOE 77.3 % (44 reviewed), at
9.2 proposed leaves per image vs 3.3 for YOLOE. It also produces tiny boxes in
leaf litter and on grass blades. **Do not use OWLv2 for leaf.**

Two contract subtleties that change the number:

- Ferns: one reviewed leaf false positive was a fern frond. Under the current
  contract (ferns count as `leaf`) that becomes a true positive, so OWLv2 leaf
  rises 11/24 → 12/24 (45.8 % → 50.0 %). YOLOE's leaf number would rise too;
  the ranking does not change.
- Broadleaf-on-conifer: OWLv2 boxes spruce foliage under the `broadleaf`
  prompt. Under a contract where `leaf` excludes needle-bearing foliage that is
  a hard false positive (it is counted as one here).

## Contract sensitivity (read before using these numbers)

The review passes above were scored under the legacy convention. Under the
coordinator's current contract rules the following deltas apply:

- **ferns = leaf** → both detectors gain a small number of leaf TPs
  (OWLv2 +1 in the reviewed sample; OWLv2 leaf 50.0 %).
- **moss / lichen / grass ≠ leaf** → already scored as false positives; no
  change.
- **Asteraceae heads = inflorescence** → shifts some reviewed `flower` TPs to
  `inflorescence` (mostly for OWLv2's single-head boxes). It does not change
  the detector ranking but it does mean the flower/inflorescence split in the
  table above is convention-dependent; treat the two as a combined
  "reproductive organ" score when deciding.
- **needle = one box per needle-bearing shoot** → this is *stricter* than how
  the audit sheets were scored. OWLv2's needle boxes are shoot-to-branch scale
  (median area 0.055) so most would survive, but whole-crown boxes would not.
  A strict re-score would lower both detectors' needle precision; OWLv2's
  tighter boxes mean it loses less.

No number in this report has been silently re-scored; the note exists so the
contract change is not mistaken for a regression.

## Operating recommendation

**Hybrid, per class.** The two detectors are complementary and the cost is one
extra pass:

| class | detector | threshold | exported rows (targeted set) |
| --- | --- | --- | --- |
| leaf | `yoloe-26s-variants` | existing | (unchanged) |
| needle | `owlv2-base` | 0.10 | 754 |
| flower | `owlv2-base` | 0.15 | 247 |
| inflorescence | `owlv2-base` | 0.15 | 268 |

Settings live in `configs/anatomy4-alt-detectors-export.json`; the run settings
(model, prompts, NMS, score floor, caps) are in
`configs/anatomy4-alt-detectors.json`. `max_box_area` 0.60 drops the
frame-dominating boxes that OWLv2 produces on wide shots (280/6,861 targeted
proposals were rejected this way at inference time; the export applies the same
guard again).

**Export (reproducible):** `/out/alt-owlv2/export-targeted/` →
`proposals.csv`, `proposals.jsonl`, `export.json`; local copy at
`artifacts/alt-review/owlv2/export-targeted/`. 1,269 rows: needle 754, flower
247, inflorescence 268, leaf 0. `export.json` carries the model revision,
prompt→class mapping, query order, NMS/cap settings and a config fingerprint,
so the proposal set can be regenerated bit-for-bit from the manifest.

Threshold honesty: needle at 0.10 is 8/18 (44 %) reviewed precision on the
targeted set. That is a *proposal* threshold, not a label threshold — the
curator still decides. Raising to 0.20 leaves 5 proposals (1 TP) in the
reviewed sample — useful as a "high-confidence first" sort key.

**Do not adopt YOLO-World v2.** Zero needle proposals across 510 images, a
0.003–0.05 score band, and 41.7 % flower / 29.2 % inflorescence with
whole-object boxes. It is fast and cheap, but speed is irrelevant when the
class of interest is never proposed.

## Repro commands

Container image: `anatomy4-alt:0.1` (`Dockerfile.alt`, adds `scipy` for OWLv2's
slow image processor; base YOLOE image lacks it). Data at `/out`, HF cache at
`/models`.

```bash
# OWLv2 proposals (targeted)
docker run --rm --network host --device nvidia.com/gpu=all -u 1002:1002 \
  -v /home/valkyrie/anatomy4-alt-detector/repo:/workspace \
  -v /srv/fast/anatomy4-shenandoah:/out \
  -v /home/valkyrie/anatomy4-alt-detector/models:/models \
  -e HF_HOME=/models -w /workspace anatomy4-alt:0.1 \
  python -m experiments.anatomy4.cli prelabel-owlv2 \
    --manifest /out/targeted/manifest.downloaded.json \
    --config experiments/anatomy4/configs/anatomy4-alt-detectors.json \
    --out /out/alt-owlv2/targeted --device cuda

# YOLO-World v2 (needs a writable Ultralytics config dir)
docker run --rm --network host --device nvidia.com/gpu=all -u 1002:1002 \
  -e YOLO_CONFIG_DIR=/tmp/ultra ... \
  python -m experiments.anatomy4.cli prelabel-world \
    --manifest /out/targeted/manifest.downloaded.json \
    --config experiments/anatomy4/configs/anatomy4-alt-detectors.json \
    --out /out/alt-worldv2/targeted --device cuda

# review sheets, empty-image/yield stats
python -m experiments.anatomy4.cli render-alt-sheets \
  --manifest /out/alt-owlv2/targeted/proposals.json \
  --out artifacts/alt-review/owlv2 --per-class 9 --needle-detail

# metrics over reviewed CSVs (comma-join class + background review dirs)
python -m experiments.anatomy4.cli compare \
  --run owlv2-base /out/alt-owlv2/targeted/proposals.json \
        artifacts/alt-review/owlv2,artifacts/alt-review/owlv2/review-bg \
  --dataset targeted --out artifacts/alt-review/compare-targeted.json

# production proposal export
python -m experiments.anatomy4.cli export-proposals \
  --manifest /out/alt-owlv2/targeted/proposals.json \
  --out /out/alt-owlv2/export-targeted \
  --thresholds experiments/anatomy4/configs/anatomy4-alt-detectors-export.json \
  --max-box-area 0.60
```

Tests: `PYTHONPATH=. python -m pytest experiments/anatomy4/tests -q`
(16 tests — box schema, NMS/cap, area guards, review CSV round-trip,
precision/tight/matched-N math, export thresholds, sheet sampling).

## Risks and limits

- Reviewed samples are small (24–44 per class targeted, 9 per class broad).
  A single verdict moves a class ~2–4 points; the ranking is robust, the exact
  percentages are not.
- The broad sample has no audit buckets, so it cannot be re-weighted by class
  prior. It is a sanity check on ordering, not a prevalence estimate.
- OWLv2's threshold behaviour is prompt-dependent (0.10 for needle is not
  comparable to 0.10 for leaf). Thresholds must be set per class, which is why
  the export config is a map rather than one number.
- Contract re-scoring (ferns, Asteraceae heads, shoot-level needle boxes) has
  not been applied to the reviewed CSVs. It is expected to help OWLv2 slightly
  and to compress the flower/inflorescence split.
- OWLv2 boxes are proposals only. Nothing here justifies training on them
  without the same human acceptance gate as the approved pipeline.
- 7× slower than YOLOE is irrelevant at 390 images and material at 100 k
  images (≈7 h vs 1 h on one 4060 Ti); batch scale-out should re-measure.

## Recommended next steps

1. Generate the hybrid proposal set on the **holdout** manifests and measure
   how many human acceptances per hour it yields against the approved
   detector — the real decision metric, not precision alone.
2. Drop `pine needles` from OWLv2's needle prompts and re-run the needle class;
   expected +5–10 points at the same row count.
3. Adopt the shoot-level needle box convention explicitly and re-score the
   needle review CSVs; report the delta for both detectors.
4. If needle remains the bottleneck, spend labeling effort there first: it is
   the class where the approved detector and both alternatives are weakest and
   where boxes are cheapest to accept (OWLv2 median box 0.055 of frame).
