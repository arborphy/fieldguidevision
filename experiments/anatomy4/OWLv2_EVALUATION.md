# OWLv2 crop-verifier evaluation + assisted-label policy

Detector: `google/owlv2-base-patch16-ensemble` proposals from the alt-detector
benchmark. Verifier: unchanged `openai/gpt-4o-mini` crop verifier
(prompt `anatomy4-verify-crop-v1`, sha `8b295eb58a7ef7fc`), balanced accept
profile, 900 px crops, 0.18 margin.

- Machine policy: `artifacts/policy-owlv2/policy.json` (`policy_id` in the file)
- Per-proposal actions: `artifacts/policy-owlv2/decisions.jsonl`
- Per-class verifier value-add: `artifacts/policy-owlv2/verifier-value.json`
- Generated readable policy: `artifacts/policy-owlv2/policy.md`
- Spend: **160 crop calls / $0.277**, no failed calls, no retries (bound 250)

Source review worktree (read-only): `fieldguidevision-anatomy4-alt-detector`,
`experiments/anatomy4/artifacts/alt-review/owlv2`. Inputs are copied/trimmed
into `artifacts/owlv2-review/inputs/` by `owlv2_prep.py`.

## Populations — kept separate, never pooled

| population | rows | photos | human TP | role |
| --- | ---: | ---: | ---: | --- |
| **targeted class sheets** | 96 (24/class) | 78 | 53 | **fit** |
| **background sheets** | 28 | 17 | 12 | negative/false-accept audit |
| **broad class sheets** | 36 (9/class) | 28 | 21 | **independent validation** |

The background sheets are *not* clean negatives: the reviewer recorded real
organs as `tp` (winter Plantae photos). They are reported separately and are not
used to fit or validate the policy. The broad sheets are a separate sampling of
the broad manifest and are used only as an independent check.

## Verifier value-add vs the raw detector

"raw P" = detector proposals the human accepted ÷ reviewed. "accept P" = the
verifier's accepts that the human accepted.

| pop | class | n | raw P | accepts | TP | FP | accept P (95% CI) | accept R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| targeted | leaf | 24 | 0.458 | 19 | 9 | 10 | 0.474 (0.27–0.68) | 0.818 |
| targeted | **needle** | 24 | 0.500 | 12 | 11 | 1 | **0.917 (0.65–0.99)** | 0.917 |
| targeted | flower | 24 | 0.708 | 13 | 11 | 2 | 0.846 (0.58–0.96) | 0.647 |
| targeted | inflorescence | 24 | 0.542 | 15 | 11 | 4 | 0.733 (0.48–0.89) | 0.846 |
| **targeted** | **all** | 96 | **0.552** | 59 | 42 | 17 | **0.712 (0.59–0.81)** | **0.792** |
| broad | leaf | 9 | 0.667 | 4 | 4 | 0 | 1.000 (0.51–1.00) | 0.667 |
| broad | needle | 9 | 0.333 | 2 | 2 | 0 | 1.000 (0.34–1.00) | 0.667 |
| broad | flower | 9 | 0.778 | 8 | 6 | 2 | 0.750 (0.41–0.93) | 0.857 |
| broad | inflorescence | 9 | 0.556 | 7 | 4 | 3 | 0.571 (0.25–0.84) | 0.800 |
| background | needle | 12 | 0.000 | 1 | 0 | 1 | 0.000 (0.00–0.79) | — |
| background | flower | 8 | 1.000 | 6 | 6 | 0 | 1.000 (0.61–1.00) | 0.750 |
| background | inflorescence | 8 | 0.500 | 5 | 3 | 2 | 0.600 (0.23–0.88) | 0.750 |

The verifier is a strong filter for **needle** (0.500 → 0.917 at 92% recall) and
a moderate one for flower/inflorescence. It adds **nothing for leaf**: OWLv2's
leaf proposals are polluted with grass blades, ferns, bracts and conifer twigs,
and the verifier accepts them as "broad, flat leaf blade".

## Joint OWLv2-score + verifier policy

Fit on the targeted class sheets with the same gates as the YOLOE policy, plus
two honest additions:

1. **assisted-lane precision ≥ 0.90** — the lane that receives labels must meet
   the bar on its own; a clean overall number diluted by val/test does not count;
2. **independent validation precision ≥ 0.80** when the broad set has ≥ 3
   accepts.

| class | action | score ≥ | verifier | accepts | TP | FP | P (95% CI) | R | train TP | LOO P | broad P | verdict |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| leaf | **review** | — (best 0.288) | balanced | 19 | 9 | 10 | 0.474 (0.27–0.68) | 0.818 | — | — | — | no band (best 0.833 at 6 accepts) |
| needle | **review** | 0.0824 candidate | balanced | 9 | 9 | 0 | 1.000 (0.70–1.00) | 0.750 | 5 | **0.750** | 1.000 (1 accept) | apparent band fails LOO |
| flower | **review** | — (best 0.148) | balanced | 13 | 11 | 2 | 0.846 (0.58–0.96) | 0.647 | — | 0.000 | 0.750 | best 0.900 overall but train lane 0.875 |
| inflorescence | **review** | — (best 0.791) | balanced | 15 | 11 | 4 | 0.733 (0.48–0.89) | 0.846 | — | — | 0.571 | best band has only 1 accept |

**No OWLv2 class reaches a validated ≥90% assisted-label band.** The JSON is
therefore all `review`, with the rejected candidates and their exact numbers
recorded per class. This is a measured result, not a threshold choice.

### Why detector-score floors do not rescue the classes

- **leaf** — the verifier's 10 false accepts are interleaved with the 9 true
  accepts (FP at 0.601, 0.273, 0.258, 0.244, 0.203, 0.179, 0.156, 0.144, 0.132).
  The best cut (≥0.288) gives 5 TP / 1 FP (P 0.833) — below target.
- **needle** — a floor at 0.082 removes the only verifier FP and 2 true needles,
  giving 9/9 on the fit set, 5/5 on train, and 1/1 on broad. But leave-one-out
  selection precision is **0.750**: with 12 positives and one FP, the cut is
  fold-dependent. Not shippable on n=24.
- **flower** — a floor at 0.148 reaches exactly 0.900 overall (9 TP / 1 FP), but
  the assisted train lane is 7/8 (0.875) because the surviving FP is in train;
  the broad set is 0.750. No band.
- **inflorescence** — every zero-FP band has ≤1 accept; the 4 verifier FPs span
  0.107–0.356 and sit between true accepts, so no floor ≥5 accepts reaches 0.90.

## Triage value (what the verifier is still good for)

On the targeted train split (71 proposals), rejecting `not_target` / `ambiguous`
/ invalid verdicts:

- **23 proposals deprioritized** (19 human-rejected, **4 human-accepted**)
- review queue 71 → 48 items (**−32%**)
- no auto-assist labels produced

The 4 human-accepted proposals sent to low priority are a real recall risk:
`a4o-116712839-needle-conifer-foliage-357` (conifer seedling), plus three
flower/inflorescence crops the verifier called unclear. Deprioritize means *last
in the queue*, not delete.

## Failure modes (cards in `artifacts/owlv2-review/*/disagreements/`)

1. **Morphological negative-list leak (leaf).** Grass/linear blades
   (`a4o-43814460-leaf-leaf-blade-089`, `a4o-50060455-leaf-broadleaf-023`),
   thistle bracts (`a4o-49330871-leaf-broadleaf-039`), conifer needles
   (`a4o-334682242-leaf-broadleaf-425`, `a4o-726982610-leaf-broadleaf-325`) and
   mossy-rock rosettes (`a4o-117819026-leaf-broadleaf-121`) are accepted as
   "broad, flat leaf blade" at 0.9–1.0. The prompt lists grass/fern/moss as
   excluded but the model does not enforce the list on blade shape. This is the
   dominant leaf failure and makes the class unusable.
2. **Single-vs-cluster boundary (flower/inflorescence).** Whole umbels accepted
   as one flower (`a4o-718294958-flower-single-flower-071`, where the model's own
   reason says "single flower with multiple blooms"), three heads accepted as a
   single bloom (`a4o-710483274-flower-single-flower-042`), and one distant bloom
   accepted as an inflorescence
   (`a4o-163157001-inflorescence-umbel-of-flowers-375`). The model can name the
   right organ family but not the count.
3. **Box-target vs crop-content.** The verifier judges the padded crop, not the
   detector's intent: an off-target box beside the organ is accepted
   (`a4o-87860202-inflorescence-umbel-of-flowers-049` — umbel outside the box;
   `a4o-163157001-needle-pine-needles-717` — brown needles in leaf litter). Wide
   conifer-canopy boxes are accepted as needle
   (`a4o-432131816-needle-conifer-foliage-011`) even though no usable needle box
   exists.
4. **Atypical needle morphology.** Flat conifer needles (yew/hemlock seedling,
   `a4o-116712839-needle-conifer-foliage-357`) are rejected as "foliage"; the
   verifier is tuned to fascicled pine/spruce needles.
5. **Subject scale / upscaling.** Tiny boxes are upscaled to 224 px, so the
   verifier sees a legible bloom in a 0.1% box
   (`a4o-719564852-flower-single-flower-454`, broad). A source-pixel floor was
   tested and does **not** help OWLv2 (FPs are not concentrated at small sizes),
   so this remains a known limit rather than a policy lever.
6. **Blur/tiny crops → under-recall.** The verifier correctly says ambiguous on
   blurred crops the human accepted (`a4o-153548318-flower-single-flower-079`,
   `a4o-39749122-flower-single-flower-045`); flower recall is 0.647 as a result.

Notable human-side noise (the verifier is right): `a4o-81617908-leaf-broadleaf-021`
is a box of fleabane flowers accepted by the human as a "small leaf"; the
verifier's `not_target`/flower reason is correct. As in the YOLOE set, the human
ground truth has its own errors, so measured precision is a lower bound.

## Production recommendations

1. **Use OWLv2 as the proposal generator, not as an auto-labeler.** It raises raw
   needle/flower/inflorescence precision over YOLOE (0.50/0.71/0.54 vs
   0.26/0.32/0.32 targeted) and its boxes are much tighter, but no class passes
   the assisted-label bar.
2. **Ship the triage, not the labels.** Reject `not_target`/`ambiguous`/invalid
   outright (23 of 71 train proposals, −32% queue) and keep every accept in
   human review. Do not auto-assign `leaf` under any OWLv2 policy.
3. **Keep needle on the watch list.** It is the only class with an apparent
   100%-precision band (9/9, 5/5 train, 1/1 broad). It needs ~50–100 more
   reviewed needle proposals (or a stronger needle prompt variant) before the
   LOO gate can pass; re-run `policy-fit` then. Do not ship it early.
4. **Fix the two semantic leaks before re-fitting**: add explicit negative rules
   and examples for grass/linear blades, bracts, and conifer foliage in the leaf
   prompt; add an explicit "count the blooms / one vs many" decision rule for
   flower vs inflorescence. Both failures are prompt/definition problems, not
   detector problems — the same crops fail the same way under YOLOE.
5. **Do not use detector-score floors as a precision lever for OWLv2.** The
   score distributions interleave TPs and FPs in every class; the only
   zero-FP bands are tiny (leaf 6 accepts, inflorescence 1 accept) or
   train-lane-dirty (flower).
6. **Never auto-accept val/test.** The test split is human truth; the policy
   scored 0/0 auto-assists there by construction, and the code enforces it.

## Limits

- 24 targeted proposals per class; every CI is wide (see table). The
  "100% precision" needle candidate is 9 accepts, CI 0.70–1.00.
- The broad validation is 9 per class — enough to contradict a 0.90 claim
  (inflorescence 0.571) but not to certify one.
- Background sheets are not clean negatives (12 of 28 human `tp`), so background
  precision is not a false-positive rate.
- One reviewer, one pass; at least one human error was found by inspection.
- These thresholds are OWLv2-specific; a detector swap invalidates the numeric
  floors while the process and JSON shape carry over.