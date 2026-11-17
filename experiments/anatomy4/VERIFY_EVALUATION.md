# Anatomy4 crop verifier — evaluation against the human YOLOE-2 review

Scope: crop-level verification of the 94 human-reviewed `yoloe2` proposals in
`experiments/anatomy4/artifacts/review-yoloe2/review.csv`, using a cheap VLM
through the existing OpenRouter credential. No training, no scaling.

- Verifier: `experiments/anatomy4/verify.py` (CLI: `verify-crops`,
  `verify-report`, `verify-cards`).
- Model: `openai/gpt-4o-mini` via OpenRouter, `temperature=0`, JSON mode.
- Prompt: `anatomy4-verify-crop-v1` (sha `8b295eb58a7ef7fc`), crops only.
- Evidence: 94 primary verdicts + 19 needle second-pass verdicts, 113 crop
  calls, **$0.25**. Total exploration spend across all variants: **$0.39 /
  176 crop calls** (bound was 200).
- Mean latency 2.9 s per call (1.4–5.6 s), zero retries, zero failures.

## What the verifier is given

Only the proposed crop's real pixels (margin 0.18 around `detector_bbox`), the
detector's proposed class, and visual class definitions. It never sees the taxon
name, the observation id, or the photo id; the audit guard rejects any reason
that names a binomial/family/order or a manifest-supplied taxon name, and
rejects any verdict carrying a box, point, or polygon field.

## Headline agreement (accept decision vs human `tp`)

Policy `strict` (class match + resolved + tight box + confidence ≥ 0.70):

| class | n | TP | FP | FN | precision (95% CI) | recall (95% CI) | trust |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| leaf | 24 | 20 | 1 | 1 | 0.952 (0.77–0.99) | 0.952 (0.77–0.99) | review-only |
| needle | 24 | 4 | 1 | 0 | 0.800 (0.38–0.96) | 1.000 (0.51–1.00) | review-only |
| flower | 22 | 12 | 2 | 2 | 0.857 (0.60–0.96) | 0.857 (0.60–0.96) | review-only |
| inflorescence | 24 | 10 | 1 | 6 | 0.909 (0.62–0.98) | 0.625 (0.39–0.82) | review-only |
| **all** | 94 | 46 | 5 | 9 | **0.902** (0.79–0.96) | 0.836 (0.72–0.91) | review-only |

Policy `balanced` (same, but self-reported confidence and tightness do not gate;
tightness is recorded as a review flag): TP 49 / FP 5 / FN 6, **P 0.907,
R 0.891**, with leaf and flower/inflorescence precision unchanged and
inflorescence recall up 0.625 → 0.75.

All three profiles are also machine-generated in `artifacts/verify-yoloe2/report.md`
(§Accept-profile comparison, `report.json.profile_sweep`) so these numbers are
reproducible without re-spend.

### Verdict-class confusion (rows = human, columns = verifier)

| human \ verifier | leaf | needle | flower | inflorescence | not_target | ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| leaf | 21 | 0 | 0 | 0 | 2 | 0 |
| needle | 0 | 4 | 0 | 0 | 0 | 0 |
| flower | 0 | 0 | 13 | 0 | 6 | 0 |
| inflorescence | 0 | 0 | 4 | 14 | 0 | 1 |
| not_target | 0 | 1 | 1 | 4 | 21 | 2 |

There is **no cross-organ confusion**: a leaf is never called a needle, a needle
never a leaf, and a bloom is never called foliage. Every class-level error is
inside the bloom pair (flower ↔ inflorescence) or is a *presence/subject* call
(`not_target`). That is the single most useful property of this verifier.

## Disagreements, inspected one by one

Fourteen accept-decision disagreements were rendered as evidence cards
(`artifacts/verify-yoloe2/disagreements/`, sheet:
`disagreements-sheet.jpg`) and read back at full resolution.

**Verifier is right, human review is wrong (2).**

- `a4y-28831444-leaf-001` — human `tp` "low confidence single blade"; the crop
  is a piece of dead wood on rock with moss. Verifier: `not_target` ("the crop
  shows wood, not a leaf"). Correct reject.
- `a4y-52720808-flower-000` — human `tp` "single thistle head"; the crop is
  dominated by a hand holding a thistle head. Verifier `not_target` at 0.00 is
  defensible; at minimum this is a labeling-convention casualty (a composite
  head is arguably an inflorescence).

**Human review is defensible, verifier is wrong (3).**

- `a4y-92726718-needle-000` — human `fp` "box on grass below pine". The box
  covers 18% of the frame: ground, moss, and distant spruce branches. Verifier
  `needle` at **1.00**. This is a whole-frame false accept, and it is why the
  `main_subject` prompt variant exists (below).
- `a4y-149161437-inflorescence-000` — human `fp` "ambiguous foliage/window box".
  Verifier `inflorescence` at **1.00**, "clearly visible among leaves". The crop
  is mostly foliage with a few pale specks at the frame edge. Overconfident.
- `a4y-42240415-flower-001` / `-003` — human `fp` "flower cluster belongs to
  inflorescence". The crop contains exactly one yellow bloom. The verifier is
  literally correct about the pixels; the human is applying the dataset's
  semantic rule that the *object* is the cluster. This is a **definition
  conflict, not a perception error**, and it is not fixable by prompting alone.

**Both sides defensible — crop-vs-scene convention (3).**

- `a4y-74975008-inflorescence-000` — human `tp` "grass inflorescence"; the crop
  is a yellow composite head. Verifier `flower`. Human note looks wrong.
- `a4y-41064040-inflorescence-000`, `a4y-624050433-inflorescence-006/-007` —
  the box lands on one bloom inside a cluster; the verifier either calls it
  flower or flags `too_tight`. These are the tightness-flag cases.

**Neither side can decide (2).**

- `a4y-719447557-inflorescence-002` — a ~35 px yellow blur. Verifier `ambiguous`
  at 0.00; human accepted. The human was generous.
- `a4y-44147331-inflorescence-000` — crop is a clear white flower cluster with
  beetles; the verifier named `inflorescence` **and then reported confidence
  0.00**. Contradiction inside one verdict; the class is right.

**One padded-crop rescue (1).**

- `a4y-719564852-leaf-004` — human `fp` "tiny box no leaf resolved". The 0.18
  margin crop clearly shows a violet leaf, so the verifier accepts. A 0.05
  margin re-run on this proposal *also* accepted, so this is a subject-scale
  judgment (a ~64 px leaf) rather than pure padding leakage — but padding can
  still rescue a box the detector did not resolve. Use the tight-margin variant
  as a confirmation pass before trusting an accept near the small-box floor.

## Findings that change the recommended policy

1. **Self-reported confidence is not a usable threshold.** 42 verdicts sit at
   0.0 and 49 at 1.0; the accept-metric sweep is *flat* from 0.00 to 0.90
   (`report.md` §sweep). gating at 0.95 costs 17 true accepts for 0.0 precision
   gain. Treat `confidence` as a triage hint only. Conversely, a 0.00 with the
   right class (`a4y-44147331`) should be treated as "class agrees, needs a
   human", not as a rejection.
2. **`bbox_tightness` is a recall tax, not a precision gain.** Rejecting
   `too_tight` removes 3 human-accepted boxes and adds 0 false accepts. Keep it
   as a **review flag** (`balanced`), not a hard gate.
3. **`main_subject` prompt variant kills whole-frame false accepts.** Re-run over
   all 24 needle proposals: 2 verdicts flipped, both from accept → reject —
   `a4y-92726718-needle-000` (the real FP) and `a4y-706362511-needle-000` (a
   distant conifer the human accepted). Needle: P 0.80 → **1.00**, R 1.00 →
   0.75 (n=3 positives). Worth using, but neither variant is under-written at
   n=4 positives.
4. **The second pass on low-confidence needles is not worth its spend.** 19
   crops / $0.059 produced no accept-decision change (it only restated
   `not_target`). Drop it until a class actually has a recall problem, and use
   `main_subject` instead.
5. **Downscaling crops saves nothing.** 384 px, 512 px, and 900 px crops of the
   same content all billed **8,950 prompt tokens**; only image *area* moves the
   number (full-frame 900 px = 25,951). Keep 900 px for legibility; the lever is
   *not sending* crops, not shrinking them.
6. **The verifier is blind to the one distinction the dataset exists for.** The
   residual errors concentrate on flower vs inflorescence. A family-level gate
   ("a bloom is present") measures P 0.867 / R 0.867 on the 46 bloom proposals —
   *not better* than the class gate, because the remaining FPs are
   presence/subject calls, not class calls.

## Recommended policy per class

| class | recommendation | why |
| --- | --- | --- |
| leaf | **review-only now; auto-accept candidate** | P 0.952 / R 0.952, the only FP is a subject-scale judgment. The 95% precision lower bound (0.77) is below any safe gate at n=24 — re-run at ≥100 leaf proposals and auto-accept if P stays ≥0.95. |
| needle | **review-only; never unattended** | only 4 human-accepted needles in the sample; CI 0.38–0.96. Use `main_subject` to suppress whole-frame accepts, and treat every needle accept as a human decision. |
| flower | **review-only** | the two FPs are the "single bloom inside a cluster" definition conflict. The verifier cannot resolve it from pixels alone. |
| inflorescence | **review-only** | lowest recall (0.63 strict / 0.75 balanced); heavy `too_tight` and `flower` interference. |
| all classes | **ship `balanced`, not `strict`** | same precision (+0.005), +0.055 recall, and the tightness signal survives as a flag instead of a silent rejection. |

Concretely: use the verifier as a **triage layer that removes no human
judgment** — reject `not_target`/`ambiguous`/invalid outright (that is 44 of 94
proposals, ≈47% of the review queue removed at P 0.90), and send every accept to
the human queue *pre-filled*, ranked by verifier class agreement first.

Thresholds, if a numeric gate is required anyway:
`min_confidence = 0.0` (it carries no signal), `accepted_tightness = keep +
too_tight with the non-keep cases flagged`, `require_resolved = true`,
`profile = "balanced"`.

## Guards that fired

- **No taxon evidence**: 0 verdicts rejected on the real run after fixing a
  guard bug — the generic binomial regex was matching sentence-initial English
  ("The crop shows bark…"), which invalidated 3 correct needle rejects. Ordinary
  capitalised words are now excluded (regression test included).
  The guard does fire on a real binomial/family (`Quercus alba`, `Fagaceae`,
  manifest common names like "eastern hemlock") — unit-tested.
- **No invented boxes**: 0 of 113 verdicts carried a box/point/polygon field.
- **Skipped proposals**: 0 (all 57 frames present; every crop decoded).

## Joint detector-score policy

The verifier verdicts are also combined with the detector's own `box_score` in
`POLICY.md` / `artifacts/policy-yoloe2/policy.json`. Leaf and inflorescence
reach a validated ≥0.90 measured-precision assisted-training policy; flower
does not survive leave-one-out, and needle has no rule with enough support. The
test split is never auto-accepted.

## Secondary runs kept as evidence

| run | purpose | crops | cost | outcome |
| --- | --- | ---: | ---: | --- |
| `verify-yoloe2` | primary, margin 0.18, 900 px, v1 prompt + needle second pass | 113 | $0.2505 | the table above |
| `verify-yoloe2-needle-mainsubject` | `main_subject` prompt on all 24 needle proposals | 24 | $0.0722 | P 0.80 → 1.00, R 1.00 → 0.75; 2 flips, both accept → reject |
| `verify-yoloe2-mainsubject` | `main_subject` spot-check on 9 mixed proposals | 9 | $0.0281 | killed the whole-frame needle FP, kept the bloom conventions |
| `verify-yoloe2-512` | 512 px crops, 6 per class | 24 | $0.0316 | 95.8% class / 91.7% accept agreement with the 900 px run; **no token saving** |
| `verify-yoloe2-tight` | margin 0.05 confirmation probe on 6 small-box proposals | 6 | $0.0079 | `leaf-004` still accepted → subject-scale, not padding |

## Reproduce

```bash
# 94 reviewed proposals, balanced-context crops
python experiments/anatomy4/cli.py verify-crops \
  --proposals <prelabel-yoloe2>/proposals.json \
  --images-dir experiments/anatomy4/artifacts/verify-yoloe2/images \
  --review experiments/anatomy4/artifacts/review-yoloe2/review.csv \
  --manifest experiments/anatomy4/artifacts/shenandoah-manifest.json \
  --out experiments/anatomy4/artifacts/verify-yoloe2 \
  --env-file /Users/valkyrie/dev/arborphy/.env \
  --batch-size 3 --max-calls 45

# re-score stored responses under a different policy — no network, no spend
python experiments/anatomy4/cli.py verify-report \
  --verdicts  experiments/anatomy4/artifacts/verify-yoloe2/verdicts.jsonl \
  --raw-calls experiments/anatomy4/artifacts/verify-yoloe2/raw_calls.jsonl \
  --review    experiments/anatomy4/artifacts/review-yoloe2/review.csv \
  --manifest  experiments/anatomy4/artifacts/shenandoah-manifest.json \
  --out       experiments/anatomy4/artifacts/verify-yoloe2

# evidence cards for every disagreement
python experiments/anatomy4/cli.py verify-cards \
  --verdicts experiments/anatomy4/artifacts/verify-yoloe2/verdicts.jsonl \
  --review   experiments/anatomy4/artifacts/review-yoloe2/review.csv \
  --images-dir experiments/anatomy4/artifacts/verify-yoloe2/images \
  --out experiments/anatomy4/artifacts/verify-yoloe2/disagreements
```

## Known limits

- One reviewer, one pass, 94 proposals. The "human ground truth" has at least
  two errors of its own, so measured agreement *understates* the verifier.
- The reviewer used the semantic-object convention (a bloom inside a cluster is
  inflorescence); the verifier sees only the crop. Until the contract states
  which convention wins, flower/inflorescence agreement cannot exceed ~0.86.
- Needle has 4 positives — no policy claim about needle precision is
  statistically supportable at this n.
- Image-token cost (~8.9k prompt tokens/crop) is an order of magnitude above
  text-only calls; a 200-crop sweep is ≈$0.30–0.60, so cost is not the
  constraint — reviewer time is.