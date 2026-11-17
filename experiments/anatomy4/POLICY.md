# Joint detector-score / verifier policy — fit against the 94-proposal audit

Machine-readable artifact: `artifacts/policy-yoloe2/policy.json`
(`policy_id` changes only when the evidence or thresholds change).
Per-proposal actions: `artifacts/policy-yoloe2/decisions.jsonl`.
CLI: `python experiments/anatomy4/cli.py policy-fit …`.

This is a re-score of stored detector scores, verifier verdicts and human
verdicts. **No new VLM calls were made.**

## Two lanes, never mixed

| lane | splits | allowed actions |
| --- | --- | --- |
| assisted training | `train` | `auto_assist` (provenance required), `review`, `deprioritize` |
| human-reviewed truth | `val`, `test` | `review` only — `auto_accept: false`, enforced in code |

The policy is scored on the test split for audit, and that audit counts against
it (see the inflorescence caveat), but test labels are never machine-written.

## How a policy is fitted

For each class: enumerate detector score floors (every observed score, plus
none) × four verifier requirement sets (`balanced`, `balanced_resolved_only`,
`strict`, `triage`) and keep the highest-yield rule that satisfies all gates:

1. measured accept precision ≥ **0.90** across all reviewed proposals;
2. at least **5** accepts in the class (support floor);
3. at least **5** accepts that land in the assisted (train) split;
4. the *selection procedure* re-run leave-one-out must still reach **0.85**
   precision — this is the anti-overfit gate.

Rule 4 is what separates a real policy from a fitted coincidence: flower passes
the apparent bar and fails held-out.

## Fitted result

| class | action | detector score ≥ | verifier | accepts | TP | FP | P (95% CI) | R | train TP | LOO P |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| leaf | **auto_assist** | 0.0951 | balanced | 18 | 18 | 0 | 1.000 (0.82–1.00) | 0.857 | 16 | 0.944 |
| needle | review | — | balanced (baseline only) | 5 | 4 | 1 | 0.800 (0.38–0.96) | 1.000 | — | — |
| flower | review | (candidate 0.1797, not shipped) | balanced | 10 | 9 | 1 | 0.900 (0.60–0.98) | 0.643 | — | **0.750** |
| inflorescence | **auto_assist** (probationary) | 0 (none) | balanced | 13 | 12 | 1 | 0.923 (0.67–0.99) | 0.750 | 10 | 0.917 |

Full per-class detail, caveats and rejected alternatives are in `policy.json`;
`policy.md` is the generated readable view.

"Highest yield" is scored on the **assisted lane**, not on total accepts: leaf's
0.0951 floor keeps all 16 train true labels, removes the one train false accept,
and gives up 2 val/test true labels (which are human-reviewed anyway).
Inflorescence keeps the no-floor rule because a floor above its 0.1111 false
accept would trade 2 train true labels for a precision number that is already
1.000 on train; that conservative alternative is recorded under
`rejected_best_alternatives`.

### Queue impact measured on the audited 94

- assisted split (train): **71 → 25 human review items (−65%)**
- 26 proposals auto-assisted (train only), **100% human-accepted**
- 20 deprioritized (19 junk, 1 human-accepted blurred crop)
- 46 train true labels stay in the pipeline: 26 auto + 19 review + 1 low-priority
- val/test: 23 proposals, all human review (9 true accepts, 14 rejects)

`deprioritize` means **last in the queue, not deleted** — the one human-accepted
proposal it catches (`a4y-719447557-inflorescence-002`, a ~35 px blur) stays
recoverable.

## Does a detector-score floor fix needle / flower / inflorescence?

**Needle — no.** The only accepted false positive (0.1996) outranks two accepted
true needles (0.1918, 0.1084). Any floor that removes the false positive also
removes half the true labels, leaving 2 accepts — below the support floor. The
verifier prompt variant is the better lever: `main_subject` measured 3 TP / 0 FP
on all 24 needle proposals (P 1.00, R 0.75, n=3). Still under-powered: recommend
review, with `main_subject` as the confirmation prompt if needle volume grows.

**Flower — no.** The two false accepts (0.2107, 0.1418) sit *between* true
accepts (0.2572 above, 0.1996 below). The best floor (≥0.1797) reaches exactly
0.900 on the fitting data with 10 accepts, but the same selection procedure
leave-one-out scores **0.750** — the 90% is a property of the two FPs we
happened to see, not a stable rule. Ship review.

**Inflorescence — the floor is redundant on the assisted lane.** The single
false accept (0.1111) is the lowest-scoring accept, so a floor above it looks
attractive (P 1.00 overall) but costs 2 true accepts (0.1177, 0.1132) and drops
recall 0.75 → 0.69. On train specifically the rule is already 10/10 without a
floor: the false accept landed in the test split. So the honest statement is
*no floor needed for the assisted lane; the all-reviewed precision is 0.923 and
that is the conservative estimate.*

**Leaf — the floor is cheap and worth taking.** The one false accept (0.0862) is
below every auto-assisted true leaf; a floor at 0.0951 removes it at the cost of
2 val/test true accepts, leaving 18/18 overall and 16/16 on train.

## Recommended use

1. Run `verify-crops` on new proposals (existing spend: ~$0.0022/crop at 900 px,
   ~$0.25 per 100 proposals).
2. Apply `policy.json` per class. On the train split:
   `auto_assist` → write the label with the full provenance bundle;
   `review` → normal human queue; `deprioritize` → end of the queue.
3. On val/test: **always** human. Never let a verifier accept become a test
   label; the policy is evaluated against test, not applied to it.
4. Audit 10% (min 5, until 50/class) of auto-accepted labels. Stop auto-assist
   for a class if the audited precision falls below 0.90.
5. Re-fit the policy whenever the review sheet grows by ~50 proposals; the
   support gates are deliberately set so a policy cannot be born from <5 accepts.

Provenance written with every auto-accepted label: `proposal_id`, `photo_id`,
`class_name`, `detector_score`, verifier class/confidence/tightness/resolved,
`model`, `prompt_hash`, `policy_schema`, `policy_action`. A label missing any of
these is re-queued for review.

## Honest limits

- One reviewer, 94 proposals, 4–26 positives per class. Every CI is wide; the
  LOO gate is a small-sample sanity check, not a substitute for volume.
- The human sheet has at least two errors of its own (a piece of wood marked
  leaf, a single composite head marked inflorescence); measured precision is
  therefore a lower bound in at least two rows.
- Inflorescence auto-assist rests on 10 train labels with one known FP in test.
  It is probationary: audit it, and drop it to review if the first audited batch
  shows a false accept.
- `deprioritize` is a review-queue ordering, not a delete — the verifier still
  produces 1 human-accepted false reject on this set (a blurred inflorescence).
- The detector floors are fitted on Shenandoah/YOLOE-2 score distributions; a
  different detector or image pipeline invalidates the numeric floors (the
  process, gates and JSON shape carry over).