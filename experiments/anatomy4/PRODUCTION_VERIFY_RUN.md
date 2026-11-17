# Bounded production crop-verification pass — training candidates

Run output is published at:

`/srv/fast/anatomy4-shenandoah/train-candidates/verified/`

The final output has 1,200 unique selected proposal verdicts, 580 assisted-label
candidates, 620 review items, all 1,200 crops used, and 304 raw OpenRouter calls
(1,204 crop submissions because four malformed batch responses were retried
individually). Total measured cost: **$1.9941342**. No blind-test data was read
or written and no verdict is marked human truth.

## Inputs and sample selection

- YOLOE leaf + inflorescence: `/srv/fast/anatomy4-shenandoah/train-candidates/proposals.json`
- OWLv2 flower: `/srv/fast/anatomy4-shenandoah/train-candidates/hybrid-proposals.json`
- Photo metadata: `/srv/fast/anatomy4-shenandoah/train-candidates/manifest.jsonl`
- Images: `/srv/fast/anatomy4-shenandoah/train-candidates/images/`
- Selection seed: `anatomy4-production-train-verify-v1`; max 4 proposals per image

| class/source | selected | unique images | score selection |
| --- | ---: | ---: | --- |
| leaf / YOLOE | 400 | 225 | 320 ≥0.20 (80%); 80 in [0.095076, 0.20) |
| inflorescence / YOLOE | 400 | 247 | 220 ≥0.20 (55%); score-stratified remainder below 0.20 |
| flower / OWLv2 | 400 | 183 | all ≥0.15; round-robin over images, at most 4 per image |

The high-score share was enriched against the eligible training pools: ≥0.20 was
14.7% of eligible leaves and 19.4% of eligible inflorescences. The inflorescence
image cap constrained its requested score-bin quotas; selection counts by bin
are in `selection.json`.

## Output files

- `verdicts.jsonl` — exactly one model verdict per selected proposal, with original proposal, crop metadata, score, detector/verifier provenance, policy decision, and `human_truth:false`.
- `assisted-labels.jsonl` — 580 entries; every label has `source:"model"`, `review_state:"accepted"`, observation/photo URL, exact detector `bbox_xyxy_norm`, detector score, model/prompt/policy IDs and provenance. These are assisted candidates, not curator-reviewed truth.
- `review-queue.jsonl` — 620 unaccepted items (flower 400, leaf 55, inflorescence 165), each carrying its original proposal and verifier verdict; all are `human_truth:false`.
- `raw_calls.jsonl` — all 304 raw responses, usage/cost, latency, status, retry attempts and prompt/model provenance.
- `stats.json` — class counts, selection details, costs, latency, yield estimates, and 58 audit-sample IDs.
- `crops/` — the 1,200 actual cropped JPEGs sent to the VLM.
- `selection.json`, `selected-proposals.json` — reproducible candidate selection and provenance.

## Applied policies and estimated yield

| class | outcome | support/policy basis | estimated yield |
|---|---|---|---:|
| leaf | 345 assisted; 55 review | YOLOE policy `d69da41c0ab1`, score ≥0.095076 and balanced verifier acceptance; prior train support 16/16, Wilson 95% CI 0.806–1.000 | 345 point estimate; 278–345 using that classwise interval |
| inflorescence | 235 assisted; 165 review | same YOLOE policy; no score floor, balanced verifier acceptance; prior train support 10/10, Wilson 95% CI 0.723–1.000 | 235 point estimate; 170–235 using that classwise interval |
| flower | 0 assisted; 400 review | OWLv2 policy `5ff2bc3c7008` is review-only | 0 assisted |

Combined point estimate: **580** accepted-candidate labels. Summing the two
classwise interval bounds gives 448–580 as a rough range; this is **not** a
formal combined 95% interval. No human review has yet measured the true yield of
this production batch; the estimates reuse the earlier small human-reviewed
YOLOE support (16 leaf / 10 inflorescence). Inflorescence is especially
provisional.

Curator audit sample: **58 / 580 assisted candidates (10%)**, randomly selected
with seed `anatomy4-curator-audit-v1` (37 leaf, 21 inflorescence). The flag is
`audit_sample:true`; audit results must be recorded as a later curator action.

## Runtime and spend

- 300 original batch requests + 4 single-crop retries; bounded concurrency 4, batch size 4.
- 0 failed calls after retries; 0 unresolved malformed verdicts.
- Cost $1.9941342; 14,890,134 prompt tokens and 40,809 completion tokens.
- Mean request latency 3.80 s, p95 6.19 s; elapsed pass + retry time 291.1 s.
- The first pass returned a mismatched verdict count for 4 inflorescence crops. Their original raw responses are retained; only those four crops were re-queried individually. The final verdict file contains the later valid verdicts.

## Safety and integrity checks

- 1,200 selected IDs = 1,200 verdict IDs; assisted + review queues partition all 1,200 exactly.
- 580 assisted labels are only YOLOE leaf/inflorescence; every box is the proposal's exact normalized xyxy box and within [0,1].
- OWLv2 flower never enters assisted labels. All labels and queue rows declare `human_truth:false`.
- 0 missing image/photo URLs, 0 failed calls, 0 unresolved verifier responses.
- `blind_test_touched:false`; no training, model update, or deploy was performed.