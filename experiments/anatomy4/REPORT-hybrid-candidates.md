# Hybrid proposals over the 2,400-image training-candidate pool

Package: `anatomy4-hybrid-candidates`
Contract: `anatomy4-v1`
Worktree: `anatomy4-alt-detector`
Status: complete — proposals only, **no review verdicts, no training, no commits**
Host: `10.0.0.245` (RTX 4060 Ti 8 GB, BioCLIP resident), container `anatomy4-alt:0.1`
Data: `/srv/fast/anatomy4-shenandoah/train-candidates` (mounted at `/out`)

## What ran

The reviewed hybrid operating point from `REPORT-alt-detectors.md`, applied to
the candidate pool: YOLOE for `leaf`, OWLv2 for `needle`/`flower`/
`inflorescence`.

| class | backend | model | prompts | floor | image scope |
| --- | --- | --- | --- | --- | --- |
| leaf | YOLOE | `yoloe-26s-seg.pt`, imgsz 1024, conf floor 0.02 | `leaf` | 0.05 | broadleaf stratum (876) |
| needle | OWLv2 | `google/owlv2-base-patch16-ensemble` | `conifer foliage`, `spruce needles` | 0.10 | **all plant strata (1,920)** |
| flower | OWLv2 | same | `single flower` | 0.15 | flower stratum (480) |
| inflorescence | OWLv2 | same | `cluster of flowers`, `umbel of flowers` | 0.15 | inflorescence stratum (480) |

Guards: max box area 0.60, min 0.0004, cross-prompt NMS IoU 0.5, no per-image
cap (see "Decisions" below). Background (480 images) was **not** sent to any
detector — it is the pool's blind negative sample; an empty result there is by
construction, not abstention.

Prompt sets follow the benchmark: `pine needles` is dropped for OWLv2 (0/8
reviewed true positives), `flower head` is dropped from inflorescence.

## Runtime and memory

| pass | images | wall | s/img | peak GPU |
| --- | --- | --- | --- | --- |
| leaf (YOLOE) | 876 | 21.6 s | 0.025 | 342 MB |
| needle (OWLv2, 4 strata) | 1,920 | 449.3 s | 0.234 | 1,018 MB |
| flower (OWLv2) | 480 | 112.7 s | 0.235 | 1,313 MB |
| inflorescence (OWLv2) | 480 | 111.9 s | 0.233 | 1,313 MB |
| **total** | 3,756 passes | **695.4 s (11.6 min)** | — | 1,313 MB |

Peak 1,313 MB is measured in a single process that keeps the YOLOE weights
resident; the isolated benchmark peak for OWLv2 was 949 MB. Model weights are
cached at `/out/models` (`HF_HOME`); no network fetch was needed.

## Blind-test exclusion

Re-verified independently against all four held-out manifests
(`/out/blind-test/batch-{1..4}/manifest.jsonl`, 240 images):

| join key | overlap |
| --- | --- |
| observation_id | **0** |
| photo_id (incl. id parsed from photo_url) | **0** |
| photo_url | **0** |
| sha256 | **0** |

`blind-check.json` records the same result (`passed: true`). The run aborts if
any overlap is found; it did not. Blind-test images were not opened, copied, or
proposed on.

## Result counts

Proposals are **unreviewed candidates**. Merged = cross-prompt duplicates
suppressed; the surviving box records every source detection.

| class | scope | images with ≥1 | coverage | proposals | raw candidates | duplicates merged | proposals spanning >1 prompt | p50 score | p50 box area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leaf | 876 | 608 | 69.4 % | 2,831 | 2,834 | 0 | 0 | 0.091 | 0.022 |
| needle | 1,920 | 1,163 | 60.6 % | **4,056** | 6,544 | 420 | 43 | 0.132 | 0.119 |
| flower | 480 | 183 | 38.1 % | 518 | 667 | 54 | 0 | 0.259 | 0.043 |
| inflorescence | 480 | 343 | 71.5 % | 1,001 | 1,404 | 94 | 29 | 0.217 | 0.072 |
| **total** | 1,920 distinct | **1,696** | **88.3 %** | **8,406** | 11,449 | 568 | 72 | — | — |

Coverage is proposals-per-scope, not recall. Flower at 38 % is the weakest:
`single flower` at 0.15 simply abstains on 62 % of flower-stratum photos, which
is consistent with the benchmark's 78 % precision at that floor (it is a
precision-first operating point).

Rejected before merge (area guards): needle 2,058 frame-dominating + 10 tiny;
inflorescence 309; flower 95; leaf 3.

### Raw vs baseline

The earlier YOLOE-only pass on this pool produced 4,749 proposals (leaf 2,834,
needle 168, flower 508, inflorescence 1,239). The hybrid pass adds **3,888
needle proposals** (168 → 4,056) and **+10 flower / −238 inflorescence**
(different prompts and floors, so the reproductive classes are not a
like-for-like comparison). Leaf is the same pass at the new 0.60 area guard:
2,834 raw, 3 rejected, 2,831 merged.

## Conifer provenance — the direct answer

Question: *which conifer candidates came from outside the conifer stratum?*

- The manifest marks **84** images as conifer stratum. Taxonomy confirms **83**
  of them (Pinaceae 76, Taxaceae 3, Cupressaceae 4); the 84th is
  `Abies balsamea phanerolepis` (Canaan fir, development origin) with
  `taxon_family: null` — a taxonomy gap, still a conifer.
- **0** conifer-family (Pinaceae/Cupressaceae/Taxaceae) images exist in the
  broadleaf, flower, or inflorescence strata. The seed is clean: no conifer was
  sampled into a non-conifer stratum.
- Therefore **0 needle proposals come from a conifer-taxon image outside the
  conifer stratum**. There was nothing to recover.
- The cross-stratum sweep still produced plenty of needle proposals:
  broadleaf 2,029; flower 861; inflorescence 610; conifer 556 — i.e.
  **3,500 of 4,056 (86 %) needle proposals sit outside the conifer stratum**.

What those extra proposals actually are (top-scoring images per non-conifer
stratum, from `hybrid-audit/needle-cross-stratum-00.jpg`):

| score | stratum | taxon | what the box showed |
| --- | --- | --- | --- |
| 0.610 | broadleaf | *Polypodium virginianum* (Polypodiaceae) | fern frond |
| 0.574 | broadleaf | *Dryopteris* (Dryopteridaceae) | fern frond |
| 0.559 | broadleaf | *Sceptridium* (Ophioglossaceae) | fern frond |
| 0.532 | flower | *Huperzia lucidula* (Lycopodiaceae) | clubmoss |
| 0.509 | broadleaf | *Dendrolycopodium obscurum* | ground-pine |
| 0.499 | broadleaf | *Thuidium delicatulum* (Thuidiaceae) | moss |
| 0.493 | inflorescence | *Cirsium* | thistle head |
| 0.471 | broadleaf | *Climacium* (Climaciaceae) | moss |
| 0.136 | broadleaf | (no family) | yew/hemlock sapling — the one plausible conifer |

So the widening behaves exactly as the benchmark predicted: it surfaces fern /
clubmoss / moss as needle, not mis-stratified conifers. Because the proposals
are unreviewed this costs review time, not label correctness. If review volume
becomes the bottleneck, triage by taxon (skip Lycopodiaceae/Thuidiaceae/
Polypodiaceae-only images) is a *sorting* aid — it must not become a gate, for
the same reason the conifer pool is small.

## Dedupe and provenance

- NMS is per image and per class at IoU 0.5. It is never applied across
  classes: `flower` and `inflorescence` may legitimately box overlapping
  structure and merging them would hide that ambiguity.
- Every surviving proposal carries `source_proposals[]` — each raw detection's
  `proposal_id`, `prompt`, `box_score`, `detector_bbox`, `detector_rank` — plus
  `prompts[]`, `prompt_count` and `duplicate_count`. 72 proposals record
  detections from two prompts; the remaining suppressed boxes were nested
  duplicates of a single prompt (OWLv2 emits heavily nested candidates).
- Raw detector output is preserved verbatim at `hybrid-raw/<class>-<backend>/proposals.json`
  with model revision, prompt→class mapping, query order, score floor, NMS,
  `keep_duplicates: true`, torch/transformers versions and a config
  fingerprint. `hybrid-proposals.json.raw_runs` records each file's SHA-256.
- No proposal has a verdict. Every merged proposal carries
  `review_status: "unreviewed"`, `proposal_state: "proposal_only"`, and the
  top-level document states "no ground truth, no review verdicts, no
  auto-accept".

## Artifacts

On `10.0.0.245`, `/srv/fast/anatomy4-shenandoah/train-candidates/`:

```
hybrid-proposals.json        54 MB, merged manifest (8,406 proposals) + provenance
stats.json                   hybrid stats (canonical name)
hybrid-stats.json            identical copy
blind-check.json             0 overlap against 240 blind images
hybrid-raw/leaf-yoloe/proposals.json            2,834 raw
hybrid-raw/needle-owlv2/proposals.json          6,544 raw
hybrid-raw/flower-owlv2/proposals.json            667 raw
hybrid-raw/inflorescence-owlv2/proposals.json   1,404 raw
hybrid-audit/                10 small sheets (4 classes + cross-stratum needle)
baseline-yoloe-stats.json    previous YOLOE-only stats, preserved
proposals.json               previous YOLOE-only proposals, untouched
```

SHA-256 (also in `artifacts/hybrid-candidates/SHA256SUMS.txt`):

```
e9852c3e582fd97bcc0e35e19a1f39f24c4a330a4cdd66bd3425dfc9a87c6d3d  hybrid-proposals.json
db5aa2ac977e399734f5a461fa1fa5668ec532469b3406336fe96e127d1068ce  stats.json
a54b037d260dd2b30f957f3ddc92126e17b2846a4d70d7bba84fe03399bf0daf  blind-check.json
```

Local evidence copy: `experiments/anatomy4/artifacts/hybrid-candidates/`
(stats, blind check, audit index, all 10 audit sheets, checksums).

## Decisions taken (and why)

1. **Background stayed detector-blind.** "All strata" was read as all *plant*
   strata. Running a detector on the negative sample would manufacture
   proposals that later curation could turn into labels, breaking the pool's
   design. Leaf also stays on its production scope (broadleaf) — YOLOE leaf
   boxes on flower/inflorescence photos would duplicate the flower work.
2. **The benchmark's per-image cap of 10 was not applied.** It existed to bound
   review sheets. On training candidates it would silently drop the 11th shoot
   on a conifer branch (it was dropping 210 leaf and 519 needle candidates).
   NMS still merges duplicates; the cap is now `null` in `stats.guards` with
   the benchmark value recorded alongside.
3. **`stats.json` was written without clobbering the baseline.** The directory
   already had a YOLOE-only `stats.json`; it was copied to
   `baseline-yoloe-stats.json` once, and `proposals.json` (the baseline
   manifest) was left untouched.
4. **Merge is a pure reindex.** `reindex-hybrid` rebuilds
   `hybrid-proposals.json` and stats from the raw files with no inference, so
   dedupe/cap policy can be revisited without GPU time. The final artifact was
   produced this way.

## Known limits

- Precision is inherited from the benchmark (needle ~33 %, flower ~78 %,
  inflorescence ~53 %, leaf ~77 % on reviewed samples). These are unreviewed
  proposals, and 8,406 rows is a review workload, not a dataset.
- Coverage is not recall: an empty class on an image is not evidence of
  absence, and no negative labels were produced.
- The cross-stratum needle sweep adds ~3,500 mostly-fern/moss proposals; see
  the triage note above.
- No multi-photo/observation-group constraint was applied here; the pool's
  2,377 observations still need observation-level train/val splitting before
  any training run (the earlier candidate audit's requirement).
- No VLM verification, no training, no deploy — none was run.

## Reproduce

```bash
# full pass (GPU, ~12 min)
docker run --rm --network host --device nvidia.com/gpu=all -u 1002:1002 \
  -v /home/valkyrie/anatomy4-alt-detector/repo:/workspace \
  -v /srv/fast/anatomy4-shenandoah:/out \
  -e HF_HOME=/out/models -e YOLO_CONFIG_DIR=/tmp/ultra -w /workspace anatomy4-alt:0.1 \
  python -m experiments.anatomy4.cli prelabel-hybrid \
    --manifest /out/train-candidates/manifest.jsonl \
    --out /out/train-candidates \
    --blind-manifest /out/blind-test/batch-1/manifest.jsonl \
    --blind-manifest /out/blind-test/batch-2/manifest.jsonl \
    --blind-manifest /out/blind-test/batch-3/manifest.jsonl \
    --blind-manifest /out/blind-test/batch-4/manifest.jsonl \
    --device cuda

# rebuild merged manifest + stats from raw outputs (CPU, no inference)
python -m experiments.anatomy4.cli reindex-hybrid --out /out/train-candidates
```

Tests: `PYTHONPATH=. python -m pytest experiments/anatomy4/tests -q` — 28
passed (12 hybrid tests: stratum scoping incl. the background-blind refusal,
conifer detection, provenance-preserving dedupe, cross-class non-merge, cap
behaviour, blind-overlap detection/pass, conifer provenance stats, guards).
