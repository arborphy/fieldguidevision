# Anatomy4 — Shenandoah detector bootstrap

Reproducible, auditable bootstrap for a four-class plant-organ detector:

| class id | class | prompt family |
| --- | --- | --- |
| 0 | `leaf` | leaf / leaves |
| 1 | `needle` | needle / conifer needle / pine needle |
| 2 | `flower` | flower / blossom |
| 3 | `inflorescence` | inflorescence / flower cluster / flower head |

Contract: `anatomy4-v1` (semantic contract approved for this sprint). Classes
are *visual* categories. Nothing in this package derives a box from taxonomy.

The pipeline is explicitly staged:

1. `build-manifest` — read-only Shenandoah observations, deterministic
   stratified sample, photo metadata/licensing, SHA-256 after download.
2. `download` — prefer the .245 media vault, then the original iNat URL.
3. `prelabel` — Grounding-DINO + SAM 2.1 proposals, per-class prompts and
   thresholds, polygon **and** bbox, raw confidence, model provenance.
4. `render` — per-image overlays and per-class contact sheets.
5. Human review — `review.csv` marks proposals `tp` / `fp` / `unsure`.
6. `verify` — crop-level VLM verification of proposals (`verify-crops`), scored
   against the human review (`verify-report`) with evidence cards for every
   disagreement (`verify-cards`). See `VERIFY_EVALUATION.md`.
7. `labels` — YOLO labels from **accepted proposals only**.
8. `audit` — counts, empty-image rate, multi-organ rate, inspected precision.

No stage falls back to rectangles: a proposal without a SAM polygon is
recorded as `mask_failed` and cannot be exported as a label.

## Quick start

```bash
# On the analysis host (read-only DB access assumed).
python experiments/anatomy4/cli.py build-manifest \
  --area-instance-id inst-osm-relation-5548542 \
  --sample-size 120 \
  --out experiments/anatomy4/artifacts/shenandoah-manifest.json

# On .245, with the output dir mounted into the GPU container.
python experiments/anatomy4/cli.py download \
  --manifest /srv/fast/anatomy4-shenandoah/manifest.json \
  --images-dir /srv/fast/anatomy4-shenandoah/images \
  --vault-root /home/valkyrie/obs-media-vault/photos \
  --vault-base-url http://10.0.0.245:55434/photos

python experiments/anatomy4/cli.py prelabel \
  --manifest /srv/fast/anatomy4-shenandoah/manifest.json \
  --config experiments/anatomy4/configs/anatomy4-v1.json \
  --out /srv/fast/anatomy4-shenandoah/prelabel \
  --device cuda --resume

python experiments/anatomy4/cli.py render \
  --manifest /srv/fast/anatomy4-shenandoah/prelabel/proposals.json \
  --out /srv/fast/anatomy4-shenandoah/overlays

# After inspecting contact sheets and filling review.csv:
python experiments/anatomy4/cli.py labels \
  --manifest /srv/fast/anatomy4-shenandoah/prelabel/proposals.json \
  --review /srv/fast/anatomy4-shenandoah/overlays/review.csv \
  --out /srv/fast/anatomy4-shenandoah/yolo

python experiments/anatomy4/cli.py audit \
  --manifest /srv/fast/anatomy4-shenandoah/prelabel/proposals.json \
  --review /srv/fast/anatomy4-shenandoah/overlays/review.csv \
  --out /srv/fast/anatomy4-shenandoah/audit
```

## Alternate detectors (OWLv2 / YOLO-World v2)

Two open-vocab proposals generators were benchmarked on the same reviewed audit
sets before further manual labeling. Full write-up:
`REPORT-alt-detectors.md`. Verdict:

| class | detector to use | threshold |
| --- | --- | --- |
| `leaf` | `yoloe-26s-variants` (keep) | unchanged |
| `needle` | `owlv2-base` | 0.10 |
| `flower` | `owlv2-base` | 0.15 |
| `inflorescence` | `owlv2-base` | 0.15 |

Reviewed precision on the targeted set: OWLv2 flower 78.1 % and inflorescence
53.1 % and needle 33.3 % (correct-and-tight needle 33.3 % vs 7.7 % for YOLOE),
but leaf 45.8 % at 9.2 boxes/image, so leaf stays on YOLOE. YOLO-World v2 is
rejected: zero needle proposals in 510 images. Proposals export to
`proposals.csv` / `proposals.jsonl` with model + prompt provenance; they are
never labels.

```bash
# proposals (container anatomy4-alt:0.1 — adds scipy for OWLv2)
python -m experiments.anatomy4.cli prelabel-owlv2 \
  --manifest /out/targeted/manifest.downloaded.json \
  --config experiments/anatomy4/configs/anatomy4-alt-detectors.json \
  --out /out/alt-owlv2/targeted --device cuda

# review sheets (large tiles, 9 per sheet) and reviewed metrics
python -m experiments.anatomy4.cli render-alt-sheets \
  --manifest /out/alt-owlv2/targeted/proposals.json \
  --out artifacts/alt-review/owlv2 --per-class 9 --needle-detail
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

## Hybrid proposals on the training-candidate pool

`prelabel-hybrid` applies the reviewed per-class operating point (YOLOE leaf,
OWLv2 needle/flower/inflorescence) to a candidate pool, with the needle prompts
run on **every plant stratum** and cross-prompt duplicates merged while keeping
every source detection. Background stays detector-blind. Write-up:
`REPORT-hybrid-candidates.md`.

```bash
# full pass (GPU)
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

Outputs: `hybrid-proposals.json` (merged, `source_proposals` provenance),
`stats.json`, `blind-check.json`, `hybrid-raw/<class>-<backend>/proposals.json`,
`hybrid-audit/*.jpg`. Proposals are unreviewed — no verdicts, no auto-accept.

## Crop-level verification (VLM)

`verify.py` sends only the proposed crop's pixels (plus the detector's proposed
class) to a cheap VLM through the existing OpenRouter credential and asks for a
strict JSON verdict: `class`, `resolved_enough`, `bbox_tightness`, `reason`,
`confidence`. It never sends the taxon name, and it rejects any response that
names a taxon or that carries a box/point/polygon field.

```bash
python experiments/anatomy4/cli.py verify-crops \
  --proposals /srv/fast/anatomy4-shenandoah/prelabel-yoloe2/proposals.json \
  --images-dir experiments/anatomy4/artifacts/verify-yoloe2/images \
  --review experiments/anatomy4/artifacts/review-yoloe2/review.csv \
  --manifest experiments/anatomy4/artifacts/shenandoah-manifest.json \
  --out experiments/anatomy4/artifacts/verify-yoloe2 \
  --batch-size 3 --max-calls 45
```

Artifacts per run: `crops/` (the actual pixels sent), `verdicts.jsonl` (one
canonical verdict + decision per proposal), `raw_calls.jsonl` (raw response,
latency, token usage, cost, model/provider, prompt hash — provenance), and
`report.md` / `report.json`.

Re-scoring is free: `verify-report --raw-calls …` re-parses stored responses, so
a prompt/guard/policy change never re-bills the same crops.

Policy profiles (`accept_decision(profile=…)`): `strict` (class + resolved +
tight + confidence ≥ 0.70), `balanced` (class + resolved; tightness recorded as
a flag — **the measured recommendation**), and `triage` (reject only
`not_target`/`ambiguous`/invalid). Measured agreement and the per-class verdict
(auto-accept / review-only / unusable) are in `VERIFY_EVALUATION.md`.

### Joint detector-score × verifier policy

`policy-fit` joins the detector's own `box_score`, the verifier verdict and the
human audit, then fits a per-class rule (score floor + verifier requirements)
under a measured-precision target, a minimum-support gate and a leave-one-out
validation gate. It separates the **assisted-training** lane (train split,
provenance required) from **human-reviewed truth** (val/test, never
auto-accepted).

```bash
python experiments/anatomy4/cli.py policy-fit \
  --proposals /srv/fast/anatomy4-shenandoah/prelabel-yoloe2/proposals.json \
  --verdicts  experiments/anatomy4/artifacts/verify-yoloe2/verdicts.jsonl \
  --review    experiments/anatomy4/artifacts/review-yoloe2/review.csv \
  --out       experiments/anatomy4/artifacts/policy-yoloe2 \
  --needle-mainsubject-verdicts \
      experiments/anatomy4/artifacts/verify-yoloe2-needle-mainsubject/verdicts.jsonl
```

Outputs `policy.json` (per-class action, score floor, verifier requirements,
measured support + CIs, rejected alternatives, `policy_id`) and
`decisions.jsonl` (per-proposal action). `labels --verdicts …` consumes verifier
accepts directly; use `--require-both` to demand the human sheet as well.
Measured policy and its limits: `POLICY.md`.

### Alternate detector (OWLv2) populations

`owlv2_prep.py` consumes the alt-detector worktree's reviewed CSVs and splits
them into three populations that are never pooled: targeted class sheets (fit),
background sheets (false-accept audit), and broad class sheets (independent
validation). `policy-fit` accepts the extra populations:

```bash
python experiments/anatomy4/cli.py policy-fit \
  --proposals experiments/anatomy4/artifacts/owlv2-review/inputs/proposals-reviewed.json \
  --verdicts  experiments/anatomy4/artifacts/owlv2-review/targeted/verdicts.jsonl \
  --review    experiments/anatomy4/artifacts/owlv2-review/inputs/review-targeted.csv \
  --validation-verdicts experiments/anatomy4/artifacts/owlv2-review/broad/verdicts.jsonl \
  --validation-review   experiments/anatomy4/artifacts/owlv2-review/inputs/review-broad.csv \
  --background-verdicts experiments/anatomy4/artifacts/owlv2-review/background/verdicts.jsonl \
  --background-review   experiments/anatomy4/artifacts/owlv2-review/inputs/review-background.csv \
  --out experiments/anatomy4/artifacts/policy-owlv2 --detector owlv2-base
```

Two extra gates apply: the assisted (train) lane must itself reach the precision
target, and the independent validation must reach 0.80 when it has ≥3 accepts.
Result: no OWLv2 class reaches a validated auto-assist band; see
`OWLv2_EVALUATION.md` for the measured support and failure modes.

## Deliberate non-goals (this sprint)

- No full training run, no thousands-of-images scale-out.
- No writes to the app DB, dev.arborphy.ai, or BioCLIP/DB services.
- No taxonomy-derived or rectangle-fallback boxes.
- Raw proposals never become YOLO labels without human acceptance.
