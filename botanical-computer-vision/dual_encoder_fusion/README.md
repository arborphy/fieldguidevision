# EfficientNet + BioCLIP controlled feature fusion

This experiment tests whether two frozen visual encoders contribute complementary
information for 63-way species recognition. It uses real spatial tokens and
cross-attention rather than concatenating two pooled vectors and calling the
operation attention.

## Model

- **EfficientNet-B0:** the 14×14 mid-level map is pooled to 7×7 and aligned with
  the final 7×7 map. Separate learned projections preserve the two scales.
- **BioCLIP 2.5 ViT-H/14:** the final 16×16 patch grid is pooled to 8×8; its CLS
  token is retained separately.
- **Fusion:** two 4-head cross-attention blocks at width 256. Bidirectional blocks
  calculate both updates from the same pre-block state. LayerScale starts at
  `1e-3`, so fusion begins near the two frozen representations instead of
  immediately overwriting them.
- **Heads:** species and organ remain separate. V2 anchors species logits to
  the pre-fusion BioCLIP representation and adds a small gated residual from
  the cross-attended representation. The residual head starts near zero and
  the gate starts at 0.076 with a `0.01 × mean(gate)` penalty, so harmful
  EfficientNet information can be rejected. Organ retains a direct fused head.
  Each original branch also has auxiliary species and organ heads.

The V2 species logits and objective are

```text
species_logits = BioCLIP_anchor_logits + gate × fused_residual_logits

CE(species_logits)
+ 0.20 × mean(CE(species_EfficientNet), CE(species_BioCLIP))
+ 0.25 × [CE(organ_fused) + 0.20 × mean(CE(organ_EfficientNet), CE(organ_BioCLIP))]
+ 0.01 × mean(gate)
```

Species macro F1 is the only early-stopping and model-selection metric. Organ is
therefore useful supervision but cannot win model selection by learning a small
set of organ classes. `unspecified` organ labels are masked; the single `stem`
example is normalized to `twig`; all other canonical organ tags remain distinct.

## Controlled comparisons

The full run uses seeds 17, 42, and 73 with the same strict individual-disjoint
split, optimizer, batch size, epoch budget, and early-stopping rule:

1. EfficientNet only
2. BioCLIP only
3. pooled-token concatenation
4. BioCLIP queries EfficientNet (one way)
5. EfficientNet queries BioCLIP (one way)
6. bidirectional cross-attention
7. bidirectional cross-attention without the EfficientNet mid-level map

The claim “bidirectional is better than one-way” passes only when mean test
species macro F1 is higher than both one-way variants across matched seeds. Raw
per-seed results and paired deltas are written to the results directory.

## Run

The first command validates extraction, both directions, and the end-to-end
training path. The second runs the complete experiment. Frozen spatial tokens
are cached in a persistent Modal volume after the first extraction.

```bash
cd botanical-computer-vision
uv run modal run dual_encoder_fusion/modal_run_experiment.py --smoke
uv run modal run dual_encoder_fusion/modal_run_experiment.py
```

If the Modal API is unavailable, the repository also contains a manually
dispatched `Run dual-encoder GPU experiment` GitHub Actions workflow. It builds
the same experiment into a temporary private Cloud Run L4 service protected by
both Google identity and a fresh random request token, downloads the result
bundle as a workflow artifact, and deletes the service in the final step so
that the GPU does not remain allocated.

Local shape/gradient tests:

```bash
uv run --with-requirements dual_encoder_fusion/requirements-dev.txt \
  pytest -q dual_encoder_fusion/tests
```

## Output contract

- `summary.csv`: mean and standard deviation by fusion mode
- `paired_deltas.csv`: seed-matched bidirectional-minus-one-way differences
- `predictions.csv`: test predictions for reproducible error analysis
- `species_macro_f1.png`: visual comparison with seed variability
- `verdict.json`: an explicit pass/not-demonstrated decision
- `REPORT.md`: short protocol and outcome summary

## Completed V1 result

The first full controlled run is archived in `results/v1_controlled`. Its
pre-specified bidirectional claim was **not demonstrated**: bidirectional
species macro-F1 was `0.8082 ± 0.0272`, compared with `0.8192 ± 0.0079` for
BioCLIP→EfficientNet and `0.8044 ± 0.0245` for EfficientNet→BioCLIP. BioCLIP
alone was strongest at `0.8439 ± 0.0092`. See the archived `ANALYSIS.md` for the
objective interpretation and the negative-transfer evidence motivating V2.

## Completed V2 result

The BioCLIP-anchored gated-residual follow-up is archived in
`results/v2_gated_residual`. It stabilized the full bidirectional model and
raised its species macro-F1 to `0.8321 ± 0.0062`, but the primary superiority
claim remained **not demonstrated**: the two one-way variants scored `0.8383 ±
0.0045` and `0.8371 ± 0.0076`, and BioCLIP alone remained strongest at `0.8466
± 0.0070`.

The final-map-only bidirectional ablation is the useful secondary result. It
scored `0.8460 ± 0.0060` and beat both one-way variants at every matched seed,
but did not beat BioCLIP alone. See its archived `ANALYSIS.md` for the paired
deltas, fusion-versus-anchor comparison, and interpretation limits.

## Locked confirmation protocol

The follow-up confirmation freezes the final-map-only architecture before
evaluation and replaces the previously inspected split with three nested,
species-complete outer folds. Individuals never cross train, validation, and
test within a fold; every one of the 465 individuals appears in an outer test
fold exactly once. Two fixed optimization seeds are run for BioCLIP-only, both
one-way directions, and final-map-only bidirectional fusion (24 fits total).

The primary decision uses pooled out-of-fold species macro-F1. Uncertainty is
estimated with 2,000 species-stratified bootstrap replicates that resample
whole individuals rather than treating correlated images as independent.
Bidirectionality is supported only if the 95% interval for the paired
bidirectional-minus-one-way difference is above zero for both directions.
Whether fusion beats BioCLIP alone is evaluated and reported separately.

```bash
cd botanical-computer-vision
uv run modal run dual_encoder_fusion/modal_run_experiment.py --confirmation
```

### Confirmation outcome

The completed run is archived in `results/v3_nested_confirmation`. The locked
claim was **not demonstrated**. Final-map bidirectional fusion scored `0.76009
± 0.00467` out-of-fold macro-F1, versus `0.75432 ± 0.00581` and `0.75833 ±
0.00839` for the one-way models. However, the paired individual-bootstrap 95%
intervals were `[-0.00191, +0.01258]` and `[-0.00678, +0.00984]`, so both
included zero. BioCLIP-only scored `0.76034 ± 0.00544`; fusion did not improve
on it. See the archived `ANALYSIS.md` and `paired_oof_deltas.png`.
