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
- **Heads:** species and organ have separate fused heads. Each original branch
  also has auxiliary species and organ heads.

The objective is

```text
species_fused
+ 0.20 × mean(species_EfficientNet, species_BioCLIP)
+ 0.25 × [organ_fused + 0.20 × mean(organ_EfficientNet, organ_BioCLIP)]
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
