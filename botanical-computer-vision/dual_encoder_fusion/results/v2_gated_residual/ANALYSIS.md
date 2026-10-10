# V2 gated-residual result analysis

## Bottom line

The primary claim is **not demonstrated**. Under the pre-specified rule, the
full bidirectional model had to exceed both one-way cross-attention models in
mean test species macro-F1 over matched seeds. It did not:

| model | species macro-F1 (mean ± SD) |
|---|---:|
| BioCLIP only | 0.8466 ± 0.0070 |
| bidirectional, final maps only | 0.8460 ± 0.0060 |
| BioCLIP queries EfficientNet | 0.8383 ± 0.0045 |
| EfficientNet queries BioCLIP | 0.8371 ± 0.0076 |
| concatenation | 0.8333 ± 0.0098 |
| full bidirectional | 0.8321 ± 0.0062 |
| EfficientNet only | 0.1819 ± 0.0050 |

For the full bidirectional model, the matched-seed mean differences were
`-0.0062` against BioCLIP→EfficientNet and `-0.0050` against
EfficientNet→BioCLIP. The full per-seed values are in `paired_deltas.csv`.

## What V2 changed

V2 did not replace attention with a gate. It retained real cross-attention but
anchored the species logits to the pre-fusion BioCLIP representation:

```text
species_logits = BioCLIP_anchor_logits + gate × fused_residual_logits
```

The residual classifier starts near zero, the gate starts near 0.076, and a
small gate-use penalty lets the model reject harmful EfficientNet information.
Species and organ keep separate heads and losses.

Compared descriptively with V1 under the same data split, seeds, and training
budget, V2 raised full bidirectional macro-F1 from 0.8082 to 0.8321 and reduced
its standard deviation from 0.0272 to 0.0062. The other fusion modes also
became more stable. This is evidence that anchoring controlled the severe V1
failure, not evidence that fusion surpassed the strong BioCLIP baseline.

## Direct negative-transfer evidence

Each fused run also reports the score of its own simultaneously trained
BioCLIP anchor before adding the gated residual. For the full bidirectional
model, the anchor averaged 0.8451 macro-F1 while the final fused prediction
averaged 0.8321, a change of `-0.0130`. The final prediction differed from its
anchor on only 3.63% of test images on average, with a mean gate of 0.286.
Those selective changes were therefore harmful overall, especially under a
macro metric sensitive to minority species.

The same fused-minus-anchor comparison was negative for every fusion mode:

| model | fused minus own BioCLIP anchor |
|---|---:|
| BioCLIP queries EfficientNet | -0.0084 |
| EfficientNet queries BioCLIP | -0.0123 |
| full bidirectional | -0.0130 |
| bidirectional, final maps only | -0.0133 |
| concatenation | -0.0157 |

## Important secondary result

The `bidirectional_final_only` ablation excludes the EfficientNet mid-level
map but retains bidirectional cross-attention over both encoders' final maps.
It exceeded both one-way models on every matched seed:

| competitor | seed 17 | seed 42 | seed 73 | mean delta |
|---|---:|---:|---:|---:|
| BioCLIP queries EfficientNet | +0.0039 | +0.0009 | +0.0181 | +0.0077 |
| EfficientNet queries BioCLIP | +0.0152 | +0.0055 | +0.0060 | +0.0089 |

This supports a narrower hypothesis: when fusion is restricted to the final
feature maps, bidirectionality is better than either one-way direction in this
experiment. It does **not** establish that dual-encoder fusion is better than
BioCLIP alone: final-only bidirectional was 0.0006 below BioCLIP-only on mean
macro-F1 and won only one of three seed-level comparisons.

## Species versus organ supervision

EfficientNet-only was weak for species (0.1819 macro-F1) but much stronger for
organ (0.6938). Concatenation produced the best organ macro-F1 (0.7099), while
BioCLIP-only reached 0.6405. EfficientNet therefore contains complementary
organ information, but the current species residual does not convert that
information into improved 63-way species decisions.

## Interpretation boundary

V2 was designed after inspecting V1 and reuses the same held-out split. It is
an exploratory follow-up, not an independent confirmatory test. A defensible
next experiment would lock the final-map-only architecture and evaluate it on
a fresh individual-disjoint holdout or nested cross-validation. The result
files here should not be presented as proof that fusion beats BioCLIP.

