# V1 controlled fusion result

This is the first complete, pre-specified comparison on the 1,641-image
individual-disjoint split. All modes used the same seeds (17, 42, 73), training
budget, optimizer, early-stopping rule, and validation species macro-F1 model
selection. The test split was not used for checkpoint selection.

## Primary result

| Mode | Species macro-F1 (mean ± SD) | Species accuracy | Organ macro-F1 |
|---|---:|---:|---:|
| BioCLIP only | **0.8439 ± 0.0092** | **0.8641** | 0.6628 |
| BioCLIP queries EfficientNet | 0.8192 ± 0.0079 | 0.8357 | 0.6929 |
| Bidirectional, final EfficientNet map only | 0.8186 ± 0.0077 | 0.8373 | **0.7235** |
| Concatenation | 0.8146 ± 0.0093 | 0.8262 | 0.6676 |
| Bidirectional | 0.8082 ± 0.0272 | 0.8341 | 0.6930 |
| EfficientNet queries BioCLIP | 0.8044 ± 0.0245 | 0.8215 | 0.6920 |
| EfficientNet only | 0.1673 ± 0.0312 | 0.2528 | 0.6035 |

The bidirectional claim is **not demonstrated**. Against matched seeds, its
mean species macro-F1 delta was -0.0110 versus BioCLIP→EfficientNet and +0.0039
versus EfficientNet→BioCLIP. Seed 73 was unstable and reversed the apparent
advantage seen at seeds 17 and 42.

## Interpretation

EfficientNet contains useful organ information but weak species information:
its species macro-F1 was 0.1673 while its organ macro-F1 was 0.6035. Unrestricted
fusion therefore lets the weaker species representation interfere with the
strong BioCLIP representation. Removing the EfficientNet mid-level map improved
bidirectional species macro-F1 by 0.0104 and organ macro-F1 by 0.0305, which is
additional evidence that the current multiscale EfficientNet contribution is
not reliably complementary for species recognition.

The next controlled iteration should retain real cross-attention but anchor the
species prediction to the pre-fusion BioCLIP logits and learn only a gated
residual correction from fused features. That design can fall back to BioCLIP
when EfficientNet is harmful while still allowing complementary evidence to
improve individual examples.
