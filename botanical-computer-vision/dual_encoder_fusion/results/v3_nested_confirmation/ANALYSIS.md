# Locked final-map confirmation analysis

## Decision

The bidirectional advantage is **not demonstrated** under the locked
confirmation rule. Final-map-only bidirectional fusion had a positive point
estimate against both one-way models, but neither 95% individual-bootstrap
interval excluded zero.

| comparison (bidirectional minus comparator) | OOF macro-F1 delta | 95% interval | bootstrap P(delta > 0) |
|---|---:|---:|---:|
| BioCLIP queries EfficientNet | +0.00577 | [-0.00191, +0.01258] | 0.926 |
| EfficientNet queries BioCLIP | +0.00175 | [-0.00678, +0.00984] | 0.655 |
| BioCLIP only | -0.00025 | [-0.00890, +0.00789] | 0.492 |

The pre-registered criterion required the lower interval bound to be above zero
against both one-way directions. Both lower bounds were negative. Fusion also
did not beat BioCLIP alone.

## Out-of-fold performance

Each number below pools the test predictions from all three outer folds, then
reports the mean and standard deviation over two fixed optimization seeds.

| model | species macro-F1 | species accuracy |
|---|---:|---:|
| BioCLIP only | 0.76034 ± 0.00544 | 0.77392 ± 0.00517 |
| final-map-only bidirectional | 0.76009 ± 0.00467 | 0.77453 ± 0.00517 |
| EfficientNet queries BioCLIP | 0.75833 ± 0.00839 | 0.77209 ± 0.01034 |
| BioCLIP queries EfficientNet | 0.75432 ± 0.00581 | 0.76965 ± 0.00776 |

These absolute scores are lower than the earlier fixed-split scores because the
outer folds reserve roughly one third of the individuals for testing and a
nested set for validation. The valid comparison is between models inside this
protocol, not between the absolute V2 and V3 values.

## Why the earlier result did not confirm

On the previously inspected split, final-map-only bidirectional fusion beat
both one-way models at all three seeds. In the locked grouped confirmation:

- outer fold 0 favored both one-way directions;
- outer fold 1 favored bidirectional fusion;
- outer fold 2 showed a small bidirectional advantage over the one-way models;
- the uncertainty across whole individuals was large enough for both paired
  intervals to include zero.

The earlier result was therefore split-sensitive. The new evidence is
compatible with a small benefit, no benefit, or a small loss; it is not strong
enough to claim superiority.

## BioCLIP anchor and organ supervision

Across fold-level fits, the bidirectional model's own BioCLIP anchor averaged
0.76150 species macro-F1 while the fused output averaged 0.75956. Fusion changed
about 6.1% of species predictions with a mean gate of 0.491, but those changes
did not improve the species metric overall.

Final-map-only bidirectional fusion did have the best mean organ macro-F1 among
the four locked models (0.70695, versus 0.68930 for BioCLIP-only). This repeats
the earlier finding that EfficientNet supplies useful organ information. The
current residual fusion still does not convert that information into a
reliable improvement in 63-way species recognition.

## Protocol audit

- 1,641 images, 63 species, 465 individuals.
- Three outer folds; every individual appears in test exactly once.
- Every train, validation, and test partition contains all 63 species.
- Two matched optimization seeds per model; 24 fitted models total.
- Validation macro-F1 controls early stopping; outer test data is not used for
  model selection.
- Uncertainty uses 2,000 species-stratified bootstrap replicates resampling
  whole individuals.

The defensible final statement is: **final-map bidirectional fusion is
competitive with BioCLIP and may slightly outperform one-way fusion, but this
dataset does not demonstrate that advantage with adequate certainty.**

