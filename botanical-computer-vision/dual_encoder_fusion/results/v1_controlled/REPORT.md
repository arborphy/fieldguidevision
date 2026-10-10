# Dual-encoder cross-attention experiment

- Status: **not_demonstrated**
- Images: 1641 (individual-disjoint train/validation/test)
- Primary selection metric: validation species macro F1
- Organ supervision: separate auxiliary head; weight 0.25 relative to species
- Encoders: frozen for this controlled phase; projections, attention blocks, and heads are trained

## Interpretation rule

Bidirectional fusion is considered supported only if its mean test species macro F1 is higher than both one-way variants under the same seeds and training budget. This run does not claim support when that criterion fails.

See `summary.csv`, `paired_deltas.csv`, and `species_macro_f1.png` for the numerical and visual comparison.