# Final-map bidirectional confirmation

- Status: **not_demonstrated**
- Images: 1641 (strictly individual-disjoint within every fold)
- Primary selection metric: validation species macro F1
- Organ supervision: separate auxiliary head; weight 0.25 relative to species
- Species fusion: BioCLIP-anchored gated residual after real cross-attention
- Encoders: frozen for this controlled phase; projections, attention blocks, and heads are trained

## Interpretation rule

The locked final-map-only bidirectional model is supported over one-way fusion only when the lower bound of its 95% species-stratified, individual-level bootstrap interval is above zero against both one-way directions. Beating BioCLIP alone is reported as a separate claim.

See `oof_summary.csv`, `bootstrap_intervals.csv`, `species_macro_f1.png`, and `paired_oof_deltas.png`.