# BioImages organ benchmark

Static, reproducible comparison of BioCLIP 2.5, DINOv3, and EfficientNet-B0 frozen representations for BioImages canonical organ/view recognition.

## Open the report

Open `index.html` directly or publish this directory with GitHub Pages. The galleries use the source BioImages thumbnail URLs; no image dataset is committed.

## Rebuild

```bash
uv run scripts/run_organ_benchmark.py
uv run scripts/verify_outputs.py
```

Inputs are the existing aligned embedding files under `../outputs/*_strict_embeddings/` and `../work/bioimages_metadata.json`. All generated numbers are preserved under `metrics/`, predictions under `predictions/`, and deterministic split definitions under `split_manifests/`.

The primary metric is **BioImages canonical-label accuracy**. It is not presented as unique visual truth; the blank audit in `ambiguity_audit/` exists to capture reasonable multi-label alternatives later by hand.
