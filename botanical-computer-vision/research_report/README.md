# BioImages research report bundle

Generated 2026-09-23 from existing project artifacts only. No model inference, embedding extraction, probe training, or clustering assignment was rerun.

## Recommended advisor-meeting order

1. `index.html` — compact meeting summary, headline results, hierarchy, and confusion pairs.
2. `error_gallery.html` — complete six-bucket visual review page.
3. `representation_analysis.html` — PCA/UMAP first, followed by frozen probes, clustering, distances, and contact sheets.

Secondary traceability pages:

- `benchmark_protocol.html` — exact model, split, candidate, prompt, parsing, runtime, and comparison caveats.
- `error_hierarchical.html` — detailed hierarchical tables and aligned error counts retained for reference.

## Traceability files

- `data/experiment_registry.csv` — one row per completed experiment/analysis.
- `data/prompt_candidate_protocol_registry.csv` — VLM run/protocol registry, including missing historical fields.
- `data/openrouter_fixed_candidate_order.csv` — reconstructed fixed order from the recorded seed and runner algorithm.
- `data/vlm_prompt_template.txt` — prompt template from the current benchmark runner.
- `data/vlm_order_sensitivity_examples.csv` — real fixed-vs-per-image order prediction changes.
- `data/metric_consistency_audit.csv` — explains repeated model names with different populations/protocols.
- `data/source_inventory.txt` — scanned source artifact inventory.

## Primary source families

- Full and strict predictions: `../outputs/`
- Four-model alignment, taxonomy, gallery, annotation template: `../error_analysis/`
- Embeddings, clustering, distances, projections, sheets: `../representation_analysis/`
- VLM integrity audit: `../reports/openrouter_vlm_audit.md`
- Executable protocol definitions: `../scripts/`

## Interpretation guardrails

- BioCLIP/DINO strict error rows use 63 candidates; VLM rows use 85.
- VLM confidences are self-assessed and not calibrated.
- BioCLIP cosine-softmax and probe probabilities are also method-specific; score magnitudes should not be compared across models.
- Labels were not used by K-means; they were attached after clustering.
- PCA/UMAP are visualization only; original-space cosine analysis provides the distance evidence.
- Human failure annotation fields are currently blank. The report does not assign image-level failure causes.
