# Full-corpus multi-label tagging report

All 1,899 BioImages photographs were tagged by three frozen representations. Predictions are out-of-fold by `individual_id`; each fold learns from Gemma multi-label references only on its training individuals, and no visual backbone was fine-tuned.

| Model | BioImages match | Top-1 exact | Gemma micro F1 | Mean Jaccard | Tags/image |
|---|---:|---:|---:|---:|---:|
| DINOv3 | 91.4% | 48.7% | 78.9% | 68.7% | 4.75 |
| BioCLIP 2.5 | 82.3% | 25.6% | 69.3% | 56.6% | 4.82 |
| EfficientNet-B0 | 88.3% | 42.9% | 74.2% | 63.2% | 4.78 |

## Interpretation notes

- BioImages match asks whether its one canonical coarse structure appears anywhere in a model's multi-label set; it is not exhaustive visual ground truth.
- Gemma is a model-generated teacher/reference, not human ground truth. Its free tags are preserved and its normalized tags supervise training folds and score only held-out folds.
- BioImages annotations define the vocabulary mapping and independent canonical-match benchmark; they do not supervise the multi-label probes.
- The two BioImages records whose coarse tag is `unspecified` remain in the gallery but are excluded from the BioImages match denominator.

## Outputs

- `predictions/`: one row per image for each frozen visual model, including every tag confidence
- `gemma_reference.csv` and `.jsonl`: free and normalized Gemma tags for every image
- `gemma_prompt.txt` and `gemma_run_summary.json`: exact prompt and full-run provenance
- `per_image_comparison.csv`: references, agreement scores, and gallery categories
- `metrics/`: summary, tag-level, leakage-audit, disagreement, and gallery-count files
- `analysis_data.json`: compact data embedded into the static browser
