# BioImages four-baseline error analysis

This directory reorganizes existing predictions only. It does not run new inference and does not assign image failure reasons automatically.

## Scope and protocol

- Common analysis set: **211 strict test images**, **63 observations/individuals**, **63 species**.
- An observation contains 1–16 images; both image-level and observation-level tables are retained.
- BioCLIP and DINOv3 use the strict 63-species candidate set. GPT-5.4 and Gemini use the 85-species full candidate set. Accuracy and error overlap are useful, but this is not a perfectly identical candidate-protocol leaderboard.
- Exact models: BioCLIP `hf-hub:imageomics/bioclip-2.5-vith14`; DINOv3 `facebook/dinov3-vitb16-pretrain-lvd1689m`; GPT `openai/gpt-5.4`; Gemini `google/gemini-2.5-flash-lite`.
- DINOv3 uses a frozen 768-dimensional CLS embedding and multinomial logistic regression (`C=100`) trained on 1,181 images and selected on 249 validation images. Its 211-image test set is individual/observation-disjoint from training and validation.
- BioCLIP score is a closed-set similarity softmax; DINOv3 score is classifier probability; GPT/Gemini scores are self-reported. Do not compare score magnitudes across model families.
- `observation_predictions.csv` uses majority vote across image-level Top-1 predictions. Ties are broken by mean score among images voting for each tied species.

## Image-level results

| Model | Top-1 | Top-5 | Observation majority Top-1 |
|---|---:|---:|---:|
| BioCLIP zero-shot | 85.8% (181/211) | 95.3% | 90.5% |
| DINOv3 + linear probe | 56.4% (119/211) | 83.4% | 63.5% |
| GPT-5.4 | 39.3% (83/211) | 64.9% | 44.4% |
| Gemini Flash Lite | 40.3% (85/211) | 65.4% | 41.3% |

At least one model is correct on **194/211 (91.9%)** images; all four are wrong on **17/211 (8.1%)**. VLMs rescue 3 images missed by both embedding methods. Embedding methods rescue 81 images missed by both VLMs. The most similar correctness sets are `bioclip` and `dinov3` (Jaccard 0.57).

## Key buckets

- `embedding_disagreement`: 101 images, 46 observations, 46 species.
- `all_four_correct`: 43 images, 18 observations, 18 species.
- `bioclip_only_correct`: 40 images, 21 observations, 21 species.
- `bioclip_and_dino_wrong`: 20 images, 11 observations, 11 species.
- `all_four_wrong`: 17 images, 10 observations, 10 species.
- `vlm_embedding_consensus_conflict`: 10 images, 8 observations, 8 species.
- `bioclip_wrong_vlm_rescue`: 4 images, 3 observations, 3 species.

The highest-priority human review buckets are:

1. `all_four_wrong`: likely shared visual/data difficulty and the best place to define the failure taxonomy.
2. `bioclip_wrong_vlm_rescue`: tests genuine general-VLM complementarity.
3. `vlm_embedding_consensus_conflict`: strongest representation-family disagreement.
4. `bioclip_only_correct`: shows where biological pretraining contributes unique signal.

## Species concentrated in shared failures

- *Acer rubrum*: 3/3 images were missed by all four.
- *Nyssa sylvatica*: 2/5 images were missed by all four.
- *Fraxinus americana*: 2/4 images were missed by all four.
- *Quercus bicolor*: 2/4 images were missed by all four.
- *Quercus rubra*: 2/4 images were missed by all four.
- *Betula lenta*: 2/3 images were missed by all four.
- *Quercus alba*: 1/7 images were missed by all four.
- *Ulmus americana*: 1/4 images were missed by all four.

## Repeated directed confusion pairs shared by models

- *Quercus macrocarpa* → *Quercus alba*: 9 occurrences across 3 models (dinov3|gemini_flash_lite|gpt_5_4).
- *Quercus palustris* → *Quercus rubra*: 9 occurrences across 3 models (dinov3|gemini_flash_lite|gpt_5_4).
- *Fraxinus americana* → *Quercus alba*: 6 occurrences across 3 models (dinov3|gemini_flash_lite|gpt_5_4).
- *Betula lenta* → *Prunus serotina*: 3 occurrences across 3 models (bioclip|dinov3|gpt_5_4).
- *Dirca palustris* → *Lindera benzoin*: 3 occurrences across 3 models (bioclip|gemini_flash_lite|gpt_5_4).
- *Quercus palustris* → *Quercus alba*: 14 occurrences across 2 models (gemini_flash_lite|gpt_5_4).
- *Quercus bicolor* → *Quercus alba*: 6 occurrences across 2 models (gemini_flash_lite|gpt_5_4).
- *Ulmus americana* → *Ostrya virginiana*: 5 occurrences across 2 models (gemini_flash_lite|gpt_5_4).

See `data/confusion_pairs.csv` for per-model pairs and `data/shared_confusion_pairs.csv` for pairs repeated across models.

## Files

- `error_gallery.html`: self-contained visual gallery for the three priority buckets; all images are embedded.
- `build_error_gallery.py`: reproducible gallery builder using only the existing bucket CSVs and image URLs.
- `data/unified_predictions_image.csv`: one row per image with all four predictions, scores, Top-5 lists, correctness, observation metadata and URLs.
- `data/observation_predictions.csv`: one row per observation with image lists, organ lists, majority-vote predictions and image-level consistency measures.
- `data/model_summary.csv`: image-level and observation-level headline metrics.
- `data/correctness_patterns.csv`: all observed four-bit correctness patterns in model order BioCLIP, DINOv3, GPT, Gemini.
- `data/pairwise_overlap.csv`: pairwise correct-set overlap and prediction agreement.
- `data/per_species_accuracy.csv`: four-model accuracy by species.
- `data/per_organ_accuracy.csv`: four-model accuracy and all-four-wrong rate by organ.
- `data/confusion_pairs.csv`: directed ground-truth → prediction error pairs per model.
- `data/shared_confusion_pairs.csv`: identical directed error pairs shared across models.
- `data/bucket_summary.csv`: size and coverage of each requested bucket.
- `buckets/*.csv`: complete image rows for each (overlapping) bucket.
- `samples/representative_samples.csv`: deterministic random samples, normally at most one image per observation before reuse.
- `samples/images/`: locally downloaded full-resolution sample images.
- `annotation/human_error_annotation.csv`: **60 unique images** ready for manual annotation; requested human fields are blank.
- `annotation/annotation_codebook.csv`: allowed values and definitions for the blank human fields. The failure taxonomy remains intentionally open.
- `summary.json`: machine-readable counts and headline findings.
- `build_error_analysis.py`: reproducible builder. Run `python3 error_analysis/build_error_analysis.py` from the repository root.
