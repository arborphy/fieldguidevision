# Taxonomy-level error analysis

This analysis re-scores the existing 211 aligned predictions only. It does not run inference and does not modify the original species-level benchmark.

## Mapping coverage

- Ground-truth species: 63
- Species appearing as ground truth or prediction: 76
- Genus mapping: first token of each validated binomial name
- Family mapping: existing `app/dashboard-data.json` metadata
- Family coverage: 76/76, with no missing values or conflicts
- BioCLIP/DINO use 63 candidate species; GPT/Gemini use 85 candidate species. The taxonomy re-scoring preserves those original predictions.

## Top-1 accuracy

| Model | Species | Genus | Family |
|---|---:|---:|---:|
| BioCLIP 2.5 zero-shot | 85.8% (181/211) | 89.6% (189/211) | 91.0% (192/211) |
| DINOv3 frozen + linear probe | 56.4% (119/211) | 61.6% (130/211) | 62.6% (132/211) |
| GPT-5.4 direct VLM | 39.3% (83/211) | 48.8% (103/211) | 56.4% (119/211) |
| Gemini 2.5 Flash Lite direct VLM | 40.3% (85/211) | 62.6% (132/211) | 66.8% (141/211) |

## Species-level error decomposition

Percentages below use each model's species-level errors as the denominator.

| Model | Species errors | Correct genus, wrong species | Wrong genus |
|---|---:|---:|---:|
| BioCLIP 2.5 zero-shot | 30 | 8 (26.7%) | 22 (73.3%) |
| DINOv3 frozen + linear probe | 92 | 11 (12.0%) | 81 (88.0%) |
| GPT-5.4 direct VLM | 128 | 20 (15.6%) | 108 (84.4%) |
| Gemini 2.5 Flash Lite direct VLM | 126 | 47 (37.3%) | 79 (62.7%) |

## Most frequent within-genus confusions

- *Quercus palustris* → *Quercus alba*: 14 prediction occurrences across 2 model(s) (11 unique image(s))
- *Quercus macrocarpa* → *Quercus alba*: 9 prediction occurrences across 3 model(s) (4 unique image(s))
- *Quercus palustris* → *Quercus rubra*: 9 prediction occurrences across 3 model(s) (8 unique image(s))
- *Quercus bicolor* → *Quercus alba*: 6 prediction occurrences across 2 model(s) (4 unique image(s))
- *Sambucus racemosa* → *Sambucus nigra*: 4 prediction occurrences across 2 model(s) (3 unique image(s))
- *Quercus rubra* → *Quercus alba*: 4 prediction occurrences across 1 model(s) (4 unique image(s))
- *Acer rubrum* → *Acer saccharum*: 3 prediction occurrences across 2 model(s) (2 unique image(s))
- *Populus deltoides* → *Populus tremuloides*: 3 prediction occurrences across 2 model(s) (2 unique image(s))
- *Fraxinus pennsylvanica* → *Fraxinus americana*: 3 prediction occurrences across 1 model(s) (3 unique image(s))
- *Aesculus glabra* → *Aesculus hippocastanum*: 2 prediction occurrences across 2 model(s) (1 unique image(s))
- *Betula lenta* → *Betula alleghaniensis*: 2 prediction occurrences across 2 model(s) (2 unique image(s))
- *Picea rubens* → *Picea glauca*: 2 prediction occurrences across 2 model(s) (1 unique image(s))

## Output files

- `taxonomy_mapping.csv`: species-to-genus-to-family mapping and where each species appears.
- `taxonomy_predictions_image.csv`: image-level ground truth and four predictions at species, genus and family levels.
- `taxonomy_level_accuracy.csv`: Top-1 accuracy for every model and taxonomy level.
- `species_error_taxonomy_breakdown.csv`: correct-genus/wrong-species, wrong-genus and family-level decomposition.
- `within_genus_confusions.csv`: directed within-genus confusion pairs aggregated across models, with per-model counts.
