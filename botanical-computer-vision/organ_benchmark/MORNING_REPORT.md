# Morning report: BioImages organ / view benchmark

## Bottom line

Supported in this benchmark: DINOv3 is stronger than BioCLIP 2.5 on both species-disjoint canonical organ classification and cross-species organ retrieval, consistent with a more species-invariant morphology representation.

These are **BioImages canonical-label** results. A photograph can contain several reasonable organs even though BioImages supplies one canonical label.

## Headline tables

### Individual-disjoint `organ_tag`

| Model           |   Top-1 | Top-1 95% CI   |   Top-3 |   Macro F1 | Macro F1 95% CI   |   Weighted F1 |   Test n |
|:----------------|--------:|:---------------|--------:|-----------:|:------------------|--------------:|---------:|
| BioCLIP 2.5     |   0.810 | 73.7–87.6%     |   0.967 |      0.669 | 56.7–84.0%        |         0.814 |      211 |
| DINOv3          |   0.896 | 84.4–94.1%     |   0.986 |      0.734 | 64.4–91.4%        |         0.900 |      211 |
| EfficientNet-B0 |   0.872 | 81.4–92.3%     |   0.991 |      0.731 | 62.0–89.2%        |         0.874 |      211 |

### Species-disjoint `organ_tag`

| Model           |   Top-1 |   Top-3 |   Macro F1 |   Weighted F1 |   Test n |
|:----------------|--------:|--------:|-----------:|--------------:|---------:|
| BioCLIP 2.5     |   0.718 |   0.938 |      0.527 |         0.717 |      259 |
| DINOv3          |   0.873 |   0.985 |      0.882 |         0.870 |      259 |
| EfficientNet-B0 |   0.815 |   0.938 |      0.754 |         0.811 |      259 |

### Cross-species organ retrieval

| Model           |   P@1 |   P@5 |   P@10 |
|:----------------|------:|------:|-------:|
| BioCLIP 2.5     | 0.586 | 0.533 |  0.494 |
| DINOv3          | 0.843 | 0.812 |  0.796 |
| EfficientNet-B0 | 0.781 | 0.772 |  0.757 |

## What was completed

- Reused three aligned, normalized frozen embedding matrices for the same 1,641 images.
- Built deterministic individual-disjoint and species-disjoint manifests and passed hard leakage assertions.
- Used one downstream method everywhere: class-balanced logistic regression, C selected by validation macro F1, then refit on train + validation.
- Evaluated `organ_tag` and original `organ_category` with Top-1, Top-3, macro F1, weighted F1, per-class metrics, support, and confusion matrices.
- Computed individual-disjoint 95% confidence intervals with 2,000 `individual_id` bootstrap replicates.
- Computed cross-species nearest-neighbor organ P@1/P@5/P@10 for every embedded query, excluding same-individual and same-species candidates.
- Built deterministic same-query retrieval galleries, DINO/BioCLIP disagreement and high-confidence error galleries.
- Created a blank, human-only 180-image multi-label audit CSV and gallery. No VLM labels were generated.
- Counted all exact `organ_category / subview` classes across the full 1,899-image metadata corpus and classified their feasibility.

## Important limitations

- The frozen embedding universe contains 1,641 of the 1,899 images: the established strict subset of 63 species with at least three independent individuals. Exact-view support is reported on all 1,899 images, but learned-probe comparisons use the shared 1,641-image universe.
- Rare canonical labels are intrinsically unstable (for example one embedded `stem` image and two `unspecified` images). They remain visible in support files but are not used to oversell fine-grained performance.
- The canonical target is not exhaustive multi-label visual truth. Human audit is the appropriate next step for measuring label ambiguity.

## Next step

Complete the 180-image human ambiguity audit, then report both strict canonical accuracy and an ambiguity-aware acceptable-label score without changing the frozen benchmark split.
