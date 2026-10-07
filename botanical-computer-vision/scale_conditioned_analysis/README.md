# Explaining species mixing with photographic scale

This directory continues the BioImages representation analysis with one focused question:

> Are different species mixed because they look morphologically similar, or because they were photographed at the same scale and composition?

Open `index.html` for the visual explanation.

Open [`review/index.html`](review/index.html) for the 92-image human audit. It shows one large photograph at a time, keeps the four operational definitions visible, supports 1–4 keyboard shortcuts, resumes progress, hides provisional evidence until submission, and exports CSV or JSON. On Vercel it uses an anonymous Supabase reviewer identity, so no email verification is required.

## Phase 1 scope

The current page deliberately completes the visual foundation before adding more metrics or algorithms:

1. Defines distant, mid-range, close-up, and uncertain with positive visual criteria.
2. Shows six real high-confidence examples for each main class and four boundary cases.
3. Records separate BioImages and Gemma evidence for every provisional image label.
4. Creates a 92-image manual review file: 10 high-confidence and 10 medium-confidence examples per main class, plus every uncertain image.
5. Redraws the saved PCA coordinates for BioCLIP 2.5, DINOv3, and EfficientNet-B0 using photographic scale.
6. Shows each scale in the same global PCA space while retaining the existing k=20 cluster colors.
7. Uses BioCLIP cluster 04 as a concrete case study and reorganizes its image gallery by scale.

No model is retrained and no new embedding is inferred.

## Operational scale definitions

| Label | Positive criterion |
|---|---|
| `distant` | The entire plant, tree, shrub, or most of the crown is visible; overall habit and scene context are readable. |
| `mid-range` | A branch system, trunk section, or several connected organs are visible; the whole organism is outside the frame. |
| `close-up` | One organ, surface, or fine structure dominates the frame and local texture or shape is readable. |
| `uncertain` | The image contains conflicting scale evidence or no positive evidence for a stable label. |

Organ and scale remain separate. A leaf can appear as a close-up, inside a mid-range branch view, or as part of a distant crown.

## Evidence and confidence

- `high`: BioImages and Gemma provide compatible evidence.
- `medium`: one source provides positive evidence.
- `conflict`: distant and close-up evidence are both present.
- `low`: none of the three operational definitions has positive evidence.

Gemma is used as an image-grounded suggestion, not ground truth. All labels remain provisional until the review sample is completed.

## Outputs

- `data/image_visual_groups.csv`: one row per image with scale, confidence, rationale, and separate source evidence.
- `data/scale_codebook.json`: the examples rendered in the visual codebook.
- `data/scale_review_sample.csv`: the 92-image human audit sheet.
- `review/`: the public, human-friendly review application and generated browser payload.
- `data/visual_group_counts.csv`: full-corpus and embedding-subset coverage.
- `assets/representation_by_scale.png`: three frozen representations colored only by scale.
- `assets/*_scale_conditioned_clusters.png`: one scale at a time in the same global PCA space.
- `assets/bioclip_cluster_04_case.png`: one mixed cluster colored by scale and species.

## Rebuild

From `botanical-computer-vision/`:

```bash
python scale_conditioned_analysis/build_scale_conditioned_analysis.py
```

Required Python packages: pandas, NumPy, scikit-learn, and Matplotlib.

## Next phase after label review

1. Compare same-species/same-scale, same-species/different-scale, and different-species/same-scale cosine similarities.
2. Add accuracy bars by scale with real correct and incorrect images.
3. Select 3–5 validated mixed clusters for local PCA across PC1–PC2, PC1–PC3, and PC2–PC3.
4. Add rotation/projection and seed-guided nearest-neighbor views using the original frozen vectors.
