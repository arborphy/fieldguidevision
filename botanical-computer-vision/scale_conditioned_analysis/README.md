# Explaining species mixing with photographic scale

This directory continues the BioImages representation analysis with one focused question:

> Are different species mixed because they look morphologically similar, or because they were photographed at the same scale and composition?

Open `index.html` for the complete visual analysis.

Open [`review/index.html`](review/index.html) for the 182-image human audit. It shows one large photograph at a time, keeps the four operational definitions visible, supports 1–4 keyboard shortcuts, resumes progress, hides provisional evidence until submission, and exports CSV or JSON. On Vercel it uses an anonymous Supabase reviewer identity, so no email verification is required.

## Implemented analysis sequence

The page follows the agreed conditional sequence instead of mixing every factor at once:

1. Defines distant, mid-range, close-up, and uncertain with positive visual criteria.
2. Redraws the three global representations by scale and provides an interactive scale-only cluster explorer with species highlighting and image inspection.
3. Compares species purity and NMI before conditioning and within each scale.
4. Draws Top-1 accuracy by scale with `correct / total` labels and real correct/error examples.
5. Calculates and plots species, organ, and scale composition for all 60 model-cluster combinations.
6. Reorganizes high-priority mixed-cluster galleries into distant, mid-range, and close-up rows.
7. Only after scale is separated, compares leaf-on/off, reproductive-visible/not-visible, and environmental/isolated groups.
8. Uses the restored original vectors to recompute local PCA, rotating projections, pairwise cosine comparisons, and Top-100 seed neighborhoods.

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
- `data/scale_review_sample.csv`: the 182-image human audit sheet.
- `review/`: the public, human-friendly review application and generated browser payload.
- `data/visual_group_counts.csv`: full-corpus and embedding-subset coverage.
- `assets/representation_by_scale.png`: three frozen representations colored only by scale.
- `assets/*_scale_conditioned_clusters.png`: one scale at a time in the same global PCA space.
- `assets/species_separation_by_scale.png`: species purity and NMI before/after scale conditioning.
- `assets/accuracy_by_scale.png`: strict-test Top-1 accuracy with `correct / total` labels.
- `assets/secondary_explanatory_variables.png`: residual comparisons for the second-layer visual variables.
- `assets/*_cluster_composition.png`: scale, organ, and ranked species composition for every k=20 cluster.
- `data/separation_metrics.csv`, `data/accuracy_by_visual_group.csv`, and `data/cluster_visual_composition.csv`: exact values behind the figures.
- `assets/pairwise_similarity_by_species_and_scale.png`: the four species/scale pair types with bootstrap confidence intervals.
- `data/scale_pairwise_similarity.csv`: exact cosine-similarity values and pair counts behind that figure.
- `data/analysis-payload.js`: interactive points, galleries, examples, five local PCA clusters and ten Top-100 seed queries per representation.

## Rebuild

From `botanical-computer-vision/`:

```bash
python scale_conditioned_analysis/build_complete_analysis.py
python scale_conditioned_analysis/verify_complete_analysis.py
```

Required Python packages: pandas, NumPy, scikit-learn, and Matplotlib.

## Original-vector source

The repository intentionally ignores `outputs/`, so the original vectors are not published with the site. To reproduce the committed deep-exploration outputs, restore these files before running the builder:

- `outputs/bioclip25_strict_embeddings/embeddings.npz`
- `outputs/dinov3_strict_embeddings/embeddings.npz`
- `outputs/efficientnet_b0_strict_embeddings/embeddings.npz`

The current committed page was built with all three files present. If they are absent on a future rebuild, the builder leaves pairwise cosine, local PCA/rotation, and Top-100 neighbors explicitly pending. It never uses saved two-dimensional coordinates as a misleading substitute.
