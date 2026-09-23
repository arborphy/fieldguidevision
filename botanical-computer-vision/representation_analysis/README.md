# BioImages representation analysis

This directory analyzes saved frozen BioCLIP and DINOv3 embeddings. It does not train a model, rerun inference, name clusters, or generate morphology explanations.

The final three-representation robustness analysis is reported in `efficientnet_control.html`. It compares BioCLIP, DINOv3 and EfficientNet-B0 across k=10/20/50, 10 prespecified seeds, original-space cosine distributions and matched PCA/UMAP views. The earlier EfficientNet-only report generator now writes `efficientnet_probe_control.html` so it cannot overwrite the final report.

## Protocol

- Embedding universe: 1,641 images from the strict individual-disjoint split.
- Test error overlay: the existing 211-image four-baseline prediction table.
- K-means: k=10, 20, 50; labels excluded; random seed 42; n_init=20.
- Similarity: cosine similarity on normalized embeddings.
- Nearest-neighbor galleries exclude all images from the query observation.
- Error enrichment: post-hoc hypergeometric tests with BH correction, plus explicitly labeled descriptive thresholds.
- PCA and UMAP: global 1,641-image projections, used only for visualization.

Mean BioCLIP/DINO cross-observation neighbor overlap@10: 0.2839.

Image download failures while building contact sheets: 0.

## Outputs

- `report.html`: meeting-ready static report.
- `efficientnet_control.html`: final three-representation multi-seed, distance and projection report.
- `data/organ_normalization_mapping.csv`: auditable raw-to-normalized organ mapping.
- `data/organ_view_metadata_normalized.csv`: image-level raw organ/detail/subview fields plus `normalized_organ`.
- `data/multiseed_cluster_runs.csv` and `data/multiseed_cluster_summary.csv`: all 90 K-means runs and mean/std/stability summaries.
- `data/distance_analysis_summary.csv` and `data/distance_similarity_histograms.csv`: matched cross-observation cosine analysis.
- `data/reference_k20_cluster_summary_normalized.csv`: normalized metadata summaries for the retained qualitative atlases.
- `assets/final_plots/`: nine matched PCA/UMAP views and the original-space similarity distribution plot.
- `data/frozen_probe_comparison.csv`: BioCLIP, DINOv3 and EfficientNet frozen linear-probe species/genus/family results.
- `data/efficientnet_k20_cluster_summary.csv` and `data/efficientnet_k20_cluster_assignments.csv`: EfficientNet k=20 composition and assignments.
- `data/three_representation_k20_summary.csv` and `data/three_representation_k20_pairwise_ari.csv`: aligned post-hoc structure metrics for all three spaces.
- `data/cluster_summary.csv`: cluster composition, purity, organ distribution, four-baseline test accuracy and enrichment statistics.
- `data/cluster_assignments.csv`: every image assignment and centroid distance for both spaces and all k values.
- `data/nearest_neighbors.csv`: all and cross-observation Top-10 neighbors for every test image in both spaces.
- `data/confusion_pair_metrics.csv`: similarity and neighbor metrics for the repeated confusion pairs.
- `data/structure_summary.csv`: post-hoc species/organ association and cross-space cluster agreement.
- `assets/cluster_sheets/`: all 160 centroid-nearest contact sheets.
- `assets/nn_galleries/`: BioCLIP and DINO galleries for all 17 all-four-wrong queries.
- `assets/pair_sheets/` and `assets/plots/`: confusion-pair representative images and global PCA/UMAP panels.

The persisted source vectors are in `outputs/bioclip25_strict_embeddings/embeddings.npz` and `outputs/dinov3_strict_embeddings/embeddings.npz`.
