# Plant embedding inversion MVP — completed

1,641 BioImages photos, 63 species; 17 points × 4 fixed seeds = 68 images. No training or fine-tuning.

OpenCLIP ViT-H-14 (laion2b_s32b_b79k), raw arithmetic means; StableUnCLIPImg2ImgPipeline, empty prompt, noise_level=0, 30 steps, seeds 0–3, 768px, NVIDIA L40S. Decoder mirror/revision and exact package versions are in checks.json.

The missing EfficientNet partitions were recomputed with explicit user authorization: full-corpus Euclidean KMeans, random_state=42, n_init=20. They are new descriptive partitions. K=7 has no fruit-majority group. The user selected the highest-fruit-share group: g0 (fruit 111/360=30.8%; actually twig-majority 179/360). Leaf endpoint is g1 (leaf 94/95). The strip labels this distinction.

Group-centre nearest-photo hits: 43/48 generated images. This is descriptive retrieval over all 1,641 real photos, not held-out biological accuracy. Each point has only four samples.

| Point | Group majority organ | Hits / 4 | Target 10-NN majority organ |
|---|---|---:|---|
| k5_g0 | leaf | 4/4 | leaf |
| k5_g1 | twig | 4/4 | twig |
| k5_g2 | whole tree (or vine) | 4/4 | whole tree (or vine) |
| k5_g3 | bark | 4/4 | bark |
| k5_g4 | inflorescence | 3/4 | inflorescence |
| k7_g0 | twig | 3/4 | twig |
| k7_g1 | leaf | 3/4 | leaf |
| k7_g2 | whole tree (or vine) | 4/4 | whole tree (or vine) |
| k7_g3 | leaf | 3/4 | leaf |
| k7_g4 | bark | 4/4 | bark |
| k7_g5 | inflorescence | 3/4 | inflorescence |
| k7_g6 | leaf | 4/4 | leaf |

Interior walk points have no prescribed group label; checks.json uses null for target-group share and reports both endpoint-group shares. The 10-NN organ summary is given for each target and each generated image.

Files: contact_k5.png (5 rows × 4 seeds); contact_k7.png (7 rows × 4 seeds); leaf_to_fruit.png (5 alpha positions × 4 seeds); checks.json (full audit); summary.json (compact table); targets.npz; config.json; scripts/ (runner, requirements, clustering and provenance). Full 768px samples and real OpenCLIP embeddings remain in the private Modal volume fieldguidevision-unclip-mvp/results_v2.

To reproduce from lindsey-cleanup, copy scripts/ to botanical-computer-vision/unclip_mvp/, install requirements.txt, and run run.py --stage all --fruit-fallback --decoder sd2-community/stable-diffusion-2-1-unclip --decoder-revision ac4aaff4a67d57bdab2e9faf277c33d8e49f4df4. Use a CUDA GPU; the manifest and EfficientNet order NPZ are in that branch.
