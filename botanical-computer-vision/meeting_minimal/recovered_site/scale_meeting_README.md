# Meeting experiments — first real runs

Preserve the existing scale analysis, NSC, strict activation preflight, and independent CPU exploration. New experiments use restored original normalized embeddings, never saved 2D points as substitutes.

## Reproduce from `botanical-computer-vision/scale_conditioned_analysis`

```sh
OPENBLAS_NUM_THREADS=2 python meeting/run_pca.py
python nsc/verify_nsc.py
python cluster_activation_mapping/exploration/verify_exploration.py
OPENBLAS_NUM_THREADS=2 python meeting/run_neo.py
OPENBLAS_NUM_THREADS=2 python meeting/build_galleries.py
OPENBLAS_NUM_THREADS=2 python meeting/run_enrichment.py
OPENBLAS_NUM_THREADS=2 python meeting/run_ae.py
python meeting/run_inversion.py --weights /absolute/path/bioclip25.safetensors --steps 12
python meeting/run_activation_extension.py --weights /absolute/path/bioclip25.safetensors --threads 4
python meeting/verify_activation_extension.py
python meeting/build_activation_report.py
python meeting/build_sections.py
python meeting/add_ae_cases.py
python meeting/build_inversion_report.py
OPENBLAS_NUM_THREADS=2 python meeting/verify_results.py
python meeting/render.py
```

Install NumPy, pandas, scipy, scikit-learn, Matplotlib, Pillow and threadpoolctl for vector-only analyses. CPU PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu, open_clip_torch 3.2.0, timm 1.0.15, safetensors 0.5.3 for frozen image inversion. This workspace uses `PYTHONPATH=/workspace/scratch/a573d6b781e4/pylibs` for these local dependencies. No API keys, paid compute or learned image generator are used.

Original vectors belong at `../outputs/{bioclip25,dinov3,efficientnet_b0}_strict_embeddings/embeddings.npz`. Source hashes, actual samples, version strings, seeds and commands are in `results/*_run.json`. Test set is never used to choose NSC threshold, AE epochs/dimensions or clustering settings. PCA/NEO/enrichment are descriptive full-corpus analyses; AE is a separate held-out comparison.

NSC is reused with archived train-only selected threshold zero and audited individual/duplicate-disjoint splits. `verify_results.py` recomputes archived zero-threshold fold metrics and test predictions from restored vectors without performing parameter selection. Threshold two remains only a training-centroid mechanism demo.

NEO implementation is a Python translation of Whang's released MATLAB data code in `vendor/neo_kmeans/neo_kmeans.m`, downloaded from https://bdi-lab.kaist.ac.kr/assets/down/neo_k_means_vector.zip. The author website states GPL distribution; preserve that license/attribution for this derivative implementation. Algorithm uses global two-stage constrained binary assignments, not top-two-center assignment or a GMM. `verify_results.py` compares one actual BioCLIP iteration to the literal repeated-minimum assignment. Source: Whang, Dhillon, Gleich, SDM 2015, pp.936–944, https://doi.org/10.1137/1.9781611974010.105.

Enrichment image composition and individual-component statistical tests have deliberately different counting units. Exact definitions and assumptions are in `results/enrichment_summary.json`. Family labels preserve the existing legacy taxonomy; no new labels enter clustering.

Archived activation maps are reused and reaudited. A separate all12 FP32 extension was actually run in this turn. All12 fixed images retain archived 3-path vector checks. A fresh single-path FP32 extension supplies real Grad-CAM and equal-area occlusion controls for all12; the archived3-case three-path maps remain unchanged. Strict original reproduction remains UNPASSED. `exploration/verify_exploration.py` checks the original page hash before adding the authorized Meeting navigation; after navigation, strict-file preservation is independently checked by `meeting/verify_delivery.py`. Neither check relaxes strict vector tolerances.

Inversion optimizes random RGB pixels through the frozen real ViT. Recomputed target and optimized image share one recorded FP32 CPU encode function; these are not exact L4 reproductions. Target preprocessing is the original224 crop; optimized64 RGB pixels are bilinearly upsampled224 before the same CLIP normalization. Two seeds per organ,12 updates; poor results are retained. Initial/intermediate/final images and every evaluated feature distance are saved. Optimizer checkpoint stores pixels after the last evaluation/update; displayed `step012` is the evaluated image at12 updates, not the extra checkpoint update. No decoder, image retrieval or generated photo is presented as reconstruction.
