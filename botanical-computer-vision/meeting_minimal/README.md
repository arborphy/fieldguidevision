# Minimal meeting experiments (2026-10-30)

One runnable minimal result for each of the seven meeting methods. Open `index.html` locally
(thumbnails load from bioimages.vanderbilt.edu). Each section states: question, what was run,
result, what it supports — and is badged *verified* / *exploratory* / *blocked* / *recovered*.

- Input: `data/embeddings/efficientnet_b0_strict/embeddings.npz` (SHA-256 checked in `common.py`),
  `organ_benchmark/split_manifests/individual_disjoint.csv` (image_id + split checked row by row),
  `scale_conditioned_analysis/data/image_visual_groups.csv` (shot_scale),
  existing cluster IDs in `representation_analysis/data/`.
- Items 1–7 run on EfficientNet. BioCLIP 2.5 / DINOv3 raw vectors were recovered from Drive afterwards
  (`data/embeddings/bioclip25_strict/`, `data/embeddings/dinov3_strict/`, SHA-256 in `common.py` and DATA_CATALOG);
  `run_models.sh` reruns the vector-only items (1, 4, 5, 6, 7) on them into `results/<model>/` (report appendix).
  Their PCA-32 test MSE reproduces the hosted AE table exactly (0.000284 / 0.000498). Item 3 still needs their weights.
- `index.html` (Chinese) and `index_en.html` (English) are built by `build_report.py` / `build_report_en.py`;
  `build_artifact.py` makes a self-contained copy (images inlined) for online publishing.
- `recovered_site/`: the hosted `scale/meeting/` page (source of the AE table in exp 5).

```bash
bash fetch_images.sh   # 1,641 originals + thumbnails -> /home/user/imgcache (not committed)
bash run_all.sh        # exp1..exp7 on EfficientNet (CPU, a few minutes) + both reports
bash run_models.sh     # vector-only items on BioCLIP / DINOv3 + both reports
```
