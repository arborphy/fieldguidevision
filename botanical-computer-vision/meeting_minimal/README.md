# Minimal meeting experiments (2026-10-30)

One runnable minimal result for each of the seven meeting methods. Open `index.html` locally
(thumbnails load from bioimages.vanderbilt.edu). Each section states: question, what was run,
result, what it supports — and is badged *verified* / *exploratory* / *blocked* / *recovered*.

- Input: `data/embeddings/efficientnet_b0_strict/embeddings.npz` (SHA-256 checked in `common.py`),
  `organ_benchmark/split_manifests/individual_disjoint.csv` (image_id + split checked row by row),
  `scale_conditioned_analysis/data/image_visual_groups.csv` (shot_scale),
  existing cluster IDs in `representation_analysis/data/`.
- BioCLIP / DINOv3 raw vectors are not in Git: only their existing K=20 cluster IDs are used (exp 2).
  Once the raw embeddings are uploaded, point `common.py` at them and rerun.
- `recovered_site/`: the hosted `scale/meeting/` page (source of the AE table in exp 5).

```bash
bash fetch_images.sh   # 1,641 originals + thumbnails -> /home/user/imgcache (not committed)
bash run_all.sh        # exp1..exp7 (CPU, a few minutes) + index.html
```
