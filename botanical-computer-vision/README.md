# FieldGuideVision — botanical computer vision

What do frozen vision models (BioCLIP 2.5, DINOv3, EfficientNet-B0) encode about plants —
species, organ, photographic scale or background — and does it explain their failures?

**Start here:** [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) — findings, open
questions, priorities.

| Doc | For |
|---|---|
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | What is settled, what is next (P0–P3) |
| [docs/DATA_CATALOG.md](docs/DATA_CATALOG.md) | Every dataset, split, embedding and label, and where the real copy lives |
| [docs/EXPERIMENT_REGISTRY.csv](docs/EXPERIMENT_REGISTRY.csv) | One row per experiment: question, data, code, result, conclusion |
| [docs/MEETING_EXPERIMENTS.md](docs/MEETING_EXPERIMENTS.md) | The seven meeting methods: purpose, what was wrong, proper design |
| [docs/DATA_EXPANSION_PLAN.md](docs/DATA_EXPANSION_PLAN.md) | Which photos to get next (Pound Ridge pilot, then iNat) |
| [docs/CLEANUP_LEDGER.md](docs/CLEANUP_LEDGER.md) | What was removed, what is kept and why, what is still missing |

## Layout

| Folder | Kind | Contents |
|---|---|---|
| `organ_benchmark/` | experiment | Split manifests (**canonical splits live here**), organ probes, retrieval |
| `error_analysis/` | experiment | Four-model species error buckets, taxonomy-level errors, gallery |
| `representation_analysis/` | experiment | Distances, neighbours, K-means K=10/20/50, multi-seed, PCA/UMAP |
| `scale_conditioned_analysis/` | experiment + review UI | Scale-stratified analysis; `review/` = 182-image scale audit |
| `dual_encoder_fusion/` | experiment (closed) | Fusion V1→V3; V3 confirmed no species gain |
| `bioimages_browser/` | data browser | 1,899-image catalog, multi-label tagging analysis (GitHub Pages) |
| `bioimages_human_loop/` | labelling app | Human Gold review (Vercel + Supabase); copies browser data by design |
| `research_report/` | published bundle | Self-contained report assembled from the folders above |
| `scripts/` | pipelines | Modal embedding/VLM runs, Pound Ridge manifest builder |
| `data/` | data | `embeddings/efficientnet_b0_strict/`, `expansion/` (Pound Ridge manifests) |
| `app/`, `worker/`, `db/`, `public/`, … | hosted site | vinext app — see [WEBAPP.md](WEBAPP.md) |

Not yet in Git (see CLEANUP_LEDGER): the meeting-experiment source on the hosted site,
BioCLIP/DINOv3 raw embeddings (Drive), human labels (Supabase).

## Rebuild the Pound Ridge manifest

```bash
python botanical-computer-vision/scripts/build_pound_ridge_manifest.py   # from repo root
```
