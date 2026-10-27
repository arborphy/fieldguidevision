# Cleanup ledger

What was removed on branch `lindsey-cleanup`, what was deliberately kept, and what is a
candidate for later. Rule: nothing that is the only copy of data, labels or evidence is
deleted; nothing a live page depends on is moved without changing its build.

## Done in this branch

| Change | Size | Why it was safe |
|---|---|---|
| Deleted `research_report_share.zip`, `representation_analysis_share.zip`, `representation_analysis_final_share.zip`, `efficientnet_baseline_share.zip` | 158 MB tracked | Every entry is byte-identical to a tracked file or an older version of a tracked README/HTML — **except** the EfficientNet run outputs, which were extracted first (next row). |
| Added `data/embeddings/efficientnet_b0_strict/` | 6 MB | The only versioned copy of the EfficientNet-B0 strict embeddings + predictions lived inside a ZIP. |
| Moved `01_general_purpose_vision_report_en.ipynb` → `notebooks/` | — | Root-level clutter; outputs are embedded, image paths are repo-root relative and unchanged. |
| Renamed `botanical-computer-vision/README.md` → `WEBAPP.md` | — | It was the vinext starter template. The new README is the research entry point. |
| Added `docs/` (status, data catalog, registry, experiment designs, expansion plan) and the Pound Ridge manifest | <0.5 MB | Documentation only. |

Git history still contains the ZIPs (the `.git` folder stays ~245 MB). Shrinking history
needs a rewrite of a shared branch — only if the team agrees.

## Kept on purpose (looks redundant, is not safe to remove yet)

| Item | Duplication | Why kept | How to remove later |
|---|---|---|---|
| `research_report/gallery_images`, `assets/images` | ~98 MB identical to `error_analysis/` | `research_report` is the self-contained published bundle; `build_research_report.py` regenerates it only from `outputs/` and a `/tmp` cache that are **not in Git** | Make the report reference `../error_analysis/` paths, or regenerate in CI; then drop the copies. |
| `error_analysis/samples/images` vs `gallery_images` | ~35 MB identical | `build_error_analysis.py` and `annotation/human_error_annotation.csv` reference `samples/images` paths | Point the gallery at `samples/images`, rebuild, delete `gallery_images`. |
| `bioimages_human_loop/` vs `bioimages_browser/` | ~17 MB identical | Human loop is a separate Vercel deployment whose `build_vercel.mjs` copies its own `tagging/`, `data/` | Have `build_vercel.mjs` copy from `../bioimages_browser/` at build time. Test the Vercel build first. |
| `app/`, `worker/`, `db/`, `drizzle/`, `build/`, `public/`, `.openai/`, `package*.json`, configs | — | They are the hosted site (ChatGPT Sites / vinext) | Keep. |
| `segmentation/`, `notebooks/`, `images/`, `tests/test_planar_filter_feasibility.py`, `scripts/live_webcam_filter_demo.py` | — | Early Grounding-DINO / SAM2 exploration; `segmentation/` also exists on `main` and its masks are needed for MEETING_EXPERIMENTS §3 | Keep; document in README as "early exploration". |
| Root `manual_of_woody_plants/`, `newcomb_wildflower_guide/`, `gobotany/`, `observations_pound_ridge/`, `trials/`, `canonical_adapter/`, `main.py`, `hierarchical_pipeline.py` | — | Come from `main` / other team members | Not this branch's to delete. |

## Missing — must be recovered (P0)

| Item | Where it is now |
|---|---|
| `scale_conditioned_analysis/meeting/`, `nsc/`, `cluster_activation_mapping/` (source + results of the 7 meeting experiments) | Hosted site version 16 only |
| BioCLIP / DINOv3 strict `embeddings.npz` | Google Drive `outputs/` |
| Human Review (100) and scale review (182) labels | Supabase |
| `outputs/` run summaries cited by `research_report/data/experiment_registry.csv` | Local machine / Drive |
