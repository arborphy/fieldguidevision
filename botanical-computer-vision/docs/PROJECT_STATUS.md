# Project status and priorities

Updated 2026-10-27 · branch `lindsey-cleanup` (from `lindsey-branch` @ `0aef4c4`).

## The research question

Frozen image models (BioCLIP 2.5, DINOv3, EfficientNet-B0) identify plants from photos.
**What do their embeddings actually encode — species, organ/view, photographic scale or
background — and does that explain when identification fails, especially on photos from
a different source?**

## What we know (settled; do not re-run)

| Finding | Evidence |
|---|---|
| BioCLIP is the strongest species model: strict 63-species test 181/211 zero-shot (85.8 %), 87.2 % linear probe | `research_report/data/experiment_registry.csv` |
| DINOv3 probe is weaker on species (56.4 %) but **better on organ** (organ_tag 0.896 vs BioCLIP 0.810) | `organ_benchmark/metrics/benchmark_summary.csv` |
| BioCLIP + DINOv3 attention fusion gives **no species gain** (macro-F1 0.760 vs 0.760, CIs span 0); organ hint only | `dual_encoder_fusion/results/v3_nested_confirmation/ANALYSIS.md` |
| 2-D projections keep ~10 % of neighbourhoods; pictures are not evidence | meeting run (site), see registry `mtg6_pca_fidelity` |
| AE adds nothing over PCA at this sample size | registry `mtg5_autoencoder` |
| Clusters exist but are not organ or species truth; overlap from NEO-K-means is parameter-driven | registry `mtg4_neo_kmeans` |

## What is open

1. Whether any embedding structure is about **organ/scale rather than species/individual**
   (needs species-stratified enrichment).
2. Whether results hold on **independent field photos** (needs Pound Ridge / iNat data).
3. How much Human Gold labelling actually exists (Supabase export).

## Priorities

| P | Work | Done when |
|---|---|---|
| **P0** | 1. Push this branch (push from this session was refused — needs repo access). 2. Recover the site-only meeting source/results into `scale_conditioned_analysis/meeting/` etc. 3. Pull BioCLIP/DINOv3 `embeddings.npz` from Drive, record SHA-256 in DATA_CATALOG. 4. Export Supabase Human Review + scale review; record counts. | A fresh clone + Drive files reproduce every number in EXPERIMENT_REGISTRY. |
| **P1** | Pound Ridge pilot: download the 383-photo pilot (not into Git), dedup, review organ/scale on the `dev` half, embed, test BioCLIP on `sealed_test`. See DATA_EXPANSION_PLAN Stage 1. | One sealed-test domain-shift table, reported apart from BioImages. |
| **P1** | Meeting experiments §2a (PCA d*), §4 (K selection + soft membership), §2b (species-stratified enrichment). See MEETING_EXPERIMENTS. | We can say which clusters are organ/scale structure, with CIs, or that none are. |
| **P2** | §5 axis reading + §7 retrieval inversion; §1 NSC in PCA basis; §3 activation mapping on surviving clusters with SAM2 masks. | Labelled axis table; per-cluster on-organ saliency mass. |
| **P2** | DATA_EXPANSION_PLAN Stage 2: iNat pull for the 19 missing + 26 thin species. | ≥ 20 independent observations per strict species. |
| **P3** | AE architecture search, generative inversion, more fusion variants, new projection pages, site redesigns. | Only with a concrete question that new data raised. |

## Working rules

- One registry row per experiment (`docs/EXPERIMENT_REGISTRY.csv`); update it in the same
  commit as the result.
- Every metric carries its denominator and split. Results on different splits are not
  compared as numbers.
- Model labels (Gemma, VLMs) are references; Human Gold is truth.
- Negative results stay in the registry. Stop conditions in MEETING_EXPERIMENTS are binding.
