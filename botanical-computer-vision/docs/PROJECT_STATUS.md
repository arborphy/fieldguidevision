# Project status and priorities

Updated 2026-10-27 · branch `lindsey-cleanup` (from `lindsey-branch` @ `0aef4c4`).

## The research question

When frozen models embed plant photos, is the space organised by **species**, by the
**organ** in frame, or by **shot scale** — and does that explain identification errors,
on BioImages and on independent field photos?

This grew out of the August evidence-bag prototype (Gemma → GroundingDINO → SAM → Gemini,
58/85 blind pilot), which worked but could not say *why*. The baselines and
representation analyses since then are the controlled version of that question.

## What we know (settled; do not re-run)

| Finding | Evidence |
|---|---|
| BioCLIP is the species model: 85.8 % zero-shot, 87.2 % probe on 211 strict test images; DINOv3 56.4 %, EfficientNet 21.8 %, VLMs ≈ 40 % | `research_report/`, `representation_analysis/data/frozen_probe_comparison.csv` |
| **Spaces organise by different things.** BioCLIP k=20 clusters follow species (NMI 0.615 vs organ 0.163); DINOv3 / EfficientNet follow organ (organ NMI 0.52 / 0.50, species 0.32 / 0.20) | `scale_conditioned_analysis/` separation table; `organ_benchmark/` |
| **Shot scale is a major axis.** Different species at the same scale vs same species at different scales: EfficientNet 0.232 > 0.176, DINOv3 0.153 ≈ 0.170, BioCLIP 0.338 ≪ 0.558 | `scale_conditioned_analysis/data/scale_pairwise_similarity.csv` |
| Conditioning on scale helps species separation at mid-range (BioCLIP purity 33 → 50 %), not at distance; BioCLIP accuracy distant 71 % vs close-up 89 % | `scale_conditioned_analysis/data/` |
| BioCLIP + DINOv3 fusion gives no species gain (0.760 vs 0.760, CIs span 0) | `dual_encoder_fusion/results/v3_nested_confirmation/ANALYSIS.md` |
| 2-D projections keep ~10 % of neighbourhoods; AE adds nothing over PCA; NEO-K-means overlap is parameter-driven | meeting runs (site only), registry `mtg*` |

## What is open

1. Is the scale effect real or an artefact of provisional labels and of species/individual
   confounding? (Human scale audit + species-stratified enrichment.)
2. After scale is controlled, what explains the remaining mixing — organ, leaf state,
   background? (Meeting methods, MEETING_EXPERIMENTS.md.)
3. Does any of it hold on **independent field photos**? (Pound Ridge / iNat.)
4. How much Human Gold labelling exists? (Supabase export.)

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
