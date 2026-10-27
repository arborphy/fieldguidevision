# Botanical computer vision

**Central question now:** when frozen vision models embed plant photographs, is the space
organised by **species**, by the **organ** in frame, or by **shot scale** (distant /
mid-range / close-up)? And does that structure explain identification errors — on
BioImages and, next, on independent field photos?

Status, priorities and next steps: [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).

## How we got here

### 1. Species baselines → BioCLIP is the model to beat
Strict set: 1,641 BioImages photos, 63 species, 465 individual plants, split so no plant
appears in both train and test (1,181 / 249 / 211).

| Model (211 strict test images) | Top-1 | Top-5 |
|---|---:|---:|
| BioCLIP 2.5 zero-shot | 85.8 % | 95.3 % |
| BioCLIP 2.5 frozen + linear probe | 87.2 % | 96.2 % |
| DINOv3 frozen + linear probe | 56.4 % | 83.4 % |
| EfficientNet-B0 frozen + linear probe | 21.8 % | 47.9 % |
| GPT-5.4 / Gemini Flash Lite (85 candidates) | 39.3 % / 40.3 % | ≈ 65 % |

Pages: [`research_report/`](research_report/index.html) ·
[`error_analysis/`](error_analysis/README.md) (shared failures: oak pairs, 17 images all
models miss).

### 2. The spaces are organised differently
K-means k=20 on 1,641 images ([`representation_analysis/`](representation_analysis/README.md),
[`organ_benchmark/`](organ_benchmark/index.html)):

| | Species NMI | Organ NMI | Organ retrieval P@1 (other species) |
|---|---:|---:|---:|
| BioCLIP 2.5 | **0.615** | 0.163 | 0.59 |
| DINOv3 | 0.316 | **0.522** | **0.84** |
| EfficientNet-B0 | 0.204 | 0.503 | 0.78 |

BioCLIP groups by species; DINOv3 and EfficientNet group by organ. Organ is linearly
decodable from all three (DINOv3 best, organ_tag top-1 0.90).

### 3. Shot scale is a major axis
[`scale_conditioned_analysis/`](scale_conditioned_analysis/README.md) asks whether
species mix because they look alike or because they were photographed at the same scale.
Mean cosine similarity between image pairs:

| | same species, different scale | different species, same scale |
|---|---:|---:|
| BioCLIP 2.5 | 0.558 | 0.338 |
| DINOv3 | 0.170 | 0.153 |
| EfficientNet-B0 | 0.176 | **0.232** |

For EfficientNet, scale outweighs species; for DINOv3 they are about equal; BioCLIP stays
species-first. Conditioning on scale raises species purity most at mid-range
(BioCLIP 33 % → 50 %), little at distance. BioCLIP accuracy: close-up 89 %, mid-range
91 %, distant 71 %. Scale labels are provisional (BioImages + Gemma) until the 182-image
human audit is in.

### 4. Combining species and organ signal does not help species ID
[`dual_encoder_fusion/`](dual_encoder_fusion/results/v3_nested_confirmation/ANALYSIS.md):
BioCLIP + DINOv3 cross-attention, locked 3-fold confirmation — species macro-F1 0.760 vs
BioCLIP 0.760. Closed.

### 5. Labels
[`bioimages_browser/`](bioimages_browser/README.md) (1,899-image catalog, multi-label
tagging vs Gemma) · [`bioimages_human_loop/`](bioimages_human_loop/README.md) (Human Gold
review, Vercel + Supabase) · [`scale_conditioned_analysis/review/`](scale_conditioned_analysis/review/index.html)
(scale audit). Human results still need exporting.

### 6. Meeting methods (Oct) — not yet in Git
NSC, enrichment, soft clustering, activation mapping, autoencoder, inversion: first runs
are on the hosted site only. What each is for and how to redo it around the question
above: [docs/MEETING_EXPERIMENTS.md](docs/MEETING_EXPERIMENTS.md).

## Docs

| | |
|---|---|
| [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | Findings, open questions, P0–P3 |
| [MEETING_EXPERIMENTS.md](docs/MEETING_EXPERIMENTS.md) | The seven methods: purpose, round-1 problems, design |
| [DATA_EXPANSION_PLAN.md](docs/DATA_EXPANSION_PLAN.md) | Which photos to get next |
| [DATA_CATALOG.md](docs/DATA_CATALOG.md) | Datasets, splits, embeddings, labels, where the real copies are |
| [EXPERIMENT_REGISTRY.csv](docs/EXPERIMENT_REGISTRY.csv) | One row per experiment |
| [CLEANUP_LEDGER.md](docs/CLEANUP_LEDGER.md) | What was removed / kept / is missing |

Other folders: `scripts/` (Modal embedding and VLM runs, Pound Ridge manifest builder),
`data/` (EfficientNet embeddings, expansion manifests), `app/ worker/ db/ public/ …`
(hosted site, see [WEBAPP.md](WEBAPP.md)).
