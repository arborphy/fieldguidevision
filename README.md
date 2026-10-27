# FieldGuideVision

Identifying plants from field photographs, and understanding **what a vision model is
actually looking at** when it does — the species, the organ in frame, or simply how far
away the camera was.

This is the research branch (`lindsey-branch`). The work lives in
[`botanical-computer-vision/`](botanical-computer-vision/README.md); start there.

## The research line in one page

| Stage | Question | Answer so far | Where |
|---|---|---|---|
| **0 · Prototype** (Aug) | Can region evidence (Gemma → GroundingDINO → SAM polygons) + Gemini name the species? | Runs end to end: 58/85 blind pilot. But no controls, so we cannot tell what drove it. → build proper baselines. | `notebooks/`, `segmentation/` |
| **1 · Baselines** (Sep) | Which frozen model identifies species on BioImages? | **BioCLIP 2.5 clearly**: 85.8 % zero-shot, 87.2 % probe on 211 strict test images. DINOv3 56.4 %, EfficientNet 21.8 %, GPT-5.4 / Gemini ≈ 40 %. | `botanical-computer-vision/research_report/`, `error_analysis/` |
| **2 · Errors** | Where do they fail together? | Oaks and other within-genus pairs; 17 images all four models miss. | `error_analysis/` |
| **3 · Representation** | How are embedding spaces organised? | **They organise by different things.** BioCLIP clusters follow species (species NMI 0.62, organ 0.16). DINOv3 and EfficientNet clusters follow **organ** (organ NMI ≈ 0.5, species ≤ 0.32). | `representation_analysis/`, `organ_benchmark/` |
| **4 · Shot scale** (Oct) | Are species mixed because they look alike, or because they were shot at the same distance? | **Scale is a major axis.** In EfficientNet, two different species at the same scale are *more* similar than one species at different scales; in DINOv3 about equal; BioCLIP stays species-first. Conditioning on scale helps mid-range, not distant. | `scale_conditioned_analysis/` |
| **5 · Fusion** | Does BioCLIP (species) + DINOv3 (organ) help? | No species gain (macro-F1 0.760 vs 0.760); organ hint only. Closed. | `dual_encoder_fusion/` |
| **6 · Labels** | Can we trust organ / scale labels? | Model labels (Gemma) are references; Human Gold review apps are live, completion to be exported. | `bioimages_browser/`, `bioimages_human_loop/`, `scale_conditioned_analysis/review/` |
| **7 · Meeting methods** (Oct) | NSC, enrichment, soft clustering, activation maps, AE, inversion | First runs exist on the hosted site only; they need recovering and a redesign around stage 4's question. | [`MEETING_EXPERIMENTS.md`](botanical-computer-vision/docs/MEETING_EXPERIMENTS.md) |

**Next:** test whether these findings hold on independent field photos (Pound Ridge /
iNaturalist), and use the meeting methods to explain the species-vs-organ-vs-scale
structure. See [`PROJECT_STATUS.md`](botanical-computer-vision/docs/PROJECT_STATUS.md).

## Repository map

| Path | What | Owner |
|---|---|---|
| `botanical-computer-vision/` | All vision research, published reports, review apps, hosted site | Lindsey |
| `notebooks/`, `images/` | Aug prototype notebooks (Grounding DINO, agentic labeling, general-purpose vision report) | Lindsey |
| `segmentation/`, `canonical_adapter/`, `hierarchical_pipeline.py`, `scripts/`, `tests/`, `Dockerfile`, [`docs/notice_tagging.md`](docs/notice_tagging.md) | Notice-first organ tagging (Grounding DINO + SAM 2.1), DeVo gates, photo-batch tooling | V W, avwilks |
| `observations_pound_ridge/` | Ward Pound Ridge iNaturalist / GBIF observations — source for the next photo set | shared |
| `manual_of_woody_plants/`, `newcomb_wildflower_guide/`, `gobotany/` | Field-guide knowledge: species × trait vocabularies, definitions, illustrations | shared |
| `archive/2026-06_jm_reproduction/` | June reproduction of the earlier Newcomb stepwise experiment; superseded, kept for record | Lindsey |
