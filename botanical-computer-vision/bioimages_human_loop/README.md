# BioImages Human Loop

An experimental, static companion to the stable [`bioimages_browser`](../bioimages_browser/). It reuses the complete 1,899-image catalog, model predictions, Full Gallery, and visual style without changing the existing browser.

The workflow is intentionally ordered:

1. discover repeated error modes across the full corpus;
2. visually audit high-signal representatives;
3. select a targeted, diverse 100-image Human Set;
4. collect blind multi-label Human Gold;
5. use Human Gold to validate the existing modes, not invent a new analysis.

No model is fine-tuned here.

## Current outputs

- 1,899 images analyzed
- 12 overlapping error modes with complete membership CSV
- 169 high-signal, species-diverse visual-audit images
- 100 Human Review images spanning all 85 species and 100 different individuals
- 8–12 representative photographs per error mode
- Full Gallery retained for good matches, BioImages mismatches, Gemma disagreements, and cross-model disagreements

The largest automated modes are high-confidence extra tags, flower/fruit/bud overlap, strong backbone disagreement, leaf-view over-interpretation, missed secondary woody structures, fruit-state/view instability, whole-plant scale splitting, and reproductive structures disappearing in context. Counts overlap and are not prevalence estimates of confirmed errors.

## Human Review UI

Open `index.html`, choose **Human review**, and label one large image at a time. Before submission, BioImages, Gemma, DINOv3, BioCLIP 2.5, and EfficientNet-B0 answers are hidden. The reviewer can select any number of:

`leaf, twig, bark, flower, fruit, cone, seed, whole plant, other`

Primary subject, ambiguity, another tag, and a short note are optional. After submission, every reference and model tag is revealed. On GitHub Pages, labels remain a browser-local preview under `bioimages-human-gold-v2`. On the Vercel deployment, reviewers sign in with a magic link and every submission is written to their RLS-protected Supabase record with revision history.

When labels exist, the home page automatically reports:

- **Tag precision**: predicted coarse tags confirmed by Human Gold / all predicted coarse tags.
- **Truth completeness**: Human Gold tags found by the model / all Human Gold tags.
- Per-mode counts of reviewed examples, model errors, and BioImages single-label ambiguity.

Unlabeled images never enter these Human Gold metrics.

## Discovery and visual audit

`scripts/discover_error_modes.py` combines BioImages mismatch, Gemma disagreement, three-model disagreement, confidence, tag co-occurrence, and known fine-view confusions. It writes the full mode membership, summary, representative IDs, and deterministic Human Set.

The visual-audit layer reuses the existing image-grounded Gemma annotations—the same Gemma run that viewed all 1,899 photographs without species or BioImages metadata. `scripts/build_gemma_visual_audit.py` tests whether those free visual tags corroborate each proposed mode. This is AI evidence, not Human Gold. Representative galleries were also visually inspected during site QA. The audit does not overwrite BioImages or fill any human field.

`scripts/modal_error_mode_audit.py` is included as an optional independent second-pass VLM audit when a valid `openrouter-bioimages-vlm` secret is available.

## Machine-readable files

- `analysis/error_modes.json` and `.csv`: summaries and model counts
- `analysis/error_mode_members.csv`: all image-mode memberships
- `analysis/vlm_audit_candidates.json`: deterministic 169-image audit set
- `analysis/vlm_error_audit.jsonl`: existing-Gemma visual corroboration
- `analysis/human_review_set.json` and `.csv`: the 100 images and explicit selection reasons
- `data/human-loop-data.js`: compact static-site payload

The copied `data/`, `tagging/`, and Full Gallery assets preserve the stable browser’s complete inputs and results. Images are loaded from BioImages and are not committed.

## Rebuild and verify

From the repository root:

```bash
python3 botanical-computer-vision/bioimages_human_loop/scripts/discover_error_modes.py
python3 botanical-computer-vision/bioimages_human_loop/scripts/build_gemma_visual_audit.py
python3 botanical-computer-vision/bioimages_human_loop/scripts/discover_error_modes.py
python3 botanical-computer-vision/bioimages_human_loop/scripts/verify_human_loop.py
node --check botanical-computer-vision/bioimages_human_loop/assets/app.js
```

Serve the directory with any static server or publish it directly under GitHub Pages. The stable `bioimages_browser/index.html` remains untouched.

For a shared, persistent multi-reviewer deployment, see [`VERCEL_HUMAN_REVIEW.md`](VERCEL_HUMAN_REVIEW.md). The recommended production path is Vercel for hosting plus Supabase Postgres/Auth for reviewer identity, RLS-protected annotations, audit history, and consensus export.

The deployment source includes `vercel.json`, a build-time runtime-config generator, and `supabase/migrations/001_human_review.sql`. The migration seeds the deterministic 100-image batch and exposes only a transactional `submit_annotation` function to authenticated reviewers.
