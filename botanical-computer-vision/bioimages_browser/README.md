# BioImages Browser and multi-label tagging analysis

A static, image-first browser for the BioImages northeastern trees collection. It preserves the original BioImages annotations while reorganizing all 1,899 photographs into a modern hierarchy:

`species → organ category → subview → images`

The site can also be browsed by individual plant, which links different views of the same documented tree. Its home page now evaluates three frozen visual models as multi-label taggers over the complete collection.

## Collection coverage

- 1,899 images
- 85 species
- 492 individual plants
- 12 original organ categories
- 36 subview types
- 45 exact `organ_category / subview` primary labels
- 1,899 / 1,899 images have a complete primary label

The eight visitor-facing organ groups are Whole tree, Bark, Twig, Leaf, Flower / Inflorescence, Fruit, Cone, and Seed. Original `organ_tag`, `organ_category`, `subview`, and `view_code` values are retained without overwriting them.

## Open and publish

Open [`index.html`](index.html) directly, serve this directory with any static server, or publish it under GitHub Pages. Images are loaded from BioImages over HTTPS and are not committed to this repository.

Available browsing routes:

- Home organ overview
- Species index and species-organ-subview pages
- Organ and subview filters
- Individual-plant index and linked galleries
- Full image catalog with filters
- Image detail with provenance, date, credit, license, and source links
- Full-corpus tagging comparison and per-tag results
- Filterable good-match, BioImages-mismatch, Gemma-disagreement, and cross-model-disagreement galleries
- Image details with BioImages, Gemma, DINOv3, BioCLIP 2.5, and EfficientNet-B0 tags
- Auxiliary coarse-organ model-evaluation page

## Multi-label tagging method

The analysis uses a 40-tag, auditable vocabulary of visible structures and supported views. The complete mapping from original `organ_tag`, `organ_category`, and `subview` values is saved under `tagging/`. Original BioImages annotations are never overwritten.

For a fair comparison, DINOv3, BioCLIP 2.5, and EfficientNet-B0 are frozen. Each uses the same per-tag balanced logistic-regression probe, deterministic five-fold `individual_id`-disjoint predictions, training-fold-only threshold selection, and evaluation code. Each fold learns from Gemma normalized tags only for its training individuals; held-out image labels never enter that model fit. Every displayed prediction is out of fold. No backbone is fine-tuned.

Gemma 3 27B independently sees each image without species or BioImages metadata and returns any number of free visual tags. Those free tags are preserved for inspection; a normalized tag list from the same response supervises training folds and is used to score held-out folds with precision, recall, F1, and Jaccard. Gemma is a model teacher/reference, not human ground truth.

BioImages match means that its single canonical coarse structure appears somewhere in a model's predicted tag set. It does not claim that other visible tags are wrong. The two `unspecified` records remain visible but are excluded from this denominator.

## Rebuild

Download `corpus.json` from the project Drive folder, then run:

```bash
python3 scripts/build_browser.py /path/to/corpus.json
python3 scripts/audit_image_urls.py
python3 scripts/verify_browser.py
node --check assets/app.js
```

The reproducible model pipeline is:

```bash
uv run --with modal modal run scripts/modal_full_embeddings.py
uv run --with modal modal run scripts/modal_gemma_multilabel.py --stage full
uv run scripts/build_tagging_analysis.py
python3 scripts/build_browser.py /path/to/corpus.json
python3 scripts/verify_browser.py
```

Images remain in Google Drive/BioImages and are not committed. Frozen embeddings are generated locally under `outputs/`; the committed `tagging/` directory contains the compact per-image predictions, references, metrics, vocabulary mapping, and report needed to audit the published site.

The build writes both machine-readable JSON/CSV and the browser-friendly `data/site-data.js`. The current source corpus SHA-256 is recorded in `data/validation.json`.

The committed URL audit checked every one of the 1,899 BioImages thumbnail URLs: all returned HTTP 200 with an image content type. Full results are in `data/image_url_audit.csv` and the summary is in `data/image_url_audit.json`.

## Annotation policy

BioImages is always the canonical source. A primary label is the original `organ_category / subview`; model predictions never replace it. Multi-label predictions supplement the source record and can include multiple visible structures and view tags. The older 211-image DINOv3 coarse-organ evaluation remains available only as an auxiliary page.

## Manual review queue

No image is missing a primary label. The generated review queue flags 194 records: 191 use an `unspecified` view term, 3 lack a named creator, and 2 fall outside the eight main display groups (some records have more than one flag). These remain browsable under their original labels and are marked in `data/site-data.js` and `data/image_manifest.csv`.
