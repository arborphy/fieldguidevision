# BioImages Browser

A static, image-first browser for the BioImages northeastern trees collection. It preserves the original BioImages annotations while reorganizing all 1,899 photographs into a modern hierarchy:

`species → organ category → subview → images`

The site can also be browsed by individual plant, which links different views of the same documented tree.

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
- Auxiliary model-evaluation page

## Rebuild

Download `corpus.json` from the project Drive folder, then run:

```bash
python3 scripts/build_browser.py /path/to/corpus.json
python3 scripts/audit_image_urls.py
python3 scripts/verify_browser.py
node --check assets/app.js
```

The build writes both machine-readable JSON/CSV and the browser-friendly `data/site-data.js`. The current source corpus SHA-256 is recorded in `data/validation.json`.

The committed URL audit checked every one of the 1,899 BioImages thumbnail URLs: all returned HTTP 200 with an image content type. Full results are in `data/image_url_audit.csv` and the summary is in `data/image_url_audit.json`.

## Annotation policy

BioImages is always the canonical source. A primary label is the original `organ_category / subview`; model predictions never replace it. Existing DINOv3 held-out predictions are attached to 211 image detail pages only as auxiliary comparison data. The model predicts organ category, not fine subview.

## Manual review queue

No image is missing a primary label. The generated review queue flags 194 records: 191 use an `unspecified` view term, 3 lack a named creator, and 2 fall outside the eight main display groups (some records have more than one flag). These remain browsable under their original labels and are marked in `data/site-data.js` and `data/image_manifest.csv`.
