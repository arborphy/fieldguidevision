# Enclosed Cheap-Look Demo

This directory contains a bounded local demonstration of organ-led image
inspection. It measures supplied pixels, builds a provenance-distinct DEVO
inspection checklist, and can optionally ask OpenRouter for unconfirmed
notices. It does not assign taxa, confirm botanical assertions, or write to
production systems.

## Report to Review

Open the generated Tuckaway index in a browser:

[fieldguidevision/scratch/tuckaway-qa-20260908/index.html](./tuckaway-qa-20260908/index.html)

The index contains a contact sheet and links to 37 photo reports. Each report
contains the photo with any supplied region outlines, RGB histograms, pixel
and edge measurements, DEVO inspection questions, and the model-proposal
section. Because the run was intentionally offline, proposals are marked
`not_requested`.

For a colleague-facing introduction, start with these three reports:

1. `tuckaway-qa-20260908/18/report.html` — foreground leaf and fruit-region
   measurements.
2. `tuckaway-qa-20260908/22/report.html` — flower-cluster measurement against
   the wider house-and-foliage scene.
3. `tuckaway-qa-20260908/32/report.html` — bark-region measurement against
   the whole forest scene.

These examples show why region markup matters: a selected leaf, flower
cluster, or bark patch has a different histogram from the surrounding scene.
They are measured pixel regions, not botanical identifications.

## Inputs and Outputs

- `scripts/fgv-demo.py` — the reusable enclosed demo.
- `tests/test_fgv_demo.py` — offline regression tests.
- `tests/verify_fgv_tuckaway.py` — the local QA driver used against the Tuckaway
  JPEGs.
- `tests/fixtures/tuckaway-qa-regions.json` — provisional assistant QA markup
  for four regions on three photos. Treat these as examples for checking the
  system, not confirmed field annotations.
- `scratch/tuckaway-qa-20260908/` — local generated evidence, report HTML,
  contact sheet, summary, and QA screenshots (in git-ignored `scratch/`).

The Tuckaway source photos remain outside this repository under the supplied
local path and were not modified or uploaded.

## Reproduce Locally

From the `fieldguidevision` directory:

```bash
uv run --extra segmentation python tests/verify_fgv_tuckaway.py \
  "/Users/valkyrie/Desktop/arborphy product/phys/tuckaway" \
  scratch/tuckaway-qa-new \
  --annotations tests/fixtures/tuckaway-qa-regions.json
```

The output directory must not already exist; this prevents accidental
overwrites between runs. The command reads the photos twice, verifies that the
evidence JSON is identical, writes per-photo reports, and creates an index.

Run the regression tests:

```bash
uv run --extra segmentation python -m unittest discover \
  -s tests -p test_fgv_demo.py -v
```

Both commands use the local Python environment, OpenCV, NumPy, Pillow, and the
canonical GoBotany release JSON. No network or GPU is required.

## Optional OpenRouter Pass

OpenRouter is off by default. To ask a vision model for proposed notices, use
`--openrouter-model` with a real model ID. That option uploads the full photo
and every supplied region crop to OpenRouter and its provider.

Model output is retained as raw response text plus a validated proposal list.
Invalid responses are kept for inspection but never become confirmed
assertions. OpenRouter is not deterministic, even at temperature zero.

## Current Limits

- This is a scratch demonstration, not a production pipeline.
- The four marked regions are provisional QA examples; 34 photos were tested
  scene-only.
- The DEVO checklist uses explicit lexical routing hints and is not a curated
  semantic mapping.
- Measurements do not establish venation, health, individual identity, or a
  taxon.
- DNG originals and video were excluded from this run.
- The scratch directory is git-ignored, so generated reports are local
  artifacts rather than committed program evidence.
