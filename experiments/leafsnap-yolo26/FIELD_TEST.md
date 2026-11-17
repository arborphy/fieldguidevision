# ArborQuest offline APK — director-led field test

**Status.** The E3 organ gate passed. The leaf-type specialist is **E4 NOT VALIDATED** and appears only as `PROPOSED` telemetry. Do not treat its output as botanical truth or as an E4 pass.

## Build and export

- Android package: `ai.arborphy.mobile.yolo26`
- Open **Offline Eyes** in ArborQuest.
- Both ONNX models and ONNX Runtime WASM are bundled; airplane mode is supported.
- Select the active **FIELD TEST CATEGORY** before each capture series.
- Tap **Export log**. Android writes `Documents/arq-telemetry/arq-telemetry-<run-id>.json` and opens the share sheet when available.
- Export one JSON package per category/session so the review units stay separate.

The actual export schema is `arq.telemetry/v1`. It contains run/session IDs, model metadata, online state, test category, gate samples, lock transitions, and specialist results. Specialist events include a 96 px crop thumbnail; full camera frames are not stored.

## Capture series

Compound whole leaves and isolated leaflets are different categories. Capture both from the same individual when possible.

| Category | Framing | Review interpretation |
|---|---|---|
| `simple_whole` | One simple leaf with blade and petiole, filling roughly 60–80% of the frame | Expected proposal: `simple_leaf` |
| `compound_whole` | Entire compound leaf: petiole, rachis, all leaflets and terminal leaflet | Expected proposal: `compound_leaf` |
| `isolated_leaflet` | One leaflet with no rachis visible, from the same compound individual | Hard negative / not assessable; do not score as a simple leaf |
| `non_leaf` | Bud, flower, twig, branch, or whole plant | Tests organ-gate false positives and leaf-lock suppression |
| `flicker_occlusion` | Continuous 20–30 s scan with partial occlusion, small pans and focus changes | Tests debounce, grace buffering and target switching |

Suggested initial coverage:

- 5 simple species × 3 individuals × 2 views
- 3 compound species × 3 individuals × 4 whole-leaf views
- The same compound individuals × 2 isolated-leaflet views
- 5 non-leaf views per available organ
- 3 flicker/occlusion sessions

Record species/individual and view conditions in your field notes. The app records the test category but does not know the true species.

## Runtime procedure

1. Select the category.
2. Point at the subject and allow the organ gate to acquire a leaf.
3. Hold for **1 second and at least 3 consistent detections**.
4. Observe `LEAF LOCKED` and the specialist's proposed result and crop preview.
5. Briefly occlude the leaf. The lock should survive approximately **2.5 seconds of grace** so the specialist can complete.
6. Use **New plant ↺** before moving to another individual.
7. Export the log before changing category.

Keep failures in the log. Do not delete false positives, lost locks, or low-confidence proposals.

## Review fields

- **Gate:** leaf detection and false positives by category, confidence, boxes and latency.
- **Lock:** acquisition timestamp, hits, lock ID, grace state/remaining time, loss and reacquisition.
- **Specialist:** raw `compound_leaf` / `simple_leaf` probabilities, margin, latency, crop thumbnail and lock ID.
- **View effect:** compare `compound_whole` against `isolated_leaflet` from the same individual. Never collapse those into one accuracy number.
- **Latency/stability:** gate FPS, cold versus warm inference, and thermal behavior during longer sessions.

## What this test does not establish

- It does not change the held-out E4 failure.
- It does not validate species identification; no species model is used.
- It does not authorize user-facing leaf-type claims.
- It does not replace the next curated whole-leaf/leaflet dataset described in `NEXT_DATA_CURATION.md`.
