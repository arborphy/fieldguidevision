# Next data curation — exact requirements

Purpose: build a field-domain, per-image-adjudicated dataset that can settle
the question E4 could not — whether a crop/frame shows a **whole compound
leaf** or an **isolated leaflet** — and supply the later margin/twig layers.
This is a data specification, not a promise of model performance.

## Why the current data is insufficient

- Dirr `leaf_type` is a **species-level** assertion; every LeafSnap crop of a
  compound species inherits "compound", even when the mask isolates one
  leaflet. V1/V2 both failed on exactly that mismatch.
- LeafSnap crops are mask foreground boxes, not field frames; they cannot
  teach the organ gate about cluttered scenes.
- Compound test coverage is only 3–4 species (V1: 193 field crops; V2: 214
  across 4 species), so the load-bearing metric is species-fragile.
- There is no individual ID, no rachis-visibility flag, and no hard-negative
  category for leaflets.

Do not "fix" this by relabelling existing crops from species metadata. The
new labels must describe **what is visible in the frame**.

## Label schema (one row per frame)

Store as JSONL alongside the original image. Keep species identity and view
labels in separate fields; never derive one from the other at read time.

```json
{
  "frame_id": "CUR-000123",
  "image": "frames/CUR-000123.jpg",
  "captured_at": "2026-10-29T18:30:12Z",
  "site_id": "WPR-NORTH-01",
  "species": "Juglans cinerea",
  "species_source": "director | botanist | herbarium",
  "species_confidence": "confirmed | probable",
  "individual_id": "JUG-WPR-03",
  "population_id": "WPR-JUG-03",
  "organ": "leaf | bud | flower | twig | bark | branch | entire_plant | non_plant",
  "leaf_architecture": "simple | compound | leaflet | unknown",
  "leaf_view": "whole_simple | whole_compound | isolated_leaflet | compound_fragment | margin_closeup | reverse_surface | not_applicable",
  "rachis": {"visible": true, "complete": true, "fragment_only": false},
  "petiole_visible": true,
  "terminal_leaflet_visible": true,
  "leaflets_visible": 7,
  "leaflet_attachment": "attached | detached | unknown",
  "pair_id": "PAIR-JUG-WPR-03-02",
  "view": {"distance_band": "near | mid | far", "occlusion_pct": 0, "blur": "none | slight | severe", "lighting": "sun | shade | backlit | flash", "background": "foliage | sky | ground | hand | structure | mixed"},
  "damage_or_disease": "none | insect | fungal | mechanical | unknown",
  "phenology": "leafing | flowering | fruiting | senescing | dormant | unknown",
  "operator": "initials",
  "notes": "free text; record label ambiguity here",
  "license": "…",
  "source": "director_capture | collaborator | existing_collection"
}
```

Rules:

- `leaf_architecture` describes the **frame**, not the species. A frame of one
  leaflet is `leaflet`, even when the species is compound.
- `leaf_view = whole_compound` requires `rachis.complete = true` and the
  terminal leaflet visible. A cut-off rachis is `compound_fragment`.
- `leaf_view = isolated_leaflet` requires no rachis in frame and
  `leaflet_attachment = detached` (or unknown if it was plucked).
- Ambiguous frames get `unknown`; do not guess. Ambiguity rate is itself a
  useful curation metric.

## A. Whole compound vs isolated leaflet (the critical axis)

Capture **paired** frames from the same individual and, where possible, the
same shoot:

1. A whole compound leaf: petiole, continuous rachis, all leaflets, terminal
   leaflet, filling 60–80% of the frame.
2. An isolated leaflet from that same leaf: no rachis visible, detached or
   clearly separated, filling the frame.
3. A second distance for the whole leaf (mid distance) so the architecture is
   visible without macro detail.
4. Optional: a compound fragment (rachis cut off) — label it
   `compound_fragment`, not `whole_compound`.

For palmately compound leaves (e.g. *Aesculus*), define the rachis as the
attachment node plus all leaflets radiating from it; mark `rachis.complete`
only when every leaflet attaches at that node and the petiole is visible.

Target: every compound species has whole/leaflet pairs across ≥5 individuals.
Never reuse a LeafSnap mask crop as a whole-compound exemplar: the mask often
isolates a leaflet.

## B. Whole leaf / rachis visibility

Record per frame, not per species:

- `rachis.visible` — any continuous axis is visible.
- `rachis.complete` — continuous from petiole to terminal leaflet (or the
  palmate attachment node), with no break at the frame edge.
- `petiole_visible`, `terminal_leaflet_visible`, `leaflets_visible` (count).
- `view.distance_band` and `occlusion_pct`.

Frames where the rachis is hidden by the hand, another leaf, or the frame edge
are **not** whole-compound examples. Keep them, labelled
`compound_fragment` or `unknown`, as hard negatives.

## C. Margin / twig / reverse views (later layers, not E4)

Collect these now, but keep them out of the simple-vs-compound training set:

| view | framing | label fields |
|---|---|---|
| `margin_closeup` | 3–5 cm of margin filling the frame; teeth/lobes/entire edge legible; include a scale reference when possible | `margin_type: entire | serrate | dentate | crenate | lobed | undulate | unknown`, `margin_secondary` |
| `twig_winter` | terminal bud + leaf scar, dormant season; one twig per frame | `bud_type`, `leaf_scar`, `twig_color`, `pubescence` |
| `reverse_surface` | abaxial face of the **same leaf** as an obverse frame; use a `pair_id` | `surface_bottom`, `venation`, `pubescence` |
| `venation_closeup` | backlit or oblique close-up of the vein network | `venation_pattern`, `vein_order_visible` |

These support the Dirr lower-level features (`leaf_shape_margins`,
`leaf_surface_bottom`, `leaf_venation`) whose EDA coverage is 64.2%, 17.0%,
and 1.8% respectively. Do not claim a layer is trainable until its own
coverage and split exist.

## D. Species + individual split

- Record a stable `individual_id` (tag, GPS + photo of the trunk, or site
  map). One individual = one genotype; do not merge nearby ramets.
- Minimum per species: **5 individuals**, ≥5 frames per individual across
  views; compound species ≥6 individuals.
- Splits are **species-disjoint** for generalization and **individual-disjoint**
  within a species. Never put two frames of one individual, or a burst, in
  different splits.
- Keep a **field-only test slice** per class; lab/pressed material is optional
  context and must be marked `domain: lab`.
- Hold out entire sites where possible, so the test measures place
  generalization, not just new individuals.

## E. Hard negatives

Negatives must be real captures, not synthetic montages:

| negative kind | examples | count target |
|---|---|---:|
| isolated leaflet | use 250 of the 500 isolated-leaflet frames above as designated hard negatives (not additive) | 250 of 500 |
| non-leaf organ | bud, flower, twig, bark, branch, whole plant | 300 (50 per organ) |
| clutter / occlusion | hand/glove, ground litter, sky, water, rock, structure, two overlapping leaves | 250 |
| degradation | severe blur, motion, glare, deep shadow, rain-on-lens | 100 |

Label them for the layer they challenge: `organ != leaf` for the gate;
`leaf_architecture = leaflet` or `leaf_view = isolated_leaflet` for the
compound specialist. A leaflet is **not** a simple leaf; it is a hard negative
for "whole compound".

## Minimum initial curated counts

These are **coverage floors** for a first curated pilot — enough to populate
every species/individual/view cell — not a sample-size calculation and not a
performance guarantee. No accuracy claim should be attached to reaching them.

| bucket | minimum frames | minimum species | minimum individuals / species | notes |
|---|---:|---:|---:|---|
| simple whole leaf (`whole_simple`) | 1,000 | 40 | 5 | field; varied sites/light |
| compound whole leaf (`whole_compound`) | 600 | 20 | 5 (6 preferred) | rachis complete; terminal leaflet |
| isolated leaflet (`isolated_leaflet`) | 500 | 20 | 5 | paired to compound frames via `pair_id` |
| non-leaf organs | 300 | 15 | 3 | 50 each: bud, flower, twig, bark, branch, entire plant |
| clutter / occlusion negatives | 250 | n/a | n/a | hands, ground, sky, water, overlapping leaves |
| degradation negatives | 100 | n/a | n/a | blur, motion, glare, shadow, wet lens |
| margin close-up (later layer) | 300 | 15 | 5 | 20 frames/species; scale reference |
| twig / winter bud (later layer) | 300 | 15 | 5 | dormant season |
| leaf reverse / venation (later layer) | 200 | 10 | 5 | paired with obverse via `pair_id` |
| **leaf-type pilot total (rows 1–6)** | **2,750** | — | — | gates a v3 leaf-type experiment |
| **later-layer total (rows 7–9)** | **800** | — | — | separate datasets |

If time is tight, do Phase 1 only (rows 1–6): 2,750 frames. Do not start a v3
training run with fewer than 600 whole-compound and 500 isolated-leaflet
frames, or without the individual IDs — the same view/label confound will
reappear.

## QA and provenance

- Double-label 100% of `whole_compound`, `isolated_leaflet`, and
  `compound_fragment` frames until inter-label agreement is stable, then 20%
  spot checks. A third adjudicator resolves disagreements; keep both original
  labels in the record.
- Reject a frame if the wrong individual, species, or organ is recorded;
  correct the label rather than deleting the evidence.
- Keep originals (EXIF, full resolution) and the curated derivatives
  separately; record the derivation command and hashes.
- Store under a new root (suggested
  `data/leafsnap-yolo26/curated-v1/{frames,labels.jsonl,individuals.json,splits.json}`)
  and build a **v3** dataset from it. Do not overwrite `classification` or
  `classification-v2`; they are the evidence for the current E4 FAIL.
- Never tune a confidence threshold on the curated test split; keep the
  validation-only selection rule used by E4.

## Explicitly out of scope for this round

- Species identification labels (the model does not identify species).
- Bark feature extraction (Dirr has no bark feature; GoBotany bark values are
  a separate provenance).
- Synthetic composites presented as field frames.
- Force-matching unmatched LeafSnap names, or removing species because they
  performed poorly on test.
