# LeafSnap x Dirr EDA — hierarchical field-vision pipeline

Training-free exploratory analysis of the decided hierarchical pipeline: 185 LeafSnap tree species joined to the Dirr descriptive vocabulary, with single-feature and joint (chordal) discriminative importance.

## Reproduce

```bash
cd /Users/valkyrie/dev/fieldguidevision-yolo26-3d
.venv/bin/python experiments/leafsnap-yolo26/eda.py
.venv/bin/python experiments/leafsnap-yolo26/test_eda.py
```

The run reads the LeafSnap manifest and the Dirr parquets, writes `eda_results.json`, `feature_assertions.json` and this report, and is deterministic (no timestamps, sorted output, fixed thresholds).

## Inputs

| input | path | sha256 | size |
|---|---|---|---|
| LeafSnap manifest | `/Users/valkyrie/Desktop/arborphy product/leafsnap/leafsnap-dataset/leafsnap-dataset-images.txt` | `d2bcc3878a77d691…` | 30,866 rows / 185 species |
| Dirr species tags | `/Users/valkyrie/dev/fieldguidevision-yolo26-3d/manual_of_woody_plants/extracted_vocabulary/parquet/dirr_species_tags.parquet` | `b650afac6d4f4da2…` | 969 rows |
| Dirr vocabulary | `/Users/valkyrie/dev/fieldguidevision-yolo26-3d/manual_of_woody_plants/extracted_vocabulary/parquet/dirr_vocabulary.parquet` | `a69144b9eaf3fd95…` | 97 value columns / 17 features |
| GoBotany (optional) | `gobotany/extracted_vocabulary/parquet/` | — | coverage / conflict check only |

## Headline findings

- **Join.** 165 of 185 LeafSnap species join to Dirr (89.2%): 151 exact + 14 audited orthographic/synonym/typo matches. 20 have no Dirr row and 0 are ambiguous.
- **Best single feature (probe IG, one observed value).** `leaf_surface_top` (Leaf Surface: Top): probe IG = 2.114 bits, probe NMI = 0.491, coverage = 63.6%; the guide's exact asserted-set upper bound is 3.268 bits. Median single-value candidate set = 45.0.
- **Best same-view pair.** `leaf_surface_top` x `leaf_shape_overall`: probe joint IG = 3.895 bits, probe synergy = 1.781 bits over the better single (n = 96).
- **Best non-local pair.** `leaf_shape_overall` (leaf_obverse) x `inflorescences` (flower): probe joint IG = 3.702 bits, probe synergy = 1.619 bits (n = 54). These cannot be read from one crop.
- **`leaf_type`.** Coverage 124/165 (75.2%); Simple = 101, Compound = 21, mixed = 2, unlabelled = 41. It is a coarse splitter, not a species discriminator: probe IG = 1.304 bits (exact-set upper bound 1.388 bits), and a single observed value leaves 103 (Simple) or 23 (Compound) candidates.
- **`leaf_type` conflicts.** 2 species assert Simple and Compound simultaneously: `Ginkgo biloba`, `Gymnocladus dioicus`. A Simple/Compound specialist must exclude or adjudicate them.
- **Bark is absent from Dirr.** No bark feature is mapped, so a leaf x bark non-local pair cannot be evaluated from the current Dirr parquets. GoBotany's bark-texture character covers 98 LeafSnap species and is the only available bark signal.
- **Adapter defect.** `DirrDevoAdapter` silently drops 49 assertions across 48 matched species (4 of 97 tag columns: Deltoid, Elliptic, Reticulate, Rhombic; affected features leaf_shape_overall, leaf_venation). This EDA uses a column-accurate mapping; the adapter should be fixed before the pipeline consumes it.

> **Fusion warning.** Non-local joint features are real information, but they are only available after fusing separate observations of the same plant (leaf obverse, leaf reverse, whole plant, flower). Never assume a non-local pair is simultaneously visible in one crop, and never train a single-crop model to emit both sides of one. Same-view pairs are the only joint features a single image can support.

## 1. Species join and image counts

| metric | value |
|---|---|
| LeafSnap rows | 30,866 |
| lab rows | 23,147 |
| field rows | 7,719 |
| LeafSnap species | 185 |
| matched to Dirr | 165 (89.2%) |
| unmatched | 20 |
| ambiguous | 0 |
| many-to-one | 0 |
| Dirr species-level rows | 967 |
| Dirr genus-only rows | 2 |

| join status | species | lab images | field images | total images |
|---|---|---|---|---|
| matched | 165 | 20,599 | 6,934 | 27,533 |
| not matched | 20 | 2,548 | 785 | 3,333 |

Match rules used: exact normalized name, normalized genus+epithet key (hyphen / hybrid-mark / spacing variants), infraspecific parent, curated synonym, one-character typo, truncated epithet. Every match must be unique; anything else is reported as ambiguous or unmatched.

**Matched but empty.** 6 matched species have a Dirr row with no asserted values at all: `Larix decidua`, `Malus floribunda`, `Malus hupehensis`, `Malus pumila`, `Pinus parviflora`, `Pinus peucea`. They stay in the denominator (coverage is deployment-honest) but cannot be keyed out by any feature. Dirr's `Malus` and `Clematis` entries are genus-only and cannot substitute for species-level identification; the matched `Malus` species rows are empty.

### Audited non-exact matches

| LeafSnap | Dirr | rule | confidence | note |
|---|---|---|---|---|
| `Aesculus hippocastamon` | `Aesculus hippocastanum` | curated_synonym | high | Misspelling of Aesculus hippocastanum in the LeafSnap manifest (three letters differ; no other Dirr row is close). |
| `Aesculus pavi` | `Aesculus pavia` | single_character_typo | medium | one-character spelling variant; no other Dirr row is within one edit |
| `Broussonettia papyrifera` | `Broussonetia papyrifera` | single_character_typo | medium | one-character spelling variant; no other Dirr row is within one edit |
| `Cladrastis lutea` | `Cladrastis kentukea` | curated_synonym | high | Cladrastis lutea is a synonym of C. kentukea (accepted name). |
| `Crataegus crus-galli` | `Crataegus crusgalli` | orthographic_variant | high | punctuation / hyphen / spacing variant of the same epithet |
| `Evodia daniellii` | `Tetradium daniellii` | curated_synonym | high | Evodia daniellii is the former name of Tetradium daniellii. |
| `Halesia tetraptera` | `Halesia carolina` | curated_synonym | medium | Halesia tetraptera and H. carolina are treated as the same species in Dirr's usage; epithet chosen by Dirr. |
| `Magnolia soulangiana` | `Magnolia xsoulangeana` | single_character_typo | medium | one-character spelling variant; no other Dirr row is within one edit |
| `Pinus peucea` | `Pinus peuce` | single_character_typo | medium | one-character spelling variant; no other Dirr row is within one edit |
| `Platanus acerifolia` | `Platanus xacerifolia` | hybrid_name_form | high | hybrid marker or spelling variant of the same epithet |
| `Prunus yedoensis` | `Prunus xyedoensis` | hybrid_name_form | high | hybrid marker or spelling variant of the same epithet |
| `Robinia pseudo-acacia` | `Robinia pseudoacacia` | orthographic_variant | high | punctuation / hyphen / spacing variant of the same epithet |
| `Styrax japonica` | `Styrax japonicus` | curated_synonym | high | Epithet gender agreement with the masculine genus Styrax. |
| `Tilia europaea` | `Tilia xeuropaea` | hybrid_name_form | high | hybrid marker or spelling variant of the same epithet |

### Unmatched LeafSnap species

| LeafSnap | diagnostic |
|---|---|
| `Acer ginnala` | suggested but not applied: Acer tataricum |
| `Asimina triloba` | no same-genus candidate within edit distance 2 |
| `Betula jacqemontii` | no same-genus candidate within edit distance 2 |
| `Carpinus betulus` | no same-genus candidate within edit distance 2 |
| `Celtis tenuifolia` | no same-genus candidate within edit distance 2 |
| `Crataegus pruinosa` | no same-genus candidate within edit distance 2 |
| `Fraxinus pennsylvanica` | no same-genus candidate within edit distance 2 |
| `Gleditsia triacanthos` | no same-genus candidate within edit distance 2 |
| `Ilex opaca` | no same-genus candidate within edit distance 2 |
| `Pinus cembra` | no same-genus candidate within edit distance 2 |
| `Pinus koraiensis` | no same-genus candidate within edit distance 2 |
| `Pinus nigra` | no same-genus candidate within edit distance 2 |
| `Pinus pungens` | no same-genus candidate within edit distance 2 |
| `Prunus pensylvanica` | no same-genus candidate within edit distance 2 |
| `Quercus michauxii` | no same-genus candidate within edit distance 2 |
| `Quercus velutina` | no same-genus candidate within edit distance 2 |
| `Salix caroliniana` | no same-genus candidate within edit distance 2 |
| `Salix matsudana` | no same-genus candidate within edit distance 2 |
| `Salix nigra` | no same-genus candidate within edit distance 2 |
| `Syringa reticulata` | no same-genus candidate within edit distance 2 |

No unmatched species is force-joined. The audited non-exact matches are listed above so a director can accept or reject them. Reconciliation with `DATASETS.md`: 151 exact matches, of which 93 are Simple-only and 16 Compound-only — the 109 species the dataset builder could use; the remaining exact matches are 2 mixed and 40 with no Simple/Compound label. The 14 audited name matches are additional and do not change that builder audit unless accepted.

## 2. Feature coverage (matched LeafSnap species, ordered by probe IG)

| feature | label | organ / view | values | coverage | multi-valued | used |
|---|---|---|---|---|---|---|
| `leaf_surface_top` | Leaf Surface: Top | leaf blade / leaf_obverse | 9 | 105/165 (63.6%) | 54.3% | 9 |
| `leaf_shape_overall` | Leaf Blade Shape: Overall | leaf blade / leaf_obverse | 15 | 127/165 (77.0%) | 75.6% | 11 |
| `leaf_shape_margins` | Leaf Blade Shape: Margins | leaf blade / leaf_obverse | 8 | 106/165 (64.2%) | 31.1% | 7 |
| `inflorescences` | Inflorescences | flower / flower | 10 | 61/165 (37.0%) | 23.0% | 9 |
| `leaf_shape_base` | Leaf Blade Shape: Base | leaf blade / leaf_obverse | 9 | 93/165 (56.4%) | 22.6% | 7 |
| `leaf_shape_apex` | Leaf Blade Shape: Apex | leaf blade / leaf_obverse | 9 | 94/165 (57.0%) | 4.3% | 5 |
| `leaf_arrangement` | Leaf Arrangement | leaf blade / leaf_obverse | 4 | 129/165 (78.2%) | 5.4% | 4 |
| `leaf_type` | Leaf Type | leaf blade / leaf_obverse | 2 | 124/165 (75.2%) | 1.6% | 2 |
| `growth_habit_types` | Growth Habit Types | whole plant / whole_plant | 4 | 86/165 (52.1%) | 19.8% | 2 |
| `leaf_surface_bottom` | Leaf Surface: Bottom | leaf blade / leaf_reverse | 7 | 28/165 (17.0%) | 3.6% | 7 |
| `leaf_type_conifer` | Conifer Leaf Types | leaf blade / leaf_obverse | 3 | 24/165 (14.5%) | 4.2% | 3 |
| `leaf_texture` | Leaf Texture | leaf blade / leaf_obverse | 4 | 19/165 (11.5%) | 5.3% | 4 |
| `compound_leaf_type` | Compound Leaf Type | leaf blade / leaf_obverse | 2 | 21/165 (12.7%) | 0.0% | 2 |
| `phenology_types` | Phenology Types | whole plant / whole_plant | 3 | 6/165 (3.6%) | 16.7% | 3 |
| `leaf_venation` | Leaf Venation | leaf blade / leaf_obverse | 4 | 3/165 (1.8%) | 0.0% | 2 |
| `leaf_shape_lobe_type` | Leaf Blade Shape: Lobe Type | leaf blade / leaf_obverse | 2 | 2/165 (1.2%) | 0.0% | 1 |
| `pinnate_types` | Pinnate Types | leaf blade / leaf_obverse | 2 | 1/165 (0.6%) | 0.0% | 1 |

Coverage is the ceiling on every other metric. `leaf_venation`, `leaf_texture`, `leaf_surface_bottom`, `leaf_type_conifer` and `phenology_types` are sparse in Dirr; a feature with high purity on 18% of species is not a general key. `compound_leaf_type` and `pinnate_types` are **conditional**: they apply only to compound (or bipinnate) species, so their 165-denominator coverage understates them — 19 of the 21 compound-labelled species carry a Pinnate/Palmate sub-type. Multi-valued rates matter too: a species with more than one asserted value cannot be treated as a single label.

## 3. Why these metrics

- **coverage** — Fraction of the candidate species for which the source asserts any value. A feature the guide does not document cannot key those species out, so coverage bounds every other metric.
- **information_gain** — IG(Y;F) = H(Y) - H(Y|F), in bits. How many bits of species identity the feature removes from a uniform species prior. Directly comparable to the log2(candidate-set) reduction and independent of feature alphabet size (unlike raw entropy of F).
- **nmi** — IG / sqrt(H(F) * H(Y)). Symmetric normalization so a feature with a large or small alphabet can be compared on a 0..1 scale; 1.0 means the feature partition determines species identity exactly.
- **conditional_entropy** — H(Y|F), in bits, and as a ratio of H(Y). The residual species uncertainty after reading the feature; the ratio is the share of the prior that survives, so lower is better and it is scale-free.
- **probe_information_gain** — IG under the single-observation model: each species' weight is split evenly over its asserted values, so the metric measures what ONE observed value tells you. This is the primary ranking key because the pipeline sees a single value at a time; the exact asserted-set IG is kept as the guide's upper bound. Where Dirr's values are co-observable (colour + lustre + indumentum in leaf_surface_top), probe IG is a conservative lower bound: a real observer can report several tags.
- **unique_resolution** — Fraction of species that end up alone in their feature cell. This is the operational question -- can this feature by itself name the species? Reported over covered species (fair to sparse features) and over all species (deployment-honest, unobserved = unresolved).
- **median_candidate_set** — Median number of species sharing a species' feature cell. The candidate set the next feature has to split. Median rather than mean because a few huge cells (e.g. all simple-leaved species) dominate the mean.
- **probe_candidate_set** — Candidate set when a single observed value is queried, averaged over the species' asserted values. Multi-valued Dirr rows make the exact cell look more specific than any real observation: a species asserting {Ovate, Elliptic} is alone in its exact cell but every single-value probe still returns the full Ovate group.
- **image_weighted** — The same metrics with each species weighted by its LeafSnap image count. Secondary only: it says what the dataset emphasises, not what the field guide can discriminate. Reported to expose the difference, never used for ranking.
- **joint_ig** — IG(Y;F1,F2) for the joint value-set cell. Upper bound on what the pair can contribute and the basis for the synergy test.
- **conditional_ig** — IG of one feature given the other = joint_IG - IG(other). The incremental information the second feature adds after the first; zero means the first already contains it.
- **synergy** — joint_IG - max(single_IG). Positive means the pair is worth more than its best member alone; zero or negative means redundancy, and the pair should not drive observation design.
- **redundancy** — IG(F1) + IG(F2) - joint_IG. Bits the two features share; high redundancy means they are proxies for the same distinction.

Ranking is by species-balanced **probe IG** (one observed value); every other metric is shown beside it so the ranking is inspectable rather than a score. The exact asserted-set IG is the guide's upper bound, and `H(Y|F)` / NMI in the table are the probe variants. Image-weighted numbers are secondary and never used to rank.

## 4. Top single features

| rank | feature | cover | probe IG | probe NMI | guide IG (upper bound) | probe H(Y\|F)/H(Y) | unique res (cov/all) | probe median cand | image-wt probe IG |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `leaf_surface_top` | 63.6% | 2.114 | 0.491 | 3.268 | 0.713 | 5.7% / 3.6% | 45.0 | 2.149 |
| 2 | `leaf_shape_overall` | 77.0% | 2.083 | 0.459 | 4.342 | 0.717 | 5.5% / 4.2% | 52.0 | 2.085 |
| 3 | `leaf_shape_margins` | 64.2% | 2.006 | 0.495 | 2.890 | 0.728 | 5.7% / 3.6% | 43.0 | 2.000 |
| 4 | `inflorescences` | 37.0% | 1.902 | 0.496 | 2.310 | 0.742 | 24.6% / 9.1% | 12.0 | 1.892 |
| 5 | `leaf_shape_base` | 56.4% | 1.890 | 0.489 | 2.492 | 0.743 | 7.5% / 4.2% | 36.0 | 1.881 |
| 6 | `leaf_shape_apex` | 57.0% | 1.489 | 0.446 | 1.612 | 0.798 | 3.2% / 1.8% | 80.0 | 1.464 |
| 7 | `leaf_arrangement` | 78.2% | 1.417 | 0.432 | 1.618 | 0.808 | 2.3% / 1.8% | 101.0 | 1.418 |
| 8 | `leaf_type` | 75.2% | 1.304 | 0.419 | 1.388 | 0.823 | 0.0% / 0.0% | 103.0 | 1.287 |
| 9 | `growth_habit_types` | 52.1% | 1.263 | 0.398 | 1.589 | 0.829 | 0.0% / 0.0% | 78.0 | 1.259 |
| 10 | `leaf_surface_bottom` | 17.0% | 1.066 | 0.379 | 1.090 | 0.855 | 10.7% / 1.8% | 8.0 | 1.015 |
| 11 | `leaf_type_conifer` | 14.5% | 0.698 | 0.306 | 0.729 | 0.905 | 8.3% / 1.2% | 21.0 | 0.692 |
| 12 | `leaf_texture` | 11.5% | 0.666 | 0.299 | 0.684 | 0.910 | 10.5% / 1.2% | 11.0 | 0.665 |

The top five features are within 11% of each other in probe IG (2.11 to 1.89 bits): no single Dirr feature is a species key. The hierarchy has to combine them.
`probe IG` is the primary ranking key: it is the information from ONE observed value, with each species' weight split over the values Dirr asserts. `guide IG` is the exact asserted-set upper bound (a perfect multi-label observation) and is shown only for contrast. `unique res` is the exact-cell fraction of species alone in their cell; the parenthesised second number counts unobserved species as unresolved. `probe median cand` is the candidate set a single observed value leaves. For `leaf_shape_overall` the exact cell looks near-unique (median 5) while a single shape leaves 52 candidates — the probe numbers are the deployment numbers. For `leaf_surface_top` the two bounds bracket the truth: its 9 values mix colour, lustre and indumentum, so one observation may return several tags (probe IG is a conservative lower bound) while the exact set assumes all of them are known at once (optimistic upper bound).

Secondary context (all 967 Dirr species-level rows, species-balanced probe IG): `leaf_shape_overall` 2.187 bits, `leaf_shape_margins` 2.010 bits, `leaf_surface_top` 1.958 bits, `leaf_shape_base` 1.696 bits, `leaf_shape_apex` 1.685 bits.

## 5. Joint (chordal) pairs

30 pairs met the eligibility rule (each feature covers >= 30% of matched species and shared coverage is >= 30%). 9 non_local, 21 same_view.

### Same-view pairs (fusable in one crop)

| pair | n | probe joint IG | probe synergy | probe cond IG (b\|a) | exact joint IG | probe unique res | probe median cand |
|---|---|---|---|---|---|---|---|
| `leaf_surface_top` x `leaf_shape_overall` | 96 | 3.895 | 1.781 | 1.781 | 6.115 | 3.1% | 17.7 |
| `leaf_surface_top` x `leaf_shape_margins` | 83 | 3.847 | 1.733 | 1.733 | 5.285 | 3.6% | 14.0 |
| `leaf_surface_top` x `leaf_shape_base` | 69 | 3.825 | 1.711 | 1.711 | 5.133 | 5.8% | 11.0 |
| `leaf_shape_overall` x `leaf_shape_margins` | 99 | 3.719 | 1.636 | 1.636 | 5.892 | 1.0% | 18.3 |
| `leaf_shape_margins` x `leaf_shape_base` | 75 | 3.683 | 1.677 | 1.677 | 4.766 | 1.3% | 12.0 |
| `leaf_shape_overall` x `leaf_shape_base` | 89 | 3.549 | 1.466 | 1.466 | 5.605 | 0.0% | 19.5 |
| `leaf_surface_top` x `leaf_shape_apex` | 74 | 3.363 | 1.249 | 1.249 | 4.434 | 10.8% | 22.0 |
| `leaf_shape_overall` x `leaf_shape_apex` | 88 | 3.321 | 1.238 | 1.238 | 5.257 | 2.3% | 31.3 |
| `leaf_surface_top` x `leaf_arrangement` | 100 | 3.237 | 1.123 | 1.123 | 4.393 | 2.0% | 29.0 |
| `leaf_shape_margins` x `leaf_shape_apex` | 74 | 3.231 | 1.225 | 1.225 | 4.044 | 2.7% | 22.0 |

### Non-local pairs (require fusion across observations / tracks)

| pair | n | probe joint IG | probe synergy | probe cond IG (b\|a) | exact joint IG | probe unique res | probe median cand |
|---|---|---|---|---|---|---|---|
| `leaf_shape_overall` x `inflorescences` | 54 | 3.702 | 1.619 | 1.619 | 5.556 | 0.0% | 4.5 |
| `leaf_surface_top` x `growth_habit_types` | 53 | 3.335 | 1.221 | 1.221 | 4.623 | 1.9% | 18.3 |
| `leaf_shape_overall` x `growth_habit_types` | 69 | 3.281 | 1.198 | 1.198 | 5.513 | 1.4% | 25.5 |
| `leaf_shape_margins` x `growth_habit_types` | 63 | 3.197 | 1.191 | 1.191 | 4.275 | 0.0% | 23.0 |
| `leaf_shape_base` x `growth_habit_types` | 52 | 3.120 | 1.230 | 1.230 | 3.914 | 0.0% | 16.1 |
| `inflorescences` x `leaf_arrangement` | 57 | 3.060 | 1.158 | 1.158 | 3.453 | 0.0% | 7.0 |
| `inflorescences` x `leaf_type` | 53 | 3.049 | 1.147 | 1.147 | 3.373 | 1.9% | 7.0 |
| `leaf_arrangement` x `growth_habit_types` | 71 | 2.626 | 1.209 | 1.209 | 3.072 | 0.0% | 46.0 |
| `leaf_type` x `growth_habit_types` | 67 | 2.558 | 1.254 | 1.254 | 2.924 | 0.0% | 53.0 |

All joint numbers use the same single-observation model as the single features: each species' weight is split over its asserted value combinations, so multi-valued rows cannot inflate a pair. `probe synergy` = joint probe IG - max(single probe IG) and is the only evidence that the second feature earns its observation; zero means one feature subsumes the other. `probe cond IG (b|a)` = joint probe IG - probe IG(a), the incremental bits of the alphabetically-second feature after the first (synergy is the smaller of the two directions). `exact joint IG` is the guide's asserted-set upper bound. For non-local pairs these bits exist only after fusing separate views — they are not a single-image model target.

### Structurally interesting but underpowered

| pair | views | n shared | probe joint IG | probe synergy | status |
|---|---|---|---|---|---|
| `leaf_surface_top` x `leaf_surface_bottom` | leaf_obverse / leaf_reverse | 26 | 2.940 | 0.826 | below coverage threshold — not ranked |
| `leaf_type` x `growth_habit_types` | leaf_obverse / whole_plant | 67 | 2.558 | 1.254 | below coverage threshold — not ranked |

## 6. Triples (small, feasibility check)

48 triples evaluated over the 9 best-covered features; every pair inside a triple also passed the pair rule. 0 fully non-local triple passed the 3-way coverage rule, so the top triples are leaf-view combinations plus at most one whole-plant feature. Triples are reported for feasibility, not as an observation plan.

| features | views | probe joint IG | probe marginal over best pair | probe synergy vs best single |
|---|---|---|---|---|
| `leaf_surface_top` x `leaf_shape_overall` x `leaf_shape_margins` | 1 (leaf_obverse) | 5.172 | 1.277 | 3.058 |
| `leaf_surface_top` x `leaf_shape_margins` x `leaf_shape_base` | 1 (leaf_obverse) | 5.146 | 1.299 | 3.032 |
| `leaf_surface_top` x `leaf_shape_overall` x `leaf_shape_base` | 1 (leaf_obverse) | 5.051 | 1.156 | 2.937 |
| `leaf_shape_overall` x `leaf_shape_margins` x `leaf_shape_base` | 1 (leaf_obverse) | 4.955 | 1.235 | 2.872 |
| `leaf_surface_top` x `leaf_shape_overall` x `growth_habit_types` | 2 (leaf_obverse, whole_plant) | 4.903 | 1.008 | 2.789 |

## 7. Feature spotlights

### `leaf_type` (decided specialist: validate labels, do not re-rank)

| class | species | lab images | field images | total images |
|---|---|---|---|---|
| Simple | 101 | 13,040 | 4,100 | 17,140 |
| Compound | 21 | 2,463 | 917 | 3,380 |
| mixed | 2 | 240 | 49 | 289 |
| unlabelled | 41 | 4,856 | 1,868 | 6,724 |

`Simple` and `Compound` are the specialist classes; `mixed` and `unlabelled` are exclusion statuses, not classes.

- Value discrimination (species per asserted value): **Simple** 103, **Compound** 23.
- Probe IG = 1.304 bits (exact asserted-set upper bound 1.388 bits); probe NMI = 0.419; probe H(Y|F) = 6.063 bits (82.3% of H(Y)); confirmed-unique probe rate = 0.0% (a Simple/Compound split never isolates a species).
- The Simple group is 103 species (Simple-only plus the mixed rows) and the Compound group is 23; 41 matched species have no Simple/Compound label (mostly conifers).
- Compound sub-type coverage: 19/21 compound species (90.5%) carry Pinnate/Palmate, so a second-stage compound split is only partly labelled.
- **Conflicts:** `Ginkgo biloba`, `Gymnocladus dioicus` assert both Simple and Compound. These are the rows the specialist must exclude or adjudicate; they are not a reason to choose another specialist.

### `leaf_arrangement`

Coverage 129/165 (78.2%); multi-valued 5.4%; probe IG 1.417 bits (guide upper bound 1.618); probe NMI 0.432; median single-value candidate set 101.0; unique resolution 2.3% covered / 1.8% all.

Vocabulary note: Four states, dominated by Alternate, so it is a coarse splitter rather than a key.

Top values: **Alternate** 101, **Opposite** 30, **Subopposite** 4, **Whorled** 2.

Exclusive-value conflicts (may reflect within-species variation): `Broussonettia papyrifera` (Opposite + Alternate), `Catalpa bignonioides` (Opposite + Whorled), `Catalpa speciosa` (Opposite + Whorled), `Chionanthus retusus` (Opposite + Alternate).

### `leaf_shape_margins`

Coverage 106/165 (64.2%); multi-valued 31.1%; probe IG 2.006 bits (guide upper bound 2.890); probe NMI 0.495; median single-value candidate set 43.0; unique resolution 5.7% covered / 3.6% all.

Vocabulary note: Mixes tooth type (Serrate, Dentate, Crenate, Doubly-Serrate), absence (Entire) and lobing/waviness (Lobed, Sinuate, Undulate). Entire + Lobed is a legitimate combination (entire lobe margins), not a conflict.

Top values: **Serrate** 51, **Entire** 43, **Lobed** 30, **Crenate** 9, **Dentate** 7, **Doubly-Serrate** 3, **Sinuate** 2.

Exclusive-value conflicts (may reflect within-species variation): `Chionanthus retusus` (Entire + Serrate), `Crataegus crus-galli` (Entire + Serrate), `Diospyros virginiana` (Entire + Serrate), `Populus deltoides` (Entire + Dentate), `Quercus virginiana` (Entire + Crenate), `Toona sinensis` (Entire + Serrate), `Ulmus pumila` (Entire + Serrate).

### `leaf_venation`

Coverage 3/165 (1.8%); multi-valued 0.0%; probe IG 0.148 bits (guide upper bound 0.148); probe NMI 0.142; median single-value candidate set 2.0; unique resolution 33.3% covered / 0.6% all.

Vocabulary note: Four venation states in the vocabulary, but Dirr asserts them for very few species; at this coverage venation cannot be a primary key.
**Low coverage (<10%): 1.8% — this feature cannot function as a primary key on this dataset.

Top values: **Reticulate** 2, **Parallel** 1.

### `leaf_surface_top`

Coverage 105/165 (63.6%); multi-valued 54.3%; probe IG 2.114 bits (guide upper bound 3.268); probe NMI 0.491; median single-value candidate set 45.0; unique resolution 5.7% covered / 3.6% all.

Vocabulary note: Mixes three axes in one feature: colour (Dark/Medium/Bright Green, Bluish, Grayish), lustre (Lustrous, Dull) and indumentum/other surface (Glabrous, Pubescent, Pale, Whitish, Silvery, Yellowish, Powdery, Woolly, Shaggy). One value is not one visual predicate, and 54% of its covered species assert more than one value (second only to leaf_shape_overall).

Top values: **Dark Green** 63, **Lustrous** 40, **Glabrous** 34, **Pubescent** 27, **Bluish** 4, **Medium Green** 4, **Grayish** 3, **Bright Green** 2.

### `leaf_surface_bottom`

Coverage 28/165 (17.0%); multi-valued 3.6%; probe IG 1.066 bits (guide upper bound 1.090); probe NMI 0.379; median single-value candidate set 8.0; unique resolution 10.7% covered / 1.8% all.

Vocabulary note: Indumentum / wax axis only (Pale, Whitish, Silvery, Yellowish, Powdery, Woolly, Shaggy). It is the abaxial face, so it requires a separate observation from leaf_surface_top.

Top values: **Pale** 8, **Powdery** 8, **Woolly** 6, **Silvery** 2, **Whitish** 2, **Yellowish** 2, **Shaggy** 1.

### `growth_habit_types` (whole plant)

Coverage 86/165 (52.1%); multi-valued 19.8%; probe IG 1.263 bits (guide upper bound 1.589); probe NMI 0.398; median single-value candidate set 78.0; unique resolution 0.0% covered / 0.0% all.

Vocabulary note: Tree / Shrub / Vine / Ground Cover. Only Tree and Shrub occur in the matched set; habit is a coarse filter with no unique resolution.

Top values: **Tree** 78, **Shrub** 25.

### `inflorescences` (flower)

Coverage 61/165 (37.0%); multi-valued 23.0%; probe IG 1.902 bits (guide upper bound 2.310); probe NMI 0.496; median single-value candidate set 12.0; unique resolution 24.6% covered / 9.1% all.

Vocabulary note: 10 inflorescence types. The most specific single feature per covered species, but it needs a flower observation and covers only about a third of the matched species.

Top values: **Panicle** 19, **Catkin** 13, **Solitary** 12, **Raceme** 9, **Corymb** 8, **Head** 6, **Umbel** 5, **Cyme** 4.

### `bark`

The Dirr extracted vocabulary has no bark feature (no value column maps to bark). Leaf x bark fusion is therefore not testable from the current Dirr parquets; it would need a new Dirr bark extraction or a GoBotany crossover that keeps provenance separate.

## 8. GoBotany coverage / conflict check (provenance separate)

GoBotany covers 98 of 185 LeafSnap species (53.0%) at the taxon level; comparisons below only use species where BOTH sources give exactly one comparable value.

| feature | GoBotany character | GoBotany coverage | compared | agree | conflict | agreement |
|---|---|---|---|---|---|---|
| leaf_type | `leaf_type_general_wa` | 98 (53.0%) | 83 | 83 | 0 | 100.0% |
| leaf_arrangement | `leaf_arrangement_wa` | 98 (53.0%) | 82 | 80 | 2 | 97.6% |
| leaf_shape_margins | `leaf_blade_margin_general_wa` | 98 (53.0%) | 48 | 41 | 7 | 85.4% |
| leaf_venation | `leaf_blade_venation_wa` | 67 (36.2%) | 0 | 0 | 0 | - |

**leaf_arrangement conflicts** (Dirr vs GoBotany): `Magnolia tripetala` Alternate vs Whorled; `Magnolia virginiana` Alternate vs Opposite.
**leaf_shape_margins conflicts** (Dirr vs GoBotany): `Acer campestre` entire vs lobed; `Amelanchier canadensis` entire vs toothed; `Carya glabra` entire vs toothed; `Castanea dentata` entire vs toothed; `Phellodendron amurense` entire vs toothed; `Quercus alba` entire vs lobed; `Quercus stellata` entire vs lobed.

View assignment is a heuristic from each feature's definition: leaf_obverse / leaf_reverse are different faces of the same leaf, flower is the reproductive structure, and whole_plant is habit or phenology. A whole-plant photo can show leaf arrangement at a distance, but it cannot resolve the close-up blade traits, so leaf x whole_plant is non-local for this pipeline. The margin comparison is coarse (entire / lobed / toothed): a Dirr `Entire` row and a GoBotany `lobed` row can describe the same leaf at different granularity (lobed blades often have entire lobe margins), so those conflicts are not proof of a data error. Venation compares zero species because Dirr's venation assertions are too sparse and `Reticulate` has no GoBotany analogue.

GoBotany's bark character values: close and smooth (38), exfoliating (14), ridged or plated (77). These are not merged into Dirr assertions.

## 9. Data quality and adapter defects

- 4 Dirr vocabulary columns carry no feature metadata: `Deltoid`, `Elliptic`, `Reticulate`, `Rhombic`. The EDA maps them explicitly; `DirrDevoAdapter.list_features` drops them.
- 10 tag columns use a disambiguating suffix, e.g. `Acute (apex)`, `Acute (base)`, `Palmate (compound)`, `Palmate (venation)`, `Pinnate (compound)`, `Pinnate (venation)`. The adapter currently keys by the column name, which is safe; a consumer that keys by the vocabulary `value_label` row instead would misattribute them (the labels repeat across features).
- Duplicated value labels across features: `Acute` -> leaf_shape_apex:Acute (apex), leaf_shape_base:Acute (base); `Palmate` -> compound_leaf_type:Palmate (compound), leaf_venation:Palmate (venation); `Pinnate` -> compound_leaf_type:Pinnate (compound), leaf_venation:Pinnate (venation); `Rounded` -> leaf_shape_apex:Rounded (apex), leaf_shape_base:Rounded (base); `Truncate` -> leaf_shape_apex:Truncate (apex), leaf_shape_base:Truncate (base).
- Adapter fidelity: `DirrDevoAdapter.get_taxon` loses 49 assertions across 48 matched species (4/97 columns). Affected features: `leaf_shape_overall`, `leaf_venation`.
- Exclusive-value checks flagged 12 species across the explicit rules; see `eda_results.json -> conflicts` for the full list. `Entire` + `Lobed` is deliberately not flagged: lobed blades can have entire lobe margins.

## 10. Warnings and limits

- **fusion-required** (critical) — Joint features MUST be fused across observations / spatial tracks. A same-view pair can share one crop; a non-local pair (leaf obverse vs leaf reverse, leaf vs whole plant, leaf vs flower) cannot. Never assume a non-local pair is visible in one image, and never train a single-crop model to predict both sides of one.
- **missing-is-not-a-value** (high) — An absent Dirr assertion is a documentation gap, not a biological value. Entropy treats it as its own cell and the deployment candidate set falls back to the full space, so sparse features are penalised rather than flattered.
- **multi-valued-optimism** (high) — Exact-value-set cells overstate resolution for multi-valued features. Compare resolution_rate_* with probe_* rates; the probe numbers describe what a single observed value returns. For features whose values are co-observable (colour + lustre + indumentum), the probe is a lower bound rather than a point estimate; the exact set is the upper bound. The truth sits between them and only the imaging model can settle it.
- **small-samples** (medium) — Pairs and triples with few shared species are noisy; n_shared is reported beside every joint metric. Underpowered spotlights are labelled rather than ranked.
- **image-weighting** (medium) — LeafSnap image counts are lab-heavy and species-unbalanced. Image-weighted metrics are secondary context only.
- **provenance** (high) — Dirr is the primary vocabulary. GoBotany values appear only in the coverage/conflict check, are labelled by source, and are never merged into the Dirr assertions.

Additional limits: Dirr's vocabulary is US-ornamental-woody oriented, so absent species are a source-coverage fact, not a taxonomy error; the Dirr-universe metrics are context for the eventual guide, while all ranked tables use the 165 matched LeafSnap species; leaf_surface_bottom is too sparse to rank as a pair (17% coverage, 26 shared species), so its top/bottom non-local contribution is reported as an underpowered spotlight rather than a recommendation. The GoBotany margin comparison is coarse (entire / lobed / toothed): `Entire` vs `lobed` can describe the same leaf at different granularity and is not proof of a data conflict. Dirr venation has no GoBotany-comparable unambiguous pair, so that row compares zero species.

## 11. Artifacts

| artifact | contents |
|---|---|
| `eda_results.json` | all metrics, join audit, pairs, triples, spotlights, warnings |
| `feature_assertions.json` | species -> feature -> values with provenance and match audit |
| `EDA.md` | this report |
| `eda.py` | reproducible analysis |
| `test_eda.py` | toy self-checks for entropy / candidate-set / join math |

