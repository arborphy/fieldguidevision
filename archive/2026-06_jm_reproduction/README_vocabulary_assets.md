# Descriptive Vocabulary Assets

This folder hands the interns the **arq-refdata equivalents of John Matter's
Spring-2026 input files** — nothing more, nothing less. Each remaining file
here corresponds directly to something John used in his thesis experiment.

If you have read John's thesis and worked with `../JM_Assets/`, this folder is
the same five conceptual inputs, but extracted with Margaret's broader pipeline
and across three reference sources instead of one.

## What John used vs. what is here

John's Spring experiment ran against Newcomb only, with five hand-curated CSVs
and a folder of 11 illustration PNGs (`../JM_Assets/data/`). The arq-refdata
pipeline produces the same conceptual pieces — vocabulary, references with
illustrations, species × feature assignments, and (separately) field
observations — but as Parquet tables, across three sources.

| John's Spring asset (`../JM_Assets/data/`)        | Equivalent here                                                              |
| ------------------------------------------------- | ---------------------------------------------------------------------------- |
| `feature_value_pairs.csv` — every (feature, value) Newcomb defines | `<source>/extracted_vocabulary/parquet/*_vocabulary.parquet` (Dirr); `<source>/extracted_vocabulary/parquet/*_feature*.parquet` + `*_character*.parquet` (Newcomb, GoBotany) |
| `references.csv` — definition + path to illustration for each value | `<source>/extracted_vocabulary/parquet/` + `<source>/illustrations/label_map.csv` |
| `newcomb_preprocessed.csv` — species × key-path assignments | `<source>/extracted_vocabulary/parquet/*_species_tags.parquet` (Dirr); `*_taxon*.parquet` (Newcomb, GoBotany) |
| `sample.csv` — species paired with iNat photo observations | `observations_pound_ridge/` (site-scoped, current; replaces the Spring slice John used) |
| `illustrations/*.png` — labeled diagram crops | `<source>/illustrations/*.png` + `label_map.csv` |

What's intentionally **not** here: raw OCR pages, source-traceable JSON, scan
intermediates, glossary dumps, editorial provenance docs, work-plans. Those
live in `arq-refdata` and are the build inputs Margaret used to *produce* the
files you see in this folder — they are not what John worked with, and they
are not what the summer's experiments need.

## Sources

| Folder                       | Source                                                | Status                                                     |
| ---------------------------- | ----------------------------------------------------- | ---------------------------------------------------------- |
| `manual_of_woody_plants/`    | Dirr's *Manual of Woody Landscape Plants*, 6th ed.    | Parquets + 104 labeled illustration crops                  |
| `newcomb_wildflower_guide/`  | Newcomb's *Wildflower Guide* (Lawrence Newcomb, 1977) | Parquets + key/locator CSVs; illustrations not yet extracted |
| `gobotany/`                  | GoBotany (New England Wild Flower Society, live API)  | Parquets + character CSVs; illustrations not mirrored      |

## Observations (the other half of the picture)

Vocabulary tells you *what features exist and which species have them*.
Observations tell you *where and when each species was actually seen*. John's
Spring experiment used a small per-species iNat photo sample (`sample.csv`);
the summer experiments use the latest site-scoped pull from Ward Pound Ridge.

| Folder                       | Source                                                | Status                                                     |
| ---------------------------- | ----------------------------------------------------- | ---------------------------------------------------------- |
| `observations_pound_ridge/`  | iNaturalist field observations at Ward Pound Ridge Reservation, NY (direct iNat export + GBIF mirror) | 11,881 iNat records + 4,225 GBIF records |

See that folder's README for the join pattern between observations and any
source's species-tag parquet.

## Folder layout per source

Every source folder has the same two compartments. They correspond directly to
the two kinds of JM input (the structured CSVs, and the illustration PNGs +
label map).

```
<source>/
├── extracted_vocabulary/
│   ├── parquet/         ← downstream contract: the (feature, value, definition)
│   │                      tables and the species × feature-value assignment matrix
│   └── *.csv            ← (Newcomb, GoBotany) lightweight CSV peeks at the same data
│
└── illustrations/
    ├── *.png            ← labeled diagram crops
    └── label_map.csv    ← filename → (feature, value) mapping
                          (placeholder README where extraction is not yet done)
```

## Where to start

1. Browse **`manual_of_woody_plants/`** — most complete. The Dirr parquets and
   illustration set are the model. Open
   `extracted_vocabulary/parquet/dirr_vocabulary.parquet` and
   `dirr_species_tags.parquet` side by side; this is the
   `feature_value_pairs.csv` + `newcomb_preprocessed.csv` analogue, scaled up.

2. Compare to **`newcomb_wildflower_guide/`** — same source John used. The
   parquets cover Newcomb's branching key and species-route assignments; the
   illustration extraction is the natural first summer task.

3. Compare to **`gobotany/`** — same conceptual shape, different source
   structure (live API, 5-layer model). Useful for thinking about how the
   pattern generalises beyond a single book.

4. Look at **`observations_pound_ridge/`** — pair with any source's
   `*_species_tags` parquet to land at the John's-`sample.csv` analogue at
   site scale.
