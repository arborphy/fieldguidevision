#!/usr/bin/env python3
# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""LeafSnap x Dirr EDA for the decided hierarchical field-vision pipeline.

Scope
-----
Reproducible, training-free exploratory analysis that answers one question:
*given the 185 LeafSnap tree species, what can Dirr's descriptive vocabulary
actually separate, and which features must be fused across observations?*

It writes three artifacts next to this file:

- ``eda_results.json``      -- all metrics, machine readable.
- ``feature_assertions.json`` -- the species x feature -> {values} table with
  provenance, the join audit, and the feature metadata.
- ``EDA.md``                -- the human report with the required tables.

Design decisions (see EDA.md for the prose version)
---------------------------------------------------
1. Species-balanced is the PRIMARY weight: every matched LeafSnap species has
   weight 1/N. Image counts are a SECONDARY, explicitly-labelled weighting
   (``image_weighted`` blocks); they measure dataset emphasis, not biology.
2. Features are categorical *sets* of asserted values. A species that asserts
   two values for a feature is multi-valued; the entropy partition uses the
   whole asserted set as the cell, and a separate "probe" analysis measures
   what happens when a single value is actually observed.
3. Unobserved (no Dirr assertion) is its own partition cell, so a sparse
   feature is penalised for the species it cannot key out. The diagnostic
   ``ig_covered`` restricts to covered species for comparison.
4. Metrics: information gain IG(Y;F) = H(Y) - H(Y|F), normalized mutual
   information NMI = IG / sqrt(H(F) H(Y)), conditional entropy H(Y|F), exact
   cell resolution / candidate-set size, and probe resolution / candidate-set
   size. Each has a stated reason in ``METRIC_DEFINITIONS``.
5. Joint ("chordal") importance is computed for every eligible pair with
   sufficient shared coverage: joint IG, conditional IG of one given the
   other, synergy over the better single feature, joint candidate sets, and a
   same-view vs non-local split. Non-local pairs are features whose required
   views are disjoint (leaf obverse / leaf reverse / flower / whole plant);
   they cannot be assessed in one crop and must be fused across observations.

No model is trained and no repository file outside this directory is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

DEFAULT_MANIFEST = Path(
    "/Users/valkyrie/Desktop/arborphy product/leafsnap/leafsnap-dataset/"
    "leafsnap-dataset-images.txt"
)
DEFAULT_DIRR_DIR = REPO_ROOT / "manual_of_woody_plants" / "extracted_vocabulary" / "parquet"
DEFAULT_GOBOTANY_DIR = REPO_ROOT / "gobotany" / "extracted_vocabulary" / "parquet"

# Eligibility thresholds (species-balanced, of the matched LeafSnap species).
MIN_FEATURE_COVERAGE = 0.30
MIN_PAIR_SHARED_COVERAGE = 0.30
MIN_TRIPLE_SHARED_COVERAGE = 0.30
MAX_TRIPLE_FEATURES = 14

UNOBSERVED = "unobserved"


# ---------------------------------------------------------------------------
# Vocabulary metadata: organ, view, metric definitions
# ---------------------------------------------------------------------------

# Mirrors segmentation/devo_adapter._DIRR_FEATURE_ORGAN. Kept local so the EDA
# records what it used; adapter_fidelity() reports any drift.
DIRR_FEATURE_ORGAN: dict[str, str] = {
    "growth_habit_types": "whole plant",
    "phenology_types": "whole plant",
    "leaf_arrangement": "leaf blade",
    "leaf_type": "leaf blade",
    "compound_leaf_type": "leaf blade",
    "leaf_venation": "leaf blade",
    "leaf_shape_overall": "leaf blade",
    "leaf_shape_margins": "leaf blade",
    "leaf_shape_lobe_type": "leaf blade",
    "leaf_shape_base": "leaf blade",
    "leaf_shape_apex": "leaf blade",
    "leaf_surface_top": "leaf blade",
    "leaf_surface_bottom": "leaf blade",
    "leaf_texture": "leaf blade",
    "leaf_type_conifer": "leaf blade",
    "pinnate_types": "leaf blade",
    "inflorescences": "flower",
}

# Observation view required to assess a feature. Derived from the organ mapping
# plus the feature's own definition ("Leaf Surface: Bottom" is the abaxial
# face; growth habit and phenology need the whole plant in frame).
FEATURE_VIEWS: dict[str, tuple[str, ...]] = {
    "leaf_arrangement": ("leaf_obverse",),
    "leaf_type": ("leaf_obverse",),
    "compound_leaf_type": ("leaf_obverse",),
    "leaf_venation": ("leaf_obverse",),
    "leaf_shape_overall": ("leaf_obverse",),
    "leaf_shape_margins": ("leaf_obverse",),
    "leaf_shape_lobe_type": ("leaf_obverse",),
    "leaf_shape_base": ("leaf_obverse",),
    "leaf_shape_apex": ("leaf_obverse",),
    "leaf_surface_top": ("leaf_obverse",),
    "leaf_texture": ("leaf_obverse",),
    "leaf_type_conifer": ("leaf_obverse",),
    "pinnate_types": ("leaf_obverse",),
    "leaf_surface_bottom": ("leaf_reverse",),
    "inflorescences": ("flower",),
    "growth_habit_types": ("whole_plant",),
    "phenology_types": ("whole_plant",),
}

# Qualitative reading of what each Dirr feature actually mixes. These are
# descriptions of the vocabulary (not rankings); every number quoted here is
# computed and checked in the report.
FEATURE_INTERPRETATION: dict[str, str] = {
    "leaf_surface_top": (
        "Mixes three axes in one feature: colour (Dark/Medium/Bright Green, "
        "Bluish, Grayish), lustre (Lustrous, Dull) and indumentum/other surface "
        "(Glabrous, Pubescent, Pale, Whitish, Silvery, Yellowish, Powdery, "
        "Woolly, Shaggy). One value is not one visual predicate, and 54% of its "
        "covered species assert more than one value (second only to "
        "leaf_shape_overall)."
    ),
    "leaf_surface_bottom": (
        "Indumentum / wax axis only (Pale, Whitish, Silvery, Yellowish, "
        "Powdery, Woolly, Shaggy). It is the abaxial face, so it requires a "
        "separate observation from leaf_surface_top."
    ),
    "leaf_shape_overall": (
        "15 blade shapes. Most covered species assert more than one, so the "
        "exact value-set cell overstates specificity; the single-value probe is "
        "the deployment number."
    ),
    "leaf_shape_margins": (
        "Mixes tooth type (Serrate, Dentate, Crenate, Doubly-Serrate), absence "
        "(Entire) and lobing/waviness (Lobed, Sinuate, Undulate). Entire + Lobed "
        "is a legitimate combination (entire lobe margins), not a conflict."
    ),
    "leaf_arrangement": (
        "Four states, dominated by Alternate, so it is a coarse splitter rather "
        "than a key."
    ),
    "growth_habit_types": (
        "Tree / Shrub / Vine / Ground Cover. Only Tree and Shrub occur in the "
        "matched set; habit is a coarse filter with no unique resolution."
    ),
    "inflorescences": (
        "10 inflorescence types. The most specific single feature per covered "
        "species, but it needs a flower observation and covers only about a "
        "third of the matched species."
    ),
    "leaf_venation": (
        "Four venation states in the vocabulary, but Dirr asserts them for very "
        "few species; at this coverage venation cannot be a primary key."
    ),
    "leaf_type": (
        "Simple / Compound. The decided specialist: a coarse, high-coverage "
        "splitter, validated on label coverage and conflicts, not re-ranked."
    ),
    "compound_leaf_type": (
        "Pinnate / Palmate. Conditional on leaf_type = Compound; read its "
        "coverage against the compound subset, not the full species list."
    ),
    "pinnate_types": (
        "Bipinnate / Tripinnate. Conditional on a compound leaf and asserted for "
        "one matched species; not usable as a key."
    ),
    "leaf_texture": (
        "Leathery / Rugose / Waxy / Glandular; sparse coverage in the matched "
        "set."
    ),
    "leaf_type_conifer": (
        "Needle-like / Awl-like / Scale-like. Mostly gymnosperms that have no "
        "Simple/Compound label, so it is complementary rather than overlapping."
    ),
    "phenology_types": (
        "Evergreen / Deciduous / Semievergreen; very sparse coverage in the "
        "matched set."
    ),
}

VIEW_DESCRIPTIONS = {
    "leaf_obverse": "upper leaf blade surface / whole blade in one frame",
    "leaf_reverse": "lower leaf blade surface (abaxial face)",
    "flower": "reproductive structure / inflorescence",
    "whole_plant": "whole-plant habit or phenology in frame",
}

METRIC_DEFINITIONS: dict[str, str] = {
    "coverage": (
        "Fraction of the candidate species for which the source asserts any "
        "value. A feature the guide does not document cannot key those species "
        "out, so coverage bounds every other metric."
    ),
    "information_gain": (
        "IG(Y;F) = H(Y) - H(Y|F), in bits. How many bits of species identity "
        "the feature removes from a uniform species prior. Directly comparable "
        "to the log2(candidate-set) reduction and independent of feature "
        "alphabet size (unlike raw entropy of F)."
    ),
    "nmi": (
        "IG / sqrt(H(F) * H(Y)). Symmetric normalization so a feature with a "
        "large or small alphabet can be compared on a 0..1 scale; 1.0 means "
        "the feature partition determines species identity exactly."
    ),
    "conditional_entropy": (
        "H(Y|F), in bits, and as a ratio of H(Y). The residual species "
        "uncertainty after reading the feature; the ratio is the share of the "
        "prior that survives, so lower is better and it is scale-free."
    ),
    "unique_resolution": (
        "Fraction of species that end up alone in their feature cell. This is "
        "the operational question -- can this feature by itself name the "
        "species? Reported over covered species (fair to sparse features) and "
        "over all species (deployment-honest, unobserved = unresolved)."
    ),
    "median_candidate_set": (
        "Median number of species sharing a species' feature cell. The "
        "candidate set the next feature has to split. Median rather than mean "
        "because a few huge cells (e.g. all simple-leaved species) dominate "
        "the mean."
    ),
    "probe_candidate_set": (
        "Candidate set when a single observed value is queried, averaged over "
        "the species' asserted values. Multi-valued Dirr rows make the exact "
        "cell look more specific than any real observation: a species "
        "asserting {Ovate, Elliptic} is alone in its exact cell but every "
        "single-value probe still returns the full Ovate group."
    ),
    "probe_information_gain": (
        "IG under the single-observation model: each species' weight is split "
        "evenly over its asserted values, so the metric measures what ONE "
        "observed value tells you. This is the primary ranking key because the "
        "pipeline sees a single value at a time; the exact asserted-set IG is "
        "kept as the guide's upper bound. Where Dirr's values are co-observable "
        "(colour + lustre + indumentum in leaf_surface_top), probe IG is a "
        "conservative lower bound: a real observer can report several tags."
    ),
    "image_weighted": (
        "The same metrics with each species weighted by its LeafSnap image "
        "count. Secondary only: it says what the dataset emphasises, not what "
        "the field guide can discriminate. Reported to expose the difference, "
        "never used for ranking."
    ),
    "joint_ig": (
        "IG(Y;F1,F2) for the joint value-set cell. Upper bound on what the "
        "pair can contribute and the basis for the synergy test."
    ),
    "conditional_ig": (
        "IG of one feature given the other = joint_IG - IG(other). The "
        "incremental information the second feature adds after the first; "
        "zero means the first already contains it."
    ),
    "synergy": (
        "joint_IG - max(single_IG). Positive means the pair is worth more "
        "than its best member alone; zero or negative means redundancy, and "
        "the pair should not drive observation design."
    ),
    "redundancy": (
        "IG(F1) + IG(F2) - joint_IG. Bits the two features share; high "
        "redundancy means they are proxies for the same distinction."
    ),
}

# Four Dirr value columns are emitted without vocabulary metadata in
# dirr_vocabulary.parquet (empty feature_name/label/section). They are still
# asserted in dirr_species_tags.parquet. The overrides are positional and
# semantic: Reticulate sits inside the venation block, and Elliptic/Deltoid/
# Rhombic sit inside the blade-shape block.
VOCAB_FEATURE_OVERRIDES: dict[str, tuple[str, str]] = {
    "Reticulate": (
        "leaf_venation",
        "column emitted without vocab metadata; sits between Parallel and the "
        "blade-shape block and is one of Dirr's four venation values",
    ),
    "Elliptic": (
        "leaf_shape_overall",
        "column emitted without vocab metadata; sits inside the Ovate..Reniform "
        "blade-shape block",
    ),
    "Deltoid": (
        "leaf_shape_overall",
        "column emitted without vocab metadata; sits inside the Ovate..Reniform "
        "blade-shape block",
    ),
    "Rhombic": (
        "leaf_shape_overall",
        "column emitted without vocab metadata; sits inside the Ovate..Reniform "
        "blade-shape block",
    ),
}

# Deliberate, auditable aliases. Every one is listed in EDA.md for review.
SYNONYMS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("cladrastis", "lutea"): (
        "Cladrastis kentukea",
        "high",
        "Cladrastis lutea is a synonym of C. kentukea (accepted name).",
    ),
    ("evodia", "daniellii"): (
        "Tetradium daniellii",
        "high",
        "Evodia daniellii is the former name of Tetradium daniellii.",
    ),
    ("halesia", "tetraptera"): (
        "Halesia carolina",
        "medium",
        "Halesia tetraptera and H. carolina are treated as the same species "
        "in Dirr's usage; epithet chosen by Dirr.",
    ),
    ("styrax", "japonica"): (
        "Styrax japonicus",
        "high",
        "Epithet gender agreement with the masculine genus Styrax.",
    ),
    ("aesculus", "hippocastamon"): (
        "Aesculus hippocastanum",
        "high",
        "Misspelling of Aesculus hippocastanum in the LeafSnap manifest "
        "(three letters differ; no other Dirr row is close).",
    ),
}

# Candidates that are botanically plausible but lose specificity. Reported as
# suggestions, never auto-applied.
SUGGESTIONS: dict[str, tuple[str, str]] = {
    "Acer ginnala": (
        "Acer tataricum",
        "Acer ginnala is treated as Acer tataricum subsp. ginnala; the Dirr "
        "row is the parent species, whose leaf shape differs, so the mapping "
        "would lose the three-lobed Amur maple distinction.",
    ),
}

# Explicit exclusive-value rules. Only pairs that describe incompatible
# states of ONE specimen are listed. Everything else multi-valued is reported
# as multi-valued, not as a conflict. Note: Entire + Lobed is deliberately NOT
# here -- a lobed blade can have entire lobe margins, and that combination is
# common in Dirr (Acer rubrum, Quercus nigra, Sassafras albidum).
CONTRADICTION_RULES: dict[str, list[tuple[str, str]]] = {
    "leaf_type": [("Simple", "Compound")],
    "leaf_arrangement": [
        ("Opposite", "Alternate"),
        ("Opposite", "Whorled"),
        ("Alternate", "Whorled"),
    ],
    "leaf_shape_margins": [
        ("Entire", "Serrate"),
        ("Entire", "Dentate"),
        ("Entire", "Crenate"),
        ("Entire", "Doubly-Serrate"),
    ],
    "compound_leaf_type": [("Pinnate (compound)", "Palmate (compound)")],
}

GOBOTANY_CHARACTERS = {
    "leaf_type": "leaf_type_general_wa",
    "leaf_arrangement": "leaf_arrangement_wa",
    "leaf_shape_margins": "leaf_blade_margin_general_wa",
    "leaf_venation": "leaf_blade_venation_wa",
    "bark": "outer_bark_main_stem_wa",
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _r(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def normalize_name(name: str) -> str:
    """Case/punctuation-insensitive display normalization."""
    text = unicodedata.normalize("NFKC", str(name)).strip().lower()
    text = text.replace("\u00d7", "x").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".")


def species_key(name: str) -> tuple[str, str] | None:
    """Genus + epithet key, ignoring punctuation, rank suffixes and hybrid marks.

    ``Acer xfreemanii`` and ``Acer freemanii`` share a key; ``Spiraea alba var.
    latifolia`` shares a key with ``Spiraea alba`` (recorded as an
    infraspecific parent match, never silently preferred over an exact row).
    """
    tokens = normalize_name(name).split()
    if len(tokens) < 2:
        return None
    genus = re.sub(r"[^a-z]", "", tokens[0])
    epithet = re.sub(r"[^a-z]", "", tokens[1])
    return (genus, epithet)


def strip_hybrid(epithet: str) -> str:
    """Drop a leading hybrid marker ('x') when what remains is a real epithet."""
    if epithet.startswith("x") and len(epithet) >= 5:
        return epithet[1:]
    return epithet


def epithet_forms(epithet: str) -> tuple[str, ...]:
    forms = [epithet, strip_hybrid(epithet)]
    return tuple(dict.fromkeys(forms))


def within_one_edit(a: str, b: str) -> bool:
    """True when the Levenshtein distance between a and b is exactly 1."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = sum(1 for x, y in zip(a, b) if x != y)
        return diffs == 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    i = j = 0
    skipped = False
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return False
        skipped = True
        j += 1
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(md_escape(header) for header in headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(path, sep="\t", dtype={"file_id": str})
    required = {"file_id", "image_path", "segmented_path", "species", "source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    frame["species"] = frame["species"].astype(str).str.strip()
    frame["source"] = frame["source"].astype(str).str.strip().str.lower()

    per_species: dict[str, dict[str, Any]] = {}
    for species, group in frame.groupby("species", sort=True):
        lab = int((group["source"] == "lab").sum())
        field = int((group["source"] == "field").sum())
        per_species[species] = {
            "species": species,
            "lab": lab,
            "field": field,
            "total": lab + field,
        }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "n_rows": int(len(frame)),
        "n_species": int(frame["species"].nunique()),
        "lab_rows": int((frame["source"] == "lab").sum()),
        "field_rows": int((frame["source"] == "field").sum()),
        "per_species": per_species,
        "species": sorted(per_species),
    }


def load_dirr(parquet_dir: Path) -> dict[str, Any]:
    """Load the wide one-hot tag matrix plus the self-describing vocabulary."""
    import duckdb

    tags_path = parquet_dir / "dirr_species_tags.parquet"
    vocab_path = parquet_dir / "dirr_vocabulary.parquet"
    con = duckdb.connect(":memory:")
    tags = con.execute(f"SELECT * FROM '{tags_path.as_posix()}'").df()
    vocab = con.execute(f"SELECT * FROM '{vocab_path.as_posix()}'").df()
    con.close()

    value_columns = [c for c in tags.columns if c not in ("species", "book_page")]
    vocab = vocab.set_index("row_type").T
    vocab.index.name = "column"
    vocab = vocab.reset_index()

    vocab_by_column: dict[str, dict[str, str]] = {}
    for _, row in vocab.iterrows():
        column = str(row["column"])
        feature = str(row.get("feature_name") or "").strip()
        override = VOCAB_FEATURE_OVERRIDES.get(column)
        if not feature and override:
            feature = override[0]
        vocab_by_column[column] = {
            "feature": feature,
            "label": str(row.get("feature_label") or "").strip(),
            "section": str(row.get("feature_section") or "").strip(),
            "value_label": str(row.get("value_label") or column).strip(),
            "definition": str(row.get("value_definition") or "").strip(),
        }

    missing_columns = [c for c in value_columns if c not in vocab_by_column]
    if missing_columns:
        raise ValueError(f"tag columns absent from vocabulary: {missing_columns}")
    unmapped = [c for c in value_columns if not vocab_by_column[c]["feature"]]
    if unmapped:
        raise ValueError(f"tag columns without a feature mapping: {unmapped}")

    feature_meta: dict[str, dict[str, Any]] = {}
    column_order = {c: i for i, c in enumerate(value_columns)}
    for column in value_columns:
        meta = vocab_by_column[column]
        feature = meta["feature"]
        entry = feature_meta.setdefault(
            feature,
            {
                "feature": feature,
                "label": meta["label"] or feature,
                "section": meta["section"],
                "required_organ": DIRR_FEATURE_ORGAN.get(feature, "whole plant"),
                "views": list(FEATURE_VIEWS.get(feature, ("whole_plant",))),
                "values": [],
                "value_columns": [],
            },
        )
        entry["values"].append(column)
        entry["value_columns"].append(
            {
                "column": column,
                "value_label": meta["value_label"],
                "definition": meta["definition"],
            }
        )

    for entry in feature_meta.values():
        entry["n_values"] = len(entry["values"])

    feature_values: dict[str, dict[str, frozenset[str]]] = {}
    book_pages: dict[str, int | None] = {}
    for _, row in tags.iterrows():
        species = str(row["species"]).strip()
        asserted: dict[str, set[str]] = defaultdict(set)
        for column in value_columns:
            try:
                flag = int(row[column])
            except (TypeError, ValueError):
                continue
            if flag == 1:
                asserted[vocab_by_column[column]["feature"]].add(column)
        feature_values[species] = {f: frozenset(v) for f, v in asserted.items()}
        page = row.get("book_page")
        try:
            book_pages[species] = int(page) if page == page else None
        except (TypeError, ValueError):
            book_pages[species] = None

    return {
        "tags_path": tags_path,
        "vocab_path": vocab_path,
        "sha256_tags": sha256_file(tags_path),
        "sha256_vocab": sha256_file(vocab_path),
        "value_columns": value_columns,
        "column_order": column_order,
        "vocab_by_column": vocab_by_column,
        "feature_meta": feature_meta,
        "feature_values": feature_values,
        "book_pages": book_pages,
        "n_rows": len(tags),
        "species": sorted(feature_values),
    }


# ---------------------------------------------------------------------------
# Species join
# ---------------------------------------------------------------------------


def resolve_species_names(
    leafsnap_species: Sequence[str],
    dirr_species: Sequence[str],
    synonyms: Mapping[tuple[str, str], tuple[str, str, str]] | None = None,
    suggestions: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Join LeafSnap names to Dirr rows with an explicit, auditable ladder.

    Rungs, in order: exact normalized string; normalized genus+epithet key
    (absorbs hyphens, hybrid marks, spacing); infraspecific parent (one Dirr
    variety for the same binomial); curated synonym; single-character typo;
    truncated/extended epithet. Each rung must yield exactly one candidate,
    otherwise the species is reported as ambiguous or unmatched.
    """
    synonyms = synonyms or SYNONYMS
    suggestions = suggestions or SUGGESTIONS

    by_norm: dict[str, list[str]] = defaultdict(list)
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_stripped_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name in dirr_species:
        by_norm[normalize_name(name)].append(name)
        key = species_key(name)
        if key:
            by_key[key].append(name)
            stripped = strip_hybrid(key[1])
            if stripped != key[1]:
                by_stripped_key[(key[0], stripped)].append(name)

    matches: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    def key_candidates(key: tuple[str, str]) -> list[str]:
        genus, epithet = key
        found: list[str] = []
        for form in epithet_forms(epithet):
            found.extend(by_key.get((genus, form), []))
        return sorted(dict.fromkeys(found))

    for name in sorted(leafsnap_species):
        norm = normalize_name(name)
        key = species_key(name)
        chosen: str | None = None
        rule: str | None = None
        confidence = "high"
        note = ""
        rung_hits: dict[str, list[str]] = {}

        exact = sorted(by_norm.get(norm, []))
        if len(exact) == 1:
            chosen, rule = exact[0], "exact"
        elif len(exact) > 1:
            rung_hits["exact"] = exact

        if chosen is None and key:
            stripped = strip_hybrid(key[1])
            cands = sorted(dict.fromkeys(
                key_candidates(key) + by_stripped_key.get((key[0], stripped), [])
            ))
            if len(cands) == 1:
                chosen = cands[0]
                dirr_tokens = normalize_name(cands[0]).split()
                if len(dirr_tokens) > 2:
                    rule = "infraspecific_parent"
                elif strip_hybrid(key[1]) != key[1] or "x" in normalize_name(cands[0]).split()[1]:
                    rule = "hybrid_name_form"
                else:
                    rule = "orthographic_variant"
            elif len(cands) > 1:
                rung_hits["normalized_key"] = cands

        if chosen is None and key and key in synonyms:
            target, confidence, note = synonyms[key]
            cands = sorted(by_norm.get(normalize_name(target), []))
            if len(cands) == 1:
                chosen, rule = cands[0], "curated_synonym"
            elif len(cands) > 1:
                rung_hits["synonym"] = cands

        if chosen is None and key:
            query = f"{key[0]}{key[1]}"
            query_stripped = f"{key[0]}{strip_hybrid(key[1])}"
            typo = sorted(
                {
                    candidate
                    for candidate_key, names in list(by_key.items()) + list(by_stripped_key.items())
                    for candidate in names
                    if len(query) >= 10
                    and (
                        within_one_edit(query, f"{candidate_key[0]}{candidate_key[1]}")
                        or within_one_edit(query_stripped, f"{candidate_key[0]}{candidate_key[1]}")
                    )
                }
            )
            if len(typo) == 1:
                chosen, rule, confidence = typo[0], "single_character_typo", "medium"
            elif len(typo) > 1:
                rung_hits["single_character_typo"] = typo

        if chosen is None and key:
            query = f"{key[0]}{key[1]}"
            prefix = sorted(
                {
                    candidate
                    for candidate_key, names in by_key.items()
                    for candidate in names
                    if len(query) >= 12
                    and (
                        f"{candidate_key[0]}{candidate_key[1]}".startswith(query)
                        or query.startswith(f"{candidate_key[0]}{candidate_key[1]}")
                        and len(f"{candidate_key[0]}{candidate_key[1]}") >= 12
                    )
                }
            )
            if len(prefix) == 1:
                chosen, rule, confidence = (
                    prefix[0],
                    "truncated_or_extended_epithet",
                    "medium",
                )
            elif len(prefix) > 1:
                rung_hits["truncated_or_extended_epithet"] = prefix

        if chosen:
            matches.append(
                {
                    "leafsnap_species": name,
                    "dirr_species": chosen,
                    "rule": rule,
                    "confidence": confidence,
                    "note": note,
                }
            )
            continue

        if rung_hits:
            ambiguous.append(
                {
                    "leafsnap_species": name,
                    "candidates": sorted({c for names in rung_hits.values() for c in names}),
                    "rungs": sorted(rung_hits),
                }
            )
            continue

        near: list[dict[str, str]] = []
        if key:
            genus = key[0]
            for candidate in sorted(by_key):
                if candidate[0] != genus:
                    continue
                distance = _prefix_distance(key[1], candidate[1])
                if distance <= 2:
                    near.append(
                        {
                            "dirr_species": by_key[candidate][0],
                            "epithet": candidate[1],
                            "edit_distance": distance,
                        }
                    )
        near.sort(key=lambda item: (item["edit_distance"], item["dirr_species"]))
        suggestion = None
        if name in suggestions:
            suggestion = {"dirr_species": suggestions[name][0], "reason": suggestions[name][1]}
        unmatched.append(
            {
                "leafsnap_species": name,
                "reason": "no Dirr row at any join rung",
                "same_genus_near_misses": near[:5],
                "suggestion_not_applied": suggestion,
            }
        )

    matched_map = {m["leafsnap_species"]: m["dirr_species"] for m in matches}
    reverse: dict[str, list[str]] = defaultdict(list)
    for leafsnap, dirr in matched_map.items():
        reverse[dirr].append(leafsnap)
    many_to_one = [
        {"dirr_species": dirr, "leafsnap_species": sorted(names)}
        for dirr, names in sorted(reverse.items())
        if len(names) > 1
    ]
    rule_counts = Counter(m["rule"] for m in matches)
    return {
        "summary": {
            "n_leafsnap_species": len(set(leafsnap_species)),
            "n_matched": len(matches),
            "n_unmatched": len(unmatched),
            "n_ambiguous": len(ambiguous),
            "n_many_to_one": len(many_to_one),
            "match_rate": _r(len(matches) / len(set(leafsnap_species))),
            "rules": dict(sorted(rule_counts.items())),
        },
        "matches": matches,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
        "many_to_one": many_to_one,
    }


def _prefix_distance(a: str, b: str) -> int:
    """Diagnostic edit distance for near-miss reporting (never used to match)."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def entropy(weights: Sequence[float]) -> float:
    """Shannon entropy in bits for non-negative weights (sum-normalized)."""
    total = float(sum(weights))
    if total <= 0:
        return 0.0
    result = 0.0
    for weight in weights:
        if weight <= 0:
            continue
        probability = weight / total
        result -= probability * math.log2(probability)
    return result


def partition_metrics(
    cell_keys: Sequence[Any],
    weights: Sequence[float],
) -> dict[str, float]:
    """Entropy, conditional entropy, IG and NMI for one categorical partition.

    ``cell_keys[i]`` is the feature state of species ``i`` (a frozenset of
    asserted values, or an empty frozenset for "not documented").
    """
    if len(cell_keys) != len(weights):
        raise ValueError("cell keys and weights must align")
    cells: dict[Any, list[int]] = defaultdict(list)
    for index, key in enumerate(cell_keys):
        cells[key].append(index)
    total_weight = float(sum(weights))
    h_y = entropy(weights)
    cell_weights = [sum(weights[i] for i in members) for members in cells.values()]
    h_f = entropy(cell_weights)
    h_cond = 0.0
    for members in cells.values():
        cell_weight = sum(weights[i] for i in members)
        if total_weight > 0 and cell_weight > 0:
            h_cond += (cell_weight / total_weight) * entropy([weights[i] for i in members])
    ig = h_y - h_cond
    denominator = math.sqrt(h_f * h_y)
    nmi = ig / denominator if denominator > 0 else 0.0
    return {
        "entropy_taxon": h_y,
        "entropy_feature": h_f,
        "conditional_entropy": h_cond,
        "conditional_entropy_ratio": (h_cond / h_y) if h_y > 0 else 0.0,
        "information_gain": ig,
        "nmi": nmi,
        "information_gain_fraction": (ig / h_y) if h_y > 0 else 0.0,
        "n_cells": float(len(cells)),
    }


def cell_stats(
    cell_keys: Sequence[Any],
    covered: Sequence[bool],
    n_species: int,
) -> dict[str, float]:
    """Exact-cell candidate-set statistics (species-balanced)."""
    sizes = Counter(cell_keys)
    covered_sizes = [sizes[key] for key, is_covered in zip(cell_keys, covered) if is_covered]
    deployment_sizes = [
        sizes[key] if is_covered else n_species
        for key, is_covered in zip(cell_keys, covered)
    ]
    covered_unique = sum(1 for size in covered_sizes if size == 1)
    deployment_unique = sum(
        1
        for key, is_covered in zip(cell_keys, covered)
        if is_covered and sizes[key] == 1
    )
    return {
        "n_covered": len(covered_sizes),
        "resolution_rate_covered": (
            covered_unique / len(covered_sizes) if covered_sizes else 0.0
        ),
        "resolution_rate_all": deployment_unique / n_species if n_species else 0.0,
        "median_candidate_set_covered": _median(covered_sizes),
        "median_candidate_set_all": _median(deployment_sizes),
        "mean_candidate_set_all": _mean(deployment_sizes),
        "max_candidate_set": max(sizes.values()) if sizes else 0,
    }


def probe_stats(
    value_sets: Sequence[frozenset[str]],
    value_to_species: Mapping[str, frozenset[str]],
    n_species: int,
    weights: Sequence[float] | None = None,
) -> dict[str, float | None]:
    """Candidate-set behaviour when ONE value is observed.

    For a multi-valued species the observed value is unknown, so the mean,
    best and worst candidate-set sizes over its asserted values are reported.
    Unobserved species fall back to the whole candidate space.
    """
    means: list[float] = []
    bests: list[float] = []
    worsts: list[float] = []
    covered: list[bool] = []
    for values in value_sets:
        if not values:
            means.append(float(n_species))
            bests.append(float(n_species))
            worsts.append(float(n_species))
            covered.append(False)
            continue
        sizes = [float(len(value_to_species[value])) for value in values]
        means.append(sum(sizes) / len(sizes))
        bests.append(min(sizes))
        worsts.append(max(sizes))
        covered.append(True)
    covered_means = [m for m, c in zip(means, covered) if c]
    confirmed_unique = sum(1 for w, c in zip(worsts, covered) if c and w == 1)
    optimistic_unique = sum(1 for b, c in zip(bests, covered) if c and b == 1)
    weighted = None
    if weights is not None:
        total = sum(weights)
        if total > 0:
            weighted = sum(w * m for w, m in zip(weights, means)) / total
    return {
        "median_probe_candidate_set_covered": _median(covered_means),
        "median_probe_candidate_set_all": _median(means),
        "mean_probe_candidate_set_covered": _mean(covered_means),
        "probe_confirmed_unique_rate_covered": (
            confirmed_unique / len(covered_means) if covered_means else 0.0
        ),
        "probe_optimistic_unique_rate_covered": (
            optimistic_unique / len(covered_means) if covered_means else 0.0
        ),
        "weighted_mean_probe_candidate_set": weighted,
    }


def build_value_index(
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
    species: Sequence[str],
    feature: str,
) -> dict[str, frozenset[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for name in species:
        for value in values_by_species[name].get(feature, frozenset()):
            index[value].add(name)
    return {value: frozenset(members) for value, members in index.items()}


def probe_partition_metrics(
    species: Sequence[str],
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
    features: Sequence[str],
    weights: Sequence[float],
) -> dict[str, float]:
    """MI between species identity and ONE observation of ``features``.

    Each species' weight is split evenly across the cartesian product of its
    observed states; an undocumented feature contributes a single
    ``unobserved`` state. Crucially, the species marginal stays the original
    weight, so ``H(Y)`` is the species prior -- not the entropy of the expanded
    observation rows. That is what makes this the information from one probe
    rather than from a perfect multi-label observation.
    """
    total = float(sum(weights))
    h_y = entropy(weights)
    cells: dict[Any, dict[str, float]] = {}
    for name, weight in zip(species, weights):
        states: list[list[str]] = []
        for feature in features:
            values = values_by_species[name].get(feature, frozenset())
            states.append(sorted(values) if values else [UNOBSERVED])
        size = 1
        for state in states:
            size *= len(state)
        share = weight / size
        for combo in _product(states):
            key = combo if len(features) > 1 else combo[0]
            cells.setdefault(key, {})
            cells[key][name] = cells[key].get(name, 0.0) + share

    cell_weights = [sum(members.values()) for members in cells.values()]
    h_obs = entropy(cell_weights)
    h_cond = 0.0
    for members in cells.values():
        cell_weight = sum(members.values())
        if total > 0 and cell_weight > 0:
            h_cond += (cell_weight / total) * entropy(list(members.values()))
    ig = h_y - h_cond
    denominator = math.sqrt(h_obs * h_y)
    return {
        "entropy_taxon": h_y,
        "entropy_feature": h_obs,
        "conditional_entropy": h_cond,
        "conditional_entropy_ratio": (h_cond / h_y) if h_y > 0 else 0.0,
        "information_gain": ig,
        "nmi": ig / denominator if denominator > 0 else 0.0,
        "information_gain_fraction": (ig / h_y) if h_y > 0 else 0.0,
        "n_cells": float(len(cells)),
    }


def _product(groups: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = [()]
    for group in groups:
        result = [prefix + (value,) for prefix in result for value in group]
    return result


# ---------------------------------------------------------------------------
# Feature evaluation
# ---------------------------------------------------------------------------


def evaluate_feature(
    feature: str,
    species: Sequence[str],
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
    meta: Mapping[str, Any],
    weights: Sequence[float] | None = None,
    image_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    n_species = len(species)
    value_sets = [values_by_species[name].get(feature, frozenset()) for name in species]
    covered = [bool(values) for values in value_sets]
    n_covered = sum(covered)
    cell_keys = [values for values in value_sets]
    unit_weights = [1.0] * n_species
    base = partition_metrics(cell_keys, unit_weights)
    exact = cell_stats(cell_keys, covered, n_species)
    value_index = build_value_index(values_by_species, species, feature)
    probe = probe_stats(value_sets, value_index, n_species)

    probe_partition = probe_partition_metrics(
        species, values_by_species, (feature,), unit_weights
    )

    covered_cells = [values for values, is_covered in zip(value_sets, covered) if is_covered]
    covered_weights = [1.0] * len(covered_cells)
    if covered_cells:
        covered_metrics = partition_metrics(covered_cells, covered_weights)
    else:
        covered_metrics = {
            "entropy_taxon": 0.0,
            "conditional_entropy": 0.0,
            "information_gain": 0.0,
            "nmi": 0.0,
        }

    multi_valued = [len(values) for values in value_sets if values]
    result: dict[str, Any] = {
        "feature": feature,
        "label": meta.get("label", feature),
        "section": meta.get("section", ""),
        "required_organ": meta.get("required_organ", ""),
        "views": list(meta.get("views", [])),
        "n_values_defined": meta.get("n_values"),
        "n_values_used": len(value_index),
        "interpretation": FEATURE_INTERPRETATION.get(feature, ""),
        "low_coverage": (n_covered / n_species) < 0.10 if n_species else True,
        "coverage": {
            "n_species": n_covered,
            "n_total": n_species,
            "rate": _r(n_covered / n_species) if n_species else 0.0,
            "n_uncovered": n_species - n_covered,
        },
        "multi_valued": {
            "n_species": sum(1 for values in value_sets if values and len(values) > 1),
            "rate_of_covered": _r(
                sum(1 for values in value_sets if values and len(values) > 1) / n_covered
            ) if n_covered else 0.0,
            "mean_values_per_covered_species": _r(
                sum(multi_valued) / len(multi_valued)
            ) if multi_valued else 0.0,
            "max_values": max(multi_valued) if multi_valued else 0,
        },
        "entropy_taxon": _r(base["entropy_taxon"]),
        "entropy_feature": _r(base["entropy_feature"]),
        "conditional_entropy": _r(base["conditional_entropy"]),
        "conditional_entropy_ratio": _r(base["conditional_entropy_ratio"]),
        "information_gain": _r(base["information_gain"]),
        "nmi": _r(base["nmi"]),
        "information_gain_fraction": _r(base["information_gain_fraction"]),
        "probe_information_gain": _r(probe_partition["information_gain"]),
        "probe_nmi": _r(probe_partition["nmi"]),
        "probe_conditional_entropy": _r(probe_partition["conditional_entropy"]),
        "probe_conditional_entropy_ratio": _r(probe_partition["conditional_entropy_ratio"]),
        "ig_covered_only": _r(covered_metrics["information_gain"]),
        "nmi_covered_only": _r(covered_metrics["nmi"]),
        "candidate_sets": {key: _r(value) if isinstance(value, float) else value for key, value in exact.items()},
        "probe": {key: _r(value) for key, value in probe.items()},
        "value_discrimination": [
            {
                "value": value,
                "display_label": _display_label(meta, value),
                "n_species": len(members),
                "rate_of_covered": _r(len(members) / n_covered) if n_covered else 0.0,
                "is_unique": len(members) == 1,
            }
            for value, members in sorted(value_index.items(), key=lambda item: (-len(item[1]), item[0]))
        ],
    }

    if image_weights is not None:
        weighted = partition_metrics(cell_keys, image_weights)
        weighted_probe = probe_stats(value_sets, value_index, n_species, image_weights)
        weighted_partition = probe_partition_metrics(
            species, values_by_species, (feature,), image_weights
        )
        result["image_weighted"] = {
            "information_gain": _r(weighted["information_gain"]),
            "nmi": _r(weighted["nmi"]),
            "conditional_entropy": _r(weighted["conditional_entropy"]),
            "probe_information_gain": _r(weighted_partition["information_gain"]),
            "probe_nmi": _r(weighted_partition["nmi"]),
            "weighted_mean_probe_candidate_set": _r(
                weighted_probe["weighted_mean_probe_candidate_set"]
            ),
        }
    return result


def _display_label(meta: Mapping[str, Any], value: str) -> str:
    for entry in meta.get("value_columns", []):
        if entry["column"] == value:
            return entry["value_label"] or value
    return value


# ---------------------------------------------------------------------------
# Joint (chordal) evaluation
# ---------------------------------------------------------------------------


def _pair_cell(values: Mapping[str, frozenset[str]], a: str, b: str) -> tuple[frozenset[str], frozenset[str]]:
    return (values.get(a, frozenset()), values.get(b, frozenset()))


def evaluate_pair(
    a: str,
    b: str,
    species: Sequence[str],
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
    meta: Mapping[str, Mapping[str, Any]],
    single_ig: Mapping[str, float],
    single_probe_ig: Mapping[str, float],
) -> dict[str, Any]:
    n_species = len(species)
    cells = [_pair_cell(values_by_species[name], a, b) for name in species]
    covered = [
        bool(values_by_species[name].get(a)) and bool(values_by_species[name].get(b))
        for name in species
    ]
    base = partition_metrics(cells, [1.0] * n_species)
    exact = cell_stats(cells, covered, n_species)
    joint_ig = base["information_gain"]
    ig_a = single_ig[a]
    ig_b = single_ig[b]
    cond_a_given_b = joint_ig - ig_b
    cond_b_given_a = joint_ig - ig_a
    synergy = joint_ig - max(ig_a, ig_b)
    redundancy = ig_a + ig_b - joint_ig

    probe_partition = probe_partition_metrics(
        species, values_by_species, (a, b), [1.0] * n_species
    )
    probe_ig_a = single_probe_ig[a]
    probe_ig_b = single_probe_ig[b]
    probe = _pair_probe(values_by_species, species, a, b, n_species)
    views_a = set(meta[a].get("views", []))
    views_b = set(meta[b].get("views", []))
    same_view = views_a == views_b
    non_local = views_a.isdisjoint(views_b)
    return {
        "feature_a": a,
        "feature_b": b,
        "views_a": sorted(views_a),
        "views_b": sorted(views_b),
        "relation": "same_view" if same_view else ("non_local" if non_local else "mixed"),
        "n_shared": sum(covered),
        "shared_coverage": _r(sum(covered) / n_species) if n_species else 0.0,
        "joint_ig": _r(joint_ig),
        "joint_nmi": _r(base["nmi"]),
        "joint_conditional_entropy": _r(base["conditional_entropy"]),
        "joint_conditional_entropy_ratio": _r(base["conditional_entropy_ratio"]),
        "joint_probe_ig": _r(probe_partition["information_gain"]),
        "joint_probe_nmi": _r(probe_partition["nmi"]),
        "joint_probe_conditional_entropy": _r(probe_partition["conditional_entropy"]),
        "joint_probe_conditional_entropy_ratio": _r(
            probe_partition["conditional_entropy_ratio"]
        ),
        "probe_ig_a": _r(probe_ig_a),
        "probe_ig_b": _r(probe_ig_b),
        "probe_cond_ig_a_given_b": _r(probe_partition["information_gain"] - probe_ig_b),
        "probe_cond_ig_b_given_a": _r(probe_partition["information_gain"] - probe_ig_a),
        "probe_synergy": _r(probe_partition["information_gain"] - max(probe_ig_a, probe_ig_b)),
        "probe_redundancy": _r(
            probe_ig_a + probe_ig_b - probe_partition["information_gain"]
        ),
        "ig_a": _r(ig_a),
        "ig_b": _r(ig_b),
        "cond_ig_a_given_b": _r(cond_a_given_b),
        "cond_ig_b_given_a": _r(cond_b_given_a),
        "synergy": _r(synergy),
        "redundancy": _r(redundancy),
        "gain_over_best_single": _r(synergy),
        "resolution_rate_covered": _r(exact["resolution_rate_covered"]),
        "resolution_rate_all": _r(exact["resolution_rate_all"]),
        "median_candidate_set_covered": _r(exact["median_candidate_set_covered"]),
        "median_candidate_set_all": _r(exact["median_candidate_set_all"]),
        "probe": {key: _r(value) for key, value in probe.items()},
    }


def _pair_probe(
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
    species: Sequence[str],
    a: str,
    b: str,
    n_species: int,
) -> dict[str, float | None]:
    index_a = build_value_index(values_by_species, species, a)
    index_b = build_value_index(values_by_species, species, b)
    means: list[float] = []
    worsts: list[float] = []
    covered: list[bool] = []
    for name in species:
        vals_a = values_by_species[name].get(a, frozenset())
        vals_b = values_by_species[name].get(b, frozenset())
        if not vals_a or not vals_b:
            means.append(float(n_species))
            worsts.append(float(n_species))
            covered.append(False)
            continue
        sizes = [
            float(len(index_a[value_a] & index_b[value_b]))
            for value_a in vals_a
            for value_b in vals_b
        ]
        means.append(sum(sizes) / len(sizes))
        worsts.append(max(sizes))
        covered.append(True)
    covered_means = [m for m, c in zip(means, covered) if c]
    confirmed_unique = sum(1 for w, c in zip(worsts, covered) if c and w == 1)
    return {
        "median_probe_candidate_set_covered": _median(covered_means),
        "median_probe_candidate_set_all": _median(means),
        "probe_confirmed_unique_rate_covered": (
            confirmed_unique / len(covered_means) if covered_means else 0.0
        ),
    }


def evaluate_triple(
    features: Sequence[str],
    species: Sequence[str],
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
    meta: Mapping[str, Mapping[str, Any]],
    single_ig: Mapping[str, float],
    single_probe_ig: Mapping[str, float],
    pair_joint: Mapping[frozenset[str], float],
    pair_probe_joint: Mapping[frozenset[str], float],
) -> dict[str, Any]:
    cells = [
        tuple(values_by_species[name].get(feature, frozenset()) for feature in features)
        for name in species
    ]
    covered = [
        all(values_by_species[name].get(feature) for feature in features)
        for name in species
    ]
    base = partition_metrics(cells, [1.0] * len(species))
    probe_partition = probe_partition_metrics(
        species, values_by_species, features, [1.0] * len(species)
    )
    exact = cell_stats(cells, covered, len(species))
    joint_ig = base["information_gain"]
    best_pair = max(
        pair_joint[frozenset(pair)]
        for pair in _combinations(list(features), 2)
    )
    best_pair_probe = max(
        pair_probe_joint[frozenset(pair)]
        for pair in _combinations(list(features), 2)
    )
    views: list[str] = []
    for feature in features:
        views.extend(meta[feature].get("views", []))
    return {
        "features": list(features),
        "views": sorted(set(views)),
        "n_distinct_views": len(set(views)),
        "fully_non_local": len(set(views)) == len(features),
        "n_shared": sum(covered),
        "joint_ig": _r(joint_ig),
        "joint_nmi": _r(base["nmi"]),
        "joint_conditional_entropy": _r(base["conditional_entropy"]),
        "joint_probe_ig": _r(probe_partition["information_gain"]),
        "joint_probe_nmi": _r(probe_partition["nmi"]),
        "best_single_ig": _r(max(single_ig[feature] for feature in features)),
        "best_single_probe_ig": _r(
            max(single_probe_ig[feature] for feature in features)
        ),
        "best_pair_joint_ig": _r(best_pair),
        "best_pair_probe_ig": _r(best_pair_probe),
        "marginal_over_best_pair": _r(joint_ig - best_pair),
        "marginal_probe_over_best_pair": _r(
            probe_partition["information_gain"] - best_pair_probe
        ),
        "synergy_over_best_single": _r(joint_ig - max(single_ig[feature] for feature in features)),
        "probe_synergy_over_best_single": _r(
            probe_partition["information_gain"]
            - max(single_probe_ig[feature] for feature in features)
        ),
        "resolution_rate_covered": _r(exact["resolution_rate_covered"]),
        "median_candidate_set_covered": _r(exact["median_candidate_set_covered"]),
    }


def _combinations(values: Sequence[str], size: int) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    if size == 2:
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                result.append((left, right))
        return result
    if size == 3:
        for i, first in enumerate(values):
            for j in range(i + 1, len(values)):
                for k in range(j + 1, len(values)):
                    result.append((first, values[j], values[k]))
        return result
    raise ValueError("only pairs and triples are supported")


# ---------------------------------------------------------------------------
# Conflict + spotlight analysis
# ---------------------------------------------------------------------------


def find_conflicts(
    species: Sequence[str],
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
) -> dict[str, Any]:
    conflicts: dict[str, list[dict[str, Any]]] = {}
    for feature, rules in CONTRADICTION_RULES.items():
        hits = []
        for name in species:
            values = values_by_species[name].get(feature, frozenset())
            for left, right in rules:
                if left in values and right in values:
                    hits.append(
                        {
                            "species": name,
                            "values": sorted(values),
                            "contradiction": [left, right],
                        }
                    )
                    break
        conflicts[feature] = hits

    orphans: list[dict[str, Any]] = []
    for name in species:
        values = values_by_species[name]
        if "leaf_type" not in values:
            continue
        types = values["leaf_type"]
        if types == frozenset({"Simple"}):
            for feature in ("compound_leaf_type", "pinnate_types"):
                if values.get(feature):
                    orphans.append(
                        {
                            "species": name,
                            "feature": feature,
                            "values": sorted(values[feature]),
                            "leaf_type": sorted(types),
                            "reason": "compound-only feature asserted on a simple-leaved species",
                        }
                    )
    return {
        "contradictions": conflicts,
        "orphan_assertions": orphans,
        "n_contradiction_species": sorted(
            {hit["species"] for hits in conflicts.values() for hit in hits}
        ),
    }


def leaf_type_spotlight(
    species: Sequence[str],
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
    leafsnap: Mapping[str, Any],
    feature_result: Mapping[str, Any],
    conflicts: Mapping[str, Any],
) -> dict[str, Any]:
    counts = Counter()
    by_class: dict[str, list[str]] = defaultdict(list)
    for name in species:
        values = values_by_species[name].get("leaf_type", frozenset())
        if values == frozenset({"Simple"}):
            counts["Simple"] += 1
            by_class["Simple"].append(name)
        elif values == frozenset({"Compound"}):
            counts["Compound"] += 1
            by_class["Compound"].append(name)
        elif values:
            counts["mixed"] += 1
            by_class["mixed"].append(name)
        else:
            counts["unlabelled"] += 1
            by_class["unlabelled"].append(name)

    image_counts = {
        key: {
            "species": len(names),
            "lab_images": sum(leafsnap["per_species"][name]["lab"] for name in names),
            "field_images": sum(leafsnap["per_species"][name]["field"] for name in names),
        }
        for key, names in by_class.items()
    }

    compound_species = by_class["Compound"]
    compound_subtype = [
        name for name in compound_species
        if values_by_species[name].get("compound_leaf_type")
    ]
    simple_with_compound_type = [
        name for name in by_class["Simple"]
        if values_by_species[name].get("compound_leaf_type")
    ]
    return {
        "counts": dict(sorted(counts.items())),
        "species": {key: sorted(names) for key, names in sorted(by_class.items())},
        "image_counts": image_counts,
        "conflicts": conflicts["contradictions"].get("leaf_type", []),
        "mixed_species": sorted(by_class["mixed"]),
        "compound_subtype_coverage": {
            "n_compound": len(compound_species),
            "n_with_subtype": len(compound_subtype),
            "rate": _r(len(compound_subtype) / len(compound_species)) if compound_species else 0.0,
            "subtype_counts": dict(
                sorted(
                    Counter(
                        "Pinnate" if "Pinnate (compound)" in values_by_species[name].get("compound_leaf_type", frozenset())
                        else "Palmate"
                        for name in compound_subtype
                    ).items()
                )
            ),
        },
        "simple_with_compound_type": sorted(simple_with_compound_type),
        "single_feature": dict(feature_result),
        "value_discrimination": feature_result.get("value_discrimination", []),
    }


def bark_spotlight(
    dirr_meta: Mapping[str, Any],
    gobotany: Mapping[str, Any] | None,
) -> dict[str, Any]:
    bark_features = sorted(feature for feature in dirr_meta if "bark" in feature.lower())
    result: dict[str, Any] = {
        "dirr_bark_features": bark_features,
        "dirr_bark_coverage": 0.0,
        "dirr_note": (
            "The Dirr extracted vocabulary has no bark feature (no value column "
            "maps to bark). Leaf x bark fusion is therefore not testable from "
            "the current Dirr parquets; it would need a new Dirr bark "
            "extraction or a GoBotany crossover that keeps provenance separate."
        ),
        "gobotany_bark": None,
    }
    if gobotany and gobotany.get("characters", {}).get("bark"):
        result["gobotany_bark"] = gobotany["characters"]["bark"]
    return result


def gobotany_check(
    leafsnap_all_species: Sequence[str],
    matched_species: Sequence[str],
    dirr_match: Mapping[str, str],
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
    gobotany_dir: Path | None,
) -> dict[str, Any] | None:
    """Coverage + conflict check only. GoBotany values never enter Dirr results."""
    if gobotany_dir is None or not (gobotany_dir / "gobotany_taxon.parquet").exists():
        return None
    import duckdb

    con = duckdb.connect(":memory:")
    taxa = con.execute(
        f"SELECT taxon_id, scientific_name FROM '{gobotany_dir / 'gobotany_taxon.parquet'}'"
    ).df()
    char_values = con.execute(
        f"SELECT taxon_id, character_short_name, value_index FROM "
        f"'{gobotany_dir / 'gobotany_taxon_character_value.parquet'}'"
    ).df()
    labels = con.execute(
        f"SELECT character_short_name, value_index, choice FROM "
        f"'{gobotany_dir / 'gobotany_character_value_label.parquet'}'"
    ).df()
    con.close()

    taxon_by_norm = {
        normalize_name(row["scientific_name"]): int(row["taxon_id"])
        for _, row in taxa.iterrows()
    }
    label_lookup = {
        (row["character_short_name"], int(row["value_index"])): str(row["choice"])
        for _, row in labels.iterrows()
        if str(row["choice"]) not in ("NA", "nan", "None")
    }
    values: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for _, row in char_values.iterrows():
        character = row["character_short_name"]
        if character not in set(GOBOTANY_CHARACTERS.values()):
            continue
        choice = label_lookup.get((character, int(row["value_index"])))
        if not choice:
            continue
        values[int(row["taxon_id"])][character].add(choice)

    by_character: dict[str, dict[str, Any]] = {}
    for feature, character in GOBOTANY_CHARACTERS.items():
        covered = []
        for name in leafsnap_all_species:
            taxon_id = taxon_by_norm.get(normalize_name(name))
            if taxon_id is not None and values[taxon_id].get(character):
                covered.append(name)
        by_character[feature] = {
            "character": character,
            "n_leafsnap_species": len(covered),
            "coverage_rate_of_leafsnap": _r(len(covered) / len(leafsnap_all_species)),
            "coverage_rate_of_dirr_matched": _r(
                len([name for name in covered if name in dirr_match])
                / len(matched_species)
            ) if matched_species else 0.0,
            "value_counts": dict(
                sorted(
                    Counter(
                        choice
                        for name in covered
                        for choice in values[taxon_by_norm[normalize_name(name)]][character]
                    ).items()
                )
            ),
        }

    comparisons = _gobotany_conflicts(
        matched_species, dirr_match, values_by_species, taxon_by_norm, values
    )
    n_gobotany_taxa = len(taxa)
    n_leafsnap_in_gobotany = sum(
        1 for name in leafsnap_all_species if normalize_name(name) in taxon_by_norm
    )
    return {
        "source": "gobotany",
        "access": (
            "GoBotany parquet tables read directly (gobotany_taxon, "
            "gobotany_taxon_character_value, gobotany_character_value_label); "
            "the pile-scoped GoBotanyDevoAdapter query layer is not used."
        ),
        "provenance_note": (
            "GoBotany is used only to assess coverage and resolve conflicts. "
            "Its labels are never merged into the Dirr feature table."
        ),
        "n_taxa": int(n_gobotany_taxa),
        "n_leafsnap_in_gobotany": int(n_leafsnap_in_gobotany),
        "leafsnap_coverage_rate": _r(n_leafsnap_in_gobotany / len(leafsnap_all_species)),
        "characters": by_character,
        "comparisons": comparisons,
    }


def _gobotany_conflicts(
    leafsnap_species: Sequence[str],
    dirr_match: Mapping[str, str],
    values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
    taxon_by_norm: Mapping[str, int],
    gb_values: Mapping[int, Mapping[str, set[str]]],
) -> dict[str, Any]:
    arrangement_map = {
        "1 leaf per node": "Alternate",
        "2 leaves per node": "Opposite",
        "3 leaves per node": "Whorled",
        "4 leaves per node": "Whorled",
    }
    margin_coarse = {
        "Entire": "entire",
        "Lobed": "lobed",
        "Pinnately Lobed": "lobed",
        "Palmately Lobed": "lobed",
        "Serrate": "toothed",
        "Dentate": "toothed",
        "Doubly-Serrate": "toothed",
        "Crenate": "toothed",
    }
    venation_map = {
        "Pinnate (venation)": "pinnate",
        "Palmate (venation)": "palmate",
        "Parallel": "parallel",
    }
    gb_venation_map = {
        "pinnate and arcuate": "pinnate",
        "pinnate and non-arcuate": "pinnate",
        "palmate": "palmate",
        "parallel": "parallel",
    }

    def dirr_values(name: str, feature: str) -> set[str]:
        return set(values_by_species[name].get(feature, frozenset()))

    def gb_values_for(name: str, character: str) -> set[str]:
        taxon_id = taxon_by_norm.get(normalize_name(name))
        return set(gb_values[taxon_id].get(character, set())) if taxon_id is not None else set()

    def compare(
        feature: str,
        character: str,
        dirr_reader,
        gb_reader,
    ) -> dict[str, Any]:
        compared = 0
        agree = 0
        conflicts = []
        for name in leafsnap_species:
            d_values = dirr_reader(name)
            g_values = gb_reader(name)
            if len(d_values) != 1 or len(g_values) != 1:
                continue
            d_value = next(iter(d_values))
            g_value = next(iter(g_values))
            compared += 1
            if d_value == g_value:
                agree += 1
            else:
                conflicts.append(
                    {
                        "leafsnap_species": name,
                        "dirr_species": dirr_match[name],
                        "dirr": d_value,
                        "gobotany": g_value,
                    }
                )
        return {
            "feature": feature,
            "gobotany_character": character,
            "n_compared_unambiguous": compared,
            "n_agree": agree,
            "n_conflict": len(conflicts),
            "agreement_rate": _r(agree / compared) if compared else None,
            "conflicts": conflicts,
        }

    return {
        "leaf_type": compare(
            "leaf_type",
            GOBOTANY_CHARACTERS["leaf_type"],
            lambda name: {value.lower() for value in dirr_values(name, "leaf_type")},
            lambda name: {value.lower() for value in gb_values_for(name, GOBOTANY_CHARACTERS["leaf_type"])},
        ),
        "leaf_arrangement": compare(
            "leaf_arrangement",
            GOBOTANY_CHARACTERS["leaf_arrangement"],
            lambda name: dirr_values(name, "leaf_arrangement"),
            lambda name: {
                arrangement_map[value]
                for value in gb_values_for(name, GOBOTANY_CHARACTERS["leaf_arrangement"])
                if value in arrangement_map
            },
        ),
        "leaf_shape_margins": compare(
            "leaf_shape_margins",
            GOBOTANY_CHARACTERS["leaf_shape_margins"],
            lambda name: {
                margin_coarse[value]
                for value in dirr_values(name, "leaf_shape_margins")
                if value in margin_coarse
            },
            lambda name: gb_values_for(name, GOBOTANY_CHARACTERS["leaf_shape_margins"]),
        ),
        "leaf_venation": compare(
            "leaf_venation",
            GOBOTANY_CHARACTERS["leaf_venation"],
            lambda name: {
                venation_map[value]
                for value in dirr_values(name, "leaf_venation")
                if value in venation_map
            },
            lambda name: {
                gb_venation_map[value]
                for value in gb_values_for(name, GOBOTANY_CHARACTERS["leaf_venation"])
                if value in gb_venation_map
            },
        ),
    }


def adapter_fidelity(
    liked: Mapping[str, Any],
    dirr_values_by_species: Mapping[str, Mapping[str, frozenset[str]]],
) -> dict[str, Any]:
    """Compare DirrDevoAdapter output with the column-accurate mapping.

    ``list_features`` skips vocabulary rows with no ``feature_name``. Four
    tag columns (Reticulate, Elliptic, Deltoid, Rhombic) have no metadata, so
    ``get_taxon`` silently drops every assertion in them. The adapter keys by
    the tag column name, which is safe for the suffixed columns
    (``Pinnate (compound)``, ``Acute (base)``, ...); a consumer that keys by
    the vocabulary ``value_label`` row instead would misattribute them. This
    block quantifies the real drop rather than relying on the adapter.
    """
    result: dict[str, Any] = {
        "adapter": "segmentation.devo_adapter.DirrDevoAdapter",
        "available": False,
        "note": "",
    }
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from segmentation.devo_adapter import DirrDevoAdapter  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import guard
        result["note"] = f"adapter import failed: {exc}"
        return result

    try:
        adapter = DirrDevoAdapter()
        adapter_features = adapter.list_features()
        adapter_map = {
            fv.value_label: fv.feature_name for fv in adapter_features
        }
        result["available"] = True
        result["n_adapter_feature_values"] = len(adapter_features)
        result["n_distinct_adapter_value_labels"] = len(adapter_map)
        dropped_columns = sorted(
            {
                column
                for column in liked["value_columns"]
                if column not in adapter_map
            }
        )
        result["n_tag_columns"] = len(liked["value_columns"])
        result["dropped_tag_columns"] = dropped_columns
        result["dropped_tag_column_count"] = len(dropped_columns)
        result["dropped_features"] = sorted(
            {
                liked["vocab_by_column"][column]["feature"]
                for column in dropped_columns
            }
        )

        dropped_assertions = 0
        affected_species = set()
        examples = []
        for name in sorted(dirr_values_by_species):
            taxon = adapter.get_taxon(name)
            if taxon is None:
                continue
            adapter_values = {
                (fv.feature_name, fv.value_label) for fv in taxon.feature_values
            }
            corrected_values = {
                (liked["vocab_by_column"][column]["feature"], column)
                for feature_values in dirr_values_by_species[name].values()
                for column in feature_values
            }
            missing = corrected_values - adapter_values
            if missing:
                dropped_assertions += len(missing)
                affected_species.add(name)
                if len(examples) < 8:
                    examples.append(
                        {
                            "dirr_species": name,
                            "missing_feature_values": sorted(
                                f"{feature}:{value}" for feature, value in missing
                            )[:8],
                        }
                    )
        result.update(
            {
                "matched_species_checked": len(dirr_values_by_species),
                "n_species_with_dropped_assertions": len(affected_species),
                "n_dropped_assertions": dropped_assertions,
                "examples": examples,
            }
        )

        adapter_organ = {}
        try:
            from segmentation.devo_adapter import _DIRR_FEATURE_ORGAN  # noqa: PLC0415

            adapter_organ = dict(_DIRR_FEATURE_ORGAN)
        except Exception:
            adapter_organ = {}
        if adapter_organ:
            result["organ_mapping_mismatches"] = sorted(
                feature
                for feature in set(adapter_organ) | set(DIRR_FEATURE_ORGAN)
                if adapter_organ.get(feature) != DIRR_FEATURE_ORGAN.get(feature)
            )
    except Exception as exc:  # pragma: no cover - defensive
        result["note"] = f"adapter evaluation failed: {exc}"
    return result


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def run_eda(
    manifest_path: Path,
    dirr_dir: Path,
    gobotany_dir: Path | None,
    include_gobotany: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    leafsnap = load_manifest(manifest_path)
    dirr = load_dirr(dirr_dir)

    resolution = resolve_species_names(leafsnap["species"], dirr["species"])
    matched = {m["leafsnap_species"]: m["dirr_species"] for m in resolution["matches"]}
    matched_species = sorted(matched)
    dirr_only_species = [name for name in dirr["species"] if species_key(name)]
    genus_only = sorted(name for name in dirr["species"] if not species_key(name))

    # Candidate space 1 (primary): the LeafSnap species that join to Dirr.
    matched_values = {name: dirr["feature_values"][matched[name]] for name in matched_species}
    matched_weights = [1.0] * len(matched_species)
    image_weights = [
        float(leafsnap["per_species"][name]["total"]) for name in matched_species
    ]
    feature_names = sorted(dirr["feature_meta"])

    single_features = [
        evaluate_feature(
            feature,
            matched_species,
            matched_values,
            dirr["feature_meta"][feature],
            weights=matched_weights,
            image_weights=image_weights,
        )
        for feature in feature_names
    ]
    single_features.sort(
        key=lambda item: (-item["probe_information_gain"], item["feature"])
    )
    for rank, item in enumerate(single_features, start=1):
        item["rank"] = rank
    single_by_name = {item["feature"]: item for item in single_features}

    # Secondary candidate space: the whole Dirr species-level guide.
    dirr_values = {name: dirr["feature_values"][name] for name in dirr_only_species}
    dirr_universe = [
        evaluate_feature(
            feature,
            dirr_only_species,
            dirr_values,
            dirr["feature_meta"][feature],
        )
        for feature in feature_names
    ]
    for rank, item in enumerate(
        sorted(dirr_universe, key=lambda x: (-x["probe_information_gain"], x["feature"])),
        start=1,
    ):
        item["rank"] = rank
    dirr_universe.sort(key=lambda item: item["rank"])

    n_matched = len(matched_species)
    eligible_features = [
        item["feature"]
        for item in single_features
        if item["coverage"]["rate"] >= MIN_FEATURE_COVERAGE
    ]
    single_ig = {item["feature"]: item["information_gain"] for item in single_features}
    single_probe_ig = {
        item["feature"]: item["probe_information_gain"] for item in single_features
    }

    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(eligible_features):
        for b in eligible_features[i + 1 :]:
            shared = sum(
                1
                for name in matched_species
                if matched_values[name].get(a) and matched_values[name].get(b)
            )
            if shared / n_matched < MIN_PAIR_SHARED_COVERAGE:
                continue
            pairs.append(
                evaluate_pair(
                    a,
                    b,
                    matched_species,
                    matched_values,
                    dirr["feature_meta"],
                    single_ig,
                    single_probe_ig,
                )
            )
    pairs.sort(
        key=lambda item: (-item["joint_probe_ig"], item["feature_a"], item["feature_b"])
    )

    pair_joint = {frozenset((p["feature_a"], p["feature_b"])): p["joint_ig"] for p in pairs}
    pair_probe_joint = {
        frozenset((p["feature_a"], p["feature_b"])): p["joint_probe_ig"] for p in pairs
    }
    triple_features = eligible_features[:MAX_TRIPLE_FEATURES]
    triples: list[dict[str, Any]] = []
    for combo in _combinations(triple_features, 3):
        shared = sum(
            1
            for name in matched_species
            if all(matched_values[name].get(feature) for feature in combo)
        )
        if shared / n_matched < MIN_TRIPLE_SHARED_COVERAGE:
            continue
        if any(frozenset(pair) not in pair_joint for pair in _combinations(list(combo), 2)):
            continue
        triples.append(
            evaluate_triple(
                combo,
                matched_species,
                matched_values,
                dirr["feature_meta"],
                single_ig,
                single_probe_ig,
                pair_joint,
                pair_probe_joint,
            )
        )
    triples.sort(key=lambda item: (-item["joint_probe_ig"], item["features"]))
    top_triples = triples[:20]

    conflicts = find_conflicts(matched_species, matched_values)
    spotlight = leaf_type_spotlight(
        matched_species,
        matched_values,
        leafsnap,
        single_by_name["leaf_type"],
        conflicts,
    )
    gobotany = (
        gobotany_check(
            leafsnap["species"],
            matched_species,
            matched,
            matched_values,
            gobotany_dir,
        )
        if include_gobotany
        else None
    )
    bark = bark_spotlight(dirr["feature_meta"], gobotany)

    underpowered = []
    for pair_features in [
        ("leaf_surface_top", "leaf_surface_bottom"),
        ("leaf_type", "growth_habit_types"),
    ]:
        a, b = pair_features
        underpowered.append(
            evaluate_pair(
                a,
                b,
                matched_species,
                matched_values,
                dirr["feature_meta"],
                single_ig,
                single_probe_ig,
            )
        )

    non_exact = [m for m in resolution["matches"] if m["rule"] != "exact"]
    results: dict[str, Any] = {
        "meta": {
            "analysis": "leafsnap-dirr-hierarchical-eda",
            "version": 1,
            "python": platform.python_version(),
            "inputs": {
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": leafsnap["sha256"],
                    "rows": leafsnap["n_rows"],
                    "species": leafsnap["n_species"],
                },
                "dirr": {
                    "tags_path": str(dirr["tags_path"]),
                    "sha256_tags": dirr["sha256_tags"],
                    "vocab_path": str(dirr["vocab_path"]),
                    "sha256_vocab": dirr["sha256_vocab"],
                    "rows": dirr["n_rows"],
                },
                "gobotany": (
                    {"path": str(gobotany_dir) if gobotany_dir else None, "used": True}
                    if gobotany
                    else {"used": False}
                ),
            },
            "config": {
                "min_feature_coverage": MIN_FEATURE_COVERAGE,
                "min_pair_shared_coverage": MIN_PAIR_SHARED_COVERAGE,
                "min_triple_shared_coverage": MIN_TRIPLE_SHARED_COVERAGE,
                "max_triple_features": MAX_TRIPLE_FEATURES,
                "primary_weight": "species-balanced (1/N per matched LeafSnap species)",
                "secondary_weight": "LeafSnap image counts (explicitly labelled)",
                "unobserved_cell": "empty frozenset; penalises sparse features",
            },
        },
        "leafsnap": {
            k: v for k, v in leafsnap.items() if k not in ("per_species", "species", "path", "sha256")
        },
        "leafsnap_per_species": leafsnap["per_species"],
        "image_counts_by_join": {
            label: {
                "species": len(names),
                "lab": sum(leafsnap["per_species"][name]["lab"] for name in names),
                "field": sum(leafsnap["per_species"][name]["field"] for name in names),
                "total": sum(leafsnap["per_species"][name]["total"] for name in names),
            }
            for label, names in (
                ("matched", sorted(matched)),
                (
                    "not_matched",
                    sorted(set(leafsnap["species"]) - set(matched)),
                ),
            )
        },
        "dirr": {
            "n_rows": dirr["n_rows"],
            "n_species_level_rows": len(dirr_only_species),
            "genus_only_rows": genus_only,
            "n_features": len(feature_names),
            "n_value_columns": len(dirr["value_columns"]),
            "features": {
                name: {
                    "label": dirr["feature_meta"][name]["label"],
                    "section": dirr["feature_meta"][name]["section"],
                    "required_organ": dirr["feature_meta"][name]["required_organ"],
                    "views": dirr["feature_meta"][name]["views"],
                    "n_values": dirr["feature_meta"][name]["n_values"],
                    "values": dirr["feature_meta"][name]["values"],
                    "interpretation": FEATURE_INTERPRETATION.get(name, ""),
                }
                for name in feature_names
            },
        },
        "name_resolution": resolution,
        "candidate_space": {
            "n_matched_leafsnap_species": n_matched,
            "n_unmatched": resolution["summary"]["n_unmatched"],
            "n_dirr_species_level": len(dirr_only_species),
            "primary": "matched LeafSnap species (species-balanced)",
            "secondary": "all Dirr species-level rows",
        },
        "data_quality": {
            "vocab_columns_without_metadata": sorted(VOCAB_FEATURE_OVERRIDES),
            "vocab_override_reasons": {k: v[1] for k, v in VOCAB_FEATURE_OVERRIDES.items()},
            "suffixed_tag_columns": sorted(
                column
                for column in dirr["value_columns"]
                if re.search(r"\((compound|venation|base|apex)\)$", column)
            ),
            "duplicate_value_labels": _duplicate_value_labels(dirr),
            "matched_species_with_no_assertions": sorted(
                name for name, values in matched_values.items() if not values
            ),
            "n_matched_species_with_no_assertions": sum(
                1 for values in matched_values.values() if not values
            ),
            "genus_only_rows_with_assertions": {
                name: sum(len(values) for values in dirr["feature_values"][name].values())
                for name in genus_only
            },
            "adapter_fidelity": adapter_fidelity(
                dirr,
                {matched[name]: values for name, values in matched_values.items()},
            ),
        },
        "single_features_leafsnap": single_features,
        "single_features_dirr_universe": dirr_universe,
        "eligible_features": eligible_features,
        "pairs": {
            "n_eligible": len(pairs),
            "relation_counts": dict(sorted(Counter(p["relation"] for p in pairs).items())),
            "top_same_view": [p for p in pairs if p["relation"] == "same_view"][:15],
            "top_non_local": [p for p in pairs if p["relation"] == "non_local"][:15],
            "all": pairs,
            "underpowered_spotlights": underpowered,
            "definitions": {
                key: METRIC_DEFINITIONS[key]
                for key in (
                    "joint_ig",
                    "conditional_ig",
                    "synergy",
                    "redundancy",
                    "probe_candidate_set",
                )
            },
        },
        "triples": {
            "n_evaluated": len(triples),
            "n_fully_non_local": sum(1 for item in triples if item["fully_non_local"]),
            "features_considered": triple_features,
            "all": triples,
            "top_by_joint_probe_ig": top_triples,
            "top_by_marginal_probe_over_best_pair": sorted(
                triples,
                key=lambda item: (-item["marginal_probe_over_best_pair"], item["features"]),
            )[:10],
        },
        "conflicts": {
            "contradictions": {
                feature: hits for feature, hits in conflicts["contradictions"].items() if hits
            },
            "n_species_with_any_contradiction": len(conflicts["n_contradiction_species"]),
            "species_with_any_contradiction": conflicts["n_contradiction_species"],
            "orphan_assertions": conflicts["orphan_assertions"],
        },
        "spotlights": {
            "leaf_type": spotlight,
            "leaf_arrangement": _feature_spotlight("leaf_arrangement", single_by_name, conflicts),
            "leaf_shape_margins": _feature_spotlight("leaf_shape_margins", single_by_name, conflicts),
            "leaf_venation": _feature_spotlight("leaf_venation", single_by_name, conflicts),
            "leaf_surface_top": _feature_spotlight("leaf_surface_top", single_by_name, conflicts),
            "leaf_surface_bottom": _feature_spotlight("leaf_surface_bottom", single_by_name, conflicts),
            "growth_habit_types": _feature_spotlight("growth_habit_types", single_by_name, conflicts),
            "inflorescences": _feature_spotlight("inflorescences", single_by_name, conflicts),
            "bark": bark,
        },
        "gobotany": gobotany,
        "metric_definitions": METRIC_DEFINITIONS,
        "view_definitions": VIEW_DESCRIPTIONS,
        "non_exact_matches": non_exact,
        "warnings": _warnings(),
    }
    assertions = _feature_assertions_artifact(
        leafsnap, dirr, resolution, matched, dirr_only_species
    )
    return results, assertions


def _duplicate_value_labels(dirr: Mapping[str, Any]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = defaultdict(list)
    for column, meta in dirr["vocab_by_column"].items():
        labels[meta["value_label"]].append(f"{meta['feature']}:{column}")
    return {
        label: sorted(entries)
        for label, entries in sorted(labels.items())
        if len({entry.split(":", 1)[0] for entry in entries}) > 1
    }


def _feature_spotlight(
    feature: str,
    single_by_name: Mapping[str, Any],
    conflicts: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(single_by_name[feature])
    result["contradictions"] = conflicts["contradictions"].get(feature, [])
    return result


def _warnings() -> list[dict[str, str]]:
    return [
        {
            "id": "fusion-required",
            "severity": "critical",
            "text": (
                "Joint features MUST be fused across observations / spatial "
                "tracks. A same-view pair can share one crop; a non-local pair "
                "(leaf obverse vs leaf reverse, leaf vs whole plant, leaf vs "
                "flower) cannot. Never assume a non-local pair is visible in one "
                "image, and never train a single-crop model to predict both "
                "sides of one."
            ),
        },
        {
            "id": "missing-is-not-a-value",
            "severity": "high",
            "text": (
                "An absent Dirr assertion is a documentation gap, not a "
                "biological value. Entropy treats it as its own cell and the "
                "deployment candidate set falls back to the full space, so "
                "sparse features are penalised rather than flattered."
            ),
        },
        {
            "id": "multi-valued-optimism",
            "severity": "high",
            "text": (
                "Exact-value-set cells overstate resolution for multi-valued "
                "features. Compare resolution_rate_* with probe_* rates; the "
                "probe numbers describe what a single observed value returns. "
                "For features whose values are co-observable (colour + lustre + "
                "indumentum), the probe is a lower bound rather than a point "
                "estimate; the exact set is the upper bound. The truth sits "
                "between them and only the imaging model can settle it."
            ),
        },
        {
            "id": "small-samples",
            "severity": "medium",
            "text": (
                "Pairs and triples with few shared species are noisy; "
                "n_shared is reported beside every joint metric. Underpowered "
                "spotlights are labelled rather than ranked."
            ),
        },
        {
            "id": "image-weighting",
            "severity": "medium",
            "text": (
                "LeafSnap image counts are lab-heavy and species-unbalanced. "
                "Image-weighted metrics are secondary context only."
            ),
        },
        {
            "id": "provenance",
            "severity": "high",
            "text": (
                "Dirr is the primary vocabulary. GoBotany values appear only in "
                "the coverage/conflict check, are labelled by source, and are "
                "never merged into the Dirr assertions."
            ),
        },
    ]


def _feature_assertions_artifact(
    leafsnap: Mapping[str, Any],
    dirr: Mapping[str, Any],
    resolution: Mapping[str, Any],
    matched: Mapping[str, str],
    dirr_species: Sequence[str],
) -> dict[str, Any]:
    species_rows = []
    for name in sorted(matched):
        dirr_name = matched[name]
        values = dirr["feature_values"][dirr_name]
        match = next(m for m in resolution["matches"] if m["leafsnap_species"] == name)
        counts = leafsnap["per_species"][name]
        species_rows.append(
            {
                "leafsnap_species": name,
                "dirr_species": dirr_name,
                "match_rule": match["rule"],
                "match_confidence": match["confidence"],
                "match_note": match["note"],
                "dirr_book_page": dirr["book_pages"].get(dirr_name),
                "lab_images": counts["lab"],
                "field_images": counts["field"],
                "total_images": counts["total"],
                "features": {
                    feature: sorted(
                        values[feature],
                        key=lambda column: dirr["column_order"][column],
                    )
                    for feature in sorted(values)
                },
                "n_asserted_features": len(values),
                "n_asserted_values": sum(len(v) for v in values.values()),
            }
        )
    return {
        "meta": {
            "analysis": "leafsnap-dirr-feature-assertions",
            "version": 1,
            "source": "dirr",
            "value_key": (
                "Feature -> values use the Dirr tag-matrix column names "
                "(e.g. 'Acute (base)', 'Pinnate (venation)') because they are "
                "the assertion keys; display labels are in the feature metadata."
            ),
            "matched_species": len(matched),
            "n_matched_species_with_no_assertions": sum(
                1 for row in species_rows if not row["features"]
            ),
            "matched_species_with_no_assertions": sorted(
                row["leafsnap_species"] for row in species_rows if not row["features"]
            ),
            "unmatched_species": len(resolution["unmatched"]),
            "dirr_species_level_rows": len(dirr_species),
            "inputs": {
                "tags_sha256": dirr["sha256_tags"],
                "vocab_sha256": dirr["sha256_vocab"],
                "manifest_sha256": leafsnap["sha256"],
            },
        },
        "feature_metadata": {
            feature: {
                "label": meta["label"],
                "section": meta["section"],
                "required_organ": meta["required_organ"],
                "views": meta["views"],
                "values": [
                    {
                        "value": entry["column"],
                        "display_label": entry["value_label"],
                        "definition": entry["definition"],
                    }
                    for entry in meta["value_columns"]
                ],
            }
            for feature, meta in sorted(dirr["feature_meta"].items())
        },
        "species": species_rows,
        "unmatched": resolution["unmatched"],
        "ambiguous": resolution["ambiguous"],
        "many_to_one": resolution["many_to_one"],
        "match_audit": [
            m for m in resolution["matches"] if m["rule"] != "exact"
        ],
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


_RULE_NOTES = {
    "orthographic_variant": "punctuation / hyphen / spacing variant of the same epithet",
    "hybrid_name_form": "hybrid marker or spelling variant of the same epithet",
    "single_character_typo": "one-character spelling variant; no other Dirr row is within one edit",
    "infraspecific_parent": "Dirr row is an infraspecific taxon of the same binomial",
}


def write_markdown(results: Mapping[str, Any], assertions: Mapping[str, Any], path: Path) -> None:
    singles = results["single_features_leafsnap"]
    single_by_name = {item["feature"]: item for item in singles}
    pairs = results["pairs"]
    meta = results["meta"]
    resolution = results["name_resolution"]
    n_matched = results["candidate_space"]["n_matched_leafsnap_species"]
    leaf_type = results["spotlights"]["leaf_type"]
    bark = results["spotlights"]["bark"]
    gobotany = results.get("gobotany")
    adapter = results["data_quality"]["adapter_fidelity"]

    def pair_row(pair: Mapping[str, Any]) -> list[Any]:
        return [
            f"`{pair['feature_a']}` x `{pair['feature_b']}`",
            pair["n_shared"],
            num(pair["joint_probe_ig"], 3),
            num(pair["probe_synergy"], 3),
            num(pair["probe_cond_ig_b_given_a"], 3),
            num(pair["joint_ig"], 3),
            pct(pair["probe"]["probe_confirmed_unique_rate_covered"]),
            num(pair["probe"]["median_probe_candidate_set_covered"], 1),
        ]

    pair_headers = [
        "pair",
        "n",
        "probe joint IG",
        "probe synergy",
        "probe cond IG (b|a)",
        "exact joint IG",
        "probe unique res",
        "probe median cand",
    ]

    lines: list[str] = []
    lines.append("# LeafSnap x Dirr EDA — hierarchical field-vision pipeline\n")
    lines.append(
        "Training-free exploratory analysis of the decided hierarchical pipeline: "
        "185 LeafSnap tree species joined to the Dirr descriptive vocabulary, "
        "with single-feature and joint (chordal) discriminative importance.\n"
    )

    lines.append("## Reproduce\n")
    lines.append("```bash")
    lines.append("cd /Users/valkyrie/dev/fieldguidevision-yolo26-3d")
    lines.append(".venv/bin/python experiments/leafsnap-yolo26/eda.py")
    lines.append(".venv/bin/python experiments/leafsnap-yolo26/test_eda.py")
    lines.append("```\n")
    lines.append(
        "The run reads the LeafSnap manifest and the Dirr parquets, writes "
        "`eda_results.json`, `feature_assertions.json` and this report, and is "
        "deterministic (no timestamps, sorted output, fixed thresholds).\n"
    )

    lines.append("## Inputs\n")
    lines.append(
        md_table(
            ["input", "path", "sha256", "size"],
            [
                [
                    "LeafSnap manifest",
                    f"`{meta['inputs']['manifest']['path']}`",
                    f"`{meta['inputs']['manifest']['sha256'][:16]}…`",
                    f"{meta['inputs']['manifest']['rows']:,} rows / "
                    f"{meta['inputs']['manifest']['species']} species",
                ],
                [
                    "Dirr species tags",
                    f"`{meta['inputs']['dirr']['tags_path']}`",
                    f"`{meta['inputs']['dirr']['sha256_tags'][:16]}…`",
                    f"{meta['inputs']['dirr']['rows']:,} rows",
                ],
                [
                    "Dirr vocabulary",
                    f"`{meta['inputs']['dirr']['vocab_path']}`",
                    f"`{meta['inputs']['dirr']['sha256_vocab'][:16]}…`",
                    "97 value columns / 17 features",
                ],
                [
                    "GoBotany (optional)",
                    "`gobotany/extracted_vocabulary/parquet/`",
                    "—",
                    "coverage / conflict check only",
                ],
            ],
        )
    )
    lines.append("")

    lines.append("## Headline findings\n")
    top_single = singles[0]
    best_same = pairs["top_same_view"][0] if pairs["top_same_view"] else None
    best_non_local = pairs["top_non_local"][0] if pairs["top_non_local"] else None
    lines.append(
        f"- **Join.** {resolution['summary']['n_matched']} of "
        f"{resolution['summary']['n_leafsnap_species']} LeafSnap species join to "
        f"Dirr ({pct(resolution['summary']['match_rate'])}): "
        f"{resolution['summary']['rules'].get('exact', 0)} exact + "
        f"{len(results['non_exact_matches'])} audited orthographic/synonym/typo "
        f"matches. {resolution['summary']['n_unmatched']} have no Dirr row and "
        f"{resolution['summary']['n_ambiguous']} are ambiguous."
    )
    lines.append(
        f"- **Best single feature (probe IG, one observed value).** "
        f"`{top_single['feature']}` ({top_single['label']}): probe IG = "
        f"{num(top_single['probe_information_gain'], 3)} bits, probe NMI = "
        f"{num(top_single['probe_nmi'], 3)}, coverage = "
        f"{pct(top_single['coverage']['rate'])}; the guide's exact asserted-set "
        f"upper bound is {num(top_single['information_gain'], 3)} bits. "
        f"Median single-value candidate set = "
        f"{num(top_single['probe']['median_probe_candidate_set_covered'], 1)}."
    )
    if best_same:
        lines.append(
            f"- **Best same-view pair.** `{best_same['feature_a']}` x "
            f"`{best_same['feature_b']}`: probe joint IG = "
            f"{num(best_same['joint_probe_ig'], 3)} bits, probe synergy = "
            f"{num(best_same['probe_synergy'], 3)} bits over the better single "
            f"(n = {best_same['n_shared']})."
        )
    if best_non_local:
        lines.append(
            f"- **Best non-local pair.** `{best_non_local['feature_a']}` "
            f"({', '.join(best_non_local['views_a'])}) x "
            f"`{best_non_local['feature_b']}` "
            f"({', '.join(best_non_local['views_b'])}): probe joint IG = "
            f"{num(best_non_local['joint_probe_ig'], 3)} bits, probe synergy = "
            f"{num(best_non_local['probe_synergy'], 3)} bits (n = "
            f"{best_non_local['n_shared']}). These cannot be read from one crop."
        )
    lines.append(
        f"- **`leaf_type`.** Coverage {leaf_type['single_feature']['coverage']['n_species']}"
        f"/{n_matched} ({pct(leaf_type['single_feature']['coverage']['rate'])}); "
        f"Simple = {leaf_type['counts'].get('Simple', 0)}, Compound = "
        f"{leaf_type['counts'].get('Compound', 0)}, mixed = "
        f"{leaf_type['counts'].get('mixed', 0)}, unlabelled = "
        f"{leaf_type['counts'].get('unlabelled', 0)}. It is a coarse splitter, "
        f"not a species discriminator: probe IG = "
        f"{num(leaf_type['single_feature']['probe_information_gain'], 3)} bits "
        f"(exact-set upper bound "
        f"{num(leaf_type['single_feature']['information_gain'], 3)} bits), and a "
        f"single observed value leaves {leaf_type['single_feature']['value_discrimination'][0]['n_species']} "
        f"(Simple) or {leaf_type['single_feature']['value_discrimination'][1]['n_species']} "
        f"(Compound) candidates."
    )
    if leaf_type["mixed_species"]:
        lines.append(
            f"- **`leaf_type` conflicts.** {len(leaf_type['mixed_species'])} species "
            f"assert Simple and Compound simultaneously: "
            f"{', '.join(f'`{s}`' for s in leaf_type['mixed_species'])}. "
            "A Simple/Compound specialist must exclude or adjudicate them."
        )
    bark_gb = (gobotany or {}).get("characters", {}).get("bark", {})
    bark_text = (
        f"GoBotany's bark-texture character covers "
        f"{bark_gb.get('n_leafsnap_species', 0)} LeafSnap species and is the "
        f"only available bark signal."
        if bark_gb
        else "The GoBotany coverage check was not run."
    )
    lines.append(
        f"- **Bark is absent from Dirr.** {bark['dirr_bark_features'] or 'No bark feature'} "
        f"is mapped, so a leaf x bark non-local pair cannot be evaluated from the "
        f"current Dirr parquets. {bark_text}"
    )
    if adapter.get("available") and adapter.get("n_dropped_assertions"):
        lines.append(
            f"- **Adapter defect.** `DirrDevoAdapter` silently drops "
            f"{adapter['n_dropped_assertions']:,} assertions across "
            f"{adapter['n_species_with_dropped_assertions']} matched species "
            f"({adapter['dropped_tag_column_count']} of {adapter['n_tag_columns']} "
            f"tag columns: "
            f"{', '.join(adapter['dropped_tag_columns'])}; affected features "
            f"{', '.join(adapter['dropped_features'])}). This EDA uses a "
            f"column-accurate mapping; the adapter should be fixed before the "
            f"pipeline consumes it."
        )
    lines.append("")
    lines.append(
        "> **Fusion warning.** Non-local joint features are real information, but "
        "they are only available after fusing separate observations of the same "
        "plant (leaf obverse, leaf reverse, whole plant, flower). Never assume a "
        "non-local pair is simultaneously visible in one crop, and never train a "
        "single-crop model to emit both sides of one. Same-view pairs are the only "
        "joint features a single image can support.\n"
    )

    lines.append("## 1. Species join and image counts\n")
    lines.append(
        md_table(
            ["metric", "value"],
            [
                ["LeafSnap rows", f"{results['leafsnap']['n_rows']:,}"],
                ["lab rows", f"{results['leafsnap']['lab_rows']:,}"],
                ["field rows", f"{results['leafsnap']['field_rows']:,}"],
                ["LeafSnap species", results["leafsnap"]["n_species"]],
                ["matched to Dirr", f"{n_matched} ({pct(n_matched / results['leafsnap']['n_species'])})"],
                ["unmatched", resolution["summary"]["n_unmatched"]],
                ["ambiguous", resolution["summary"]["n_ambiguous"]],
                ["many-to-one", resolution["summary"]["n_many_to_one"]],
                ["Dirr species-level rows", results["candidate_space"]["n_dirr_species_level"]],
                ["Dirr genus-only rows", f"{len(results['dirr']['genus_only_rows'])}"],
            ],
        )
    )
    lines.append("")
    split = results["image_counts_by_join"]
    lines.append(
        md_table(
            ["join status", "species", "lab images", "field images", "total images"],
            [
                [
                    label.replace("_", " "),
                    data["species"],
                    f"{data['lab']:,}",
                    f"{data['field']:,}",
                    f"{data['total']:,}",
                ]
                for label, data in split.items()
            ],
        )
    )
    lines.append("")
    lines.append(
        "Match rules used: exact normalized name, normalized genus+epithet key "
        "(hyphen / hybrid-mark / spacing variants), infraspecific parent, "
        "curated synonym, one-character typo, truncated epithet. Every match "
        "must be unique; anything else is reported as ambiguous or unmatched.\n"
    )
    empty = results["data_quality"]["matched_species_with_no_assertions"]
    if empty:
        lines.append(
            f"**Matched but empty.** {len(empty)} matched species have a Dirr row "
            f"with no asserted values at all: "
            + ", ".join(f"`{name}`" for name in empty)
            + ". They stay in the denominator (coverage is deployment-honest) but "
            "cannot be keyed out by any feature. Dirr's `Malus` and `Clematis` "
            "entries are genus-only and cannot substitute for species-level "
            "identification; the matched `Malus` species rows are empty."
        )
        lines.append("")
    lines.append("### Audited non-exact matches\n")
    lines.append(
        md_table(
            ["LeafSnap", "Dirr", "rule", "confidence", "note"],
            [
                [
                    f"`{m['leafsnap_species']}`",
                    f"`{m['dirr_species']}`",
                    m["rule"],
                    m["confidence"],
                    m["note"] or _RULE_NOTES.get(m["rule"], m["rule"]),
                ]
                for m in results["non_exact_matches"]
            ],
        )
    )
    lines.append("")
    lines.append("### Unmatched LeafSnap species\n")
    unmatched_rows = []
    for item in resolution["unmatched"]:
        near = item["same_genus_near_misses"]
        suggestion = item["suggestion_not_applied"]
        hint = ""
        if suggestion:
            hint = f"suggested but not applied: {suggestion['dirr_species']}"
        elif near:
            hint = "nearest same-genus: " + ", ".join(
                f"{n['dirr_species']} (d={n['edit_distance']})" for n in near[:2]
            )
        unmatched_rows.append(
            [
                f"`{item['leafsnap_species']}`",
                hint or "no same-genus candidate within edit distance 2",
            ]
        )
    lines.append(md_table(["LeafSnap", "diagnostic"], unmatched_rows))
    lines.append("")
    exact_simple = sum(
        1
        for s in assertions["species"]
        if s["match_rule"] == "exact" and s["features"].get("leaf_type") == ["Simple"]
    )
    exact_compound = sum(
        1
        for s in assertions["species"]
        if s["match_rule"] == "exact" and s["features"].get("leaf_type") == ["Compound"]
    )
    lines.append(
        f"No unmatched species is force-joined. The audited non-exact matches are "
        f"listed above so a director can accept or reject them. Reconciliation "
        f"with `DATASETS.md`: {resolution['summary']['rules'].get('exact', 0)} exact "
        f"matches, of which {exact_simple} are Simple-only and {exact_compound} "
        f"Compound-only — the {exact_simple + exact_compound} species the dataset "
        f"builder could use; the remaining exact matches are 2 mixed and 40 with no "
        f"Simple/Compound label. The 14 audited name matches are additional and do "
        f"not change that builder audit unless accepted.\n"
    )

    lines.append("## 2. Feature coverage (matched LeafSnap species, ordered by probe IG)\n")
    lines.append(
        md_table(
            ["feature", "label", "organ / view", "values", "coverage", "multi-valued", "used"],
            [
                [
                    f"`{item['feature']}`",
                    item["label"],
                    f"{item['required_organ']} / {', '.join(item['views'])}",
                    f"{item['n_values_defined']}",
                    f"{item['coverage']['n_species']}/{n_matched} ({pct(item['coverage']['rate'])})",
                    pct(item["multi_valued"]["rate_of_covered"]),
                    f"{item['n_values_used']}",
                ]
                for item in singles
            ],
        )
    )
    lines.append("")
    lines.append(
        "Coverage is the ceiling on every other metric. `leaf_venation`, "
        "`leaf_texture`, `leaf_surface_bottom`, `leaf_type_conifer` and "
        "`phenology_types` are sparse in Dirr; a feature with high purity on "
        "18% of species is not a general key. `compound_leaf_type` and "
        "`pinnate_types` are **conditional**: they apply only to compound (or "
        "bipinnate) species, so their 165-denominator coverage understates "
        "them — 19 of the 21 compound-labelled species carry a Pinnate/Palmate "
        "sub-type. Multi-valued rates matter too: a species with more than one "
        "asserted value cannot be treated as a single label.\n"
    )

    lines.append("## 3. Why these metrics\n")
    for key in (
        "coverage",
        "information_gain",
        "nmi",
        "conditional_entropy",
        "probe_information_gain",
        "unique_resolution",
        "median_candidate_set",
        "probe_candidate_set",
        "image_weighted",
        "joint_ig",
        "conditional_ig",
        "synergy",
        "redundancy",
    ):
        lines.append(f"- **{key}** — {METRIC_DEFINITIONS[key]}")
    lines.append("")
    lines.append(
        "Ranking is by species-balanced **probe IG** (one observed value); every "
        "other metric is shown beside it so the ranking is inspectable rather "
        "than a score. The exact asserted-set IG is the guide's upper bound, and "
        "`H(Y|F)` / NMI in the table are the probe variants. Image-weighted "
        "numbers are secondary and never used to rank.\n"
    )

    lines.append("## 4. Top single features\n")
    lines.append(
        md_table(
            [
                "rank",
                "feature",
                "cover",
                "probe IG",
                "probe NMI",
                "guide IG (upper bound)",
                "probe H(Y|F)/H(Y)",
                "unique res (cov/all)",
                "probe median cand",
                "image-wt probe IG",
            ],
            [
                [
                    item["rank"],
                    f"`{item['feature']}`",
                    pct(item["coverage"]["rate"]),
                    num(item["probe_information_gain"], 3),
                    num(item["probe_nmi"], 3),
                    num(item["information_gain"], 3),
                    num(item["probe_conditional_entropy_ratio"], 3),
                    f"{pct(item['candidate_sets']['resolution_rate_covered'])} / "
                    f"{pct(item['candidate_sets']['resolution_rate_all'])}",
                    num(item["probe"]["median_probe_candidate_set_covered"], 1),
                    num(item["image_weighted"]["probe_information_gain"], 3),
                ]
                for item in singles[:12]
            ],
        )
    )
    lines.append("")
    top_five = [item["probe_information_gain"] for item in singles[:5]]
    lines.append(
        f"The top five features are within "
        f"{num(100 * (1 - min(top_five) / max(top_five)), 0)}% of each other in "
        f"probe IG ({num(max(top_five), 2)} to {num(min(top_five), 2)} bits): no "
        f"single Dirr feature is a species key. The hierarchy has to combine "
        f"them."
    )
    lines.append(
        "`probe IG` is the primary ranking key: it is the information from ONE "
        "observed value, with each species' weight split over the values Dirr "
        "asserts. `guide IG` is the exact asserted-set upper bound (a perfect "
        "multi-label observation) and is shown only for contrast. `unique res` "
        "is the exact-cell fraction of species alone in their cell; the "
        "parenthesised second number counts unobserved species as unresolved. "
        "`probe median cand` is the candidate set a single observed value "
        "leaves. For `leaf_shape_overall` the exact cell looks near-unique "
        "(median 5) while a single shape leaves 52 candidates — the probe "
        "numbers are the deployment numbers. For `leaf_surface_top` the two "
        "bounds bracket the truth: its 9 values mix colour, lustre and "
        "indumentum, so one observation may return several tags (probe IG is a "
        "conservative lower bound) while the exact set assumes all of them are "
        "known at once (optimistic upper bound).\n"
    )
    lines.append(
        "Secondary context (all 967 Dirr species-level rows, species-balanced "
        "probe IG): "
        + ", ".join(
            f"`{item['feature']}` {num(item['probe_information_gain'], 3)} bits"
            for item in results["single_features_dirr_universe"][:5]
        )
        + ".\n"
    )

    lines.append("## 5. Joint (chordal) pairs\n")
    lines.append(
        f"{pairs['n_eligible']} pairs met the eligibility rule (each feature "
        f"covers >= {pct(MIN_FEATURE_COVERAGE, 0)} of matched species and "
        f"shared coverage is >= {pct(MIN_PAIR_SHARED_COVERAGE, 0)}). "
        + ", ".join(f"{count} {name}" for name, count in pairs["relation_counts"].items())
        + ".\n"
    )
    lines.append("### Same-view pairs (fusable in one crop)\n")
    lines.append(
        md_table(pair_headers, [pair_row(pair) for pair in pairs["top_same_view"][:10]])
    )
    lines.append("")
    lines.append("### Non-local pairs (require fusion across observations / tracks)\n")
    lines.append(
        md_table(pair_headers, [pair_row(pair) for pair in pairs["top_non_local"][:10]])
    )
    lines.append("")
    lines.append(
        "All joint numbers use the same single-observation model as the single "
        "features: each species' weight is split over its asserted value "
        "combinations, so multi-valued rows cannot inflate a pair. `probe "
        "synergy` = joint probe IG - max(single probe IG) and is the only "
        "evidence that the second feature earns its observation; zero means one "
        "feature subsumes the other. `probe cond IG (b|a)` = joint probe IG - "
        "probe IG(a), the incremental bits of the alphabetically-second feature "
        "after the first (synergy is the smaller of the two directions). `exact "
        "joint IG` is the guide's asserted-set upper bound. For non-local pairs "
        "these bits exist only after fusing separate views — they are not a "
        "single-image model target.\n"
    )
    if pairs["underpowered_spotlights"]:
        lines.append("### Structurally interesting but underpowered\n")
        lines.append(
            md_table(
                ["pair", "views", "n shared", "probe joint IG", "probe synergy", "status"],
                [
                    [
                        f"`{p['feature_a']}` x `{p['feature_b']}`",
                        " / ".join(sorted(set(p["views_a"]) | set(p["views_b"]))),
                        p["n_shared"],
                        num(p["joint_probe_ig"], 3),
                        num(p["probe_synergy"], 3),
                        "below coverage threshold — not ranked",
                    ]
                    for p in pairs["underpowered_spotlights"]
                ],
            )
        )
        lines.append("")

    lines.append("## 6. Triples (small, feasibility check)\n")
    triples = results["triples"]
    lines.append(
        f"{triples['n_evaluated']} triples evaluated over the "
        f"{len(triples['features_considered'])} best-covered features; every "
        f"pair inside a triple also passed the pair rule. "
        f"{triples['n_fully_non_local']} fully non-local triple passed the 3-way "
        f"coverage rule, so the top triples are leaf-view combinations plus at "
        f"most one whole-plant feature. Triples are reported for feasibility, "
        f"not as an observation plan.\n"
    )
    lines.append(
        md_table(
            [
                "features",
                "views",
                "probe joint IG",
                "probe marginal over best pair",
                "probe synergy vs best single",
            ],
            [
                [
                    " x ".join(f"`{f}`" for f in item["features"]),
                    f"{item['n_distinct_views']} ({', '.join(item['views'])})",
                    num(item["joint_probe_ig"], 3),
                    num(item["marginal_probe_over_best_pair"], 3),
                    num(item["probe_synergy_over_best_single"], 3),
                ]
                for item in triples["top_by_joint_probe_ig"][:5]
            ],
        )
    )
    lines.append("")

    lines.append("## 7. Feature spotlights\n")
    lines.append("### `leaf_type` (decided specialist: validate labels, do not re-rank)\n")
    lines.append(
        md_table(
            ["class", "species", "lab images", "field images", "total images"],
            [
                [
                    key,
                    value["species"],
                    f"{value['lab_images']:,}",
                    f"{value['field_images']:,}",
                    f"{value['lab_images'] + value['field_images']:,}",
                ]
                for key, value in sorted(
                    leaf_type["image_counts"].items(),
                    key=lambda item: ["Simple", "Compound", "mixed", "unlabelled"].index(item[0]),
                )
            ],
        )
    )
    lines.append("")
    lines.append(
        "`Simple` and `Compound` are the specialist classes; `mixed` and "
        "`unlabelled` are exclusion statuses, not classes.\n"
    )
    lines.append(
        f"- Value discrimination (species per asserted value): "
        + ", ".join(
            f"**{entry['display_label']}** {entry['n_species']}"
            for entry in leaf_type["value_discrimination"]
        )
        + "."
    )
    lines.append(
        f"- Probe IG = {num(leaf_type['single_feature']['probe_information_gain'], 3)} bits "
        f"(exact asserted-set upper bound "
        f"{num(leaf_type['single_feature']['information_gain'], 3)} bits); probe NMI = "
        f"{num(leaf_type['single_feature']['probe_nmi'], 3)}; probe H(Y|F) = "
        f"{num(leaf_type['single_feature']['probe_conditional_entropy'], 3)} bits "
        f"({pct(leaf_type['single_feature']['probe_conditional_entropy_ratio'])} of H(Y)); "
        f"confirmed-unique probe rate = "
        f"{pct(leaf_type['single_feature']['probe']['probe_confirmed_unique_rate_covered'])} "
        f"(a Simple/Compound split never isolates a species)."
    )
    lines.append(
        f"- The Simple group is "
        f"{leaf_type['value_discrimination'][0]['n_species'] if leaf_type['value_discrimination'] else 0} "
        f"species (Simple-only plus the mixed rows) and the Compound group is "
        f"{leaf_type['value_discrimination'][1]['n_species'] if len(leaf_type['value_discrimination']) > 1 else 0}; "
        f"{leaf_type['counts'].get('unlabelled', 0)} matched species have no "
        f"Simple/Compound label (mostly conifers)."
    )
    lines.append(
        f"- Compound sub-type coverage: "
        f"{leaf_type['compound_subtype_coverage']['n_with_subtype']}/"
        f"{leaf_type['compound_subtype_coverage']['n_compound']} compound species "
        f"({pct(leaf_type['compound_subtype_coverage']['rate'])}) carry "
        f"Pinnate/Palmate, so a second-stage compound split is only partly labelled."
    )
    if leaf_type["mixed_species"]:
        lines.append(
            "- **Conflicts:** "
            + ", ".join(f"`{name}`" for name in leaf_type["mixed_species"])
            + " assert both Simple and Compound. These are the rows the specialist "
            "must exclude or adjudicate; they are not a reason to choose another "
            "specialist."
        )
    if leaf_type["simple_with_compound_type"]:
        lines.append(
            "- Orphan assertions (compound sub-type on a Simple-only species): "
            + ", ".join(f"`{name}`" for name in leaf_type["simple_with_compound_type"])
        )
    lines.append("")

    for feature, heading in (
        ("leaf_arrangement", "`leaf_arrangement`"),
        ("leaf_shape_margins", "`leaf_shape_margins`"),
        ("leaf_venation", "`leaf_venation`"),
        ("leaf_surface_top", "`leaf_surface_top`"),
        ("leaf_surface_bottom", "`leaf_surface_bottom`"),
        ("growth_habit_types", "`growth_habit_types` (whole plant)"),
        ("inflorescences", "`inflorescences` (flower)"),
    ):
        item = results["spotlights"][feature]
        lines.append(f"### {heading}\n")
        lines.append(
            f"Coverage {item['coverage']['n_species']}/{n_matched} "
            f"({pct(item['coverage']['rate'])}); multi-valued "
            f"{pct(item['multi_valued']['rate_of_covered'])}; probe IG "
            f"{num(item['probe_information_gain'], 3)} bits (guide upper bound "
            f"{num(item['information_gain'], 3)}); probe NMI "
            f"{num(item['probe_nmi'], 3)}; median single-value candidate set "
            f"{num(item['probe']['median_probe_candidate_set_covered'], 1)}; "
            f"unique resolution {pct(item['candidate_sets']['resolution_rate_covered'])} "
            f"covered / {pct(item['candidate_sets']['resolution_rate_all'])} all.\n"
        )
        if item.get("interpretation"):
            lines.append(f"Vocabulary note: {item['interpretation']}")
        if item.get("low_coverage"):
            lines.append(
                f"**Low coverage (<10%): {pct(item['coverage']['rate'])} — this "
                f"feature cannot function as a primary key on this dataset."
            )
        lines.append("")
        lines.append(
            "Top values: "
            + ", ".join(
                f"**{entry['display_label']}** {entry['n_species']}"
                for entry in item["value_discrimination"][:8]
            )
            + "."
        )
        if item.get("contradictions"):
            lines.append(
                "\nExclusive-value conflicts (may reflect within-species variation): "
                + ", ".join(
                    f"`{hit['species']}` ({' + '.join(hit['contradiction'])})"
                    for hit in item["contradictions"][:10]
                )
                + "."
            )
        lines.append("")

    lines.append("### `bark`\n")
    lines.append(bark["dirr_note"] + "\n")

    lines.append("## 8. GoBotany coverage / conflict check (provenance separate)\n")
    if not gobotany:
        lines.append("GoBotany check not run.\n")
    else:
        lines.append(
            f"GoBotany covers {gobotany['n_leafsnap_in_gobotany']} of 185 LeafSnap "
            f"species ({pct(gobotany['leafsnap_coverage_rate'])}) at the taxon "
            f"level; comparisons below only use species where BOTH sources give "
            f"exactly one comparable value.\n"
        )
        lines.append(
            md_table(
                ["feature", "GoBotany character", "GoBotany coverage", "compared", "agree", "conflict", "agreement"],
                [
                    [
                        feature,
                        f"`{data['gobotany_character']}`",
                        f"{gobotany['characters'][feature]['n_leafsnap_species']} "
                        f"({pct(gobotany['characters'][feature]['coverage_rate_of_leafsnap'])})",
                        data["n_compared_unambiguous"],
                        data["n_agree"],
                        data["n_conflict"],
                        pct(data["agreement_rate"]) if data["agreement_rate"] is not None else "-",
                    ]
                    for feature, data in gobotany["comparisons"].items()
                ],
            )
        )
        lines.append("")
        for feature, data in gobotany["comparisons"].items():
            if data["conflicts"]:
                lines.append(
                    f"**{feature} conflicts** (Dirr vs GoBotany): "
                    + "; ".join(
                        f"`{item['leafsnap_species']}` {item['dirr']} vs {item['gobotany']}"
                        for item in data["conflicts"][:10]
                    )
                    + "."
                )
        lines.append("")
        lines.append(
            "View assignment is a heuristic from each feature's definition: "
        "leaf_obverse / leaf_reverse are different faces of the same leaf, "
        "flower is the reproductive structure, and whole_plant is habit or "
        "phenology. A whole-plant photo can show leaf arrangement at a "
        "distance, but it cannot resolve the close-up blade traits, so "
        "leaf x whole_plant is non-local for this pipeline. "
        "The margin comparison is coarse (entire / lobed / toothed): a Dirr "
            "`Entire` row and a GoBotany `lobed` row can describe the same leaf "
            "at different granularity (lobed blades often have entire lobe "
            "margins), so those conflicts are not proof of a data error. "
            "Venation compares zero species because Dirr's venation assertions "
            "are too sparse and `Reticulate` has no GoBotany analogue."
        )
        lines.append("")
        lines.append(
            "GoBotany's bark character values: "
            + ", ".join(
                f"{value} ({count})"
                for value, count in gobotany["characters"]["bark"]["value_counts"].items()
            )
            + ". These are not merged into Dirr assertions.\n"
        )

    lines.append("## 9. Data quality and adapter defects\n")
    dq = results["data_quality"]
    lines.append(
        f"- {len(dq['vocab_columns_without_metadata'])} Dirr vocabulary columns "
        f"carry no feature metadata: "
        + ", ".join(f"`{c}`" for c in dq["vocab_columns_without_metadata"])
        + ". The EDA maps them explicitly; `DirrDevoAdapter.list_features` drops them."
    )
    lines.append(
        f"- {len(dq['suffixed_tag_columns'])} tag columns use a disambiguating "
        f"suffix, e.g. "
        + ", ".join(f"`{c}`" for c in dq["suffixed_tag_columns"][:6])
        + ". The adapter currently keys by the column name, which is safe; a "
        "consumer that keys by the vocabulary `value_label` row instead would "
        "misattribute them (the labels repeat across features)."
    )
    if dq["duplicate_value_labels"]:
        lines.append(
            "- Duplicated value labels across features: "
            + "; ".join(
                f"`{label}` -> {', '.join(entries)}"
                for label, entries in dq["duplicate_value_labels"].items()
            )
            + "."
        )
    if adapter.get("available"):
        lines.append(
            f"- Adapter fidelity: `DirrDevoAdapter.get_taxon` loses "
            f"{adapter['n_dropped_assertions']:,} assertions across "
            f"{adapter['n_species_with_dropped_assertions']} matched species "
            f"({adapter['dropped_tag_column_count']}/{adapter['n_tag_columns']} "
            f"columns). Affected features: "
            + ", ".join(f"`{f}`" for f in adapter["dropped_features"])
            + "."
        )
        if adapter.get("organ_mapping_mismatches"):
            lines.append(
                "- Organ mapping drift vs `_DIRR_FEATURE_ORGAN`: "
                + ", ".join(adapter["organ_mapping_mismatches"])
            )
    lines.append(
        f"- Exclusive-value checks flagged {results['conflicts']['n_species_with_any_contradiction']} "
        f"species across the explicit rules; see "
        f"`eda_results.json -> conflicts` for the full list. `Entire` + `Lobed` "
        "is deliberately not flagged: lobed blades can have entire lobe margins."
    )
    if results["conflicts"]["orphan_assertions"]:
        lines.append(
            f"- {len(results['conflicts']['orphan_assertions'])} orphan compound-only "
            "assertions on Simple-leaved species."
        )
    lines.append("")

    lines.append("## 10. Warnings and limits\n")
    for warning in results["warnings"]:
        lines.append(f"- **{warning['id']}** ({warning['severity']}) — {warning['text']}")
    lines.append("")
    lines.append(
        f"Additional limits: Dirr's vocabulary is US-ornamental-woody oriented, so "
        f"absent species are a source-coverage fact, not a taxonomy error; the "
        f"Dirr-universe metrics are context for the eventual guide, while all "
        f"ranked tables use the {n_matched} matched LeafSnap species; "
        f"leaf_surface_bottom is too sparse to rank as a pair (17% coverage, 26 "
        f"shared species), so its top/bottom non-local contribution is reported "
        f"as an underpowered spotlight rather than a recommendation. The GoBotany "
        f"margin comparison is coarse (entire / lobed / toothed): `Entire` vs "
        f"`lobed` can describe the same leaf at different granularity and is not "
        f"proof of a data conflict. Dirr venation has no GoBotany-comparable "
        f"unambiguous pair, so that row compares zero species.\n"
    )

    lines.append("## 11. Artifacts\n")
    lines.append(
        md_table(
            ["artifact", "contents"],
            [
                ["`eda_results.json`", "all metrics, join audit, pairs, triples, spotlights, warnings"],
                ["`feature_assertions.json`", "species -> feature -> values with provenance and match audit"],
                ["`EDA.md`", "this report"],
                ["`eda.py`", "reproducible analysis"],
                ["`test_eda.py`", "toy self-checks for entropy / candidate-set / join math"],
            ],
        )
    )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_manifest() -> Path:
    env = os.environ.get("LEAFSNAP_MANIFEST")
    return Path(env) if env else DEFAULT_MANIFEST


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, default=_default_manifest())
    parser.add_argument("--dirr-dir", type=Path, default=DEFAULT_DIRR_DIR)
    parser.add_argument("--gobotany-dir", type=Path, default=DEFAULT_GOBOTANY_DIR)
    parser.add_argument("--no-gobotany", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)

    results, assertions = run_eda(
        manifest_path=args.manifest,
        dirr_dir=args.dirr_dir,
        gobotany_dir=args.gobotany_dir,
        include_gobotany=not args.no_gobotany,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "eda_results.json"
    assertions_path = out_dir / "feature_assertions.json"
    markdown_path = out_dir / "EDA.md"

    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    assertions_path.write_text(
        json.dumps(assertions, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(results, assertions, markdown_path)

    top = results["single_features_leafsnap"][:5]
    best_same = results["pairs"]["top_same_view"][0] if results["pairs"]["top_same_view"] else None
    best_non_local = results["pairs"]["top_non_local"][0] if results["pairs"]["top_non_local"] else None
    print(
        f"matched {results['name_resolution']['summary']['n_matched']}/"
        f"{results['name_resolution']['summary']['n_leafsnap_species']} LeafSnap species"
    )
    print("top single features (species-balanced probe IG):")
    for item in top:
        print(
            f"  {item['rank']}. {item['feature']:<24} "
            f"probeIG={item['probe_information_gain']:.3f} "
            f"probeNMI={item['probe_nmi']:.3f} guideIG={item['information_gain']:.3f} "
            f"cover={item['coverage']['rate']:.1%} "
            f"probeCand={item['probe']['median_probe_candidate_set_covered']:.1f}"
        )
    if best_same:
        print(
            f"best same-view pair: {best_same['feature_a']} x {best_same['feature_b']} "
            f"(probe joint={best_same['joint_probe_ig']:.3f}, "
            f"probe synergy={best_same['probe_synergy']:.3f}, n={best_same['n_shared']})"
        )
    if best_non_local:
        print(
            f"best non-local pair: {best_non_local['feature_a']} x {best_non_local['feature_b']} "
            f"(probe joint={best_non_local['joint_probe_ig']:.3f}, "
            f"probe synergy={best_non_local['probe_synergy']:.3f}, "
            f"n={best_non_local['n_shared']})"
        )
    print(f"wrote {results_path}")
    print(f"wrote {assertions_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
