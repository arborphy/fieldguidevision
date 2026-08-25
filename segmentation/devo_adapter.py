"""DeVo-agnostic vocabulary adapter protocol for Field Guide Vision.

Any descriptive-vocabulary source (Dirr, GoBotany, Newcomb, …) is normalized into
one shape the statement compiler consumes: taxa, each carrying the feature values
that source asserts for it, and each feature value mapped to the organ required
to assess it. The compiler and the agentic resolver read only these types — they
never see a source-specific schema.

Dirr is the primary source for the P17 experiment (leaf-focused, curated,
self-describing value definitions). GoBotany is the comparison baseline via the
existing ``gobotany_devo_adapter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# Normalized vocabulary types (source-agnostic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedFeatureValue:
    """One (feature, value) pair a source asserts, with its grounding context."""

    feature_name: str  # e.g. "leaf_shape_margins"
    feature_label: str  # e.g. "Leaf Blade Shape: Margins"
    feature_section: str  # e.g. "PLANT MORPHOLOGY / LEAF MORPHOLOGY"
    value_label: str  # e.g. "Serrate"
    value_definition: str | None  # e.g. "saw-toothed, the teeth pointing forward."
    required_organ: str  # canonical organ needed to assess it, e.g. "leaf blade"


@dataclass(frozen=True)
class ResolvedTaxon:
    """A taxon and the feature values the source asserts for it."""

    scientific_name: str
    common_name: str | None
    family: str | None
    feature_values: tuple[ResolvedFeatureValue, ...]


@runtime_checkable
class DeVoAdapter(Protocol):
    """Protocol every source vocabulary adapter implements."""

    @property
    def devo_name(self) -> str: ...

    def list_features(self) -> tuple[ResolvedFeatureValue, ...]:
        """Every (feature, value) the source defines, with required organ."""
        ...

    def resolve_taxa_by_feature_values(
        self, value_labels: Sequence[str]
    ) -> tuple[ResolvedTaxon, ...]:
        """Taxa the source asserts carry ALL of the given value labels."""
        ...

    def get_taxon(self, scientific_name: str) -> ResolvedTaxon | None:
        """One taxon with its asserted feature values, or None."""
        ...


# ---------------------------------------------------------------------------
# Organ mapping (Dirr primary; other sources reuse the canonical organ names)
# ---------------------------------------------------------------------------

# Canonical organ names align with pipeline.DEFAULT_PROMPTS / notice categories.
ORGAN_WHOLE_PLANT = "whole plant"
ORGAN_LEAF_BLADE = "leaf blade"
ORGAN_REPRODUCTIVE = "flower"  # reproductive structure / inflorescence carrier

# Dirr feature_name -> organ required to assess it. Leaf morphology all hangs off
# the leaf blade; classification features (habit, phenology) need the whole plant.
_DIRR_FEATURE_ORGAN: dict[str, str] = {
    "growth_habit_types": ORGAN_WHOLE_PLANT,
    "phenology_types": ORGAN_WHOLE_PLANT,
    "leaf_arrangement": ORGAN_LEAF_BLADE,
    "leaf_type": ORGAN_LEAF_BLADE,
    "compound_leaf_type": ORGAN_LEAF_BLADE,
    "leaf_venation": ORGAN_LEAF_BLADE,
    "leaf_shape_overall": ORGAN_LEAF_BLADE,
    "leaf_shape_margins": ORGAN_LEAF_BLADE,
    "leaf_shape_lobe_type": ORGAN_LEAF_BLADE,
    "leaf_shape_base": ORGAN_LEAF_BLADE,
    "leaf_shape_apex": ORGAN_LEAF_BLADE,
    "leaf_surface_top": ORGAN_LEAF_BLADE,
    "leaf_surface_bottom": ORGAN_LEAF_BLADE,
    "leaf_texture": ORGAN_LEAF_BLADE,
    "leaf_type_conifer": ORGAN_LEAF_BLADE,
    "pinnate_types": ORGAN_LEAF_BLADE,
    "inflorescences": ORGAN_REPRODUCTIVE,
}


# ---------------------------------------------------------------------------
# Dirr adapter (primary) — normalizes the wide one-hot parquets
# ---------------------------------------------------------------------------


class DirrDevoAdapter:
    """Normalizes Dirr's wide one-hot parquets into the ResolvedFeature protocol.

    ``dirr_vocabulary.parquet`` is self-describing: each value column carries its
    ``feature_name`` / ``feature_label`` / ``feature_section`` /
    ``value_definition`` across ``row_type`` rows. ``dirr_species_tags.parquet``
    is the species × value one-hot matrix (1 = asserted).
    """

    def __init__(self, parquet_dir: str | Path | None = None) -> None:
        base = Path(parquet_dir) if parquet_dir else (
            Path(__file__).resolve().parent.parent
            / "manual_of_woody_plants"
            / "extracted_vocabulary"
            / "parquet"
        )
        self.parquet_dir = base
        self._vocab_path = base / "dirr_vocabulary.parquet"
        self._tags_path = base / "dirr_species_tags.parquet"
        self._vernacular_path = base / "dirr_taxa_vernacular.parquet"
        self._feature_values: tuple[ResolvedFeatureValue, ...] | None = None

    @property
    def devo_name(self) -> str:
        return "dirr"

    # -- vocabulary ---------------------------------------------------------

    def list_features(self) -> tuple[ResolvedFeatureValue, ...]:
        if self._feature_values is not None:
            return self._feature_values
        import duckdb

        con = duckdb.connect(":memory:")
        df = con.execute(f"SELECT * FROM '{self._vocab_path.as_posix()}'").df()
        con.close()
        # Transpose: rows are row_type records, columns are value labels. The value
        # label is the original column name (the index after transpose); there is
        # also a ``value_label`` row_type row carrying the same string.
        df = df.set_index("row_type").T
        df.index.name = "value_col"
        df = df.reset_index()

        values: list[ResolvedFeatureValue] = []
        for _, row in df.iterrows():
            feature_name = str(row.get("feature_name") or "").strip()
            value_label = str(row.get("value_col") or "").strip()
            if not feature_name or not value_label:
                continue
            organ = _DIRR_FEATURE_ORGAN.get(feature_name, ORGAN_WHOLE_PLANT)
            values.append(
                ResolvedFeatureValue(
                    feature_name=feature_name,
                    feature_label=str(row.get("feature_label") or feature_name),
                    feature_section=str(row.get("feature_section") or ""),
                    value_label=value_label,
                    value_definition=(
                        str(row["value_definition"]).strip()
                        if row.get("value_definition") and str(row.get("value_definition")) != "nan"
                        else None
                    ),
                    required_organ=organ,
                )
            )
        self._feature_values = tuple(values)
        return self._feature_values

    # -- taxa ---------------------------------------------------------------

    def _asserted_value_labels(self, species_row) -> tuple[str, ...]:
        """Value columns asserted (==1) for one species_tags row."""
        labels = []
        for col, val in species_row.items():
            if col in ("species", "book_page"):
                continue
            try:
                if int(val) == 1:
                    labels.append(str(col))
            except (TypeError, ValueError):
                continue
        return tuple(labels)

    def _build_taxon(self, scientific_name: str, value_labels: Sequence[str]) -> ResolvedTaxon:
        vocab = {v.value_label: v for v in self.list_features()}
        fvs = tuple(vocab[lbl] for lbl in value_labels if lbl in vocab)
        return ResolvedTaxon(
            scientific_name=scientific_name,
            common_name=self._common_name(scientific_name),
            family=None,  # Dirr parquets here do not carry family; resolved via backbone if needed.
            feature_values=fvs,
        )

    def _common_name(self, scientific_name: str) -> str | None:
        if not self._vernacular_path.exists():
            return None
        import duckdb

        con = duckdb.connect(":memory:")
        row = con.execute(
            f"SELECT common_names FROM '{self._vernacular_path.as_posix()}' "
            "WHERE lower(taxon_name) = ? LIMIT 1",
            [scientific_name.lower()],
        ).fetchone()
        con.close()
        return str(row[0]) if row and row[0] else None

    def get_taxon(self, scientific_name: str) -> ResolvedTaxon | None:
        import duckdb

        con = duckdb.connect(":memory:")
        df = con.execute(
            f"SELECT * FROM '{self._tags_path.as_posix()}' WHERE lower(species) = ? LIMIT 1",
            [scientific_name.lower()],
        ).df()
        con.close()
        if df.empty:
            return None
        return self._build_taxon(scientific_name, self._asserted_value_labels(df.iloc[0]))

    def resolve_taxa_by_feature_values(
        self, value_labels: Sequence[str]
    ) -> tuple[ResolvedTaxon, ...]:
        """Species whose one-hot row asserts every requested value label."""
        import duckdb

        if not value_labels:
            return ()
        # Build a WHERE clause: each requested value column must equal 1.
        con = duckdb.connect(":memory:")
        conditions = " AND ".join(f'"{lbl}" = 1' for lbl in value_labels)
        df = con.execute(
            f"SELECT * FROM '{self._tags_path.as_posix()}' WHERE {conditions}"
        ).df()
        con.close()
        taxa = [
            self._build_taxon(str(row["species"]), self._asserted_value_labels(row))
            for _, row in df.iterrows()
        ]
        return tuple(taxa)
