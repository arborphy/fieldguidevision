"""GoBotany Descriptive Vocabulary (DeVo) Query Adapter for Field Guide Vision.

Fast query layer over GoBotany parquet datasets in `gobotany/extracted_vocabulary/parquet/`
extracting candidate taxa, discriminative characters, and Grounding DINO prompt vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import duckdb


@dataclass(frozen=True)
class GoBotanyCharacterValue:
    character_name: str
    friendly_name: str
    value_label: str
    friendly_text: str | None
    character_group: str | None


@dataclass(frozen=True)
class GoBotanyTaxonCandidate:
    taxon_id: int
    scientific_name: str
    common_name: str | None
    genus: str
    family: str | None
    pile_slug: str | None
    discriminative_characters: tuple[GoBotanyCharacterValue, ...]
    prompt_vocabulary: tuple[str, ...]


HENRICO_VA_COMMON_YARD_TAXA = (
    ("Quercus alba", "White Oak", "Fagaceae", "woody-angiosperms"),
    ("Quercus rubra", "Northern Red Oak", "Fagaceae", "woody-angiosperms"),
    ("Acer rubrum", "Red Maple", "Sapindaceae", "woody-angiosperms"),
    ("Liriodendron tulipifera", "Tulip Poplar", "Magnoliaceae", "woody-angiosperms"),
    ("Liquidambar styraciflua", "Sweetgum", "Altingiaceae", "woody-angiosperms"),
    ("Cornus florida", "Flowering Dogwood", "Cornaceae", "woody-angiosperms"),
    ("Cercis canadensis", "Eastern Redbud", "Fabaceae", "woody-angiosperms"),
    ("Prunus serotina", "Black Cherry", "Rosaceae", "woody-angiosperms"),
    ("Parthenocissus quinquefolia", "Virginia Creeper", "Vitaceae", "woody-angiosperms"),
    ("Toxicodendron radicans", "Poison Ivy", "Anacardiaceae", "woody-angiosperms"),
    ("Pinus taeda", "Loblolly Pine", "Pinaceae", "woody-gymnosperms"),
    ("Pinus virginiana", "Virginia Pine", "Pinaceae", "woody-gymnosperms"),
    ("Trifolium repens", "White Clover", "Fabaceae", "non-alternate-remaining-non-monocots"),
    ("Taraxacum officinale", "Common Dandelion", "Asteraceae", "composites"),
    ("Plantago lanceolata", "Narrowleaf Plantain", "Plantaginaceae", "remaining-non-monocots"),
)


class GoBotanyDevoAdapter:
    """Queries GoBotany Parquet files for taxonomic priors and DeVo character prompts."""

    def __init__(self, parquet_dir: str | Path | None = None) -> None:
        if parquet_dir is None:
            # Default to workspace gobotany parquet path
            base_dir = Path(__file__).resolve().parent.parent / "gobotany" / "extracted_vocabulary" / "parquet"
        else:
            base_dir = Path(parquet_dir)
        
        self.parquet_dir = base_dir
        self.con = duckdb.connect(":memory:")
        self._init_views()

    def _init_views(self) -> None:
        """Register parquet tables as views in DuckDB."""
        p = self.parquet_dir
        if not p.exists():
            return

        tables = {
            "taxon": p / "gobotany_taxon.parquet",
            "character": p / "gobotany_character.parquet",
            "character_value_label": p / "gobotany_character_value_label.parquet",
            "taxon_character_value": p / "gobotany_taxon_character_value.parquet",
            "pile": p / "gobotany_pile.parquet",
            "feature": p / "gobotany_feature.parquet",
            "feature_value": p / "gobotany_feature_value.parquet",
        }

        for view_name, path in tables.items():
            if path.exists():
                self.con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM '{path.as_posix()}'")

    def list_piles(self) -> list[dict[str, Any]]:
        """Return all GoBotany morphological piles."""
        query = "SELECT pile_id, pile_name, pile_slug, pile_friendly_name, description FROM pile ORDER BY pile_id"
        return self.con.execute(query).df().to_dict(orient="records")

    def get_candidate_taxa(
        self,
        pile_slug: str | None = None,
        search_query: str | None = None,
        limit: int = 15,
    ) -> tuple[GoBotanyTaxonCandidate, ...]:
        """Fetch candidate taxa matching pile or search query with discriminative characters."""
        where_clauses = []
        params: list[Any] = []

        if pile_slug:
            where_clauses.append("tcv.pile_slug = ?")
            params.append(pile_slug)
        if search_query:
            where_clauses.append("(LOWER(t.scientific_name) LIKE ? OR LOWER(t.common_name) LIKE ?)")
            q = f"%{search_query.lower()}%"
            params.extend([q, q])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT DISTINCT t.taxon_id, t.scientific_name, t.common_name, t.genus, t.family, tcv.pile_slug
            FROM taxon t
            JOIN taxon_character_value tcv ON t.taxon_id = tcv.taxon_id
            {where_sql}
            ORDER BY t.scientific_name
            LIMIT {limit}
        """

        rows = self.con.execute(sql, params).fetchall()
        candidates: list[GoBotanyTaxonCandidate] = []

        for row in rows:
            taxon_id, scientific_name, common_name, genus, family, p_slug = row
            chars = self.get_taxon_discriminative_characters(taxon_id, limit=8)
            prompts = self._extract_prompts_from_characters(scientific_name, common_name, chars)
            candidates.append(
                GoBotanyTaxonCandidate(
                    taxon_id=taxon_id,
                    scientific_name=scientific_name,
                    common_name=common_name,
                    genus=genus,
                    family=family,
                    pile_slug=p_slug,
                    discriminative_characters=chars,
                    prompt_vocabulary=prompts,
                )
            )

        return tuple(candidates)

    def get_taxon_discriminative_characters(
        self,
        taxon_id: int,
        limit: int = 10,
    ) -> tuple[GoBotanyCharacterValue, ...]:
        """Retrieve key discriminative morphological characters for a specific taxon."""
        sql = """
            SELECT 
                c.character_short_name,
                COALESCE(c.friendly_name, c.character_short_name) AS friendly_name,
                COALESCE(cvl.display_label, cvl.choice) AS value_label,
                cvl.friendly_text,
                c.character_group
            FROM taxon_character_value tcv
            JOIN character c ON tcv.character_short_name = c.character_short_name AND tcv.pile_slug = c.pile_slug
            LEFT JOIN character_value_label cvl 
                ON tcv.character_short_name = cvl.character_short_name 
                AND tcv.pile_slug = cvl.pile_slug 
                AND tcv.value_index = cvl.value_index
            WHERE tcv.taxon_id = ?
            AND cvl.display_label IS NOT NULL
            ORDER BY tcv.ease ASC, c.character_id ASC
            LIMIT ?
        """
        rows = self.con.execute(sql, [taxon_id, limit]).fetchall()
        chars: list[GoBotanyCharacterValue] = []
        for r in rows:
            chars.append(
                GoBotanyCharacterValue(
                    character_name=r[0],
                    friendly_name=r[1],
                    value_label=r[2],
                    friendly_text=r[3],
                    character_group=r[4],
                )
            )
        return tuple(chars)

    def get_henrico_va_yard_candidates(self) -> tuple[GoBotanyTaxonCandidate, ...]:
        """Return standard GoBotany candidate taxa tailored to Henrico, VA yard / woodland edge."""
        candidates: list[GoBotanyTaxonCandidate] = []
        for sci_name, com_name, fam, p_slug in HENRICO_VA_COMMON_YARD_TAXA:
            # Query taxon id from database
            sql = "SELECT taxon_id, genus FROM taxon WHERE scientific_name = ? LIMIT 1"
            row = self.con.execute(sql, [sci_name]).fetchone()
            if row:
                t_id, genus = row
                chars = self.get_taxon_discriminative_characters(t_id, limit=8)
                prompts = self._extract_prompts_from_characters(sci_name, com_name, chars)
                candidates.append(
                    GoBotanyTaxonCandidate(
                        taxon_id=t_id,
                        scientific_name=sci_name,
                        common_name=com_name,
                        genus=genus,
                        family=fam,
                        pile_slug=p_slug,
                        discriminative_characters=chars,
                        prompt_vocabulary=prompts,
                    )
                )
        return tuple(candidates)

    def _extract_prompts_from_characters(
        self,
        scientific_name: str,
        common_name: str | None,
        chars: Sequence[GoBotanyCharacterValue],
    ) -> tuple[str, ...]:
        """Map GoBotany character labels into clean Grounding DINO detection prompts."""
        prompts: set[str] = {"whole plant", "leaf", "bark", "branch", "stem"}
        
        # Add organ specific terms based on character values
        for c in chars:
            lbl = c.value_label.lower()
            fname = c.friendly_name.lower()
            if "lobe" in lbl or "lobe" in fname:
                prompts.add("leaf lobe")
            if "serrate" in lbl or "toothed" in lbl:
                prompts.add("toothed leaf margin")
            if "furrow" in lbl or "ridge" in lbl:
                prompts.add("furrowed bark")
            if "scale" in lbl or "plate" in lbl:
                prompts.add("scaly bark")
            if "petiole" in fname or "petiole" in lbl:
                prompts.add("leaf petiole")
            if "compound" in lbl:
                prompts.add("compound leaflet")
            if "needle" in lbl or "needle" in fname:
                prompts.add("pine needle cluster")
            if "flower" in fname or "petal" in lbl:
                prompts.add("flower")
            if "fruit" in fname or "nut" in lbl or "acorn" in lbl:
                prompts.add("fruit")
                
        return tuple(sorted(prompts))
