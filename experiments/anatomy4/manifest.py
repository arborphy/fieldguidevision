# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Build the Shenandoah observation/media manifest.

Read-only. The app DB is queried through a configurable command (default:
``docker exec arq-compose-postgres psql``) and parsed as CSV so JSON payloads
can never break field splitting.
"""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import (
    CONTRACT,
    photo_id_from_url,
    stable_unit_interval,
)

AREA_INSTANCE_ID = "inst-osm-relation-5548542"

# Gymnosperm genera at Shenandoah. Used ONLY to guarantee the audit sample has
# needle candidates; boxes are never derived from this list.
NEEDLE_CANDIDATE_GENERA = {
    "abies",
    "chamaecyparis",
    "cryptomeria",
    "juniperus",
    "larix",
    "picea",
    "pinus",
    "taxus",
    "thuja",
    "tsuga",
}

SEASONS = ("winter", "spring", "summer", "fall")

CONIFER_FAMILIES = {"pinaceae", "cupressaceae"}

TARGETED_QUERY = """
SELECT
  o.arborphy_id, o.inat_id, o.taxon_name, o.taxon_rank, o.common_name, o.observed_on,
  o.quality_grade, o.iconic_taxon_name, o.photo_url, o.source_payload,
  fam.canonical_name AS taxon_family, gen.canonical_name AS taxon_genus
FROM observations o
LEFT JOIN gbif_backbone fam ON fam.gbif_id = o.backbone_family_fk
LEFT JOIN gbif_backbone gen ON gen.gbif_id = o.backbone_genus_fk
WHERE o.area_instance_id = %s
  AND o.photo_url IS NOT NULL AND o.photo_url <> ''
  AND o.deleted_at IS NULL
ORDER BY o.arborphy_id
"""

DEFAULT_QUERY = """
SELECT
  arborphy_id, inat_id, taxon_name, taxon_rank, common_name, observed_on,
  quality_grade, iconic_taxon_name, photo_url, source_payload
FROM observations
WHERE area_instance_id = %s
  AND photo_url IS NOT NULL AND photo_url <> ''
  AND deleted_at IS NULL
ORDER BY arborphy_id
"""


@dataclass
class ImageRecord:
    photo_id: str
    url: str | None
    license: str | None = None
    attribution: str | None = None
    width: int | None = None
    height: int | None = None
    position: int | None = None
    is_cover: bool = False
    selected_for_audit: bool = False
    vault_relpath: str | None = None
    download: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "url": self.url,
            "license": self.license,
            "attribution": self.attribution,
            "width": self.width,
            "height": self.height,
            "position": self.position,
            "is_cover": self.is_cover,
            "selected_for_audit": self.selected_for_audit,
            "vault_relpath": self.vault_relpath,
            "download": self.download,
        }


@dataclass
class ObservationRecord:
    observation_id: str
    inat_id: str | None
    taxon_name: str | None
    taxon_rank: str | None
    common_name: str | None
    observed_on: str | None
    quality_grade: str | None
    iconic_taxon_name: str | None
    images: list[ImageRecord] = field(default_factory=list)
    stratum: str | None = None
    taxon_family: str | None = None
    taxon_genus: str | None = None
    audit_bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "inat_id": self.inat_id,
            "taxon_name": self.taxon_name,
            "taxon_rank": self.taxon_rank,
            "common_name": self.common_name,
            "observed_on": self.observed_on,
            "quality_grade": self.quality_grade,
            "iconic_taxon_name": self.iconic_taxon_name,
            "stratum": self.stratum,
            "taxon_family": self.taxon_family,
            "taxon_genus": self.taxon_genus,
            "audit_bucket": self.audit_bucket,
            "images": [image.to_dict() for image in self.images],
        }


def _run_psql(db_command: Sequence[str], query: str) -> list[dict[str, str]]:
    csv.field_size_limit(64 * 1024 * 1024)
    command = [*db_command, "-c", f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER true)"]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    reader = csv.DictReader(io.StringIO(completed.stdout))
    return list(reader)


def _payload_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _upgrade_inat_url(url: str | None) -> str | None:
    """iNat payloads carry /square.; audit needs the /large. rendition."""

    if not url:
        return None
    for size in ("square", "small", "medium"):
        marker = f"/{size}."
        if marker in url:
            return url.replace(marker, "/large.")
    return url


def _images_from_payload(payload: dict[str, Any], cover_url: str | None) -> list[ImageRecord]:
    photos = payload.get("photos")
    if not isinstance(photos, list):
        photos = []
    observation_photos = payload.get("observation_photos")
    positions: dict[str, int] = {}
    if isinstance(observation_photos, list):
        for entry in observation_photos:
            if not isinstance(entry, dict):
                continue
            photo = entry.get("photo") if isinstance(entry.get("photo"), dict) else {}
            photo_id = str(photo.get("id") or entry.get("photo_id") or "")
            if photo_id:
                positions[photo_id] = int(entry.get("position") or 0)

    images: list[ImageRecord] = []
    seen: set[str] = set()
    for raw in photos:
        if not isinstance(raw, dict):
            continue
        photo_id = raw.get("id")
        if photo_id is None:
            continue
        photo_id = str(photo_id)
        if photo_id in seen:
            continue
        seen.add(photo_id)
        dims = raw.get("original_dimensions") or {}
        images.append(
            ImageRecord(
                photo_id=photo_id,
                url=_upgrade_inat_url(raw.get("url")),
                license=raw.get("license_code"),
                attribution=raw.get("attribution"),
                width=dims.get("width"),
                height=dims.get("height"),
                position=positions.get(photo_id),
            )
        )

    cover_id = photo_id_from_url(cover_url)
    if cover_id and cover_id not in seen:
        images.insert(
            0,
            ImageRecord(
                photo_id=cover_id,
                url=_upgrade_inat_url(cover_url),
                position=0,
            ),
        )
        seen.add(cover_id)

    for image in images:
        image.is_cover = image.photo_id == cover_id
        if image.url and not image.url.startswith("http"):
            image.url = None
        image.vault_relpath = f"photos/{image.photo_id}"
    if not any(image.is_cover for image in images) and images:
        images[0].is_cover = True
    return images


def parse_observations(rows: Iterable[dict[str, str]]) -> list[ObservationRecord]:
    records: list[ObservationRecord] = []
    for row in rows:
        payload = _payload_dict(row.get("source_payload"))
        images = _images_from_payload(payload, row.get("photo_url"))
        if not images:
            continue
        records.append(
            ObservationRecord(
                observation_id=str(row.get("arborphy_id") or ""),
                inat_id=str(row.get("inat_id") or "") or None,
                taxon_name=row.get("taxon_name") or None,
                taxon_rank=row.get("taxon_rank") or None,
                common_name=row.get("common_name") or None,
                observed_on=row.get("observed_on") or None,
                quality_grade=row.get("quality_grade") or None,
                iconic_taxon_name=row.get("iconic_taxon_name") or None,
                taxon_family=row.get("taxon_family") or None,
                taxon_genus=row.get("taxon_genus") or None,
                images=images,
            )
        )
    return records


def fetch_observations(
    area_instance_id: str = AREA_INSTANCE_ID,
    db_command: Sequence[str] | None = None,
) -> list[ObservationRecord]:
    command = list(
        db_command
        or ["docker", "exec", "arq-compose-postgres", "psql", "-U", "arq", "-d", "arq"]
    )
    # psql does not interpolate COPY parameters; the area id is a fixed,
    # internally generated identifier. Validate its shape before interpolation.
    if not re.fullmatch(r"inst-[a-z0-9-]+", area_instance_id):
        raise ValueError(f"unexpected area instance id: {area_instance_id!r}")
    query = DEFAULT_QUERY.replace("%s", f"'{area_instance_id}'")
    rows = _run_psql(command, query)
    return parse_observations(rows)


def month_of(date_value: str | None) -> int | None:
    if not date_value:
        return None
    match = re.match(r"\d{4}-(\d{2})", date_value)
    return int(match.group(1)) if match else None


def season_of(date_value: str | None) -> str:
    month = month_of(date_value)
    if month is None:
        return "unknown"
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def is_plant(record: ObservationRecord) -> bool:
    return (record.iconic_taxon_name or "").lower() == "plantae"


def is_needle_candidate(record: ObservationRecord) -> bool:
    name = (record.taxon_name or "").lower()
    genus = name.split()[0] if name else ""
    return genus in NEEDLE_CANDIDATE_GENERA


def _pick_diverse(
    records: Sequence[ObservationRecord], count: int, seed: str
) -> list[ObservationRecord]:
    """Round-robin over distinct taxon names, deterministically ordered."""

    if count <= 0:
        return []
    groups: dict[str, list[ObservationRecord]] = defaultdict(list)
    for record in records:
        groups[record.taxon_name or record.observation_id].append(record)
    ordered = sorted(
        groups.values(),
        key=lambda group: stable_unit_interval(group[0].observation_id, seed),
    )
    for group in ordered:
        group.sort(key=lambda r: stable_unit_interval(r.observation_id, f"{seed}:within"))
    picked: list[ObservationRecord] = []
    while len(picked) < count and ordered:
        next_round: list[list[ObservationRecord]] = []
        for group in ordered:
            if not group:
                continue
            picked.append(group.pop(0))
            if len(picked) >= count:
                break
            if group:
                next_round.append(group)
        ordered = next_round
    return picked


def select_sample(
    records: Sequence[ObservationRecord],
    sample_size: int = 120,
    seed: str = CONTRACT,
    plant_fraction: float = 0.72,
    needle_minimum: int = 10,
) -> list[ObservationRecord]:
    """Deterministic stratified sample.

    Strata (plants): needle candidates, then season (spring/summer/fall/winter).
    Non-plants are stratified by iconic taxon. Every stratum falls back to the
    unused plant/other pool if it is short, so the requested size is met when
    enough records exist.
    """

    usable = [r for r in records if r.images]
    plants = [r for r in usable if is_plant(r)]
    others = [r for r in usable if not is_plant(r)]

    plant_target = min(len(plants), round(sample_size * plant_fraction))
    other_target = max(0, sample_size - plant_target)

    needle_pool = [r for r in plants if is_needle_candidate(r)]
    season_pools: dict[str, list[ObservationRecord]] = {s: [] for s in SEASONS}
    season_pools["unknown"] = []
    for record in plants:
        if is_needle_candidate(record):
            continue
        season = season_of(record.observed_on)
        season_pools[season if season in season_pools else "unknown"].append(record)

    selected: list[ObservationRecord] = []
    taken: set[str] = set()

    def take(items: Sequence[ObservationRecord], count: int, stratum: str) -> None:
        fresh = [r for r in items if r.observation_id not in taken]
        picked = _pick_diverse(fresh, count, f"{seed}:{stratum}")
        for record in picked:
            record.stratum = stratum
            selected.append(record)
            taken.add(record.observation_id)

    needle_count = min(needle_minimum, len(needle_pool), plant_target)
    take(needle_pool, needle_count, "plant:needle_candidate")

    remaining = plant_target - needle_count
    # Season weights: spring/summer richest for flowers/inflorescences.
    weights = {"spring": 0.30, "summer": 0.30, "fall": 0.22, "winter": 0.18}
    quotas = {
        season: int(round(remaining * weight)) for season, weight in weights.items()
    }
    quotas["summer"] += remaining - sum(quotas.values())
    for season in SEASONS:
        take(season_pools[season], quotas[season], f"plant:{season}")

    if len([r for r in selected if is_plant(r)]) < plant_target:
        missing = plant_target - len([r for r in selected if is_plant(r)])
        take(plants, missing, "plant:fallback")

    other_strata: dict[str, list[ObservationRecord]] = defaultdict(list)
    for record in others:
        other_strata[(record.iconic_taxon_name or "unknown").lower()].append(record)
    if other_target > 0:
        per_group = max(1, other_target // max(1, len(other_strata)))
        for group in sorted(other_strata):
            take(other_strata[group], per_group, f"other:{group}")
        missing = other_target - len([r for r in selected if not is_plant(r)])
        if missing > 0:
            take(others, missing, "other:fallback")

    # Trim to requested size while preserving stratum order (which is also the
    # audit review order: needle candidates first).
    selected = selected[:sample_size]
    for record in selected:
        cover = next((image for image in record.images if image.is_cover), record.images[0])
        cover.selected_for_audit = True
    return selected


def is_conifer(record: ObservationRecord) -> bool:
    """Taxonomy-only candidate flag. Never used to derive a box or a label."""

    family = (record.taxon_family or "").lower()
    if family in CONIFER_FAMILIES:
        return True
    name = (record.taxon_name or "").lower()
    genus = (record.taxon_genus or (name.split()[0] if name else "")).lower()
    return genus in NEEDLE_CANDIDATE_GENERA


def is_flowering_season(record: ObservationRecord) -> bool:
    return month_of(record.observed_on) in (3, 4, 5, 6, 7, 8)


def _photos_for(record: ObservationRecord, limit: int) -> list[ImageRecord]:
    if limit <= 0:
        return []
    ordered = sorted(
        record.images,
        key=lambda image: (
            not image.is_cover,
            image.position if image.position is not None else 99,
        ),
    )
    return ordered[:limit]


def select_targeted_sample(
    records: Sequence[ObservationRecord],
    *,
    seed: str = CONTRACT,
    needle_observations: int = 60,
    photos_per_needle_observation: int = 2,
    leaf_observations: int = 110,
    flower_observations: int = 126,
    background_observations: int = 48,
) -> list[ObservationRecord]:
    """Class-specific audit sample for the targeted YOLOE pass.

    Buckets are disjoint and assigned by precedence: needle (conifer taxa) >
    flower/inflorescence (spring-summer Plantae) > leaf (broadleaf Plantae) >
    background (non-plant plus winter/unknown Plantae). Taxonomy is used ONLY
    to pick candidates; proposals never inherit the taxon.
    """

    usable = [r for r in records if r.images]
    conifer = [r for r in usable if is_plant(r) and is_conifer(r)]
    broadleaf = [r for r in usable if is_plant(r) and not is_conifer(r)]
    flowering = [r for r in broadleaf if is_flowering_season(r)]
    non_plant = [r for r in usable if not is_plant(r)]
    winter_plant = [
        r
        for r in broadleaf
        if is_flowering_season(r) is False and month_of(r.observed_on) is not None
    ]

    selected: list[ObservationRecord] = []
    taken: set[str] = set()

    def take(
        items: Sequence[ObservationRecord],
        count: int,
        bucket: str,
        photos: int = 1,
    ) -> None:
        fresh = [r for r in items if r.observation_id not in taken]
        picked = _pick_diverse(fresh, count, f"{seed}:{bucket}")
        for record in picked:
            record.stratum = bucket
            record.audit_bucket = bucket
            for image in _photos_for(record, photos):
                image.selected_for_audit = True
            selected.append(record)
            taken.add(record.observation_id)

    take(conifer, needle_observations, "needle", photos_per_needle_observation)
    take(flowering, flower_observations, "flower")
    take(broadleaf, leaf_observations, "leaf")
    # Background: half non-plant, half winter/unknown plant, so reviewed empty
    # images cover both "no organ in frame" and "dormant plant" cases.
    take(non_plant, background_observations // 2, "background")
    take(winter_plant, background_observations - background_observations // 2, "background")
    if len(selected) < sum(
        [
            needle_observations,
            leaf_observations,
            flower_observations,
            background_observations,
        ]
    ):
        missing = (
            needle_observations
            + leaf_observations
            + flower_observations
            + background_observations
            - len(selected)
        )
        take(usable, missing, "background")
    return selected


def build_targeted_manifest(
    *,
    area_instance_id: str = AREA_INSTANCE_ID,
    seed: str = "anatomy4-targeted-v1",
    needle_observations: int = 60,
    photos_per_needle_observation: int = 2,
    leaf_observations: int = 110,
    flower_observations: int = 126,
    background_observations: int = 48,
    db_command: Sequence[str] | None = None,
    records: Sequence[ObservationRecord] | None = None,
) -> dict[str, Any]:
    if records is None:
        command = list(
            db_command
            or ["docker", "exec", "arq-compose-postgres", "psql", "-U", "arq", "-d", "arq"]
        )
        if not re.fullmatch(r"inst-[a-z0-9-]+", area_instance_id):
            raise ValueError(f"unexpected area instance id: {area_instance_id!r}")
        rows = _run_psql(command, TARGETED_QUERY.replace("%s", f"'{area_instance_id}'"))
        all_records = parse_observations(rows)
    else:
        all_records = list(records)

    selected = select_targeted_sample(
        all_records,
        seed=seed,
        needle_observations=needle_observations,
        photos_per_needle_observation=photos_per_needle_observation,
        leaf_observations=leaf_observations,
        flower_observations=flower_observations,
        background_observations=background_observations,
    )
    strata = Counter(record.stratum for record in selected)
    observations = [record.to_dict() for record in selected]
    images = [
        {**image, "observation_id": record["observation_id"], "audit_bucket": record["audit_bucket"]}
        for record in observations
        for image in record["images"]
    ]
    audit_images = [image for image in images if image["selected_for_audit"]]
    bucket_images = Counter(
        image["audit_bucket"] for image in audit_images
    )
    return {
        "schema": "arq.anatomy4.manifest/v1",
        "contract": CONTRACT,
        "mode": "targeted",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "area_instance_id": area_instance_id,
        "seed": seed,
        "sample_size_requested": len(audit_images),
        "source_counts": {
            "observations_with_photos": len(all_records),
            "plants_with_photos": len([r for r in all_records if is_plant(r)]),
            "conifer_observations_with_photos": len(
                [r for r in all_records if is_plant(r) and is_conifer(r)]
            ),
            "selected_observations": len(observations),
            "selected_images_for_audit": len(audit_images),
            "audit_images_by_bucket": dict(sorted(bucket_images.items())),
        },
        "strata": dict(sorted(strata.items())),
        "observations": observations,
        "images": images,
    }


def build_manifest(
    *,
    area_instance_id: str = AREA_INSTANCE_ID,
    sample_size: int = 120,
    seed: str = CONTRACT,
    plant_fraction: float = 0.72,
    needle_minimum: int = 10,
    db_command: Sequence[str] | None = None,
    records: Sequence[ObservationRecord] | None = None,
) -> dict[str, Any]:
    all_records = (
        list(records)
        if records is not None
        else fetch_observations(area_instance_id, db_command)
    )
    selected = select_sample(
        all_records,
        sample_size=sample_size,
        seed=seed,
        plant_fraction=plant_fraction,
        needle_minimum=needle_minimum,
    )
    strata = Counter(record.stratum for record in selected)
    observations = [record.to_dict() for record in selected]
    images = [
        {**image, "observation_id": record["observation_id"]}
        for record in observations
        for image in record["images"]
    ]
    audit_images = [image for image in images if image["selected_for_audit"]]
    return {
        "schema": "arq.anatomy4.manifest/v1",
        "contract": CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "area_instance_id": area_instance_id,
        "seed": seed,
        "sample_size_requested": sample_size,
        "source_counts": {
            "observations_with_photos": len(all_records),
            "plants_with_photos": len([r for r in all_records if is_plant(r)]),
            "selected_observations": len(observations),
            "selected_images_for_audit": len(audit_images),
        },
        "strata": dict(sorted(strata.items())),
        "observations": observations,
        "images": images,
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=1))


def audit_images(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [image for image in manifest.get("images", []) if image.get("selected_for_audit")]
