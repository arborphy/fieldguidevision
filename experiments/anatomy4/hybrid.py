# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Hybrid proposal pass over the 2,400-image training-candidate pool.

Per-class detector choice comes from the reviewed alternate-detector benchmark
(``REPORT-alt-detectors.md``): YOLOE stays on ``leaf``; OWLv2 takes ``needle``,
``flower`` and ``inflorescence``.

Two deliberate differences from the earlier per-stratum candidate pass:

* **Needle prompts run on every plant stratum**, not only conifer-tagged
  images. The conifer stratum is small (84 images) and the taxon tag is a
  sampling aid, not a fact about the pixels.
* **Cross-prompt duplicates are merged without losing provenance.** Every
  surviving proposal carries ``source_proposals`` with each raw detection's
  id, prompt, score and bbox, so prompt-level precision stays auditable.

The background stratum stays detector-blind: it is the pool's negative sample
and "the detector found nothing" must never be manufactured into a label.
Nothing here assigns a review verdict.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import CLASS_NAMES

PLANT_STRATA: tuple[str, ...] = ("broadleaf", "conifer", "flower", "inflorescence")
BACKGROUND_STRATUM = "background"

#: Reviewed operating point. ``strata`` is either an explicit list or
#: ``"all_plant"`` for the cross-stratum needle sweep.
HYBRID_PLAN: dict[str, dict[str, Any]] = {
    "leaf": {
        "backend": "yoloe",
        "model": "yoloe-26s-seg.pt",
        "prompts": ["leaf"],
        "strata": ["broadleaf"],
        "score_floor": 0.05,
        "imgsz": 1024,
        "conf_floor": 0.02,
        "why": "YOLOE 77.3% reviewed precision vs OWLv2 45.8-50.0% and 9.2 boxes/image",
    },
    "needle": {
        "backend": "owlv2",
        "model": "google/owlv2-base-patch16-ensemble",
        "prompts": ["conifer foliage", "spruce needles"],
        "strata": "all_plant",
        "score_floor": 0.10,
        "imgsz": None,
        "conf_floor": None,
        "why": "OWLv2 33.3% reviewed precision, 33.3% correct+tight, median box 0.055; 'pine needles' dropped (0/8 reviewed)",
    },
    "flower": {
        "backend": "owlv2",
        "model": "google/owlv2-base-patch16-ensemble",
        "prompts": ["single flower"],
        "strata": ["flower"],
        "score_floor": 0.15,
        "imgsz": None,
        "conf_floor": None,
        "why": "OWLv2 78.1% reviewed precision vs YOLOE 31.8%",
    },
    "inflorescence": {
        "backend": "owlv2",
        "model": "google/owlv2-base-patch16-ensemble",
        "prompts": ["cluster of flowers", "umbel of flowers"],
        "strata": ["inflorescence"],
        "score_floor": 0.15,
        "imgsz": None,
        "conf_floor": None,
        "why": "OWLv2 53.1% reviewed precision, median box 0.028 vs YOLOE 0.188; 'flower head' dropped",
    },
}

#: Guards shared by every class in this pass.
MAX_BOX_AREA = 0.60
MIN_BOX_AREA = 0.0004
NMS_IOU = 0.5
MAX_PER_CLASS_PER_IMAGE = 10

#: Conifer taxa are a *provenance* question here, not a filter. This mirrors
#: ``manifest.is_conifer`` without importing the DB-facing module.
CONIFER_FAMILIES = {
    "cupressaceae",
    "pinaceae",
    "taxaceae",
    "taxodiaceae",
    "sciadopityaceae",
    "podocarpaceae",
    "araucariaceae",
}
CONIFER_GENERA = {
    "abies",
    "chamaecyparis",
    "juniperus",
    "larix",
    "picea",
    "pinus",
    "taxodium",
    "taxus",
    "thuja",
    "tsuga",
    "cryptomeria",
    "metasequoia",
    "sciadopitys",
    "cupressus",
    "sequoia",
    "sequoiadendron",
    "pseudotsuga",
    "cedrus",
    "calocedrus",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Pool loading and scoping
# --------------------------------------------------------------------------


def load_candidate_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_conifer_row(row: Mapping[str, Any]) -> bool:
    family = str(row.get("taxon_family") or "").strip().lower()
    genus = str(row.get("taxon_genus") or "").strip().lower()
    return family in CONIFER_FAMILIES or genus in CONIFER_GENERA


def strata_for(class_name: str, plan: Mapping[str, Any] | None = None) -> list[str]:
    entry = (plan or HYBRID_PLAN)[class_name]
    strata = entry["strata"]
    if strata == "all_plant":
        return list(PLANT_STRATA)
    return list(strata)


def scope_rows(
    rows: Sequence[dict[str, Any]],
    class_name: str,
    *,
    plan: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Candidate rows a class's prompts are run on.

    The background stratum is never in a scope: it is the detector-blind
    negative sample.
    """

    allowed = set(strata_for(class_name, plan))
    if BACKGROUND_STRATUM in allowed:
        raise ValueError("background stratum must stay detector-blind")
    return [row for row in rows if row.get("stratum") in allowed]


def detector_manifest(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Minimal manifest shape both detector backends accept."""

    return {
        "schema": "arq.anatomy4.manifest/v1",
        "contract": "anatomy4-v1",
        "images": [
            {
                "photo_id": row["image_id"],
                "image_id": row["image_id"],
                "observation_id": row.get("observation_id"),
                "audit_bucket": row.get("stratum"),
                "selected_for_audit": True,
                "download": {"path": row["local_image_path"]},
            }
            for row in rows
        ],
        "observations": [],
    }


def _single_class_config(class_name: str, entry: Mapping[str, Any]):
    from .contracts import AnatomyConfig, ClassSpec

    spec = ClassSpec(
        id=CLASS_NAMES.index(class_name),
        name=class_name,
        prompts=tuple(entry["prompts"]),
        box_threshold=float(entry["score_floor"]),
        text_threshold=0.0,
    )
    return AnatomyConfig(
        contract="anatomy4-v1",
        classes=(spec,),
        detector=str(entry["model"]),
        segmenter="facebook/sam2.1-hiera-tiny",
        max_image_side=1536,
        nms_iou=NMS_IOU,
        polygon_epsilon=0.003,
        multimask_output=False,
        min_polygon_points=3,
        empty_cache_each_image=True,
        max_box_area=MAX_BOX_AREA,
        min_box_area=MIN_BOX_AREA,
    )


# --------------------------------------------------------------------------
# Provenance-preserving dedupe
# --------------------------------------------------------------------------


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _source_ref(proposal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "proposal_id": proposal["proposal_id"],
        "prompt": proposal.get("prompt"),
        "box_score": round(float(proposal["box_score"]), 6),
        "detector_bbox": list(proposal["detector_bbox"]),
        "detector_rank": proposal.get("detector_rank"),
    }


def merge_prompt_duplicates(
    proposals: Iterable[Mapping[str, Any]],
    *,
    iou_threshold: float = NMS_IOU,
    max_per_class: int | None = MAX_PER_CLASS_PER_IMAGE,
) -> list[dict[str, Any]]:
    """Greedy per-image, per-class NMS that keeps every source detection.

    The survivor is the highest-scoring box. Suppressed detections do not
    disappear: they are recorded in ``source_proposals`` (id, prompt, score,
    bbox), so a proposal still says *which prompts* found it. Dedupe is never
    performed across classes — ``flower`` and ``inflorescence`` may legitimately
    describe overlapping structure, and merging them would hide that ambiguity.
    """

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        if proposal.get("status") != "proposed":
            continue
        groups[(str(proposal["image_id"]), str(proposal["class_name"]))].append(proposal)

    merged: list[dict[str, Any]] = []
    for _, group in sorted(groups.items()):
        kept: list[dict[str, Any]] = []
        for proposal in sorted(group, key=lambda p: float(p["box_score"]), reverse=True):
            duplicate = next(
                (
                    other
                    for other in kept
                    if _iou(other["detector_bbox"], proposal["detector_bbox"]) >= iou_threshold
                ),
                None,
            )
            if duplicate is not None:
                duplicate["source_proposals"].append(_source_ref(proposal))
                duplicate["duplicate_count"] += 1
                continue
            if max_per_class is not None and len(kept) >= max_per_class:
                continue
            survivor = dict(proposal)
            survivor["source_proposals"] = [_source_ref(proposal)]
            survivor["duplicate_count"] = 0
            kept.append(survivor)
        for survivor in kept:
            prompts = {source["prompt"] for source in survivor["source_proposals"]}
            survivor["prompt_count"] = len(prompts)
            survivor["prompts"] = sorted(prompts)
            survivor["review_status"] = "unreviewed"
            survivor["proposal_state"] = "proposal_only"
        merged.extend(kept)
    merged.sort(key=lambda p: (CLASS_NAMES.index(p["class_name"]), -float(p["box_score"])))
    return merged


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _score_block(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "p50": round(_percentile(values, 0.5), 6),
        "p90": round(_percentile(values, 0.9), 6),
    }


def build_stats(
    *,
    rows: Sequence[dict[str, Any]],
    merged: Sequence[dict[str, Any]],
    raw_by_class: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any] | None = None,
    blind_check: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = plan or HYBRID_PLAN
    row_by_image = {row["image_id"]: row for row in rows}
    scope_by_class = {
        class_name: {row["image_id"] for row in scope_rows(rows, class_name, plan=plan)}
        for class_name in CLASS_NAMES
    }

    per_class: dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        proposals = [p for p in merged if p["class_name"] == class_name]
        scope = scope_by_class[class_name]
        raw = raw_by_class.get(class_name, {})
        run = runs.get(class_name, {})
        summary = raw.get("summary", {}) if raw else {}
        raw_proposals = raw.get("proposals", []) if raw else []
        rejected = summary.get("rejected_by_reason") or dict(
            Counter(
                proposal.get("status")
                for proposal in raw_proposals
                if proposal.get("status") and proposal.get("status") != "proposed"
            )
        )
        by_stratum = Counter(p["stratum"] for p in proposals)
        by_prompt = Counter(
            source["prompt"]
            for proposal in proposals
            for source in proposal["source_proposals"]
        )
        per_class[class_name] = {
            "backend": plan[class_name]["backend"],
            "model": plan[class_name]["model"],
            "prompts": list(plan[class_name]["prompts"]),
            "score_floor": plan[class_name]["score_floor"],
            "strata": strata_for(class_name, plan),
            "scope_images": len(scope),
            "images_processed": run.get("images_processed"),
            "images_with_proposals": len({p["image_id"] for p in proposals}),
            "image_coverage": round(
                len({p["image_id"] for p in proposals}) / max(1, len(scope)), 4
            ),
            "proposals": len(proposals),
            "raw_proposals": run.get("raw_detections_above_floor")
            if run.get("raw_detections_above_floor") is not None
            else len(raw_proposals),
            "duplicates_merged": sum(int(p.get("duplicate_count") or 0) for p in proposals),
            "proposals_spanning_multiple_prompts": sum(
                1 for p in proposals if len(p.get("prompts") or []) > 1
            ),
            "proposals_by_stratum": dict(sorted(by_stratum.items())),
            "proposals_by_prompt": dict(sorted(by_prompt.items())),
            "score_summary": _score_block([float(p["box_score"]) for p in proposals]),
            "box_area_summary": _score_block(
                [float(p["detector_box_area"]) for p in proposals]
            ),
            "rejected_by_reason": rejected,
            "runtime_seconds": run.get("runtime_seconds"),
            "peak_gpu_mb": run.get("peak_gpu_mb"),
        }

    scored_images = set().union(*scope_by_class.values())
    images_with_any = {p["image_id"] for p in merged}
    pooled = [row for row in rows if row.get("stratum") != BACKGROUND_STRATUM]
    conifer_taxon = [row for row in pooled if is_conifer_row(row)]
    outside = [row for row in conifer_taxon if row.get("stratum") != "conifer"]
    outside_ids = {row["image_id"] for row in outside}
    inside_ids = {row["image_id"] for row in conifer_taxon if row.get("stratum") == "conifer"}
    needle = [p for p in merged if p["class_name"] == "needle"]
    needle_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in needle:
        needle_by_image[proposal["image_id"]].append(proposal)

    conifer_outside = {
        "taxon_is_conifer_images": len(conifer_taxon),
        "in_conifer_stratum": len(inside_ids),
        "outside_conifer_stratum": len(outside),
        "outside_by_stratum": dict(
            sorted(Counter(row["stratum"] for row in outside).items())
        ),
        "needle_proposals_from_outside": sum(
            len(needle_by_image.get(image_id, [])) for image_id in outside_ids
        ),
        "images_outside_with_needle": len(outside_ids & set(needle_by_image)),
        "examples": [
            {
                "image_id": row["image_id"],
                "stratum": row["stratum"],
                "taxon_name": row.get("taxon_name"),
                "taxon_family": row.get("taxon_family"),
                "needle_proposals": len(needle_by_image.get(row["image_id"], [])),
                "max_score": max(
                    (float(p["box_score"]) for p in needle_by_image.get(row["image_id"], [])),
                    default=None,
                ),
            }
            for row in sorted(
                outside,
                key=lambda r: -len(needle_by_image.get(r["image_id"], [])),
            )
            if needle_by_image.get(row["image_id"])
        ],
    }

    total_runtime = sum(
        float(entry.get("runtime_seconds") or 0.0) for entry in runs.values()
    )
    return {
        "schema": "arq.anatomy4.hybrid-stats/v1",
        "created_at": utc_now(),
        "purpose": "training-candidate proposals; no ground truth, no review verdicts",
        "images_in_pool": len(rows),
        "images_scored": len(scored_images),
        "images_with_any_proposal": len(images_with_any),
        "image_coverage": round(len(images_with_any) / max(1, len(scored_images)), 4),
        "background_stratum": {
            "images": len(rows) - len(pooled),
            "detector_blind": True,
            "note": "never sent to any detector; empty proposals are by construction, not abstention",
        },
        "proposals_total": len(merged),
        "proposals_by_class": dict(
            Counter(p["class_name"] for p in merged)
        ),
        "per_class": per_class,
        "conifer_provenance": conifer_outside,
        "runtime_seconds_total": round(total_runtime, 2),
        "peak_gpu_mb": max(
            (float(entry.get("peak_gpu_mb") or 0.0) for entry in runs.values()),
            default=0.0,
        ),
        "runs": {name: dict(entry) for name, entry in runs.items()},
        "blind_test_check": dict(blind_check or {}),
        "guards": {
            "max_box_area": MAX_BOX_AREA,
            "min_box_area": MIN_BOX_AREA,
            "nms_iou": NMS_IOU,
            # The benchmark's per-image cap of 10 was a review-volume bound; it
            # is not applied to training candidates (see reindex_notes).
            "per_image_cap": None,
            "benchmark_review_cap": MAX_PER_CLASS_PER_IMAGE,
        },
        "plan": {
            name: {
                "backend": entry["backend"],
                "model": entry["model"],
                "prompts": list(entry["prompts"]),
                "strata": strata_for(name, plan),
                "score_floor": entry["score_floor"],
            }
            for name, entry in plan.items()
        },
    }


# --------------------------------------------------------------------------
# Blind-test exclusion check
# --------------------------------------------------------------------------


def _photo_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    marker = "/photos/"
    if marker in url:
        tail = url.split(marker, 1)[1]
        digits = "".join(character for character in tail.split("/")[0] if character.isdigit())
        return digits or None
    return None


def verify_blind_exclusion(
    rows: Sequence[dict[str, Any]], blind_manifest_paths: Sequence[str | Path]
) -> dict[str, Any]:
    """Re-verify zero overlap against the held-out blind manifests."""

    blind_observations: set[str] = set()
    blind_photos: set[str] = set()
    blind_urls: set[str] = set()
    blind_sha: set[str] = set()
    per_manifest: dict[str, int] = {}
    for path in blind_manifest_paths:
        count = 0
        with Path(path).open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                count += 1
                if record.get("observation_id"):
                    blind_observations.add(str(record["observation_id"]))
                if record.get("photo_id"):
                    blind_photos.add(str(record["photo_id"]))
                if record.get("photo_url"):
                    blind_urls.add(str(record["photo_url"]))
                    derived = _photo_id_from_url(str(record["photo_url"]))
                    if derived:
                        blind_photos.add(derived)
                if record.get("sha256"):
                    blind_sha.add(str(record["sha256"]))
        per_manifest[str(path)] = count

    candidate_observations = {str(row.get("observation_id")) for row in rows}
    candidate_photos = {str(row.get("photo_id")) for row in rows}
    candidate_urls = {str(row.get("photo_url")) for row in rows}
    candidate_sha = {str(row.get("sha256")) for row in rows}

    overlap = {
        "observation_ids": sorted(candidate_observations & blind_observations),
        "photo_ids": sorted(candidate_photos & blind_photos),
        "photo_urls": sorted(candidate_urls & blind_urls),
        "sha256": sorted(candidate_sha & blind_sha),
    }
    return {
        "blind_images": sum(per_manifest.values()),
        "blind_observations": len(blind_observations),
        "manifests": per_manifest,
        "candidate_images": len(rows),
        "overlap_counts": {name: len(values) for name, values in overlap.items()},
        "overlap": overlap,
        "passed": not any(overlap.values()),
    }


# --------------------------------------------------------------------------
# Small contact-sheet audit
# --------------------------------------------------------------------------


def render_audit_sheets(
    merged: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    out_dir: str | Path,
    *,
    per_class: int = 12,
    columns: int = 3,
) -> dict[str, list[str]]:
    """One small sheet per class plus one cross-stratum needle sheet.

    Score-stratified sample (highest, middle, lowest) rather than top-N only,
    so the sheet shows the operating range instead of the easy cases.
    """

    from PIL import Image, ImageDraw

    from .altsheets import CLASS_COLORS, _tile, sample_evenly

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    path_by_image = {row["image_id"]: row["local_image_path"] for row in rows}
    tile_w, tile_h, caption_h = 520, 390, 34
    rows_per_sheet = 3
    per_sheet = columns * rows_per_sheet

    def draw_sheet(title: str, proposals: Sequence[dict[str, Any]], name: str) -> list[str]:
        sample = sample_evenly(list(proposals), per_class)
        pages: list[str] = []
        for page, start in enumerate(range(0, len(sample), per_sheet)):
            chunk = sample[start : start + per_sheet]
            sheet = Image.new(
                "RGB", (columns * tile_w, rows_per_sheet * (tile_h + caption_h)), (10, 10, 10)
            )
            draw = ImageDraw.Draw(sheet)
            draw.text((6, 4), title, fill=(255, 255, 255))
            for index, proposal in enumerate(chunk):
                row_index, col_index = divmod(index, columns)
                x = col_index * tile_w
                y = row_index * (tile_h + caption_h)
                image_path = path_by_image.get(proposal["image_id"])
                if image_path and Path(image_path).is_file():
                    sheet.paste(_tile(image_path, proposal, tile_w, tile_h), (x, y))
                draw.rectangle(
                    [x, y, x + tile_w, y + tile_h],
                    outline=CLASS_COLORS.get(proposal["class_name"], (255, 255, 255)),
                    width=2,
                )
                caption = (
                    f"{proposal['box_score']:.3f} {'+'.join(proposal['prompts'])} "
                    f"area {proposal['detector_box_area']:.2f} | {proposal['stratum']} | "
                    f"{proposal['image_id']}"
                )
                draw.text((x + 4, y + tile_h + 8), caption, fill=(235, 235, 235))
            target = output / f"{name}-{page:02d}.jpg"
            sheet.save(target, quality=90)
            pages.append(str(target))
        return pages

    sheets: dict[str, list[str]] = {}
    for class_name in CLASS_NAMES:
        proposals = [p for p in merged if p["class_name"] == class_name]
        proposals.sort(key=lambda p: -float(p["box_score"]))
        sheets[class_name] = draw_sheet(
            f"{class_name} — {len(proposals)} proposals", proposals, class_name
        )

    cross = [
        p
        for p in merged
        if p["class_name"] == "needle" and p["stratum"] != "conifer"
    ]
    cross.sort(key=lambda p: -float(p["box_score"]))
    sheets["needle-cross-stratum"] = draw_sheet(
        f"needle outside the conifer stratum — {len(cross)} proposals",
        cross,
        "needle-cross-stratum",
    )
    (output / "index.json").write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "per_class": per_class,
                "sheets": sheets,
                "note": "audit aid only; tiles carry no review verdict",
            },
            indent=1,
        )
    )
    return sheets

# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _enrich(proposal: dict[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(proposal)
    enriched.update(
        {
            "stratum": row.get("stratum"),
            "origin": row.get("origin"),
            "taxon_name": row.get("taxon_name"),
            "taxon_rank": row.get("taxon_rank"),
            "taxon_family": row.get("taxon_family"),
            "taxon_genus": row.get("taxon_genus"),
            "is_conifer_taxon": bool(is_conifer_row(row)),
            "common_name": row.get("common_name"),
            "observed_on": row.get("observed_on"),
            "hybrid": True,
        }
    )
    return enriched


def run_hybrid(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    device: str = "cuda",
    classes: Sequence[str] | None = None,
    limit_per_class: int | None = None,
    blind_manifest_paths: Sequence[str | Path] = (),
    plan: Mapping[str, Mapping[str, Any]] | None = None,
    render_sheets: bool = True,
    log: Any = print,
) -> dict[str, Any]:
    """Run the reviewed hybrid operating point over the candidate pool."""

    import torch

    from .owlv2 import run_owlv2
    from .yoloe import run_yoloe_variants

    plan = plan or HYBRID_PLAN
    output = Path(out_dir)
    raw_root = output / "hybrid-raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    rows = load_candidate_manifest(manifest_path)
    row_by_image = {row["image_id"]: row for row in rows}
    selected_classes = list(classes or CLASS_NAMES)

    blind_check: dict[str, Any] = {
        "performed": bool(blind_manifest_paths),
        "passed": None,
    }
    if blind_manifest_paths:
        blind_check = verify_blind_exclusion(rows, blind_manifest_paths)
        (output / "blind-check.json").write_text(json.dumps(blind_check, indent=1))
        log(
            f"blind exclusion: {blind_check['overlap_counts']} "
            f"passed={blind_check['passed']}"
        )
        if not blind_check["passed"]:
            raise SystemExit("blind-test overlap detected; refusing to propose")

    runs: dict[str, dict[str, Any]] = {}
    raw_by_class: dict[str, dict[str, Any]] = {}
    all_proposals: list[dict[str, Any]] = []

    for class_name in selected_classes:
        entry = plan[class_name]
        scoped = scope_rows(rows, class_name, plan=plan)
        if limit_per_class is not None:
            scoped = scoped[:limit_per_class]
        manifest = detector_manifest(scoped)
        config = _single_class_config(class_name, entry)
        out_path = raw_root / f"{class_name}-{entry['backend']}"
        log(
            f"[{class_name}] {entry['backend']} on {len(scoped)} images, "
            f"prompts={entry['prompts']} floor={entry['score_floor']}"
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if entry["backend"] == "yoloe":
            result = run_yoloe_variants(
                manifest,
                config,
                [
                    {
                        "class_name": class_name,
                        "prompt": entry["prompts"][0],
                        "box_threshold": entry["score_floor"],
                    }
                ],
                out_path,
                model_name=entry["model"],
                imgsz=entry["imgsz"],
                device=device,
                conf_floor=entry["conf_floor"],
            )
        elif entry["backend"] == "owlv2":
            result = run_owlv2(
                manifest,
                config,
                out_path,
                model_name=entry["model"],
                device=device,
                score_threshold=0.02,
                iou=NMS_IOU,
                max_per_class=MAX_PER_CLASS_PER_IMAGE,
                prompts_override={class_name: entry["prompts"]},
                keep_duplicates=True,
            )
        else:  # pragma: no cover - plan is static
            raise ValueError(f"unknown backend {entry['backend']}")

        peak = (
            round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available()
            else 0.0
        )
        summary = result["summary"]
        runs[class_name] = {
            "backend": entry["backend"],
            "model": entry["model"],
            "prompts": list(entry["prompts"]),
            "score_floor": entry["score_floor"],
            "scope_images": len(scoped),
            "images_processed": summary.get("images_processed"),
            "runtime_seconds": round(float(summary.get("total_runtime_seconds") or 0.0), 2),
            "peak_gpu_mb": peak,
            "raw_detections_above_floor": summary.get("raw_detections_above_floor"),
            "rejected_by_reason": rejected,
            "raw_manifest": str(out_path / "proposals.json"),
            "raw_manifest_sha256": sha256_file(out_path / "proposals.json"),
        }
        raw_by_class[class_name] = result

        for proposal in result.get("proposals", []):
            if proposal.get("status") != "proposed":
                continue
            row = row_by_image.get(str(proposal["image_id"]))
            if row is None:
                continue
            all_proposals.append(_enrich(proposal, row))

        log(
            f"[{class_name}] raw={len(result.get('proposals', []))} "
            f"kept={sum(1 for p in all_proposals if p['class_name'] == class_name)} "
            f"{runs[class_name]['runtime_seconds']}s peak={peak}MB"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The benchmark's per-image cap of 10 was a review-volume bound. For
    # training candidates it would silently drop the 11th shoot on a conifer
    # branch, so the cap is off here; NMS still merges cross-prompt duplicates.
    merged = merge_prompt_duplicates(all_proposals, max_per_class=None)
    merged_by_class = Counter(p["class_name"] for p in merged)
    log(f"merged proposals by class: {dict(merged_by_class)}")

    stats = build_stats(
        rows=rows,
        merged=merged,
        raw_by_class=raw_by_class,
        runs=runs,
        plan=plan,
        blind_check=blind_check,
    )

    doc = {
        "schema": "arq.anatomy4.hybrid-proposals/v1",
        "contract": "anatomy4-v1",
        "created_at": utc_now(),
        "purpose": (
            "training-candidate proposals from the reviewed hybrid operating "
            "point; no ground truth, no review verdicts, no auto-accept"
        ),
        "source_manifest": str(manifest_path),
        "plan": stats["plan"],
        "guards": stats["guards"],
        "pool": {
            "images": len(rows),
            "images_by_stratum": dict(Counter(row["stratum"] for row in rows)),
            "background_detector_blind": True,
            "blind_check": blind_check,
        },
        "raw_runs": runs,
        "images_count": len(rows),
        "proposals": merged,
        "summary": {
            "proposals_total": len(merged),
            "proposals_by_class": dict(merged_by_class),
            "images_with_any_proposal": stats["images_with_any_proposal"],
            "duplicates_merged": sum(int(p.get("duplicate_count") or 0) for p in merged),
        },
    }
    if render_sheets:
        sheets = render_audit_sheets(merged, rows, output / "hybrid-audit")
        stats["audit_sheets"] = sheets

    # ``stats.json`` is the requested canonical name, but this directory also
    # holds the earlier YOLOE-only candidate pass. Keep that file intact by
    # copying it aside once, then write the hybrid stats to the canonical name.
    canonical = output / "stats.json"
    backup = output / "baseline-yoloe-stats.json"
    if backup.exists():
        stats.setdefault("notes", []).append(
            f"previous YOLOE-only candidate stats preserved at {backup.name}"
        )
    if canonical.exists() and not backup.exists():
        try:
            existing = json.loads(canonical.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        if existing.get("schema") != "arq.anatomy4.hybrid-stats/v1":
            backup.write_text(canonical.read_text())
            stats["notes"] = [
                f"previous YOLOE-only candidate stats preserved at {backup.name}",
                "baseline proposals remain at proposals.json; this file describes the hybrid pass",
            ]

    (output / "hybrid-proposals.json").write_text(json.dumps(doc, indent=1))
    (output / "hybrid-stats.json").write_text(json.dumps(stats, indent=1))
    canonical.write_text(json.dumps(stats, indent=1))

    return stats


def reindex_hybrid(
    out_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    iou_threshold: float = NMS_IOU,
    max_per_class: int | None = None,
    render_sheets: bool = True,
    log: Any = print,
) -> dict[str, Any]:
    """Rebuild the merged manifest and stats from the raw detector outputs.

    Raw proposals are the durable artifact: dedupe, the per-image cap and the
    stats can be recomputed without touching the GPU. The default cap here is
    ``None`` — the reviewed per-class cap of 10 was a *benchmark* bound, and for
    training candidates silently dropping the 11th shoot on a conifer branch
    would be the wrong trade. NMS still merges cross-prompt duplicates.
    """

    output = Path(out_dir)
    doc_path = output / "hybrid-proposals.json"
    doc = json.loads(doc_path.read_text())
    stats_path = output / "stats.json"
    previous = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    manifest_path = manifest_path or doc.get("source_manifest")
    if not manifest_path:
        raise ValueError("no source manifest recorded; pass --manifest")
    rows = load_candidate_manifest(manifest_path)
    row_by_image = {row["image_id"]: row for row in rows}

    raw_by_class: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, Any]] = {}
    all_proposals: list[dict[str, Any]] = []
    for class_name, stored in (doc.get("raw_runs") or {}).items():
        raw_path = Path(stored["raw_manifest"])
        raw = json.loads(raw_path.read_text())
        raw_by_class[class_name] = raw
        peak = (
            (previous.get("per_class", {}).get(class_name) or {}).get("peak_gpu_mb")
            or stored.get("peak_gpu_mb")
            or 0.0
        )
        runs[class_name] = {
            "backend": stored.get("backend"),
            "model": stored.get("model"),
            "prompts": stored.get("prompts"),
            "score_floor": stored.get("score_floor"),
            "scope_images": stored.get("scope_images"),
            "images_processed": raw.get("summary", {}).get("images_processed"),
            "runtime_seconds": round(
                float(raw.get("summary", {}).get("total_runtime_seconds") or 0.0), 2
            ),
            "peak_gpu_mb": peak,
            "raw_detections_above_floor": len(raw.get("proposals", [])),
            "rejected_by_reason": stored.get("rejected_by_reason", {}),
            "raw_manifest": str(raw_path),
            "raw_manifest_sha256": sha256_file(raw_path),
        }
        for proposal in raw.get("proposals", []):
            if proposal.get("status") != "proposed":
                continue
            row = row_by_image.get(str(proposal["image_id"]))
            if row is None:
                continue
            all_proposals.append(_enrich(proposal, row))
        log(f"[{class_name}] raw proposed={len(raw.get('proposals', []))}")

    merged = merge_prompt_duplicates(
        all_proposals, iou_threshold=iou_threshold, max_per_class=max_per_class
    )
    merged_by_class = Counter(p["class_name"] for p in merged)

    stats = build_stats(
        rows=rows,
        merged=merged,
        raw_by_class=raw_by_class,
        runs=runs,
        blind_check=doc.get("pool", {}).get("blind_check") or {},
    )
    if render_sheets:
        stats["audit_sheets"] = render_audit_sheets(merged, rows, output / "hybrid-audit")
    stats["reindexed_at"] = utc_now()
    if (output / "baseline-yoloe-stats.json").exists():
        stats.setdefault("notes", []).extend(
            [
                "previous YOLOE-only candidate stats preserved at baseline-yoloe-stats.json",
                "proposals.json is still the YOLOE-only baseline; hybrid-proposals.json is this pass",
            ]
        )
    stats["reindex_notes"] = [
        "merged manifest rebuilt from raw detector outputs; no inference was re-run",
        f"cross-prompt NMS IoU = {iou_threshold}",
        f"per-image cap = {max_per_class if max_per_class is not None else 'none'}",
    ]

    doc["proposals"] = merged
    doc["created_at"] = doc.get("created_at")
    doc["reindexed_at"] = stats["reindexed_at"]
    doc["raw_runs"] = runs
    doc["summary"] = {
        "proposals_total": len(merged),
        "proposals_by_class": dict(merged_by_class),
        "images_with_any_proposal": stats["images_with_any_proposal"],
        "duplicates_merged": sum(int(p.get("duplicate_count") or 0) for p in merged),
    }
    doc_path.write_text(json.dumps(doc, indent=1))
    (output / "hybrid-stats.json").write_text(json.dumps(stats, indent=1))
    (output / "stats.json").write_text(json.dumps(stats, indent=1))
    log(f"merged={dict(merged_by_class)} total={len(merged)}")
    return stats
