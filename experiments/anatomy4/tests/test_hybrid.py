# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import json

from experiments.anatomy4.contracts import load_config
from experiments.anatomy4.hybrid import (
    MAX_BOX_AREA,
    PLANT_STRATA,
    build_stats,
    detector_manifest,
    is_conifer_row,
    load_candidate_manifest,
    merge_prompt_duplicates,
    scope_rows,
    strata_for,
    verify_blind_exclusion,
)
from experiments.anatomy4.altdet import make_box_proposal

CONFIG = "experiments/anatomy4/configs/anatomy4-targeted.json"


def _row(image_id: str, stratum: str, **extra):
    return {
        "image_id": image_id,
        "photo_id": image_id.replace("tc-", ""),
        "observation_id": f"inat-{image_id}",
        "photo_url": f"https://example.org/photos/{image_id}/large.jpeg",
        "sha256": f"sha-{image_id}",
        "stratum": stratum,
        "origin": "new",
        "taxon_name": extra.pop("taxon_name", "Acer rubrum"),
        "taxon_family": extra.pop("taxon_family", "Sapindaceae"),
        "taxon_genus": extra.pop("taxon_genus", "Acer"),
        "local_image_path": f"/out/images/{image_id}.jpg",
        **extra,
    }


def _proposal(image_id: str, class_name: str, prompt: str, score: float, box, status="proposed"):
    config = load_config(CONFIG)
    proposal = make_box_proposal(
        photo_id=image_id,
        observation_id=f"inat-{image_id}",
        image_path=f"/out/images/{image_id}.jpg",
        audit_bucket="conifer",
        class_name=class_name,
        prompt=prompt,
        score=score,
        box=box,
        width=100.0,
        height=100.0,
        provenance={"backend": "test"},
        detector_rank=0,
        slug="h",
        config=config,
    )
    proposal["status"] = status
    return proposal


def test_plant_strata_exclude_background() -> None:
    assert "background" not in PLANT_STRATA
    assert strata_for("needle") == list(PLANT_STRATA)
    assert strata_for("leaf") == ["broadleaf"]


def test_scope_needle_runs_on_every_plant_stratum() -> None:
    rows = [_row(f"tc-{i}", stratum) for i, stratum in enumerate(PLANT_STRATA + ("background",))]
    needle_scope = {row["stratum"] for row in scope_rows(rows, "needle")}
    assert needle_scope == set(PLANT_STRATA)
    assert "background" not in needle_scope

    leaf_scope = {row["stratum"] for row in scope_rows(rows, "leaf")}
    assert leaf_scope == {"broadleaf"}


def test_scope_refuses_background_blindness_breach() -> None:
    rows = [_row("tc-1", "background")]
    plan = {"needle": {"strata": ["background"]}}
    try:
        scope_rows(rows, "needle", plan=plan)
    except ValueError as error:
        assert "detector-blind" in str(error)
    else:  # pragma: no cover
        raise AssertionError("background scope must raise")


def test_conifer_detection_uses_taxonomy_not_stratum() -> None:
    assert is_conifer_row({"taxon_family": "Pinaceae"})
    assert is_conifer_row({"taxon_genus": "Tsuga"})
    assert not is_conifer_row({"taxon_family": "Sapindaceae", "taxon_genus": "Acer"})
    assert not is_conifer_row({})


def test_detector_manifest_keeps_image_identity_and_path() -> None:
    rows = [_row("tc-abc", "conifer")]
    manifest = detector_manifest(rows)
    image = manifest["images"][0]
    assert image["photo_id"] == "tc-abc"
    assert image["audit_bucket"] == "conifer"
    assert image["download"]["path"] == "/out/images/tc-abc.jpg"


def test_merge_preserves_cross_prompt_provenance() -> None:
    image = "tc-1"
    proposals = [
        _proposal(image, "needle", "conifer foliage", 0.31, [10, 10, 30, 30]),
        _proposal(image, "needle", "spruce needles", 0.22, [11, 11, 31, 31]),
        _proposal(image, "needle", "spruce needles", 0.12, [60, 60, 90, 90]),
    ]
    merged = merge_prompt_duplicates(proposals)
    assert len(merged) == 2
    top = max(merged, key=lambda p: p["box_score"])
    assert top["prompt"] == "conifer foliage"
    assert top["duplicate_count"] == 1
    assert top["prompt_count"] == 2
    assert {source["prompt"] for source in top["source_proposals"]} == {
        "conifer foliage",
        "spruce needles",
    }
    assert all("verdict" not in p for p in merged)
    assert all(p["review_status"] == "unreviewed" for p in merged)


def test_merge_is_per_class_not_across_classes() -> None:
    image = "tc-2"
    proposals = [
        _proposal(image, "flower", "single flower", 0.4, [10, 10, 30, 30]),
        _proposal(image, "inflorescence", "umbel of flowers", 0.3, [10, 10, 30, 30]),
    ]
    merged = merge_prompt_duplicates(proposals)
    assert len(merged) == 2
    assert {p["class_name"] for p in merged} == {"flower", "inflorescence"}


def test_merge_ignores_rejected_and_applies_cap() -> None:
    image = "tc-3"
    rejected = _proposal(
        image, "needle", "conifer foliage", 0.9, [0, 0, 99, 99],
        status="rejected_area:dominates_frame",
    )
    distant = [
        _proposal(image, "needle", "conifer foliage", 0.5 - index * 0.01,
                  [index * 5, index * 5, index * 5 + 4, index * 5 + 4])
        for index in range(12)
    ]
    merged = merge_prompt_duplicates([rejected, *distant], max_per_class=3)
    assert len(merged) == 3
    assert all(p["status"] == "proposed" for p in merged)


def test_verify_blind_exclusion_flags_overlap(tmp_path) -> None:
    rows = [_row("tc-1", "conifer"), _row("tc-2", "broadleaf")]
    blind = tmp_path / "blind.jsonl"
    blind.write_text(
        json.dumps(
            {
                "review_image_id": "b-1",
                "observation_id": "inat-tc-1",
                "photo_url": "https://example.org/photos/tc-1/large.jpeg",
                "sha256": "sha-tc-1",
            }
        )
        + "\n"
    )
    result = verify_blind_exclusion(rows, [blind])
    assert result["passed"] is False
    assert result["overlap_counts"]["observation_ids"] == 1
    assert result["overlap_counts"]["photo_urls"] == 1
    assert result["overlap_counts"]["sha256"] == 1


def test_verify_blind_exclusion_passes_on_clean_pool(tmp_path) -> None:
    rows = [_row("tc-1", "conifer")]
    blind = tmp_path / "blind.jsonl"
    blind.write_text(
        json.dumps(
            {
                "review_image_id": "b-1",
                "observation_id": "inat-other",
                "photo_url": "https://example.org/photos/999/large.jpeg",
                "sha256": "sha-other",
            }
        )
        + "\n"
    )
    result = verify_blind_exclusion(rows, [blind])
    assert result["passed"] is True
    assert sum(result["overlap_counts"].values()) == 0


def test_stats_report_conifer_provenance_outside_stratum() -> None:
    rows = [
        _row("tc-1", "conifer", taxon_family="Pinaceae", taxon_genus="Pinus"),
        _row("tc-2", "flower", taxon_family="Pinaceae", taxon_genus="Pinus"),
        _row("tc-3", "broadleaf"),
        _row("tc-4", "background"),
    ]
    proposals = [
        {**_proposal("tc-1", "needle", "conifer foliage", 0.4, [10, 10, 30, 30]),
         "stratum": "conifer", "is_conifer_taxon": True, "source_proposals": [], "duplicate_count": 0},
        {**_proposal("tc-2", "needle", "spruce needles", 0.3, [10, 10, 30, 30]),
         "stratum": "flower", "is_conifer_taxon": True, "source_proposals": [], "duplicate_count": 0},
        {**_proposal("tc-3", "leaf", "leaf", 0.5, [10, 10, 30, 30]),
         "stratum": "broadleaf", "is_conifer_taxon": False, "source_proposals": [], "duplicate_count": 0},
    ]
    for proposal in proposals:
        proposal["source_proposals"] = [
            {"proposal_id": proposal["proposal_id"], "prompt": proposal["prompt"],
             "box_score": proposal["box_score"], "detector_bbox": proposal["detector_bbox"],
             "detector_rank": 0}
        ]
        proposal["prompts"] = [proposal["prompt"]]
    stats = build_stats(
        rows=rows,
        merged=proposals,
        raw_by_class={},
        runs={},
        blind_check={"passed": True},
    )
    conifer = stats["conifer_provenance"]
    assert conifer["taxon_is_conifer_images"] == 2
    assert conifer["in_conifer_stratum"] == 1
    assert conifer["outside_conifer_stratum"] == 1
    assert conifer["outside_by_stratum"] == {"flower": 1}
    assert conifer["needle_proposals_from_outside"] == 1
    assert stats["background_stratum"]["detector_blind"] is True
    assert stats["proposals_by_class"] == {"leaf": 1, "needle": 2}


def test_guards_are_the_reviewed_operating_point() -> None:
    assert MAX_BOX_AREA == 0.60