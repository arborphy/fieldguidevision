# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import json

from experiments.anatomy4.altsheets import empty_image_stats, sample_evenly
from experiments.anatomy4.altdet import dedupe_same_class, make_box_proposal
from experiments.anatomy4.compare import summarize
from experiments.anatomy4.contracts import load_config

CONFIG = "experiments/anatomy4/configs/anatomy4-targeted.json"


def _proposal(
    score: float,
    box,
    class_name: str = "leaf",
    prompt: str = "leaf",
    config=None,
    photo_id: str = "1",
):
    config = config or load_config(CONFIG)
    return make_box_proposal(
        photo_id=photo_id,
        observation_id="obs-1",
        image_path="/tmp/1.jpg",
        audit_bucket="leaf",
        class_name=class_name,
        prompt=prompt,
        score=score,
        box=box,
        width=100.0,
        height=100.0,
        provenance={"backend": "test"},
        detector_rank=int(score * 100),
        slug="t",
        config=config,
    )


def test_box_proposal_is_normalized_and_has_no_fake_polygon() -> None:
    proposal = _proposal(0.5, (200, -10, 50, 60))
    assert proposal["detector_bbox"] == [0.5, 0.0, 1.0, 0.6]
    assert proposal["proposal_kind"] == "box"
    assert proposal["polygon"] == []
    assert proposal["bbox"] == proposal["detector_bbox"]
    assert proposal["status"] == "proposed"


def test_box_proposal_rejects_frame_dominating_and_tiny_boxes() -> None:
    huge = _proposal(0.9, (-5, -5, 105, 105))
    assert huge["status"] == "rejected_area:dominates_frame"
    tiny = _proposal(0.9, (10, 10, 11, 11))
    assert tiny["status"] == "rejected_area:too_small"


def test_dedupe_keeps_only_same_class_overlaps() -> None:
    same = [
        _proposal(0.8, (0, 0, 50, 50), prompt="leaf"),
        _proposal(0.6, (1, 1, 50, 50), prompt="leaf blade"),
        _proposal(0.5, (60, 60, 90, 90), prompt="broadleaf"),
    ]
    other = _proposal(0.7, (0, 0, 50, 50), class_name="flower", prompt="flower")
    kept = dedupe_same_class([*same, other])
    assert len(kept) == 3
    assert {p["class_name"] for p in kept} == {"leaf", "flower"}
    leaf_scores = sorted(p["box_score"] for p in kept if p["class_name"] == "leaf")
    assert leaf_scores == [0.5, 0.8]


def test_review_sample_is_deterministic_and_covers_score_range() -> None:
    proposals = [{"box_score": value / 100} for value in range(100)]
    first = sample_evenly(proposals, 24)
    second = sample_evenly(proposals, 24)
    assert [p["box_score"] for p in first] == [p["box_score"] for p in second]
    assert len(first) == 24
    assert first[0]["box_score"] == 0.0
    assert first[-1]["box_score"] == 0.99


def test_compare_scores_precision_tightness_and_matched_n() -> None:
    proposals = [
        _proposal(0.9, (0, 0, 80, 80)),  # dominates -> would be filtered; keep small
        _proposal(0.8, (0, 0, 40, 40)),
        _proposal(0.7, (0, 0, 40, 40)),
        _proposal(0.6, (0, 0, 40, 40)),
    ]
    reviews = {
        proposals[0]["proposal_id"]: {"verdict": "tp"},
        proposals[1]["proposal_id"]: {"verdict": "tp"},
        proposals[2]["proposal_id"]: {"verdict": "fp"},
        proposals[3]["proposal_id"]: {"verdict": "fp"},
    }
    manifest = {"backend": "test", "proposals": proposals}
    result = summarize(manifest, reviews, dataset="unit")
    leaf = result["per_class"]["leaf"]
    assert leaf["reviewed"] == 4
    assert leaf["precision"] == 0.5
    # 0.64 area is not tight at 0.60; only the 0.16-area tp counts.
    assert leaf["correct_tight"] == 1
    # Only four proposals were reviewed, so no matched-N estimate applies.
    assert leaf["precision_at_n"] == {}
    assert leaf["score_sweep"][">=0.30"]["precision"] == 0.5
    assert result["overall"]["precision"] == 0.5


def test_empty_image_stats_per_bucket() -> None:
    manifest = {
        "images": [
            {"photo_id": "a", "selected_for_audit": True, "audit_bucket": "leaf"},
            {"photo_id": "b", "selected_for_audit": True, "audit_bucket": "leaf"},
            {"photo_id": "c", "selected_for_audit": True, "audit_bucket": "background"},
            {"photo_id": "d", "selected_for_audit": False, "audit_bucket": "leaf"},
        ],
        "proposals": [_proposal(0.5, (0, 0, 20, 20), photo_id="a")],
    }
    stats = empty_image_stats(manifest)
    assert stats["leaf"]["images"] == 2
    assert stats["leaf"]["with_proposals"] == 1
    assert stats["leaf"]["empty_rate"] == 0.5
    assert stats["background"]["empty_rate"] == 1.0


def test_export_applies_class_thresholds_and_area_guard(tmp_path) -> None:
    from experiments.anatomy4.export import export_proposals

    manifest = {
        "backend": "test",
        "provenance": {"detector": "unit"},
        "proposals": [
            _proposal(0.30, (0, 0, 20, 20), class_name="needle", prompt="spruce needles"),
            _proposal(0.05, (0, 0, 20, 20), class_name="needle", prompt="spruce needles"),
            _proposal(0.40, (0, 0, 90, 90), class_name="flower", prompt="single flower"),
            _proposal(0.20, (0, 0, 20, 20), class_name="flower", prompt="single flower"),
        ],
    }
    summary = export_proposals(
        manifest, tmp_path, thresholds={"needle": 0.10, "flower": 0.25}
    )
    assert summary["needle"] == 1
    # 0.81 area exceeds the review-guided 0.60 cap; the small 0.20 box is below 0.25.
    assert summary["flower"] == 0
    assert summary["rows"] == 1
    rows = (tmp_path / "proposals.csv").read_text().strip().splitlines()
    assert len(rows) == 2
    assert rows[1].startswith("needle,0,")
