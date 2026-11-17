# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import json
from pathlib import Path

from experiments.anatomy4.policy import (
    ASSIST_SPLIT,
    TEST_SPLIT,
    build_rows,
    decision_for_row,
    default_verifier_requirement,
    fit_class,
    fit_policies,
    leave_one_out,
    rule_accepts,
    score_rule,
    split_support,
    verdict_passes,
)


def _verdict(verdict_class="leaf", *, valid=True, resolved=True, tightness="keep", confidence=1.0):
    return {
        "verdict_class": verdict_class,
        "resolved_enough": resolved,
        "bbox_tightness": tightness,
        "reason": "visible anatomy",
        "confidence": confidence,
        "valid": valid,
        "invalid_reason": None if valid else "taxon_evidence",
        "invented_box_fields": [],
        "taxon_evidence": [],
    }


def _row(score, human, split="train", verdict=None, cls="leaf", proposal_id=None):
    return {
        "proposal_id": proposal_id or f"{cls}-{score}-{split}-{human}",
        "class": cls,
        "detector_score": score,
        "human_accept": human,
        "human_notes": "",
        "split": split,
        "verdict": verdict or _verdict(cls),
        "pass": 1,
    }


def test_score_rule_counts_and_wilson_intervals() -> None:
    rows = [_row(0.5, True), _row(0.3, True), _row(0.2, False), _row(0.1, False)]
    rule = {"detector_score_min": 0.25, "verifier": default_verifier_requirement("balanced")}
    metrics = score_rule(rows, rule)
    assert (metrics["accepts"], metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 2, 0, 0)
    assert metrics["precision"] == 1.0 and metrics["recall"] == 1.0
    low, high = metrics["precision_ci95"]
    assert 0.0 < low < 1.0 and high == 1.0  # 2/2 is not certainty

    # Inclusive minimum: the 0.25 row would be included if it existed; 0.2 is not.
    assert not rule_accepts(rows[2], rule)
    assert not rule_accepts(rows[3], rule)

    # 1 FP in 4 accepts -> precision 0.75 and zero tolerance at a 0.9 target.
    loose = score_rule(rows, {"detector_score_min": 0.0, "verifier": default_verifier_requirement("balanced")})
    assert (loose["accepts"], loose["tp"], loose["fp"]) == (4, 2, 2)
    assert loose["extra_fp_tolerated_at_target"] == 0


def test_verifier_requirement_gates_resolution_and_tightness() -> None:
    balanced = default_verifier_requirement("balanced")
    strict = default_verifier_requirement("strict")
    too_tight = _verdict("leaf", tightness="too_tight")
    assert verdict_passes(too_tight, "leaf", balanced) is True
    assert verdict_passes(too_tight, "leaf", strict) is False
    unresolved = _verdict("leaf", resolved=False)
    assert verdict_passes(unresolved, "leaf", balanced) is False
    assert verdict_passes(_verdict("leaf", valid=False), "leaf", balanced) is False
    # A right-class verdict never passes for another class.
    assert verdict_passes(_verdict("flower"), "leaf", balanced) is False


def _clean_class_rows() -> list[dict]:
    rows = []
    # 10 human-accepted with scores above 0.20, 10 rejected below it.
    for index in range(10):
        rows.append(_row(0.60 - index * 0.03, True, "train" if index < 7 else "test"))
    for index in range(10):
        rows.append(_row(0.18 - index * 0.01, False, "train" if index < 5 else "val"))
    return rows


def test_fit_class_finds_a_clean_threshold_and_counts_assisted_support() -> None:
    rows = _clean_class_rows()
    # At a 0.90 target the fit tolerates one FP to keep every TP; at 0.95 it
    # must cut above the rejected band. Assert the stricter, unambiguous case.
    fitted = fit_class(rows, target_precision=0.95, min_accepts=5, min_assisted_support=5)
    assert fitted["action"] == "auto_assist"
    assert fitted["validated"] is True
    assert fitted["policy"]["detector_score_min"] >= 0.33
    assert fitted["policy"]["applies_to_splits"] == [ASSIST_SPLIT]
    assert fitted["policy"]["never_auto_accept_splits"] == ["val", "test"]
    support = fitted["measured_support"]
    assert support["all_reviewed"]["precision"] == 1.0
    assert support["all_reviewed"]["recall"] == 1.0
    assert support["assisted_train"]["accepts"] == 7
    assert support["human_reviewed_test"]["accepts"] == 3
    assert support["held_out_validation"]["precision"] == 1.0
    # No accepted proposal outside train is ever eligible for assistance.
    for row in rows:
        if row["split"] != "train":
            assert decision_for_row(row, fitted)["action"] == "review"


def test_fit_class_refuses_a_class_without_support() -> None:
    rows = [
        _row(0.9, True),
        _row(0.8, True),
        _row(0.7, False),
        _row(0.6, False),
    ]
    fitted = fit_class(rows)
    assert fitted["action"] == "review"
    assert fitted["confidence"] == "insufficient_support"
    assert fitted["policy"] is None


def test_fit_class_rejects_an_apparent_fit_that_fails_validation() -> None:
    rows = _clean_class_rows()
    fitted = fit_class(
        rows,
        target_precision=0.90,
        min_accepts=5,
        min_assisted_support=5,
        loo_min_precision=1.01,  # unreachable on purpose
    )
    assert fitted["action"] == "review"
    assert fitted["confidence"] == "unvalidated_fit"
    assert fitted["policy"] is None
    assert "unvalidated_candidate" in fitted
    assert fitted["unvalidated_candidate"]["why_not_shipped"]


def test_decision_fallbacks_deprioritize_verifier_rejects() -> None:
    class_policy = {
        "action": "auto_assist",
        "policy": {"detector_score_min": 0.2, "verifier": default_verifier_requirement("balanced")},
    }
    accepted = _row(0.5, True)
    assert decision_for_row(accepted, class_policy)["action"] == "auto_assist"

    below = _row(0.1, True)
    assert decision_for_row(below, class_policy) == {"action": "review", "reason": "policy_reject"}

    junk = _row(0.5, False, verdict=_verdict("not_target", resolved=False, confidence=0.0))
    assert decision_for_row(junk, class_policy) == {
        "action": "deprioritize",
        "reason": "verifier_not_target",
    }

    test_row = _row(0.9, True, "test")
    assert decision_for_row(test_row, class_policy) == {
        "action": "review",
        "reason": "non_train_split",
    }


def test_build_rows_joins_score_verdict_review_and_split(tmp_path: Path) -> None:
    proposals = {
        "proposals": [
            {
                "proposal_id": "p1",
                "class_name": "leaf",
                "box_score": 0.42,
                "observation_id": "inat-1",
            },
            {
                "proposal_id": "p2",
                "class_name": "needle",
                "box_score": 0.11,
                "observation_id": "inat-2",
            },
        ]
    }
    (tmp_path / "proposals.json").write_text(json.dumps(proposals))
    (tmp_path / "review.csv").write_text(
        "proposal_id,class_name,verdict,notes\np1,leaf,tp,\np2,needle,fp,moss\n"
    )
    with (tmp_path / "verdicts.jsonl").open("w") as handle:
        handle.write(json.dumps({"proposal_id": "p1", "proposed_class": "leaf", "pass": 1,
                                 "verdict": _verdict("leaf")}) + "\n")
        handle.write(json.dumps({"proposal_id": "p2", "proposed_class": "needle", "pass": 1,
                                 "verdict": _verdict("not_target", resolved=False)}) + "\n")
    rows = build_rows(
        proposals_path=tmp_path / "proposals.json",
        verdicts_path=tmp_path / "verdicts.jsonl",
        review_path=tmp_path / "review.csv",
    )
    by_id = {row["proposal_id"]: row for row in rows}
    assert by_id["p1"]["detector_score"] == 0.42
    assert by_id["p1"]["human_accept"] is True
    assert by_id["p2"]["human_accept"] is False
    assert by_id["p1"]["split"] in {"train", "val", "test"}
    assert by_id["p2"]["verdict"]["verdict_class"] == "not_target"


def test_leave_one_out_applies_the_selection_procedure() -> None:
    rows = _clean_class_rows()

    def select(pool):
        best = None
        for threshold in sorted({row["detector_score"] for row in pool}):
            rule = {"detector_score_min": threshold, "verifier": default_verifier_requirement("balanced")}
            metrics = score_rule(pool, rule)
            if metrics["accepts"] >= 5 and (metrics["precision"] or 0) >= 0.95:
                best = rule
                break
        return best

    loo = leave_one_out(rows, select=select)
    assert loo["accepts"] > 0
    assert loo["precision"] == 1.0
    assert loo["recall"] is not None


def test_fit_policies_writes_decisions_only_for_train(tmp_path: Path) -> None:
    # Reuse build_rows' real split assignment by generating many observations.
    proposals = []
    verdicts = []
    review_lines = ["proposal_id,class_name,verdict,notes"]
    for index in range(40):
        proposal_id = f"p{index}"
        proposals.append(
            {
                "proposal_id": proposal_id,
                "class_name": "leaf",
                "box_score": 0.5 if index % 2 == 0 else 0.1,
                "observation_id": f"inat-{index}",
            }
        )
        verdicts.append(
            {
                "proposal_id": proposal_id,
                "proposed_class": "leaf",
                "pass": 1,
                "verdict": _verdict("leaf"),
            }
        )
        review_lines.append(f"{proposal_id},leaf,{'tp' if index % 2 == 0 else 'fp'},")
    (tmp_path / "proposals.json").write_text(json.dumps({"proposals": proposals}))
    (tmp_path / "verdicts.jsonl").write_text("\n".join(json.dumps(v) for v in verdicts) + "\n")
    (tmp_path / "review.csv").write_text("\n".join(review_lines) + "\n")

    policy = fit_policies(
        proposals_path=tmp_path / "proposals.json",
        verdicts_path=tmp_path / "verdicts.jsonl",
        review_path=tmp_path / "review.csv",
        decision_out=tmp_path / "decisions.jsonl",
    )
    assert policy["schema"] == "arq.anatomy4.verify-policy/v1"
    assert policy["lanes"]["human_reviewed_truth"]["auto_accept"] is False
    assert policy["policy_id"]

    decisions = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert len(decisions) == 40
    for decision in decisions:
        if decision["split"] != "train":
            assert decision["action"] == "review"
            assert decision["action_reason"] == "non_train_split"
        if decision["action"] == "auto_assist":
            assert decision["verifier_verdict_class"] == "leaf"

def test_operational_summary_never_assists_non_train() -> None:
    from experiments.anatomy4.policy import _operational_summary

    per_class = {
        "leaf": {
            "action": "auto_assist",
            "policy": {
                "detector_score_min": 0.2,
                "verifier": default_verifier_requirement("balanced"),
            },
        }
    }
    rows = [
        _row(0.5, True, "train", cls="leaf", proposal_id="a"),
        _row(0.5, False, "train", cls="leaf", proposal_id="b"),
        _row(0.5, True, "test", cls="leaf", proposal_id="c"),
        _row(0.1, True, "train", cls="leaf", proposal_id="d"),
    ]
    summary = _operational_summary(rows, per_class)
    # The policy cannot see human labels: both 0.5-score train rows are accepted,
    # one of which is a false accept. The summary must report that honestly.
    assert summary["assisted_split"]["auto_assist"] == 2
    assert summary["assisted_split"]["auto_assist_precision"] == 0.5
    assert summary["assisted_split"]["review"] == 1
    assert summary["human_reviewed_splits"]["auto_assist"] == 0
    assert summary["human_reviewed_splits"]["review"] == 1
    assert summary["human_positives_auto_assisted"] == 1
    assert summary["human_positives_left_for_review"] == 2
    assert summary["assisted_split_review_before"] == 3
    assert summary["assisted_split_review_after"] == 1


def test_assisted_precision_gate_blocks_a_rule_with_a_train_fp() -> None:
    rows = []
    # 6 accepted true leaves in test/val, 1 accepted false leaf in train.
    for index in range(6):
        rows.append(_row(0.60 - index * 0.02, True, "test" if index % 2 else "val"))
    rows.append(_row(0.50, False, "train"))
    # Enough train true labels to satisfy the count gate.
    for index in range(5):
        rows.append(_row(0.40 - index * 0.02, True, "train"))
    fitted = fit_class(
        rows,
        target_precision=0.90,
        min_accepts=5,
        min_assisted_support=5,
        min_assisted_precision=0.90,
    )
    # The no-floor rule is 11/12 overall (0.917) but the train lane is 5/6 (0.833).
    assert fitted["action"] == "review"
    assert fitted["policy"] is None
    assert "0.90" in fitted["decision_reason"]


def test_validation_gate_blocks_an_inflated_fit() -> None:
    rows = [_row(0.60 - index * 0.02, True, "train" if index < 8 else "test") for index in range(10)]
    rows += [_row(0.10 - index * 0.01, False, "train") for index in range(6)]
    # A separate validation population where the same rule accepts only FPs.
    validation = [
        _row(0.60, False, "test", proposal_id="v1"),
        _row(0.58, False, "test", proposal_id="v2"),
        _row(0.55, False, "test", proposal_id="v3"),
    ]
    fitted = fit_class(
        rows,
        target_precision=0.90,
        min_accepts=5,
        min_assisted_support=5,
        validation_rows=validation,
        validation_min_precision=0.80,
        validation_min_accepts=3,
    )
    assert fitted["action"] == "review"
    assert fitted["policy"] is None
    assert "validation precision" in fitted["decision_reason"]
    assert fitted["measured_support"]["independent_validation"]["fp"] == 3


def test_owlv2_prep_keeps_populations_separate(tmp_path: Path) -> None:
    from experiments.anatomy4.owlv2_prep import build_inputs, classify_source

    alt = tmp_path / "alt"
    (alt / "broad-review").mkdir(parents=True)
    header = "tile,proposal_id,class_name,box_score,photo_id,bucket,prompt,area,verdict,notes\n"
    (alt / "review-leaf.csv").write_text(
        header + "0,a4o-1-leaf-leaf-001,leaf,0.5,11,leaf,leaf,0.2,tp,blade\n"
    )
    (alt / "review-needle-bg.csv").write_text(
        header + "0,a4o-2-needle-pine-needles-002,needle,0.4,12,background,pine needles,0.2,fp,litter\n"
    )
    (alt / "broad-review" / "review-flower.csv").write_text(
        header + "0,a4o-3-flower-single-flower-003,flower,0.3,13,,single flower,0.2,tp,bloom\n"
    )
    square = [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]
    proposals = {
        "proposals": [
            {
                "proposal_id": proposal_id,
                "photo_id": str(photo),
                "class_name": cls,
                "box_score": 0.5,
                "detector_bbox": [0.1, 0.1, 0.2, 0.2],
                "polygon": square,
                "observation_id": f"obs-{index}",
            }
            for index, proposal_id, cls, photo in (
                (1, "a4o-1-leaf-leaf-001", "leaf", 11),
                (2, "a4o-2-needle-pine-needles-002", "needle", 12),
                (3, "a4o-3-flower-single-flower-003", "flower", 13),
            )
        ]
    }
    (alt / "targeted-proposals.json").write_text(json.dumps(proposals))
    (alt / "broad-proposals.json").write_text(json.dumps({"proposals": []}))

    assert classify_source(alt / "review-needle-bg.csv") == "background"
    assert classify_source(alt / "broad-review" / "review-flower.csv") == "broad"
    assert classify_source(alt / "review-leaf.csv") == "targeted"

    summary = build_inputs(alt_root=alt, out_dir=tmp_path / "out", image_source={
        "targeted": "/tmp/t", "background": "/tmp/t", "broad": "/tmp/b"
    })
    assert summary["populations"]["targeted"]["rows"] == 1
    assert summary["populations"]["background"]["rows"] == 1
    assert summary["populations"]["broad"]["rows"] == 1
    assert summary["unique_photos"] == 3
    assert (tmp_path / "out" / "review-targeted.csv").exists()
    manifest = json.loads((tmp_path / "out" / "image-manifest.json").read_text())
    assert manifest["11"]["populations"] == ["targeted"]
    assert manifest["12"]["source_dir"] == "/tmp/t"
    assert manifest["13"]["source_dir"] == "/tmp/b"


def test_production_selection_is_stratified_and_caps_per_image() -> None:
    from experiments.anatomy4.train_queue import select_production_candidates

    yoloe = []
    for cls, floor in (("leaf", 0.10), ("inflorescence", 0.06)):
        for index in range(40):
            score = floor + (index % 4) * 0.15
            yoloe.append(
                {
                    "proposal_id": f"y-{cls}-{index}",
                    "class_name": cls,
                    "photo_id": f"yp-{index // 2}",
                    "observation_id": f"obs-y-{index}",
                    "status": "proposed",
                    "box_score": score,
                    "model": {"backend": "yoloe", "detector": "yoloe-26s-seg.pt"},
                }
            )
    owl = []
    for index in range(45):
        owl.append(
            {
                "proposal_id": f"o-{index}",
                "class_name": "flower",
                "photo_id": f"op-{index // 3}",
                "observation_id": f"obs-o-{index}",
                "status": "proposed",
                "box_score": 0.15 + index * 0.01,
                "model": {"backend": "owlv2", "detector": "google/owlv2-base-patch16-ensemble"},
            }
        )
    selected, meta = select_production_candidates(
        yoloe_proposals=yoloe,
        owlv2_proposals=owl,
        leaf_count=20,
        inflorescence_count=20,
        flower_count=30,
        max_per_photo=3,
    )
    assert len(selected) == 70
    assert all(sum(1 for p in selected if p["class_name"] == cls) == count for cls, count in (
        ("leaf", 20), ("inflorescence", 20), ("flower", 30)
    ))
    for cls in ("leaf", "inflorescence", "flower"):
        rows = [p for p in selected if p["class_name"] == cls]
        assert max(
            sum(1 for p in rows if p["photo_id"] == photo)
            for photo in {p["photo_id"] for p in rows}
        ) <= 3
    assert meta["unique_photos"]["flower_owlv2"] == 15
    assert meta["score_strata_selected"]["leaf_yoloe"]
    assert all(p["box_score"] >= 0.15 for p in selected if p["class_name"] == "flower")


def test_production_accept_only_uses_supported_class_policy() -> None:
    from experiments.anatomy4.train_queue import _policy_accepts

    yoloe_policy = {
        "policy_id": "yoloe-leaf",
        "per_class": {
            "leaf": {
                "action": "auto_assist",
                "policy": {
                    "applies_to_splits": ["train"],
                    "detector_score_min": 0.2,
                    "verifier": default_verifier_requirement("balanced"),
                },
            },
            "inflorescence": {"action": "review", "policy": None},
        },
    }
    leaf = {"class_name": "leaf", "box_score": 0.25}
    assert _policy_accepts(leaf, _verdict("leaf"), yoloe_policy)[0] is True
    low = {"class_name": "leaf", "box_score": 0.1}
    assert _policy_accepts(low, _verdict("leaf"), yoloe_policy)[0] is False
    inflo = {"class_name": "inflorescence", "box_score": 0.9}
    assert _policy_accepts(inflo, _verdict("inflorescence"), yoloe_policy) == (False, None)


def test_train_candidates_filter_excludes_development_and_nonproposed() -> None:
    from experiments.anatomy4.train_queue import _is_new_train_candidate

    assert _is_new_train_candidate({"status": "proposed"})
    assert _is_new_train_candidate({"status": "proposed", "origin": "new", "proposal_state": "proposal_only", "review_status": "unreviewed"})
    assert not _is_new_train_candidate({"status": "proposed", "origin": "development"})
    assert not _is_new_train_candidate({"status": "rejected"})
