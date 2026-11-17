# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.anatomy4.verify import (
    VerdictParseError,
    accept_decision,
    build_report,
    crop_pixels,
    detect_taxon_evidence,
    extract_json_payload,
    infer_human_class,
    make_crop,
    normalize_class,
    parse_verdicts,
    read_env_file,
)


def _verdict(**overrides):
    payload = {
        "class": "leaf",
        "resolved_enough": True,
        "bbox_tightness": "keep",
        "reason": "a flat green blade with visible veins",
        "confidence": 0.9,
    }
    if "class_" in overrides:
        overrides["class"] = overrides.pop("class_")
    payload.update(overrides)
    return payload


def test_extract_json_tolerates_fences_and_prose_and_rejects_empty() -> None:
    fenced = '```json\n{"verdicts": []}\n```'
    assert extract_json_payload(fenced) == {"verdicts": []}
    cited = 'Here is the result:\n{"verdicts": []}\nThank you.'
    assert extract_json_payload(cited) == {"verdicts": []}
    with pytest.raises(VerdictParseError):
        extract_json_payload("no json here")
    with pytest.raises(VerdictParseError):
        extract_json_payload("")


def test_parse_verdicts_requires_one_object_per_crop() -> None:
    content = json.dumps({"verdicts": [_verdict(), _verdict(class_="needle")]})
    # Unknown key is ignored; the class field is authoritative.
    verdicts = parse_verdicts(content, expected=2, proposed_classes=["leaf", "needle"])
    assert [v["verdict_class"] for v in verdicts] == ["leaf", "needle"]
    assert all(v["valid"] for v in verdicts)

    with pytest.raises(VerdictParseError):
        parse_verdicts(content, expected=1, proposed_classes=["leaf"])


def test_verdict_class_and_tightness_are_normalized() -> None:
    assert normalize_class("Pine Needles") == "needle"
    assert normalize_class("cluster of flowers") == "inflorescence"
    assert normalize_class("uncertain") == "ambiguous"
    content = json.dumps(
        {
            "verdicts": [
                {
                    **_verdict(class_="leaves", bbox_tightness="too broad"),
                    "confidence": 92,
                }
            ]
        }
    )
    verdict = parse_verdicts(content, expected=1, proposed_classes=["leaf"])[0]
    assert verdict["verdict_class"] == "leaf"
    assert verdict["bbox_tightness"] == "too_broad"
    assert verdict["confidence"] == 0.92


def test_missing_or_bad_fields_are_rejected() -> None:
    with pytest.raises(VerdictParseError):
        parse_verdicts(
            json.dumps({"verdicts": [{"class": "leaf"}]}), expected=1
        )
    with pytest.raises(VerdictParseError):
        parse_verdicts(
            json.dumps({"verdicts": [{"class": "mushroom", "resolved_enough": True,
                                      "bbox_tightness": "keep", "reason": "x", "confidence": 0.5}]}),
            expected=1,
        )


def test_no_taxon_evidence_guard_flags_binomials_families_and_known_names() -> None:
    assert detect_taxon_evidence("conifer needles along the twig") == []
    assert detect_taxon_evidence("a broad leaf with serrated margin") == []
    # Sentence-initial capitalised English must not read as a genus.
    assert detect_taxon_evidence("The crop shows tree bark and trunk, not a needle.") == []
    assert detect_taxon_evidence("This image shows moss.") == []
    assert detect_taxon_evidence("Single flower visible.") == []
    assert "Toxicodendron radicans" in detect_taxon_evidence(
        "this is Toxicodendron radicans"
    )
    assert detect_taxon_evidence("a member of Fagaceae") == ["Fagaceae"]
    assert detect_taxon_evidence("within Fagales") == ["Fagales"]
    assert detect_taxon_evidence("eastern hemlock foliage", ["Tsuga canadensis", "eastern hemlock"]) == [
        "eastern hemlock"
    ]
    # The reason is kept verbatim, but the verdict is invalid and cannot be accepted.
    verdict = parse_verdicts(
        json.dumps({"verdicts": [_verdict(reason="Quercus alba leaf")]}),
        expected=1,
        proposed_classes=["leaf"],
        known_names=["Quercus alba", "white oak"],
    )[0]
    assert verdict["valid"] is False
    assert verdict["invalid_reason"] == "taxon_evidence"
    assert accept_decision(verdict, "leaf")["accept"] is False


def test_invented_boxes_are_flagged_and_cannot_be_accepted() -> None:
    verdict = parse_verdicts(
        json.dumps({"verdicts": [_verdict(bbox=[0, 0, 1, 1])]}),
        expected=1,
        proposed_classes=["leaf"],
    )[0]
    assert verdict["valid"] is False
    assert verdict["invalid_reason"] == "invented_box"
    decision = accept_decision(verdict, "leaf")
    assert decision["accept"] is False
    assert "invented_box" in decision["reasons"]


def test_accept_policy_requires_class_match_resolution_tightness_and_confidence() -> None:
    good = parse_verdicts(
        json.dumps({"verdicts": [_verdict()]}), expected=1, proposed_classes=["leaf"]
    )[0]
    assert accept_decision(good, "leaf")["accept"] is True

    mismatch = parse_verdicts(
        json.dumps({"verdicts": [_verdict(class_="flower")]}),
        expected=1,
        proposed_classes=["leaf"],
    )[0]
    assert accept_decision(mismatch, "leaf")["accept"] is False
    assert "class_mismatch:flower" in accept_decision(mismatch, "leaf")["reasons"]

    not_resolved = parse_verdicts(
        json.dumps({"verdicts": [_verdict(resolved_enough=False)]}),
        expected=1,
        proposed_classes=["leaf"],
    )[0]
    assert accept_decision(not_resolved, "leaf")["accept"] is False

    too_tight = parse_verdicts(
        json.dumps({"verdicts": [_verdict(bbox_tightness="too_tight")]}),
        expected=1,
        proposed_classes=["leaf"],
    )[0]
    assert accept_decision(too_tight, "leaf")["accept"] is False
    assert accept_decision(too_tight, "leaf", accepted_tightness=("keep", "too_tight"))["accept"] is True

    low = parse_verdicts(
        json.dumps({"verdicts": [_verdict(confidence=0.4)]}),
        expected=1,
        proposed_classes=["leaf"],
    )[0]
    assert accept_decision(low, "leaf", min_confidence=0.7)["accept"] is False
    assert accept_decision(low, "leaf", min_confidence=0.3)["accept"] is True


def test_not_target_and_ambiguous_never_accept() -> None:
    for value in ("not_target", "ambiguous"):
        verdict = parse_verdicts(
            json.dumps({"verdicts": [_verdict(class_=value)]}),
            expected=1,
            proposed_classes=["leaf"],
        )[0]
        assert accept_decision(verdict, "leaf")["accept"] is False


def test_crop_pixels_clips_and_pads_real_pixels() -> None:
    left, top, right, bottom = crop_pixels((0.4, 0.4, 0.6, 0.6), 1000, 500, margin=0.25)
    # 0.2*1000=200 wide, 0.2*500=100 tall, pad=50px.
    assert (left, top, right, bottom) == (350, 150, 650, 350)
    assert crop_pixels((0.0, 0.0, 1.0, 1.0), 100, 100, margin=0.5) == (0, 0, 100, 100)
    with pytest.raises(ValueError):
        crop_pixels((0.5, 0.5, 0.5, 0.5), 100, 100)


def test_make_crop_writes_actual_cropped_pixels(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "source.jpg"
    Image.new("RGB", (400, 200), (10, 20, 30)).save(source)
    out = tmp_path / "crop.jpg"
    meta = make_crop(source, (0.25, 0.25, 0.75, 0.75), out, margin=0.0, max_side=900)
    assert out.exists()
    assert meta["source_size"] == [400, 200]
    assert meta["crop_pixels"] == [100, 50, 300, 150]
    assert meta["sha256"] == __import__("hashlib").sha256(out.read_bytes()).hexdigest()


def test_env_file_reader_does_not_require_export_syntax(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('# comment\nOPENROUTER_API_KEY="test-key"\nexport OTHER=1\n')
    values = read_env_file(env)
    assert values["OPENROUTER_API_KEY"] == "test-key"
    assert values["OTHER"] == "1"


def test_human_class_inference_uses_review_notes() -> None:
    assert infer_human_class("leaf", "tp", "") == "leaf"
    assert infer_human_class("leaf", "fp", "yellow bloom in hand - flower not leaf") == "flower"
    assert infer_human_class("flower", "fp", "flower cluster belongs to inflorescence") == "inflorescence"
    assert infer_human_class("needle", "fp", "daylily leaves not needles") == "leaf"
    assert infer_human_class("needle", "fp", "grass") == "not_target"
    assert infer_human_class("leaf", "fp", "box on grass stalk") == "not_target"


def _row(proposal_id: str, cls: str, accept: bool, verdict_class: str, human: bool, pass_number: int = 1):
    return {
        "schema": "arq.anatomy4.verify/v1",
        "proposal_id": proposal_id,
        "photo_id": "1",
        "pass": pass_number,
        "proposed_class": cls,
        "proposal_box_score": 0.3,
        "review_verdict": "tp" if human else "fp",
        "review_notes": "",
        "model": "test-model",
        "prompt_hash": "test",
        "verdict": {
            "verdict_class": verdict_class,
            "resolved_enough": True,
            "bbox_tightness": "keep",
            "reason": "visual",
            "confidence": 0.9,
            "valid": True,
            "invalid_reason": None,
            "invented_box_fields": [],
            "taxon_evidence": [],
        },
        "decision": {"accept": accept, "reasons": []},
        "crop": {},
        "latency_ms": 10.0,
        "usage": {"total_tokens": 10},
    }


def test_report_metrics_and_second_pass_override() -> None:
    class FakeSettings:
        def public(self):
            return {"model": "test-model", "temperature": 0.0}

    review_rows = [
        {"proposal_id": "a", "class_name": "leaf", "verdict": "tp", "notes": ""},
        {"proposal_id": "b", "class_name": "leaf", "verdict": "fp", "notes": "grass"},
        {"proposal_id": "c", "class_name": "needle", "verdict": "tp", "notes": ""},
        {"proposal_id": "d", "class_name": "needle", "verdict": "fp", "notes": "moss"},
    ]
    verdict_rows = [
        _row("a", "leaf", True, "leaf", True),
        _row("b", "leaf", True, "leaf", False),
        _row("c", "needle", False, "not_target", True),
        _row("d", "needle", False, "not_target", False),
        # second pass reverses the low-confidence needle rejection.
        _row("c", "needle", True, "needle", True, pass_number=2),
    ]
    report = build_report(
        review_rows=review_rows,
        verdict_rows=verdict_rows,
        settings=FakeSettings(),
        review_path="review.csv",
        proposals_path="proposals.json",
        crops_dir="crops",
        calls_used=3,
        skipped=[],
        min_confidence=0.7,
        batch_size=2,
    )
    leaf = report["per_class"]["leaf"]
    assert (leaf["tp"], leaf["fp"], leaf["tn"], leaf["fn"]) == (1, 1, 0, 0)
    assert leaf["precision"] == 0.5
    needle = report["per_class"]["needle"]
    assert (needle["tp"], needle["fp"], needle["tn"], needle["fn"]) == (1, 0, 1, 0)
    assert needle["precision"] == 1.0 and needle["recall"] == 1.0
    assert report["second_pass_verdicts"] == 1
    assert report["verified_proposals"] == 4
    assert report["overall"]["accuracy"] == 0.75

def test_policy_profiles_encode_the_measured_tradeoffs() -> None:
    from experiments.anatomy4.verify import POLICY_PROFILES

    assert set(POLICY_PROFILES) == {"strict", "balanced", "triage"}

    # A tight-box, right-class verdict: strict rejects, balanced accepts + flags.
    too_tight = parse_verdicts(
        json.dumps(
            {"verdicts": [_verdict(class_="inflorescence", bbox_tightness="too_tight", confidence=0.0)]}
        ),
        expected=1,
        proposed_classes=["inflorescence"],
    )[0]
    strict = accept_decision(too_tight, "inflorescence", profile="strict")
    balanced = accept_decision(too_tight, "inflorescence", profile="balanced")
    assert strict["accept"] is False
    assert balanced["accept"] is True
    assert balanced["flags"] == ["bbox_too_tight"]

    # triage never accepts a not_target verdict, and never accepts invalid payloads.
    not_target = parse_verdicts(
        json.dumps({"verdicts": [_verdict(class_="not_target")]}),
        expected=1,
        proposed_classes=["leaf"],
    )[0]
    assert accept_decision(not_target, "leaf", profile="triage")["accept"] is False
    assert "verdict_not_target" in accept_decision(
        not_target, "leaf", profile="triage"
    )["reasons"]
    invalid = parse_verdicts(
        json.dumps({"verdicts": [_verdict(reason="Quercus rubra leaf")]}),
        expected=1,
        proposed_classes=["leaf"],
    )[0]
    for profile in POLICY_PROFILES:
        assert accept_decision(invalid, "leaf", profile=profile)["accept"] is False

    with pytest.raises(ValueError):
        accept_decision(balanced, "leaf", profile="nope")


def test_wilson_interval_and_policy_sweep_are_pure() -> None:
    from experiments.anatomy4.verify import policy_sweep, wilson_interval

    low, high = wilson_interval(10, 10)
    assert low is not None and high == 1.0
    assert low < 1.0  # 10/10 is not certainty
    assert wilson_interval(0, 0) == (None, None)

    review_rows = [
        {"proposal_id": "a", "class_name": "leaf", "verdict": "tp", "notes": ""},
        {"proposal_id": "b", "class_name": "leaf", "verdict": "fp", "notes": "grass"},
    ]
    verdict_rows = [
        _row("a", "leaf", True, "leaf", True),
        _row("b", "leaf", False, "not_target", False),
    ]
    sweep = policy_sweep(
        review_rows=review_rows,
        verdict_rows=verdict_rows,
        confidences=(0.0, 0.9),
        tightness_sets=(("keep",),),
    )
    assert len(sweep) == 2
    assert all(entry["n"] == 2 for entry in sweep)
    # policy_sweep must not mutate the input verdicts.
    assert verdict_rows[0]["decision"]["accept"] is True


def test_sample_evenly_per_class_spreads_the_confidence_range() -> None:
    from experiments.anatomy4.verify import sample_evenly_per_class

    proposals = [
        {"proposal_id": f"leaf-{index}", "class_name": "leaf"} for index in range(10)
    ] + [{"proposal_id": "needle-0", "class_name": "needle"}]
    picked = sample_evenly_per_class(proposals, 3)
    leaf_ids = [p["proposal_id"] for p in picked if p["class_name"] == "leaf"]
    assert len(leaf_ids) == 3
    assert leaf_ids[0] == "leaf-0" and leaf_ids[-1] == "leaf-9"
    assert "needle-0" in {p["proposal_id"] for p in picked}


def test_labels_can_be_gated_by_the_verifier(tmp_path: Path) -> None:
    from experiments.anatomy4.labels import (
        accepted_proposals,
        is_verified_accepted,
        load_verification,
        write_yolo_dataset,
    )

    square = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
    manifest = {
        "contract": "anatomy4-v1",
        "images": [
            {"photo_id": "1", "observation_id": "obs-1", "download": {}},
            {"photo_id": "2", "observation_id": "obs-2", "download": {}},
        ],
        "proposals": [
            {
                "proposal_id": "p1",
                "photo_id": "1",
                "observation_id": "obs-1",
                "class_name": "leaf",
                "status": "proposed",
                "polygon": square,
            },
            {
                "proposal_id": "p2",
                "photo_id": "2",
                "observation_id": "obs-2",
                "class_name": "needle",
                "status": "proposed",
                "polygon": square,
            },
        ],
    }
    verdicts_path = tmp_path / "verdicts.jsonl"
    verdicts_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "proposal_id": "p1",
                    "proposed_class": "leaf",
                    "pass": 1,
                    "verdict": {
                        "verdict_class": "leaf",
                        "resolved_enough": True,
                        "bbox_tightness": "keep",
                        "reason": "flat blade",
                        "confidence": 0.9,
                        "valid": True,
                        "invalid_reason": None,
                        "taxon_evidence": [],
                        "invented_box_fields": [],
                    },
                },
                {
                    "proposal_id": "p2",
                    "proposed_class": "needle",
                    "pass": 1,
                    "verdict": {
                        "verdict_class": "not_target",
                        "resolved_enough": False,
                        "bbox_tightness": "keep",
                        "reason": "moss",
                        "confidence": 0.0,
                        "valid": True,
                        "invalid_reason": None,
                        "taxon_evidence": [],
                        "invented_box_fields": [],
                    },
                },
            )
        )
        + "\n"
    )
    verification = load_verification(verdicts_path)
    assert is_verified_accepted(manifest["proposals"][0], verification) is True
    assert is_verified_accepted(manifest["proposals"][1], verification) is False

    machine_only = accepted_proposals(manifest, {}, verification)
    assert [p["proposal_id"] for p in machine_only] == ["p1"]

    # require_both: machine accepts p1, the human sheet rejects it.
    reviews = {"p1": {"verdict": "fp"}}
    assert accepted_proposals(manifest, reviews, verification, require_both=True) == []
    assert [
        p["proposal_id"]
        for p in accepted_proposals(manifest, reviews, verification, require_both=False)
    ] == ["p1"]

    summary = write_yolo_dataset(
        manifest,
        tmp_path / "out",
        reviews={},
        verification=verification,
        include_segmentation=False,
    )
    assert summary["accepted_proposals"] == 1
    assert summary["verification"] is True
    assert (tmp_path / "out" / "accepted.json").exists()
