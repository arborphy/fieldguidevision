"""Tests for the P17 agentic statement-tagging modules.

Covers the Dirr DeVo adapter normalization, the statement compiler's value
resolution + disambiguation, and the agentic scene-resolution loop driven by a
mock Gemma (no GPU required).
"""

from __future__ import annotations

import json

from PIL import Image

from segmentation.agentic_resolver import (
    OUTCOME_ABSTAIN,
    OUTCOME_DISCARDED,
    OUTCOME_RESOLVED,
    AgenticSceneResolver,
)
from segmentation.devo_adapter import DirrDevoAdapter
from segmentation.pipeline import CaptureFrame, RawSegment
from segmentation.statement_compiler import StatementCompiler


# ---------------------------------------------------------------------------
# Dirr adapter
# ---------------------------------------------------------------------------


def test_dirr_adapter_lists_feature_values_with_organs() -> None:
    adapter = DirrDevoAdapter()
    feats = adapter.list_features()
    assert len(feats) > 80  # ~93 curated values
    by_label = {f.value_label: f for f in feats}
    # Curated definitions ride along as grounding references.
    assert by_label["Serrate"].value_definition == "saw-toothed, the teeth pointing forward."
    # Organ mapping: leaf morphology hangs off the leaf blade; habit off whole plant.
    assert by_label["Serrate"].required_organ == "leaf blade"
    assert by_label["Tree"].required_organ == "whole plant"
    assert by_label["Deciduous"].required_organ == "whole plant"


def test_dirr_adapter_resolves_taxa_by_values() -> None:
    adapter = DirrDevoAdapter()
    trees = adapter.resolve_taxa_by_feature_values(["Tree"])
    assert len(trees) > 100
    quercus = adapter.get_taxon("Quercus alba")
    assert quercus is not None
    assert any(fv.value_label == "Tree" for fv in quercus.feature_values)


# ---------------------------------------------------------------------------
# Statement compiler
# ---------------------------------------------------------------------------


def test_compiler_deciduous_tree_prefers_whole_specimen() -> None:
    compiler = StatementCompiler(DirrDevoAdapter())
    plan = compiler.compile(
        "all observations which show a deciduous tree "
        "(highest candidate is one that shows the entire specimen)"
    )
    targets = {(t.feature_name, t.value_label) for t in plan.tag_targets}
    # "deciduous tree" -> phenology + habit; "entire specimen" must NOT match the
    # leaf-margin value "Entire".
    assert ("phenology_types", "Deciduous") in targets
    assert ("growth_habit_types", "Tree") in targets
    assert ("leaf_shape_margins", "Entire") not in targets
    assert plan.prefer_whole_specimen
    # Whole-plant organ is required and gates acceptance.
    required = {o.organ for o in plan.required_organs}
    assert "whole plant" in required


def test_compiler_leaf_statement_uses_leaf_organ() -> None:
    compiler = StatementCompiler(DirrDevoAdapter())
    plan = compiler.compile("observations showing a serrate lobed leaf with cuneate base")
    targets = {(t.feature_name, t.value_label) for t in plan.tag_targets}
    assert ("leaf_shape_margins", "Serrate") in targets
    assert ("leaf_shape_margins", "Lobed") in targets
    assert ("leaf_shape_base", "Cuneate") in targets
    assert {o.organ for o in plan.required_organs} == {"leaf blade"}


def test_compiler_rejects_unmatched_statement() -> None:
    compiler = StatementCompiler(DirrDevoAdapter())
    try:
        compiler.compile("show me something with no vocabulary overlap zzz")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unmatched statement")


# ---------------------------------------------------------------------------
# Agentic resolver (mock Gemma + mock segmenter)
# ---------------------------------------------------------------------------


def _frame() -> CaptureFrame:
    return CaptureFrame(
        observation_id="obs-1",
        image_url="file:///tmp/x.png",
        subject_id="Quercus alba",
        species_name="Quercus alba",
    )


def _image() -> Image.Image:
    return Image.new("RGB", (64, 64), (34, 139, 34))


def _whole_plant_segment() -> tuple[RawSegment, ...]:
    poly = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))  # ~36% area
    return (RawSegment(label="whole plant", polygon=poly, detection_confidence=0.5,
                       mask_quality=0.7, prompt="whole plant", detector_label="whole plant"),)


def _plan() -> object:
    compiler = StatementCompiler(DirrDevoAdapter())
    return compiler.compile("deciduous tree, entire specimen")


def test_resolver_discards_on_negative_cheap_look() -> None:
    def vlm(image, prompt):
        return json.dumps({"action": "discard", "is_target_plausible": False,
                           "rationale": "landscape, no specimen"})

    def segment(image, prompts):
        raise AssertionError("detector must not fire after a discard")

    resolver = AgenticSceneResolver(vlm=vlm, segment=segment)
    res = resolver.resolve(_image(), _plan(), _frame())
    assert res.outcome == OUTCOME_DISCARDED
    assert res.turns_used == 1
    assert not res.feature_tags


def test_resolver_resolves_and_tags() -> None:
    calls = {"ground": 0}

    def vlm(image, prompt):
        if "Decide ONLY whether" in prompt:
            return json.dumps({"action": "ground", "is_target_plausible": True, "rationale": "ok"})
        if "directing an object-detection tool" in prompt:
            calls["ground"] += 1
            return json.dumps({"action": "ground", "rec_phrase": "whole plant", "rationale": "try"})
        return json.dumps({
            "action": "resolve",
            "feature_assessments": [
                {"feature_name": "growth_habit_types", "value_label": "Tree",
                 "confidence": 0.9, "abstain": False, "rationale": "single trunk"},
                {"feature_name": "phenology_types", "value_label": "Deciduous",
                 "confidence": 0.4, "abstain": True, "rationale": "cannot tell season"},
            ],
            "rationale": "resolved",
        })

    resolver = AgenticSceneResolver(vlm=vlm, segment=lambda i, p: _whole_plant_segment())
    res = resolver.resolve(_image(), _plan(), _frame())
    assert res.outcome == OUTCOME_RESOLVED
    # Only the confident, non-abstained tag survives.
    assert [(t.feature_name, t.value_label) for t in res.feature_tags] == [
        ("growth_habit_types", "Tree")
    ]
    assert any(s.label == "whole plant" for s in res.grounded_segments)


def test_resolver_rephrases_after_null_detection() -> None:
    rec_phrases: list[str] = []
    segments_by_call = [(), (), _whole_plant_segment()]  # two nulls, then a hit

    def vlm(image, prompt):
        if "Decide ONLY whether" in prompt:
            return json.dumps({"action": "ground", "is_target_plausible": True, "rationale": "ok"})
        if "directing an object-detection tool" in prompt:
            n = len(rec_phrases)
            phrase = ["whole plant", "the mature tree in the center", "the bare trunk and crown"][min(n, 2)]
            rec_phrases.append(phrase)
            return json.dumps({"action": "rephrase", "rec_phrase": phrase, "rationale": "retry"})
        return json.dumps({"action": "resolve", "feature_assessments": [], "rationale": "done"})

    def segment(image, prompts):
        return segments_by_call[min(len(rec_phrases) - 1, 2)]

    resolver = AgenticSceneResolver(vlm=vlm, segment=segment)
    res = resolver.resolve(_image(), _plan(), _frame())
    assert res.outcome == OUTCOME_RESOLVED
    # Null detections drove rephrasing (null != negative).
    assert len(rec_phrases) == 3
    assert res.turns_used >= 4  # cheap look + 3 ground attempts + assess


def test_resolver_abstains_when_agent_gives_up() -> None:
    def vlm(image, prompt):
        if "Decide ONLY whether" in prompt:
            return json.dumps({"action": "ground", "is_target_plausible": True, "rationale": "ok"})
        if "directing an object-detection tool" in prompt:
            return json.dumps({"action": "abstain", "rec_phrase": None,
                               "rationale": "only a flower is visible; cannot assess habit"})
        raise AssertionError("assess must not run after abstain")

    resolver = AgenticSceneResolver(vlm=vlm, segment=lambda i, p: ())
    res = resolver.resolve(_image(), _plan(), _frame())
    assert res.outcome == OUTCOME_ABSTAIN
    assert "flower" in res.rationale


if __name__ == "__main__":
    import unittest

    unittest.main()
