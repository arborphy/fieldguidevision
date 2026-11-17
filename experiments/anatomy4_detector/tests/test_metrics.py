# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import pytest

from experiments.anatomy4_detector.metrics import (
    PixelBox,
    PixelDetection,
    evaluate,
    match_detections,
)


def box(class_id: int, x0: float, y0: float, x1: float, y1: float) -> PixelBox:
    return PixelBox(class_id, x0, y0, x1, y1)


def det(class_id: int, confidence: float, x0=0.0, y0=0.0, x1=10.0, y1=10.0) -> PixelDetection:
    return PixelDetection(class_id, x0, y0, x1, y1, confidence)


def test_perfect_predictions_score_one() -> None:
    truth = {"a": [box(0, 0, 0, 10, 10)]}
    predictions = {"a": [det(0, 0.9)]}
    result = evaluate(truth, predictions, split="test")
    leaf = result["per_class"]["leaf"]
    assert (leaf["tp"], leaf["fp"], leaf["fn"]) == (1, 0, 0)
    assert leaf["precision"] == 1.0
    assert leaf["recall"] == 1.0
    assert leaf["ap50"] == pytest.approx(1.0)
    assert leaf["ap50_95"] == pytest.approx(1.0)
    assert result["macro"]["ap50"] == pytest.approx(1.0)
    assert result["micro"]["precision"] == 1.0
    # Classes with no test support must be undefined, not zero.
    assert result["per_class"]["needle"]["ap50"] is None
    assert result["per_class"]["needle"]["support"] == 0
    assert "needle" not in result["macro"]["classes_with_support"]
    assert result["confusion_matrix"]["matrix"][0][0] == 1


def test_false_positive_reduces_precision_and_background_cell() -> None:
    truth = {"a": [box(0, 0, 0, 10, 10)]}
    predictions = {"a": [det(0, 0.9), det(0, 0.8, 20, 20, 30, 30)]}
    result = evaluate(truth, predictions)
    leaf = result["per_class"]["leaf"]
    assert (leaf["tp"], leaf["fp"], leaf["fn"]) == (1, 1, 0)
    assert leaf["precision"] == pytest.approx(0.5)
    assert leaf["recall"] == pytest.approx(1.0)
    background_row = result["confusion_matrix"]["labels"].index("background")
    leaf_column = result["confusion_matrix"]["labels"].index("leaf")
    assert result["confusion_matrix"]["matrix"][background_row][leaf_column] == 1


def test_missed_ground_truth_reduces_recall_and_marks_background_column() -> None:
    truth = {"a": [box(0, 0, 0, 10, 10)]}
    result = evaluate(truth, {"a": []})
    leaf = result["per_class"]["leaf"]
    assert (leaf["tp"], leaf["fp"], leaf["fn"]) == (0, 0, 1)
    assert leaf["recall"] == 0.0
    assert leaf["ap50"] == 0.0
    background_column = result["confusion_matrix"]["labels"].index("background")
    leaf_row = result["confusion_matrix"]["labels"].index("leaf")
    assert result["confusion_matrix"]["matrix"][leaf_row][background_column] == 1


def test_wrong_class_is_a_confusion_not_a_match() -> None:
    truth = {"a": [box(0, 0, 0, 10, 10)]}
    predictions = {"a": [det(2, 0.9)]}
    result = evaluate(truth, predictions)
    assert result["per_class"]["leaf"]["tp"] == 0
    assert result["per_class"]["leaf"]["fn"] == 1
    assert result["per_class"]["flower"]["fp"] == 1
    leaf_row = result["confusion_matrix"]["labels"].index("leaf")
    flower_column = result["confusion_matrix"]["labels"].index("flower")
    assert result["confusion_matrix"]["matrix"][leaf_row][flower_column] == 1


def test_ap_rewards_recall_and_ignores_trailing_false_positives() -> None:
    truth = {"a": [box(0, 0, 0, 10, 10), box(0, 20, 20, 30, 30)]}
    one_hit = {"a": [det(0, 0.9), det(0, 0.8, 40, 40, 50, 50)]}
    both_hits_with_trailing_fp = {
        "a": [det(0, 0.9), det(0, 0.8, 20, 20, 30, 30), det(0, 0.7, 40, 40, 50, 50)]
    }
    partial = evaluate(truth, one_hit)["per_class"]["leaf"]
    complete = evaluate(truth, both_hits_with_trailing_fp)["per_class"]["leaf"]
    # COCO 101-point interpolation with the precision envelope: one of two GT
    # found gives AP 51/101; a trailing FP after all GT does not lower AP.
    assert partial["ap50"] == pytest.approx(51 / 101, abs=1e-6)
    assert complete["ap50"] == pytest.approx(1.0)
    # The trailing FP still costs precision at the operating point.
    assert complete["precision"] == pytest.approx(2 / 3)


def test_conf_threshold_filters_operating_point_but_not_ap() -> None:
    truth = {"a": [box(0, 0, 0, 10, 10)]}
    predictions = {"a": [det(0, 0.1)]}
    result = evaluate(truth, predictions, conf_threshold=0.25)
    assert result["per_class"]["leaf"]["tp"] == 0
    assert result["per_class"]["leaf"]["fn"] == 1
    # AP still ranks the low-confidence detection (COCO convention).
    assert result["per_class"]["leaf"]["ap50"] == pytest.approx(1.0)


def test_negative_images_count_false_positives() -> None:
    truth = {"negative": []}
    predictions = {"negative": [det(1, 0.7)]}
    result = evaluate(truth, predictions)
    assert result["counts"]["negative_images"] == 1
    assert result["per_class"]["needle"]["fp"] == 1
    assert result["macro"]["precision"] == 0.0


def test_iou_threshold_controls_matching_but_ap50_is_always_iou_050() -> None:
    truth = {"a": [box(0, 0, 0, 10, 10)]}
    predictions = {"a": [det(0, 0.9, 0, 0, 10, 8)]}  # IoU 0.8
    strict = evaluate(truth, predictions, iou_threshold=0.9)
    loose = evaluate(truth, predictions, iou_threshold=0.5)
    assert strict["per_class"]["leaf"]["tp"] == 0
    assert loose["per_class"]["leaf"]["tp"] == 1
    # AP50 is defined at IoU 0.5 regardless of the operating threshold...
    assert strict["per_class"]["leaf"]["ap50"] == pytest.approx(1.0)
    # ...and the strict AP column reflects the 0.90 operating IoU.
    assert strict["per_class"]["leaf"]["ap_per_iou"]["0.90"] == 0.0
    assert loose["per_class"]["leaf"]["ap50"] == pytest.approx(1.0)


def test_greedy_matching_uses_each_gt_once() -> None:
    truth = [box(0, 0, 0, 10, 10)]
    dets = [det(0, 0.9), det(0, 0.8)]
    match = match_detections(truth, dets, 0.5)
    assert match.det_to_gt[0] == 0
    assert match.det_to_gt[1] is None
    assert match.unmatched_gt == []


def test_tagged_bbox_metrics_call_out_leaf() -> None:
    from experiments.anatomy4_detector.metrics import tagged_bbox_metrics

    truth = {"a": [box(0, 0, 0, 10, 10), box(0, 20, 20, 30, 30)]}
    predictions = {"a": [det(0, 0.9), det(0, 0.8, 20, 20, 30, 30)]}
    result = evaluate(truth, predictions)
    tagged = tagged_bbox_metrics(result)
    assert tagged["source"] == "reviewed_ground_truth"
    assert tagged["leaf"]["recall"] == 1.0
    assert tagged["leaf"]["ap50"] == pytest.approx(1.0)
    assert tagged["leaf"]["support"] == 2
    assert tagged["per_class"]["needle"]["support"] == 0
    assert tagged["per_class"]["needle"]["ap50"] is None


def test_threshold_selection_uses_best_macro_f1_then_closest_default() -> None:
    from experiments.anatomy4_detector.metrics import select_threshold_by_macro_f1

    sweep = {
        "results": {
            "0.10": {"macro": {"f1": 0.5}},
            "0.25": {"macro": {"f1": 0.7}},
            "0.30": {"macro": {"f1": 0.7}},
            "0.50": {"macro": {"f1": 0.6}},
            "0.60": {"macro": {"f1": None}},
        }
    }
    assert select_threshold_by_macro_f1(sweep, default=0.25) == 0.25
    sweep["results"]["0.20"] = {"macro": {"f1": 0.8}}
    assert select_threshold_by_macro_f1(sweep, default=0.25) == 0.20
    assert select_threshold_by_macro_f1({"results": {}}, default=0.25) == 0.25


def test_threshold_parser_always_includes_default() -> None:
    from experiments.anatomy4_detector.cli import _parse_thresholds

    assert _parse_thresholds("0.10,0.50", 0.25) == [0.1, 0.25, 0.5]
    assert _parse_thresholds("", 0.25) == [0.25]
    assert _parse_thresholds("0.25,0.25", 0.25) == [0.25]
    assert _parse_thresholds("1.5,0,0.4", 0.25) == [0.25, 0.4]
