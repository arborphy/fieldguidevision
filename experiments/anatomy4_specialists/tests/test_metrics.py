# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import numpy as np
import pytest

from experiments.anatomy4_specialists.metrics import (
    abstention_report,
    accuracy,
    apply_abstention,
    confidence_and_margin,
    confusion_matrix,
    full_report,
    macro_prf,
    per_class_prf,
    select_abstain_threshold,
    top_k_accuracy,
)

LABELS = ["Simple", "Compound"]


def test_confusion_matrix_rows_actual_columns_predicted() -> None:
    matrix = confusion_matrix([0, 0, 1, 1, 0], [0, 1, 1, 1, 0], LABELS)
    assert matrix == [[2, 1], [0, 2]]


def test_per_class_precision_recall_f1_math() -> None:
    # Simple: tp=2 fp=0 fn=1 -> P=1, R=2/3, F1=0.8
    # Compound: tp=2 fp=1 fn=0 -> P=2/3, R=1, F1=0.8
    per_class = per_class_prf([0, 0, 1, 1, 0], [0, 1, 1, 1, 0], LABELS)
    assert per_class["Simple"]["tp"] == 2
    assert per_class["Simple"]["fp"] == 0
    assert per_class["Simple"]["fn"] == 1
    assert per_class["Simple"]["precision"] == pytest.approx(1.0)
    assert per_class["Simple"]["recall"] == pytest.approx(2 / 3)
    assert per_class["Simple"]["f1"] == pytest.approx(0.8)
    assert per_class["Compound"]["precision"] == pytest.approx(2 / 3)
    assert per_class["Compound"]["recall"] == pytest.approx(1.0)
    macro = macro_prf([0, 0, 1, 1, 0], [0, 1, 1, 1, 0], LABELS)
    assert macro["precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert macro["recall"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert macro["f1"] == pytest.approx(0.8)


def test_absent_class_is_excluded_from_default_macro() -> None:
    per_class = per_class_prf([0, 0], [0, 0], LABELS)
    assert per_class["Compound"]["support"] == 0
    macro = macro_prf([0, 0], [0, 0], LABELS)
    assert macro["n_classes"] == 1
    assert macro["f1"] == pytest.approx(1.0)
    all_macro = macro_prf([0, 0], [0, 0], LABELS, include_absent=True)
    assert all_macro["n_classes"] == 2
    assert all_macro["f1"] == pytest.approx(0.5)


def test_accuracy_and_top_k() -> None:
    probs = [[0.9, 0.1], [0.4, 0.6], [0.7, 0.3]]
    assert accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)
    assert top_k_accuracy([0, 1, 1], probs, k=1) == pytest.approx(2 / 3)
    assert top_k_accuracy([0, 1, 1], probs, k=2) == pytest.approx(1.0)


def test_abstention_and_margin() -> None:
    probs = [[0.9, 0.1], [0.55, 0.45], [0.3, 0.7]]
    confidence, margin = confidence_and_margin(probs)
    assert confidence.tolist() == pytest.approx([0.9, 0.55, 0.7])
    assert margin.tolist() == pytest.approx([0.8, 0.1, 0.4])
    assert apply_abstention(probs, confidence_threshold=0.6) == [0, -1, 1]
    assert apply_abstention(probs, confidence_threshold=0.0, margin_threshold=0.2) == [0, -1, 1]


def test_abstain_threshold_selection_maximizes_coverage_at_precision() -> None:
    y_true = [0, 0, 1, 1, 0, 1]
    # 0.60 covers rows 0,1,3,4,5 (all correct); row 2 (conf 0.55) abstains,
    # so precision is 1.0 with coverage 5/6. 0.50 also covers the wrong row.
    # 0.80 abstains row 5 too (coverage 4/6).
    probs = [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.55, 0.45],
        [0.05, 0.95],
        [0.7, 0.3],
        [0.4, 0.6],
    ]
    selection = select_abstain_threshold(
        y_true,
        probs,
        LABELS,
        target_precision=1.0,
        min_coverage=0.25,
        grid=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.96],
    )
    assert selection["target_met"] is True
    assert selection["confidence_threshold"] == 0.6
    assert selection["coverage"] == pytest.approx(5 / 6)


def test_abstain_threshold_falls_back_when_target_unmet() -> None:
    y_true = [0, 1]
    probs = [[0.6, 0.4], [0.55, 0.45]]
    selection = select_abstain_threshold(
        y_true, probs, LABELS, target_precision=0.99, min_coverage=0.9, grid=[0.0, 0.5]
    )
    assert selection["target_met"] is False
    assert "fallback" in selection["rule"]


def test_abstention_report_treats_abstain_as_unknown() -> None:
    y_true = [0, 0, 1, 1]
    probs = [[0.9, 0.1], [0.4, 0.6], [0.55, 0.45], [0.1, 0.9]]
    report = abstention_report(y_true, probs, LABELS, confidence_threshold=0.56)
    # Row 1 is wrong but covered; row 2 abstains.
    assert report["coverage"] == pytest.approx(0.75)
    assert report["abstained"] == 1
    assert report["unknown_rate"] == pytest.approx(0.25)
    assert report["confusion_matrix"]["labels"] == LABELS + ["abstain"]
    assert report["confusion_matrix"]["rows_actual_columns_predicted"][1][2] == 1
    assert report["selective_accuracy"] == pytest.approx(2 / 3)
    assert report["per_class_recall_all_rows"]["Compound"] == pytest.approx(0.5)


def test_full_report_shape_and_confusion_labels() -> None:
    probs = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    report = full_report([0, 1], probs, LABELS, confidence_threshold=0.5)
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["confusion_matrix"]["labels"] == LABELS
    assert report["confusion_matrix"]["rows_actual_columns_predicted"] == [[1, 0], [0, 1]]
    assert report["per_class_recall"] == {"Simple": 1.0, "Compound": 1.0}
    assert report["abstention"]["coverage"] == pytest.approx(1.0)


def test_empty_inputs_are_defined() -> None:
    assert accuracy([], []) == 0.0
    assert top_k_accuracy([], [], k=2) == 0.0
    assert apply_abstention([], confidence_threshold=0.5) == []
    with pytest.raises(ValueError):
        select_abstain_threshold([], [], LABELS)
