# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Classification metrics, per-class recall, confusion matrix, and abstention.

All functions are pure Python + numpy. No model/DB imports.

Recall semantics
----------------
Two per-class recalls are always reported:

* **all rows** — abstained predictions count as false negatives. This is the
  conservative "if the expert is asked, what does it recover?" number.
* **covered only** — only non-abstained rows count; this measures accuracy on
  the subset the expert chose to answer. It can look much better than
  all-row recall when coverage is low, so both are recorded side by side.

Abstention contract
-------------------
A prediction is **covered** when both:

* ``max(probabilities) >= confidence_threshold``, and
* ``max(probabilities) - second_max(probabilities) >= margin_threshold``

Covered rows use argmax; all others are reported as ``abstain`` (never mapped
to a real class). The threshold is selected on the validation split only.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

ABSTAIN_LABEL = "abstain"


def _as_arrays(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(list(y_true), dtype=np.int64)
    pred = np.asarray(list(y_pred), dtype=np.int64)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    return true, pred


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]
) -> list[list[int]]:
    """Rows = actual, columns = predicted (including an out-of-range guard)."""

    true, pred = _as_arrays(y_true, y_pred)
    n = len(labels)
    matrix = np.zeros((n, n), dtype=np.int64)
    for actual, predicted in zip(true, pred):
        if actual < 0 or actual >= n or predicted < 0 or predicted >= n:
            raise ValueError(f"label index out of range: true={actual} pred={predicted}")
        matrix[actual, predicted] += 1
    return matrix.tolist()


def per_class_prf(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """One-vs-rest precision/recall/F1 for every class (support included)."""

    matrix = np.asarray(confusion_matrix(y_true, y_pred, labels), dtype=np.int64)
    out: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(labels):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def macro_prf(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
    *,
    include_absent: bool = False,
) -> dict[str, Any]:
    """Macro precision/recall/F1.

    By default averages over classes with support > 0 (the honest "present
    classes" macro). ``include_absent=True`` averages over every declared
    class, counting absent classes' zero scores — useful only as a secondary
    view.
    """

    per_class = per_class_prf(y_true, y_pred, labels)
    names = list(labels) if include_absent else [n for n in labels if per_class[n]["support"] > 0]
    if not names:
        return {"precision": None, "recall": None, "f1": None, "n_classes": 0}
    return {
        "precision": float(sum(per_class[n]["precision"] for n in names) / len(names)),
        "recall": float(sum(per_class[n]["recall"] for n in names) / len(names)),
        "f1": float(sum(per_class[n]["f1"] for n in names) / len(names)),
        "n_classes": len(names),
        "classes": names,
    }


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true, pred = _as_arrays(y_true, y_pred)
    if true.size == 0:
        return 0.0
    return float((true == pred).mean())


def top_k_accuracy(
    y_true: Sequence[int], probabilities: Sequence[Sequence[float]], *, k: int = 2
) -> float:
    true = np.asarray(list(y_true), dtype=np.int64)
    probs = np.asarray(list(probabilities), dtype=np.float64)
    if true.size == 0:
        return 0.0
    if probs.ndim != 2 or probs.shape[0] != true.size:
        raise ValueError("probabilities must be [n, classes]")
    k = max(1, min(k, probs.shape[1]))
    top = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([actual in row for actual, row in zip(true, top)]))


def confidence_and_margin(probabilities: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(list(probabilities), dtype=np.float64)
    if probs.size == 0:
        return np.zeros(0), np.zeros(0)
    if probs.ndim != 2:
        raise ValueError("probabilities must be [n, classes]")
    ordered = np.sort(probs, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2] if probs.shape[1] > 1 else ordered[:, -1]
    return confidence, margin


def apply_abstention(
    probabilities: Sequence[Sequence[float]],
    *,
    confidence_threshold: float = 0.0,
    margin_threshold: float = 0.0,
) -> list[int]:
    """Return class indices, or ``-1`` for abstained rows."""

    probs = np.asarray(list(probabilities), dtype=np.float64)
    if probs.size == 0:
        return []
    if probs.ndim != 2:
        raise ValueError("probabilities must be [n, classes]")
    confidence, margin = confidence_and_margin(probs)
    argmax = probs.argmax(axis=1)
    return [
        int(index)
        if conf >= confidence_threshold and marg >= margin_threshold
        else -1
        for index, conf, marg in zip(argmax, confidence, margin)
    ]


def _selective_precision(y_true: np.ndarray, pred: np.ndarray) -> float:
    covered = pred >= 0
    if not covered.any():
        return 0.0
    return float((y_true[covered] == pred[covered]).mean())


def select_abstain_threshold(
    y_true: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    target_precision: float = 0.95,
    min_coverage: float = 0.30,
    grid: Sequence[float] | None = None,
    margin_threshold: float = 0.0,
) -> dict[str, Any]:
    """Pick a confidence threshold on VALIDATION predictions.

    Rule: among thresholds whose selective precision >= ``target_precision``
    AND coverage >= ``min_coverage``, choose the one with the highest
    coverage (lowest threshold). If none qualify, fall back to the threshold
    maximizing selective F1 (then coverage), and state that the target was not
    met. The returned object records the whole curve so the decision is
    auditable.
    """

    true = np.asarray(list(y_true), dtype=np.int64)
    probs = np.asarray(list(probabilities), dtype=np.float64)
    if true.size == 0 or probs.size == 0:
        raise ValueError("cannot select an abstain threshold from empty validation data")
    thresholds = list(grid) if grid is not None else [round(0.05 * i, 2) for i in range(0, 20)]
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        pred = np.asarray(
            apply_abstention(probs, confidence_threshold=threshold, margin_threshold=margin_threshold),
            dtype=np.int64,
        )
        covered = pred >= 0
        coverage = float(covered.mean())
        selective_precision = _selective_precision(true, pred)
        selective_f1 = None
        if covered.any():
            selective_f1 = macro_prf(
                true[covered].tolist(), pred[covered].tolist(), list(labels)
            )["f1"]
        curve.append(
            {
                "confidence_threshold": float(threshold),
                "coverage": coverage,
                "selective_precision": selective_precision,
                "selective_macro_f1": selective_f1,
                "abstained": int((~covered).sum()),
            }
        )

    qualifying = [
        item
        for item in curve
        if item["selective_precision"] >= target_precision and item["coverage"] >= min_coverage
    ]
    if qualifying:
        chosen = max(qualifying, key=lambda item: (item["coverage"], -item["confidence_threshold"]))
        met = True
    else:
        # Fallback: maximise selective macro F1, then coverage. A threshold of
        # 0.0 covers everything, so this is always defined.
        chosen = max(
            curve,
            key=lambda item: (
                item["selective_macro_f1"] if item["selective_macro_f1"] is not None else -1.0,
                item["coverage"],
            ),
        )
        met = False
    chosen = dict(chosen)
    chosen.update(
        {
            "target_precision": target_precision,
            "min_coverage": min_coverage,
            "margin_threshold": margin_threshold,
            "target_met": met,
            "rule": (
                "max coverage s.t. selective precision >= target and coverage >= min; "
                "fallback max selective macro-F1 then coverage"
            ),
            "curve": curve,
        }
    )
    return chosen


def abstention_report(
    y_true: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    confidence_threshold: float,
    margin_threshold: float = 0.0,
) -> dict[str, Any]:
    """Full metrics with abstentions, coverage, and per-class recall."""

    true = np.asarray(list(y_true), dtype=np.int64)
    probs = np.asarray(list(probabilities), dtype=np.float64)
    if true.size != probs.shape[0]:
        raise ValueError("y_true and probabilities length mismatch")
    predictions = np.asarray(
        apply_abstention(
            probs, confidence_threshold=confidence_threshold, margin_threshold=margin_threshold
        ),
        dtype=np.int64,
    )
    covered = predictions >= 0
    confidence, margin = confidence_and_margin(probs)

    # Treat abstain as its own predicted class in the all-rows confusion view.
    all_labels = list(labels) + [ABSTAIN_LABEL]
    mapped = np.where(covered, predictions, len(labels)).tolist()
    matrix = confusion_matrix(true.tolist(), mapped, all_labels)
    per_class_all = per_class_prf(true.tolist(), mapped, all_labels)
    per_class_covered = per_class_prf(
        true[covered].tolist(), predictions[covered].tolist(), labels
    ) if covered.any() else {name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0, "tp": 0, "fp": 0, "fn": 0} for name in labels}

    return {
        "n": int(true.size),
        "coverage": float(covered.mean()) if true.size else 0.0,
        "abstained": int((~covered).sum()),
        "unknown_rate": float((~covered).mean()) if true.size else 0.0,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "overall_accuracy_treating_abstain_as_error": accuracy(true.tolist(), mapped),
        "selective_accuracy": float((true[covered] == predictions[covered]).mean()) if covered.any() else None,
        "selective_macro_prf": macro_prf(
            true[covered].tolist(), predictions[covered].tolist(), labels
        ) if covered.any() else {"precision": None, "recall": None, "f1": None, "n_classes": 0},
        "recall_semantics": {
            "all_rows": "abstained predictions count as false negatives",
            "covered_only": "abstained rows are excluded from the denominator",
        },
        "per_class_recall_all_rows": {
            name: per_class_all[name]["recall"] for name in labels
        },
        "per_class_recall_covered": {
            name: per_class_covered[name]["recall"] for name in labels
        },
        "per_class_recall_covered_only": {
            name: per_class_covered[name]["recall"] for name in labels
        },
        "per_class": per_class_covered,
        "confusion_matrix": {
            "labels": all_labels,
            "rows_actual_columns_predicted": matrix,
        },
        "mean_confidence": float(confidence.mean()) if confidence.size else None,
        "mean_margin": float(margin.mean()) if margin.size else None,
    }


def full_report(
    y_true: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    confidence_threshold: float,
    margin_threshold: float = 0.0,
    top_k: Iterable[int] = (2,),
) -> dict[str, Any]:
    """Threshold-free + abstention metrics for one split."""

    probs = np.asarray(list(probabilities), dtype=np.float64)
    if probs.size == 0:
        raise ValueError("empty probabilities")
    predictions = probs.argmax(axis=1).tolist()
    matrix = confusion_matrix(y_true, predictions, labels)
    report = {
        "n": len(predictions),
        "accuracy": accuracy(y_true, predictions),
        "top_k_accuracy": {str(k): top_k_accuracy(y_true, probs, k=k) for k in top_k},
        "macro_prf_present_classes": macro_prf(y_true, predictions, labels),
        "macro_prf_all_classes": macro_prf(y_true, predictions, labels, include_absent=True),
        "per_class": per_class_prf(y_true, predictions, labels),
        "per_class_recall": {
            name: per_class_prf(y_true, predictions, labels)[name]["recall"] for name in labels
        },
        "confusion_matrix": {
            "labels": list(labels),
            "rows_actual_columns_predicted": matrix,
        },
        "abstention": abstention_report(
            y_true,
            probs,
            labels,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        ),
    }
    return report
