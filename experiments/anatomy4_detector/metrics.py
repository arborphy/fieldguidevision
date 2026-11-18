# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Real detection metrics from held-out predictions (no placeholders).

Boxes are supplied in **pixel** coordinates so IoU is aspect-correct (normalized
IoU is wrong whenever images have different aspect ratios). Matching is greedy
by descending confidence, each ground-truth box matched at most once.

Reported:

* per-class precision / recall / F1 at an operating point (conf, IoU);
* per-class AP50 and AP50-95 (COCO 101-point interpolation);
* per-class support (ground-truth box count);
* a confusion matrix (class-agnostic IoU matching; wrong class is a confusion);
* micro and macro scores. AP for a class with zero support is ``None``, never 0.
  Macro precision/F1 average only classes where the metric is defined; macro
  recall/AP average only classes with support.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import CLASS_NAMES, CONTRACT, iou_xyxy

DEFAULT_AP_THRESHOLDS: tuple[float, ...] = tuple(
    round(0.5 + 0.05 * index, 2) for index in range(10)
)
BACKGROUND = "background"


@dataclass(frozen=True)
class PixelBox:
    class_id: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class PixelDetection:
    class_id: int
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class MatchResult:
    det_to_gt: list[int | None]
    det_iou: list[float]
    gt_matched: list[int | None]

    @property
    def unmatched_gt(self) -> list[int]:
        return [index for index, value in enumerate(self.gt_matched) if value is None]


def match_detections(
    ground_truth: Sequence[PixelBox],
    detections: Sequence[PixelDetection],
    iou_threshold: float,
    *,
    class_aware: bool = True,
) -> MatchResult:
    """Greedy one-to-one matching of detections to GT boxes."""

    order = sorted(
        range(len(detections)),
        key=lambda index: (-detections[index].confidence, index),
    )
    gt_taken = [False] * len(ground_truth)
    det_to_gt: list[int | None] = [None] * len(detections)
    det_iou: list[float] = [0.0] * len(detections)
    gt_matched: list[int | None] = [None] * len(ground_truth)

    for det_index in order:
        detection = detections[det_index]
        best_index: int | None = None
        best_iou = 0.0
        for gt_index, gt_box in enumerate(ground_truth):
            if gt_taken[gt_index]:
                continue
            if class_aware and gt_box.class_id != detection.class_id:
                continue
            overlap = iou_xyxy(detection.xyxy, gt_box.xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index
        if best_index is not None and best_iou >= iou_threshold:
            gt_taken[best_index] = True
            det_to_gt[det_index] = best_index
            det_iou[det_index] = best_iou
            gt_matched[best_index] = det_index
    return MatchResult(det_to_gt=det_to_gt, det_iou=det_iou, gt_matched=gt_matched)


def _interpolated_ap(
    true_positives: Sequence[int], confidences: Sequence[float], n_gt: int
) -> float | None:
    if n_gt <= 0:
        return None
    if not true_positives:
        return 0.0
    order = sorted(
        range(len(true_positives)), key=lambda index: (-confidences[index], index)
    )
    tp = 0
    fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for index in order:
        if true_positives[index]:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    envelope = list(precisions)
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    total = 0.0
    for step in range(101):
        target = step / 100.0
        position = bisect_left(recalls, target)
        total += envelope[position] if position < len(envelope) else 0.0
    return total / 101.0


def _class_ap(
    ground_truth: Mapping[str, Sequence[PixelBox]],
    detections: Mapping[str, Sequence[PixelDetection]],
    class_id: int,
    iou_threshold: float,
) -> float | None:
    n_gt = sum(1 for boxes in ground_truth.values() for box in boxes if box.class_id == class_id)
    if n_gt == 0:
        return None
    flags: list[int] = []
    scores: list[float] = []
    for image in ground_truth:
        gt = [box for box in ground_truth[image] if box.class_id == class_id]
        dets = [det for det in detections.get(image, []) if det.class_id == class_id]
        if not dets:
            continue
        match = match_detections(gt, dets, iou_threshold, class_aware=True)
        for index, detection in enumerate(dets):
            flags.append(1 if match.det_to_gt[index] is not None else 0)
            scores.append(detection.confidence)
    return _interpolated_ap(flags, scores, n_gt)


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _ratio(numerator: float, denominator: float) -> float | None:
    return (numerator / denominator) if denominator else None


def evaluate(
    ground_truth: Mapping[str, Sequence[PixelBox]],
    detections: Mapping[str, Sequence[PixelDetection]],
    *,
    split: str | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    ap_thresholds: Sequence[float] = DEFAULT_AP_THRESHOLDS,
    max_detections: int | None = None,
    classes: Sequence[str] = CLASS_NAMES,
    contract: str = CONTRACT,
) -> dict:
    """Evaluate all images, including images with no GT (their boxes are FPs)."""

    images = sorted(set(ground_truth) | set(detections))
    per_class: dict[str, dict] = {}
    for class_id, name in enumerate(classes):
        support = sum(
            1
            for boxes in ground_truth.values()
            for box in boxes
            if box.class_id == class_id
        )
        true_positives = 0
        false_positives = 0
        operating_detections = 0
        for image in images:
            gt = [box for box in ground_truth.get(image, []) if box.class_id == class_id]
            dets = [
                det
                for det in detections.get(image, [])
                if det.class_id == class_id and det.confidence >= conf_threshold
            ]
            operating_detections += len(dets)
            if not dets:
                continue
            match = match_detections(gt, dets, iou_threshold, class_aware=True)
            true_positives += sum(1 for value in match.det_to_gt if value is not None)
            false_positives += sum(1 for value in match.det_to_gt if value is None)
        false_negatives = support - true_positives
        precision = _ratio(true_positives, true_positives + false_positives)
        recall = _ratio(true_positives, support)
        f1 = (
            (2 * precision * recall / (precision + recall))
            if precision is not None and recall is not None and (precision + recall)
            else None
        )
        ap_by_iou = {
            f"{threshold:.2f}": _class_ap(
                ground_truth, detections, class_id, threshold
            )
            for threshold in ap_thresholds
        }
        per_class[name] = {
            "class_id": class_id,
            "support": support,
            "predictions": operating_detections,
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap_by_iou.get("0.50"),
            "ap50_95": _mean(ap_by_iou[value] for value in ap_by_iou),
            "ap_per_iou": ap_by_iou,
        }

    total_tp = sum(item["tp"] for item in per_class.values())
    total_fp = sum(item["fp"] for item in per_class.values())
    total_fn = sum(item["fn"] for item in per_class.values())
    micro_precision = _ratio(total_tp, total_tp + total_fp)
    micro_recall = _ratio(total_tp, total_tp + total_fn)
    micro_f1 = (
        (2 * micro_precision * micro_recall / (micro_precision + micro_recall))
        if micro_precision is not None
        and micro_recall is not None
        and (micro_precision + micro_recall)
        else None
    )

    supported = [name for name, item in per_class.items() if item["support"] > 0]
    precision_defined = [
        name for name, item in per_class.items() if item["precision"] is not None
    ]
    f1_defined = [name for name, item in per_class.items() if item["f1"] is not None]
    macro = {
        "classes_with_support": supported,
        "classes_without_support": [
            name for name in classes if name not in supported
        ],
        "classes_with_precision": precision_defined,
        "precision": _mean(per_class[name]["precision"] for name in precision_defined),
        "recall": _mean(per_class[name]["recall"] for name in supported),
        "f1": _mean(per_class[name]["f1"] for name in f1_defined),
        "ap50": _mean(per_class[name]["ap50"] for name in supported),
        "ap50_95": _mean(per_class[name]["ap50_95"] for name in supported),
    }

    confusion = confusion_matrix(
        ground_truth,
        detections,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes,
    )

    negative_images = sum(
        1 for image in images if not ground_truth.get(image)
    )
    gt_boxes = sum(len(boxes) for boxes in ground_truth.values())
    total_detections = sum(len(boxes) for boxes in detections.values())
    return {
        "schema": "arq.anatomy4.metrics/v1",
        "contract": contract,
        "class_names": list(classes),
        "split": split,
        "operating_point": {
            "conf": conf_threshold,
            "iou": iou_threshold,
            "max_detections": max_detections,
            "ap_iou_thresholds": list(ap_thresholds),
        },
        "per_class": per_class,
        "macro": macro,
        "micro": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "confusion_matrix": confusion,
        "counts": {
            "images": len(images),
            "negative_images": negative_images,
            "gt_boxes": gt_boxes,
            "detections": total_detections,
            "detections_at_operating_point": sum(
                item["predictions"] for item in per_class.values()
            ),
        },
    }


def confusion_matrix(
    ground_truth: Mapping[str, Sequence[PixelBox]],
    detections: Mapping[str, Sequence[PixelDetection]],
    *,
    conf_threshold: float,
    iou_threshold: float,
    classes: Sequence[str] = CLASS_NAMES,
) -> dict:
    """Class-agnostic IoU matching so a wrong-class box is a confusion."""

    labels = list(classes) + [BACKGROUND]
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for image in sorted(set(ground_truth) | set(detections)):
        gt = list(ground_truth.get(image, []))
        dets = [
            det
            for det in detections.get(image, [])
            if det.confidence >= conf_threshold
        ]
        match = match_detections(gt, dets, iou_threshold, class_aware=False)
        for det_index, gt_index in enumerate(match.det_to_gt):
            predicted = classes[dets[det_index].class_id]
            if gt_index is None:
                matrix[index[BACKGROUND]][index[predicted]] += 1
            else:
                truth = classes[gt[gt_index].class_id]
                matrix[index[truth]][index[predicted]] += 1
        for gt_index, det_index in enumerate(match.gt_matched):
            if det_index is None:
                truth = classes[gt[gt_index].class_id]
                matrix[index[truth]][index[BACKGROUND]] += 1
    normalized: list[list[float | None]] = []
    for row in matrix:
        total = sum(row)
        normalized.append([(value / total) if total else None for value in row])
    return {
        "labels": labels,
        "matrix": matrix,
        "normalized": normalized,
        "rows": "truth",
        "columns": "prediction",
        "iou": iou_threshold,
        "conf": conf_threshold,
        "matching": "class-agnostic",
    }


def tagged_bbox_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Per-class metrics over the reviewed/tagged GT boxes, leaf called out.

    "Tagged bbox" means curator/reviewer-accepted ground-truth boxes (the blind
    benchmark and reviewed sources), not detector proposals. The leaf block is
    duplicated at the top level so downstream consumers do not have to dig.
    """

    per_class: dict[str, dict[str, Any]] = {}
    for name in CLASS_NAMES:
        item = (result.get("per_class") or {}).get(name) or {}
        per_class[name] = {
            "precision": item.get("precision"),
            "recall": item.get("recall"),
            "f1": item.get("f1"),
            "ap50": item.get("ap50"),
            "ap50_95": item.get("ap50_95"),
            "support": item.get("support"),
            "tp": item.get("tp"),
            "fp": item.get("fp"),
            "fn": item.get("fn"),
        }
    return {
        "source": "reviewed_ground_truth",
        "description": (
            "precision/recall/AP over curator-reviewed GT boxes at the reported "
            "operating point; support is the tagged GT box count"
        ),
        "per_class": per_class,
        "leaf": per_class["leaf"],
    }


def select_threshold_by_macro_f1(
    sweep: Mapping[str, Any],
    *,
    default: float = 0.25,
) -> float:
    """Pick the confidence threshold with the best macro F1 (tie -> closest default)."""

    results = sweep.get("results") or {}
    best: tuple[float, float, float] | None = None
    best_threshold = float(default)
    for key, entry in results.items():
        try:
            threshold = float(key)
        except (TypeError, ValueError):
            continue
        f1 = ((entry.get("macro") or {}).get("f1"))
        if f1 is None:
            continue
        score = (float(f1), -abs(threshold - float(default)), threshold)
        if best is None or score > best:
            best = score
            best_threshold = threshold
    return best_threshold


def render_metrics_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"# anatomy4 metrics — {result.get('split') or 'all'}",
        "",
        f"operating point: conf {result['operating_point']['conf']:.2f}, "
        f"IoU {result['operating_point']['iou']:.2f}",
        "",
        "| class | support | pred | TP | FP | FN | P | R | F1 | AP50 | AP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASS_NAMES:
        item = result["per_class"][name]
        lines.append(
            f"| {name} | {item['support']} | {item['predictions']} | {item['tp']} | "
            f"{item['fp']} | {item['fn']} | {_fmt(item['precision'])} | "
            f"{_fmt(item['recall'])} | {_fmt(item['f1'])} | {_fmt(item['ap50'], 3)} | "
            f"{_fmt(item['ap50_95'], 3)} |"
        )
    macro = result["macro"]
    lines += [
        "",
        f"macro over {len(macro['classes_with_support'])} class(es): "
        f"P {_fmt(macro['precision'])} · R {_fmt(macro['recall'])} · "
        f"F1 {_fmt(macro['f1'])} · AP50 {_fmt(macro['ap50'], 3)} · "
        f"AP50-95 {_fmt(macro['ap50_95'], 3)}",
        "",
        f"micro: P {_fmt(result['micro']['precision'])} · "
        f"R {_fmt(result['micro']['recall'])} · F1 {_fmt(result['micro']['f1'])}",
        "",
        f"images {result['counts']['images']} "
        f"({result['counts']['negative_images']} negative) · "
        f"GT {result['counts']['gt_boxes']} · "
        f"detections {result['counts']['detections']}",
    ]
    if macro["classes_without_support"]:
        lines += [
            "",
            "Classes without test support (AP undefined, excluded from macro): "
            + ", ".join(macro["classes_without_support"]),
        ]
    lines += ["", "## Confusion matrix (truth × prediction)", ""]
    labels = result["confusion_matrix"]["labels"]
    lines.append("| truth \\ pred | " + " | ".join(labels) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in labels) + " |")
    for name, row in zip(labels, result["confusion_matrix"]["matrix"]):
        lines.append(f"| {name} | " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"
