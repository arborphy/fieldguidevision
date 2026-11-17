# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Evaluator utilities for the 5-class organ-gate v2 detector.

v1's evaluator assumed exactly one ground-truth box per image.  v2 adds
multi-leaf composite scenes, so this module supports an arbitrary number of
ground-truth boxes per row and performs greedy confidence-ordered matching
against all of them.

Nothing here trains or mutates data.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

CLASS_NAMES: tuple[str, ...] = ("leaf", "bud", "flower", "twig", "branch")
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
NONLEAF_CLASSES: tuple[str, ...] = ("bud", "flower", "twig", "branch")

IOU_MATCH = 0.5
AP_IOU = 0.5
CONF_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        uid = row["uid"]
        if uid in seen:
            raise ValueError(f"duplicate uid {uid!r} in {path}:{lineno}")
        seen.add(uid)
        boxes = row.get("boxes")
        if not boxes:
            raise ValueError(f"no boxes for {uid}")
        for box in boxes:
            if box["class_name"] not in CLASS_IDS:
                raise ValueError(f"unknown class {box['class_name']!r} for {uid}")
            if len(box["bbox_xyxy"]) != 4:
                raise ValueError(f"bad bbox for {uid}: {box['bbox_xyxy']}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def make_record(row: dict[str, Any], preds: Iterable[dict[str, Any]]) -> dict[str, Any]:
    kept = []
    for p in preds:
        x1, y1, x2, y2 = (float(v) for v in p["bbox_xyxy"])
        kept.append(
            {
                "c": int(p["class_id"]),
                "s": round(float(p["confidence"]), 6),
                "b": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )
    kept.sort(key=lambda d: d["s"], reverse=True)
    return {
        "uid": row["uid"],
        "split": row["split"],
        "source": row["source"],
        "domain": row["domain"],
        "weak_box": bool(row.get("weak_box", False)),
        "box_kind": row.get("box_kind"),
        "gt_boxes": [
            {
                "c": CLASS_IDS[b["class_name"]],
                "b": [round(float(v), 4) for v in b["bbox_xyxy"]],
                "weak": bool(b.get("weak_box", False)),
                "kind": b.get("box_kind"),
            }
            for b in row["boxes"]
        ],
        "preds": kept,
    }


def _match_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Greedy confidence-ordered matching of predictions to unmatched GT boxes."""
    npos_by_class: dict[int, int] = defaultdict(int)
    flags: list[tuple[float, bool, int]] = []
    cross_events: list[tuple[float, str, str]] = []

    for rec in records:
        gts = rec["gt_boxes"]
        for gt in gts:
            npos_by_class[gt["c"]] += 1
        used = [False] * len(gts)
        for p in rec["preds"]:  # already confidence-sorted
            best_j = -1
            best_iou = 0.0
            for j, gt in enumerate(gts):
                if used[j] or gt["c"] != p["c"]:
                    continue
                iou = iou_xyxy(p["b"], gt["b"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            is_tp = best_j >= 0 and best_iou >= IOU_MATCH
            flags.append((p["s"], bool(is_tp), p["c"]))
            if is_tp:
                used[best_j] = True
                continue
            for gt in gts:
                if gt["c"] != p["c"] and iou_xyxy(p["b"], gt["b"]) >= IOU_MATCH:
                    cross_events.append((p["s"], CLASS_NAMES[gt["c"]], CLASS_NAMES[p["c"]]))
                    break

    return {"npos": npos_by_class, "flags": flags, "cross_events": cross_events}


def prf_from_flags(matched: dict[str, Any], conf: float) -> dict[str, Any]:
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    for score, hit, class_id in matched["flags"]:
        if score < conf:
            continue
        if hit:
            tp[class_id] += 1
        else:
            fp[class_id] += 1
    fn: dict[int, int] = defaultdict(int)
    for class_id, n in matched["npos"].items():
        fn[class_id] = max(0, n - tp.get(class_id, 0))
    cross: dict[str, int] = defaultdict(int)
    for score, gt_name, pred_name in matched["cross_events"]:
        if score >= conf:
            cross[f"{gt_name}->{pred_name}"] += 1
    return {"tp": tp, "fp": fp, "fn": fn, "npos": matched["npos"], "cross_class": cross}


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_gt": tp + fn,
    }


def average_precision(
    flags: Sequence[tuple[float, bool, int]], class_id: int, npos: int
) -> float | None:
    if npos <= 0:
        return None
    cls = [(s, hit) for s, hit, c in flags if c == class_id]
    if not cls:
        return 0.0
    cls.sort(key=lambda t: t[0], reverse=True)
    tp = fp = 0
    points: list[tuple[float, float]] = []
    for _, hit in cls:
        if hit:
            tp += 1
        else:
            fp += 1
        points.append((tp / npos, tp / (tp + fp)))
    envelope: list[tuple[float, float]] = []
    best = 0.0
    for recall, precision in reversed(points):
        best = max(best, precision)
        envelope.append((recall, best))
    envelope.reverse()
    ap = 0.0
    prev_r = 0.0
    for recall, precision in envelope:
        ap += (recall - prev_r) * precision
        prev_r = recall
    if prev_r > 1.0 + 1e-9 or ap > 1.0 + 1e-9:
        raise AssertionError(f"AP invariant violated: recall={prev_r} ap={ap} npos={npos}")
    return float(ap)


def _class_block(matched: dict[str, Any], class_id: int, conf: float) -> dict[str, Any]:
    prf = prf_from_flags(matched, conf)
    block = _prf(
        prf["tp"].get(class_id, 0),
        prf["fp"].get(class_id, 0),
        prf["fn"].get(class_id, 0),
    )
    block["ap50"] = average_precision(
        matched["flags"], class_id, matched["npos"].get(class_id, 0)
    )
    if block["ap50"] is not None:
        block["ap50"] = round(block["ap50"], 4)
    block["class"] = CLASS_NAMES[class_id]
    return block


def _subset(records: Sequence[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    return [
        rec
        for rec in records
        if all(rec.get(k) == v for k, v in criteria.items())
    ]


def evaluate(
    records: Sequence[dict[str, Any]],
    conf: float,
    *,
    label: str = "test",
) -> dict[str, Any]:
    matched = _match_records(records)
    per_class = {name: _class_block(matched, i, conf) for i, name in enumerate(CLASS_NAMES)}
    totals_prf = prf_from_flags(matched, conf)
    totals = _prf(
        sum(totals_prf["tp"].values()),
        sum(totals_prf["fp"].values()),
        sum(totals_prf["fn"].values()),
    )
    present = [n for n in CLASS_NAMES if per_class[n]["n_gt"] > 0]
    macro_f1 = sum(per_class[n]["f1"] for n in present) / len(present) if present else 0.0

    nonleaf = [n for n in NONLEAF_CLASSES if per_class[n]["n_gt"] > 0]
    nonleaf_recall = (
        sum(per_class[n]["recall"] for n in nonleaf) / len(nonleaf) if nonleaf else None
    )

    ap_by_class = {n: per_class[n]["ap50"] for n in CLASS_NAMES}
    ap_values = [v for v in ap_by_class.values() if v is not None]
    map50 = sum(ap_values) / len(ap_values) if ap_values else None

    leaf_field = _subset(records, source="leafsnap", domain="field")
    leaf_match = _match_records(leaf_field)
    leaf_block = _class_block(leaf_match, CLASS_IDS["leaf"], conf)

    bio = _subset(records, source="bioimages")
    bio_match = _match_records(bio)
    bio_classes = {
        name: _class_block(bio_match, i, conf)
        for i, name in enumerate(CLASS_NAMES)
        if bio_match["npos"].get(i, 0) > 0
    }
    bio_recalls = [b["recall"] for b in bio_classes.values()]
    bio_aps = [b["ap50"] for b in bio_classes.values() if b["ap50"] is not None]

    return {
        "label": label,
        "confidence_threshold": conf,
        "iou_match": IOU_MATCH,
        "n_images": len(records),
        "n_gt_boxes": sum(len(r["gt_boxes"]) for r in records),
        "overall": totals,
        "macro_f1_present_classes": round(macro_f1, 4),
        "map50_iou0.5": round(map50, 4) if map50 is not None else None,
        "per_class": per_class,
        "nonleaf_macro_recall": round(nonleaf_recall, 4)
        if nonleaf_recall is not None
        else None,
        "sources": {
            "leafsnap_field_leaf": {
                **leaf_block,
                "n_images": len(leaf_field),
                "box_kind": "segmentation-derived tight boxes (real localization)",
            },
            "bioimages_weak_box": {
                "n_images": len(bio),
                "box_kind": "near-full-frame weak labels (NOT localization quality)",
                "per_class": bio_classes,
                "macro_recall": round(sum(bio_recalls) / len(bio_recalls), 4)
                if bio_recalls
                else None,
                "map50_iou0.5": round(sum(bio_aps) / len(bio_aps), 4) if bio_aps else None,
            },
        },
        "cross_class_matches": dict(sorted(totals_prf["cross_class"].items())),
        "ap_note": "AP is threshold-free over all saved predictions (conf floor 0.001).",
    }


def prf_curve(records: Sequence[dict[str, Any]], conf: float) -> dict[str, Any]:
    matched = _match_records(records)
    prf = prf_from_flags(matched, conf)
    out = {}
    for i, name in enumerate(CLASS_NAMES):
        if matched["npos"].get(i, 0) == 0:
            continue
        out[name] = _prf(prf["tp"].get(i, 0), prf["fp"].get(i, 0), prf["fn"].get(i, 0))
    return out


def select_confidence(
    val_records: Sequence[dict[str, Any]], grid: Sequence[float] = CONF_GRID
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    curve = []
    for conf in grid:
        per_class = prf_curve(val_records, conf)
        if not per_class:
            continue
        macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
        curve.append({"confidence": conf, "macro_f1": round(macro_f1, 4)})
        candidate = {
            "confidence": conf,
            "macro_f1": round(macro_f1, 4),
            "per_class": per_class,
        }
        if best is None or candidate["macro_f1"] > best["macro_f1"]:
            best = candidate
        elif candidate["macro_f1"] == best["macro_f1"] and conf < best["confidence"]:
            best = candidate
    if best is None:
        raise ValueError("no validation records with ground truth")
    best["rule"] = "max unweighted mean F1 across validation classes; ties -> lower confidence"
    best["grid"] = list(grid)
    best["curve"] = curve
    return best


def _rec(
    uid: str,
    gt_classes: Sequence[str],
    gt_boxes: Sequence[Sequence[float]],
    preds: Sequence[tuple[int, float, Sequence[float]]],
    source: str = "leafsnap",
    domain: str = "field",
    split: str = "test",
    weak: bool = False,
) -> dict[str, Any]:
    return {
        "uid": uid,
        "split": split,
        "source": source,
        "domain": domain,
        "weak_box": weak,
        "box_kind": "weak" if weak else "mask",
        "gt_boxes": [
            {"c": CLASS_IDS[c], "b": list(b), "weak": weak, "kind": "t"}
            for c, b in zip(gt_classes, gt_boxes)
        ],
        "preds": sorted(
            ({"c": c, "s": s, "b": list(b)} for c, s, b in preds),
            key=lambda d: d["s"],
            reverse=True,
        ),
    }


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _self_test() -> None:
    _check(_approx(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]), 1.0), "identical IoU")
    _check(_approx(iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0.0), "disjoint IoU")
    _check(_approx(iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3), "half-overlap IoU")

    perfect = [
        _rec(f"a{i}", ["leaf"], [[10, 10, 50, 50]], [(0, 0.9, [10, 10, 50, 50])])
        for i in range(4)
    ]
    m = evaluate(perfect, 0.25)
    _check(m["per_class"]["leaf"]["precision"] == 1.0, "perfect leaf P")
    _check(m["per_class"]["leaf"]["recall"] == 1.0, "perfect leaf R")
    _check(m["per_class"]["leaf"]["ap50"] == 1.0, "perfect leaf AP")
    _check(m["sources"]["leafsnap_field_leaf"]["recall"] == 1.0, "source leaf recall")

    multi = _rec(
        "multi",
        ["leaf", "leaf", "leaf"],
        [[0, 0, 20, 20], [30, 0, 50, 20], [60, 0, 80, 20]],
        [
            (0, 0.9, [0, 0, 20, 20]),
            (0, 0.8, [30, 0, 50, 20]),
            (0, 0.7, [60, 0, 80, 20]),
        ],
    )
    m = evaluate([multi], 0.25)
    _check(m["per_class"]["leaf"]["tp"] == 3, "three leaves matched")
    _check(m["per_class"]["leaf"]["recall"] == 1.0, "three-leaf recall")

    dup_gt = _rec(
        "dupgt",
        ["leaf", "leaf"],
        [[0, 0, 10, 10], [10, 0, 20, 10]],
        [(0, 0.9, [0, 0, 10, 10]), (0, 0.8, [0, 0, 10, 10])],
    )
    m = evaluate([dup_gt], 0.25)
    _check(m["per_class"]["leaf"]["tp"] == 1, "duplicate box cannot match both GTs")
    _check(m["per_class"]["leaf"]["fn"] == 1, "unmatched GT is FN")

    recs = [
        _rec("b0", ["leaf"], [[10, 10, 50, 50]], [(0, 0.10, [10, 10, 50, 50])]),
        _rec("b1", ["leaf"], [[10, 10, 50, 50]], [(4, 0.80, [10, 10, 50, 50])]),
    ]
    m = evaluate(recs, 0.25)
    _check(m["per_class"]["leaf"]["recall"] == 0.0, "suppressed + wrong class -> no recall")
    _check(m["per_class"]["leaf"]["fn"] == 2, "two leaf FN")
    _check(m["per_class"]["branch"]["fp"] == 1, "wrong class FP")
    _check(m["cross_class_matches"] == {"leaf->branch": 1}, "cross-class diagnostic")

    low = [_rec("d0", ["leaf"], [[0, 0, 100, 100]], [(0, 0.9, [0, 0, 40, 100])])]
    _check(evaluate(low, 0.25)["per_class"]["leaf"]["tp"] == 0, "IoU<0.5 rejected")

    ladder = [
        _rec("e0", ["leaf"], [[0, 0, 10, 10]], [(0, 0.9, [0, 0, 10, 10])]),
        _rec("e1", ["leaf"], [[0, 0, 10, 10]], []),
        _rec("e2", ["leaf"], [[0, 0, 10, 10]], [(0, 0.7, [0, 0, 10, 10])]),
        _rec("e3", ["leaf"], [[100, 100, 110, 110]], [(0, 0.8, [0, 0, 10, 10])]),
    ]
    ap = evaluate(ladder, 0.05)["per_class"]["leaf"]["ap50"]
    _check(_approx(ap, 0.25 * 1.0 + 0.25 * (2 / 3), 1e-3), f"AP ladder got {ap}")

    val = [
        _rec("v0", ["leaf"], [[0, 0, 10, 10]], [(0, 0.9, [0, 0, 10, 10])], split="val"),
        _rec("v1", ["leaf"], [[0, 0, 10, 10]], [(0, 0.2, [0, 0, 10, 10])], split="val"),
        _rec(
            "v2",
            ["bud"],
            [[0, 0, 10, 10]],
            [(1, 0.8, [0, 0, 10, 10])],
            split="val",
            source="bioimages",
            domain="bioimages",
            weak=True,
        ),
    ]
    sel = select_confidence(val, (0.1, 0.25, 0.85))
    _check(sel["confidence"] in (0.1, 0.25, 0.85), "selection returns a grid point")
    _check(sel["per_class"]["leaf"]["recall"] == 1.0, "selection sees both leaves at 0.1")
    print("organ-gate v2 evaluator self-checks: OK")


if __name__ == "__main__":
    _self_test()
