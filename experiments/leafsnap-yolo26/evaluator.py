# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Transparent detection evaluator for the LeafSnap / BioImages organ gate.

Why this exists
---------------
Ultralytics reports box mAP50 / mAP50-95 and per-class P/R, but it cannot
break those numbers down by *data source*. The organ gate needs that split:
LeafSnap field boxes are segmentation-derived (real localization), while
BioImages boxes are full-frame / near-full-frame WEAK labels. Mixing them in
one aggregate hides the only localization claim the gate makes.

This module is deliberately small and explicit:

* one ground-truth box per image (guaranteed by the dataset build: 8,560
  images, 8,560 manifest rows, one label line per image),
* greedy confidence-ordered matching at IoU >= ``IOU_MATCH`` (default 0.50),
* per-class precision / recall / F1 at a *given* confidence threshold,
* all-point average precision (AP) per class, threshold-free, at IoU 0.50,
* per-source slices (LeafSnap field vs BioImages weak-box organ),
* confidence selection ON VALIDATION ONLY.

Nothing here touches the test split for tuning: ``select_confidence`` accepts
validation records, and ``evaluate`` is a pure function of the records plus
the confidence chosen beforehand.

Run the self-checks with::

    .venv/bin/python experiments/leafsnap-yolo26/evaluator.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

CLASS_NAMES: tuple[str, ...] = (
    "leaf",
    "bud",
    "flower",
    "twig",
    "branch",
    "entire_plant",
)
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
NONLEAF_CLASSES: tuple[str, ...] = ("bud", "flower", "twig", "branch", "entire_plant")

IOU_MATCH = 0.5
AP_IOU = 0.5
CONF_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05..0.95


# --------------------------------------------------------------------------
# manifest + geometry
# --------------------------------------------------------------------------
def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load the detector manifest (one row per image, one GT box per row)."""
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
        if row["class_name"] not in CLASS_IDS:
            raise ValueError(f"unknown class {row['class_name']!r} for {uid}")
        if len(row["bbox_xyxy"]) != 4:
            raise ValueError(f"bad bbox for {uid}: {row['bbox_xyxy']}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
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


def make_record(
    row: dict[str, Any],
    preds: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Turn a manifest row + raw predictions into one evaluation record.

    ``preds`` items look like ``{"class_id": int, "confidence": float,
    "bbox_xyxy": [x1, y1, x2, y2]}``. Raw predictions are kept (not filtered)
    so any confidence threshold can be applied later without re-running
    inference.
    """
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
        "gt_class_id": CLASS_IDS[row["class_name"]],
        "gt_box": [round(float(v), 2) for v in row["bbox_xyxy"]],
        "preds": kept,
    }


# --------------------------------------------------------------------------
# matching / PRF at a confidence threshold
# --------------------------------------------------------------------------
def _match_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Greedy IoU-0.5 matching over ALL saved predictions (confidence floor).

    Matching runs once per image in confidence order, so a threshold can be
    applied afterwards by filtering the flags: a prediction below the threshold
    can never have "used up" the ground-truth box for a higher-confidence one.
    This keeps AP threshold-free and PRF exactly consistent with matching.
    """
    npos_by_class: dict[int, int] = defaultdict(int)
    flags: list[tuple[float, bool, int]] = []
    # diagnostic: wrong-class prediction overlapping the GT box
    cross_events: list[tuple[float, str, str]] = []

    for rec in records:
        gt_c = rec["gt_class_id"]
        npos_by_class[gt_c] += 1
        matched = False
        for p in rec["preds"]:
            iou = iou_xyxy(p["b"], rec["gt_box"])
            ok = p["c"] == gt_c and iou >= IOU_MATCH
            # Only the FIRST same-class match is a true positive. Later
            # duplicates (including additional correct-class boxes on the same
            # single GT) are false positives in both PRF and the AP flags,
            # otherwise AP could exceed 1.0.
            is_tp = bool(ok and not matched)
            flags.append((p["s"], is_tp, p["c"]))
            if is_tp:
                matched = True
            elif p["c"] != gt_c and iou >= IOU_MATCH:
                cross_events.append((p["s"], CLASS_NAMES[gt_c], CLASS_NAMES[p["c"]]))

    return {
        "npos": npos_by_class,
        "flags": flags,
        "cross_events": cross_events,
    }


def prf_from_flags(matched: dict[str, Any], conf: float) -> dict[str, Any]:
    """Per-class TP/FP/FN at ``conf`` from the pre-computed matching flags."""
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


def average_precision(flags: Sequence[tuple[float, bool, int]], class_id: int, npos: int) -> float | None:
    """All-point AP at IoU 0.50 for one class (None when no positives)."""
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
    # Precision envelope, then integrate the monotonically increasing recall.
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
        # Duplicate/incorrect matching would silently inflate AP; fail loudly.
        raise AssertionError(f"AP invariant violated: recall={prev_r} ap={ap} npos={npos}")
    return float(ap)


def _class_block(matched: dict[str, Any], class_id: int, conf: float) -> dict[str, Any]:
    name = CLASS_NAMES[class_id]
    prf = prf_from_flags(matched, conf)
    block = _prf(
        prf["tp"].get(class_id, 0),
        prf["fp"].get(class_id, 0),
        prf["fn"].get(class_id, 0),
    )
    block["ap50"] = average_precision(matched["flags"], class_id, matched["npos"].get(class_id, 0))
    if block["ap50"] is not None:
        block["ap50"] = round(block["ap50"], 4)
    block["class"] = name
    return block


def _subset(records: Sequence[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        if all(rec.get(k) == v for k, v in criteria.items()):
            out.append(rec)
    return out


def evaluate(
    records: Sequence[dict[str, Any]],
    conf: float,
    *,
    label: str = "test",
) -> dict[str, Any]:
    """Full metric bundle at one confidence threshold.

    ``conf`` MUST have been chosen on validation data only. The function does
    not look at any other split.
    """
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
    nonleaf_f1 = sum(per_class[n]["f1"] for n in nonleaf) / len(nonleaf) if nonleaf else None

    ap_by_class = {n: per_class[n]["ap50"] for n in CLASS_NAMES}
    ap_values = [v for v in ap_by_class.values() if v is not None]
    map50 = sum(ap_values) / len(ap_values) if ap_values else None

    # ---- per-source slices -------------------------------------------------
    leaf_leafsnap = _subset(records, source="leafsnap", domain="field")
    leafsnap_match = _match_records(leaf_leafsnap)
    leaf_block = _class_block(leafsnap_match, CLASS_IDS["leaf"], conf)

    bio = _subset(records, source="bioimages")
    bio_match = _match_records(bio)
    bio_classes = {
        name: _class_block(bio_match, i, conf)
        for i, name in enumerate(CLASS_NAMES)
        if name in NONLEAF_CLASSES and (bio_match["npos"].get(i, 0) > 0)
    }
    bio_recalls = [b["recall"] for b in bio_classes.values()]
    bio_aps = [b["ap50"] for b in bio_classes.values() if b["ap50"] is not None]

    return {
        "label": label,
        "confidence_threshold": conf,
        "iou_match": IOU_MATCH,
        "n_images": len(records),
        "overall": totals,
        "macro_f1_present_classes": round(macro_f1, 4),
        "map50_iou0.5": round(map50, 4) if map50 is not None else None,
        "per_class": per_class,
        "nonleaf_macro_recall": round(nonleaf_recall, 4) if nonleaf_recall is not None else None,
        "nonleaf_macro_f1": round(nonleaf_f1, 4) if nonleaf_f1 is not None else None,
        "sources": {
            "leafsnap_field_leaf": {
                **leaf_block,
                "n_images": len(leaf_leafsnap),
                "box_kind": "segmentation-derived (real localization)",
            },
            "bioimages_weak_box": {
                "n_images": len(bio),
                "box_kind": "full-frame / near-full-frame weak labels (NOT localization quality)",
                "per_class": bio_classes,
                "macro_recall": round(sum(bio_recalls) / len(bio_recalls), 4) if bio_recalls else None,
                "map50_iou0.5": round(sum(bio_aps) / len(bio_aps), 4) if bio_aps else None,
            },
        },
        "cross_class_matches": dict(sorted(totals_prf["cross_class"].items())),
        "ap_note": "AP is threshold-free and computed over all saved predictions (conf floor 0.001).",
    }


# --------------------------------------------------------------------------
# validation-only confidence selection
# --------------------------------------------------------------------------
def prf_curve(records: Sequence[dict[str, Any]], conf: float) -> dict[str, Any]:
    """Light-weight per-class P/R at ``conf`` (used by the selection sweep)."""
    matched = _match_records(records)
    prf = prf_from_flags(matched, conf)
    out = {}
    for i, name in enumerate(CLASS_NAMES):
        if matched["npos"].get(i, 0) == 0:
            continue
        out[name] = _prf(
            prf["tp"].get(i, 0), prf["fp"].get(i, 0), prf["fn"].get(i, 0)
        )
    return out


def select_confidence(
    val_records: Sequence[dict[str, Any]],
    grid: Sequence[float] = CONF_GRID,
) -> dict[str, Any]:
    """Pick one global confidence threshold on VALIDATION ONLY.

    Rule (documented, not tuned on test): maximize the unweighted mean F1 over
    the classes that have ground truth in validation. Ties break toward the
    lower threshold (higher recall), then toward the grid order.
    """
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


# --------------------------------------------------------------------------
# self-checks
# --------------------------------------------------------------------------
def _rec(
    uid: str,
    cls: str,
    gt: Sequence[float],
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
        "box_kind": "full_frame_weak" if weak else "mask_bbox",
        "gt_class_id": CLASS_IDS[cls],
        "gt_box": list(gt),
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
    # IoU: identical boxes -> 1.0; disjoint -> 0.0; half overlap -> 1/3.
    _check(_approx(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]), 1.0), "identical IoU")
    _check(_approx(iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0.0), "disjoint IoU")
    _check(_approx(iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3), "half-overlap IoU")
    _check(_approx(iou_xyxy([0, 0, 0, 0], [0, 0, 0, 0]), 0.0), "degenerate IoU")

    # Perfect leaf detections: P = R = F1 = AP = 1.
    perfect = [_rec(f"a{i}", "leaf", [10, 10, 50, 50], [(0, 0.9, [10, 10, 50, 50])]) for i in range(4)]
    m = evaluate(perfect, 0.25)
    leaf = m["per_class"]["leaf"]
    _check(leaf["precision"] == 1.0 and leaf["recall"] == 1.0, "perfect leaf PR")
    _check(leaf["ap50"] == 1.0, "perfect leaf AP")
    _check(m["sources"]["leafsnap_field_leaf"]["recall"] == 1.0, "source leaf recall")

    # Misses below threshold count as FN; misses above also FN, plus FP if wrong class.
    recs = [
        _rec("b0", "leaf", [10, 10, 50, 50], [(0, 0.10, [10, 10, 50, 50])]),  # suppressed by conf
        _rec("b1", "leaf", [10, 10, 50, 50], [(5, 0.80, [10, 10, 50, 50])]),  # wrong class
    ]
    m = evaluate(recs, 0.25)
    _check(m["per_class"]["leaf"]["recall"] == 0.0, "suppressed + wrong-class => no recall")
    _check(m["per_class"]["leaf"]["fn"] == 2, "two leaf FN")
    _check(m["per_class"]["entire_plant"]["fp"] == 1, "wrong class FP")
    _check(m["cross_class_matches"] == {"leaf->entire_plant": 1}, "cross-class diagnostic")

    # Duplicate detections: earliest matched TP, later duplicate FP.
    dup = [_rec("c0", "leaf", [10, 10, 50, 50], [
        (0, 0.9, [10, 10, 50, 50]),
        (0, 0.5, [11, 11, 50, 50]),
    ])]
    m = evaluate(dup, 0.25)
    _check(m["per_class"]["leaf"]["tp"] == 1 and m["per_class"]["leaf"]["fp"] == 1, "duplicate penalized")
    _check(m["per_class"]["leaf"]["ap50"] == 1.0, "duplicate detections cannot push AP above 1.0")

    # Many duplicate correct boxes on one GT: recall stays 1.0 and AP stays 1.0,
    # extra boxes only add false positives.
    many = [_rec("c1", "leaf", [0, 0, 10, 10], [(0, 0.9 - i * 0.01, [0, 0, 10, 10]) for i in range(6)])]
    m = evaluate(many, 0.05)
    _check(m["per_class"]["leaf"]["recall"] == 1.0, "duplicate recall <= 1")
    _check(m["per_class"]["leaf"]["ap50"] == 1.0, "duplicate AP clamped at 1")
    _check(m["per_class"]["leaf"]["fp"] == 5, "duplicate extras are FP")

    # IoU below 0.5 is not a match.
    low = [_rec("d0", "leaf", [0, 0, 100, 100], [(0, 0.9, [0, 0, 40, 100])])]  # IoU 0.4
    _check(evaluate(low, 0.25)["per_class"]["leaf"]["tp"] == 0, "IoU<0.5 rejected")

    # AP on a 4-positive, 3-detection ladder: confs 0.9 TP, 0.8 FP, 0.7 TP.
    # precision at recall 1/4 -> 1.0, at 2/4 -> 2/3 => AP = 0.25*1 + 0.25*(2/3).
    ladder = [
        _rec("e0", "leaf", [0, 0, 10, 10], [(0, 0.9, [0, 0, 10, 10])]),
        _rec("e1", "leaf", [0, 0, 10, 10], []),
        _rec("e2", "leaf", [0, 0, 10, 10], [(0, 0.7, [0, 0, 10, 10])]),
        _rec("e3", "leaf", [100, 100, 110, 110], [(0, 0.8, [0, 0, 10, 10])]),
    ]
    ap = evaluate(ladder, 0.05)["per_class"]["leaf"]["ap50"]
    _check(_approx(ap, 0.25 * 1.0 + 0.25 * (2 / 3), 1e-3), f"AP ladder got {ap}")

    # Confidence selection uses only the records it is given.
    val = [
        _rec("v0", "leaf", [0, 0, 10, 10], [(0, 0.9, [0, 0, 10, 10])], split="val"),
        _rec("v1", "leaf", [0, 0, 10, 10], [(0, 0.2, [0, 0, 10, 10])], split="val"),
        _rec("v2", "bud", [0, 0, 10, 10], [(1, 0.8, [0, 0, 10, 10])], split="val",
             source="bioimages", domain="bioimages", weak=True),
    ]
    sel = select_confidence(val, (0.1, 0.25, 0.85))
    _check(sel["confidence"] in (0.1, 0.25, 0.85), "selection returns a grid point")
    _check(sel["per_class"]["leaf"]["recall"] == 1.0, "selection sees both leafs at 0.1")
    print("evaluator self-checks: OK")


if __name__ == "__main__":
    _self_test()
