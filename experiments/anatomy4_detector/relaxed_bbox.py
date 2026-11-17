# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Relaxed (IoU 0.25 / any-overlap) tagged-bbox scoring from eval artifacts.

Reads an ``evaluate`` output directory (``predictions.jsonl`` +
``ground_truth.jsonl``) and recomputes per-class operating-point metrics at
several IoU criteria, including the any-overlap "box hit" rate demanded by the
anatomy4 acceptance discussion. Emits JSON and a markdown table.

    python -m experiments.anatomy4_detector.relaxed_bbox <eval-dir> [more...]

Nothing here trains or writes into datasets; it is pure read-only scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CLASS_NAMES = ("leaf", "needle", "flower", "inflorescence")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
CSS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_split(eval_dir: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    gt: dict[str, list[dict]] = {}
    for row in _load(eval_dir / "ground_truth.jsonl"):
        width = row.get("width") or 1
        height = row.get("height") or 1
        boxes = []
        for box in row.get("boxes") or []:
            boxes.append(
                {
                    "class_id": int(box["class_id"]),
                    "xyxy": [
                        box["x0"] * width,
                        box["y0"] * height,
                        box["x1"] * width,
                        box["y1"] * height,
                    ],
                }
            )
        gt[row["stem"]] = boxes
    det: dict[str, list[dict]] = {}
    for row in _load(eval_dir / "predictions.jsonl"):
        det.setdefault(row["stem"], []).append(
            {
                "class_id": int(row["class_id"]),
                "conf": float(row["confidence"]),
                "xyxy": list(row["xyxy_px"]),
            }
        )
    return gt, det


def _greedy(gt: list[dict], det: list[dict], iou_thr: float, class_aware: bool):
    order = sorted(range(len(det)), key=lambda i: (-det[i]["conf"], i))
    taken = [False] * len(gt)
    matched = [None] * len(det)
    best_ious = [0.0] * len(det)
    for index in order:
        best_i, best = None, 0.0
        for g_index, g in enumerate(gt):
            if taken[g_index]:
                continue
            if class_aware and g["class_id"] != det[index]["class_id"]:
                continue
            value = iou(det[index]["xyxy"], g["xyxy"])
            if value > best:
                best, best_i = value, g_index
        matched_ok = best >= iou_thr if iou_thr > 0 else best > 0.0
        if best_i is not None and matched_ok:
            taken[best_i] = True
            matched[index] = best_i
            best_ious[index] = best
    return matched, best_ious, taken


def _ap(flags: list[int], confs: list[float], n_gt: int) -> float | None:
    if n_gt <= 0:
        return None
    if not flags:
        return 0.0
    order = sorted(range(len(flags)), key=lambda i: (-confs[i], i))
    tp = fp = 0
    rec: list[float] = []
    prec: list[float] = []
    for index in order:
        if flags[index]:
            tp += 1
        else:
            fp += 1
        rec.append(tp / n_gt)
        prec.append(tp / (tp + fp))
    env = list(prec)
    for index in range(len(env) - 2, -1, -1):
        env[index] = max(env[index], env[index + 1])
    total = 0.0
    for step in range(101):
        target = step / 100.0
        position = next((i for i, value in enumerate(rec) if value >= target), None)
        total += env[position] if position is not None else 0.0
    return total / 101.0


def score(
    gt: Mapping[str, list[dict]],
    det: Mapping[str, list[dict]],
    *,
    conf: float,
    iou_thr: float,
    class_aware: bool,
    class_id: int | None = None,
    confs_floor: float = 0.001,
) -> dict[str, Any]:
    tp = fp = 0
    n_gt = 0
    hits = 0
    flags: list[int] = []
    scores: list[float] = []
    neg_images = 0
    neg_with_fp = 0
    for stem in sorted(set(gt) | set(det)):
        g = gt.get(stem, [])
        d = [item for item in det.get(stem, []) if item["conf"] >= conf]
        if class_id is not None:
            g = [item for item in g if item["class_id"] == class_id]
            if class_aware:
                d = [item for item in d if item["class_id"] == class_id]
        n_gt += len(g)
        if stem in gt and not gt[stem]:
            neg_images += 1
            if any(item["conf"] >= conf for item in det.get(stem, [])):
                neg_with_fp += 1
        matched, _, taken = _greedy(g, d, iou_thr, class_aware)
        tp += sum(1 for item in matched if item is not None)
        fp += sum(1 for item in matched if item is None)
        hits += sum(1 for value in taken if value)
        # AP over the low floor (independent of the operating point)
        d_ap = [item for item in det.get(stem, []) if item["conf"] >= confs_floor]
        if class_id is not None and class_aware:
            d_ap = [item for item in d_ap if item["class_id"] == class_id]
        m_ap, _, _ = _greedy(g, sorted(d_ap, key=lambda x: -x["conf"]), iou_thr, class_aware)
        for index, item in enumerate(sorted(d_ap, key=lambda x: -x["conf"])):
            flags.append(1 if m_ap[index] is not None else 0)
            scores.append(item["conf"])
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / n_gt if n_gt else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else None
    )
    return {
        "conf": conf,
        "iou": iou_thr,
        "class_aware": class_aware,
        "n_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": n_gt - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "any_overlap_hit_rate": (hits / n_gt) if n_gt else None,
        "ap": _ap(flags, scores, n_gt),
        "negative_images": neg_images,
        "negative_fp_images": neg_with_fp,
        "negative_fp_rate": (neg_with_fp / neg_images) if neg_images else None,
    }


def report(eval_dir: Path) -> dict[str, Any]:
    gt, det = load_split(eval_dir)
    out: dict[str, Any] = {
        "eval_dir": str(eval_dir),
        "images": len(gt),
        "gt_boxes": sum(len(v) for v in gt.values()),
        "per_class": {},
    }
    criteria = (
        ("iou50", 0.50, True),
        ("iou25", 0.25, True),
        ("any_overlap", 0.0, True),
        ("any_overlap_anyclass", 0.0, False),
    )
    for name in CLASS_NAMES:
        block: dict[str, Any] = {}
        for label, thr, aware in criteria:
            block[label] = {
                f"{conf:.2f}": score(
                    gt,
                    det,
                    conf=conf,
                    iou_thr=thr,
                    class_aware=aware,
                    class_id=CLASS_IDS[name],
                )
                for conf in CSS
            }
        out["per_class"][name] = block
    out["micro"] = {
        label: {
            f"{conf:.2f}": score(
                gt, det, conf=conf, iou_thr=thr, class_aware=aware
            )
            for conf in CSS
        }
        for label, thr, aware in criteria
    }
    return out


def markdown(results: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(f"# relaxed tagged-bbox scoring — {result['eval_dir']}")
        lines.append("")
        lines.append(
            f"images {result['images']} · gt boxes {result['gt_boxes']}"
        )
        lines.append("")
        for name, block in result["per_class"].items():
            lines.append(f"## {name}")
            lines.append("")
            lines.append(
                "| criterion | conf | P | R | F1 | AP | any-overlap hit | neg FP rate |"
            )
            lines.append("|---|---|---|---|---|---|---|---|")
            for label, by_conf in block.items():
                for conf, row in by_conf.items():
                    def fmt(value: Any) -> str:
                        return "—" if value is None else f"{value:.3f}"
                    lines.append(
                        f"| {label} | {conf} | {fmt(row['precision'])} | "
                        f"{fmt(row['recall'])} | {fmt(row['f1'])} | {fmt(row['ap'])} | "
                        f"{fmt(row['any_overlap_hit_rate'])} | "
                        f"{fmt(row['negative_fp_rate'])} |"
                    )
            lines.append("")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dirs", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    results = [report(path) for path in args.eval_dirs]
    payload = json.dumps(results, indent=1)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n")
    else:
        print(payload)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown(results) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
