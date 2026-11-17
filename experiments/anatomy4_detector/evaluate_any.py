# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Evaluate an anatomy4 checkpoint on any split of a train/val dataset.

The stock ``evaluate`` command requires the dataset to contain a ``test``
split, which the training dataset deliberately does not (the frozen benchmark
is a separate test-only dataset). This helper is the same code path with
``require_test=False`` so ``train``/``val`` can be scored directly.

    python -m experiments.anatomy4_detector.evaluate_any --dataset <root> \
        --weights <best.pt> --split train --out <dir> \
        --prepared <run>/prepared --conf-thresholds 0.10,0.25,0.50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepared", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--conf-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-overlays", type=int, default=48)
    parser.add_argument("--max-contact-images", type=int, default=240)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args(list(argv) if argv is not None else None)

    from experiments.anatomy4_detector.dataset import (
        prepare_ultralytics,
        validate_dataset,
    )
    from experiments.anatomy4_detector.evaluate import evaluate_split
    from experiments.anatomy4_detector.train import configure_process_env

    configure_process_env()
    audit = validate_dataset(args.dataset, require_test=False)
    if audit.get("blocking"):
        print(json.dumps({"status": "invalid", "errors": audit.get("errors")}))
        return 2
    prepared_root = args.prepared
    if prepared_root is None:
        prepared = prepare_ultralytics(audit, Path(args.out) / "prepared")
        prepared_root = prepared["root"]
    thresholds = sorted(
        {
            round(float(part), 4)
            for part in ([str(args.conf)] + str(args.conf_thresholds).split(","))
            if str(part).strip()
        }
    )
    result = evaluate_split(
        audit=audit,
        split=args.split,
        prepared_root=prepared_root,
        weights=args.weights,
        out_dir=args.out,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        conf_threshold=args.conf,
        conf_thresholds=thresholds,
        iou_threshold=args.iou,
        max_det=args.max_det,
        max_overlays=args.max_overlays,
        max_contact_images=args.max_contact_images,
    )
    print(
        json.dumps(
            {
                "split": result["split"],
                "images": result.get("images"),
                "gt_boxes": result.get("gt_boxes"),
                "macro": result.get("macro"),
                "micro": result.get("micro"),
                "per_class": result.get("per_class"),
                "leaf_tagged_bbox": result.get("leaf_tagged_bbox"),
                "metrics_json": result.get("metrics_json"),
                "contact_sheets": result.get("contact_sheets"),
            },
            indent=1,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
