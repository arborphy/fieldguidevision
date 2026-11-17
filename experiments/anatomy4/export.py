# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Export review-ready proposal tables from an alternate-detector run.

This is the hand-off artifact for manual bounding-box labeling: one row per
proposal, class-specific thresholds, no taxonomy, provenance attached. It never
writes YOLO labels — accepted labels still require human review.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import CLASS_NAMES

DEFAULT_THRESHOLDS: dict[str, float] = {
    "leaf": 0.25,
    "needle": 0.10,
    "flower": 0.15,
    "inflorescence": 0.15,
}


def export_proposals(
    manifest: Mapping[str, Any],
    out_dir: str | Path,
    *,
    thresholds: Mapping[str, float] | None = None,
    max_box_area: float = 0.60,
    per_class_limit: int | None = None,
) -> dict[str, Any]:
    thresholds = dict(thresholds or DEFAULT_THRESHOLDS)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "backend": manifest.get("backend"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "model": manifest.get("provenance"),
    }
    selected: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_NAMES}
    for proposal in manifest.get("proposals", []):
        if proposal.get("status") != "proposed":
            continue
        class_name = proposal["class_name"]
        if float(proposal["box_score"]) < thresholds.get(class_name, 1.0):
            continue
        if float(proposal.get("detector_box_area") or 1.0) > max_box_area:
            continue
        selected[class_name].append(proposal)

    summary: dict[str, Any] = {"thresholds": thresholds, "max_box_area": max_box_area}
    rows: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        group = sorted(
            selected[class_name], key=lambda p: p["box_score"], reverse=True
        )
        if per_class_limit is not None:
            group = group[:per_class_limit]
        summary[class_name] = len(group)
        for rank, proposal in enumerate(group):
            rows.append(
                {
                    "class_name": class_name,
                    "rank": rank,
                    "proposal_id": proposal["proposal_id"],
                    "photo_id": proposal["photo_id"],
                    "observation_id": proposal.get("observation_id", ""),
                    "image_path": proposal.get("image_path", ""),
                    "box_score": f"{proposal['box_score']:.4f}",
                    "prompt": proposal.get("prompt", ""),
                    "x0": f"{proposal['detector_bbox'][0]:.6f}",
                    "y0": f"{proposal['detector_bbox'][1]:.6f}",
                    "x1": f"{proposal['detector_bbox'][2]:.6f}",
                    "y1": f"{proposal['detector_bbox'][3]:.6f}",
                    "box_area": f"{proposal.get('detector_box_area', 0):.6f}",
                    "proposal_kind": proposal.get("proposal_kind", "mask"),
                    "review_verdict": "",
                    "review_notes": "",
                }
            )

    csv_path = output / "proposals.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["class_name"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "proposals.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else "")
    )
    summary["rows"] = len(rows)
    summary["provenance"] = provenance
    (output / "export.json").write_text(json.dumps(summary, indent=1))
    return summary
