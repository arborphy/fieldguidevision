# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Prepare the OWLv2 reviewed populations for crop verification (read-only).

Consumes the alt-detector worktree's reviewed CSVs and proposal dumps and
writes three separate populations under ``artifacts/owlv2-review/inputs``:

- ``targeted``   — the per-class contact-sheet rows (24 per class)
- ``background`` — the negative/background contact-sheet rows
- ``broad``      — the independent 9-per-class broad-sample rows

Populations are never pooled: the targeted set fits the policy, the broad set
validates it, and the background set measures false accepts on negatives.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ALT_ROOT = Path(
    "/Users/valkyrie/dev/fieldguidevision-anatomy4-alt-detector/experiments/anatomy4/artifacts/alt-review/owlv2"
)
POPULATIONS = ("targeted", "background", "broad")
FIELDNAMES = [
    "proposal_id",
    "class_name",
    "box_score",
    "photo_id",
    "bucket",
    "prompt",
    "area",
    "verdict",
    "notes",
    "source_csv",
]
PROPOSAL_FIELDS = [
    "proposal_id",
    "image_id",
    "photo_id",
    "observation_id",
    "audit_bucket",
    "class_name",
    "class_id",
    "prompt",
    "prompt_variant",
    "detector_label",
    "box_score",
    "detector_bbox",
    "detector_box_area",
    "polygon",
    "status",
    "model",
]


def classify_source(path: Path) -> str:
    name = path.name
    parent = path.parent.name
    if parent == "broad-review":
        return "broad"
    if name.endswith("-bg.csv"):
        return "background"
    return "targeted"


def load_review_rows(alt_root: Path = DEFAULT_ALT_ROOT) -> dict[str, list[dict[str, str]]]:
    populations: dict[str, list[dict[str, str]]] = {name: [] for name in POPULATIONS}
    files = sorted(glob.glob(str(alt_root / "review-*.csv"))) + sorted(
        glob.glob(str(alt_root / "broad-review" / "review-*.csv"))
    )
    for raw_path in files:
        path = Path(raw_path)
        population = classify_source(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not (row.get("proposal_id") or "").strip():
                    continue
                cleaned = {
                    key: (row.get(key) or "").strip() for key in FIELDNAMES if key != "source_csv"
                }
                cleaned["verdict"] = cleaned["verdict"].lower()
                if cleaned["verdict"] not in {"tp", "fp"}:
                    raise ValueError(f"unexpected verdict {cleaned['verdict']!r} in {path}")
                cleaned["source_csv"] = str(path.relative_to(alt_root))
                populations[population].append(cleaned)
    return populations


def load_proposals(alt_root: Path = DEFAULT_ALT_ROOT) -> dict[str, dict[str, Any]]:
    proposals: dict[str, dict[str, Any]] = {}
    for name in ("targeted-proposals.json", "broad-proposals.json"):
        data = json.loads((alt_root / name).read_text())
        for proposal in data.get("proposals") or []:
            proposals[str(proposal["proposal_id"])] = proposal
    return proposals


def build_inputs(
    *,
    alt_root: Path = DEFAULT_ALT_ROOT,
    out_dir: Path,
    image_source: dict[str, str] | None = None,
) -> dict[str, Any]:
    populations = load_review_rows(alt_root)
    proposals = load_proposals(alt_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    referenced_proposals: set[str] = set()
    referenced_photos: set[str] = set()
    summary: dict[str, Any] = {}
    for name, rows in populations.items():
        missing = sorted({row["proposal_id"] for row in rows} - proposals.keys())
        if missing:
            raise ValueError(f"{name}: {len(missing)} proposals missing (e.g. {missing[:3]})")
        with (out_dir / f"review-{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        referenced_proposals.update(row["proposal_id"] for row in rows)
        referenced_photos.update(row["photo_id"] for row in rows)
        summary[name] = {
            "rows": len(rows),
            "photos": len({row["photo_id"] for row in rows}),
            "tp": sum(1 for row in rows if row["verdict"] == "tp"),
            "fp": sum(1 for row in rows if row["verdict"] == "fp"),
            "by_class": {
                cls: {
                    "n": sum(1 for row in rows if row["class_name"] == cls),
                    "tp": sum(
                        1
                        for row in rows
                        if row["class_name"] == cls and row["verdict"] == "tp"
                    ),
                }
                for cls in ("leaf", "needle", "flower", "inflorescence")
                if any(row["class_name"] == cls for row in rows)
            },
        }

    trimmed = {
        "schema": "arq.anatomy4.owlv2-reviewed/v1",
        "source": str(alt_root),
        "populations": summary,
        "proposals": [
            {key: proposals[pid][key] for key in PROPOSAL_FIELDS if key in proposals[pid]}
            for pid in sorted(referenced_proposals)
        ],
    }
    (out_dir / "proposals-reviewed.json").write_text(json.dumps(trimmed, indent=1))

    default_sources = image_source or {
        "targeted": "/srv/fast/anatomy4-shenandoah/targeted/images",
        "background": "/srv/fast/anatomy4-shenandoah/targeted/images",
        "broad": "/srv/fast/anatomy4-shenandoah/images",
    }
    image_manifest: dict[str, dict[str, str]] = {}
    for name, rows in populations.items():
        for row in rows:
            photo_id = row["photo_id"]
            entry = image_manifest.setdefault(
                photo_id, {"source_dir": default_sources[name], "populations": []}
            )
            if name not in entry["populations"]:
                entry["populations"].append(name)
    (out_dir / "image-manifest.json").write_text(json.dumps(image_manifest, indent=1, sort_keys=True))

    return {
        "out_dir": str(out_dir),
        "populations": summary,
        "unique_photos": len(referenced_photos),
        "proposals_written": len(trimmed["proposals"]),
    }


def image_photo_ids(manifest_path: Path, populations: Iterable[str] | None = None) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    if populations is None:
        return sorted(manifest)
    wanted = set(populations)
    return sorted(
        photo_id
        for photo_id, entry in manifest.items()
        if wanted & set(entry["populations"])
    )
