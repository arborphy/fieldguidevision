# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Detector comparison metrics over anatomy4 review CSVs.

All numbers are *reviewed* numbers: a proposal only counts if a human marked it
``tp`` or ``fp``. Nothing here infers a label from taxonomy, and a box-only
proposal is scored exactly like a masked one (the mask is not the question).

Reported per class:
- precision (tp / decided) on the reviewed sample
- correct+tight (tp whose box area <= TIGHT_MAX_AREA) among reviewed proposals
- precision at matched N (top-N reviewed proposals by score) so detectors
  reviewed at different sample sizes are compared on the same proposal count
- score-threshold sweep
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .contracts import CLASS_NAMES

TIGHT_MAX_AREA = 0.60

TP = {"tp", "accept", "accepted"}
FP = {"fp", "reject", "rejected"}
UNSURE = {"unsure", "?"}


def load_review_csv(path: str | Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            proposal_id = (row.get("proposal_id") or "").strip()
            if proposal_id:
                rows[proposal_id] = row
    return rows


def load_review_dir(path: str | Path) -> dict[str, dict[str, str]]:
    """Merge ``review-<class>.csv`` files written by ``altsheets``.

    ``path`` may be a comma-separated list of directories, so a detector's
    class sample and its background/negative sample can be merged into one
    full-pool estimate.
    """

    merged: dict[str, dict[str, str]] = {}
    for part in str(path).split(","):
        for csv_path in sorted(Path(part).glob("review-*.csv")):
            merged.update(load_review_csv(csv_path))
    return merged


def _proposals_by_id(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(proposal["proposal_id"]): proposal
        for proposal in manifest.get("proposals", [])
        if proposal.get("status") == "proposed"
    }


def _decided(review: Mapping[str, str] | None) -> str:
    verdict = (review.get("verdict") or "").strip().lower() if review else ""
    if verdict in TP:
        return "tp"
    if verdict in FP:
        return "fp"
    if verdict in UNSURE:
        return "unsure"
    return "unreviewed"


def summarize(
    manifest: Mapping[str, Any],
    reviews: Mapping[str, Mapping[str, str]],
    *,
    tight_max_area: float = TIGHT_MAX_AREA,
    dataset: str = "",
) -> dict[str, Any]:
    proposals = _proposals_by_id(manifest)
    per_class: dict[str, dict[str, Any]] = {
        name: {
            "reviewed": 0,
            "tp": 0,
            "fp": 0,
            "unsure": 0,
            "correct_tight": 0,
            "areas": [],
            "rows": [],
        }
        for name in CLASS_NAMES
    }
    for proposal_id, proposal in proposals.items():
        review = reviews.get(proposal_id)
        verdict = _decided(review)
        if verdict not in {"tp", "fp", "unsure"}:
            continue
        bucket = per_class[proposal["class_name"]]
        bucket["reviewed"] += 1
        if verdict == "tp":
            bucket["tp"] += 1
        elif verdict == "fp":
            bucket["fp"] += 1
        else:
            bucket["unsure"] += 1
        area = float(proposal.get("detector_box_area") or 0.0)
        bucket["areas"].append(area)
        tight = verdict == "tp" and area <= tight_max_area
        if tight:
            bucket["correct_tight"] += 1
        bucket["rows"].append(
            {
                "proposal_id": proposal_id,
                "score": float(proposal["box_score"]),
                "area": area,
                "verdict": verdict,
                "tight": tight,
                "bucket": proposal.get("audit_bucket"),
                "prompt": proposal.get("prompt"),
            }
        )

    results: dict[str, Any] = {}
    for name, bucket in per_class.items():
        rows = sorted(bucket["rows"], key=lambda row: row["score"], reverse=True)
        decided = bucket["tp"] + bucket["fp"]

        def precision_at(n: int) -> float | None:
            head = rows[:n]
            decided_head = [row for row in head if row["verdict"] in {"tp", "fp"}]
            if not decided_head:
                return None
            return sum(1 for row in decided_head if row["verdict"] == "tp") / len(
                decided_head
            )

        sweep = {}
        for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            above = [row for row in rows if row["score"] >= threshold]
            decided_above = [row for row in above if row["verdict"] in {"tp", "fp"}]
            sweep[f">={threshold:.2f}"] = {
                "n": len(above),
                "tp": sum(1 for row in above if row["verdict"] == "tp"),
                "precision": (
                    sum(1 for row in decided_above if row["verdict"] == "tp")
                    / len(decided_above)
                    if decided_above
                    else None
                ),
            }

        prompts: dict[str, dict[str, int]] = {}
        for row in rows:
            prompt = str(row.get("prompt") or "unknown")
            entry = prompts.setdefault(prompt, {"tp": 0, "fp": 0, "reviewed": 0})
            if row["verdict"] in {"tp", "fp"}:
                entry[row["verdict"]] += 1
                entry["reviewed"] += 1
        for entry in prompts.values():
            entry["precision"] = (
                entry["tp"] / entry["reviewed"] if entry["reviewed"] else None
            )

        results[name] = {
            "reviewed": bucket["reviewed"],
            "tp": bucket["tp"],
            "fp": bucket["fp"],
            "unsure": bucket["unsure"],
            "precision": (bucket["tp"] / decided) if decided else None,
            "correct_tight": bucket["correct_tight"],
            "correct_tight_per_reviewed": (
                bucket["correct_tight"] / bucket["reviewed"] if bucket["reviewed"] else None
            ),
            "median_area": _median(bucket["areas"]),
            "precision_at_n": {
                str(n): precision_at(n) for n in (12, 24) if n <= bucket["reviewed"]
            },
            "score_sweep": sweep,
            "by_prompt": prompts,
        }

    reviewed = sum(value["reviewed"] for value in results.values())
    tp = sum(value["tp"] for value in results.values())
    fp = sum(value["fp"] for value in results.values())
    tight = sum(value["correct_tight"] for value in results.values())
    return {
        "dataset": dataset,
        "backend": manifest.get("backend"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "summary": manifest.get("summary", {}),
        "per_class": results,
        "overall": {
            "reviewed": reviewed,
            "tp": tp,
            "fp": fp,
            "precision": (tp / (tp + fp)) if tp + fp else None,
            "correct_tight": tight,
            "correct_tight_per_reviewed": (tight / reviewed) if reviewed else None,
        },
    }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def markdown_table(results: Sequence[Mapping[str, Any]]) -> str:
    header = "| detector | set | " + " | ".join(
        f"{name} P (n)" for name in CLASS_NAMES
    ) + " | tight | overall P |"
    lines = [header, "| --- | --- | " + " | ".join("---" for _ in CLASS_NAMES) + " | --- | --- |"]
    for result in results:
        cells = []
        for name in CLASS_NAMES:
            cell = result["per_class"][name]
            cells.append(f"{_fmt(cell['precision'])} ({cell['reviewed']})")
        lines.append(
            "| {backend} | {dataset} | {cells} | {tight} | {overall} |".format(
                backend=result["backend"],
                dataset=result["dataset"],
                cells=" | ".join(cells),
                tight=_fmt(result["overall"]["correct_tight_per_reviewed"]),
                overall=_fmt(result["overall"]["precision"]),
            )
        )
    return "\n".join(lines)


def _load_reviews(path: str) -> dict[str, dict[str, str]]:
    if any(Path(part).is_dir() for part in str(path).split(",")):
        return load_review_dir(path)
    return load_review_csv(path)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Compare anatomy4 detector reviews")
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("NAME", "MANIFEST", "REVIEW"),
        required=True,
        help="name manifest reviews (repeatable); REVIEW may be a CSV or a "
        "directory of review-<class>.csv files",
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    results = []
    for name, manifest_path, review_path in args.run:
        manifest = json.loads(Path(manifest_path).read_text())
        reviews = _load_reviews(review_path)
        result = summarize(manifest, reviews, dataset=f"{args.dataset}:{name}" if args.dataset else name)
        results.append(result)

    print(markdown_table(results))
    print("\n```json")
    print(json.dumps(results, indent=1, default=str))
    print("```")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
