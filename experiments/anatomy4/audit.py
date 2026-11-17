# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Audit metrics for the anatomy4 bootstrap."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .contracts import CLASS_NAMES


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def reviewed_metrics(
    manifest: Mapping[str, Any], reviews: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    proposals = {
        str(proposal["proposal_id"]): proposal
        for proposal in manifest.get("proposals", [])
        if proposal.get("status") == "proposed"
    }
    per_class: dict[str, dict[str, int]] = {
        name: {"tp": 0, "fp": 0, "unsure": 0, "unreviewed": 0} for name in CLASS_NAMES
    }
    score_buckets: dict[str, dict[str, list[int]]] = {
        name: defaultdict(lambda: [0, 0]) for name in CLASS_NAMES
    }
    notes: dict[str, list[str]] = defaultdict(list)
    prompt_counts: dict[str, dict[str, dict[str, int]]] = {
        name: {} for name in CLASS_NAMES
    }
    for proposal_id, proposal in proposals.items():
        review = reviews.get(proposal_id)
        class_name = proposal["class_name"]
        verdict = (review.get("verdict") or "").strip().lower() if review else ""
        if verdict in {"tp", "accept", "accepted"}:
            per_class[class_name]["tp"] += 1
            bucket = _score_bucket(float(proposal["box_score"]))
            score_buckets[class_name][bucket][0] += 1
        elif verdict in {"fp", "reject", "rejected"}:
            per_class[class_name]["fp"] += 1
            bucket = _score_bucket(float(proposal["box_score"]))
            score_buckets[class_name][bucket][1] += 1
            if review.get("notes"):
                notes[class_name].append(review["notes"])
        elif verdict in {"unsure", "?"}:
            per_class[class_name]["unsure"] += 1
        else:
            per_class[class_name]["unreviewed"] += 1
        prompt = str(
            proposal.get("prompt_variant")
            or proposal.get("prompt")
            or proposal.get("detector_label")
            or "unknown"
        )
        bucket = prompt_counts[class_name].setdefault(
            prompt, {"tp": 0, "fp": 0, "unsure": 0, "unreviewed": 0}
        )
        if verdict in {"tp", "accept", "accepted"}:
            bucket["tp"] += 1
        elif verdict in {"fp", "reject", "rejected"}:
            bucket["fp"] += 1
        elif verdict in {"unsure", "?"}:
            bucket["unsure"] += 1
        else:
            bucket["unreviewed"] += 1

    results: dict[str, Any] = {}
    for name, counts in per_class.items():
        decided = counts["tp"] + counts["fp"]
        results[name] = {
            **counts,
            "precision": (counts["tp"] / decided) if decided else None,
            "reviewed": decided + counts["unsure"],
            "score_buckets": {
                bucket: {"tp": values[0], "fp": values[1]}
                for bucket, values in sorted(score_buckets[name].items())
            },
            "fp_notes": notes.get(name, [])[:20],
            "usable_per_1000": (
                round(1000 * counts["tp"] / decided) if decided else None
            ),
            "by_prompt": {
                prompt: {
                    **values,
                    "precision": (
                        values["tp"] / (values["tp"] + values["fp"])
                        if values["tp"] + values["fp"]
                        else None
                    ),
                    "usable_per_1000": (
                        round(1000 * values["tp"] / (values["tp"] + values["fp"]))
                        if values["tp"] + values["fp"]
                        else None
                    ),
                }
                for prompt, values in sorted(prompt_counts[name].items())
            },
        }
    total_tp = sum(value["tp"] for value in results.values())
    total_fp = sum(value["fp"] for value in results.values())
    return {
        "per_class": results,
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "precision": (total_tp / (total_tp + total_fp)) if total_tp + total_fp else None,
            "reviewed": sum(value["reviewed"] for value in results.values()),
            "proposals": len(proposals),
            "coverage": (
                sum(value["reviewed"] for value in results.values()) / len(proposals)
                if proposals
                else None
            ),
        },
    }


def _score_bucket(score: float) -> str:
    if score < 0.25:
        return "<0.25"
    if score < 0.35:
        return "0.25-0.35"
    if score < 0.5:
        return "0.35-0.50"
    return ">=0.50"


def _image_key(image: Mapping[str, Any]) -> str:
    return str(image.get("photo_id") or image.get("image_id") or "")


def compute_audit(
    manifest: Mapping[str, Any],
    reviews: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    images = manifest.get("images", [])
    proposals = [
        proposal
        for proposal in manifest.get("proposals", [])
        if proposal.get("status") == "proposed" and proposal.get("polygon")
    ]
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        by_image[str(proposal["photo_id"])].append(proposal)

    per_class_counts: Counter[str] = Counter()
    by_source = Counter()
    labels_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    scores: dict[str, list[float]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    for proposal in manifest.get("proposals", []):
        status_counts[proposal.get("status", "unknown")] += 1
        if proposal.get("status") != "proposed":
            continue
        per_class_counts[proposal["class_name"]] += 1
        scores[proposal["class_name"]].append(float(proposal["box_score"]))
        labels_by_class[proposal["class_name"]][proposal.get("detector_label", "")] += 1
        by_source[str(proposal.get("model", {}).get("detector", "unknown"))] += 1

    image_class_sets = {
        image_id: {proposal["class_name"] for proposal in items}
        for image_id, items in by_image.items()
    }
    empty_images = [
        _image_key(image)
        for image in images
        if not by_image.get(_image_key(image))
    ]
    multi = [
        image_id
        for image_id, classes in image_class_sets.items()
        if len(classes) >= 2
    ]
    multi_counter: Counter[str] = Counter(
        "+".join(sorted(classes)) for classes in image_class_sets.values() if len(classes) >= 2
    )
    audit: dict[str, Any] = {
        "schema": "arq.anatomy4.audit/v1",
        "contract": manifest.get("contract"),
        "images_selected": len(images),
        "images_with_proposals": len(by_image),
        "images_empty": len(empty_images),
        "empty_image_rate": (len(empty_images) / len(images)) if images else None,
        "empty_image_ids": empty_images[:40],
        "multi_organ_images": len(multi),
        "multi_organ_rate": (len(multi) / len(images)) if images else None,
        "multi_organ_combinations": dict(multi_counter.most_common(20)),
        "proposals_total": len(proposals),
        "proposals_by_class": {name: per_class_counts.get(name, 0) for name in CLASS_NAMES},
        "proposals_per_image_by_class": {
            name: (
                per_class_counts.get(name, 0) / len(images) if images else None
            )
            for name in CLASS_NAMES
        },
        "score_summary": {
            name: {
                "min": min(values) if values else None,
                "mean": mean(values) if values else None,
                "p50": _quantile(values, 0.5),
                "p90": _quantile(values, 0.9),
                "max": max(values) if values else None,
            }
            for name, values in scores.items()
        },
        "detector_labels": {
            name: dict(counter.most_common(12)) for name, counter in labels_by_class.items()
        },
        "status_counts": dict(status_counts),
        "runtime_seconds_total": sum(
            record.get("elapsed_seconds", 0.0) for record in manifest.get("images", [])
        ),
        "provenance": manifest.get("provenance"),
    }
    if reviews:
        audit["reviewed"] = reviewed_metrics(manifest, reviews)
    return audit


def render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Anatomy4 Shenandoah detector audit",
        "",
        f"- images selected: {audit['images_selected']}",
        f"- images with proposals: {audit['images_with_proposals']}",
        f"- empty-image rate: {_pct(audit['empty_image_rate'])}",
        f"- multi-organ rate: {_pct(audit['multi_organ_rate'])}",
        f"- proposals: {audit['proposals_total']}",
        f"- runtime: {audit['runtime_seconds_total']:.1f}s",
        "",
        "| class | proposals | per image | p50 score | p90 score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in CLASS_NAMES:
        score = audit["score_summary"].get(name, {})
        lines.append(
            f"| {name} | {audit['proposals_by_class'].get(name, 0)} | "
            f"{_num(audit['proposals_per_image_by_class'].get(name))} | "
            f"{_num(score.get('p50'))} | {_num(score.get('p90'))} |"
        )
    if "reviewed" in audit:
        lines += [
            "",
            "## Manually inspected precision",
            "",
            "| class | tp | fp | unsure | precision | reviewed |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in CLASS_NAMES:
            value = audit["reviewed"]["per_class"][name]
            lines.append(
                f"| {name} | {value['tp']} | {value['fp']} | {value['unsure']} | "
                f"{_pct(value['precision'])} | {value['reviewed']} |"
            )
        overall = audit["reviewed"]["overall"]
        lines += [
            "",
            f"Overall inspected precision: {_pct(overall['precision'])} "
            f"({overall['tp']} TP / {overall['fp']} FP; "
            f"coverage {_pct(overall['coverage'])})",
        ]
        lines += [
            "",
            "## Per-prompt precision",
            "",
            "| class | prompt | tp | fp | precision | usable / 1000 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for name in CLASS_NAMES:
            for prompt, value in audit["reviewed"]["per_class"][name]["by_prompt"].items():
                lines.append(
                    f"| {name} | {prompt} | {value['tp']} | {value['fp']} | "
                    f"{_pct(value['precision'])} | "
                    f"{value['usable_per_1000'] if value['usable_per_1000'] is not None else '-'} |"
                )
        empty_review = audit.get("empty_review") or {}
        if empty_review:
            lines += [
                "",
                "## Reviewed empty images",
                "",
                f"- inspected: {empty_review.get('inspected', 0)}",
                f"- correctly empty: {empty_review.get('empty_correct', 0)}",
                f"- missed organs: {empty_review.get('missed_organ', 0)}",
                f"- abstention accuracy: {_pct(empty_review.get('accuracy'))}",
            ]
    return "\n".join(lines) + "\n"


def _num(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100 * float(value):.1f}%"


def write_audit(audit: Mapping[str, Any], out_dir: str | Path) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "audit.json").write_text(json.dumps(audit, indent=1))
    (target / "audit.md").write_text(render_markdown(audit))
