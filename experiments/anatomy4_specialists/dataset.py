# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Crop-manifest loading, dataset/label validation, and YOLO classification layout."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    ANATOMY_CLASSES,
    CROP_MANIFEST_SCHEMA,
    SPLITS,
    TargetSpec,
    normalize_crop_row,
    read_jsonl,
    write_jsonl,
)
from .splits import (
    apply_assignments,
    assign_splits,
    verify_isolation,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_BOX_SOURCES = {"detector", "stage1_anatomy4_yolo"}
ALLOWED_REVIEW_VERDICTS = {
    "tp",
    "accept",
    "accepted",
    "true_positive",
    "stage1_predicted",
    "unreviewed_allowed",
}


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    exclusions: dict[str, int] = field(default_factory=dict)
    isolation: dict[str, Any] = field(default_factory=dict)
    supervised_rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "counts": self.counts,
            "exclusions": self.exclusions,
            "isolation": self.isolation,
        }


def load_crop_manifest(path: str | Path) -> list[dict[str, Any]]:
    return [normalize_crop_row(row) for row in read_jsonl(path)]


def _row_label(row: Mapping[str, Any]) -> str | list[str] | None:
    label = row.get("target_label")
    if isinstance(label, list):
        return [str(v) for v in label]
    if label is None or label == "":
        return None
    return str(label)


def select_supervised(
    rows: Sequence[Mapping[str, Any]],
    target: TargetSpec,
    *,
    min_detector_confidence: float = 0.0,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Rows usable as single-label supervision, plus exclusion reasons."""

    selected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        anatomy = str(row.get("anatomy_class") or "")
        if anatomy not in ANATOMY_CLASSES:
            reasons["unknown_anatomy_class"] += 1
            continue
        if not target.anatomy_allows(anatomy):
            reasons[f"anatomy_not_in_target:{anatomy}"] += 1
            continue
        if row.get("weak_label_eligible") is False:
            reasons[f"weak_label_ineligible:{row.get('weak_label_reason')}"] += 1
            continue
        label = _row_label(row)
        if label is None:
            reasons["missing_label"] += 1
            continue
        if isinstance(label, list):
            if len(label) != 1:
                reasons["multi_label"] += 1
                continue
            label = label[0]
        if label not in target.classes:
            reasons[f"label_outside_target:{label}"] += 1
            continue
        try:
            confidence = float(row.get("detector_confidence", 0.0))
        except (TypeError, ValueError):
            reasons["invalid_detector_confidence"] += 1
            continue
        if confidence < min_detector_confidence:
            reasons["below_detector_confidence_floor"] += 1
            continue
        updated = dict(row)
        updated["target_label"] = label
        selected.append(updated)
    return selected, reasons


def validate_dataset(
    rows: Sequence[Mapping[str, Any]],
    target: TargetSpec,
    *,
    require_labels: bool = True,
    check_files: bool = False,
    require_species_disjoint: bool = False,
    min_per_class_per_split: int = 2,
    min_detector_confidence: float = 0.0,
    allow_unlabeled: bool = False,
) -> ValidationReport:
    """Validate a crop manifest against a target contract.

    Hard errors include: malformed contract fields, labels not in the target,
    a label attached to a crop whose anatomy does not match the claim, split
    isolation failures, and missing class support in train/val.
    """

    report = ValidationReport(ok=True)
    normalized = [normalize_crop_row(row) for row in rows]
    if not normalized:
        report.errors.append("empty crop manifest")
        report.ok = False
        return report

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    required_errors = 0
    for index, row in enumerate(normalized):
        where = f"row {index} ({row.get('crop_id') or 'no crop_id'})"
        for field_name in (
            "observation_id",
            "taxon_name",
            "image",
            "crop_path",
            "anatomy_class",
            "detector_confidence",
            "observation_split",
        ):
            if row.get(field_name) in (None, ""):
                report.errors.append(f"{where}: missing {field_name}")
                required_errors += 1
        observation_id = row.get("observation_id")
        if observation_id is not None and (
            not isinstance(observation_id, str)
            or not observation_id
            or observation_id != observation_id.strip()
        ):
            report.errors.append(
                f"{where}: observation_id must be a non-empty, trimmed string"
            )
            required_errors += 1
        crop_id = str(row.get("crop_id") or "")
        if crop_id:
            if crop_id in seen_ids:
                report.errors.append(f"{where}: duplicate crop_id")
            seen_ids.add(crop_id)
        crop_path = str(row.get("crop_path") or "")
        if crop_path:
            if crop_path in seen_paths:
                report.errors.append(f"{where}: duplicate crop_path")
            seen_paths.add(crop_path)
            if check_files and not Path(crop_path).is_file():
                report.errors.append(f"{where}: crop_path does not exist: {crop_path}")
            elif check_files and Path(crop_path).suffix.lower() not in IMAGE_EXTS:
                report.errors.append(f"{where}: crop_path is not an image: {crop_path}")
        anatomy = str(row.get("anatomy_class") or "")
        if anatomy and anatomy not in ANATOMY_CLASSES:
            report.errors.append(f"{where}: unknown anatomy_class {anatomy!r}")
        split = str(row.get("observation_split") or "")
        if split and split not in SPLITS:
            report.errors.append(f"{where}: invalid observation_split {split!r}")
        try:
            confidence = float(row.get("detector_confidence"))
        except (TypeError, ValueError):
            report.errors.append(f"{where}: invalid detector_confidence")
            confidence = 0.0
        if confidence < min_detector_confidence:
            report.errors.append(f"{where}: detector_confidence below floor")
        if row.get("schema") not in (None, CROP_MANIFEST_SCHEMA):
            report.warnings.append(f"{where}: unexpected schema {row.get('schema')!r}")
        box_source = row.get("detector_box_source")
        if box_source is not None and str(box_source) not in ALLOWED_BOX_SOURCES:
            report.errors.append(f"{where}: unsupported detector_box_source {box_source!r}")
        verdict = row.get("review_verdict")
        if verdict is not None and str(verdict) not in ALLOWED_REVIEW_VERDICTS:
            report.errors.append(f"{where}: unsupported review_verdict {verdict!r}")

    supervised, exclusions = select_supervised(
        normalized, target, min_detector_confidence=min_detector_confidence
    )
    report.exclusions = dict(sorted(exclusions.items()))
    report.supervised_rows = supervised

    if require_labels and exclusions:
        # Rows without a usable label are expected before a join, but a
        # dataset build must be explicit: either join first or pass
        # --allow-unlabeled to audit them as excluded.
        if not allow_unlabeled:
            for reason, count in sorted(exclusions.items()):
                report.errors.append(f"{count} row(s) excluded: {reason}")

    # Defense in depth: a weak label attached with a mismatched anatomy is a
    # contract violation even if the join was run by another tool.
    for row in normalized:
        if row.get("target_label") is None:
            continue
        anatomy = str(row.get("anatomy_class") or "")
        if not target.anatomy_allows(anatomy):
            report.errors.append(
                f"crop {row.get('crop_id')}: target_label present on anatomy "
                f"{anatomy!r}, which target {target.target_id} does not accept"
            )

    # Split isolation (the assignments are already in the manifest). If there
    # are no assignments, this reports the violation rather than inventing one.
    assignments = [str(row.get("observation_split") or "") for row in normalized]
    if all(a in SPLITS for a in assignments):
        report.isolation = verify_isolation(
            normalized, assignments, require_species_disjoint=require_species_disjoint
        )
        if not report.isolation.get("ok"):
            report.errors.append(f"split isolation failed: {report.isolation}")
    else:
        report.errors.append("split isolation cannot be checked: missing/invalid splits")

    # Class support.
    per_split_label: dict[str, Counter[str]] = defaultdict(Counter)
    for row in supervised:
        per_split_label[str(row["observation_split"])][str(row["target_label"])] += 1
    missing_classes = {
        split: [name for name in target.classes if per_split_label[split][name] == 0]
        for split in SPLITS
    }
    for split in ("train", "val"):
        if missing_classes[split]:
            report.errors.append(
                f"{split} split is missing target classes: {missing_classes[split]}"
            )
    if min_per_class_per_split > 0:
        for split in ("train", "val"):
            for name in target.classes:
                count = per_split_label[split][name]
                if 0 < count < min_per_class_per_split:
                    report.errors.append(
                        f"{split} split has {count} < {min_per_class_per_split} crops for class {name}"
                    )
    if missing_classes["test"]:
        report.warnings.append(
            f"test split is missing target classes (recall will be undefined there): {missing_classes['test']}"
        )

    report.counts = {
        "rows": len(normalized),
        "supervised_rows": len(supervised),
        "by_split": {split: sum(per_split_label[split].values()) for split in SPLITS},
        "by_split_label": {split: dict(sorted(counter.items())) for split, counter in per_split_label.items()},
        "by_anatomy": dict(sorted(Counter(str(r.get("anatomy_class")) for r in supervised).items())),
        "target": target.to_dict(),
        "n_species": len(
            {
                str(r.get("taxon_name") or "").strip().lower()
                for r in supervised
                if str(r.get("taxon_name") or "").strip()
            }
        ),
        "n_observations": len({str(r.get("observation_id")) for r in supervised}),
    }
    if required_errors:
        report.ok = False
    report.ok = not report.errors
    return report


def _link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if mode == "symlink":
        try:
            target.symlink_to(source.resolve())
            return
        except OSError:
            mode = "hardlink"
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def build_classification_dataset(
    rows: Sequence[Mapping[str, Any]],
    target: TargetSpec,
    out_dir: str | Path,
    *,
    split_mode: str = "species",
    seed: str = "anatomy4-specialists-v1",
    fractions: Mapping[str, float] | None = None,
    link_mode: str = "symlink",
    min_per_class_per_split: int = 2,
    min_detector_confidence: float = 0.0,
    check_files: bool = True,
    use_manifest_splits: bool = False,
    prefer_test_unseen: bool = False,
) -> dict[str, Any]:
    """Materialize an Ultralytics classification folder tree.

    ``use_manifest_splits=False`` (default) recomputes splits with the chosen
    splitter so isolation is guaranteed for the exact row set. Set it to True
    only when the manifest splits were produced by the same splitter.
    """

    if use_manifest_splits and prefer_test_unseen:
        raise ValueError(
            "--holdout-prefer-stage1-unseen requires split recomputation; "
            "do not combine it with --use-manifest-splits"
        )
    supervised, exclusions = select_supervised(
        rows, target, min_detector_confidence=min_detector_confidence
    )
    if not supervised:
        raise ValueError(f"no supervised rows for target {target.target_id}")

    if use_manifest_splits:
        missing = [row for row in supervised if str(row.get("observation_split") or "") not in SPLITS]
        if missing:
            raise ValueError(f"{len(missing)} row(s) missing observation_split")
        assignments = [str(row["observation_split"]) for row in supervised]
        split_report = {
            "mode": "manifest",
            "assignments": assignments,
            "isolation": verify_isolation(
                supervised,
                assignments,
                require_species_disjoint=split_mode in {"species", "species_then_observation"},
            ),
            "counts": {},
        }
    else:
        split_report = assign_splits(
            supervised,
            mode=split_mode,
            seed=seed,
            fractions=fractions,
            required_classes=target.classes,
            required_min_per_class=min_per_class_per_split,
            prefer_test_unseen=prefer_test_unseen,
        )
        assignments = split_report["assignments"]
    if not split_report["isolation"].get("ok"):
        raise ValueError(f"split isolation failed: {split_report['isolation']}")
    supervised = apply_assignments(supervised, assignments)

    validation = validate_dataset(
        supervised,
        target,
        require_labels=True,
        check_files=check_files,
        require_species_disjoint=split_mode in {"species", "species_then_observation"},
        min_per_class_per_split=min_per_class_per_split,
        min_detector_confidence=min_detector_confidence,
    )
    if not validation.ok:
        raise ValueError("dataset validation failed:\n" + "\n".join(validation.errors))

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for row in supervised:
        split = str(row["observation_split"])
        label = str(row["target_label"])
        source = Path(str(row["crop_path"]))
        target_path = output / split / label / f"{row.get('crop_id') or source.stem}{source.suffix.lower()}"
        _link_or_copy(source, target_path, link_mode)
        counts[split][label] += 1

    labels = {
        "schema": "arq.anatomy4_specialists.labels/v1",
        "target": target.to_dict(),
        "classes": list(target.classes),
        "label_to_index": target.label_to_index,
        "index_to_label": {str(k): v for k, v in target.index_to_label.items()},
        "num_classes": len(target.classes),
        "abstain": {
            "class": "abstain",
            "rule": "max softmax probability below the validation-selected threshold",
        },
    }
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    write_jsonl(output / "manifest.jsonl", supervised)
    summary = {
        "schema": "arq.anatomy4_specialists.dataset/v1",
        "target": target.to_dict(),
        "split_mode": split_mode,
        "prefer_test_unseen": prefer_test_unseen,
        "split_counts": split_report.get("counts"),
        "stage1_overlap_counts": split_report.get("stage1_overlap_counts"),
        "isolation": split_report.get("isolation"),
        "counts": {
            "total": len(supervised),
            "by_split": {split: sum(counter.values()) for split, counter in counts.items()},
            "by_split_label": {split: dict(sorted(counter.items())) for split, counter in counts.items()},
            "excluded": dict(sorted(exclusions.items())),
        },
        "validation": validation.to_dict(),
    }
    (output / "counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "validation.json").write_text(json.dumps(validation.to_dict(), indent=2) + "\n")
    return summary
