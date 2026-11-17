# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Validate and prepare a materialized anatomy4 YOLO dataset.

On-disk contract (``anatomy4-v1``)
----------------------------------
::

    <root>/
      dataset.json                 # optional but recommended (authoritative)
      images/<split>/<stem>.<ext>
      labels/<split>/<stem>.txt    # one line per box, normalized xyxy
      split_map.json               # optional: {"<image relpath or stem>": "<obs id>"}

Rules enforced by :func:`validate_dataset`:

* class ids are exactly ``0..3`` in the contract order;
* every image has a label file (an explicit empty file is a negative image);
* boxes are normalized xyxy with ``0 <= x0 < x1 <= 1`` and ``0 <= y0 < y1 <= 1``;
* ``dataset.json`` entries agree with disk (path, split, negative flag);
* no observation id appears in more than one split (leakage check);
* per-class support is reported per split, never estimated.

Nothing here writes placeholder metrics: missing support is reported as zero,
and split isolation that cannot be checked is reported as ``unverifiable``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import (
    CLASS_NAMES,
    CONTRACT,
    IMAGE_SUFFIXES,
    SPLITS,
    TEST_ONLY_MARKER,
    Box,
    ContractError,
    observation_leaks,
    parse_label_text,
    xyxy_to_cxcywh,
)

DATASET_SCHEMA = "arq.anatomy4.dataset/v1"


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.path:
            payload["path"] = self.path
        return payload


@dataclass
class ImageEntry:
    relative_path: str
    split: str
    observation_id: str | None
    negative: bool
    width: int | None
    height: int | None
    boxes: list[Box] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.relative_path,
            "split": self.split,
            "observation_id": self.observation_id,
            "negative": self.negative,
            "width": self.width,
            "height": self.height,
            "boxes": [
                {
                    "class_id": box.class_id,
                    "class_name": box.class_name,
                    "x0": box.x0,
                    "y0": box.y0,
                    "x1": box.x1,
                    "y1": box.y1,
                }
                for box in self.boxes
            ],
        }


def _suffix_ok(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def _scan_images(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        return []
    return sorted(
        (path for path in split_dir.iterdir() if path.is_file() and _suffix_ok(path)),
        key=lambda path: path.name,
    )


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image  # local import: keeps module import cheap

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_dataset_metadata(root: Path) -> dict[str, Any] | None:
    path = root / "dataset.json"
    if not path.is_file():
        return None
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ContractError("dataset.json must contain an object")
    return data


def load_observation_map(root: Path) -> dict[str, str] | None:
    path = root / "split_map.json"
    if not path.is_file():
        return None
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ContractError("split_map.json must contain an object")
    return {str(key): str(value) for key, value in data.items()}


def _observation_for(
    relative_path: str,
    stem: str,
    metadata_entry: Mapping[str, Any] | None,
    observation_map: Mapping[str, str] | None,
) -> str | None:
    if metadata_entry and metadata_entry.get("observation_id") is not None:
        return str(metadata_entry["observation_id"])
    if observation_map:
        for key in (relative_path, stem, Path(relative_path).name, stem):
            if key in observation_map:
                return observation_map[key]
    return None


def validate_dataset(
    root: str | Path,
    *,
    require_test: bool = True,
    require_val: bool = True,
    observation_map: Mapping[str, str] | None = None,
    class_warn_support: int = 5,
    labels_sha256: bool = True,
) -> dict[str, Any]:
    """Return a dataset audit dictionary. Never raises for dataset defects."""

    dataset_root = Path(root).expanduser().resolve()
    errors: list[Issue] = []
    warnings: list[Issue] = []
    entries: list[ImageEntry] = []

    def error(code: str, message: str, path: str | None = None) -> None:
        errors.append(Issue("error", code, message, path))

    def warn(code: str, message: str, path: str | None = None) -> None:
        warnings.append(Issue("warning", code, message, path))

    if not dataset_root.is_dir():
        error("dataset_missing", f"dataset root does not exist: {dataset_root}")
        return _empty_audit(dataset_root, errors, warnings)

    try:
        metadata = load_dataset_metadata(dataset_root)
    except (ContractError, json.JSONDecodeError) as exc:
        error("dataset_json_invalid", str(exc), "dataset.json")
        metadata = None

    explicit_map = observation_map
    if explicit_map is None:
        try:
            explicit_map = load_observation_map(dataset_root)
        except (ContractError, json.JSONDecodeError) as exc:
            error("split_map_invalid", str(exc), "split_map.json")
            explicit_map = None

    marker_path = dataset_root / TEST_ONLY_MARKER
    marker: dict[str, Any] | None = None
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text())
        except json.JSONDecodeError as exc:
            error("test_only_marker_invalid", str(exc), TEST_ONLY_MARKER)
            marker = None
        if marker is not None:
            if marker.get("contract") not in (None, CONTRACT):
                error(
                    "test_only_marker_contract",
                    f"marker contract {marker.get('contract')!r} != {CONTRACT!r}",
                    TEST_ONLY_MARKER,
                )
            if marker.get("allow_train") is True:
                error(
                    "test_only_marker_trainable",
                    "FROZEN_TEST_ONLY.json must not allow training",
                    TEST_ONLY_MARKER,
                )
    test_only = bool((metadata or {}).get("test_only")) or marker is not None
    if test_only:
        require_val = False
        if marker is None:
            warn(
                "test_only_without_marker",
                "dataset declares test_only without FROZEN_TEST_ONLY.json",
                "dataset.json",
            )

    declared_classes: list[str] = list(CLASS_NAMES)
    declared_contract = CONTRACT
    metadata_images: dict[str, Mapping[str, Any]] = {}
    if metadata is not None:
        declared_contract = str(metadata.get("contract", CONTRACT))
        if declared_contract != CONTRACT:
            error(
                "contract_mismatch",
                f"dataset.json contract {declared_contract!r} != {CONTRACT!r}",
                "dataset.json",
            )
        raw_classes = metadata.get("classes", list(CLASS_NAMES))
        declared_classes = [str(name) for name in raw_classes]
        if tuple(declared_classes) != CLASS_NAMES:
            error(
                "class_order_mismatch",
                f"dataset.json classes {declared_classes} != {list(CLASS_NAMES)}",
                "dataset.json",
            )
        raw_images = metadata.get("images")
        if raw_images is None:
            error("dataset_json_images_missing", "dataset.json has no images[]", "dataset.json")
        elif not isinstance(raw_images, list):
            error("dataset_json_images_invalid", "images must be a list", "dataset.json")
        else:
            for item in raw_images:
                if not isinstance(item, dict) or not item.get("image"):
                    error(
                        "dataset_json_entry_invalid",
                        f"image entry needs an 'image' field: {item!r}",
                        "dataset.json",
                    )
                    continue
                relative = str(item["image"])
                if relative in metadata_images:
                    error(
                        "dataset_json_duplicate",
                        f"duplicate image entry {relative}",
                        "dataset.json",
                    )
                    continue
                metadata_images[relative] = item

    seen_stems: dict[str, str] = {}
    split_observations: dict[str, set[str]] = {split: set() for split in SPLITS}
    label_hash = hashlib.sha256()

    for split in SPLITS:
        split_dir = dataset_root / "images" / split
        if not split_dir.is_dir():
            if split == "test" and require_test:
                error("split_missing", "test split directory is missing", f"images/{split}")
            elif split == "val" and require_val:
                error("split_missing", "val split directory is missing", f"images/{split}")
            else:
                warn("split_empty", f"no images/{split} directory", f"images/{split}")
            continue
        images = _scan_images(split_dir)
        if not images:
            if split == "test" and require_test:
                error("split_empty", "test split has no images", f"images/{split}")
            elif split == "val" and require_val:
                error("split_empty", "val split has no images", f"images/{split}")
        for image_path in images:
            relative = image_path.relative_to(dataset_root).as_posix()
            stem = image_path.stem
            if stem in seen_stems:
                warn(
                    "duplicate_image_stem",
                    f"stem {stem!r} also appears at {seen_stems[stem]}",
                    relative,
                )
            else:
                seen_stems[stem] = relative

            label_path = dataset_root / "labels" / split / f"{stem}.txt"
            label_relative = label_path.relative_to(dataset_root).as_posix()
            boxes: list[Box] = []
            label_ok = True
            if not label_path.is_file():
                error(
                    "missing_label_file",
                    (
                        "no label file; negatives must have an explicit empty "
                        f"labels/{split}/{stem}.txt"
                    ),
                    label_relative,
                )
                label_ok = False
            else:
                label_text = label_path.read_text()
                if labels_sha256:
                    label_hash.update(relative.encode())
                    label_hash.update(label_text.encode())
                try:
                    boxes = parse_label_text(label_text, source=label_relative)
                except ContractError as exc:
                    error("label_invalid", str(exc), label_relative)
                    boxes = []
                    label_ok = False

            width, height = _image_size(image_path)
            if width is None:
                warn("image_unreadable", "could not read image dimensions", relative)

            metadata_entry = metadata_images.get(relative)
            if metadata is not None and metadata_entry is None:
                error(
                    "image_not_in_dataset_json",
                    "image on disk is not declared in dataset.json",
                    relative,
                )
            declared_negative = bool(metadata_entry.get("negative")) if metadata_entry else False
            # A negative is an explicitly empty label file. A malformed label
            # is NOT a negative: counting it as one would inflate the negative
            # rate and understate the annotation failure.
            actual_negative = label_ok and not boxes
            if declared_negative and not actual_negative:
                error(
                    "negative_has_boxes",
                    "dataset.json marks this image negative but the label file is not empty",
                    label_relative,
                )
            if metadata_entry is not None and not declared_negative and actual_negative:
                warn(
                    "undeclared_negative",
                    "empty label file is not marked negative in dataset.json",
                    label_relative,
                )

            observation_id = _observation_for(
                relative, stem, metadata_entry, explicit_map
            )
            if observation_id is None:
                warn(
                    "observation_id_missing",
                    "no observation id; split isolation cannot be verified for this image",
                    relative,
                )
            else:
                split_observations[split].add(observation_id)

            exact: dict[tuple[int, float, float, float, float], int] = {}
            for box in boxes:
                key = (box.class_id, round(box.x0, 6), round(box.y0, 6), round(box.x1, 6), round(box.y1, 6))
                exact[key] = exact.get(key, 0) + 1
                if box.area < 1e-5:
                    warn("degenerate_box", f"box area {box.area:.2e} is tiny", label_relative)
            duplicates = [key for key, count in exact.items() if count > 1]
            if duplicates:
                warn(
                    "duplicate_boxes",
                    f"{len(duplicates)} exactly duplicated box(es)",
                    label_relative,
                )

            entries.append(
                ImageEntry(
                    relative_path=relative,
                    split=split,
                    observation_id=observation_id,
                    negative=actual_negative,
                    width=width,
                    height=height,
                    boxes=boxes,
                )
            )

    if metadata is not None:
        disk = {entry.relative_path for entry in entries}
        for declared in metadata_images:
            if declared not in disk:
                error(
                    "declared_image_missing",
                    "dataset.json declares an image that is not on disk",
                    declared,
                )

    for split in SPLITS:
        label_dir = dataset_root / "labels" / split
        if not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            image_exists = any(
                (dataset_root / "images" / split / f"{label_path.stem}{suffix}").is_file()
                for suffix in IMAGE_SUFFIXES
            )
            if not image_exists:
                error(
                    "orphan_label_file",
                    "label file has no matching image",
                    label_path.relative_to(dataset_root).as_posix(),
                )

    if test_only:
        for split in ("train", "val"):
            offenders = _scan_images(dataset_root / "images" / split)
            if offenders:
                error(
                    "test_only_has_" + split,
                    f"test-only dataset contains {len(offenders)} image(s) in {split}/",
                    f"images/{split}",
                )
        warn(
            "test_only_dataset",
            "frozen test-only dataset; only images/test may be used",
        )

    leaks = observation_leaks(split_observations)
    for observation, splits in sorted(leaks.items()):
        error(
            "observation_leak",
            f"observation {observation!r} appears in splits {splits}",
        )

    with_observations = any(split_observations[split] for split in SPLITS)
    missing_observations = any(entry.observation_id is None for entry in entries)
    if not with_observations:
        isolation = "unverifiable"
    elif leaks:
        isolation = "leaked"
    elif missing_observations:
        isolation = "partial"
    else:
        isolation = "clean"

    stats = _split_stats(entries)
    overall_support = {
        name: sum(stats[split]["boxes_per_class"].get(name, 0) for split in stats)
        for name in CLASS_NAMES
    }
    test_support = stats.get("test", {}).get("boxes_per_class", {})

    if require_test:
        for name in CLASS_NAMES:
            if test_support.get(name, 0) < class_warn_support:
                warn(
                    "low_test_support",
                    f"test support for {name} is {test_support.get(name, 0)} "
                    f"(< {class_warn_support}); per-class test metrics will be weak",
                )

    total_images = len(entries)
    negatives = sum(1 for entry in entries if entry.negative)
    if total_images and negatives == 0:
        warn(
            "no_negatives",
            "dataset has no negative (empty-label) images; false-positive rate is unmeasurable",
        )

    audit: dict[str, Any] = {
        "schema": "arq.anatomy4.dataset-audit/v1",
        "contract": CONTRACT,
        "declared_contract": declared_contract,
        "dataset_root": str(dataset_root),
        "classes": list(CLASS_NAMES),
        "declared_classes": declared_classes,
        "status": "invalid" if errors else "ok",
        "blocking": bool(errors),
        "test_only": test_only,
        "frozen_marker": marker,
        "split_isolation": isolation,
        "has_observation_map": bool(explicit_map) or any(
            entry.observation_id is not None for entry in entries
        ),
        "totals": {
            "images": total_images,
            "negatives": negatives,
            "boxes": sum(len(entry.boxes) for entry in entries),
            "observations": len(
                {entry.observation_id for entry in entries if entry.observation_id}
            ),
            "boxes_per_class": overall_support,
            "negative_rate": (negatives / total_images) if total_images else None,
        },
        "splits": stats,
        "support": {
            "boxes_per_class": overall_support,
            "test_boxes_per_class": {
                name: test_support.get(name, 0) for name in CLASS_NAMES
            },
        },
        "images": [entry.to_dict() for entry in entries],
        "observations_by_split": {
            split: sorted(values) for split, values in split_observations.items()
        },
        "errors": [issue.to_dict() for issue in errors],
        "warnings": [issue.to_dict() for issue in warnings],
        "labels_sha256": label_hash.hexdigest() if labels_sha256 else None,
    }
    return audit


def _split_stats(entries: Iterable[ImageEntry]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    grouped: dict[str, list[ImageEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.split].append(entry)
    for split in SPLITS:
        items = grouped.get(split, [])
        boxes_per_class: dict[str, int] = {name: 0 for name in CLASS_NAMES}
        images_per_class: dict[str, int] = {name: 0 for name in CLASS_NAMES}
        for entry in items:
            present: set[int] = set()
            for box in entry.boxes:
                boxes_per_class[box.class_name] += 1
                present.add(box.class_id)
            for class_id in present:
                images_per_class[CLASS_NAMES[class_id]] += 1
        observations = {entry.observation_id for entry in items if entry.observation_id}
        stats[split] = {
            "images": len(items),
            "negatives": sum(1 for entry in items if entry.negative),
            "boxes": sum(len(entry.boxes) for entry in items),
            "observations": len(observations),
            "boxes_per_class": boxes_per_class,
            "images_per_class": images_per_class,
        }
    return stats


def _empty_audit(root: Path, errors: list[Issue], warnings: list[Issue]) -> dict[str, Any]:
    return {
        "schema": "arq.anatomy4.dataset-audit/v1",
        "contract": CONTRACT,
        "dataset_root": str(root),
        "classes": list(CLASS_NAMES),
        "status": "invalid",
        "blocking": True,
        "test_only": False,
        "frozen_marker": None,
        "split_isolation": "unverifiable",
        "totals": {
            "images": 0,
            "negatives": 0,
            "boxes": 0,
            "observations": 0,
            "boxes_per_class": {name: 0 for name in CLASS_NAMES},
            "negative_rate": None,
        },
        "splits": {split: {"images": 0, "negatives": 0, "boxes": 0, "observations": 0,
                           "boxes_per_class": {name: 0 for name in CLASS_NAMES},
                           "images_per_class": {name: 0 for name in CLASS_NAMES}}
                   for split in SPLITS},
        "support": {
            "boxes_per_class": {name: 0 for name in CLASS_NAMES},
            "test_boxes_per_class": {name: 0 for name in CLASS_NAMES},
        },
        "images": [],
        "observations_by_split": {split: [] for split in SPLITS},
        "errors": [issue.to_dict() for issue in errors],
        "warnings": [issue.to_dict() for issue in warnings],
        "labels_sha256": None,
    }


def dataset_identity(root: str | Path) -> dict[str, Any]:
    """Identity sets for split-isolation checks between two datasets.

    Uses dataset.json because the audit omits source SHA-256 values. Returns
    counts and the sets so callers can report or hard-fail on overlap.
    """

    dataset_root = Path(root).expanduser().resolve()
    path = dataset_root / "dataset.json"
    if not path.is_file():
        raise ContractError(f"dataset.json not found under {dataset_root}")
    doc = json.loads(path.read_text())
    observations: set[str] = set()
    photos: set[str] = set()
    sha256: set[str] = set()
    for entry in doc.get("images") or []:
        if entry.get("observation_id"):
            observations.add(str(entry["observation_id"]))
        photo_id = entry.get("photo_id")
        relative = str(entry.get("image") or "")
        photos.add(str(photo_id) if photo_id else Path(relative).stem)
        source = entry.get("source") if isinstance(entry.get("source"), Mapping) else {}
        digest = (
            entry.get("source_sha256")
            or entry.get("sha256")
            or source.get("source_sha256")
            or source.get("sha256")
        )
        if digest:
            sha256.add(str(digest))
    return {
        "root": str(dataset_root),
        "observation_ids": observations,
        "photo_ids": photos,
        "sha256": sha256,
        "counts": {
            "observations": len(observations),
            "photos": len(photos),
            "sha256": len(sha256),
        },
    }


def assert_disjoint_datasets(
    first: str | Path,
    second: str | Path,
) -> dict[str, Any]:
    """Raise ContractError when two datasets share observations/photos/bytes."""

    left = dataset_identity(first)
    right = dataset_identity(second)
    overlaps = {
        "observation_ids": sorted(left["observation_ids"] & right["observation_ids"]),
        "photo_ids": sorted(left["photo_ids"] & right["photo_ids"]),
        "sha256": sorted(left["sha256"] & right["sha256"]),
    }
    total = sum(len(values) for values in overlaps.values())
    report = {
        "first": left["root"],
        "second": right["root"],
        "counts": {"first": left["counts"], "second": right["counts"]},
        "overlaps": {key: values[:20] for key, values in overlaps.items()},
        "overlap_counts": {key: len(values) for key, values in overlaps.items()},
        "total_overlap": total,
    }
    if total:
        raise ContractError(
            "dataset isolation violated: "
            + ", ".join(
                f"{len(values)} shared {key}" for key, values in overlaps.items() if values
            )
            + f" (first: {left['root']}, second: {right['root']})"
        )
    return report


def write_audit(audit: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=1) + "\n")
    return target


def render_audit_markdown(audit: Mapping[str, Any]) -> str:
    totals = audit["totals"]
    lines = [
        "# anatomy4 dataset audit",
        "",
        f"- contract: {audit['contract']}",
        f"- status: **{audit['status']}**",
        f"- split isolation: {audit['split_isolation']}",
        f"- images: {totals['images']} ({totals['negatives']} negative)",
        f"- boxes: {totals['boxes']}",
        f"- observations: {totals['observations']}",
        "",
        "| split | images | negatives | boxes | observations | "
        + " | ".join(CLASS_NAMES)
        + " |",
        "| --- | ---: | ---: | ---: | ---: | "
        + " | ".join("---:" for _ in CLASS_NAMES)
        + " |",
    ]
    for split in SPLITS:
        stats = audit["splits"][split]
        counts = " | ".join(
            str(stats["boxes_per_class"].get(name, 0)) for name in CLASS_NAMES
        )
        lines.append(
            f"| {split} | {stats['images']} | {stats['negatives']} | "
            f"{stats['boxes']} | {stats['observations']} | {counts} |"
        )
    lines += ["", "## Support (boxes)", ""]
    for name in CLASS_NAMES:
        lines.append(
            f"- {name}: {audit['support']['boxes_per_class'].get(name, 0)} total, "
            f"{audit['support']['test_boxes_per_class'].get(name, 0)} test"
        )
    if audit["errors"]:
        lines += ["", "## Errors", ""]
        lines += [f"- `{item['code']}` {item['message']}" for item in audit["errors"]]
    if audit["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- `{item['code']}` {item['message']}" for item in audit["warnings"]]
    if audit["labels_sha256"]:
        lines += ["", f"labels sha256: `{audit['labels_sha256']}`"]
    return "\n".join(lines) + "\n"


def _link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "symlink":
        os.symlink(source, target)
    elif mode == "hardlink":
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"unknown link mode {mode!r}")


def prepare_ultralytics(
    audit: Mapping[str, Any],
    out_dir: str | Path,
    *,
    link_mode: str = "symlink",
    allow_test_only: bool = False,
    oversample_class: str | None = None,
    oversample_factor: int = 1,
) -> dict[str, Any]:
    """Write the Ultralytics view (cx cy w h) from the audited xyxy dataset.

    Frozen test-only datasets are refused by default: they exist to evaluate a
    checkpoint, never to train one. ``allow_test_only`` is for the evaluator.
    """

    if audit.get("blocking"):
        errors = ", ".join(item["code"] for item in audit.get("errors", []))
        raise ContractError(f"refusing to prepare an invalid dataset: {errors}")
    test_only = bool(audit.get("test_only"))
    if test_only and not allow_test_only:
        raise ContractError(
            "frozen test-only dataset: refusing to prepare for training "
            "(evaluation may pass allow_test_only=True)"
        )
    required_splits = ("test",) if test_only else ("train", "val")
    for split in required_splits:
        stats = audit["splits"].get(split, {})
        if not stats.get("images"):
            raise ContractError(f"{split} split is empty; cannot prepare")

    source_root = Path(audit["dataset_root"])
    target_root = Path(out_dir).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for entry in audit["images"]:
        split = entry["split"]
        source_image = source_root / entry["image"]
        stem = source_image.stem
        _link_or_copy(
            source_image,
            target_root / "images" / split / source_image.name,
            link_mode,
        )
        source_label = source_root / "labels" / split / f"{stem}.txt"
        target_label = target_root / "labels" / split / f"{stem}.txt"
        target_label.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                box["class_id"], *xyxy_to_cxcywh(
                    (box["x0"], box["y0"], box["x1"], box["y1"])
                )
            )
            for box in entry["boxes"]
        ]
        # Explicit empty file for negatives: never omit the file.
        target_label.write_text(("\n".join(lines) + "\n") if lines else "")
        if not source_label.is_file():
            raise ContractError(f"label disappeared during prepare: {source_label}")

    oversample_info = None
    if oversample_class is not None and int(oversample_factor) > 1:
        class_name = str(oversample_class).strip().lower()
        if class_name not in CLASS_NAMES:
            raise ContractError(f"unknown oversample class {class_name!r}")
        if test_only:
            raise ContractError("cannot oversample a frozen test-only dataset")
        extra = 0
        train_dir = target_root / "images" / "train"
        label_dir = target_root / "labels" / "train"
        # Remove repeats from a prior prepare so reruns are deterministic.
        for directory in (train_dir, label_dir):
            if directory.is_dir():
                for entry in directory.iterdir():
                    if entry.is_file() and "__rep" in entry.stem:
                        entry.unlink()
        for entry in audit["images"]:
            if entry["split"] != "train":
                continue
            if not any(box["class_name"] == class_name for box in entry["boxes"]):
                continue
            source_image = source_root / entry["image"]
            suffix = source_image.suffix.lower()
            stem = source_image.stem
            source_label = label_dir / f"{stem}.txt"
            if not source_label.is_file():
                raise ContractError(f"missing prepared label for {stem}")
            for repeat in range(1, int(oversample_factor)):
                target_image = train_dir / f"{stem}__rep{repeat}{suffix}"
                target_label = label_dir / f"{stem}__rep{repeat}.txt"
                if not target_image.exists():
                    _link_or_copy(source_image, target_image, link_mode)
                    target_label.write_text(source_label.read_text())
                    extra += 1
        oversample_info = {
            "class": class_name,
            "factor": int(oversample_factor),
            "source_images": sum(
                1
                for entry in audit["images"]
                if entry["split"] == "train"
                and any(box["class_name"] == class_name for box in entry["boxes"])
            ),
            "extra_train_images": extra,
            "train_only": True,
        }

    data_yaml = target_root / "data.yaml"
    yaml_lines = [f"path: {target_root}"]
    if test_only:
        yaml_lines += [
            "test: images/test",
            "# FROZEN TEST-ONLY DATASET: train/val intentionally omitted.",
        ]
    else:
        yaml_lines += ["train: images/train", "val: images/val", "test: images/test"]
    yaml_lines += [f"names: {json.dumps(list(CLASS_NAMES))}", ""]
    data_yaml.write_text("\n".join(yaml_lines))
    prepared = {
        "schema": "arq.anatomy4.prepared/v1",
        "contract": CONTRACT,
        "source_root": str(source_root),
        "root": str(target_root),
        "data_yaml": str(data_yaml),
        "link_mode": link_mode,
        "classes": list(CLASS_NAMES),
        "splits": {
            split: audit["splits"][split]["images"] for split in SPLITS
        },
        "label_format": "cx cy w h (normalized)",
        "source_label_format": "x0 y0 x1 y1 (normalized)",
        "test_only": test_only,
        "oversample": oversample_info,
        "dataset_audit_sha256": hashlib.sha256(
            json.dumps(audit, sort_keys=True).encode()
        ).hexdigest(),
    }
    if oversample_info:
        prepared["splits"]["train"] = (
            audit["splits"]["train"]["images"]
            + oversample_info["extra_train_images"]
        )
    (target_root / "anatomy4_prepared.json").write_text(
        json.dumps(prepared, indent=1) + "\n"
    )
    return prepared
