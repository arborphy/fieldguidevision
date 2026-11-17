# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Import the FROZEN anatomy4 blind benchmark as a test-only eval dataset.

Source (curator-reviewed truth, all rows ``split=test``):

* ``annotations.jsonl`` — one row per image: class/bbox labels, ``negative``,
  ``needs_review``, observation id, reviewer provenance.
* ``manifest.jsonl`` — image paths, sha256, batch, observation id.
* optional ``audit.json`` — the freeze record (SHA-256s + counts) to verify.

This is a different artifact from the DINO proposal/review flow: there is no
proposal id and no review CSV. Boxes are copied exactly (normalized xyxy) into
``labels/test/<review_image_id>.txt``; negatives get explicit empty files.

Safety:

* only ``split=test`` rows are accepted; anything else aborts;
* the output is marked ``test_only`` in ``dataset.json`` and by
  ``FROZEN_TEST_ONLY.json``; ``prepare_ultralytics`` and the training CLI refuse
  it unless an evaluator explicitly asks for the test-only view;
* ``needs_review`` rows are excluded from truth by default (their annotations
  are preserved in ``excluded/needs_review.jsonl``); ``--include-needs-review``
  includes them but keeps the flag in ``dataset.json``;
* observations remain test-only multi-photo groups (no train/val), so no
  observation-safety question arises; ids are preserved verbatim.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    CLASS_IDS,
    CLASS_NAMES,
    CONTRACT,
    TEST_ONLY_MARKER,
    ContractError,
)
from .dataset import render_audit_markdown, validate_dataset, write_audit
from .render import CLASS_COLORS

BLIND_SCHEMA = "arq.anatomy4.blind-import/v1"
FROZEN_SCHEMA = "arq.anatomy4.frozen-test-only/v1"


class BlindImportError(RuntimeError):
    """The frozen benchmark is inconsistent or cannot be materialized."""


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if not source.is_file():
        raise BlindImportError(f"input not found: {source}")
    for number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BlindImportError(f"{source}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise BlindImportError(f"{source}:{number}: row must be an object")
        rows.append(row)
    return rows


def _validate_bbox(value: Any, *, where: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise BlindImportError(f"{where}: bbox must be 4 values, got {value!r}")
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise BlindImportError(f"{where}: bbox values must be numbers") from exc
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise BlindImportError(
            f"{where}: bbox is not normalized xyxy with x0<x1, y0<y1 in [0,1]: {value!r}"
        )
    return (x0, y0, x1, y1)


def _validate_label(raw: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise BlindImportError(f"{where}: label must be an object")
    class_name = str(raw.get("class") or raw.get("class_name") or "").strip().lower()
    if class_name not in CLASS_IDS:
        raise BlindImportError(f"{where}: class {class_name!r} not in {list(CLASS_NAMES)}")
    bbox = _validate_bbox(raw.get("bbox", raw.get("bbox_xyxy")), where=where)
    review_state = str(raw.get("review_state") or "").strip().lower()
    if review_state not in {"accepted", "confirmed"}:
        raise BlindImportError(
            f"{where}: label review_state {review_state!r} is not accepted/confirmed"
        )
    source = str(raw.get("source") or "").strip().lower()
    if not source or any(
        marker in source for marker in ("fallback", "synthetic", "heuristic", "mock", "fake")
    ):
        raise BlindImportError(f"{where}: invalid label provenance source {source!r}")
    return {
        "class": class_name,
        "bbox": bbox,
        "review_state": review_state,
        "source": source,
        "provenance": dict(raw.get("provenance") or {}),
        "raw": dict(raw),
    }


def load_benchmark(
    annotations_path: str | Path,
    manifest_path: str | Path,
    *,
    frozen_audit_path: str | Path | None = None,
    verify_frozen: bool = True,
) -> dict[str, Any]:
    annotations = _read_jsonl(annotations_path)
    manifest_rows = _read_jsonl(manifest_path)
    manifest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(manifest_rows, start=1):
        image_id = str(row.get("review_image_id") or "").strip()
        if not image_id:
            raise BlindImportError(f"manifest row {index} has no review_image_id")
        if image_id in manifest:
            raise BlindImportError(f"duplicate manifest review_image_id {image_id!r}")
        if row.get("split") != "test":
            raise BlindImportError(
                f"manifest {image_id}: split {row.get('split')!r} is not 'test'; "
                "the blind benchmark is frozen test-only"
            )
        manifest[image_id] = row
    if not annotations:
        raise BlindImportError("annotations.jsonl is empty")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(annotations, start=1):
        image_id = str(row.get("review_image_id") or "").strip()
        if not image_id:
            raise BlindImportError(f"annotation row {index} has no review_image_id")
        if image_id in seen:
            raise BlindImportError(f"duplicate annotation review_image_id {image_id!r}")
        seen.add(image_id)
        manifest_row = manifest.get(image_id)
        if manifest_row is None:
            raise BlindImportError(f"annotation {image_id} has no manifest row")
        if row.get("split") != "test":
            raise BlindImportError(
                f"annotation {image_id}: split {row.get('split')!r} is not 'test'"
            )
        if str(row.get("observation_id") or "") != str(
            manifest_row.get("observation_id") or ""
        ):
            raise BlindImportError(
                f"annotation {image_id}: observation_id disagrees with the manifest"
            )
        if str(row.get("batch") or "") != str(manifest_row.get("batch") or ""):
            raise BlindImportError(f"annotation {image_id}: batch disagrees with the manifest")

        negative = row.get("negative") is True
        needs_review = row.get("needs_review") is True
        labels = [
            _validate_label(raw, where=f"{image_id}:label[{position}]")
            for position, raw in enumerate(row.get("labels") or [])
        ]
        if negative and labels:
            raise BlindImportError(f"annotation {image_id}: negative row carries labels")
        if not negative and not labels:
            raise BlindImportError(
                f"annotation {image_id}: no labels and not marked negative"
            )
        exact = Counter(
            (
                CLASS_IDS[label["class"]],
                *(round(value, 6) for value in label["bbox"]),
            )
            for label in labels
        )
        duplicates = [key for key, count in exact.items() if count > 1]
        if duplicates:
            raise BlindImportError(
                f"annotation {image_id}: {len(duplicates)} exact duplicate box(es)"
            )
        records.append(
            {
                "review_image_id": image_id,
                "observation_id": str(row["observation_id"]),
                "batch": str(row.get("batch") or ""),
                "negative": negative,
                "needs_review": needs_review,
                "needs_review_note": row.get("needs_review_note"),
                "review_state": str(row.get("review_state") or ""),
                "source": str(row.get("source") or ""),
                "provenance": dict(row.get("provenance") or {}),
                "photo_url": row.get("photo_url") or manifest_row.get("photo_url"),
                "manifest": manifest_row,
                "labels": labels,
            }
        )
    missing = sorted(set(manifest) - seen)
    if missing:
        raise BlindImportError(
            f"{len(missing)} manifest image(s) have no annotation row: {missing[:5]}"
        )

    source_counts = _counts(records)
    frozen = None
    audit_path = Path(frozen_audit_path) if frozen_audit_path else Path(annotations_path).parent / "audit.json"
    if audit_path.is_file():
        frozen = json.loads(audit_path.read_text())
        if verify_frozen:
            _verify_frozen(
                frozen,
                annotations_path=annotations_path,
                manifest_path=manifest_path,
                counts=source_counts,
            )
    elif verify_frozen:
        raise BlindImportError(
            f"freeze audit not found at {audit_path}; pass --frozen-audit or "
            "--no-verify-frozen"
        )
    return {
        "records": records,
        "counts": source_counts,
        "frozen_audit": frozen,
        "annotations_sha256": _sha256(annotations_path),
        "manifest_sha256": _sha256(manifest_path),
        "annotations_path": str(Path(annotations_path).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "frozen_audit_path": str(audit_path.resolve()) if frozen else None,
    }


def _counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    boxes_per_class = {name: 0 for name in CLASS_NAMES}
    images_per_class = {name: 0 for name in CLASS_NAMES}
    for record in records:
        present = set()
        for label in record["labels"]:
            boxes_per_class[label["class"]] += 1
            present.add(label["class"])
        for name in present:
            images_per_class[name] += 1
    return {
        "rows": len(records),
        "positive_rows": sum(1 for record in records if not record["negative"]),
        "negative_rows": sum(1 for record in records if record["negative"]),
        "needs_review_rows": sum(1 for record in records if record["needs_review"]),
        "boxes": sum(boxes_per_class.values()),
        "boxes_per_class": boxes_per_class,
        "images_per_class": images_per_class,
        "observations": len({record["observation_id"] for record in records}),
        "multi_photo_observations": sum(
            1 for count in Counter(record["observation_id"] for record in records).values() if count > 1
        ),
        "batches": dict(sorted(Counter(record["batch"] for record in records).items())),
    }


def _verify_frozen(
    frozen: Mapping[str, Any],
    *,
    annotations_path: str | Path,
    manifest_path: str | Path,
    counts: Mapping[str, Any],
) -> None:
    expected_annotations = frozen.get("frozen_annotations_sha256")
    expected_manifest = frozen.get("frozen_manifest_sha256")
    actual_annotations = _sha256(annotations_path)
    actual_manifest = _sha256(manifest_path)
    if expected_annotations and expected_annotations != actual_annotations:
        raise BlindImportError(
            "annotations.jsonl does not match the frozen record "
            f"({actual_annotations} != {expected_annotations})"
        )
    if expected_manifest and expected_manifest != actual_manifest:
        raise BlindImportError(
            "manifest.jsonl does not match the frozen record "
            f"({actual_manifest} != {expected_manifest})"
        )
    merged = frozen.get("merged") or {}
    checks = {
        "rows": merged.get("rows"),
        "boxes_total": merged.get("boxes_total"),
        "negative_rows": merged.get("negative_rows"),
        "needs_review_rows": merged.get("needs_review_rows"),
        "observations": merged.get("observations"),
    }
    actual = {
        "rows": counts["rows"],
        "boxes_total": counts["boxes"],
        "negative_rows": counts["negative_rows"],
        "needs_review_rows": counts["needs_review_rows"],
        "observations": counts["observations"],
    }
    for key, expected in checks.items():
        if expected is not None and int(expected) != int(actual[key]):
            raise BlindImportError(
                f"frozen audit {key}={expected} disagrees with the rows ({actual[key]})"
            )
    expected_classes = merged.get("boxes_by_class") or {}
    for name, expected in expected_classes.items():
        if name in counts["boxes_per_class"] and int(expected) != int(
            counts["boxes_per_class"][name]
        ):
            raise BlindImportError(
                f"frozen audit {name} boxes={expected} disagrees with rows "
                f"({counts['boxes_per_class'][name]})"
            )


def _resolve_image(record: Mapping[str, Any], images_root: Path | None) -> Path:
    manifest = record["manifest"]
    remote = manifest.get("local_image_path")
    basename = Path(str(remote or f"{record['review_image_id']}.jpg")).name
    candidates: list[Path] = []
    if images_root is not None:
        batch = str(record.get("batch") or "")
        candidates.extend(
            [
                images_root / batch / "images" / basename,
                images_root / batch / basename,
                images_root / "images" / basename,
                images_root / basename,
            ]
        )
    if remote:
        candidates.append(Path(str(remote)))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise BlindImportError(
        f"image not found for {record['review_image_id']} (tried {[str(c) for c in candidates][:3]})"
    )


def _copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        target.hardlink_to(source)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        raise ValueError(f"unknown copy mode {mode!r}")


def render_gt_overlays(
    dataset_root: str | Path,
    out_dir: str | Path,
    *,
    count: int = 6,
    max_side: int = 1280,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Draw a handful of GT-only overlays (class colors) for coordinate QA."""

    from PIL import Image, ImageDraw, ImageFont

    root = Path(dataset_root)
    if records is None:
        dataset = json.loads((root / "dataset.json").read_text())
        records = dataset.get("images") or []
    chosen: list[Mapping[str, Any]] = []
    used: set[str] = set()
    # One image per class (most boxes), then one negative, then the busiest.
    for class_name in CLASS_NAMES:
        candidates = [
            record
            for record in records
            if any(box.get("class_name") == class_name for box in record.get("boxes") or [])
        ]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda record: sum(
                1 for box in record.get("boxes") or [] if box.get("class_name") == class_name
            ),
        )
        photo_id = str(best.get("photo_id") or Path(best["image"]).stem)
        if photo_id not in used:
            chosen.append(best)
            used.add(photo_id)
    negative = next(
        (record for record in records if not (record.get("boxes") or [])),
        None,
    )
    if negative is not None:
        photo_id = str(negative.get("photo_id") or Path(negative["image"]).stem)
        if photo_id not in used:
            chosen.append(negative)
            used.add(photo_id)
    for record in sorted(records, key=lambda item: -len(item.get("boxes") or [])):
        if len(chosen) >= count:
            break
        photo_id = str(record.get("photo_id") or Path(record["image"]).stem)
        if photo_id not in used:
            chosen.append(record)
            used.add(photo_id)
    chosen = chosen[: max(1, int(count))]

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    results: list[dict[str, Any]] = []
    for record in chosen:
        source = root / record["image"]
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        scale = 1.0
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for index, box in enumerate(record.get("boxes") or []):
            color = CLASS_COLORS.get(box["class_name"], (255, 255, 255))
            x0 = float(box["x0"]) * width
            y0 = float(box["y0"]) * height
            x1 = float(box["x1"]) * width
            y1 = float(box["y1"]) * height
            draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
            label = f"{index + 1} {box['class_name']}"
            try:
                left, top, right, bottom = draw.textbbox((x0 + 2, y0 + 2), label, font=font)
            except Exception:
                left, top, right, bottom = x0 + 2, y0 + 2, x0 + 80, y0 + 14
            draw.rectangle((left - 1, top - 1, right + 2, bottom + 2), fill=(0, 0, 0))
            draw.text((x0 + 2, y0 + 2), label, fill=color, font=font)
        target = output / f"gt-{record.get('photo_id') or Path(record['image']).stem}.jpg"
        image.save(target, quality=90)
        results.append(
            {
                "photo_id": record.get("photo_id") or Path(record["image"]).stem,
                "observation_id": record.get("observation_id"),
                "image": str(source),
                "overlay": str(target),
                "width": image.width,
                "height": image.height,
                "boxes": len(record.get("boxes") or []),
                "classes": sorted({box["class_name"] for box in record.get("boxes") or []}),
            }
        )
    return results


def import_blind_test(
    *,
    annotations_path: str | Path,
    manifest_path: str | Path,
    out_dir: str | Path,
    images_root: str | Path | None = None,
    frozen_audit_path: str | Path | None = None,
    verify_frozen: bool = True,
    verify_sha256: bool = True,
    include_needs_review: bool = False,
    copy_mode: str = "copy",
    overlay_count: int = 6,
) -> dict[str, Any]:
    benchmark = load_benchmark(
        annotations_path,
        manifest_path,
        frozen_audit_path=frozen_audit_path,
        verify_frozen=verify_frozen,
    )
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_root).expanduser().resolve() if images_root else None
    if images_dir is not None and not images_dir.is_dir():
        raise BlindImportError(f"images root is not a directory: {images_dir}")

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    needs_review_archive: list[dict[str, Any]] = []
    images_out: list[dict[str, Any]] = []
    boxes_per_class: Counter[str] = Counter()
    negatives = 0

    for record in benchmark["records"]:
        if record["needs_review"]:
            payload = {
                "review_image_id": record["review_image_id"],
                "observation_id": record["observation_id"],
                "batch": record["batch"],
                "needs_review": True,
                "needs_review_note": record.get("needs_review_note"),
                "negative": record["negative"],
                "labels": [
                    {
                        "class": label["class"],
                        "class_id": CLASS_IDS[label["class"]],
                        "bbox": list(label["bbox"]),
                        "source": label["source"],
                        "review_state": label["review_state"],
                        "provenance": label["provenance"],
                    }
                    for label in record["labels"]
                ],
                "photo_url": record.get("photo_url"),
                "manifest": record["manifest"],
                "reason": "needs_review",
                "included_in_eval": bool(include_needs_review),
            }
            needs_review_archive.append(payload)
            if not include_needs_review:
                excluded.append(payload)
                continue
        source = _resolve_image(record, images_dir)
        expected_sha = record["manifest"].get("sha256")
        actual_sha = _sha256(source)
        if verify_sha256 and expected_sha and actual_sha != expected_sha:
            raise BlindImportError(
                f"sha256 mismatch for {record['review_image_id']}: "
                f"{actual_sha} != {expected_sha}"
            )
        target = root / "images" / "test" / f"{record['review_image_id']}{source.suffix.lower()}"
        _copy(source, target, copy_mode)
        label_path = root / "labels" / "test" / f"{record['review_image_id']}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        boxes = []
        for label in record["labels"]:
            x0, y0, x1, y1 = label["bbox"]
            class_id = CLASS_IDS[label["class"]]
            lines.append(f"{class_id} {x0:.6f} {y0:.6f} {x1:.6f} {y1:.6f}")
            boxes_per_class[label["class"]] += 1
            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": label["class"],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )
        # Explicit empty file for a reviewed negative.
        label_path.write_text(("\n".join(lines) + "\n") if lines else "")
        negative = not boxes
        if negative:
            negatives += 1
        entry = {
            "image": f"images/test/{record['review_image_id']}{source.suffix.lower()}",
            "photo_id": record["review_image_id"],
            "observation_id": record["observation_id"],
            "negative": negative,
            "needs_review": record["needs_review"],
            "included_in_eval": True,
            "source": {
                "batch": record["batch"],
                "photo_url": record.get("photo_url"),
                "review_image_id": record["review_image_id"],
                "source_sha256": expected_sha,
                "source_image": str(source),
            },
            "review": {
                "review_state": record["review_state"],
                "source": record["source"],
                "provenance": record["provenance"],
                "needs_review": record["needs_review"],
                "needs_review_note": record.get("needs_review_note"),
            },
            "boxes": boxes,
        }
        images_out.append(entry)
        included.append(entry)

    included_counts = {
        "images": len(images_out),
        "positive_images": sum(1 for entry in images_out if not entry["negative"]),
        "negative_images": negatives,
        "boxes": sum(boxes_per_class.values()),
        "boxes_per_class": {name: boxes_per_class.get(name, 0) for name in CLASS_NAMES},
        "observations": len({entry["observation_id"] for entry in images_out}),
    }
    excluded_counts = {
        "images": len(excluded),
        "boxes": sum(len(item["labels"]) for item in excluded),
        "boxes_per_class": {
            name: sum(
                1
                for item in excluded
                for label in item["labels"]
                if label["class"] == name
            )
            for name in CLASS_NAMES
        },
        "ids": [item["review_image_id"] for item in excluded],
    }

    excluded_dir = root / "excluded"
    excluded_dir.mkdir(parents=True, exist_ok=True)
    excluded_path = excluded_dir / "needs_review.jsonl"
    with excluded_path.open("w") as handle:
        for item in needs_review_archive:
            handle.write(json.dumps(item) + "\n")

    dataset_json = {
        "schema": "arq.anatomy4.dataset/v1",
        "contract": CONTRACT,
        "classes": list(CLASS_NAMES),
        "test_only": True,
        "frozen_benchmark": {
            "schema": BLIND_SCHEMA,
            "annotations": benchmark["annotations_path"],
            "annotations_sha256": benchmark["annotations_sha256"],
            "manifest": benchmark["manifest_path"],
            "manifest_sha256": benchmark["manifest_sha256"],
            "frozen_audit": benchmark["frozen_audit_path"],
            "source_counts": benchmark["counts"],
        },
        "excluded": {
            "needs_review": {
                **excluded_counts,
                "included_in_eval": bool(include_needs_review),
                "archive_rows": len(needs_review_archive),
                "sidecar": "excluded/needs_review.jsonl",
                "include_flag": "--include-needs-review",
            }
        },
        "images": images_out,
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=1) + "\n")

    marker = {
        "schema": FROZEN_SCHEMA,
        "contract": CONTRACT,
        "test_only": True,
        "allow_train": False,
        "splits": ["test"],
        "dataset": "dataset.json",
        "source": {
            "annotations": {
                "path": benchmark["annotations_path"],
                "sha256": benchmark["annotations_sha256"],
            },
            "manifest": {
                "path": benchmark["manifest_path"],
                "sha256": benchmark["manifest_sha256"],
            },
            "frozen_audit": benchmark["frozen_audit_path"],
        },
        "counts": {
            "source": benchmark["counts"],
            "included": included_counts,
            "excluded_needs_review": excluded_counts,
        },
        "note": (
            "FROZEN blind benchmark. Only images/test exists by design. "
            "Do not copy these images or labels into train/val."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / TEST_ONLY_MARKER).write_text(json.dumps(marker, indent=1) + "\n")
    (root / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {root}",
                "test: images/test",
                f"names: {json.dumps(list(CLASS_NAMES))}",
                "# FROZEN TEST-ONLY DATASET: train/val intentionally omitted.",
                "",
            ]
        )
    )

    audit = validate_dataset(root, require_test=True, require_val=False)
    audit_path = write_audit(audit, root / "dataset-audit.json")
    (root / "dataset-audit.md").write_text(render_audit_markdown(audit))

    overlays = render_gt_overlays(root, root / "overlays", count=overlay_count)
    import_doc = {
        "schema": BLIND_SCHEMA,
        "contract": CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_only": True,
        "source": {
            "annotations": {
                "path": benchmark["annotations_path"],
                "sha256": benchmark["annotations_sha256"],
            },
            "manifest": {
                "path": benchmark["manifest_path"],
                "sha256": benchmark["manifest_sha256"],
            },
            "frozen_audit": benchmark["frozen_audit_path"],
            "images_root": str(images_dir) if images_dir else None,
        },
        "source_counts": benchmark["counts"],
        "included": included_counts,
        "excluded_needs_review": excluded_counts,
        "include_needs_review": include_needs_review,
        "copy_mode": copy_mode,
        "verify_sha256": verify_sha256,
        "observation_groups": {
            observation: ids
            for observation, ids in sorted(_group_photos(images_out).items())
            if len(ids) > 1
        },
        "overlays": overlays,
        "audit": str(audit_path),
        "marker": str(root / TEST_ONLY_MARKER),
        "status": audit["status"],
        "blocking": audit["blocking"],
    }
    (root / "blind-benchmark.json").write_text(json.dumps(import_doc, indent=1) + "\n")
    (root / "blind-benchmark-audit.md").write_text(
        _render_import_markdown(import_doc, audit)
    )
    if audit["blocking"]:
        codes = sorted({item["code"] for item in audit["errors"]})
        raise BlindImportError(
            f"imported dataset failed validation {codes}; inspect {audit_path}"
        )
    return import_doc


def _group_photos(entries: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        groups[str(entry["observation_id"])].append(str(entry["photo_id"]))
    return groups


def _render_import_markdown(import_doc: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    source = import_doc["source_counts"]
    included = import_doc["included"]
    excluded = import_doc["excluded_needs_review"]
    lines = [
        "# Anatomy4 frozen blind benchmark — imported eval dataset",
        "",
        f"- contract: {import_doc['contract']}",
        f"- status: **{audit['status']}** · test-only: yes",
        f"- source: {source['rows']} rows / {source['boxes']} boxes",
        f"- included: {included['images']} images / {included['boxes']} boxes "
        f"({included['negative_images']} negatives)",
        f"- excluded needs_review: {excluded['images']} images / "
        f"{excluded['boxes']} boxes: {', '.join(excluded['ids']) or 'none'}",
        f"- annotations sha256: `{import_doc['source']['annotations']['sha256']}`",
        f"- manifest sha256: `{import_doc['source']['manifest']['sha256']}`",
        "",
        "## Per-class (included)",
        "",
        "| class | boxes |",
        "| --- | ---: |",
    ]
    for name in CLASS_NAMES:
        lines.append(f"| {name} | {included['boxes_per_class'].get(name, 0)} |")
    lines += [
        "",
        "## Source (frozen)",
        "",
        "| class | boxes | images |",
        "| --- | ---: | ---: |",
    ]
    for name in CLASS_NAMES:
        lines.append(
            f"| {name} | {source['boxes_per_class'].get(name, 0)} | "
            f"{source['images_per_class'].get(name, 0)} |"
        )
    if import_doc["observation_groups"]:
        lines += ["", "## Multi-photo observations (all in test)", ""]
        for observation, photos in import_doc["observation_groups"].items():
            lines.append(f"- `{observation}`: {len(photos)} photos")
    if import_doc["overlays"]:
        lines += ["", "## GT overlays", ""]
        for overlay in import_doc["overlays"]:
            lines.append(
                f"- `{overlay['photo_id']}` ({overlay['boxes']} boxes, "
                f"{overlay['width']}×{overlay['height']}): {overlay['overlay']}"
            )
    if audit["errors"]:
        lines += ["", "## Errors", ""]
        lines += [f"- `{item['code']}` {item['message']}" for item in audit["errors"]]
    if audit["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- `{item['code']}` {item['message']}" for item in audit["warnings"]]
    return "\n".join(lines) + "\n"
