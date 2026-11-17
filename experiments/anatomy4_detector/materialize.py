# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Materialize the DINO/YOLOE audit artifacts into the anatomy4-v1 dataset.

Inputs (read-only):

* ``proposals.json`` — ``experiments/anatomy4`` prelabel output. Every proposal
  carries ``proposal_id``, ``photo_id``, ``observation_id``, ``class_name``,
  ``box_score``, ``model`` provenance, and a normalized **xyxy** ``bbox`` (plus
  an optional polygon). Labels are copied from ``bbox`` as-is; a polygon-only
  proposal gets its tight normalized bbox. ``cx cy w h`` is never interpreted
  here, so the only x-y-w-h conversion in the pipeline happens once, in
  ``dataset.prepare_ultralytics``.
* ``review.csv`` — human verdicts keyed by ``proposal_id``
  (``tp``/``fp``/``unsure``, case-insensitive). Only ``tp`` becomes truth;
  rejected and **unreviewed** proposals never become boxes.
* ``manifest.downloaded.json`` — optional; used to find images and the
  observation ids of photos the prelabel images list does not carry.
* an images directory (default: ``<proposals dir>/../images``).

Output (the exact layout ``validate``/``pipeline`` consume)::

    <out>/dataset.json            # images, observation ids, negative flags, review counts
    <out>/images/<split>/<photo>.<ext>
    <out>/labels/<split>/<photo>.txt   # normalized xyxy; empty file for negatives
    <out>/provenance.jsonl        # one row per accepted box (confidence, model, review row)
    <out>/materialization.json    # input hashes, policies, counts, partial-label warnings
    <out>/data.yaml               # Ultralytics class order
    <out>/dataset-audit.{json,md} # result of validate_dataset on the output

Negative policy (``--negative-policy``):

* ``reviewed-rejected`` (default) — an image becomes an explicit negative when
  **all** of its proposals were reviewed and **all** were rejected. Images with
  no proposals are never assumed negative (the detector may have missed an
  organ; nobody reviewed it). With ``--classes``, the image must have at least
  one reviewed proposal of a selected class, so a negative is only claimed for
  the classes actually looked at.
* ``explicit`` — only ``--negative-image``/``--negative-file`` entries.
* ``none`` — no negatives beyond reviewed positives.

Observation-safe splits use the canonical deterministic hash
(``contracts.split_for_observation``, 70/15/15). All images of an observation
land in one split; the emitted dataset is validated for leakage before the
command succeeds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    CLASS_IDS,
    CLASS_NAMES,
    CONTRACT,
    IMAGE_SUFFIXES,
    SPLITS,
    split_for_observation,
)
from .dataset import (
    render_audit_markdown,
    validate_dataset,
    write_audit,
)

MATERIALIZATION_SCHEMA = "arq.anatomy4.materialization/v1"

ACCEPT_VERDICTS = frozenset({"tp", "accept", "accepted", "true_positive", "true-positive", "yes", "1"})
REJECT_VERDICTS = frozenset({"fp", "reject", "rejected", "false_positive", "false-positive", "no", "0"})
UNDECIDED_VERDICTS = frozenset({"", "unsure", "?", "maybe", "unreviewed", "pending", "-"})


class MaterializationError(RuntimeError):
    """The audit artifacts cannot be materialized without inventing truth."""


@dataclass
class ReviewEntry:
    proposal_id: str
    verdict: str  # "accept" | "reject" | "undecided"
    raw_verdict: str
    notes: str
    row: int
    conflicts: list[str] = field(default_factory=list)


@dataclass
class ReviewSummary:
    entries: dict[str, ReviewEntry]
    rows: int
    accepted: int
    rejected: int
    undecided: int
    conflicts: int
    unknown_ids: list[str]


def normalize_verdict(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in ACCEPT_VERDICTS:
        return "accept"
    if raw in REJECT_VERDICTS:
        return "reject"
    if raw in UNDECIDED_VERDICTS:
        return "undecided"
    raise ValueError(f"unknown review verdict {value!r}")


def load_reviews(path: str | Path) -> ReviewSummary:
    """Load a review sheet. Later rows win; conflicting verdicts are flagged."""

    source = Path(path)
    entries: dict[str, ReviewEntry] = {}
    rows = 0
    accepted = 0
    rejected = 0
    undecided = 0
    conflicts = 0
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "proposal_id" not in reader.fieldnames:
            raise MaterializationError(
                f"review sheet {source} has no 'proposal_id' column: {reader.fieldnames}"
            )
        for index, row in enumerate(reader, start=1):
            rows += 1
            proposal_id = (row.get("proposal_id") or "").strip()
            if not proposal_id:
                continue
            verdict = normalize_verdict(row.get("verdict"))
            notes = (row.get("notes") or "").strip()
            if proposal_id in entries:
                previous = entries[proposal_id]
                if (
                    previous.verdict != "undecided"
                    and verdict != "undecided"
                    and previous.verdict != verdict
                ):
                    conflicts += 1
                    previous.conflicts.append(previous.raw_verdict)
                prior_conflicts = previous.conflicts
                entry = ReviewEntry(
                    proposal_id=proposal_id,
                    verdict=verdict if verdict != "undecided" else previous.verdict,
                    raw_verdict=(row.get("verdict") or "").strip() or previous.raw_verdict,
                    notes=notes or previous.notes,
                    row=index,
                    conflicts=prior_conflicts,
                )
                entries[proposal_id] = entry
            else:
                entries[proposal_id] = ReviewEntry(
                    proposal_id=proposal_id,
                    verdict=verdict,
                    raw_verdict=(row.get("verdict") or "").strip(),
                    notes=notes,
                    row=index,
                )
    for entry in entries.values():
        if entry.verdict == "accept":
            accepted += 1
        elif entry.verdict == "reject":
            rejected += 1
        else:
            undecided += 1
    return ReviewSummary(
        entries=entries,
        rows=rows,
        accepted=accepted,
        rejected=rejected,
        undecided=undecided,
        conflicts=conflicts,
        unknown_ids=[],
    )


def load_background_review(path: str | Path) -> dict[str, Any]:
    """Image-level background review (``review-background.csv``).

    Verdicts: ``empty_correct`` = a human confirmed no target organ (explicit
    negative); ``missed_organ`` = an organ is present but the prelabel missed it
    (never a negative). Anything else is ignored.
    """

    source = Path(path)
    verdicts: dict[str, str] = {}
    conflicts = 0
    rows = 0
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"photo_id", "verdict"}.issubset(fields):
            raise MaterializationError(
                f"background review {source} needs photo_id+verdict columns, got {sorted(fields)}"
            )
        for row in reader:
            rows += 1
            photo_id = (row.get("photo_id") or "").strip()
            raw = (row.get("verdict") or "").strip().lower()
            if not photo_id:
                continue
            verdict = (
                "empty" if raw in {"empty_correct", "empty", "empty-correct"}
                else "missed" if raw in {"missed_organ", "missed", "missed-organ", "organ_present"}
                else "undecided"
            )
            previous = verdicts.get(photo_id)
            if previous in {"empty", "missed"} and verdict in {"empty", "missed"} and previous != verdict:
                conflicts += 1
            if verdict != "undecided":
                verdicts[photo_id] = verdict
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "rows": rows,
        "verdicts": verdicts,
        "empty": sorted(pid for pid, verdict in verdicts.items() if verdict == "empty"),
        "missed": sorted(pid for pid, verdict in verdicts.items() if verdict == "missed"),
        "conflicts": conflicts,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise MaterializationError(f"{path} must contain a JSON object")
    return data


def _validate_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise MaterializationError(f"bbox must have 4 values, got {len(box)}")
    x0, y0, x1, y1 = (float(value) for value in box)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise MaterializationError(
            f"bbox is not normalized xyxy with x0<x1, y0<y1 in [0,1]: {(x0, y0, x1, y1)}"
        )
    return (x0, y0, x1, y1)


def proposal_xyxy(proposal: Mapping[str, Any]) -> tuple[tuple[float, float, float, float], str]:
    """Return the normalized xyxy box and the field it came from.

    ``bbox`` is the reviewed box (polygon-tight); ``detector_bbox`` is the raw
    detection box. Both are normalized xyxy in the anatomy4 artifacts. A
    parsed ``cx cy w h`` is rejected, never reinterpreted.
    """

    for field_name in ("bbox", "detector_bbox"):
        value = proposal.get(field_name)
        if value:
            return _validate_xyxy(value), field_name
    polygon = proposal.get("polygon")
    if polygon and len(polygon) >= 3:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        return _validate_xyxy((min(xs), min(ys), max(xs), max(ys))), "polygon"
    raise MaterializationError(
        f"proposal {proposal.get('proposal_id')!r} has no bbox/detector_bbox/polygon"
    )


def _image_suffix_index(images_dir: Path | None) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if images_dir is None or not images_dir.is_dir():
        return index
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in index:
                raise MaterializationError(
                    f"duplicate image stem {path.stem!r}: {index[path.stem]} and {path}"
                )
            index[path.stem] = path
    return index


def resolve_image(
    photo_id: str,
    image_record: Mapping[str, Any] | None,
    *,
    images_dir: Path | None,
    suffix_index: Mapping[str, Path],
    manifest_index: Mapping[str, Path],
) -> Path | None:
    candidates: list[Path] = []
    if photo_id in suffix_index:
        candidates.append(suffix_index[photo_id])
    record = image_record or {}
    raw_path = record.get("image_path")
    if raw_path:
        candidates.append(Path(raw_path))
        if images_dir is not None:
            candidates.append(images_dir / Path(raw_path).name)
    download = record.get("download") or {}
    download_path = download.get("path")
    if download_path:
        candidates.append(Path(download_path))
        if images_dir is not None:
            candidates.append(images_dir / Path(download_path).name)
    if photo_id in manifest_index:
        candidates.append(manifest_index[photo_id])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _copy_image(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        raise ValueError(f"unknown copy mode {mode!r}")


def _explicit_negative_ids(values: Iterable[str]) -> set[str]:
    ids: set[str] = set()
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                ids.add(token)
    return ids


def _load_id_file(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    source = Path(path)
    if not source.is_file():
        raise MaterializationError(f"negative list not found: {source}")
    return {
        line.strip()
        for line in source.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def write_data_yaml(out_dir: Path) -> Path:
    target = out_dir / "data.yaml"
    target.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {json.dumps(list(CLASS_NAMES))}",
                "",
            ]
        )
    )
    return target


def materialize(
    *,
    proposals_path: str | Path,
    review_path: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
    images_dir: str | Path | None = None,
    classes: Sequence[str] | None = None,
    seed: str = CONTRACT,
    negative_policy: str = "reviewed-rejected",
    negative_images: Iterable[str] = (),
    negative_file: str | Path | None = None,
    background_review_path: str | Path | None = None,
    require_fully_reviewed: bool = False,
    allow_missing_images: bool = False,
    copy_mode: str = "copy",
    dedupe_exact: bool = True,
    verify_sha256: bool = True,
    require_test: bool = True,
    require_val: bool = True,
) -> dict[str, Any]:
    """Materialize one dataset; raises :class:`MaterializationError` on conflict."""

    if negative_policy not in {"reviewed-rejected", "explicit", "none"}:
        raise MaterializationError(f"unknown negative policy {negative_policy!r}")

    proposals_source = Path(proposals_path)
    review_source = Path(review_path)
    manifest_source = Path(manifest_path) if manifest_path else None
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    images_root = (
        Path(images_dir).expanduser().resolve()
        if images_dir is not None
        else (proposals_source.parent.parent / "images").resolve()
    )

    proposals_doc = _load_json(proposals_source)
    contract = str(proposals_doc.get("contract", CONTRACT))
    if contract != CONTRACT:
        raise MaterializationError(
            f"proposals contract {contract!r} != {CONTRACT!r}"
        )
    raw_proposals = proposals_doc.get("proposals")
    if not isinstance(raw_proposals, list):
        raise MaterializationError("proposals.json has no proposals[] list")
    image_records: dict[str, dict[str, Any]] = {}
    for record in proposals_doc.get("images") or []:
        photo_id = str(record.get("image_id") or record.get("photo_id") or "")
        if photo_id:
            image_records[photo_id] = record
    manifest_doc: dict[str, Any] = {}
    manifest_index: dict[str, Path] = {}
    manifest_records: dict[str, dict[str, Any]] = {}
    if manifest_source is not None:
        if not manifest_source.is_file():
            raise MaterializationError(f"observations manifest not found: {manifest_source}")
        manifest_doc = _load_json(manifest_source)
        for record in manifest_doc.get("images") or []:
            photo_id = str(record.get("photo_id") or record.get("image_id") or "")
            download_path = (record.get("download") or {}).get("path")
            if photo_id:
                manifest_records[photo_id] = record
            if photo_id and download_path:
                manifest_index[photo_id] = Path(download_path)

    reviews = load_reviews(review_source)
    proposal_ids = {str(item.get("proposal_id")) for item in raw_proposals if item.get("proposal_id")}
    reviews.unknown_ids = sorted(set(reviews.entries) - proposal_ids)
    overlap = [pid for pid in reviews.entries if pid in proposal_ids]
    if not overlap:
        raise MaterializationError(
            "review sheet and proposals share no proposal_id; refusing to materialize"
        )

    selected_classes = list(CLASS_NAMES)
    if classes:
        requested = [name.strip().lower() for name in classes if name.strip()]
        invalid = [name for name in requested if name not in CLASS_IDS]
        if invalid:
            raise MaterializationError(
                f"unknown classes {invalid}; contract classes are {list(CLASS_NAMES)}"
            )
        selected_classes = [name for name in CLASS_NAMES if name in requested]
    selected_ids = {CLASS_IDS[name] for name in selected_classes}

    cli_negative_ids = _explicit_negative_ids(negative_images) | _load_id_file(negative_file)
    background = (
        load_background_review(background_review_path)
        if background_review_path
        else None
    )
    background_empty = set(background["empty"]) if background else set()
    background_missed = set(background["missed"]) if background else set()
    explicit_ids = cli_negative_ids | {
        photo_id for photo_id in background_empty if photo_id in image_records
    }
    suffix_index = _image_suffix_index(images_root if images_root.is_dir() else None)

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_not_proposed = 0
    for proposal in raw_proposals:
        if proposal.get("status") not in (None, "proposed"):
            skipped_not_proposed += 1
            continue
        photo_id = str(proposal.get("photo_id") or proposal.get("image_id") or "")
        if not photo_id:
            raise MaterializationError(
                f"proposal {proposal.get('proposal_id')!r} has no photo_id"
            )
        by_image[photo_id].append(proposal)

    images_out: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    boxes_per_class: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    split_observations: dict[str, set[str]] = {split: set() for split in SPLITS}
    missing_images: list[str] = []
    dropped: Counter[str] = Counter()
    partial_images = 0
    unreviewed_on_labeled = 0
    excluded_fully_reviewed = 0
    deduped_boxes: list[dict[str, Any]] = []
    positive_images: set[str] = set()
    negative_images_out: set[str] = set()
    model_ids: set[str] = set()

    def find_explicit(identifier: str) -> str | None:
        for photo_id, record in image_records.items():
            raw_path = str(record.get("image_path") or "")
            if identifier in {
                photo_id,
                f"{photo_id}.jpg",
                Path(raw_path).name,
                raw_path,
                Path(raw_path).stem,
            }:
                return photo_id
        return None

    for identifier in sorted(explicit_ids):
        photo_id = find_explicit(identifier)
        if photo_id is None:
            raise MaterializationError(
                f"--negative-image {identifier!r} does not match any prelabel image"
            )
    explicit_photo_ids = {
        find_explicit(identifier) for identifier in explicit_ids
    }

    for photo_id in sorted(set(by_image) | explicit_photo_ids):
        proposals = by_image.get(photo_id, [])
        record = image_records.get(photo_id, {})
        observation_id = str(record.get("observation_id") or "")
        if not observation_id:
            # Fall back to the proposals themselves.
            observation_id = str(proposals[0].get("observation_id") or "")
        if not observation_id:
            raise MaterializationError(
                f"image {photo_id} has no observation_id in the prelabel/manifest"
            )

        decisions: list[dict[str, Any]] = []
        for proposal in proposals:
            if proposal.get("class_name") not in CLASS_IDS:
                raise MaterializationError(
                    f"proposal {proposal.get('proposal_id')!r} class "
                    f"{proposal.get('class_name')!r} is not in {list(CLASS_NAMES)}"
                )
            review = reviews.entries.get(str(proposal.get("proposal_id")))
            verdict = review.verdict if review else "undecided"
            decisions.append(
                {
                    "proposal": proposal,
                    "class_name": proposal["class_name"],
                    "verdict": verdict,
                    "review": review,
                }
            )
        n_accepted = sum(1 for item in decisions if item["verdict"] == "accept")
        n_rejected = sum(1 for item in decisions if item["verdict"] == "reject")
        n_undecided = sum(1 for item in decisions if item["verdict"] == "undecided")
        accepted_selected = [
            item
            for item in decisions
            if item["verdict"] == "accept"
            and CLASS_IDS[item["class_name"]] in selected_ids
        ]

        negative_basis: str | None = None
        include = False
        if accepted_selected:
            if photo_id in background_empty:
                raise MaterializationError(
                    f"image {photo_id} is marked empty_correct in the background "
                    f"review but has {len(accepted_selected)} accepted proposal(s)"
                )
            if photo_id in explicit_photo_ids:
                raise MaterializationError(
                    f"image {photo_id} is listed as an explicit negative but has "
                    f"{len(accepted_selected)} accepted proposal(s)"
                )
            include = True
        elif photo_id in explicit_photo_ids:
            include = True
            negative_basis = (
                "explicit"
                if photo_id in cli_negative_ids
                else "background_empty"
            )
        elif negative_policy == "reviewed-rejected":
            fully_decided = n_undecided == 0
            reviewed_classes = {
                item["class_name"]
                for item in decisions
                if item["verdict"] in {"accept", "reject"}
            }
            class_ok = not classes or bool(reviewed_classes & set(selected_classes))
            if fully_decided and n_accepted == 0 and n_rejected > 0 and class_ok:
                include = True
                negative_basis = "reviewed_all_rejected"
        else:
            include = False

        if include and photo_id in background_missed and negative_basis is not None:
            # A human saw an organ; never export this as an empty image.
            include = False
            dropped["background_missed_organ"] += 1
        if include and require_fully_reviewed and n_undecided and negative_basis not in {
            "explicit",
            "background_empty",
        }:
            include = False
            excluded_fully_reviewed += 1

        if not include:
            dropped["images_without_accepted_truth"] += 1
            continue
        if n_undecided and accepted_selected:
            partial_images += 1
            unreviewed_on_labeled += n_undecided

        image_path = resolve_image(
            photo_id,
            record,
            images_dir=images_root if images_root.is_dir() else None,
            suffix_index=suffix_index,
            manifest_index=manifest_index,
        )
        if image_path is None:
            missing_images.append(photo_id)
            dropped["missing_images"] += 1
            continue

        digest = (record.get("download") or {}).get("sha256") or (
            manifest_records.get(photo_id, {}).get("download") or {}
        ).get("sha256")
        if verify_sha256 and digest:
            actual = sha256_file(image_path)
            if actual != digest:
                raise MaterializationError(
                    f"image sha256 mismatch for {photo_id}: {actual} != {digest}"
                )

        split = split_for_observation(observation_id, seed=seed)
        target = output / "images" / split / f"{photo_id}{image_path.suffix.lower()}"
        _copy_image(image_path, target, copy_mode)
        split_observations[split].add(observation_id)
        split_counts[split]["images"] += 1

        prepared_boxes: list[dict[str, Any]] = []
        for item in accepted_selected:
            proposal = item["proposal"]
            box, box_source = proposal_xyxy(proposal)
            prepared_boxes.append(
                {"item": item, "proposal": proposal, "box": box, "box_source": box_source}
            )
        if dedupe_exact and prepared_boxes:
            unique: dict[tuple[int, float, float, float, float], dict[str, Any]] = {}
            for prepared in prepared_boxes:
                key = (
                    CLASS_IDS[prepared["item"]["class_name"]],
                    *(round(value, 6) for value in prepared["box"]),
                )
                existing = unique.get(key)
                if existing is None:
                    unique[key] = prepared
                    continue
                existing_score = float(existing["proposal"].get("box_score") or 0.0)
                candidate_score = float(prepared["proposal"].get("box_score") or 0.0)
                keep = (
                    prepared
                    if (candidate_score, str(prepared["proposal"].get("proposal_id")))
                    > (existing_score, str(existing["proposal"].get("proposal_id")))
                    else existing
                )
                dropped_item = existing if keep is prepared else prepared
                unique[key] = keep
                deduped_boxes.append(
                    {
                        "image": photo_id,
                        "kept_proposal_id": keep["proposal"].get("proposal_id"),
                        "dropped_proposal_id": dropped_item["proposal"].get("proposal_id"),
                        "class_name": keep["item"]["class_name"],
                        "xyxy_norm": list(keep["box"]),
                    }
                )
            prepared_boxes = list(unique.values())

        label_lines: list[str] = []
        for prepared in prepared_boxes:
            item = prepared["item"]
            proposal = prepared["proposal"]
            box = prepared["box"]
            box_source = prepared["box_source"]
            label_lines.append(
                f"{CLASS_IDS[item['class_name']]} {box[0]:.6f} {box[1]:.6f} "
                f"{box[2]:.6f} {box[3]:.6f}"
            )
            boxes_per_class[item["class_name"]] += 1
            model = proposal.get("model") or proposals_doc.get("provenance") or {}
            detector = model.get("detector") if isinstance(model, dict) else None
            if detector:
                model_ids.add(str(detector))
            review = item["review"]
            provenance_rows.append(
                {
                    "schema": "arq.anatomy4.proposal-provenance/v1",
                    "contract": CONTRACT,
                    "proposal_id": proposal.get("proposal_id"),
                    "image": f"images/{split}/{photo_id}{image_path.suffix.lower()}",
                    "photo_id": photo_id,
                    "observation_id": observation_id,
                    "split": split,
                    "class_id": CLASS_IDS[item["class_name"]],
                    "class_name": item["class_name"],
                    "xyxy_norm": list(box),
                    "box_source": box_source,
                    "box_score": proposal.get("box_score"),
                    "detector_label": proposal.get("detector_label"),
                    "prompt": proposal.get("prompt"),
                    "mask_iou": proposal.get("mask_iou"),
                    "model": model,
                    "created_at": proposal.get("created_at"),
                    "review": {
                        "verdict": "tp",
                        "raw_verdict": review.raw_verdict if review else "tp",
                        "notes": review.notes if review else "",
                        "sheet": str(review_source),
                        "row": review.row if review else None,
                    },
                }
            )

        label_path = output / "labels" / split / f"{photo_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        # Explicit empty file for every negative (never omit it).
        label_path.write_text(("\n".join(label_lines) + "\n") if label_lines else "")

        negative = not label_lines
        if negative:
            negative_images_out.add(photo_id)
            split_counts[split]["negatives"] += 1
        else:
            positive_images.add(photo_id)
        split_counts[split]["boxes"] += len(label_lines)

        images_out.append(
            {
                "image": f"images/{split}/{photo_id}{image_path.suffix.lower()}",
                "observation_id": observation_id,
                "negative": negative,
                "photo_id": photo_id,
                "source_image": str(image_path),
                "source_sha256": digest,
                "review": {
                    "reviewed_proposals": n_accepted + n_rejected,
                    "accepted": n_accepted,
                    "rejected": n_rejected,
                    "undecided": n_undecided,
                    "negative_basis": negative_basis,
                },
                "model": next(
                    (
                        (item["proposal"].get("model") or {}).get("detector")
                        for item in decisions
                        if item["proposal"].get("model")
                    ),
                    None,
                ),
            }
        )

    if missing_images and not allow_missing_images:
        raise MaterializationError(
            f"{len(missing_images)} image file(s) missing (first: {missing_images[:5]}); "
            "pass --images-dir or --allow-missing-images"
        )
    if not images_out:
        raise MaterializationError(
            "no images selected; check the review/proposal ids and negative policy"
        )

    leakage = {
        observation: sorted(
            split for split, values in split_observations.items() if observation in values
        )
        for observation in set().union(*split_observations.values())
        if sum(observation in values for values in split_observations.values()) > 1
    }
    if leakage:
        raise MaterializationError(f"observation split leakage: {leakage}")

    dataset_json = {
        "schema": "arq.anatomy4.dataset/v1",
        "contract": CONTRACT,
        "classes": list(CLASS_NAMES),
        "materialization": "materialization.json",
        "images": images_out,
    }
    (output / "dataset.json").write_text(json.dumps(dataset_json, indent=1) + "\n")
    with (output / "provenance.jsonl").open("w") as handle:
        for row in provenance_rows:
            handle.write(json.dumps(row) + "\n")
    data_yaml = write_data_yaml(output)

    audit = validate_dataset(
        output, require_test=require_test, require_val=require_val
    )
    audit_path = write_audit(audit, output / "dataset-audit.json")
    (output / "dataset-audit.md").write_text(render_audit_markdown(audit))

    inputs = {
        "proposals": {
            "path": str(proposals_source.resolve()),
            "sha256": sha256_file(proposals_source),
        },
        "review": {
            "path": str(review_source.resolve()),
            "sha256": sha256_file(review_source),
        },
        "images_dir": str(images_root),
    }
    if manifest_source is not None:
        inputs["manifest"] = {
            "path": str(manifest_source.resolve()),
            "sha256": sha256_file(manifest_source),
        }
    if background:
        inputs["background_review"] = {
            "path": background["path"],
            "sha256": background["sha256"],
            "rows": background["rows"],
            "empty": len(background["empty"]),
            "missed": len(background["missed"]),
            "unknown_ids": len(
                [pid for pid in background["verdicts"] if pid not in image_records]
            ),
            "conflicts": background["conflicts"],
        }
    materialization = {
        "schema": MATERIALIZATION_SCHEMA,
        "contract": CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "selection": {
            "classes": selected_classes,
            "negative_policy": negative_policy,
            "require_fully_reviewed": require_fully_reviewed,
            "explicit_negatives": sorted(explicit_photo_ids),
            "seed": seed,
            "copy_mode": copy_mode,
            "dedupe_exact": dedupe_exact,
            "verify_sha256": verify_sha256,
        },
        "review": {
            "rows": reviews.rows,
            "unique_proposals": len(reviews.entries),
            "accepted": reviews.accepted,
            "rejected": reviews.rejected,
            "undecided": reviews.undecided,
            "conflicts": reviews.conflicts,
            "unknown_ids": len(reviews.unknown_ids),
            "unknown_ids_sample": reviews.unknown_ids[:20],
        },
        "counts": {
            "images": len(images_out),
            "positives": len(positive_images),
            "negatives": len(negative_images_out),
            "boxes": sum(boxes_per_class.values()),
            "boxes_per_class": {name: boxes_per_class.get(name, 0) for name in CLASS_NAMES},
            "splits": {
                split: {
                    "images": split_counts[split]["images"],
                    "negatives": split_counts[split]["negatives"],
                    "boxes": split_counts[split]["boxes"],
                    "observations": len(split_observations[split]),
                }
                for split in SPLITS
            },
            "skipped_not_proposed": skipped_not_proposed,
            "dropped": dict(dropped),
            "excluded_fully_reviewed": excluded_fully_reviewed,
            "deduped_exact_boxes": len(deduped_boxes),
        },
        "dedupe": {
            "policy": "exact class+xyxy (6dp), highest box_score kept"
            if dedupe_exact
            else "disabled",
            "dropped": deduped_boxes,
        },
        "partial_labeling": {
            "images_with_unreviewed_proposals": partial_images,
            "unreviewed_proposals_on_labeled_images": unreviewed_on_labeled,
            "note": (
                "Unreviewed proposals are excluded from truth. On labeled images "
                "they are unlabeled, so the trainer treats them as background; "
                "use --require-fully-reviewed for a stricter subset."
            ),
        },
        "model_ids": sorted(model_ids),
        "output": {
            "dataset": str(output / "dataset.json"),
            "data_yaml": str(data_yaml),
            "provenance": str(output / "provenance.jsonl"),
            "audit": str(audit_path),
            "audit_markdown": str(output / "dataset-audit.md"),
        },
        "status": audit["status"],
        "blocking": audit["blocking"],
        "split_isolation": audit["split_isolation"],
        "errors": audit["errors"],
        "warnings": audit["warnings"],
    }
    (output / "materialization.json").write_text(
        json.dumps(materialization, indent=1) + "\n"
    )
    if audit["blocking"]:
        codes = sorted({item["code"] for item in audit["errors"]})
        raise MaterializationError(
            f"materialized dataset failed validation {codes}; "
            f"inspect {output / 'dataset-audit.json'}"
        )
    return materialization
