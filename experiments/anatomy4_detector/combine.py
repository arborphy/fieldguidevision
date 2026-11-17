# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Combine mixed reviewed Shenandoah sources into one train/val dataset.

Sources (all JSONL; any may be absent until its reviewer finishes):

* ``train-candidates/verified/assisted-labels.jsonl`` — accepted model labels
  with policy provenance;
* ``train-candidates/curated-needle/batch-{1..4}/annotations.jsonl`` — accepted
  curator needle labels;
* ``train-candidates/negative-review/batch-{1,2}.jsonl`` — accepted explicit
  negatives;
* optional ``--review-queue`` outputs from later manual review.

Rules:

* only accepted/confirmed records/labels contribute truth. ``needs_review``,
  ``proposed`` and unknown review states are excluded and written to
  ``quarantine/`` (never silently dropped);
* a negative for an image that also has an accepted positive is invalid: the
  positive wins and the negative record is quarantined;
* identical ``class + bbox`` (6 dp) boxes from multiple sources collapse to one
  box, retaining every source in the merged provenance;
* splits are deterministic per OBSERVATION (default 85/15 train/val), so all
  photos of a plant land together;
* the frozen blind benchmark is a hard denylist (observation id, photo id,
  photo URL, sha256). Any overlap aborts the combine;
* class-specific source selection is supported with
  ``--class-source needle=curated-needle`` (JSONL sources are named
  ``assisted``, ``curated-needle``, ``negative-review``, ``review-queue``, or
  whatever ``--source name=path`` supplies).

Output is the same validated layout as ``materialize``, plus
``merge-report.json``, ``provenance.jsonl``, ``quarantine/`` and GT overlays.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    CLASS_IDS,
    CLASS_NAMES,
    CONTRACT,
    IMAGE_SUFFIXES,
    ContractError,
    stable_unit_interval,
)
from .dataset import prepare_ultralytics, render_audit_markdown, validate_dataset, write_audit
from .blindtest import render_gt_overlays

COMBINE_SCHEMA = "arq.anatomy4.combine/v1"
DEFAULT_SEED = "anatomy4-train-v1"
DEFAULT_TRAIN_RATIO = 0.85
DEFAULT_BLIND_ROOT = Path("/srv/fast/anatomy4-shenandoah/blind-test/merged")
DEFAULT_CANDIDATE_ROOT = Path("/srv/fast/anatomy4-shenandoah/train-candidates")

ACCEPT_STATES = frozenset({"accepted", "confirmed", "approved"})
EXCLUDE_STATES = frozenset(
    {"proposed", "needs_review", "needs-review", "pending", "unreviewed", "rejected"}
)
NEGATIVE_VERDICTS = frozenset(
    {"empty_correct", "empty-correct", "empty", "negative", "no_target", "no-target"}
)
ACCEPT_VERDICTS = frozenset({"accept", "accepted", "tp", "confirm", "confirmed"})


class CombineError(RuntimeError):
    """The sources cannot be combined without inventing or leaking truth."""


class BlindLeakageError(CombineError):
    """A source image overlaps the frozen blind benchmark; always fatal."""


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if not source.is_file():
        raise CombineError(f"source not found: {source}")
    for number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CombineError(f"{source}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise CombineError(f"{source}:{number}: row must be an object")
        rows.append(row)
    return rows


def _norm_state(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _accepted(row: Mapping[str, Any]) -> bool:
    state = _norm_state(row.get("review_state") or row.get("status"))
    if state in ACCEPT_STATES:
        return True
    if state in EXCLUDE_STATES:
        return False
    if row.get("accepted") is True:
        return True
    if _norm_state(row.get("verdict")) in ACCEPT_VERDICTS:
        return True
    labels = _labels(row)
    if labels:
        states = {
            _norm_state(label.get("review_state") or label.get("status"))
            for label in labels
        }
        if states and states <= ACCEPT_STATES:
            return True
        if all(label.get("accepted") is True for label in labels):
            return True
    return False


def _excluded_state(row: Mapping[str, Any]) -> str | None:
    state = _norm_state(row.get("review_state") or row.get("status"))
    if row.get("needs_review") is True:
        return "needs_review"
    if state in EXCLUDE_STATES:
        return state
    return None


def _is_negative(row: Mapping[str, Any]) -> bool:
    verdict = _norm_state(row.get("verdict") or row.get("review"))
    return (
        row.get("negative") is True
        or row.get("is_negative") is True
        or verdict in NEGATIVE_VERDICTS
        or _norm_state(row.get("label_state")) in {"empty", "negative"}
    )


def _bbox(value: Any, where: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise CombineError(f"{where}: bbox must be 4 values, got {value!r}")
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise CombineError(f"{where}: bbox values must be numbers") from exc
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise CombineError(f"{where}: bbox is not normalized xyxy: {value!r}")
    return (x0, y0, x1, y1)


def _label_class(raw: Mapping[str, Any], where: str) -> str:
    name = _norm_state(raw.get("class") or raw.get("class_name") or raw.get("label"))
    if name not in CLASS_IDS:
        raise CombineError(f"{where}: class {name!r} is not in {list(CLASS_NAMES)}")
    return name


def _labels(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("labels", "annotations", "boxes", "detections"):
        value = raw.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    # Assisted verification emits one flat row per accepted box:
    # {class, bbox_xyxy_norm, proposal_id, review_state, ...}.
    if raw.get("class") and any(
        key in raw for key in ("bbox", "bbox_xyxy", "bbox_xyxy_norm")
    ):
        return [raw]
    return []


class Source:
    def __init__(self, name: str, paths: Sequence[Path]) -> None:
        self.name = name
        self.paths = list(paths)


def _expand(pattern: str | Path) -> list[Path]:
    text = str(pattern)
    if any(ch in text for ch in "*?["):
        return [Path(item) for item in sorted(glob.glob(text))]
    return [Path(text)]


def _expand_many(
    patterns: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(patterns, (str, Path)):
        values = [patterns]
    else:
        values = list(patterns)
    paths: list[Path] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                paths.extend(_expand(part))
    return sorted({path for path in paths})


def discover_sources(
    *,
    assisted: str | Path | None,
    curated_needle_glob: str | Path | Sequence[str | Path] | None,
    negative_glob: str | Path | Sequence[str | Path] | None,
    review_queue: Iterable[str | Path] = (),
    curated_globs: Sequence[str | Path] = (),
    extra_sources: Iterable[str] = (),
) -> list[Source]:
    sources: list[Source] = []

    def add(name: str, candidates: Iterable[Path]) -> None:
        paths = [path for path in candidates if path.is_file()]
        if paths:
            sources.append(Source(name, sorted(paths)))

    if assisted is not None:
        add("assisted", [Path(assisted)])
    if curated_needle_glob is not None:
        add("curated-needle", _expand_many(curated_needle_glob))
    if curated_globs:
        add("curated", _expand_many(curated_globs))
    if negative_glob is not None:
        add("negative-review", _expand_many(negative_glob))
    add("review-queue", [Path(path) for path in review_queue])
    for spec in extra_sources:
        if "=" not in spec:
            raise CombineError(f"--source must be name=path, got {spec!r}")
        name, pattern = spec.split("=", 1)
        add(name.strip(), _expand(pattern))
    return sources


def _manifest_index(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return {}
    index: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        path_values = [
            row.get("local_image_path"),
            row.get("image_path"),
            row.get("path"),
        ]
        keys = [
            row.get("image_id"),
            row.get("review_image_id"),
            row.get("photo_id"),
            *[Path(str(value)).stem for value in path_values if value],
            f"url:{row.get('photo_url')}" if row.get("photo_url") else None,
        ]
        for key in keys:
            if key:
                index[str(key)] = row
    return index


def _image_id(row: Mapping[str, Any], manifest: Mapping[str, Mapping[str, Any]]) -> str:
    """Canonical image identity: the candidate manifest id, not a review-local id.

    Curated batches use ids like ``cn2-tc-…`` while the image/candidate is
    ``tc-…``; cross-source dedupe and negative/positive conflict detection must
    use the same key, so path stems that exist in the manifest win.
    """

    stems = []
    for key in ("local_image_path", "image_path", "image_file", "source_image", "path"):
        value = row.get(key)
        if value:
            stems.append(Path(str(value)).stem)
    for stem in stems:
        if stem in manifest:
            return stem
    proposal_id = str(row.get("proposal_id") or "")
    if proposal_id:
        match = re.search(r"(?:^|-)tc-([0-9a-f]+)-", proposal_id)
        if match and f"tc-{match.group(1)}" in manifest:
            return f"tc-{match.group(1)}"
    photo_url = row.get("photo_url")
    if photo_url:
        url_key = f"url:{photo_url}"
        if url_key in manifest:
            return str(manifest[url_key].get("image_id") or manifest[url_key].get("review_image_id"))
    for key in ("image_id", "review_image_id", "candidate_id", "tc_id"):
        value = row.get(key)
        if value:
            value = str(value)
            if value in manifest:
                return value
            if value.startswith(("cn", "rv", "b1-", "b2-", "b3-", "b4-")):
                parts = value.split("-tc-", 1)
                if len(parts) == 2 and f"tc-{parts[1]}" in manifest:
                    return f"tc-{parts[1]}"
    for key in ("image_id", "review_image_id", "candidate_id", "tc_id"):
        value = row.get(key)
        if value:
            return str(value)
    if stems:
        return stems[0]
    sha = row.get("sha256")
    if sha:
        for candidate_id, record in manifest.items():
            if record.get("sha256") == sha:
                return candidate_id
    raise CombineError("record has no image_id/local_image_path/sha256")


def _image_path(
    row: Mapping[str, Any],
    manifest_row: Mapping[str, Any] | None,
    images_root: Path | None,
    suffix_index: Mapping[str, Path],
    image_id: str,
) -> Path | None:
    candidates: list[Path] = []
    for key in ("local_image_path", "image_path", "source_image", "path"):
        value = row.get(key)
        if value:
            candidates.append(Path(str(value)))
    if manifest_row:
        for key in ("local_image_path", "image_path", "path"):
            value = manifest_row.get(key)
            if value:
                candidates.append(Path(str(value)))
    if image_id in suffix_index:
        candidates.append(suffix_index[image_id])
    if images_root is not None:
        for candidate in list(candidates):
            candidates.append(images_root / candidate.name)
            candidates.append(images_root / "images" / candidate.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _split_for(observation_id: str, seed: str, train_ratio: float) -> str:
    value = stable_unit_interval(observation_id, seed)
    return "train" if value < train_ratio else "val"


def _blind_denylist(
    manifest_path: str | Path | None,
    annotations_path: str | Path | None,
) -> dict[str, set[str]]:
    deny = {"observations": set(), "photos": set(), "urls": set(), "sha256": set()}
    if manifest_path and Path(manifest_path).is_file():
        for row in _read_jsonl(manifest_path):
            for key, target in (
                ("observation_id", "observations"),
                ("review_image_id", "photos"),
                ("photo_url", "urls"),
                ("sha256", "sha256"),
            ):
                if row.get(key):
                    deny[target].add(str(row[key]))
    if annotations_path and Path(annotations_path).is_file():
        for row in _read_jsonl(annotations_path):
            for key, target in (
                ("observation_id", "observations"),
                ("review_image_id", "photos"),
                ("photo_url", "urls"),
            ):
                if row.get(key):
                    deny[target].add(str(row[key]))
    return deny


def _check_blind(
    *,
    image_id: str,
    observation_id: str,
    photo_url: str | None,
    sha256: str | None,
    deny: Mapping[str, set[str]],
) -> None:
    matches = []
    if observation_id in deny["observations"]:
        matches.append("observation_id")
    if image_id in deny["photos"]:
        matches.append("photo_id")
    if photo_url and photo_url in deny["urls"]:
        matches.append("photo_url")
    if sha256 and sha256 in deny["sha256"]:
        matches.append("sha256")
    if matches:
        raise BlindLeakageError(
            f"blind-test leakage: image {image_id} ({observation_id}) matches "
            f"frozen blind fields {matches}"
        )


def _parse_class_sources(specs: Sequence[str]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise CombineError(f"--class-source must be CLASS=source[,source], got {spec!r}")
        class_name, sources = spec.split("=", 1)
        class_name = _norm_state(class_name)
        if class_name not in CLASS_IDS:
            raise CombineError(f"--class-source class {class_name!r} is not in {list(CLASS_NAMES)}")
        names = {name.strip() for name in sources.split(",") if name.strip()}
        if not names:
            raise CombineError(f"--class-source {spec!r} names no sources")
        mapping[class_name] = names
    return mapping


def _safe_copy(source: Path, target: Path, mode: str) -> None:
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


def _suffix_index(root: Path | None) -> dict[str, Path]:
    """Index images directly under root and under root/images.

    The CLI default images root is the candidate pool root, while files live in
    ``<root>/images``; curated/negative rows often carry no inline path.
    """

    index: dict[str, Path] = {}
    if root is None or not root.is_dir():
        return index
    directories = [root]
    if (root / "images").is_dir():
        directories.append(root / "images")
    for directory in directories:
        for entry in directory.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
                index.setdefault(entry.stem, entry)
    return index


def combine_sources(
    *,
    out_dir: str | Path,
    assisted: str | Path | None = None,
    curated_needle_glob: str | Path | Sequence[str | Path] | None = None,
    curated_globs: Sequence[str | Path] = (),
    negative_glob: str | Path | Sequence[str | Path] | None = None,
    review_queue: Sequence[str | Path] = (),
    extra_sources: Sequence[str] = (),
    class_sources: Sequence[str] = (),
    candidate_manifest: str | Path | None = None,
    images_root: str | Path | None = None,
    blind_manifest: str | Path | None = None,
    blind_annotations: str | Path | None = None,
    seed: str = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    include_negatives: bool = True,
    copy_mode: str = "copy",
    overlay_count: int = 6,
    allow_missing_sources: bool = False,
    require_assisted: bool = True,
    require_blind_denylist: bool = True,
) -> dict[str, Any]:
    if not (0.0 < train_ratio < 1.0):
        raise CombineError("train_ratio must be between 0 and 1")
    sources = discover_sources(
        assisted=assisted,
        curated_needle_glob=curated_needle_glob,
        curated_globs=curated_globs,
        negative_glob=negative_glob,
        review_queue=review_queue,
        extra_sources=extra_sources,
    )
    if not sources:
        raise CombineError(
            "no source files found; pass --assisted / --curated-needle / "
            "--negative-review / --source"
        )
    if not allow_missing_sources:
        expected = {"curated-needle", "negative-review"}
        if require_assisted:
            expected.add("assisted")
        present = {source.name for source in sources}
        missing = sorted(expected - present)
        if missing:
            raise CombineError(
                f"expected sources are not ready: {missing}. They are being "
                "produced by review agents; pass --allow-missing-sources to "
                "combine the subset that exists."
            )
    images_dir = Path(images_root).expanduser().resolve() if images_root else None
    manifest_path = Path(candidate_manifest) if candidate_manifest else None
    if (manifest_path is None or not manifest_path.is_file()) and images_dir is not None:
        fallback = images_dir / "manifest.jsonl"
        if fallback.is_file():
            manifest_path = fallback
    manifest = _manifest_index(manifest_path)
    suffix_index = _suffix_index(images_dir)
    blind_root = DEFAULT_BLIND_ROOT
    if images_dir is not None:
        fallback_root = images_dir.parent / "blind-test" / "merged"
        if fallback_root.is_dir() and (
            not (blind_root / "manifest.jsonl").is_file()
        ):
            blind_root = fallback_root
    deny_manifest = (
        Path(blind_manifest) if blind_manifest else blind_root / "manifest.jsonl"
    )
    deny_annotations = (
        Path(blind_annotations) if blind_annotations else blind_root / "annotations.jsonl"
    )
    deny = _blind_denylist(deny_manifest, deny_annotations)
    blind_available = bool(deny["observations"] or deny["photos"])
    if require_blind_denylist and not blind_available:
        raise CombineError(
            "blind-test denylist is unavailable; refusing to combine without a "
            "leakage check (pass --allow-missing-blind-denylist only if the "
            "blind benchmark does not exist on this host)"
        )
    class_filter = _parse_class_sources(class_sources)

    records: list[dict[str, Any]] = []
    quarantine: dict[str, list[dict[str, Any]]] = {
        "conflicts": [],
        "needs_review": [],
        "unreviewed": [],
        "invalid": [],
    }
    excluded_non_negative: list[dict[str, Any]] = []
    per_source: dict[str, Counter] = {
        source.name: Counter(
            {
                "files": len(source.paths),
                "rows": 0,
                "accepted_rows": 0,
                "images": 0,
                "boxes": 0,
                "negatives": 0,
                "excluded": 0,
                "duplicate_boxes_merged": 0,
                "excluded_non_negative": 0,
            }
        )
        for source in sources
    }
    for source in sources:
        for path in source.paths:
            for index, row in enumerate(_read_jsonl(path), start=1):
                per_source[source.name]["rows"] += 1
                provenance = {
                    "source": source.name,
                    "file": str(path.resolve()),
                    "sha256": _sha256(path),
                    "row": index,
                    "record": {
                        key: row.get(key)
                        for key in (
                            "policy",
                            "policy_id",
                            "policy_action",
                            "model",
                            "model_id",
                            "confidence",
                            "detector_score",
                            "detector_provenance",
                            "verifier_provenance",
                            "verifier_verdict",
                            "human_truth",
                            "prompt",
                            "provenance",
                            "reviewer",
                            "review_state",
                            "source",
                            "verdict",
                        )
                        if row.get(key) is not None
                    },
                }
                state = _excluded_state(row)
                if source.name == "negative-review" and not _is_negative(row):
                    # Negative-review batches contain reviewed positives/exclusions
                    # alongside true negatives. negative=false rows are neither
                    # training truth nor errors; count and skip them.
                    per_source[source.name]["excluded_non_negative"] += 1
                    excluded_non_negative.append(
                        {
                            "source": source.name,
                            "file": str(path.resolve()),
                            "row": index,
                            "image_id": row.get("image_id") or row.get("review_image_id"),
                            "observation_id": row.get("observation_id"),
                            "review_state": row.get("review_state"),
                            "reason": "negative_review_row_negative_false",
                            "row_payload": row,
                        }
                    )
                    continue
                if state is not None:
                    per_source[source.name]["excluded"] += 1
                    record = {
                        "source": source.name,
                        "file": str(path.resolve()),
                        "row": index,
                        "reason": state,
                        "row_payload": row,
                    }
                    quarantine["needs_review" if state == "needs_review" else "unreviewed"].append(record)
                    continue
                if not _accepted(row):
                    per_source[source.name]["excluded"] += 1
                    quarantine["unreviewed"].append(
                        {
                            "source": source.name,
                            "file": str(path.resolve()),
                            "row": index,
                            "reason": "not_accepted",
                            "row_payload": row,
                        }
                    )
                    continue
                try:
                    image_id = _image_id(row, manifest)
                    manifest_row = manifest.get(image_id) or {}
                    observation_id = str(
                        row.get("observation_id")
                        or manifest_row.get("observation_id")
                        or ""
                    )
                    if not observation_id:
                        raise CombineError(
                            f"{source.name}:{path.name}:{index}: no observation_id"
                        )
                    image_path = _image_path(
                        row, manifest_row, images_dir, suffix_index, image_id
                    )
                    if image_path is None:
                        raise CombineError(
                            f"{source.name}:{path.name}:{index}: image not found for {image_id}"
                        )
                    with image_path.open("rb") as stream:
                        image_sha = _sha256_bytes(stream.read())
                    expected_sha = row.get("sha256") or manifest_row.get("sha256")
                    if expected_sha and image_sha != expected_sha:
                        raise CombineError(
                            f"{source.name}:{path.name}:{index}: sha256 mismatch for "
                            f"{image_id} ({image_sha} != {expected_sha})"
                        )
                    photo_url = row.get("photo_url") or manifest_row.get("photo_url")
                    _check_blind(
                        image_id=image_id,
                        observation_id=observation_id,
                        photo_url=str(photo_url) if photo_url else None,
                        sha256=image_sha,
                        deny=deny,
                    )
                    raw_labels = list(_labels(row))
                    negative = _is_negative(row)
                    labels: list[dict[str, Any]] = []
                    for position, raw_label in enumerate(raw_labels):
                        where = f"{source.name}:{path.name}:{index}:label[{position}]"
                        label_state = _norm_state(
                            raw_label.get("review_state") or raw_label.get("status")
                        )
                        if (
                            raw_label.get("needs_review") is True
                            or label_state in EXCLUDE_STATES
                        ):
                            quarantine["needs_review"].append(
                                {
                                    "source": source.name,
                                    "file": str(path.resolve()),
                                    "row": index,
                                    "label_index": position,
                                    "reason": "label_needs_review",
                                    "label": raw_label,
                                }
                            )
                            continue
                        class_name = _label_class(raw_label, where)
                        allowed = class_filter.get(class_name)
                        if allowed is not None and source.name not in allowed:
                            continue
                        box = _bbox(
                            raw_label.get("bbox")
                            or raw_label.get("bbox_xyxy")
                            or raw_label.get("bbox_xyxy_norm"),
                            where,
                        )
                        labels.append(
                            {
                                "class_name": class_name,
                                "class_id": CLASS_IDS[class_name],
                                "bbox": box,
                                "confidence": raw_label.get("confidence"),
                                "provenance": dict(raw_label.get("provenance") or {}),
                            }
                        )
                    if negative and labels:
                        quarantine["conflicts"].append(
                            {
                                "source": source.name,
                                "file": str(path.resolve()),
                                "row": index,
                                "image_id": image_id,
                                "reason": "negative_and_positive_in_same_record",
                                "resolved": "positive_wins",
                                "labels": len(labels),
                                "row_payload": row,
                            }
                        )
                        negative = False
                        per_source[source.name]["excluded"] += 1
                    if not labels and not negative:
                        quarantine["invalid"].append(
                            {
                                "source": source.name,
                                "file": str(path.resolve()),
                                "row": index,
                                "image_id": image_id,
                                "reason": "accepted_record_with_no_labels_and_not_negative",
                                "row_payload": row,
                            }
                        )
                        per_source[source.name]["excluded"] += 1
                        continue
                    if not labels and not negative:
                        continue
                    records.append(
                        {
                            "image_id": image_id,
                            "observation_id": observation_id,
                            "image_path": image_path,
                            "sha256": image_sha,
                            "photo_url": photo_url,
                            "labels": labels,
                            "negative": negative and not labels,
                            "provenance": provenance,
                            "source_name": source.name,
                        }
                    )
                    per_source[source.name]["accepted_rows"] += 1
                except BlindLeakageError:
                    raise
                except CombineError as exc:
                    quarantine["invalid"].append(
                        {
                            "source": source.name,
                            "file": str(path.resolve()),
                            "row": index,
                            "reason": "invalid",
                            "error": str(exc),
                            "row_payload": row,
                        }
                    )
                    per_source[source.name]["excluded"] += 1

    if not records:
        raise CombineError("no accepted records; nothing to combine")

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_image[record["image_id"]].append(record)

    images_out: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    boxes_per_class: Counter[str] = Counter()
    split_stats: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    merges: list[dict[str, Any]] = []
    observation_splits: dict[str, str] = {}

    for image_id in sorted(by_image):
        group = by_image[image_id]
        observation_id = group[0]["observation_id"]
        for record in group:
            if record["observation_id"] != observation_id:
                raise CombineError(
                    f"image {image_id} is assigned to multiple observations "
                    f"({observation_id}, {record['observation_id']})"
                )
        positives = [record for record in group if record["labels"]]
        negatives = [record for record in group if record["negative"]]
        if positives:
            for negative_record in negatives:
                quarantine["conflicts"].append(
                    {
                        "source": negative_record["source_name"],
                        "file": negative_record["provenance"]["file"],
                        "row": negative_record["provenance"]["row"],
                        "image_id": image_id,
                        "reason": "negative_for_positive_image",
                        "resolved": "positive_wins",
                        "row_payload": negative_record["provenance"]["record"],
                    }
                )
            negatives = []
        if not positives and negatives and not include_negatives:
            continue
        split = observation_splits.setdefault(
            observation_id, _split_for(observation_id, seed, train_ratio)
        )
        suffix = group[0]["image_path"].suffix.lower()
        relative = f"images/{split}/{image_id}{suffix}"
        if not positives and negatives:
            images_out.append(
                {
                    "image": relative,
                    "photo_id": image_id,
                    "observation_id": observation_id,
                    "negative": True,
                    "needs_review": False,
                    "included_in_eval": True,
                    "source": {
                        "merged": True,
                        "sources": [record["provenance"] for record in negatives],
                        "photo_url": group[0]["photo_url"],
                        "source_sha256": group[0]["sha256"],
                    },
                    "boxes": [],
                }
            )
            continue
        if not positives:
            continue

        merged: dict[tuple[int, tuple[float, float, float, float]], dict[str, Any]] = {}
        for record in positives:
            for label in record["labels"]:
                key = (
                    label["class_id"],
                    tuple(round(value, 6) for value in label["bbox"]),
                )
                entry = merged.get(key)
                source_ref = {
                    **record["provenance"],
                    "confidence": label["confidence"],
                    "label_provenance": label["provenance"],
                }
                if entry is None:
                    merged[key] = {
                        "class_name": label["class_name"],
                        "class_id": label["class_id"],
                        "bbox": label["bbox"],
                        "sources": [source_ref],
                    }
                else:
                    entry["sources"].append(source_ref)
                    per_source[record["source_name"]]["duplicate_boxes_merged"] += 1
        for entry in merged.values():
            if len(entry["sources"]) > 1:
                merges.append(
                    {
                        "image_id": image_id,
                        "class_name": entry["class_name"],
                        "bbox": list(entry["bbox"]),
                        "sources": [
                            {
                                "source": source["source"],
                                "file": source["file"],
                                "row": source["row"],
                            }
                            for source in entry["sources"]
                        ],
                    }
                )
        images_out.append(
            {
                "image": relative,
                "photo_id": image_id,
                "observation_id": observation_id,
                "negative": False,
                "needs_review": False,
                "included_in_eval": True,
                "source": {
                    "merged": True,
                    "sources": [record["provenance"] for record in positives],
                    "photo_url": group[0]["photo_url"],
                    "source_sha256": group[0]["sha256"],
                },
                "boxes": [
                    {
                        "class_id": entry["class_id"],
                        "class_name": entry["class_name"],
                        "x0": entry["bbox"][0],
                        "y0": entry["bbox"][1],
                        "x1": entry["bbox"][2],
                        "y1": entry["bbox"][3],
                    }
                    for entry in merged.values()
                ],
            }
        )
        boxes_per_class.update(entry["class_name"] for entry in merged.values())
        provenance_rows.extend(
            {
                "schema": "arq.anatomy4.combined-box/v1",
                "contract": CONTRACT,
                "image": relative,
                "photo_id": image_id,
                "observation_id": observation_id,
                "class_id": entry["class_id"],
                "class_name": entry["class_name"],
                "xyxy_norm": list(entry["bbox"]),
                "sources": entry["sources"],
            }
            for entry in merged.values()
        )

    split_counts = Counter(observation_splits.values())
    if split_counts.get("train", 0) == 0 or split_counts.get("val", 0) == 0:
        raise CombineError(
            f"deterministic split produced train={split_counts.get('train', 0)}, "
            f"val={split_counts.get('val', 0)} observations; need both non-empty"
        )

    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for entry in images_out:
        split = observation_splits[entry["observation_id"]]
        source_path = by_image[entry["photo_id"]][0]["image_path"]
        target = root / entry["image"]
        _safe_copy(source_path, target, copy_mode)
        label_lines = [
            f"{box['class_id']} {box['x0']:.6f} {box['y0']:.6f} "
            f"{box['x1']:.6f} {box['y1']:.6f}"
            for box in entry["boxes"]
        ]
        label_path = root / "labels" / split / f"{entry['photo_id']}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(("\n".join(label_lines) + "\n") if label_lines else "")
        split_stats[split]["images"] += 1
        split_stats[split]["boxes"] += len(entry["boxes"])
        split_stats[split]["negatives"] += 1 if entry["negative"] else 0
    dataset_json = {
        "schema": "arq.anatomy4.dataset/v1",
        "contract": CONTRACT,
        "classes": list(CLASS_NAMES),
        "merge": {
            "schema": COMBINE_SCHEMA,
            "seed": seed,
            "train_ratio": train_ratio,
            "sources": [source.name for source in sources],
            "class_sources": {name: sorted(values) for name, values in class_filter.items()},
            "blind_denylist": {
                "observations": len(deny["observations"]),
                "photos": len(deny["photos"]),
                "urls": len(deny["urls"]),
                "sha256": len(deny["sha256"]),
                "available": blind_available,
            },
        },
        "images": images_out,
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=1) + "\n")
    with (root / "provenance.jsonl").open("w") as handle:
        for row in provenance_rows:
            handle.write(json.dumps(row) + "\n")
    quarantine_dir = root / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in quarantine.items():
        with (quarantine_dir / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    with (root / "excluded_non_negative.jsonl").open("w") as handle:
        for row in excluded_non_negative:
            handle.write(json.dumps(row) + "\n")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                f"names: {json.dumps(list(CLASS_NAMES))}",
                "",
            ]
        )
    )

    audit = validate_dataset(root, require_test=False, require_val=True)
    audit_path = write_audit(audit, root / "dataset-audit.json")
    (root / "dataset-audit.md").write_text(render_audit_markdown(audit))
    overlays = render_gt_overlays(root, root / "overlays", count=overlay_count)

    for source in sources:
        counts = per_source[source.name]
        counts["images"] = len(
            {
                record["image_id"]
                for record in records
                if record["source_name"] == source.name
            }
        )
        counts["boxes"] = sum(
            len(record["labels"])
            for record in records
            if record["source_name"] == source.name
        )
        counts["negatives"] = sum(
            1
            for record in records
            if record["source_name"] == source.name and record["negative"]
        )
    report = {
        "schema": COMBINE_SCHEMA,
        "contract": CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            source.name: {
                "paths": [str(path.resolve()) for path in source.paths],
                "sha256": [_sha256(path) for path in source.paths],
            }
            for source in sources
        },
        "selection": {
            "seed": seed,
            "train_ratio": train_ratio,
            "include_negatives": include_negatives,
            "class_sources": {name: sorted(values) for name, values in class_filter.items()},
            "copy_mode": copy_mode,
        },
        "counts": {
            "images": len(images_out),
            "positives": sum(1 for entry in images_out if not entry["negative"]),
            "negatives": sum(1 for entry in images_out if entry["negative"]),
            "boxes": sum(boxes_per_class.values()),
            "boxes_per_class": {name: boxes_per_class.get(name, 0) for name in CLASS_NAMES},
            "observations": len(observation_splits),
            "splits": {
                split: {
                    "observations": sum(1 for value in observation_splits.values() if value == split),
                    "images": split_stats[split]["images"],
                    "boxes": split_stats[split]["boxes"],
                    "negatives": split_stats[split]["negatives"],
                }
                for split in ("train", "val")
            },
        },
        "per_source": {
            name: dict(counts) for name, counts in per_source.items()
        },
        "dedupe": {
            "merged_boxes": len(merges),
            "merges": merges,
        },
        "conflicts": {
            "quarantined": len(quarantine["conflicts"]),
            "records": quarantine["conflicts"],
        },
        "excluded": {
            "needs_review": len(quarantine["needs_review"]),
            "unreviewed": len(quarantine["unreviewed"]),
            "invalid": len(quarantine["invalid"]),
            "non_negative": len(excluded_non_negative),
            "non_negative_sidecar": str(root / "excluded_non_negative.jsonl"),
        },
        "blind_check": {
            "available": blind_available,
            "denylist": {
                "observations": len(deny["observations"]),
                "photos": len(deny["photos"]),
                "urls": len(deny["urls"]),
                "sha256": len(deny["sha256"]),
            },
            "matches": 0,
        },
        "split_isolation": audit["split_isolation"],
        "audit": str(audit_path),
        "overlays": overlays,
        "status": audit["status"],
        "blocking": audit["blocking"],
    }
    (root / "merge-report.json").write_text(json.dumps(report, indent=1) + "\n")
    (root / "merge-report.md").write_text(_render_report_markdown(report))
    if audit["blocking"]:
        codes = sorted({item["code"] for item in audit["errors"]})
        raise CombineError(
            f"combined dataset failed validation {codes}; inspect {audit_path}"
        )
    return report


def _render_report_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Anatomy4 mixed-source combine report",
        "",
        f"- contract: {report['contract']}",
        f"- status: **{report['status']}** · split isolation: {report['split_isolation']}",
        f"- images: {counts['images']} ({counts['positives']} positive, "
        f"{counts['negatives']} negative)",
        f"- boxes: {counts['boxes']}",
        f"- seed: `{report['selection']['seed']}` · train ratio "
        f"{report['selection']['train_ratio']}",
        "",
        "| class | boxes |",
        "| --- | ---: |",
    ]
    for name in CLASS_NAMES:
        lines.append(f"| {name} | {counts['boxes_per_class'].get(name, 0)} |")
    lines += [
        "",
        "## Splits",
        "",
        "| split | observations | images | boxes | negatives |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "val"):
        item = counts["splits"][split]
        lines.append(
            f"| {split} | {item['observations']} | {item['images']} | "
            f"{item['boxes']} | {item['negatives']} |"
        )
    lines += [
        "",
        "## Per source",
        "",
        "| source | files | rows | accepted rows | images | boxes | negatives | "
        "duplicates merged | excluded | excluded non-negative |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in report["per_source"].items():
        lines.append(
            f"| {name} | {item.get('files', 0)} | {item.get('rows', 0)} | "
            f"{item.get('accepted_rows', 0)} | {item.get('images', 0)} | "
            f"{item.get('boxes', 0)} | {item.get('negatives', 0)} | "
            f"{item.get('duplicate_boxes_merged', 0)} | {item.get('excluded', 0)} | "
            f"{item.get('excluded_non_negative', 0)} |"
        )
    lines += [
        "",
        f"- deduped boxes: {report['dedupe']['merged_boxes']}",
        f"- quarantined conflicts: {report['conflicts']['quarantined']}",
        f"- excluded needs_review: {report['excluded']['needs_review']} · "
        f"unreviewed: {report['excluded']['unreviewed']} · "
        f"invalid: {report['excluded']['invalid']} · "
        f"non_negative: {report['excluded']['non_negative']}",
        f"- blind denylist available: {report['blind_check']['available']} · "
        f"matches: {report['blind_check']['matches']}",
    ]
    if report["overlays"]:
        lines += ["", "## GT overlays", ""]
        for overlay in report["overlays"]:
            lines.append(
                f"- `{overlay['photo_id']}` ({overlay['boxes']} boxes): "
                f"{overlay['overlay']}"
            )
    return "\n".join(lines) + "\n"
