# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""GT-vs-prediction overlays and per-class contact sheets.

Colors: each anatomy4 class has a stable color for ground truth. Prediction
boxes are colored by match status at the requested IoU/confidence (TP green,
FP red, FN orange, dashed). This is the visual audit surface for the metric
numbers, not a substitute for them.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .contracts import CLASS_NAMES, CONTRACT, iou_xyxy
from .metrics import PixelBox, PixelDetection, match_detections

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "leaf": (46, 158, 79),
    "needle": (31, 111, 235),
    "flower": (217, 72, 139),
    "inflorescence": (224, 161, 0),
}
PALETTE: tuple[tuple[int, int, int], ...] = (
    (46, 158, 79),
    (31, 111, 235),
    (217, 72, 139),
    (224, 161, 0),
    (139, 92, 246),
    (14, 165, 233),
    (234, 88, 12),
    (20, 184, 166),
    (244, 63, 94),
    (132, 204, 22),
    (168, 85, 247),
    (250, 204, 21),
)
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "tp": (34, 197, 94),
    "fp": (239, 68, 68),
    "fn": (245, 158, 11),
}


def class_color(name: str, index: int = 0) -> tuple[int, int, int]:
    """Stable color for any class name (anatomy4 keeps its canonical colors)."""

    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    if index < len(PALETTE):
        return PALETTE[index]
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (64 + digest[0] % 160, 64 + digest[1] % 160, 64 + digest[2] % 160)


def _class_name(item: Any, classes: Sequence[str]) -> str:
    class_id = int(item.class_id)
    if 0 <= class_id < len(classes):
        return str(classes[class_id])
    return str(getattr(item, "class_name", class_id))


def _font(size: int = 13) -> ImageFont.ImageFont:
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    color: tuple[int, int, int],
    width: int = 2,
    dash: int = 7,
) -> None:
    x0, y0, x1, y1 = box
    for start, end, horizontal, fixed in (
        (x0, x1, True, y0),
        (x0, x1, True, y1),
        (y0, y1, False, x0),
        (y0, y1, False, x1),
    ):
        position = start
        while position < end:
            segment_end = min(position + dash, end)
            if horizontal:
                draw.line((position, fixed, segment_end, fixed), fill=color, width=width)
            else:
                draw.line((fixed, position, fixed, segment_end), fill=color, width=width)
            position += dash * 2


def _text_with_backdrop(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    except Exception:
        left, top, right, bottom = x, y, x + 8 * len(text), y + 12
    pad = 2
    draw.rectangle(
        (left - pad, top - pad, right + pad, bottom + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=color, font=font)


def annotate_image(
    image: Image.Image,
    ground_truth: Sequence[PixelBox],
    detections: Sequence[PixelDetection],
    *,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    show_labels: bool = True,
    line_width: int = 3,
    classes: Sequence[str] = CLASS_NAMES,
) -> tuple[Image.Image, dict]:
    """Draw GT and predictions; return the image and an audit summary."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(max(11, min(16, canvas.width // 45)))
    summary = {"tp": 0, "fp": 0, "fn": 0}

    for box in ground_truth:
        name = _class_name(box, classes)
        color = class_color(name)
        draw.rectangle(box.xyxy, outline=color, width=line_width)
        if show_labels:
            _text_with_backdrop(
                draw,
                (box.x0 + 2, max(0, box.y0 - 16)),
                f"GT {name}",
                color,
                font,
            )

    dets = [det for det in detections if det.confidence >= conf_threshold]
    match = match_detections(ground_truth, dets, iou_threshold, class_aware=True)
    for index, detection in enumerate(dets):
        matched = match.det_to_gt[index] is not None
        status = "tp" if matched else "fp"
        summary[status] += 1
        color = STATUS_COLORS[status]
        draw.rectangle(detection.xyxy, outline=color, width=2)
        if show_labels:
            _text_with_backdrop(
                draw,
                (detection.x0 + 2, detection.y1 + 1),
                f"{_class_name(detection, classes)} {detection.confidence:.2f} {status.upper()}",
                color,
                font,
            )
    for gt_index in match.unmatched_gt:
        box = ground_truth[gt_index]
        summary["fn"] += 1
        _dashed_rectangle(draw, box.xyxy, STATUS_COLORS["fn"], width=2)
        if show_labels:
            _text_with_backdrop(
                draw,
                (box.x0 + 2, min(canvas.height - 16, box.y1 + 1)),
                f"MISSED {_class_name(box, classes)}",
                STATUS_COLORS["fn"],
                font,
            )
    return canvas, summary


def render_overlay(
    image_path: str | Path,
    ground_truth: Sequence[PixelBox],
    detections: Sequence[PixelDetection],
    out_path: str | Path,
    *,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    max_side: int = 1280,
    classes: Sequence[str] = CLASS_NAMES,
) -> dict:
    source = Path(image_path)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        scale = 1.0
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )
        drawn, summary = annotate_image(
            image,
            ground_truth,
            detections,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            classes=classes,
        )
        drawn.save(target, quality=88)
    return {
        "image": str(source),
        "overlay": str(target),
        "width": drawn.width,
        "height": drawn.height,
        "scale": scale,
        **summary,
    }


def _thumb(image: Image.Image, size: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((size, size), Image.LANCZOS)
    return copy


def render_contact_sheets(
    records: Iterable[Mapping],
    out_dir: str | Path,
    *,
    per_sheet: int = 12,
    columns: int = 3,
    thumb_size: int = 320,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    classes: Sequence[str] = CLASS_NAMES,
) -> list[dict]:
    """One sheet family per class (plus ``all``), paginated at ``per_sheet``."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    font = _font(12)
    tiles: dict[str, list[tuple[Mapping, Image.Image, dict]]] = {
        name: [] for name in (*classes, "all")
    }
    for record in records:
        image_path = Path(record["image"])
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        except OSError:
            continue
        rendered, summary = annotate_image(
            image,
            record["ground_truth"],
            record["detections"],
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            classes=classes,
        )
        present = {
            _class_name(box, classes) for box in record["ground_truth"]
        } | {
            _class_name(det, classes)
            for det in record["detections"]
            if det.confidence >= conf_threshold
        }
        for name in present:
            if name in tiles:
                tiles[name].append((record, rendered, summary))
        tiles["all"].append((record, rendered, summary))

    written: list[dict] = []
    for name, items in tiles.items():
        for page in range(0, len(items), per_sheet):
            chunk = items[page : page + per_sheet]
            rows = (len(chunk) + columns - 1) // columns
            sheet = Image.new(
                "RGB",
                (columns * thumb_size, rows * (thumb_size + 22) + 18),
                (18, 18, 18),
            )
            draw = ImageDraw.Draw(sheet)
            heading = f"{CONTRACT} · {name} · {len(items)} images"
            draw.text((8, 4), heading, fill=(240, 240, 240), font=font)
            for position, (record, rendered, summary) in enumerate(chunk):
                row, column = divmod(position, columns)
                tile = _thumb(rendered, thumb_size)
                x = column * thumb_size
                y = 18 + row * (thumb_size + 22)
                sheet.paste(tile, (x, y))
                caption = (
                    f"{Path(record['image']).stem} · "
                    f"TP {summary['tp']} FP {summary['fp']} FN {summary['fn']}"
                )
                draw.text((x + 4, y + thumb_size + 4), caption, fill=(200, 200, 200), font=font)
            target = output / f"contact-{name}-{page // per_sheet:02d}.jpg"
            sheet.save(target, quality=86)
            written.append(
                {
                    "class": name,
                    "page": page // per_sheet,
                    "images": len(chunk),
                    "path": str(target),
                }
            )
    return written


# ── Stratified visual sampling ───────────────────────────────────────────────


def _xyxy(item: Any) -> tuple[float, float, float, float]:
    if hasattr(item, "xyxy"):
        return tuple(float(value) for value in item.xyxy)  # type: ignore[return-value]
    if isinstance(item, Mapping):
        if "xyxy" in item:
            return tuple(float(value) for value in item["xyxy"])  # type: ignore[return-value]
        if "bbox" in item:
            return tuple(float(value) for value in item["bbox"])  # type: ignore[return-value]
        return (
            float(item["x0"]),
            float(item["y0"]),
            float(item["x1"]),
            float(item["y1"]),
        )
    raise TypeError(f"unsupported box {item!r}")


def _class_id(item: Any, classes: Sequence[str]) -> int:
    if hasattr(item, "class_id"):
        return int(item.class_id)
    if isinstance(item, Mapping):
        if item.get("class_id") is not None:
            return int(item["class_id"])
        if item.get("class") is not None:
            return list(classes).index(str(item["class"]))
    raise TypeError(f"unsupported box {item!r}")


def _confidence(item: Any) -> float:
    return float(getattr(item, "confidence", 0.0))


def _norm_bbox(
    xyxy: Sequence[float], width: float, height: float
) -> list[float]:
    x0, y0, x1, y1 = (float(value) for value in xyxy)
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(min(max(x0 / width, 0.0), 1.0), 6),
        round(min(max(y0 / height, 0.0), 1.0), 6),
        round(min(max(x1 / width, 0.0), 1.0), 6),
        round(min(max(y1 / height, 0.0), 1.0), 6),
    ]


def classify_eval_record(
    record: Mapping[str, Any],
    *,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    classes: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Bucket one evaluated image into TP/FP/FN/TN plus confusion cells.

    Matching is greedy by descending confidence. A same-class overlap at or
    above ``iou_threshold`` is a true positive. An unmatched detection is a
    false positive; when it overlaps an unmatched ground-truth box of another
    class it also records that ``(actual, predicted)`` confusion. Unmatched
    ground truth is a false negative. An image with no ground truth and no
    detections at the operating point is a clean negative.
    """

    ground_truth = list(record.get("ground_truth") or [])
    detections = [
        det for det in (record.get("detections") or [])
        if _confidence(det) >= conf_threshold
    ]
    width = float(record.get("width") or 0.0)
    height = float(record.get("height") or 0.0)

    gt_taken = [False] * len(ground_truth)
    det_rows: list[dict[str, Any]] = []
    classes_tp: list[str] = []
    classes_fp: list[str] = []
    classes_fn: list[str] = []
    confusion_pairs: list[list[str]] = []

    order = sorted(
        range(len(detections)),
        key=lambda index: (-_confidence(detections[index]), index),
    )
    for det_index in order:
        detection = detections[det_index]
        det_class_id = _class_id(detection, classes)
        det_name = str(classes[det_class_id]) if 0 <= det_class_id < len(classes) else str(det_class_id)
        det_box = _xyxy(detection)

        best_index: int | None = None
        best_iou = 0.0
        for gt_index, gt in enumerate(ground_truth):
            if gt_taken[gt_index]:
                continue
            if _class_id(gt, classes) != det_class_id:
                continue
            overlap = iou_xyxy(det_box, _xyxy(gt))
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index

        if best_index is not None and best_iou >= iou_threshold:
            gt_taken[best_index] = True
            classes_tp.append(det_name)
            det_rows.append(
                {
                    "class": det_name,
                    "bbox": _norm_bbox(det_box, width, height),
                    "confidence": round(_confidence(detection), 6),
                    "match": "tp",
                }
            )
            continue

        actual_name: str | None = None
        confusion_iou = 0.0
        for gt_index, gt in enumerate(ground_truth):
            if gt_taken[gt_index]:
                continue
            overlap = iou_xyxy(det_box, _xyxy(gt))
            if overlap >= iou_threshold and overlap > confusion_iou:
                confusion_iou = overlap
                actual_name = str(classes[_class_id(gt, classes)])
        classes_fp.append(det_name)
        row = {
            "class": det_name,
            "bbox": _norm_bbox(det_box, width, height),
            "confidence": round(_confidence(detection), 6),
            "match": "fp",
        }
        if actual_name is not None:
            pair = [actual_name, det_name]
            if pair not in confusion_pairs:
                confusion_pairs.append(pair)
            row["confused_with"] = actual_name
        det_rows.append(row)

    gt_rows: list[dict[str, Any]] = []
    for gt_index, gt in enumerate(ground_truth):
        gt_class_id = _class_id(gt, classes)
        gt_name = str(classes[gt_class_id]) if 0 <= gt_class_id < len(classes) else str(gt_class_id)
        gt_rows.append(
            {
                "class": gt_name,
                "bbox": _norm_bbox(_xyxy(gt), width, height),
            }
        )
        if not gt_taken[gt_index]:
            classes_fn.append(gt_name)

    tp, fp, fn = len(classes_tp), len(classes_fp), len(classes_fn)
    tn = 1 if not ground_truth and not detections else 0
    buckets: list[str] = []
    if tp:
        buckets.append("tp")
    if fp:
        buckets.append("fp")
    if fn:
        buckets.append("fn")
    if tn:
        buckets.append("tn")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "buckets": buckets,
        "classes_tp": classes_tp,
        "classes_fp": classes_fp,
        "classes_fn": classes_fn,
        "confusion_pairs": confusion_pairs,
        "ground_truth": gt_rows,
        "detections": det_rows,
    }


def _item_key(record: Mapping[str, Any], index: int) -> str:
    return str(record.get("item_key") or record.get("stem") or index)


def build_eval_samples_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    contract: str = CONTRACT,
    model_id: str | None = None,
    dataset_id: str | None = None,
    split: str = "test",
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    classes: Sequence[str] = CLASS_NAMES,
    sample_per_bucket: int = 24,
    seed: int = 42,
    murals: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Full-split bucket counts plus a bounded, deterministic sample sheet."""

    classes = list(classes)
    classified: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = []
    for index, record in enumerate(records):
        key = _item_key(record, index)
        classified.append((key, record, classify_eval_record(
            record,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            classes=classes,
        )))

    total_images = len(classified)
    tp_images = sum(1 for _, _, item in classified if item["tp"] > 0)
    fp_images = sum(1 for _, _, item in classified if item["fp"] > 0)
    fn_images = sum(1 for _, _, item in classified if item["fn"] > 0)
    tn_images = sum(1 for _, _, item in classified if item["tn"])
    by_class: dict[str, dict[str, int]] = {}
    for name in classes:
        by_class[name] = {
            "tp": sum(item["classes_tp"].count(name) for _, _, item in classified),
            "fp": sum(item["classes_fp"].count(name) for _, _, item in classified),
            "fn": sum(item["classes_fn"].count(name) for _, _, item in classified),
        }

    pools: list[list[tuple[str, Mapping[str, Any], dict[str, Any]]]] = []
    for name in classes:
        for kind in ("tp", "fp", "fn"):
            pools.append(
                [entry for entry in classified if name in entry[2][f"classes_{kind}"]]
            )
    confusion_cells: set[tuple[str, str]] = set()
    for _, _, item in classified:
        for pair in item["confusion_pairs"]:
            confusion_cells.add((pair[0], pair[1]))
    for actual, predicted in sorted(confusion_cells):
        pools.append(
            [
                entry
                for entry in classified
                if [actual, predicted] in entry[2]["confusion_pairs"]
            ]
        )
    pools.append([entry for entry in classified if entry[2]["tn"]])

    rng = random.Random(seed)
    selected: dict[str, tuple[str, Mapping[str, Any], dict[str, Any]]] = {}
    per_bucket = max(0, int(sample_per_bucket))
    for pool in pools:
        unique = {entry[0]: entry for entry in pool}
        ordered = [unique[key] for key in sorted(unique)]
        if not ordered or per_bucket == 0:
            continue
        take = ordered if len(ordered) <= per_bucket else rng.sample(ordered, per_bucket)
        for entry in take:
            selected.setdefault(entry[0], entry)

    capped = [selected[key] for key in sorted(selected)][:240]
    samples: list[dict[str, Any]] = []
    for key, record, item in capped:
        samples.append(
            {
                "item_key": key,
                "observation_id": record.get("observation_id"),
                "photo_url": record.get("photo_url") or record.get("url"),
                "tp": item["tp"],
                "fp": item["fp"],
                "fn": item["fn"],
                "buckets": item["buckets"],
                "classes_tp": item["classes_tp"],
                "classes_fp": item["classes_fp"],
                "classes_fn": item["classes_fn"],
                "confusion_pairs": item["confusion_pairs"],
                "ground_truth": item["ground_truth"],
                "detections": item["detections"],
            }
        )

    return {
        "schema": "arq.stage1.eval_samples/v1",
        "contract": contract,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "split": split,
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "class_names": classes,
        "bucket_counts": {
            "total_images": total_images,
            "sampled_images": len(samples),
            "tp_images": tp_images,
            "fp_images": fp_images,
            "fn_images": fn_images,
            "tn_images": tn_images,
            "by_class": by_class,
        },
        "samples": samples,
        "murals": [dict(meta) for meta in murals],
    }


def _slug(text: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in str(text))
    return "-".join(part for part in safe.split("-") if part) or "class"


def _save_bounded_jpeg(sheet: Image.Image, target: Path, max_bytes: int = 4 * 1024 * 1024) -> None:
    for quality in (84, 72, 60, 45):
        sheet.save(target, quality=quality, optimize=True)
        if target.stat().st_size < max_bytes:
            return
    scale = 0.75
    while target.stat().st_size >= max_bytes and scale > 0.2:
        resized = sheet.resize(
            (max(1, int(sheet.width * scale)), max(1, int(sheet.height * scale))),
            Image.LANCZOS,
        )
        resized.save(target, quality=55, optimize=True)
        scale -= 0.15


def render_stratified_murals(
    records: Iterable[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    contract: str = CONTRACT,
    classes: Sequence[str] = CLASS_NAMES,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    tiles_per_mural: int = 12,
    columns: int = 4,
    thumb_size: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Render a bounded overview sheet, an errors sheet, and one sheet/class."""

    classes = list(classes)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    font = _font(12)
    rng = random.Random(seed)

    classified: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = []
    for index, record in enumerate(records):
        classified.append(
            (
                _item_key(record, index),
                record,
                classify_eval_record(
                    record,
                    iou_threshold=iou_threshold,
                    conf_threshold=conf_threshold,
                    classes=classes,
                ),
            )
        )

    def _pick(pool: list[tuple[str, Mapping[str, Any], dict[str, Any]]]) -> list[tuple[str, Mapping[str, Any], dict[str, Any]]]:
        unique = {entry[0]: entry for entry in pool}
        ordered = [unique[key] for key in sorted(unique)]
        if len(ordered) <= tiles_per_mural:
            return ordered
        return rng.sample(ordered, max(0, int(tiles_per_mural)))

    overview_pool: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    strata = [
        sorted([e for e in classified if name in e[2][f"classes_{kind}"]], key=lambda e: e[0])
        for kind in ("tp", "fp", "fn")
        for name in classes
    ]
    strata.append(sorted([e for e in classified if e[2]["tn"]], key=lambda e: e[0]))
    depth = 0
    while depth < max((len(group) for group in strata), default=0):
        for group in strata:
            if depth < len(group) and group[depth][0] not in seen:
                overview_pool.append(group[depth])
                seen.add(group[depth][0])
        depth += 1
        if len(overview_pool) >= tiles_per_mural * 2:
            break

    sheets: list[tuple[str, str, str | None, list[tuple[str, Mapping[str, Any], dict[str, Any]]]]] = [
        ("mural-overview.jpg", "Overview · stratified sample", None, _pick(overview_pool)),
        (
            "mural-errors.jpg",
            "Errors · false alarms and missed boxes",
            None,
            _pick([e for e in classified if e[2]["fp"] > 0 or e[2]["fn"] > 0]),
        ),
    ]
    for name in classes:
        pool = [
            e
            for e in classified
            if name in e[2]["classes_tp"]
            or name in e[2]["classes_fp"]
            or name in e[2]["classes_fn"]
        ]
        chosen = _pick(pool)
        if chosen:
            sheets.append(
                (
                    f"mural-{_slug(name)}.jpg",
                    f"Class · {name}",
                    name,
                    chosen,
                )
            )

    written: list[dict[str, Any]] = []
    for filename, title, class_name, chosen in sheets:
        if not chosen:
            continue
        columns_used = max(1, int(columns))
        rows = (len(chosen) + columns_used - 1) // columns_used
        sheet = Image.new(
            "RGB",
            (columns_used * thumb_size, rows * (thumb_size + 22) + 18),
            (18, 18, 18),
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 4), f"{title} · {len(chosen)} images", fill=(240, 240, 240), font=font)
        rendered_count = 0
        for position, (_, record, summary) in enumerate(chosen):
            image_path = Path(str(record.get("image") or record.get("prepared_image") or ""))
            try:
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
            except OSError:
                continue
            rendered, _ = annotate_image(
                image,
                list(record.get("ground_truth") or []),
                list(record.get("detections") or []),
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold,
                classes=classes,
            )
            row, column = divmod(position, columns_used)
            tile = _thumb(rendered, thumb_size)
            x = column * thumb_size
            y = 18 + row * (thumb_size + 22)
            sheet.paste(tile, (x, y))
            caption = (
                f"{Path(str(record.get('image') or '')).stem} · "
                f"TP {summary['tp']} FP {summary['fp']} FN {summary['fn']}"
            )
            draw.text((x + 4, y + thumb_size + 4), caption, fill=(200, 200, 200), font=font)
            rendered_count += 1
        target = output / filename
        _save_bounded_jpeg(sheet, target)
        written.append(
            {
                "filename": filename,
                "path": str(target),
                "title": title,
                "bucket": "all" if class_name is None and filename == "mural-overview.jpg"
                else ("errors" if class_name is None else "class"),
                "class": class_name,
                "images": rendered_count,
                "bytes": target.stat().st_size,
            }
        )
    return written
