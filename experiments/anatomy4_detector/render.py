# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""GT-vs-prediction overlays and per-class contact sheets.

Colors: each anatomy4 class has a stable color for ground truth. Prediction
boxes are colored by match status at the requested IoU/confidence (TP green,
FP red, FN orange, dashed). This is the visual audit surface for the metric
numbers, not a substitute for them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .contracts import CLASS_NAMES, CONTRACT
from .metrics import PixelBox, PixelDetection, match_detections

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "leaf": (46, 158, 79),
    "needle": (31, 111, 235),
    "flower": (217, 72, 139),
    "inflorescence": (224, 161, 0),
}
STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "tp": (34, 197, 94),
    "fp": (239, 68, 68),
    "fn": (245, 158, 11),
}


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
) -> tuple[Image.Image, dict]:
    """Draw GT and predictions; return the image and an audit summary."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(max(11, min(16, canvas.width // 45)))
    summary = {"tp": 0, "fp": 0, "fn": 0}

    for box in ground_truth:
        color = CLASS_COLORS[box.class_name]
        draw.rectangle(box.xyxy, outline=color, width=line_width)
        if show_labels:
            _text_with_backdrop(
                draw,
                (box.x0 + 2, max(0, box.y0 - 16)),
                f"GT {box.class_name}",
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
                f"{detection.class_name} {detection.confidence:.2f} {status.upper()}",
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
                f"MISSED {box.class_name}",
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
        )
        present = {
            box.class_name for box in record["ground_truth"]
        } | {
            det.class_name
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
