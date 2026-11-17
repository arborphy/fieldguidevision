# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Overlay + contact-sheet renderer and review sheet generator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

CLASS_COLORS = {
    "leaf": (46, 204, 113),
    "needle": (52, 152, 219),
    "flower": (241, 196, 15),
    "inflorescence": (231, 76, 60),
}


def _draw_proposal(draw, proposal: dict[str, Any], width: int, height: int, *, label: bool = True) -> None:
    from PIL import ImageDraw

    color = CLASS_COLORS.get(proposal["class_name"], (255, 255, 255))
    polygon = proposal.get("polygon") or []
    if len(polygon) >= 3:
        points = [(x * width, y * height) for x, y in polygon]
        draw.polygon(points, outline=color, fill=(*color, 42))
    bbox = proposal.get("bbox") or proposal.get("detector_bbox")
    if bbox:
        x0, y0, x1, y1 = bbox
        draw.rectangle(
            [x0 * width, y0 * height, x1 * width, y1 * height],
            outline=color,
            width=3,
        )
    if label:
        text = f"{proposal['class_name']} {proposal['box_score']:.2f}"
        tx, ty = (bbox[0] * width + 4, bbox[1] * height + 4) if bbox else (6, 6)
        box = draw.textbbox((tx, ty), text)
        draw.rectangle([box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1], fill=(0, 0, 0))
        draw.text((tx, ty), text, fill=color)


def render_image_overlay(
    image_path: str | Path,
    proposals: Sequence[dict[str, Any]],
    out_path: str | Path,
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for proposal in proposals:
        _draw_proposal(draw, proposal, image.width, image.height)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=88)


def _tile(image_path: str, proposals: Sequence[dict[str, Any]], width: int = 420, height: int = 320):
    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((width, height))
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    draw = ImageDraw.Draw(canvas, "RGBA")
    # Coordinates are normalized to the original, but the thumbnail preserves
    # aspect ratio, so map through the pasted image rectangle.
    offset_x = (width - image.width) // 2
    offset_y = (height - image.height) // 2
    for proposal in proposals:
        polygon = proposal.get("polygon") or []
        color = CLASS_COLORS.get(proposal["class_name"], (255, 255, 255))
        if len(polygon) >= 3:
            draw.polygon(
                [
                    (offset_x + x * image.width, offset_y + y * image.height)
                    for x, y in polygon
                ],
                outline=color,
                fill=(*color, 46),
            )
        bbox = proposal.get("bbox") or proposal.get("detector_bbox")
        if bbox:
            draw.rectangle(
                [
                    offset_x + bbox[0] * image.width,
                    offset_y + bbox[1] * image.height,
                    offset_x + bbox[2] * image.width,
                    offset_y + bbox[3] * image.height,
                ],
                outline=color,
                width=2,
            )
    return canvas


def render_contact_sheets(
    manifest: dict[str, Any],
    out_dir: str | Path,
    *,
    columns: int = 4,
    rows: int = 4,
    per_class: bool = True,
) -> list[str]:
    from PIL import Image, ImageDraw

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_paths = {
        str(image.get("photo_id") or image.get("image_id")): image.get("download", {}).get("path")
        for image in manifest.get("images", [])
    }
    proposals = [
        proposal
        for proposal in manifest.get("proposals", [])
        if proposal.get("status") == "proposed"
        and (proposal.get("polygon") or proposal.get("detector_bbox"))
    ]
    by_class: dict[str, list[dict[str, Any]]] = {}
    for proposal in proposals:
        by_class.setdefault(proposal["class_name"], []).append(proposal)
    for group in by_class.values():
        group.sort(key=lambda p: p["box_score"], reverse=True)

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    if per_class:
        groups = [(name, by_class.get(name, [])) for name in CLASS_COLORS]
    groups.append(("all", sorted(proposals, key=lambda p: p["box_score"], reverse=True)))

    tile_w, tile_h, caption_h = 420, 320, 28
    sheet_w = columns * tile_w
    sheet_h = rows * (tile_h + caption_h)
    written: list[str] = []
    for name, items in groups:
        if not items:
            continue
        page = 0
        for start in range(0, len(items), columns * rows):
            chunk = items[start : start + columns * rows]
            sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 10, 10))
            draw = ImageDraw.Draw(sheet)
            for index, proposal in enumerate(chunk):
                row, col = divmod(index, columns)
                path = image_paths.get(proposal["photo_id"])
                if not path or not Path(path).is_file():
                    continue
                tile = _tile(path, [proposal], tile_w, tile_h)
                x = col * tile_w
                y = row * (tile_h + caption_h)
                sheet.paste(tile, (x, y))
                caption = (
                    f"{proposal['class_name']} {proposal['box_score']:.2f} "
                    f"mask {(proposal.get('mask_iou') or 0):.2f} | {proposal['proposal_id']}"
                )
                draw.text((x + 4, y + tile_h + 6), caption, fill=(230, 230, 230))
            target = output / f"{name}-{page:02d}.jpg"
            sheet.save(target, quality=90)
            written.append(str(target))
            page += 1
    return written


def write_review_sheet(manifest: dict[str, Any], out_path: str | Path) -> str:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    proposals = [
        proposal
        for proposal in manifest.get("proposals", [])
        if proposal.get("status") == "proposed"
    ]
    proposals.sort(key=lambda p: (p["class_name"], -p["box_score"]))
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "proposal_id",
                "image_id",
                "observation_id",
                "class_name",
                "box_score",
                "mask_iou",
                "verdict",
                "notes",
            ]
        )
        for proposal in proposals:
            writer.writerow(
                [
                    proposal["proposal_id"],
                    proposal["photo_id"],
                    proposal.get("observation_id", ""),
                    proposal["class_name"],
                    f"{proposal['box_score']:.4f}",
                    f"{(proposal.get('mask_iou') or 0):.4f}",
                    "",
                    "",
                ]
            )
    return str(target)


def _sample_evenly(items: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministic score-stratified sample across the class range."""

    if len(items) <= count:
        return list(items)
    step = (len(items) - 1) / (count - 1)
    indexes = sorted({round(i * step) for i in range(count)})
    return [items[index] for index in indexes]


def render_review_sample(
    manifest: dict[str, Any],
    out_dir: str | Path,
    *,
    per_class: int = 24,
    columns: int = 4,
    rows: int = 4,
) -> tuple[list[str], str]:
    """Score-stratified review sheets + an index-aligned verdict CSV."""

    from PIL import Image, ImageDraw

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_paths = {
        str(image.get("photo_id") or image.get("image_id")): image.get("download", {}).get("path")
        for image in manifest.get("images", [])
    }
    proposals = [
        proposal
        for proposal in manifest.get("proposals", [])
        if proposal.get("status") == "proposed"
        and (proposal.get("polygon") or proposal.get("detector_bbox"))
    ]
    by_class: dict[str, list[dict[str, Any]]] = {}
    for proposal in proposals:
        by_class.setdefault(proposal["class_name"], []).append(proposal)
    for group in by_class.values():
        group.sort(key=lambda p: p["box_score"], reverse=True)

    tile_w, tile_h, caption_h = 420, 320, 40
    sheet_w = columns * tile_w
    sheet_h = rows * (tile_h + caption_h)
    written: list[str] = []
    rows_out: list[list[Any]] = []
    tile_index = 0
    for class_name in CLASS_COLORS:
        class_items = by_class.get(class_name, [])
        prompts = sorted({str(p.get("prompt_variant") or p.get("prompt") or "") for p in class_items})
        planned: list[dict[str, Any]] = []
        if len(prompts) > 1:
            per_prompt = max(6, per_class // len(prompts))
            for prompt in prompts:
                group = [p for p in class_items if str(p.get("prompt_variant") or p.get("prompt") or "") == prompt]
                planned.extend(_sample_evenly(group, per_prompt))
        else:
            planned = _sample_evenly(class_items, per_class)
        # Guarantee designed-negative examples in every class sheet so the
        # measured precision includes false positives on background images.
        seen_ids = {p["proposal_id"] for p in planned}
        negatives = [
            p
            for p in class_items
            if p.get("audit_bucket") == "background" and p["proposal_id"] not in seen_ids
        ]
        planned.extend(_sample_evenly(negatives, min(8, len(negatives))))
        planned.sort(key=lambda p: -p["box_score"])
        items = planned
        for start in range(0, len(items), columns * rows):
            chunk = items[start : start + columns * rows]
            sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 10, 10))
            draw = ImageDraw.Draw(sheet)
            for index, proposal in enumerate(chunk):
                row, col = divmod(index, columns)
                path = image_paths.get(proposal["photo_id"])
                if not path or not Path(path).is_file():
                    continue
                tile = _tile(path, [proposal], tile_w, tile_h)
                x = col * tile_w
                y = row * (tile_h + caption_h)
                sheet.paste(tile, (x, y))
                prompt_label = str(proposal.get("prompt_variant") or proposal.get("prompt") or "")
                caption = (
                    f"#{tile_index:02d} {class_name} {proposal['box_score']:.2f} "
                    f"[{prompt_label}] | {proposal['proposal_id']}"
                )
                draw.text((x + 4, y + tile_h + 6), caption, fill=(235, 235, 235))
                rows_out.append(
                    [
                        tile_index,
                        proposal["proposal_id"],
                        class_name,
                        f"{proposal['box_score']:.4f}",
                        proposal["photo_id"],
                        "",
                        "",
                        prompt_label,
                    ]
                )
                tile_index += 1
            target = output / f"review-{class_name}-{start // (columns * rows):02d}.jpg"
            sheet.save(target, quality=92)
            written.append(str(target))

    csv_path = output / "review-sample.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["tile", "proposal_id", "class_name", "box_score", "photo_id", "verdict", "notes", "prompt"]
        )
        writer.writerows(rows_out)
    return written, str(csv_path)


def render_background_sample(
    manifest: dict[str, Any],
    out_dir: str | Path,
    *,
    per_bucket: int = 32,
    columns: int = 4,
    rows: int = 4,
) -> tuple[list[str], str]:
    """Contact sheets of images with NO proposals (reviewed empty images).

    Deliberately caption-free about taxonomy: the reviewer must not be primed
    to find an organ. ``verdict`` is ``empty_correct`` when no target organ is
    present, ``missed_organ`` when the detector abstained on a visible one.
    """

    from PIL import Image, ImageDraw

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    images = [image for image in manifest.get("images", []) if image.get("selected_for_audit")]
    if not images:
        images = list(manifest.get("images", []))
    with_proposals = {
        str(proposal.get("photo_id") or proposal.get("image_id"))
        for proposal in manifest.get("proposals", [])
        if proposal.get("status") == "proposed"
    }
    # The designed-negative bucket is reviewed in full regardless of whether
    # the detector fired on it: these sheets measure missed organs, not boxes.
    empties = [
        image
        for image in images
        if image.get("audit_bucket") == "background"
        or str(image.get("photo_id") or image.get("image_id")) not in with_proposals
    ]
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for image in empties:
        by_bucket.setdefault(str(image.get("audit_bucket") or "unknown"), []).append(image)
    for group in by_bucket.values():
        group.sort(key=lambda image: str(image.get("photo_id") or image.get("image_id")))

    tile_w, tile_h, caption_h = 420, 320, 26
    sheet_w, sheet_h = columns * tile_w, rows * (tile_h + caption_h)
    written: list[str] = []
    rows_out: list[list[Any]] = []
    tile_index = 0
    for bucket in sorted(by_bucket):
        items = by_bucket[bucket][:per_bucket]
        for start in range(0, len(items), columns * rows):
            chunk = items[start : start + columns * rows]
            sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 10, 10))
            draw = ImageDraw.Draw(sheet)
            for index, image in enumerate(chunk):
                row, col = divmod(index, columns)
                photo_id = str(image.get("photo_id") or image.get("image_id"))
                path = image.get("download", {}).get("path")
                if not path or not Path(path).is_file():
                    continue
                tile = _tile(path, [], tile_w, tile_h)
                x, y = col * tile_w, row * (tile_h + caption_h)
                sheet.paste(tile, (x, y))
                draw.text(
                    (x + 4, y + tile_h + 5),
                    f"#{tile_index:02d} empty/{bucket} | {photo_id}",
                    fill=(235, 235, 235),
                )
                rows_out.append([tile_index, photo_id, bucket, "", ""])
                tile_index += 1
            target = output / f"review-empty-{bucket}-{start // (columns * rows):02d}.jpg"
            sheet.save(target, quality=92)
            written.append(str(target))

    csv_path = output / "review-background.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tile", "photo_id", "audit_bucket", "verdict", "notes"])
        writer.writerows(rows_out)
    return written, str(csv_path)
