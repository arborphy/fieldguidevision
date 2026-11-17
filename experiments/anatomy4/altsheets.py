# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Large-tile review sheets for box-only proposal generators.

The broad audits used 420x320 contact tiles, which is enough to judge "right
organ" but not box tightness. These sheets keep the deterministic
score-stratified sample rule but render bigger tiles and caption the prompt, so
the reviewer sees exactly which phrase produced each box.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

CLASS_COLORS = {
    "leaf": (46, 204, 113),
    "needle": (52, 152, 219),
    "flower": (241, 196, 15),
    "inflorescence": (231, 76, 60),
}


def sample_evenly(items: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Same deterministic rule as ``render.render_review_sample``."""

    if len(items) <= count:
        return list(items)
    step = (len(items) - 1) / (count - 1)
    indexes = sorted({round(i * step) for i in range(count)})
    return [items[index] for index in indexes]


def _tile(
    image_path: str,
    proposal: dict[str, Any],
    width: int,
    height: int,
) -> Any:
    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((width, height))
    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    offset_x = (width - image.width) // 2
    offset_y = (height - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas, "RGBA")
    color = CLASS_COLORS.get(proposal["class_name"], (255, 255, 255))
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
            width=4,
        )
    return canvas


def render_class_sheets(
    manifest: dict[str, Any],
    out_dir: str | Path,
    *,
    class_name: str,
    per_class: int = 24,
    columns: int = 3,
    tile_w: int = 520,
    tile_h: int = 390,
    caption_h: int = 34,
    background_only: bool = False,
) -> tuple[list[str], str]:
    """Render one class's sampled proposals; returns (sheet paths, CSV path)."""

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
        and proposal.get("class_name") == class_name
        and (proposal.get("audit_bucket") == "background") == background_only
    ]
    proposals.sort(key=lambda p: p["box_score"], reverse=True)
    sample = sample_evenly(proposals, per_class)

    rows = 3
    per_sheet = columns * rows
    sheet_paths: list[str] = []
    for page, start in enumerate(range(0, len(sample), per_sheet)):
        chunk = sample[start : start + per_sheet]
        sheet = Image.new(
            "RGB", (columns * tile_w, rows * (tile_h + caption_h)), (10, 10, 10)
        )
        draw = ImageDraw.Draw(sheet)
        for index, proposal in enumerate(chunk):
            row, col = divmod(index, columns)
            path = image_paths.get(proposal["photo_id"])
            x = col * tile_w
            y = row * (tile_h + caption_h)
            if path and Path(path).is_file():
                sheet.paste(_tile(path, proposal, tile_w, tile_h), (x, y))
            caption = (
                f"{proposal['box_score']:.3f} {proposal.get('prompt', '')} "
                f"area {proposal.get('detector_box_area', 0):.2f} | "
                f"{proposal['proposal_id']}"
            )
            draw.text((x + 4, y + tile_h + 8), caption, fill=(235, 235, 235))
        target = output / f"{class_name}{'-bg' if background_only else ''}-{page:02d}.jpg"
        sheet.save(target, quality=90)
        sheet_paths.append(str(target))

    csv_path = output / f"review-{class_name}{'-bg' if background_only else ''}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["tile", "proposal_id", "class_name", "box_score", "photo_id", "bucket", "prompt", "area", "verdict", "notes"]
        )
        for index, proposal in enumerate(sample):
            writer.writerow(
                [
                    index,
                    proposal["proposal_id"],
                    proposal["class_name"],
                    f"{proposal['box_score']:.4f}",
                    proposal["photo_id"],
                    proposal.get("audit_bucket", ""),
                    proposal.get("prompt", ""),
                    f"{proposal.get('detector_box_area', 0):.4f}",
                    "",
                    "",
                ]
            )
    return sheet_paths, str(csv_path)


def render_needle_detail(
    manifest: dict[str, Any],
    out_dir: str | Path,
    *,
    per_class: int = 24,
    tile_w: int = 820,
    tile_h: int = 615,
) -> tuple[list[str], str]:
    """2x2 close-up sheets: needle tightness is a judgment call at full size."""

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
        and proposal.get("class_name") == "needle"
        and proposal.get("audit_bucket") != "background"
    ]
    proposals.sort(key=lambda p: p["box_score"], reverse=True)
    sample = sample_evenly(proposals, per_class)

    caption_h = 30
    paths: list[str] = []
    for page, start in enumerate(range(0, len(sample), 4)):
        chunk = sample[start : start + 4]
        sheet = Image.new("RGB", (2 * tile_w, 2 * (tile_h + caption_h)), (10, 10, 10))
        draw = ImageDraw.Draw(sheet)
        for index, proposal in enumerate(chunk):
            row, col = divmod(index, 2)
            path = image_paths.get(proposal["photo_id"])
            x = col * tile_w
            y = row * (tile_h + caption_h)
            if path and Path(path).is_file():
                sheet.paste(_tile(path, proposal, tile_w, tile_h), (x, y))
            caption = (
                f"{proposal['box_score']:.3f} {proposal.get('prompt', '')} "
                f"area {proposal.get('detector_box_area', 0):.2f} | {proposal['proposal_id']}"
            )
            draw.text((x + 4, y + tile_h + 6), caption, fill=(235, 235, 235))
        target = output / f"needle-detail-{page:02d}.jpg"
        sheet.save(target, quality=90)
        paths.append(str(target))
    return paths, ""


def empty_image_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    """Per-bucket proposal/empty rates (abstention behavior)."""

    from collections import defaultdict

    per_bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"images": 0, "with_proposals": 0}
    )
    by_photo = defaultdict(int)
    for proposal in manifest.get("proposals", []):
        if proposal.get("status") == "proposed":
            by_photo[proposal["photo_id"]] += 1
    for image in manifest.get("images", []):
        if not image.get("selected_for_audit"):
            continue
        key = str(image.get("photo_id") or image.get("image_id"))
        bucket = str(image.get("audit_bucket") or "unknown")
        per_bucket[bucket]["images"] += 1
        if by_photo.get(key, 0) > 0:
            per_bucket[bucket]["with_proposals"] += 1
    for values in per_bucket.values():
        values["empty"] = values["images"] - values["with_proposals"]
        values["empty_rate"] = (
            round(values["empty"] / values["images"], 4) if values["images"] else None
        )
    return dict(per_bucket)
