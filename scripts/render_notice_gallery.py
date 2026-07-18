#!/usr/bin/env python3
"""Render an auditable contact sheet from a notice-review gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "leaf": "#39d353", "flower": "#ff6b9a", "fruit": "#ffb000",
    "bud": "#b692ff", "bark": "#c58a54", "branch": "#45c8ff",
    "stem": "#45c8ff", "whole plant": "#e7e7e7", "ambiguous organ": "#ff7b00",
}


def _scale_polygon(points: list[list[float]], width: int, height: int) -> list[tuple[float, float]]:
    return [(point[0] * width, point[1] * height) for point in points]


def render(gallery: dict, batch: dict, output: Path, cell_width: int = 420) -> None:
    paths = {frame["observation_id"]: frame["local_path"] for frame in batch["input_manifest"]["frames"]}
    entries = gallery["annotations"]
    rows = (len(entries) + 2) // 3
    cell_height = int(cell_width * 0.94)
    sheet = Image.new("RGB", (cell_width * 3, cell_height * rows), "#101417")
    font = ImageFont.load_default()
    for index, annotation in enumerate(entries):
        image = Image.open(paths[annotation["observation_id"]]).convert("RGB")
        image.thumbnail((cell_width, cell_height - 46))
        canvas = Image.new("RGB", (cell_width, cell_height), "#101417")
        x = (cell_width - image.width) // 2
        canvas.paste(image, (x, 0))
        draw = ImageDraw.Draw(canvas)
        category = annotation["notice_category_label"]
        color = COLORS.get(category, "#ffffff")
        polygon = _scale_polygon(annotation["polygon"], image.width, image.height)
        polygon = [(point_x + x, point_y) for point_x, point_y in polygon]
        draw.line(polygon, fill=color, width=4, joint="curve")
        label = f"{category}  score {annotation['confidence']:.2f}  {annotation['annotation_id'][-6:]}"
        draw.rectangle((0, cell_height - 46, cell_width, cell_height), fill="#101417")
        draw.text((8, cell_height - 37), label, fill=color, font=font)
        draw.text((8, cell_height - 20), annotation["observation_id"], fill="#dce3e8", font=font)
        sheet.paste(canvas, ((index % 3) * cell_width, (index // 3) * cell_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gallery", type=Path)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Gallery image already exists: {args.output}")
    render(json.loads(args.gallery.read_text()), json.loads(args.batch.read_text()), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
