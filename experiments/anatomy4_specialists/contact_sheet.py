# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""GT crop contact-sheet renderer.

Reads the final crop manifest (after the weak-label join) and renders one
sheet per split x target label, plus an all-labels overview. Captions include
the crop id, taxon, anatomy class, detector confidence, and weak-label source
so a human can audit the evidence that will supervise training.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import TargetSpec, normalize_crop_row, read_jsonl

LABEL_COLORS = {
    "Simple": (46, 204, 113),
    "Compound": (241, 196, 15),
    "Entire": (52, 152, 219),
    "Serrate": (231, 76, 60),
    "Lobed": (155, 89, 182),
    "Alternate": (26, 188, 156),
    "Opposite": (230, 126, 34),
    "Raceme": (46, 204, 113),
    "Corymb": (241, 196, 15),
    "Cyme": (52, 152, 219),
    "Panicle": (231, 76, 60),
}


def _load_font(size: int):
    """Portable font: DejaVu when present, otherwise Pillow's bundled default."""

    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def load_sheet_rows(
    *, crop_manifest: str | Path | None = None, dataset_root: str | Path | None = None
) -> list[dict[str, Any]]:
    """Load rows from a crop manifest or the **trained dataset's** manifest.

    A dataset build may recompute splits, so sheets must read
    ``<dataset>/manifest.jsonl`` when one exists; the pre-split crop manifest
    can disagree with the actual training split.
    """

    if dataset_root is not None:
        path = Path(dataset_root) / "manifest.jsonl"
    elif crop_manifest is not None:
        path = Path(crop_manifest)
    else:
        raise ValueError("provide crop_manifest or dataset_root")
    if not path.is_file():
        raise FileNotFoundError(f"contact-sheet manifest not found: {path}")
    return [normalize_crop_row(row) for row in read_jsonl(path)]


def _tile(
    path: Path,
    caption: str,
    *,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
        image.thumbnail((width - 4, height - 34), Image.Resampling.LANCZOS)
        canvas.paste(image, ((width - image.width) // 2, 2))
    except Exception:
        draw = ImageDraw.Draw(canvas)
        draw.text((8, height // 2), "MISSING", fill=(255, 80, 80))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, height - 32, width - 1, height - 1], fill=(0, 0, 0))
    draw.rectangle([0, height - 34, width - 1, height - 32], fill=color)
    font = _load_font(11)
    draw.text((4, height - 28), caption[: max(8, width // 6)], fill=(235, 235, 235), font=font)
    return canvas


def _page(
    items: Sequence[Mapping[str, Any]],
    *,
    columns: int,
    rows: int,
    tile_w: int,
    tile_h: int,
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (8, 8, 8))
    for index, row in enumerate(items[: columns * rows]):
        col, line = index % columns, index // columns
        label = str(row.get("target_label") or "unlabeled")
        color = LABEL_COLORS.get(label, (200, 200, 200))
        caption = (
            f"{label} | {row.get('taxon_name')} | {row.get('anatomy_class')} "
            f"| conf {float(row.get('detector_confidence', 0)):.2f} | {row.get('crop_id')}"
        )
        tile = _tile(
            Path(str(row.get("crop_path"))),
            caption,
            width=tile_w,
            height=tile_h,
            color=color,
        )
        sheet.paste(tile, (col * tile_w, line * tile_h))
    return sheet


def render_contact_sheets(
    rows: Sequence[Mapping[str, Any]],
    target: TargetSpec,
    out_dir: str | Path,
    *,
    columns: int = 6,
    rows_per_page: int = 4,
    tile_width: int = 220,
    tile_height: int = 200,
    splits: Sequence[str] = ("train", "val", "test"),
    include_unlabeled: bool = False,
) -> dict[str, Any]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    usable = [
        dict(row)
        for row in rows
        if row.get("target_label") is not None or include_unlabeled
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        split = str(row.get("observation_split") or "unsorted")
        label = str(row.get("target_label") or "unlabeled")
        groups[(split, label)].append(row)

    written: list[str] = []
    counts: Counter[str] = Counter()
    per_page = max(1, columns * rows_per_page)
    for split in splits:
        labels = [name for name in target.classes] + (["unlabeled"] if include_unlabeled else [])
        for label in labels:
            items = groups.get((split, label), [])
            if not items:
                continue
            items = sorted(items, key=lambda r: (-float(r.get("detector_confidence", 0)), str(r.get("crop_id"))))
            for page_index in range(math.ceil(len(items) / per_page)):
                chunk = items[page_index * per_page : (page_index + 1) * per_page]
                sheet = _page(
                    chunk,
                    columns=columns,
                    rows=rows_per_page,
                    tile_w=tile_width,
                    tile_h=tile_height,
                )
                path = output / f"{split}-{label.replace(' ', '_')}-{page_index:02d}.jpg"
                sheet.save(path, quality=90, optimize=True)
                written.append(str(path))
                counts[f"{split}:{label}"] += len(chunk)

    return {
        "schema": "arq.anatomy4_specialists.contact_sheets/v1",
        "target": target.target_id,
        "sheets": written,
        "counts": dict(sorted(counts.items())),
        "columns": columns,
        "rows_per_page": rows_per_page,
    }
