# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from experiments.anatomy4_detector.contracts import CLASS_NAMES


def write_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (40, 60, 40)).save(path)


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(lines) + "\n") if lines else "")


def make_dataset(
    root: Path,
    *,
    entries: list[dict] | None = None,
    write_metadata: bool = True,
) -> dict:
    """Build a small valid dataset.

    Default: one train box (leaf), one train negative, one val box (needle),
    one test box (flower).
    """

    if entries is None:
        entries = [
            {
                "image": "images/train/a.jpg",
                "observation_id": "obs-a",
                "negative": False,
                "boxes": ["0 0.10 0.10 0.50 0.50"],
            },
            {
                "image": "images/train/neg.jpg",
                "observation_id": "obs-neg",
                "negative": True,
                "boxes": [],
            },
            {
                "image": "images/val/b.jpg",
                "observation_id": "obs-b",
                "negative": False,
                "boxes": ["1 0.20 0.20 0.60 0.60"],
            },
            {
                "image": "images/test/c.jpg",
                "observation_id": "obs-c",
                "negative": False,
                "boxes": ["2 0.30 0.30 0.70 0.70"],
            },
        ]
    for entry in entries:
        image_path = root / entry["image"]
        write_image(image_path)
        stem = image_path.stem
        split = image_path.parent.name
        write_label(root / "labels" / split / f"{stem}.txt", entry.get("boxes", []))
    if write_metadata:
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "schema": "arq.anatomy4.dataset/v1",
                    "contract": "anatomy4-v1",
                    "classes": list(CLASS_NAMES),
                    "images": [
                        {
                            "image": entry["image"],
                            "observation_id": entry.get("observation_id"),
                            "negative": entry.get("negative", False),
                        }
                        for entry in entries
                    ],
                },
                indent=1,
            )
        )
    return {"root": root, "entries": entries}
