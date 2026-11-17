# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""CLIP conifer-vs-not filter over needle candidate crops.

The second YOLOE prompt pass on a crop still fires on non-conifer crops (the
detector always produces a box). This stage asks a contrastive question
instead: is this crop conifer foliage, or one of the known confusers (forest
floor, mushroom, broadleaf, trunk, animal)?

Inputs are the crops written by ``needle-second-pass``, so the crop geometry is
identical and the two filters are directly comparable on the same review set.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

POSITIVE_PROMPTS = [
    "a close-up photo of conifer needles",
    "pine needles on a branch",
    "spruce or fir foliage",
    "a hemlock branch with needles",
]
NEGATIVE_PROMPTS = [
    "a mushroom",
    "forest floor with dead leaves",
    "a broadleaf plant",
    "a tree trunk",
    "an animal",
    "moss on a log",
    "a flower",
]


def run_needle_clip_filter(
    second_pass_dir: str | Path,
    out_dir: str | Path,
    *,
    model_name: str = "ViT-B/32",
    device: str = "cuda",
    batch_size: int = 32,
    download_root: str | None = None,
) -> dict[str, Any]:
    import clip
    import torch
    from PIL import Image

    source = Path(second_pass_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    crops_dir = source / "crops"
    rows = list(csv.DictReader((source / "needle-second-pass.csv").open(newline="")))
    rows = [row for row in rows if (crops_dir / f"{row['proposal_id']}.jpg").is_file()]

    model, preprocess = clip.load(model_name, device=device, download_root=download_root)
    texts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_features = model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    started = time.perf_counter()
    scores: dict[str, float] = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        images = [preprocess(Image.open(crops_dir / f"{row['proposal_id']}.jpg").convert("RGB")) for row in chunk]
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(batch).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.t()
            probabilities = logits.softmax(dim=-1)
        positive_mass = probabilities[:, : len(POSITIVE_PROMPTS)].sum(dim=-1)
        for row, value in zip(chunk, positive_mass.tolist()):
            scores[row["proposal_id"]] = float(value)

    for row in rows:
        row["clip_conifer_score"] = round(scores.get(row["proposal_id"], 0.0), 4)
    csv_path = output / "needle-clip-filter.csv"
    fieldnames = list(rows[0].keys()) if rows else ["proposal_id", "clip_conifer_score"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "crops_scored": len(rows),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "provenance": {
            "model": model_name,
            "positive_prompts": POSITIVE_PROMPTS,
            "negative_prompts": NEGATIVE_PROMPTS,
            "device": device,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
