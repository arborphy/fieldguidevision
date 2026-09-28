"""Create a full-corpus free multi-label visual reference with Gemma 3 27B."""

from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import modal

from tagging_vocabulary import TAGS, VOCABULARY


DRIVE_FILE_ID = "18U0D61HWmdFH9kyPSuog6mD6zKgfugTc"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
MODEL = "google/gemma-3-27b-it"
RUN_VERSION = "bioimages-gemma-multilabel-v2"
MAX_IMAGE_SIDE = 1024

runtime = modal.Image.debian_slim(python_version="3.11").pip_install(
    "gdown==5.2.0", "pillow==11.3.0", "requests==2.32.5"
).add_local_python_source("tagging_vocabulary")
app = modal.App("bioimages-gemma-multilabel-reference", image=runtime)
openrouter_secret = modal.Secret.from_name("openrouter-bioimages-vlm")
checkpoint_volume = modal.Volume.from_name("bioimages-gemma-multilabel", create_if_missing=True)


VOCABULARY_TEXT = "\n".join(f'- {item["tag"]}: {item["description"]}' for item in VOCABULARY)
PROMPT = f"""Inspect only the visible content of this botanical photograph. Do not identify
the species and do not use filenames or metadata. Return every clearly visible plant
structure and meaningful photographic view/condition. Multiple structures may coexist;
do not force a single label and do not invent hidden structures.

Return JSON with exactly two arrays:
1. "tags": 1 to 12 short, natural English visual tags of your own choosing.
2. "normalized_tags": every applicable item from the controlled vocabulary below.
There is no fixed number of normalized tags. Use exact vocabulary spelling and omit an
item when it is not visibly supported. General structure tags and specific view tags may
both be included.

Controlled vocabulary:
{VOCABULARY_TEXT}

Return JSON only, for example:
{{"tags":["broad green leaves","woody twig","immature fruit"],
 "normalized_tags":["leaf","twig","fruit","immature fruit"]}}"""


def _package(files: dict[str, bytes]) -> bytes:
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name); info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return sink.getvalue()


@app.function(
    timeout=10 * 60 * 60,
    cpu=6.0,
    memory=8192,
    secrets=[openrouter_secret],
    volumes={"/results": checkpoint_volume},
)
def run_reference(stage: str = "smoke", concurrency: int = 10) -> bytes:
    import base64
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import hashlib
    import os
    import random
    import time
    import zipfile

    import gdown
    from PIL import Image, ImageFile, ImageOps
    import requests

    if stage not in {"smoke", "full"}: raise ValueError(stage)
    if not 1 <= concurrency <= 12: raise ValueError("concurrency must be 1..12")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key: raise RuntimeError("OPENROUTER_API_KEY is missing")
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    zip_path = "/tmp/bioimages.zip"; root = Path("/tmp/bioimages")
    if not gdown.download(DRIVE_URL, zip_path, quiet=False, fuzzy=True):
        raise RuntimeError("Google Drive download failed")
    with zipfile.ZipFile(zip_path) as archive: archive.extractall(root)
    corpus = json.loads((root / "bioimages-ne-trees" / "corpus.json").read_text())
    image_dir = root / "bioimages-ne-trees" / "images"
    rows = []
    for scene in corpus["scenes"]:
        for image in scene["images"]:
            rows.append({
                "image_id": image["image_id"], "file": image["file"],
                "path": str(image_dir / image["file"]), "species": scene["scientific_name"],
            })
    rows.sort(key=lambda row: row["image_id"])
    if len(rows) != 1899: raise RuntimeError(f"Expected 1899 images, got {len(rows)}")
    if stage == "smoke":
        rows = sorted(rows, key=lambda row: hashlib.sha256(("smoke|" + row["image_id"]).encode()).hexdigest())[:12]

    checkpoint = Path("/results") / f"{RUN_VERSION}.jsonl"
    completed: dict[str, dict] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            try:
                item = json.loads(line)
                if item.get("run_version") == RUN_VERSION and item.get("status") == "ok":
                    completed[item["image_id"]] = item
            except Exception:
                pass

    headers = {
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://arborphy.github.io/fieldguidevision/",
        "X-Title": "BioImages multi-label visual tagging reference",
    }

    def encode(path: str) -> tuple[str, int, int]:
        output = io.BytesIO()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            width, height = image.size
            image.save(output, format="JPEG", quality=85, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode(), width, height

    def parse_content(content) -> tuple[list[str], list[str]]:
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        clean = str(content).strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if clean.startswith("json"): clean = clean[4:].lstrip()
        if not clean.startswith("{"):
            clean = clean[clean.find("{"):clean.rfind("}") + 1]
        value = json.loads(clean)
        free = value.get("tags"); normalized = value.get("normalized_tags")
        if not isinstance(free, list) or not 1 <= len(free) <= 12 or any(not isinstance(x, str) or not x.strip() for x in free):
            raise ValueError("tags must be 1..12 non-empty strings")
        if not isinstance(normalized, list):
            raise ValueError("normalized_tags must be an array")
        # Preserve every unconstrained free tag for inspection, but make the
        # scoring array robust to a repeated item or an occasional extra term
        # (for example, "thorn") that is not part of the controlled vocabulary.
        normalized = list(dict.fromkeys(x for x in normalized if x in TAGS))
        if not normalized: raise ValueError("normalized_tags may not be empty")
        return [x.strip() for x in free], normalized

    def call_one(row: dict) -> dict:
        started = time.perf_counter(); last_error = ""; total_cost = 0.0
        try:
            image_url, width, height = encode(row["path"])
        except Exception as exc:
            return {**row, "run_version": RUN_VERSION, "status": "decode_error", "error": repr(exc)}
        correction = ""
        for attempt in range(1, 5):
            payload = {
                "model": MODEL, "temperature": 0, "max_tokens": 700,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT + correction},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}], "usage": {"include": True},
            }
            try:
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=180)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text[:400]}"; time.sleep(2 ** attempt + random.random()); continue
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text[:800]}"; break
                raw = response.json(); usage = raw.get("usage") or {}
                if usage.get("cost") is not None: total_cost += float(usage["cost"])
                content = raw["choices"][0]["message"]["content"]
                free, normalized = parse_content(content)
                return {
                    "image_id": row["image_id"], "file": row["file"], "species": row["species"],
                    "run_version": RUN_VERSION, "model": raw.get("model", MODEL),
                    "tags": free, "normalized_tags": normalized, "raw_response": content,
                    "generation_id": raw.get("id", ""), "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"), "cost_usd": total_cost,
                    "image_width": width, "image_height": height, "attempts": attempt,
                    "latency_seconds": time.perf_counter() - started, "status": "ok", "error": "",
                }
            except Exception as exc:
                last_error = repr(exc)
                correction = "\n\nYour previous response was invalid. Return complete JSON with both arrays and exact normalized vocabulary spelling."
                if attempt < 4: time.sleep(2 ** attempt)
        return {
            "image_id": row["image_id"], "file": row["file"], "species": row["species"],
            "run_version": RUN_VERSION, "model": MODEL, "tags": [], "normalized_tags": [],
            "cost_usd": total_cost, "attempts": 4, "latency_seconds": time.perf_counter() - started,
            "status": "api_error", "error": last_error,
        }

    pending = [row for row in rows if row["image_id"] not in completed]
    with checkpoint.open("a") as sink, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(call_one, row): row["image_id"] for row in pending}
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result(); sink.write(json.dumps(result, ensure_ascii=False) + "\n"); sink.flush()
            if result.get("status") == "ok": completed[result["image_id"]] = result
            if n % 50 == 0: checkpoint_volume.commit()
    checkpoint_volume.commit()

    selected_ids = {row["image_id"] for row in rows}
    selected = [completed[image_id] for image_id in sorted(selected_ids) if image_id in completed]
    failures = [row for row in rows if row["image_id"] not in completed]
    if failures:
        raise RuntimeError(f"Gemma incomplete: {len(selected)}/{len(rows)}; first missing={failures[:3]}")

    jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
    csv_sink = io.StringIO(); fields = [
        "image_id", "file", "species", "model", "tags", "normalized_tags", "cost_usd",
        "prompt_tokens", "completion_tokens", "attempts", "latency_seconds", "status", "error",
    ]
    writer = csv.DictWriter(csv_sink, fieldnames=fields); writer.writeheader()
    for row in selected:
        writer.writerow({**{k: row.get(k, "") for k in fields}, "tags": json.dumps(row["tags"]), "normalized_tags": json.dumps(row["normalized_tags"])})
    summary = {
        "run_version": RUN_VERSION, "stage": stage, "model": MODEL, "images": len(selected),
        "vocabulary_size": len(TAGS), "mean_tags": sum(len(r["normalized_tags"]) for r in selected) / len(selected),
        "cost_usd": sum(float(r.get("cost_usd") or 0) for r in selected),
        "latency_seconds_sum": sum(float(r.get("latency_seconds") or 0) for r in selected),
        "attempt_histogram": {str(a): sum(r.get("attempts") == a for r in selected) for a in range(1, 5)},
        "status": "complete",
    }
    return _package({
        "predictions.jsonl": jsonl, "predictions.csv": csv_sink.getvalue().encode(),
        "summary.json": json.dumps(summary, indent=2).encode(),
        "prompt.txt": PROMPT.encode(), "vocabulary.json": json.dumps(VOCABULARY, indent=2).encode(),
    })


@app.local_entrypoint()
def main(stage: str = "smoke", concurrency: int = 10, output: str = "botanical-computer-vision/outputs/gemma_multilabel_reference") -> None:
    payload = run_reference.remote(stage, concurrency)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(out)
    print((out / "summary.json").read_text())
