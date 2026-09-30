"""Use Gemma to visually audit high-signal recurring error-mode candidates."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import modal


DRIVE_FILE_ID = "18U0D61HWmdFH9kyPSuog6mD6zKgfugTc"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
MODEL = "google/gemma-3-27b-it"
RUN_VERSION = "bioimages-error-mode-audit-v1"
HUMAN_TAGS = ["leaf", "twig", "bark", "flower", "fruit", "cone", "seed", "whole plant", "other"]

runtime = modal.Image.debian_slim(python_version="3.11").pip_install(
    "gdown==5.2.0", "pillow==11.3.0", "requests==2.32.5"
)
app = modal.App("bioimages-error-mode-vlm-audit", image=runtime)
secret = modal.Secret.from_name("openrouter-bioimages-vlm")
volume = modal.Volume.from_name("bioimages-error-mode-vlm-audit", create_if_missing=True)


PROMPT = """You are auditing recurring error patterns in botanical image tagging.
Judge the photograph itself first, then use the supplied annotations and predictions to
explain the disagreement. BioImages is a single intended-subject label, not exhaustive
visual truth. Gemma and CNN outputs may also be wrong.

Return JSON with exactly these fields:
- visible_tags: every clearly visible item from [leaf, twig, bark, flower, fruit, cone,
  seed, whole plant, other]. Multiple tags are expected when appropriate.
- other_visible: a short phrase for visible structure not covered above, or "".
- supported_modes: only candidate mode IDs that the pixels genuinely support.
- annotation_ambiguity: true when multiple reasonable labels or a non-exhaustive
  BioImages label explain the apparent mismatch.
- bioimages_label_visually_supported: true/false.
- short_pattern: a concrete error-pattern name, at most 12 words.
- likely_cause: one short sentence grounded in this image.

Do not infer a species. Do not accept a candidate mode merely because it was suggested.
Return JSON only."""


def package(files: dict[str, bytes]) -> bytes:
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return sink.getvalue()


@app.function(
    timeout=3 * 60 * 60,
    cpu=6.0,
    memory=8192,
    secrets=[secret],
    volumes={"/results": volume},
)
def run_audit(candidates: list[dict], concurrency: int = 10) -> bytes:
    import base64
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    import random
    import time
    import zipfile

    import gdown
    from PIL import Image, ImageFile, ImageOps
    import requests

    if not 1 <= concurrency <= 12:
        raise ValueError("concurrency must be 1..12")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    zip_path = "/tmp/bioimages.zip"
    root = Path("/tmp/bioimages")
    if not gdown.download(DRIVE_URL, zip_path, quiet=False, fuzzy=True):
        raise RuntimeError("Google Drive download failed")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    image_dir = root / "bioimages-ne-trees" / "images"

    checkpoint = Path("/results") / f"{RUN_VERSION}.jsonl"
    completed = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            try:
                item = json.loads(line)
                if item.get("run_version") == RUN_VERSION and item.get("status") == "ok":
                    completed[item["image_id"]] = item
            except Exception:
                pass

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://arborphy.github.io/fieldguidevision/",
        "X-Title": "BioImages error mode discovery",
    }

    def encode(path: Path) -> str:
        output = io.BytesIO()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image.save(output, format="JPEG", quality=85, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()

    def parse(content, candidate_modes: set[str]) -> dict:
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        clean = str(content).strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if clean.startswith("json"):
                clean = clean[4:].lstrip()
        if not clean.startswith("{"):
            clean = clean[clean.find("{"):clean.rfind("}") + 1]
        value = json.loads(clean)
        visible = value.get("visible_tags")
        modes = value.get("supported_modes")
        if not isinstance(visible, list) or not visible:
            raise ValueError("visible_tags missing")
        visible = list(dict.fromkeys(item for item in visible if item in HUMAN_TAGS))
        if not visible:
            visible = ["other"]
        if not isinstance(modes, list):
            raise ValueError("supported_modes missing")
        modes = list(dict.fromkeys(item for item in modes if item in candidate_modes))
        return {
            "visible_tags": visible,
            "other_visible": str(value.get("other_visible", ""))[:160],
            "supported_modes": modes,
            "annotation_ambiguity": bool(value.get("annotation_ambiguity", False)),
            "bioimages_label_visually_supported": bool(value.get("bioimages_label_visually_supported", False)),
            "short_pattern": str(value.get("short_pattern", ""))[:160],
            "likely_cause": str(value.get("likely_cause", ""))[:400],
        }

    def call_one(row: dict) -> dict:
        started = time.perf_counter()
        total_cost = 0.0
        image_url = encode(image_dir / row["file"])
        context = {
            "bioimages_label": row["bioimages_label"],
            "gemma_previous_tags": row["gemma_tags"],
            "gemma_previous_normalized_tags": row["gemma_normalized_tags"],
            "model_tags_with_confidence": row["model_tags"],
            "candidate_mode_ids": row["candidate_modes"],
        }
        last_error = ""
        for attempt in range(1, 5):
            payload = {
                "model": MODEL,
                "temperature": 0,
                "max_tokens": 650,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT + "\n\nCase data:\n" + json.dumps(context, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}],
                "usage": {"include": True},
            }
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    time.sleep(2 ** attempt + random.random())
                    continue
                response.raise_for_status()
                raw = response.json()
                usage = raw.get("usage") or {}
                total_cost += float(usage.get("cost") or 0)
                parsed = parse(raw["choices"][0]["message"]["content"], set(row["candidate_modes"]))
                return {
                    "image_id": row["image_id"],
                    "run_version": RUN_VERSION,
                    "model": raw.get("model", MODEL),
                    **parsed,
                    "candidate_modes": row["candidate_modes"],
                    "cost_usd": total_cost,
                    "attempts": attempt,
                    "latency_seconds": time.perf_counter() - started,
                    "status": "ok",
                    "error": "",
                }
            except Exception as exc:
                last_error = repr(exc)
                if attempt < 4:
                    time.sleep(2 ** attempt)
        return {
            "image_id": row["image_id"], "run_version": RUN_VERSION,
            "candidate_modes": row["candidate_modes"], "cost_usd": total_cost,
            "attempts": 4, "latency_seconds": time.perf_counter() - started,
            "status": "api_error", "error": last_error,
        }

    pending = [row for row in candidates if row["image_id"] not in completed]
    with checkpoint.open("a") as sink, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(call_one, row): row["image_id"] for row in pending}
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            if result.get("status") == "ok":
                completed[result["image_id"]] = result
            if number % 40 == 0:
                volume.commit()
    volume.commit()

    selected = [completed[row["image_id"]] for row in candidates if row["image_id"] in completed]
    if len(selected) != len(candidates):
        missing = [row["image_id"] for row in candidates if row["image_id"] not in completed]
        raise RuntimeError(f"VLM audit incomplete: {len(selected)}/{len(candidates)}; missing={missing[:5]}")
    jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
    summary = {
        "run_version": RUN_VERSION,
        "model": MODEL,
        "images": len(selected),
        "cost_usd": sum(float(row.get("cost_usd") or 0) for row in selected),
        "annotation_ambiguity": sum(row["annotation_ambiguity"] for row in selected),
        "bioimages_label_supported": sum(row["bioimages_label_visually_supported"] for row in selected),
        "status": "complete",
    }
    return package({
        "predictions.jsonl": jsonl,
        "summary.json": json.dumps(summary, indent=2).encode(),
        "prompt.txt": PROMPT.encode(),
    })


@app.local_entrypoint()
def main(
    candidates: str = "botanical-computer-vision/bioimages_human_loop/analysis/vlm_audit_candidates.json",
    output: str = "botanical-computer-vision/outputs/error_mode_vlm_audit",
    concurrency: int = 10,
) -> None:
    rows = json.loads(Path(candidates).read_text())
    payload = run_audit.remote(rows, concurrency)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(destination)
    print((destination / "summary.json").read_text())
