"""Closed-set BioImages benchmark through OpenRouter, executed entirely on Modal.

Stages:
  smoke: three deterministic strict-test images
  pilot: one deterministic strict-test image from each of 63 eligible species
  full:  all 1,899 images and all 85 candidate species

The per-image ground truth is removed from the inference manifest before any API
request is made and is loaded again only after both models have finished.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import modal


DRIVE_FILE_ID = "18U0D61HWmdFH9kyPSuog6mD6zKgfugTc"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
SEED = 42
TARGET_HOLDOUT_FRACTION = 0.15
MAX_IMAGE_SIDE = 1536
MODELS = {
    "gemini_flash_lite": "google/gemini-2.5-flash-lite",
    "gpt_5_4": "openai/gpt-5.4",
}

runtime = modal.Image.debian_slim(python_version="3.11").pip_install(
    "gdown==5.2.0",
    "pillow==11.3.0",
    "requests==2.32.5",
)
app = modal.App("bioimages-openrouter-vlm-benchmark", image=runtime)
openrouter_secret = modal.Secret.from_name("openrouter-bioimages-vlm")


def _csv_bytes(rows: list[dict], fieldnames: list[str]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode(
        "utf-8"
    )


def _package(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


@app.function(
    timeout=6 * 60 * 60,
    cpu=4.0,
    memory=8192,
    secrets=[openrouter_secret],
)
def run_benchmark(
    stage: str = "pilot",
    concurrency: int = 6,
    model_keys_csv: str = "gemini_flash_lite,gpt_5_4",
    gpt_reasoning_effort: str = "minimal",
    gpt_image_detail: str = "auto",
    candidate_order_strategy: str = "fixed",
) -> bytes:
    import base64
    import hashlib
    import os
    import platform
    import random
    import shutil
    import time
    import zipfile
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import gdown
    import requests
    from PIL import Image, ImageFile, ImageOps

    if stage not in {"smoke", "pilot", "full"}:
        raise ValueError(f"Unsupported stage: {stage}")
    if not 1 <= concurrency <= 12:
        raise ValueError("concurrency must be between 1 and 12")
    requested_model_keys = [key.strip() for key in model_keys_csv.split(",") if key.strip()]
    if not requested_model_keys or any(key not in MODELS for key in requested_model_keys):
        raise ValueError(f"Unsupported model selection: {model_keys_csv}")
    selected_models = {key: MODELS[key] for key in requested_model_keys}
    if gpt_reasoning_effort not in {"minimal", "low", "medium", "high"}:
        raise ValueError(f"Unsupported GPT reasoning effort: {gpt_reasoning_effort}")
    if gpt_image_detail not in {"auto", "high", "original"}:
        raise ValueError(f"Unsupported GPT image detail: {gpt_image_detail}")
    if candidate_order_strategy not in {"fixed", "per_image"}:
        raise ValueError(f"Unsupported candidate order strategy: {candidate_order_strategy}")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")

    started = time.perf_counter()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    random.seed(SEED)

    zip_path = "/tmp/bioimages.zip"
    data_root = Path("/tmp/bioimages")
    truth_path = Path(f"/tmp/{stage}_truth.json")
    download_started = time.perf_counter()
    if not gdown.download(DRIVE_URL, zip_path, quiet=False, fuzzy=True):
        raise RuntimeError("Google Drive download failed")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_root)
    dataset_seconds = time.perf_counter() - download_started

    corpus_path = data_root / "bioimages-ne-trees" / "corpus.json"
    images_dir = data_root / "bioimages-ne-trees" / "images"
    with corpus_path.open("r", encoding="utf-8") as handle:
        corpus = json.load(handle)

    scenes = sorted(corpus["scenes"], key=lambda scene: scene["scientific_name"].strip())
    all_classes = [
        {
            "scientific_name": scene["scientific_name"].strip(),
            "vernacular": (scene.get("vernacular") or "").strip(),
        }
        for scene in scenes
    ]
    if len(all_classes) != 85:
        raise RuntimeError(f"Expected 85 species, found {len(all_classes)}")
    if sum(len(scene["images"]) for scene in scenes) != 1899:
        raise RuntimeError("Expected 1,899 images")

    all_rows_with_truth = []
    for scene in scenes:
        species = scene["scientific_name"].strip()
        for image in scene["images"]:
            all_rows_with_truth.append(
                {
                    "image_id": image["image_id"],
                    "file": image["file"],
                    "path": str(images_dir / image["file"]),
                    "individual_id": image["individual_id"],
                    "organ_category": image.get("organ_category") or "unknown",
                    "ground_truth_species": species,
                }
            )

    # Recompute the exact strict test individual for each of the 63 species that
    # have at least three independent individuals, then choose one image from it.
    strict_pilot_with_truth = []
    for scene in scenes:
        species = scene["scientific_name"].strip()
        groups: dict[str, list[dict]] = defaultdict(list)
        for image in scene["images"]:
            groups[image["individual_id"]].append(image)
        if len(groups) < 3:
            continue
        target = max(1.0, len(scene["images"]) * TARGET_HOLDOUT_FRACTION)
        candidates = []
        for validation_group in sorted(groups):
            for test_group in sorted(groups):
                if validation_group == test_group:
                    continue
                score = (
                    abs(len(groups[validation_group]) - target)
                    + abs(len(groups[test_group]) - target)
                    + 0.25 * abs(len(groups[validation_group]) - len(groups[test_group]))
                )
                tie = hashlib.sha256(
                    f"{SEED}|{species}|{validation_group}|{test_group}".encode()
                ).hexdigest()
                candidates.append((score, tie, test_group))
        _, _, test_group = min(candidates)
        selected = min(
            groups[test_group],
            key=lambda image: hashlib.sha256(
                f"{SEED}|pilot|{image['image_id']}".encode()
            ).hexdigest(),
        )
        strict_pilot_with_truth.append(
            {
                "image_id": selected["image_id"],
                "file": selected["file"],
                "path": str(images_dir / selected["file"]),
                "individual_id": test_group,
                "organ_category": selected.get("organ_category") or "unknown",
                "ground_truth_species": species,
            }
        )
    strict_pilot_with_truth.sort(key=lambda row: row["ground_truth_species"])
    if len(strict_pilot_with_truth) != 63:
        raise RuntimeError(
            f"Expected one pilot image for 63 species, found {len(strict_pilot_with_truth)}"
        )

    selected_with_truth = (
        all_rows_with_truth
        if stage == "full"
        else strict_pilot_with_truth[:3]
        if stage == "smoke"
        else strict_pilot_with_truth
    )
    truth_path.write_text(
        json.dumps(
            {
                row["image_id"]: {
                    "species": row["ground_truth_species"],
                    "organ_category": row["organ_category"],
                }
                for row in selected_with_truth
            }
        ),
        encoding="utf-8",
    )
    inference_manifest = [
        {
            key: row[key]
            for key in ("image_id", "file", "path", "individual_id", "organ_category")
        }
        for row in selected_with_truth
    ]
    del selected_with_truth, all_rows_with_truth, corpus, scenes, strict_pilot_with_truth

    scientific_names = [item["scientific_name"] for item in all_classes]
    alias_by_species = {
        item["scientific_name"]: item["vernacular"] for item in all_classes
    }
    response_schema = {
        "type": "object",
        "properties": {
            "ranked_species": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {"type": "string", "enum": scientific_names},
            },
            "confidences": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "required": ["ranked_species", "confidences"],
        "additionalProperties": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://bioimages-bioclip-benchmark.lm3879.chatgpt.site/",
        "X-Title": "BioImages direct VLM benchmark",
        "X-OpenRouter-Metadata": "enabled",
    }

    def encode_image(path: str) -> tuple[str, int, int]:
        output = io.BytesIO()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            width, height = image.size
            image.save(output, format="JPEG", quality=90, optimize=True)
        payload = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}", width, height

    def candidate_prompt(image_id: str) -> str:
        order_key = "candidate-order" if candidate_order_strategy == "fixed" else image_id
        ordered_candidates = sorted(
            all_classes,
            key=lambda item: hashlib.sha256(
                f"{SEED}|{order_key}|{item['scientific_name']}".encode()
            ).hexdigest(),
        )
        lines = []
        for item in ordered_candidates:
            label = item["scientific_name"]
            if item["vernacular"] and item["vernacular"].lower() != label.lower():
                label += f" ({item['vernacular']})"
            lines.append(f"- {label}")
        return (
            "Perform closed-set botanical species identification from the image. "
            "Use visible morphology only. Select exactly five distinct candidates from "
            "the list, rank them most likely first, and return each scientific name "
            "exactly as written in ranked_species. Put the five corresponding self-assessed "
            "scores between 0 and 1 in confidences; they need not sum to 1. Do not use "
            "filenames, metadata, web search, or any information outside the image and "
            "candidate list.\n\nCandidates:\n" + "\n".join(lines)
        )

    def generation_metadata(generation_id: str) -> dict:
        for delay in (0.4, 0.8, 1.6, 3.2):
            time.sleep(delay)
            try:
                response = requests.get(
                    "https://openrouter.ai/api/v1/generation",
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"id": generation_id},
                    timeout=30,
                )
                if response.status_code == 200:
                    return response.json().get("data", {})
            except requests.RequestException:
                pass
        return {}

    def call_one(model_key: str, model_id: str, row: dict) -> dict:
        request_started = time.perf_counter()
        try:
            image_url, width, height = encode_image(row["path"])
        except Exception as exc:
            return {
                **row,
                "model_key": model_key,
                "model_id": model_id,
                "status": "decode_error",
                "error": repr(exc),
                "latency_seconds": time.perf_counter() - request_started,
            }
        image_part = {"type": "image_url", "image_url": {"url": image_url}}
        if model_key == "gpt_5_4":
            image_part["image_url"]["detail"] = gpt_image_detail
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": candidate_prompt(row["image_id"])},
                        image_part,
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": (
                2000
                if model_key == "gpt_5_4" and gpt_reasoning_effort == "high"
                else 1200
                if model_key == "gpt_5_4" and gpt_reasoning_effort == "medium"
                else 700
                if model_key == "gpt_5_4"
                else 300
            ),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "botanical_species_top5",
                    "strict": True,
                    "schema": response_schema,
                },
            },
            "usage": {"include": True},
        }
        if model_key == "gpt_5_4":
            payload["reasoning_effort"] = gpt_reasoning_effort
            payload["service_tier"] = "flex"

        last_error = ""
        attempt_costs: list[float] = []
        attempt_generation_ids: list[str] = []
        for attempt in range(1, 4):
            try:
                api_started = time.perf_counter()
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
                api_seconds = time.perf_counter() - api_started
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    time.sleep(2**attempt)
                    continue
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
                    break
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage") or {}
                generation_id = data.get("id", "")
                if generation_id:
                    attempt_generation_ids.append(generation_id)
                cost = usage.get("cost")
                metadata = {}
                if cost is None and generation_id:
                    metadata = generation_metadata(generation_id)
                    cost = metadata.get("total_cost", metadata.get("usage"))
                if cost is not None:
                    attempt_costs.append(float(cost))
                openrouter_metadata = data.get("openrouter_metadata") or {}
                raw_content = data["choices"][0]["message"]["content"]
                if isinstance(raw_content, list):
                    raw_content = "".join(
                        part.get("text", "") for part in raw_content if isinstance(part, dict)
                    )
                parsed = json.loads(raw_content)
                names = parsed["ranked_species"]
                confidences = parsed["confidences"]
                if len(names) != 5 or len(confidences) != 5 or len(set(names)) != 5:
                    raise ValueError(f"Top-5 is not five distinct classes: {names}")
                if any(name not in scientific_names for name in names):
                    raise ValueError(f"Out-of-list prediction: {names}")
                top5 = [
                    {"species": name, "confidence": float(confidence)}
                    for name, confidence in zip(names, confidences)
                ]

                return {
                    **row,
                    "model_key": model_key,
                    "model_id": model_id,
                    "served_model": data.get("model", ""),
                    "provider": metadata.get("provider_name")
                    or openrouter_metadata.get("provider_name", ""),
                    "service_tier": data.get("service_tier")
                    or metadata.get("service_tier", ""),
                    "generation_id": generation_id,
                    "predicted_species": names[0],
                    "confidence": float(top5[0]["confidence"]),
                    "top5": top5,
                    "raw_response": raw_content,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens"
                    ),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    )
                    or metadata.get("native_tokens_reasoning"),
                    "cost_usd": sum(attempt_costs) if attempt_costs else None,
                    "final_attempt_cost_usd": float(cost) if cost is not None else None,
                    "attempt_generation_ids": attempt_generation_ids,
                    "image_width": width,
                    "image_height": height,
                    "api_seconds": api_seconds,
                    "latency_seconds": time.perf_counter() - request_started,
                    "attempts": attempt,
                    "status": "ok",
                    "error": "",
                }
            except Exception as exc:
                last_error = repr(exc)
                if attempt < 3:
                    payload["messages"][0]["content"][0]["text"] += (
                        "\n\nVALIDATION CORRECTION: The previous response was invalid or "
                        "contained a duplicate. Return five different species names and "
                        "valid complete JSON."
                    )
                    time.sleep(2**attempt)
        return {
            **row,
            "model_key": model_key,
            "model_id": model_id,
            "status": "api_error",
            "error": last_error,
            "cost_usd": sum(attempt_costs) if attempt_costs else None,
            "attempt_generation_ids": attempt_generation_ids,
            "latency_seconds": time.perf_counter() - request_started,
            "attempts": 3,
        }

    predictions_without_truth = []
    model_wall_seconds = {}
    for model_key, model_id in selected_models.items():
        model_started = time.perf_counter()
        # Finish one request first so providers can cache the identical text/schema
        # prefix before the remaining image requests are submitted concurrently.
        predictions_without_truth.append(
            call_one(model_key, model_id, inference_manifest[0])
        )
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(call_one, model_key, model_id, row): row["image_id"]
                for row in inference_manifest[1:]
            }
            for future in as_completed(futures):
                predictions_without_truth.append(future.result())
        model_wall_seconds[model_key] = time.perf_counter() - model_started

    # Ground truth becomes visible only after every requested model has completed.
    truth_by_id = json.loads(truth_path.read_text(encoding="utf-8"))
    evaluated = []
    for row in predictions_without_truth:
        truth = truth_by_id[row["image_id"]]
        output = {key: value for key, value in row.items() if key != "path"}
        output["ground_truth_species"] = truth["species"]
        output["organ_category"] = truth["organ_category"]
        if row["status"] == "ok":
            names = [item["species"] for item in row["top5"]]
            output["correct"] = names[0] == truth["species"]
            output["correct_top5"] = truth["species"] in names
        else:
            output["correct"] = False
            output["correct_top5"] = False
        evaluated.append(output)
    evaluated.sort(key=lambda row: (row["model_key"], row["image_id"]))

    summaries = {}
    for model_key, model_id in selected_models.items():
        rows = [row for row in evaluated if row["model_key"] == model_key]
        successful = [row for row in rows if row["status"] == "ok"]
        costs = [row["cost_usd"] for row in rows if row.get("cost_usd") is not None]
        latencies = [row["latency_seconds"] for row in successful]
        summaries[model_key] = {
            "model_id": model_id,
            "requested_images": len(rows),
            "successful_images": len(successful),
            "top1_correct": sum(row["correct"] for row in rows),
            "top1_accuracy": sum(row["correct"] for row in rows) / len(rows),
            "top5_correct": sum(row["correct_top5"] for row in rows),
            "top5_accuracy": sum(row["correct_top5"] for row in rows) / len(rows),
            "cost_records": len(costs),
            "total_cost_usd": sum(costs) if costs else None,
            "mean_cost_per_image_usd": sum(costs) / len(rows) if costs else None,
            "projected_1899_cost_usd": (
                sum(costs) / len(rows) * 1899 if costs else None
            ),
            "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
            "max_latency_seconds": max(latencies) if latencies else None,
            "wall_seconds": model_wall_seconds[model_key],
            "status_counts": dict(Counter(row["status"] for row in rows)),
        }

    total_pilot_cost = sum(
        summary["total_cost_usd"] or 0 for summary in summaries.values()
    )
    projected_full_cost = sum(
        summary["projected_1899_cost_usd"] or 0 for summary in summaries.values()
    )
    summary = {
        "stage": stage,
        "models": summaries,
        "candidate_species": len(scientific_names),
        "candidate_protocol": "85 scientific names with vernacular aliases",
        "selected_images": len(inference_manifest),
        "selected_species": len({truth["species"] for truth in truth_by_id.values()}),
        "pilot_selection": (
            "one deterministic image from each strict held-out individual"
            if stage != "full"
            else "all original BioImages images"
        ),
        "ground_truth_isolation": (
            "The API inference manifest contained no per-image species labels. Ground truth "
            "was read only after every requested model completed all requested predictions."
        ),
        "image_preprocessing": (
            f"EXIF transpose, RGB conversion, longest side capped at {MAX_IMAGE_SIDE}px, "
            "JPEG quality 90, metadata removed"
        ),
        "total_actual_cost_usd": total_pilot_cost,
        "projected_1899_both_models_cost_usd": projected_full_cost,
        "dataset_download_and_extract_seconds": dataset_seconds,
        "total_run_seconds": time.perf_counter() - started,
        "concurrency": concurrency,
        "requested_models": requested_model_keys,
        "gpt_reasoning_effort": gpt_reasoning_effort,
        "gpt_image_detail": gpt_image_detail,
        "candidate_order_strategy": candidate_order_strategy,
        "seed": SEED,
        "python_version": platform.python_version(),
    }

    prediction_fields = [
        "image_id",
        "file",
        "individual_id",
        "organ_category",
        "model_key",
        "model_id",
        "served_model",
        "provider",
        "service_tier",
        "generation_id",
        "predicted_species",
        "confidence",
        "top5",
        "ground_truth_species",
        "correct",
        "correct_top5",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cost_usd",
        "image_width",
        "image_height",
        "api_seconds",
        "latency_seconds",
        "attempts",
        "status",
        "error",
    ]
    csv_rows = []
    for row in evaluated:
        csv_row = {key: row.get(key, "") for key in prediction_fields}
        csv_row["top5"] = json.dumps(row.get("top5", []), ensure_ascii=False)
        csv_rows.append(csv_row)

    readme = f"""# BioImages OpenRouter direct-VLM benchmark ({stage})

- Requested models: `{', '.join(selected_models.values())}`
- Candidate set: 85 species, scientific names plus vernacular aliases
- Requested images per model: {len(inference_manifest)}
- Actual combined API cost: ${total_pilot_cost:.6f}
- Projected combined cost for 1,899 images per model: ${projected_full_cost:.4f}
- GPT reasoning effort: `{gpt_reasoning_effort}`
- GPT image detail: `{gpt_image_detail}`
- Candidate order strategy: `{candidate_order_strategy}`

Ground truth was unavailable to the inference requests and was loaded only after
both models completed. Confidence is model-self-reported and is not calibrated.
"""
    files = {
        "README.md": readme.encode("utf-8"),
        "summary.json": json.dumps(summary, indent=2, ensure_ascii=False).encode(),
        "predictions.csv": _csv_bytes(csv_rows, prediction_fields),
        "raw_responses.jsonl": _jsonl_bytes(evaluated),
        "candidate_species.json": json.dumps(
            [
                {
                    "scientific_name": name,
                    "vernacular": alias_by_species[name],
                }
                for name in scientific_names
            ],
            indent=2,
            ensure_ascii=False,
        ).encode(),
    }
    shutil.rmtree(data_root, ignore_errors=True)
    Path(zip_path).unlink(missing_ok=True)
    truth_path.unlink(missing_ok=True)
    return _package(files)


@app.local_entrypoint()
def main(
    stage: str = "pilot",
    concurrency: int = 6,
    output: str = "",
    model_keys: str = "gemini_flash_lite,gpt_5_4",
    gpt_reasoning_effort: str = "minimal",
    gpt_image_detail: str = "auto",
    candidate_order_strategy: str = "fixed",
) -> None:
    if not output:
        output = f"outputs/openrouter_vlm_{stage}"
    payload = run_benchmark.remote(
        stage,
        concurrency,
        model_keys,
        gpt_reasoning_effort,
        gpt_image_detail,
        candidate_order_strategy,
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(output_path)
    print((output_path / "summary.json").read_text(encoding="utf-8"))
