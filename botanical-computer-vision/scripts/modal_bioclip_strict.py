"""BioCLIP 2.5 zero-shot on the exact DINOv3 strict test split.

The Google Drive archive and model stay in Modal. The returned bundle contains
only predictions and aggregate metrics. Candidate classes and test image IDs
are recomputed with the same deterministic protocol as modal_dinov3_strict.py.
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
MODEL_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"
SEED = 42
TARGET_HOLDOUT_FRACTION = 0.15

runtime = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gdown==5.2.0",
        "numpy==2.2.6",
        "open_clip_torch==3.2.0",
        "pillow==11.3.0",
        "torch==2.7.1",
        "torchvision==0.22.1",
    )
)
app = modal.App("bioimages-bioclip25-strict-zero-shot", image=runtime)


def _csv_bytes(rows: list[dict], fieldnames: list[str]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _package(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


@app.function(gpu="L4", timeout=60 * 60, cpu=4.0, memory=32768)
def run_benchmark() -> bytes:
    import hashlib
    import os
    import platform
    import shutil
    import time
    import zipfile
    from collections import Counter, defaultdict

    import gdown
    import numpy as np
    import open_clip
    import torch
    from PIL import Image, ImageFile

    started = time.perf_counter()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the requested Modal L4 container")

    zip_path = "/tmp/bioimages.zip"
    data_root = Path("/tmp/bioimages")
    download_started = time.perf_counter()
    if not gdown.download(DRIVE_URL, zip_path, quiet=False, fuzzy=True):
        raise RuntimeError("Google Drive download failed")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_root)
    data_seconds = time.perf_counter() - download_started

    corpus_path = data_root / "bioimages-ne-trees" / "corpus.json"
    images_dir = data_root / "bioimages-ne-trees" / "images"
    with corpus_path.open("r", encoding="utf-8") as handle:
        corpus = json.load(handle)

    eligible_scenes = []
    for scene in corpus["scenes"]:
        groups = {image["individual_id"] for image in scene["images"]}
        if len(groups) >= 3:
            eligible_scenes.append(scene)
    eligible_scenes.sort(key=lambda scene: scene["scientific_name"].strip())
    if len(eligible_scenes) != 63:
        raise RuntimeError(f"Expected 63 eligible species, found {len(eligible_scenes)}")

    classes = [
        {
            "scientific_name": scene["scientific_name"].strip(),
            "vernacular": (scene.get("vernacular") or "").strip(),
        }
        for scene in eligible_scenes
    ]

    test_rows_with_truth = []
    for scene in eligible_scenes:
        species = scene["scientific_name"].strip()
        groups: dict[str, list[dict]] = defaultdict(list)
        for image in scene["images"]:
            groups[image["individual_id"]].append(image)
        target = max(1.0, len(scene["images"]) * TARGET_HOLDOUT_FRACTION)
        candidates = []
        for validation_group in sorted(groups):
            for test_group in sorted(groups):
                if validation_group == test_group:
                    continue
                score = (
                    abs(len(groups[validation_group]) - target)
                    + abs(len(groups[test_group]) - target)
                    + 0.25
                    * abs(len(groups[validation_group]) - len(groups[test_group]))
                )
                tie = hashlib.sha256(
                    f"{SEED}|{species}|{validation_group}|{test_group}".encode()
                ).hexdigest()
                candidates.append((score, tie, validation_group, test_group))
        _, _, _, test_group = min(candidates)
        for image in groups[test_group]:
            test_rows_with_truth.append(
                {
                    "image_id": image["image_id"],
                    "file": image["file"],
                    "path": str(images_dir / image["file"]),
                    "individual_id": test_group,
                    "organ_category": image.get("organ_category") or "unknown",
                    "ground_truth_species": species,
                }
            )

    if len(test_rows_with_truth) != 211:
        raise RuntimeError(f"Expected 211 strict test images, found {len(test_rows_with_truth)}")

    truth_path = Path("/tmp/strict_test_truth.json")
    truth_path.write_text(
        json.dumps(
            {
                row["image_id"]: {
                    "species": row["ground_truth_species"],
                    "organ_category": row["organ_category"],
                }
                for row in test_rows_with_truth
            }
        ),
        encoding="utf-8",
    )
    inference_manifest = [
        {
            key: row[key]
            for key in ("image_id", "file", "path", "individual_id", "organ_category")
        }
        for row in test_rows_with_truth
    ]
    del test_rows_with_truth, corpus, eligible_scenes

    model_started = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.eval().to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_seconds = time.perf_counter() - model_started

    text_embeddings = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in classes:
            prompts = [item["scientific_name"]]
            if item["vernacular"] and item["vernacular"].lower() != item["scientific_name"].lower():
                prompts.append(item["vernacular"])
            features = model.encode_text(tokenizer(prompts).to("cuda"))
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            text_embeddings.append(
                torch.nn.functional.normalize(features.mean(dim=0), dim=-1)
            )
        text_matrix = torch.stack(text_embeddings, dim=1)
    logit_scale = float(model.logit_scale.exp().detach().cpu())

    def predict(records: list[dict], batch_size: int, stage: str):
        outputs = []
        gpu_seconds = 0.0
        wall_started = time.perf_counter()
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            tensors = []
            valid_records = []
            for row in batch_records:
                try:
                    with Image.open(row["path"]) as image:
                        tensors.append(preprocess(image.convert("RGB")))
                    valid_records.append(row)
                except Exception as exc:
                    outputs.append(
                        {**row, "stage": stage, "status": "decode_error", "error": repr(exc)}
                    )
            if not tensors:
                continue
            batch = torch.stack(tensors).to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            gpu_started = time.perf_counter()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    image_features = torch.nn.functional.normalize(
                        model.encode_image(batch).float(), dim=-1
                    )
                    probabilities = (logit_scale * image_features @ text_matrix).softmax(dim=-1)
                    top_probs, top_indices = probabilities.topk(5, dim=-1)
                torch.cuda.synchronize()
                gpu_seconds += time.perf_counter() - gpu_started
                for row, probs, indices in zip(valid_records, top_probs.cpu(), top_indices.cpu()):
                    top5 = [
                        {
                            "species": classes[int(index)]["scientific_name"],
                            "confidence": float(prob),
                        }
                        for prob, index in zip(probs, indices)
                    ]
                    outputs.append(
                        {
                            **row,
                            "stage": stage,
                            "status": "ok",
                            "error": "",
                            "predicted_species": top5[0]["species"],
                            "confidence": top5[0]["confidence"],
                            "top5": top5,
                        }
                    )
            except Exception as exc:
                torch.cuda.synchronize()
                gpu_seconds += time.perf_counter() - gpu_started
                for row in valid_records:
                    outputs.append(
                        {**row, "stage": stage, "status": "inference_error", "error": repr(exc)}
                    )
        return outputs, gpu_seconds, time.perf_counter() - wall_started

    predict(inference_manifest[:1], 1, "warmup")
    small_rows, small_gpu_seconds, _ = predict(inference_manifest[:8], 8, "small_test")
    small_ok = sum(row["status"] == "ok" for row in small_rows)
    seconds_per_image = small_gpu_seconds / max(small_ok, 1)
    if small_ok != 8:
        raise RuntimeError(f"Small test failed: {small_ok}/8")
    if seconds_per_image > 10.0:
        raise RuntimeError(f"Abnormal GPU inference speed: {seconds_per_image:.2f}s/image")

    prediction_rows, gpu_seconds, wall_seconds = predict(
        inference_manifest, 32, "strict_test"
    )
    if len(prediction_rows) != 211 or any(row["status"] != "ok" for row in prediction_rows):
        raise RuntimeError("Strict test inference did not produce 211 successful rows")

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    per_species_total = Counter()
    per_species_top1 = Counter()
    per_species_top5 = Counter()
    per_organ_total = Counter()
    per_organ_top1 = Counter()
    per_organ_top5 = Counter()
    error_pairs = Counter()
    evaluated = []
    for row in prediction_rows:
        ground_truth = truth[row["image_id"]]["species"]
        organ = truth[row["image_id"]]["organ_category"]
        top5_species = [item["species"] for item in row["top5"]]
        correct = row["predicted_species"] == ground_truth
        correct_top5 = ground_truth in top5_species
        per_species_total[ground_truth] += 1
        per_species_top1[ground_truth] += int(correct)
        per_species_top5[ground_truth] += int(correct_top5)
        per_organ_total[organ] += 1
        per_organ_top1[organ] += int(correct)
        per_organ_top5[organ] += int(correct_top5)
        if not correct:
            error_pairs[(ground_truth, row["predicted_species"])] += 1
        evaluated.append(
            {
                "image_id": row["image_id"],
                "file": row["file"],
                "individual_id": row["individual_id"],
                "organ_category": organ,
                "predicted_species": row["predicted_species"],
                "confidence": round(row["confidence"], 8),
                "top5": json.dumps(row["top5"], ensure_ascii=False),
                "ground_truth_species": ground_truth,
                "correct": correct,
                "correct_top5": correct_top5,
                "status": row["status"],
                "error": row["error"],
            }
        )

    per_species = []
    for item in classes:
        species = item["scientific_name"]
        total = per_species_total[species]
        per_species.append(
            {
                "species": species,
                "vernacular": item["vernacular"],
                "total": total,
                "top1_correct": per_species_top1[species],
                "top1_accuracy": per_species_top1[species] / total,
                "top5_correct": per_species_top5[species],
                "top5_accuracy": per_species_top5[species] / total,
            }
        )
    per_organ = [
        {
            "organ_category": organ,
            "total": per_organ_total[organ],
            "top1_correct": per_organ_top1[organ],
            "top1_accuracy": per_organ_top1[organ] / per_organ_total[organ],
            "top5_correct": per_organ_top5[organ],
            "top5_accuracy": per_organ_top5[organ] / per_organ_total[organ],
        }
        for organ in sorted(per_organ_total)
    ]
    pair_rows = [
        {"ground_truth_species": actual, "predicted_species": predicted, "count": count}
        for (actual, predicted), count in error_pairs.most_common()
    ]
    top1_correct = sum(row["correct"] for row in evaluated)
    top5_correct = sum(row["correct_top5"] for row in evaluated)
    macro_top1 = sum(row["top1_accuracy"] for row in per_species) / len(per_species)
    summary = {
        "model": MODEL_NAME,
        "method": "zero-shot closed-set cosine similarity",
        "candidate_classes": "63 strict-split species; scientific name plus vernacular alias",
        "training_required": False,
        "ground_truth_isolation": (
            "The inference manifest contained no species labels. Test truth was re-read only "
            "after all 211 predictions and Top-5 lists had been created."
        ),
        "total_images": len(evaluated),
        "successful_images": len(evaluated),
        "species_count": len(classes),
        "top1_correct": top1_correct,
        "top1_accuracy": top1_correct / len(evaluated),
        "top5_correct": top5_correct,
        "top5_accuracy": top5_correct / len(evaluated),
        "macro_top1_accuracy": macro_top1,
        "small_test_images": small_ok,
        "small_test_gpu_seconds": small_gpu_seconds,
        "small_test_seconds_per_image": seconds_per_image,
        "strict_test_gpu_seconds": gpu_seconds,
        "strict_test_wall_seconds": wall_seconds,
        "dataset_download_and_extract_seconds": data_seconds,
        "model_download_and_load_seconds": model_seconds,
        "total_run_seconds": time.perf_counter() - started,
        "batch_size": 32,
        "precision": "bfloat16 autocast",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "open_clip_version": getattr(open_clip, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "seed": SEED,
    }

    files = {
        "predictions.csv": _csv_bytes(
            evaluated,
            [
                "image_id", "file", "individual_id", "organ_category",
                "predicted_species", "confidence", "top5", "ground_truth_species",
                "correct", "correct_top5", "status", "error",
            ],
        ),
        "per_species.csv": _csv_bytes(
            per_species,
            [
                "species", "vernacular", "total", "top1_correct", "top1_accuracy",
                "top5_correct", "top5_accuracy",
            ],
        ),
        "per_organ.csv": _csv_bytes(
            per_organ,
            [
                "organ_category", "total", "top1_correct", "top1_accuracy",
                "top5_correct", "top5_accuracy",
            ],
        ),
        "error_pairs.csv": _csv_bytes(
            pair_rows, ["ground_truth_species", "predicted_species", "count"]
        ),
        "summary.json": json.dumps(summary, indent=2, ensure_ascii=False).encode(),
        "run_status.json": json.dumps(
            {"small_test": small_rows, "strict_test_status": "ok"},
            indent=2,
            ensure_ascii=False,
        ).encode(),
    }
    shutil.rmtree(data_root, ignore_errors=True)
    Path(zip_path).unlink(missing_ok=True)
    return _package(files)


@app.local_entrypoint()
def main(output: str = "outputs/bioclip25_strict_zero_shot") -> None:
    payload = run_benchmark.remote()
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(output_path)
    print((output_path / "summary.json").read_text(encoding="utf-8"))
