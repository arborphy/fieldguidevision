"""Strict individual-disjoint frozen-backbone linear-probe benchmark on Modal.

Images and model weights are downloaded inside Modal. Test labels are written to
a separate file and are not re-read until test predictions have been produced.
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
DINO_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
BIOCLIP_MODEL_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"
EFFICIENTNET_MODEL_NAME = "torchvision/efficientnet_b0-imagenet1k-v1"
SEED = 42
TARGET_HOLDOUT_FRACTION = 0.15

runtime = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gdown==5.2.0",
        "huggingface_hub==0.36.0",
        "numpy==2.2.6",
        "open_clip_torch==3.2.0",
        "pillow==11.3.0",
        "scikit-learn==1.7.2",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers==4.57.1",
    )
)

app = modal.App("bioimages-dinov3-strict-linear-probe", image=runtime)
hf_secret = modal.Secret.from_name("huggingface-secret")


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


@app.function(
    gpu="L4",
    timeout=60 * 60,
    cpu=8.0,
    memory=32768,
    secrets=[hf_secret],
)
def run_benchmark(backbone: str = "dinov3") -> bytes:
    import hashlib
    import os
    import platform
    import time
    import zipfile
    from collections import Counter, defaultdict

    import gdown
    import numpy as np
    import open_clip
    import torch
    from PIL import Image, ImageFile
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    from transformers import AutoImageProcessor, AutoModel

    started = time.perf_counter()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the requested Modal L4 container")
    if backbone not in {"dinov3", "bioclip", "efficientnet_b0"}:
        raise ValueError(f"Unsupported backbone: {backbone}")
    hf_token = os.environ.get("HF_TOKEN")
    if backbone == "dinov3" and not hf_token:
        raise RuntimeError("HF_TOKEN is missing from Modal secret 'huggingface-secret'")

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

    # Strict benchmark universe: only species with at least three independent
    # individual_id groups, so every class can appear in train/validation/test.
    eligible_scenes = []
    for scene in corpus["scenes"]:
        groups = {image["individual_id"] for image in scene["images"]}
        if len(groups) >= 3:
            eligible_scenes.append(scene)

    classes = sorted(scene["scientific_name"].strip() for scene in eligible_scenes)
    class_to_index = {name: index for index, name in enumerate(classes)}
    if len(classes) != 63:
        raise RuntimeError(f"Expected 63 eligible species, found {len(classes)}")

    # For each species, choose one validation and one test individual. The pair
    # is selected to be as close as possible to 15% of that species' images.
    # SHA-256 provides deterministic seed-based tie breaking across Python runs.
    split_manifest_with_truth = []
    split_groups: dict[str, dict[str, list[str]]] = {}
    for scene in eligible_scenes:
        species = scene["scientific_name"].strip()
        groups: dict[str, list[dict]] = defaultdict(list)
        for image in scene["images"]:
            groups[image["individual_id"]].append(image)
        group_ids = sorted(groups)
        target = max(1.0, len(scene["images"]) * TARGET_HOLDOUT_FRACTION)

        candidates = []
        for validation_group in group_ids:
            for test_group in group_ids:
                if validation_group == test_group:
                    continue
                validation_size = len(groups[validation_group])
                test_size = len(groups[test_group])
                score = (
                    abs(validation_size - target)
                    + abs(test_size - target)
                    + 0.25 * abs(validation_size - test_size)
                )
                tie = hashlib.sha256(
                    f"{SEED}|{species}|{validation_group}|{test_group}".encode()
                ).hexdigest()
                candidates.append((score, tie, validation_group, test_group))
        _, _, validation_group, test_group = min(candidates)
        split_groups[species] = {
            "validation": [validation_group],
            "test": [test_group],
            "train": [
                group_id
                for group_id in group_ids
                if group_id not in {validation_group, test_group}
            ],
        }

        for group_id, images in groups.items():
            split = (
                "validation"
                if group_id == validation_group
                else "test"
                if group_id == test_group
                else "train"
            )
            for image in images:
                split_manifest_with_truth.append(
                    {
                        "image_id": image["image_id"],
                        "file": image["file"],
                        "path": str(images_dir / image["file"]),
                        "split": split,
                        "individual_id": group_id,
                        "organ_category": image.get("organ_category") or "unknown",
                        "ground_truth_species": species,
                    }
                )

    if len(split_manifest_with_truth) != 1641:
        raise RuntimeError(
            f"Expected 1,641 eligible images, found {len(split_manifest_with_truth)}"
        )

    # Hard leakage checks before any model work.
    for species in classes:
        rows = [row for row in split_manifest_with_truth if row["ground_truth_species"] == species]
        group_sets = {
            split: {row["individual_id"] for row in rows if row["split"] == split}
            for split in ("train", "validation", "test")
        }
        if any(not values for values in group_sets.values()):
            raise RuntimeError(f"Empty split for {species}: {group_sets}")
        if (
            group_sets["train"] & group_sets["validation"]
            or group_sets["train"] & group_sets["test"]
            or group_sets["validation"] & group_sets["test"]
        ):
            raise RuntimeError(f"individual_id leakage for {species}: {group_sets}")

    split_counts = Counter(row["split"] for row in split_manifest_with_truth)
    species_by_id = {
        row["image_id"]: row["ground_truth_species"]
        for row in split_manifest_with_truth
    }
    train_labels = {
        row["image_id"]: class_to_index[row["ground_truth_species"]]
        for row in split_manifest_with_truth
        if row["split"] == "train"
    }
    validation_labels = {
        row["image_id"]: class_to_index[row["ground_truth_species"]]
        for row in split_manifest_with_truth
        if row["split"] == "validation"
    }
    test_truth_path = Path("/tmp/test_truth.json")
    test_truth_path.write_text(
        json.dumps(
            {
                row["image_id"]: row["ground_truth_species"]
                for row in split_manifest_with_truth
                if row["split"] == "test"
            }
        ),
        encoding="utf-8",
    )

    # The embedding manifest deliberately contains no species labels.
    inference_manifest = [
        {
            key: row[key]
            for key in (
                "image_id",
                "file",
                "path",
                "split",
                "individual_id",
                "organ_category",
            )
        }
        for row in split_manifest_with_truth
    ]
    pilot_ids = []
    for species in classes:
        pilot_ids.append(
            next(
                row["image_id"]
                for row in split_manifest_with_truth
                if row["ground_truth_species"] == species
            )
        )
    pilot_id_set = set(pilot_ids)
    pilot_manifest = [
        row for row in inference_manifest if row["image_id"] in pilot_id_set
    ]

    # Keep the final output manifest on disk, but remove in-memory test truth
    # before feature extraction and classifier training.
    output_manifest = [
        {key: value for key, value in row.items() if key != "path"}
        for row in split_manifest_with_truth
    ]
    del split_manifest_with_truth, corpus, eligible_scenes, species_by_id

    model_started = time.perf_counter()
    if backbone == "dinov3":
        model_name = DINO_MODEL_NAME
        processor = AutoImageProcessor.from_pretrained(model_name, token=hf_token)
        model = AutoModel.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16,
        ).eval().to("cuda")
        preprocess = None
        batch_size = 64
    elif backbone == "bioclip":
        model_name = BIOCLIP_MODEL_NAME
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        model = model.eval().to("cuda")
        processor = None
        batch_size = 32
    else:
        model_name = EFFICIENTNET_MODEL_NAME
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights).eval().to("cuda")
        preprocess = weights.transforms()
        processor = None
        batch_size = 128
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_seconds = time.perf_counter() - model_started

    def extract_embeddings(records: list[dict], batch_size: int, stage: str):
        ids: list[str] = []
        features: list[np.ndarray] = []
        statuses: list[dict] = []
        gpu_seconds = 0.0
        wall_started = time.perf_counter()

        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            images = []
            valid_records = []
            for row in batch_records:
                try:
                    with Image.open(row["path"]) as image:
                        rgb = image.convert("RGB")
                        images.append(preprocess(rgb) if preprocess is not None else rgb)
                    valid_records.append(row)
                except Exception as exc:
                    statuses.append(
                        {
                            "image_id": row["image_id"],
                            "stage": stage,
                            "status": "decode_error",
                            "error": repr(exc),
                        }
                    )
            if not images:
                continue

            if backbone == "dinov3":
                inputs = processor(images=images, return_tensors="pt")
                inputs = {
                    key: value.to("cuda", non_blocking=True)
                    for key, value in inputs.items()
                }
            else:
                inputs = torch.stack(images).to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            gpu_started = time.perf_counter()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if backbone == "dinov3":
                        outputs = model(**inputs)
                        pooled = outputs.pooler_output.float()
                    elif backbone == "bioclip":
                        pooled = model.encode_image(inputs).float()
                    else:
                        feature_map = model.features(inputs)
                        pooled = torch.flatten(model.avgpool(feature_map), 1).float()
                    pooled = torch.nn.functional.normalize(pooled, dim=-1)
                torch.cuda.synchronize()
                gpu_seconds += time.perf_counter() - gpu_started
                pooled_cpu = pooled.cpu().numpy()
                for row, feature in zip(valid_records, pooled_cpu):
                    ids.append(row["image_id"])
                    features.append(feature)
                    statuses.append(
                        {
                            "image_id": row["image_id"],
                            "stage": stage,
                            "status": "ok",
                            "error": "",
                        }
                    )
            except Exception as exc:
                torch.cuda.synchronize()
                gpu_seconds += time.perf_counter() - gpu_started
                for row in valid_records:
                    statuses.append(
                        {
                            "image_id": row["image_id"],
                            "stage": stage,
                            "status": "inference_error",
                            "error": repr(exc),
                        }
                    )

        matrix = np.stack(features) if features else np.empty((0, 0), dtype=np.float32)
        return ids, matrix, statuses, gpu_seconds, time.perf_counter() - wall_started

    # Warm-up is excluded from timing.
    extract_embeddings(inference_manifest[:1], batch_size=1, stage="warmup")

    small_ids, small_x, small_status, small_gpu, small_wall = extract_embeddings(
        inference_manifest[:8], batch_size=8, stage="small_test"
    )
    if len(small_ids) != 8 or small_x.shape[0] != 8:
        raise RuntimeError(
            f"Small test failed: ids={len(small_ids)}, shape={small_x.shape}"
        )
    seconds_per_image = small_gpu / len(small_ids)
    if seconds_per_image > 10.0:
        raise RuntimeError(
            f"Abnormal {backbone} GPU speed: {seconds_per_image:.2f} seconds/image"
        )

    pilot_ids_out, pilot_x, pilot_status, pilot_gpu, pilot_wall = extract_embeddings(
        pilot_manifest, batch_size=32, stage="pilot"
    )
    if len(pilot_ids_out) != 63 or pilot_x.shape[0] != 63:
        raise RuntimeError(
            f"Pilot failed: ids={len(pilot_ids_out)}, shape={pilot_x.shape}"
        )

    full_ids, full_x, full_status, full_gpu, full_wall = extract_embeddings(
        inference_manifest, batch_size=batch_size, stage="full"
    )
    if len(full_ids) != 1641 or full_x.shape[0] != 1641:
        failures = [row for row in full_status if row["status"] != "ok"]
        raise RuntimeError(
            f"Full embedding failed: ids={len(full_ids)}, shape={full_x.shape}, "
            f"failures={failures[:3]}"
        )

    feature_by_id = {image_id: feature for image_id, feature in zip(full_ids, full_x)}
    train_ids = sorted(train_labels)
    validation_ids = sorted(validation_labels)
    test_ids = sorted(
        row["image_id"] for row in inference_manifest if row["split"] == "test"
    )
    train_x = np.stack([feature_by_id[image_id] for image_id in train_ids])
    train_y = np.array([train_labels[image_id] for image_id in train_ids])
    validation_x = np.stack([feature_by_id[image_id] for image_id in validation_ids])
    validation_y = np.array([validation_labels[image_id] for image_id in validation_ids])
    test_x = np.stack([feature_by_id[image_id] for image_id in test_ids])

    # Tune only the linear probe regularization on validation labels.
    probe_started = time.perf_counter()
    validation_grid = []
    candidate_cs = [0.01, 0.1, 1.0, 10.0, 100.0]
    for c_value in candidate_cs:
        classifier = LogisticRegression(
            C=c_value,
            max_iter=3000,
            solver="lbfgs",
            random_state=SEED,
        )
        classifier.fit(train_x, train_y)
        validation_predictions = classifier.predict(validation_x)
        validation_probabilities = classifier.predict_proba(validation_x)
        validation_top5 = np.argsort(-validation_probabilities, axis=1)[:, :5]
        validation_grid.append(
            {
                "C": c_value,
                "top1_accuracy": float(
                    accuracy_score(validation_y, validation_predictions)
                ),
                "macro_accuracy": float(
                    balanced_accuracy_score(validation_y, validation_predictions)
                ),
                "top5_accuracy": float(
                    np.mean(
                        [
                            truth in candidates
                            for truth, candidates in zip(validation_y, validation_top5)
                        ]
                    )
                ),
                "iterations": int(classifier.n_iter_.max()),
            }
        )

    best_validation = max(
        validation_grid,
        key=lambda row: (
            row["macro_accuracy"],
            row["top1_accuracy"],
            -abs(np.log10(row["C"])),
        ),
    )
    best_c = best_validation["C"]
    final_classifier = LogisticRegression(
        C=best_c,
        max_iter=3000,
        solver="lbfgs",
        random_state=SEED,
    )
    train_validation_x = np.concatenate([train_x, validation_x], axis=0)
    train_validation_y = np.concatenate([train_y, validation_y], axis=0)
    final_classifier.fit(train_validation_x, train_validation_y)

    # Produce test predictions before loading any test labels.
    test_probabilities = final_classifier.predict_proba(test_x)
    test_top5_indices = np.argsort(-test_probabilities, axis=1)[:, :5]
    test_prediction_indices = test_top5_indices[:, 0]
    test_prediction_records = []
    for image_id, probabilities, top_indices in zip(
        test_ids, test_probabilities, test_top5_indices
    ):
        test_prediction_records.append(
            {
                "image_id": image_id,
                "predicted_species": classes[int(top_indices[0])],
                "confidence": float(probabilities[int(top_indices[0])]),
                "top5": [
                    {
                        "species": classes[int(index)],
                        "confidence": float(probabilities[int(index)]),
                    }
                    for index in top_indices
                ],
            }
        )

    # Test ground truth becomes visible only after predictions exist.
    test_truth_by_id = json.loads(test_truth_path.read_text(encoding="utf-8"))
    test_y = np.array([class_to_index[test_truth_by_id[image_id]] for image_id in test_ids])
    probe_seconds = time.perf_counter() - probe_started

    metadata_by_id = {row["image_id"]: row for row in output_manifest}
    prediction_rows = []
    for prediction, predicted_index, truth_index in zip(
        test_prediction_records, test_prediction_indices, test_y
    ):
        image_id = prediction["image_id"]
        truth = test_truth_by_id[image_id]
        top5_species = [item["species"] for item in prediction["top5"]]
        prediction_rows.append(
            {
                "image_id": image_id,
                "file": metadata_by_id[image_id]["file"],
                "split": "test",
                "individual_id": metadata_by_id[image_id]["individual_id"],
                "organ_category": metadata_by_id[image_id]["organ_category"],
                "predicted_species": prediction["predicted_species"],
                "confidence": round(prediction["confidence"], 8),
                "top5": json.dumps(prediction["top5"], ensure_ascii=False),
                "ground_truth_species": truth,
                "correct": bool(int(predicted_index) == int(truth_index)),
                "correct_top5": truth in top5_species,
                "status": "ok",
                "error": "",
            }
        )

    top1_correct = sum(row["correct"] for row in prediction_rows)
    top5_correct = sum(row["correct_top5"] for row in prediction_rows)
    per_species_rows = []
    for species in classes:
        rows = [row for row in prediction_rows if row["ground_truth_species"] == species]
        top1_count = sum(row["correct"] for row in rows)
        top5_count = sum(row["correct_top5"] for row in rows)
        per_species_rows.append(
            {
                "species": species,
                "total": len(rows),
                "top1_correct": top1_count,
                "top1_accuracy": top1_count / len(rows),
                "top5_correct": top5_count,
                "top5_accuracy": top5_count / len(rows),
            }
        )

    per_organ_rows = []
    organs = sorted({row["organ_category"] for row in prediction_rows})
    for organ in organs:
        rows = [row for row in prediction_rows if row["organ_category"] == organ]
        top1_count = sum(row["correct"] for row in rows)
        top5_count = sum(row["correct_top5"] for row in rows)
        per_organ_rows.append(
            {
                "organ_category": organ,
                "total": len(rows),
                "top1_correct": top1_count,
                "top1_accuracy": top1_count / len(rows),
                "top5_correct": top5_count,
                "top5_accuracy": top5_count / len(rows),
            }
        )

    error_counter = Counter(
        (row["ground_truth_species"], row["predicted_species"])
        for row in prediction_rows
        if not row["correct"]
    )
    error_rows = [
        {
            "ground_truth_species": truth,
            "predicted_species": prediction,
            "count": count,
        }
        for (truth, prediction), count in error_counter.most_common()
    ]

    total_seconds = time.perf_counter() - started
    summary = {
        "model": model_name,
        "method": {
            "dinov3": "frozen DINOv3 CLS embedding + multinomial logistic regression",
            "bioclip": "frozen BioCLIP image embedding + multinomial logistic regression",
            "efficientnet_b0": (
                "frozen ImageNet-supervised EfficientNet-B0 global pooled penultimate-layer "
                "embedding + multinomial logistic regression"
            ),
        }[backbone],
        "backbone_trained": False,
        "embedding_dimension": int(full_x.shape[1]),
        "split_protocol": (
            "63 species with >=3 individual_id groups; exactly one validation and "
            "one test individual per species; no individual overlap"
        ),
        "seed": SEED,
        "total_images": len(inference_manifest),
        "species_count": len(classes),
        "train_images": split_counts["train"],
        "validation_images": split_counts["validation"],
        "test_images": split_counts["test"],
        "test_top1_correct": top1_correct,
        "test_top1_accuracy": top1_correct / len(prediction_rows),
        "test_top5_correct": top5_correct,
        "test_top5_accuracy": top5_correct / len(prediction_rows),
        "test_macro_top1_accuracy": float(
            np.mean([row["top1_accuracy"] for row in per_species_rows])
        ),
        "best_C": best_c,
        "best_validation_macro_accuracy": best_validation["macro_accuracy"],
        "small_test_images": len(small_ids),
        "small_test_gpu_seconds": small_gpu,
        "small_test_seconds_per_image": seconds_per_image,
        "pilot_images": len(pilot_ids_out),
        "pilot_gpu_seconds": pilot_gpu,
        "full_embedding_gpu_seconds": full_gpu,
        "full_embedding_wall_seconds": full_wall,
        "probe_training_and_prediction_seconds": probe_seconds,
        "dataset_download_and_extract_seconds": data_seconds,
        "model_download_and_load_seconds": model_seconds,
        "total_run_seconds": total_seconds,
        "batch_size": batch_size,
        "precision": "bfloat16 autocast",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "ground_truth_isolation": (
            "The frozen embedding manifest contained no species labels. Only train/validation "
            "labels were loaded for probe selection. Test truth was re-read after test "
            "probabilities and Top-5 predictions had been created."
        ),
    }

    display_name = {
        "dinov3": "DINOv3",
        "bioclip": "BioCLIP 2.5",
        "efficientnet_b0": "EfficientNet-B0 ImageNet",
    }[backbone]
    readme = f"""# Strict {display_name} frozen linear-probe benchmark

- Model: `{model_name}`
- Protocol: 63 species, 1,641 images, individual-disjoint train/validation/test
- Test Top-1: {summary['test_top1_accuracy']:.4%}
- Test Top-5: {summary['test_top5_accuracy']:.4%}
- Test macro Top-1: {summary['test_macro_top1_accuracy']:.4%}
- Best validation-selected C: {best_c}
- Backbone trainable parameters: 0
- `embeddings.npz`: all 1,641 normalized frozen embeddings with image IDs and split names

Test labels were not loaded until test predictions were produced.
"""

    prediction_fields = [
        "image_id",
        "file",
        "split",
        "individual_id",
        "organ_category",
        "predicted_species",
        "confidence",
        "top5",
        "ground_truth_species",
        "correct",
        "correct_top5",
        "status",
        "error",
    ]
    manifest_fields = [
        "image_id",
        "file",
        "split",
        "individual_id",
        "organ_category",
        "ground_truth_species",
    ]

    # Persist the label-free representation so exploratory clustering and
    # nearest-neighbor analyses never need to run the backbone again.
    embedding_output = io.BytesIO()
    np.savez_compressed(
        embedding_output,
        image_ids=np.asarray(full_ids),
        embeddings=full_x.astype(np.float32, copy=False),
        splits=np.asarray([metadata_by_id[image_id]["split"] for image_id in full_ids]),
        model=np.asarray(model_name),
        normalized=np.asarray(True),
    )

    files = {
        "README.md": readme.encode("utf-8"),
        "embeddings.npz": embedding_output.getvalue(),
        "summary.json": json.dumps(summary, indent=2).encode("utf-8"),
        "split_groups.json": json.dumps(split_groups, indent=2).encode("utf-8"),
        "split_manifest.csv": _csv_bytes(output_manifest, manifest_fields),
        "predictions.csv": _csv_bytes(prediction_rows, prediction_fields),
        "per_species.csv": _csv_bytes(
            per_species_rows,
            [
                "species",
                "total",
                "top1_correct",
                "top1_accuracy",
                "top5_correct",
                "top5_accuracy",
            ],
        ),
        "per_organ.csv": _csv_bytes(
            per_organ_rows,
            [
                "organ_category",
                "total",
                "top1_correct",
                "top1_accuracy",
                "top5_correct",
                "top5_accuracy",
            ],
        ),
        "error_pairs.csv": _csv_bytes(
            error_rows,
            ["ground_truth_species", "predicted_species", "count"],
        ),
        "validation_grid.csv": _csv_bytes(
            validation_grid,
            [
                "C",
                "top1_accuracy",
                "macro_accuracy",
                "top5_accuracy",
                "iterations",
            ],
        ),
        "run_status.json": json.dumps(
            {
                "small_test": small_status,
                "pilot": pilot_status,
                "full": full_status,
            },
            indent=2,
        ).encode("utf-8"),
    }
    return _package(files)


@app.local_entrypoint()
def main(backbone: str = "dinov3", output: str = "") -> None:
    if not output:
        output = {
            "dinov3": "outputs/dinov3_strict_linear_probe",
            "bioclip": "outputs/bioclip25_strict_linear_probe",
            "efficientnet_b0": "outputs/efficientnet_b0_strict_embeddings",
        }[backbone]
    payload = run_benchmark.remote(backbone)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(output_path)
    summary = json.loads((output_path / "summary.json").read_text())
    print(json.dumps(summary, indent=2))
