"""Extract full-corpus frozen embeddings for DINOv3, BioCLIP, and EfficientNet-B0."""

from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import modal


DRIVE_FILE_ID = "18U0D61HWmdFH9kyPSuog6mD6zKgfugTc"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
BIOCLIP_MODEL = "hf-hub:imageomics/bioclip-2.5-vith14"
EFFICIENTNET_MODEL = "torchvision/efficientnet_b0-imagenet1k-v1"
SEED = 42

runtime = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gdown==5.2.0",
        "huggingface_hub==0.36.0",
        "numpy==2.2.6",
        "open_clip_torch==3.2.0",
        "pillow==11.3.0",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers==4.57.1",
    )
)
app = modal.App("bioimages-full-multilabel-embeddings", image=runtime)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    sink = io.StringIO()
    writer = csv.DictWriter(sink, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    return sink.getvalue().encode()


def _package(files: dict[str, bytes]) -> bytes:
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name); info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return sink.getvalue()


@app.function(gpu="L4", timeout=2 * 60 * 60, cpu=8.0, memory=32768, secrets=[hf_secret])
def extract_all() -> bytes:
    import gc
    import os
    import platform
    import time
    import zipfile

    import gdown
    import numpy as np
    import open_clip
    import torch
    from PIL import Image, ImageFile
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    from transformers import AutoImageProcessor, AutoModel

    started = time.perf_counter()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    torch.manual_seed(SEED); np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    zip_path = "/tmp/bioimages.zip"
    root = Path("/tmp/bioimages")
    if not gdown.download(DRIVE_URL, zip_path, quiet=False, fuzzy=True):
        raise RuntimeError("Google Drive download failed")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    corpus_path = root / "bioimages-ne-trees" / "corpus.json"
    image_dir = root / "bioimages-ne-trees" / "images"
    corpus = json.loads(corpus_path.read_text())
    rows = []
    for scene in sorted(corpus["scenes"], key=lambda s: s["scientific_name"]):
        for image in scene["images"]:
            rows.append({
                "image_id": image["image_id"],
                "file": image["file"],
                "path": str(image_dir / image["file"]),
                "species": scene["scientific_name"],
                "individual_id": image["individual_id"],
                "organ_tag": image["organ_tag"],
                "organ_category": image["organ_category"],
                "subview": image["subview"],
            })
    if len(rows) != 1899 or len({r["image_id"] for r in rows}) != 1899:
        raise RuntimeError("Expected 1,899 unique images")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing")
    payloads: dict[str, bytes] = {}
    summaries = []

    def run_model(key: str):
        model_started = time.perf_counter()
        if key == "dinov3":
            processor = AutoImageProcessor.from_pretrained(DINO_MODEL, token=hf_token)
            model = AutoModel.from_pretrained(DINO_MODEL, token=hf_token, torch_dtype=torch.bfloat16).eval().to("cuda")
            preprocess = None; batch_size = 64; model_name = DINO_MODEL
        elif key == "bioclip":
            model, _, preprocess = open_clip.create_model_and_transforms(BIOCLIP_MODEL)
            model = model.eval().to("cuda"); processor = None; batch_size = 32; model_name = BIOCLIP_MODEL
        else:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights).eval().to("cuda")
            preprocess = weights.transforms(); processor = None; batch_size = 128; model_name = EFFICIENTNET_MODEL
        for parameter in model.parameters(): parameter.requires_grad_(False)
        load_seconds = time.perf_counter() - model_started

        ids, features, failures = [], [], []
        infer_started = time.perf_counter(); gpu_seconds = 0.0
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            pictures, valid = [], []
            for row in batch_rows:
                try:
                    with Image.open(row["path"]) as opened:
                        rgb = opened.convert("RGB")
                        pictures.append(preprocess(rgb) if preprocess is not None else rgb.copy())
                    valid.append(row)
                except Exception as exc:
                    failures.append({"image_id": row["image_id"], "error": repr(exc)})
            if not pictures: continue
            inputs = processor(images=pictures, return_tensors="pt") if key == "dinov3" else torch.stack(pictures)
            if key == "dinov3": inputs = {k: v.to("cuda", non_blocking=True) for k, v in inputs.items()}
            else: inputs = inputs.to("cuda", non_blocking=True)
            torch.cuda.synchronize(); gpu_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if key == "dinov3": pooled = model(**inputs).pooler_output.float()
                elif key == "bioclip": pooled = model.encode_image(inputs).float()
                else: pooled = torch.flatten(model.avgpool(model.features(inputs)), 1).float()
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            torch.cuda.synchronize(); gpu_seconds += time.perf_counter() - gpu_started
            for row, feature in zip(valid, pooled.cpu().numpy()):
                ids.append(row["image_id"]); features.append(feature)
        matrix = np.stack(features).astype(np.float32, copy=False)
        if len(ids) != 1899 or matrix.shape[0] != 1899 or failures:
            raise RuntimeError(f"{key} incomplete: ids={len(ids)} shape={matrix.shape} failures={failures[:3]}")
        sink = io.BytesIO()
        np.savez_compressed(sink, image_ids=np.asarray(ids), embeddings=matrix, model=np.asarray(model_name), normalized=np.asarray(True))
        summary = {
            "model_key": key, "model": model_name, "images": len(ids), "dimensions": matrix.shape[1],
            "model_load_seconds": load_seconds, "inference_wall_seconds": time.perf_counter() - infer_started,
            "gpu_seconds": gpu_seconds, "batch_size": batch_size,
        }
        del model, processor, preprocess, matrix, features, pictures, inputs, pooled
        gc.collect(); torch.cuda.empty_cache()
        return sink.getvalue(), summary

    for key in ("dinov3", "bioclip", "efficientnet_b0"):
        payload, summary = run_model(key)
        payloads[f"{key}.npz"] = payload; summaries.append(summary)

    manifest_fields = ["image_id", "file", "species", "individual_id", "organ_tag", "organ_category", "subview"]
    payloads["manifest.csv"] = _csv_bytes([{k: r[k] for k in manifest_fields} for r in rows], manifest_fields)
    payloads["summary.json"] = json.dumps({
        "images": 1899, "models": summaries, "seed": SEED,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "python": platform.python_version(), "total_seconds": time.perf_counter() - started,
    }, indent=2).encode()
    return _package(payloads)


@app.local_entrypoint()
def main(output: str = "botanical-computer-vision/outputs/full_multilabel_embeddings") -> None:
    payload = extract_all.remote()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(out)
    print((out / "summary.json").read_text())
