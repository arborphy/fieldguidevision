# /// script
# requires-python = ">=3.11"
# dependencies = ["modal==1.2.1"]
# ///
"""Extract frozen spatial tokens and run controlled fusion ablations on Modal.

Examples:
  uv run modal run dual_encoder_fusion/modal_run_experiment.py --smoke
  uv run modal run dual_encoder_fusion/modal_run_experiment.py
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "organ_benchmark" / "split_manifests" / "individual_disjoint.csv"
DATASET_FILE_ID = "18U0D61HWmdFH9kyPSuog6mD6zKgfugTc"
BIOCLIP_MODEL = "hf-hub:imageomics/bioclip-2.5-vith14"
CACHE_PATH = "/cache/bioclip25_effb0_spatial_tokens_v1.pt"

runtime = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gdown==5.2.0",
        "matplotlib==3.10.1",
        "numpy==2.2.6",
        "open_clip_torch==3.2.0",
        "pandas==2.2.3",
        "pillow==11.3.0",
        "scikit-learn==1.7.2",
        "torch==2.7.1",
        "torchvision==0.22.1",
    )
    .add_local_dir(str(HERE), remote_path="/root/dual_encoder_fusion", copy=True)
    .add_local_file(str(MANIFEST), remote_path="/root/individual_disjoint.csv", copy=True)
)
app = modal.App("bioimages-dual-encoder-cross-attention", image=runtime)
cache_volume = modal.Volume.from_name("bioimages-dual-encoder-cache", create_if_missing=True)


def _archive(files: dict[str, bytes]) -> bytes:
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w:gz") as bundle:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    return sink.getvalue()


@app.function(
    gpu="L4",
    cpu=8.0,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={"/cache": cache_volume},
)
def run_experiment(smoke: bool = False, rebuild_cache: bool = False) -> bytes:
    import csv
    import gc
    import hashlib
    import json
    import math
    import os
    import platform
    import shutil
    import sys
    import time
    import zipfile

    import gdown
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import open_clip
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageFile
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    sys.path.insert(0, "/root")
    from dual_encoder_fusion.experiment import TrainConfig, run_training

    started = time.perf_counter()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the requested Modal L4 container")

    def load_or_create_cache() -> dict[str, object]:
        if os.path.exists(CACHE_PATH) and not rebuild_cache:
            value = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
            if value.get("cache_version") == 1:
                print(f"Using cached tokens from {CACHE_PATH}", flush=True)
                return value

        data_root = Path("/tmp/bioimages")
        zip_path = Path("/tmp/bioimages.zip")
        url = f"https://drive.google.com/uc?id={DATASET_FILE_ID}"
        if not gdown.download(url, str(zip_path), quiet=False, fuzzy=True):
            raise RuntimeError("BioImages Google Drive download failed")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(data_root)
        image_dir = data_root / "bioimages-ne-trees" / "images"
        manifest = pd.read_csv("/root/individual_disjoint.csv")
        if len(manifest) != 1641 or manifest.image_id.nunique() != 1641:
            raise RuntimeError("Expected the committed 1,641-image strict manifest")
        missing = [name for name in manifest.file if not (image_dir / name).exists()]
        if missing:
            raise RuntimeError(f"Missing source images: {missing[:5]}")

        class ImageDataset(Dataset):
            def __init__(self, transform):
                self.transform = transform

            def __len__(self):
                return len(manifest)

            def __getitem__(self, index):
                with Image.open(image_dir / manifest.iloc[index].file) as opened:
                    return index, self.transform(opened.convert("RGB"))

        # EfficientNet 14x14 mid-map (pooled to 7x7) plus the final 7x7 map.
        eff_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        eff_model = efficientnet_b0(weights=eff_weights).eval().to("cuda")
        for parameter in eff_model.parameters():
            parameter.requires_grad_(False)
        eff_loader = DataLoader(
            ImageDataset(eff_weights.transforms()), batch_size=96, shuffle=False,
            num_workers=8, pin_memory=True, persistent_workers=True,
        )
        eff_mid = torch.empty((len(manifest), 49, 112), dtype=torch.float16)
        eff_final = torch.empty((len(manifest), 49, 1280), dtype=torch.float16)
        with torch.inference_mode():
            for indices, images in eff_loader:
                x = images.to("cuda", non_blocking=True)
                middle = None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for layer_index, layer in enumerate(eff_model.features):
                        x = layer(x)
                        if layer_index == 5:
                            middle = x
                    assert middle is not None
                    middle = F.adaptive_avg_pool2d(middle, (7, 7))
                eff_mid[indices] = middle.flatten(2).transpose(1, 2).cpu().half()
                eff_final[indices] = x.flatten(2).transpose(1, 2).cpu().half()
        del eff_model, eff_loader, x, middle
        gc.collect(); torch.cuda.empty_cache()

        # BioCLIP final 16x16 patch tokens are pooled to 8x8. The CLS token is
        # retained separately, before CLIP's final projection.
        bio_model, _, bio_preprocess = open_clip.create_model_and_transforms(BIOCLIP_MODEL)
        bio_model = bio_model.eval().to("cuda")
        for parameter in bio_model.parameters():
            parameter.requires_grad_(False)
        bio_loader = DataLoader(
            ImageDataset(bio_preprocess), batch_size=20, shuffle=False,
            num_workers=8, pin_memory=True, persistent_workers=True,
        )
        bio_spatial = None
        bio_global = None
        with torch.inference_mode():
            for indices, images in bio_loader:
                images = images.to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = bio_model.forward_intermediates(
                        image=images,
                        image_indices=1,
                        normalize_intermediates=True,
                        image_output_fmt="NLC",
                        image_output_extra_tokens=True,
                    )
                    spatial = output["image_intermediates"][-1]
                    prefix = output["image_intermediates_prefix"][-1][:, 0]
                    grid = math.isqrt(spatial.shape[1])
                    if grid * grid != spatial.shape[1]:
                        raise RuntimeError(f"BioCLIP patch count is not square: {spatial.shape}")
                    spatial = spatial.transpose(1, 2).reshape(
                        spatial.shape[0], spatial.shape[2], grid, grid
                    )
                    spatial = F.adaptive_avg_pool2d(spatial, (8, 8))
                    spatial = spatial.flatten(2).transpose(1, 2)
                if bio_spatial is None:
                    width = spatial.shape[-1]
                    bio_spatial = torch.empty((len(manifest), 64, width), dtype=torch.float16)
                    bio_global = torch.empty((len(manifest), width), dtype=torch.float16)
                bio_spatial[indices] = spatial.cpu().half()
                bio_global[indices] = prefix.cpu().half()
        assert bio_spatial is not None and bio_global is not None

        species_names = sorted(manifest.species.unique().tolist())
        species_lookup = {name: index for index, name in enumerate(species_names)}
        normalized_organ = manifest.organ_tag.replace({"stem": "twig"})
        organ_names = sorted(name for name in normalized_organ.unique() if name != "unspecified")
        organ_lookup = {name: index for index, name in enumerate(organ_names)}
        cache = {
            "cache_version": 1,
            "model_names": {
                "efficientnet": "torchvision/efficientnet_b0-imagenet1k-v1",
                "bioclip": BIOCLIP_MODEL,
            },
            "image_ids": manifest.image_id.astype(str).tolist(),
            "files": manifest.file.astype(str).tolist(),
            "splits": manifest.split.astype(str).tolist(),
            "species_names": species_names,
            "organ_names": organ_names,
            "species_targets": torch.tensor(
                [species_lookup[value] for value in manifest.species], dtype=torch.long
            ),
            "organ_targets": torch.tensor(
                [organ_lookup.get(value, -1) for value in normalized_organ], dtype=torch.long
            ),
            "eff_mid": eff_mid,
            "eff_final": eff_final,
            "bio_spatial": bio_spatial,
            "bio_global": bio_global,
            "manifest_sha256": hashlib.sha256(
                Path("/root/individual_disjoint.csv").read_bytes()
            ).hexdigest(),
        }
        temporary = CACHE_PATH + ".tmp"
        torch.save(cache, temporary)
        os.replace(temporary, CACHE_PATH)
        cache_volume.commit()
        shutil.rmtree(data_root, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        return cache

    cache = load_or_create_cache()
    if smoke:
        modes = ["bio_queries_eff", "eff_queries_bio", "bidirectional"]
        seeds = [42]
        config = TrainConfig(width=64, heads=4, blocks=1, epochs=3, patience=3, batch_size=32)
    else:
        modes = [
            "efficientnet_only", "bioclip_only", "concat",
            "bio_queries_eff", "eff_queries_bio", "bidirectional",
            "bidirectional_final_only",
        ]
        seeds = [17, 42, 73]
        config = TrainConfig()

    results = []
    prediction_rows = []
    device = torch.device("cuda")
    for mode in modes:
        for seed in seeds:
            print(f"Training {mode}, seed={seed}", flush=True)
            result, predictions, _ = run_training(cache, mode, seed, config, device)
            history = result.pop("history")
            result["history"] = history
            results.append(result)
            for row in predictions:
                index = row.pop("index")
                row.update({
                    "mode": mode,
                    "seed": seed,
                    "image_id": cache["image_ids"][index],
                    "species_truth": cache["species_names"][row["species_target"]],
                    "species_predicted": cache["species_names"][row["species_prediction"]],
                    "species_top5_names": [cache["species_names"][value] for value in row["species_top5"]],
                })
                prediction_rows.append(row)
            torch.cuda.empty_cache()

    flat_results = [{key: value for key, value in row.items() if key not in {"config", "history"}} for row in results]
    results_frame = pd.DataFrame(flat_results)
    metric_columns = [
        "species_accuracy", "species_macro_f1", "species_top5",
        "organ_accuracy", "organ_macro_f1",
    ]
    summary = results_frame.groupby("mode")[metric_columns + ["trainable_parameters"]].agg(
        ["mean", "std"]
    ).reset_index()
    summary.columns = ["_".join(part for part in column if part) if isinstance(column, tuple) else column for column in summary.columns]

    paired_rows = []
    if "bidirectional" in set(results_frame["mode"]):
        for competitor in ("bio_queries_eff", "eff_queries_bio"):
            paired = results_frame[results_frame["mode"].isin(["bidirectional", competitor])].pivot(
                index="seed", columns="mode", values="species_macro_f1"
            ).dropna()
            for seed, row in paired.iterrows():
                paired_rows.append({
                    "competitor": competitor,
                    "seed": int(seed),
                    "bidirectional_species_macro_f1": float(row["bidirectional"]),
                    "competitor_species_macro_f1": float(row[competitor]),
                    "paired_delta": float(row["bidirectional"] - row[competitor]),
                })
    paired_frame = pd.DataFrame(paired_rows)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    ordering = summary.sort_values("species_macro_f1_mean", ascending=False)
    positions = np.arange(len(ordering))
    axis.bar(
        positions, ordering.species_macro_f1_mean,
        yerr=ordering.species_macro_f1_std.fillna(0),
        color=["#246b45" if value == "bidirectional" else "#9bb89f" for value in ordering["mode"]],
        capsize=4,
    )
    axis.set_xticks(positions, ordering["mode"], rotation=25, ha="right")
    axis.set_ylabel("Test species macro F1")
    axis.set_title("Controlled EfficientNet + BioCLIP fusion ablation")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_sink = io.BytesIO(); fig.savefig(figure_sink, format="png", dpi=180); plt.close(fig)

    def dataframe_bytes(frame: pd.DataFrame) -> bytes:
        return frame.to_csv(index=False).encode("utf-8")

    results_json = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
    metadata = {
        "smoke": smoke,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "images": len(cache["image_ids"]),
        "species_classes": len(cache["species_names"]),
        "organ_classes": len(cache["organ_names"]),
        "organ_names": cache["organ_names"],
        "species_is_primary": True,
        "selection_metric": "validation species macro F1",
        "test_used_for_selection": False,
        "modes": modes,
        "seeds": seeds,
    }
    verdict = {"status": "smoke_only"}
    if not smoke:
        deltas = paired_frame.groupby("competitor").paired_delta.mean().to_dict()
        verdict = {
            "status": "pass" if deltas and all(value > 0 for value in deltas.values()) else "not_demonstrated",
            "criterion": "mean bidirectional species macro F1 exceeds each one-way direction over matched seeds",
            "mean_paired_deltas": deltas,
        }
    report_lines = [
        "# Dual-encoder cross-attention experiment",
        "",
        f"- Status: **{verdict['status']}**",
        f"- Images: {metadata['images']} (individual-disjoint train/validation/test)",
        f"- Primary selection metric: {metadata['selection_metric']}",
        "- Organ supervision: separate auxiliary head; weight 0.25 relative to species",
        "- Encoders: frozen for this controlled phase; projections, attention blocks, and heads are trained",
        "",
        "## Interpretation rule",
        "",
        "Bidirectional fusion is considered supported only if its mean test species macro F1 is higher than both one-way variants under the same seeds and training budget. This run does not claim support when that criterion fails.",
        "",
        "See `summary.csv`, `paired_deltas.csv`, and `species_macro_f1.png` for the numerical and visual comparison.",
    ]
    files = {
        "run_metadata.json": json.dumps(metadata, indent=2).encode(),
        "verdict.json": json.dumps(verdict, indent=2).encode(),
        "results.csv": dataframe_bytes(results_frame),
        "results.json": results_json,
        "summary.csv": dataframe_bytes(summary),
        "paired_deltas.csv": dataframe_bytes(paired_frame),
        "predictions.csv": dataframe_bytes(pd.DataFrame(prediction_rows)),
        "species_macro_f1.png": figure_sink.getvalue(),
        "REPORT.md": "\n".join(report_lines).encode("utf-8"),
    }
    return _archive(files)


@app.local_entrypoint()
def main(
    smoke: bool = False,
    rebuild_cache: bool = False,
    output: str = "botanical-computer-vision/dual_encoder_fusion/results",
) -> None:
    payload = run_experiment.remote(smoke=smoke, rebuild_cache=rebuild_cache)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(destination)
    print((destination / "REPORT.md").read_text(encoding="utf-8"))
