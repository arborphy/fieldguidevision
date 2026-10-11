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
def run_experiment(
    smoke: bool = False,
    confirmation: bool = False,
    rebuild_cache: bool = False,
    commit_modal_cache: bool = True,
) -> bytes:
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
    from dual_encoder_fusion.experiment import (
        TrainConfig,
        make_nested_group_splits,
        run_training,
    )

    started = time.perf_counter()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the requested Modal L4 container")

    def load_or_create_cache() -> dict[str, object]:
        Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
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
            # BioCLIP 2.5 ViT-H uses substantially more activation memory than
            # EfficientNet. Four images leaves headroom on a 24 GiB L4 while
            # keeping extraction deterministic and fully batched.
            ImageDataset(bio_preprocess), batch_size=4, shuffle=False,
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
        # The raw function is also reused by the temporary Cloud Run GPU
        # service.  Only a real Modal invocation can commit this volume.
        if commit_modal_cache:
            cache_volume.commit()
        shutil.rmtree(data_root, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        return cache

    cache = load_or_create_cache()
    if smoke and confirmation:
        raise ValueError("smoke and confirmation protocols are mutually exclusive")

    manifest = pd.read_csv("/root/individual_disjoint.csv")
    manifest = manifest.set_index("image_id").loc[cache["image_ids"]].reset_index()
    if confirmation:
        modes = [
            "bioclip_only",
            "bio_queries_eff",
            "eff_queries_bio",
            "bidirectional_final_only",
        ]
        seeds = [17, 42]
        config = TrainConfig()
        generated_splits = make_nested_group_splits(
            manifest.species.tolist(),
            manifest.individual_id.tolist(),
            folds=3,
            seed=20261012,
        )
        split_definitions = list(enumerate(generated_splits))
    elif smoke:
        modes = ["bio_queries_eff", "eff_queries_bio", "bidirectional"]
        seeds = [42]
        config = TrainConfig(width=64, heads=4, blocks=1, epochs=3, patience=3, batch_size=32)
        split_definitions = [(-1, np.asarray(cache["splits"]))]
    else:
        modes = [
            "efficientnet_only", "bioclip_only", "concat",
            "bio_queries_eff", "eff_queries_bio", "bidirectional",
            "bidirectional_final_only",
        ]
        seeds = [17, 42, 73]
        config = TrainConfig()
        split_definitions = [(-1, np.asarray(cache["splits"]))]

    results = []
    prediction_rows = []
    device = torch.device("cuda")
    split_audit = []
    for outer_fold, split_labels in split_definitions:
        fold_cache = dict(cache)
        fold_cache["splits"] = split_labels.tolist()
        split_audit.append({
            "outer_fold": outer_fold,
            **{
                f"{split_name}_images": int(np.sum(split_labels == split_name))
                for split_name in ("train", "validation", "test")
            },
            **{
                f"{split_name}_individuals": int(
                    manifest.loc[split_labels == split_name, "individual_id"].nunique()
                )
                for split_name in ("train", "validation", "test")
            },
        })
        for mode in modes:
            for seed in seeds:
                print(
                    f"Training {mode}, fold={outer_fold}, seed={seed}", flush=True
                )
                result, predictions, _ = run_training(
                    fold_cache, mode, seed, config, device
                )
                result["outer_fold"] = outer_fold
                results.append(result)
                for row in predictions:
                    index = row.pop("index")
                    row.update({
                        "mode": mode,
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "image_id": cache["image_ids"][index],
                        "individual_id": manifest.iloc[index].individual_id,
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
        "species_anchor_accuracy", "species_anchor_macro_f1",
        "species_changed_from_anchor", "species_fusion_gate_mean",
        "organ_accuracy", "organ_macro_f1",
    ]
    summary = results_frame.groupby("mode")[metric_columns + ["trainable_parameters"]].agg(
        ["mean", "std"]
    ).reset_index()
    summary.columns = ["_".join(part for part in column if part) if isinstance(column, tuple) else column for column in summary.columns]

    predictions_frame = pd.DataFrame(prediction_rows)
    paired_rows = []
    candidate = "bidirectional_final_only" if confirmation else "bidirectional"
    competitors = (
        ("bio_queries_eff", "eff_queries_bio", "bioclip_only")
        if confirmation else ("bio_queries_eff", "eff_queries_bio")
    )
    if candidate in set(results_frame["mode"]):
        pair_index = ["outer_fold", "seed"] if confirmation else ["seed"]
        for competitor in competitors:
            paired = results_frame[
                results_frame["mode"].isin([candidate, competitor])
            ].pivot(index=pair_index, columns="mode", values="species_macro_f1").dropna()
            for pair_key, row in paired.iterrows():
                outer_fold, seed = (
                    pair_key if confirmation else (-1, pair_key)
                )
                paired_rows.append({
                    "competitor": competitor,
                    "outer_fold": int(outer_fold),
                    "seed": int(seed),
                    "candidate_species_macro_f1": float(row[candidate]),
                    "competitor_species_macro_f1": float(row[competitor]),
                    "paired_delta": float(row[candidate] - row[competitor]),
                })
    paired_frame = pd.DataFrame(paired_rows)

    oof_per_seed = pd.DataFrame()
    oof_summary = pd.DataFrame()
    bootstrap_frame = pd.DataFrame()
    delta_figure_sink = None
    if confirmation:
        from sklearn.metrics import accuracy_score, f1_score

        oof_rows = []
        for (mode, seed), frame in predictions_frame.groupby(["mode", "seed"]):
            if len(frame) != len(cache["image_ids"]) or frame.image_id.nunique() != len(frame):
                raise RuntimeError(
                    f"Incomplete out-of-fold predictions for {mode}, seed={seed}"
                )
            truth = frame.species_target.to_numpy()
            predicted = frame.species_prediction.to_numpy()
            oof_rows.append({
                "mode": mode,
                "seed": int(seed),
                "species_accuracy": float(accuracy_score(truth, predicted)),
                "species_macro_f1": float(f1_score(
                    truth,
                    predicted,
                    labels=np.arange(len(cache["species_names"])),
                    average="macro",
                    zero_division=0,
                )),
            })
        oof_per_seed = pd.DataFrame(oof_rows)
        oof_summary = oof_per_seed.groupby("mode")[[
            "species_accuracy", "species_macro_f1"
        ]].agg(["mean", "std"]).reset_index()
        oof_summary.columns = [
            "_".join(part for part in column if part)
            if isinstance(column, tuple) else column
            for column in oof_summary.columns
        ]

        wide = predictions_frame.pivot(
            index=["image_id", "individual_id", "species_target"],
            columns=["mode", "seed"],
            values="species_prediction",
        ).sort_index()
        if wide.isna().any().any() or len(wide) != len(cache["image_ids"]):
            raise RuntimeError("Confirmation prediction matrix is incomplete")
        truth = wide.index.get_level_values("species_target").to_numpy(dtype=int)
        individual_ids = wide.index.get_level_values("individual_id").to_numpy(dtype=str)
        labels = np.arange(len(cache["species_names"]))

        positions_by_species_group = []
        for species_target in labels:
            species_positions = np.flatnonzero(truth == species_target)
            species_groups = np.unique(individual_ids[species_positions])
            positions_by_species_group.append([
                species_positions[individual_ids[species_positions] == group]
                for group in species_groups
            ])
        bootstrap_rng = np.random.default_rng(20261012)
        bootstrap_indices = []
        for _ in range(2000):
            sampled_positions = []
            for group_positions in positions_by_species_group:
                choices = bootstrap_rng.integers(0, len(group_positions), len(group_positions))
                sampled_positions.extend(group_positions[index] for index in choices)
            bootstrap_indices.append(np.concatenate(sampled_positions))

        def macro_f1(target: np.ndarray, predicted: np.ndarray) -> float:
            return float(f1_score(
                target,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            ))

        bootstrap_rows = []
        for competitor in competitors:
            candidate_predictions = {
                seed: wide[(candidate, seed)].to_numpy(dtype=int) for seed in seeds
            }
            competitor_predictions = {
                seed: wide[(competitor, seed)].to_numpy(dtype=int) for seed in seeds
            }
            observed = float(np.mean([
                macro_f1(truth, candidate_predictions[seed])
                - macro_f1(truth, competitor_predictions[seed])
                for seed in seeds
            ]))
            samples = np.asarray([
                np.mean([
                    macro_f1(truth[indices], candidate_predictions[seed][indices])
                    - macro_f1(truth[indices], competitor_predictions[seed][indices])
                    for seed in seeds
                ])
                for indices in bootstrap_indices
            ])
            bootstrap_rows.append({
                "competitor": competitor,
                "observed_oof_macro_f1_delta": observed,
                "ci_95_lower": float(np.quantile(samples, 0.025)),
                "ci_95_upper": float(np.quantile(samples, 0.975)),
                "bootstrap_probability_positive": float(np.mean(samples > 0)),
                "bootstrap_iterations": len(samples),
            })
        bootstrap_frame = pd.DataFrame(bootstrap_rows)

        delta_fig, delta_axis = plt.subplots(figsize=(8.5, 4.6))
        delta_positions = np.arange(len(bootstrap_frame))
        observed = bootstrap_frame.observed_oof_macro_f1_delta.to_numpy()
        lower = bootstrap_frame.ci_95_lower.to_numpy()
        upper = bootstrap_frame.ci_95_upper.to_numpy()
        delta_axis.errorbar(
            observed,
            delta_positions,
            xerr=np.vstack([observed - lower, upper - observed]),
            fmt="o",
            color="#246b45",
            ecolor="#466a55",
            capsize=5,
        )
        delta_axis.axvline(0, color="#9b3d3d", linestyle="--", linewidth=1.5)
        delta_axis.set_yticks(delta_positions, bootstrap_frame.competitor)
        delta_axis.set_xlabel("Final-only bidirectional minus comparator macro F1")
        delta_axis.set_title("Individual-bootstrap 95% intervals on out-of-fold predictions")
        delta_axis.grid(axis="x", alpha=0.25)
        delta_fig.tight_layout()
        delta_figure_sink = io.BytesIO()
        delta_fig.savefig(delta_figure_sink, format="png", dpi=180)
        plt.close(delta_fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    plot_summary = oof_summary if confirmation else summary
    ordering = plot_summary.sort_values("species_macro_f1_mean", ascending=False)
    positions = np.arange(len(ordering))
    axis.bar(
        positions, ordering.species_macro_f1_mean,
        yerr=ordering.species_macro_f1_std.fillna(0),
        color=["#246b45" if value == candidate else "#9bb89f" for value in ordering["mode"]],
        capsize=4,
    )
    axis.set_xticks(positions, ordering["mode"], rotation=25, ha="right")
    axis.set_ylabel("Test species macro F1")
    axis.set_title(
        "Nested group confirmation: out-of-fold performance"
        if confirmation else "Controlled EfficientNet + BioCLIP fusion ablation"
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_sink = io.BytesIO(); fig.savefig(figure_sink, format="png", dpi=180); plt.close(fig)

    def dataframe_bytes(frame: pd.DataFrame) -> bytes:
        return frame.to_csv(index=False).encode("utf-8")

    results_json = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
    metadata = {
        "smoke": smoke,
        "confirmation": confirmation,
        "protocol": (
            "three_fold_nested_individual_confirmation"
            if confirmation else ("smoke" if smoke else "fixed_split_ablation")
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "images": len(cache["image_ids"]),
        "species_classes": len(cache["species_names"]),
        "organ_classes": len(cache["organ_names"]),
        "organ_names": cache["organ_names"],
        "species_is_primary": True,
        "architecture_version": "v2_bioclip_anchored_gated_residual",
        "selection_metric": "validation species macro F1",
        "test_used_for_selection": False,
        "modes": modes,
        "seeds": seeds,
        "outer_split_seed": 20261012 if confirmation else None,
        "split_audit": split_audit,
    }
    verdict = {"status": "smoke_only"}
    if confirmation:
        interval_lookup = bootstrap_frame.set_index("competitor").to_dict("index")
        one_way_competitors = ("bio_queries_eff", "eff_queries_bio")
        bidirectionality_supported = all(
            interval_lookup[name]["ci_95_lower"] > 0
            for name in one_way_competitors
        )
        fusion_beats_bioclip = interval_lookup["bioclip_only"]["ci_95_lower"] > 0
        verdict = {
            "status": (
                "bidirectionality_supported"
                if bidirectionality_supported else "not_demonstrated"
            ),
            "criterion": (
                "final-map-only bidirectional fusion must have a positive lower "
                "bound of the 95% species-stratified individual bootstrap interval "
                "against both one-way directions"
            ),
            "bidirectionality_supported": bidirectionality_supported,
            "fusion_beats_bioclip": fusion_beats_bioclip,
            "comparisons": interval_lookup,
        }
    elif not smoke:
        deltas = paired_frame.groupby("competitor").paired_delta.mean().to_dict()
        verdict = {
            "status": "pass" if deltas and all(value > 0 for value in deltas.values()) else "not_demonstrated",
            "criterion": "mean bidirectional species macro F1 exceeds each one-way direction over matched seeds",
            "mean_paired_deltas": deltas,
        }
    report_lines = [
        (
            "# Final-map bidirectional confirmation"
            if confirmation else "# Dual-encoder cross-attention experiment"
        ),
        "",
        f"- Status: **{verdict['status']}**",
        f"- Images: {metadata['images']} (strictly individual-disjoint within every fold)",
        f"- Primary selection metric: {metadata['selection_metric']}",
        "- Organ supervision: separate auxiliary head; weight 0.25 relative to species",
        "- Species fusion: BioCLIP-anchored gated residual after real cross-attention",
        "- Encoders: frozen for this controlled phase; projections, attention blocks, and heads are trained",
        "",
        "## Interpretation rule",
        "",
        (
            "The locked final-map-only bidirectional model is supported over one-way "
            "fusion only when the lower bound of its 95% species-stratified, "
            "individual-level bootstrap interval is above zero against both one-way "
            "directions. Beating BioCLIP alone is reported as a separate claim."
            if confirmation else
            "Bidirectional fusion is considered supported only if its mean test species "
            "macro F1 is higher than both one-way variants under the same seeds and "
            "training budget. This run does not claim support when that criterion fails."
        ),
        "",
        (
            "See `oof_summary.csv`, `bootstrap_intervals.csv`, "
            "`species_macro_f1.png`, and `paired_oof_deltas.png`."
            if confirmation else
            "See `summary.csv`, `paired_deltas.csv`, and `species_macro_f1.png` for "
            "the numerical and visual comparison."
        ),
    ]
    files = {
        "run_metadata.json": json.dumps(metadata, indent=2).encode(),
        "verdict.json": json.dumps(verdict, indent=2).encode(),
        "results.csv": dataframe_bytes(results_frame),
        "results.json": results_json,
        "summary.csv": dataframe_bytes(summary),
        "paired_deltas.csv": dataframe_bytes(paired_frame),
        "predictions.csv": dataframe_bytes(predictions_frame),
        "species_macro_f1.png": figure_sink.getvalue(),
        "REPORT.md": "\n".join(report_lines).encode("utf-8"),
    }
    if confirmation:
        files.update({
            "split_audit.csv": dataframe_bytes(pd.DataFrame(split_audit)),
            "oof_per_seed.csv": dataframe_bytes(oof_per_seed),
            "oof_summary.csv": dataframe_bytes(oof_summary),
            "bootstrap_intervals.csv": dataframe_bytes(bootstrap_frame),
            "paired_oof_deltas.png": delta_figure_sink.getvalue(),
        })
    return _archive(files)


@app.local_entrypoint()
def main(
    smoke: bool = False,
    confirmation: bool = False,
    rebuild_cache: bool = False,
    output: str = "botanical-computer-vision/dual_encoder_fusion/results",
) -> None:
    payload = run_experiment.remote(
        smoke=smoke,
        confirmation=confirmation,
        rebuild_cache=rebuild_cache,
    )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(destination)
    print((destination / "REPORT.md").read_text(encoding="utf-8"))
