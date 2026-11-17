# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Evaluation for a conditioned expert.

Metrics are only ever written from an actual checkpoint + actual predictions.
There is no placeholder/fabricated metric path: :func:`evaluate_run` raises if
the checkpoint or dataset is missing.

The abstain threshold is selected on the validation split and then applied
unchanged to test. ``metrics.json`` records both the whole selection curve and
the final test metrics (macro P/R/F1, per-class recall, confusion matrix,
coverage/abstain rate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contracts import TargetSpec
from .dataset import IMAGE_EXTS
from .device import resolve_device
from .metrics import full_report, select_abstain_threshold


@dataclass
class SplitData:
    paths: list[Path]
    labels: list[int]


def collect_split(dataset_root: str | Path, split: str, target: TargetSpec) -> SplitData:
    root = Path(dataset_root) / split
    if not root.is_dir():
        raise FileNotFoundError(f"missing split directory: {root}")
    paths: list[Path] = []
    labels: list[int] = []
    for class_index, class_name in enumerate(target.classes):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS and path.is_file():
                paths.append(path)
                labels.append(class_index)
    if not paths:
        raise ValueError(f"no images in split {split!r} under {root}")
    return SplitData(paths=paths, labels=labels)


def load_classifier(checkpoint: str | Path, device: str = "auto", *, imgsz: int = 224):
    """Load a YOLO26 classification checkpoint as a torch module."""

    import torch
    from ultralytics import YOLO

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    model = YOLO(str(path)).model
    model.eval()
    model.to(resolve_device(device))
    return model


def classifier_names(model: Any) -> list[str]:
    """Return the model's index->class-name order (Ultralytics sorts folders)."""

    names = getattr(model, "names", None)
    if isinstance(names, dict):
        try:
            keys = sorted(names, key=lambda value: int(value))
        except (TypeError, ValueError):
            keys = sorted(names)
        return [str(names[index]) for index in keys]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    raise ValueError("model does not expose a class-name mapping")


def reorder_probabilities(
    probabilities: Any, model_names: Sequence[str], target_classes: Sequence[str]
) -> np.ndarray:
    """Reorder model-order columns into the target's canonical class order.

    Ultralytics builds class folders alphabetically, so ``model.names`` is not
    necessarily the config order. Metrics must use the target order.
    """

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be [n, classes]")
    if set(model_names) != set(target_classes):
        raise ValueError(
            f"model classes {list(model_names)} do not match target classes {list(target_classes)}"
        )
    if len(model_names) != len(target_classes):
        raise ValueError("model and target class counts differ")
    index = {name: position for position, name in enumerate(model_names)}
    order = [index[name] for name in target_classes]
    return probs[:, order]


def probabilities_from_output(output: Any) -> np.ndarray:
    """Return probabilities without double-applying softmax.

    Ultralytics classification inference already applies softmax in its
    ``Classify`` head. Some exported/raw models return logits. A probability
    matrix is non-negative and each row sums to 1; logits are softmaxed once.
    """

    values = np.asarray(output, dtype=np.float64)
    if values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2:
        raise ValueError(f"unexpected classifier output shape: {values.shape}")
    non_negative = bool((values >= -1e-6).all())
    row_sums = values.sum(axis=1)
    looks_like_probabilities = non_negative and bool(
        np.allclose(row_sums, 1.0, atol=1e-3)
    )
    if looks_like_probabilities:
        return np.clip(values, 0.0, None)
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def predict_paths(
    model: Any,
    paths: Sequence[Path],
    *,
    imgsz: int = 224,
    batch_size: int = 32,
    device: str = "auto",
) -> np.ndarray:
    """Return an ``[n, classes]`` softmax matrix for the given image paths."""

    import torch
    from PIL import Image
    from ultralytics.data.augment import classify_transforms

    device = resolve_device(device)
    transform = classify_transforms(imgsz)
    outputs: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            tensors.append(transform(image))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            output = model(batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
        outputs.append(probabilities_from_output(output.detach().cpu().numpy()))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0))


def write_predictions(
    out_path: str | Path,
    paths: Sequence[Path],
    labels: Sequence[int],
    probabilities: np.ndarray,
    target: TargetSpec,
    *,
    confidence_threshold: float,
    margin_threshold: float = 0.0,
) -> None:
    from .metrics import apply_abstention, confidence_and_margin

    predictions = apply_abstention(
        probabilities,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    confidence, margin = confidence_and_margin(probabilities)
    target_path = Path(out_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w") as handle:
        for path, actual, predicted, conf, marg, row in zip(
            paths, labels, predictions, confidence, margin, probabilities
        ):
            handle.write(
                json.dumps(
                    {
                        "path": str(path),
                        "actual": target.index_to_label[int(actual)],
                        "predicted": target.index_to_label[int(predicted)]
                        if predicted >= 0
                        else "abstain",
                        "confidence": float(conf),
                        "margin": float(marg),
                        "probabilities": {
                            name: float(prob) for name, prob in zip(target.classes, row)
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def evaluate_run(
    checkpoint: str | Path,
    dataset_root: str | Path,
    target: TargetSpec,
    out_dir: str | Path,
    *,
    device: str = "auto",
    imgsz: int = 224,
    batch_size: int = 32,
    target_precision: float = 0.95,
    min_coverage: float = 0.30,
    margin_threshold: float = 0.0,
    threshold_grid: Sequence[float] | None = None,
    max_vram_fraction: float | None = None,
    min_free_vram_gb: float | None = None,
    split: str = "both",
) -> dict[str, Any]:
    """Evaluate val+test and write metrics/predictions. Real outputs only.

    ``split`` selects the reported focus (``val``, ``test``, or ``both``);
    validation is always run because the abstain threshold is selected there
    only. Applies the same VRAM cap/preflight as training before loading the
    model.
    """

    if split not in {"val", "test", "both"}:
        raise ValueError("split must be val, test, or both")
    from .device import configure_memory

    memory_kwargs: dict[str, float] = {}
    if max_vram_fraction is not None:
        memory_kwargs["max_vram_fraction"] = max_vram_fraction
    if min_free_vram_gb is not None:
        memory_kwargs["min_free_vram_gb"] = min_free_vram_gb
    memory = configure_memory(device, **memory_kwargs)
    device = memory["device"]
    model = load_classifier(checkpoint, device, imgsz=imgsz)
    model_names = classifier_names(model)
    val = collect_split(dataset_root, "val", target)
    test = collect_split(dataset_root, "test", target)
    val_probs = reorder_probabilities(
        predict_paths(model, val.paths, imgsz=imgsz, batch_size=batch_size, device=device),
        model_names,
        target.classes,
    )
    test_probs = reorder_probabilities(
        predict_paths(model, test.paths, imgsz=imgsz, batch_size=batch_size, device=device),
        model_names,
        target.classes,
    )

    selection = select_abstain_threshold(
        val.labels,
        val_probs,
        target.classes,
        target_precision=target_precision,
        min_coverage=min_coverage,
        grid=threshold_grid,
        margin_threshold=margin_threshold,
    )
    threshold = float(selection["confidence_threshold"])
    test_report = full_report(
        test.labels,
        test_probs,
        target.classes,
        confidence_threshold=threshold,
        margin_threshold=margin_threshold,
    )
    val_report = full_report(
        val.labels,
        val_probs,
        target.classes,
        confidence_threshold=threshold,
        margin_threshold=margin_threshold,
    )

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_predictions(
        output / "predictions_val.jsonl",
        val.paths,
        val.labels,
        val_probs,
        target,
        confidence_threshold=threshold,
        margin_threshold=margin_threshold,
    )
    write_predictions(
        output / "predictions_test.jsonl",
        test.paths,
        test.labels,
        test_probs,
        target,
        confidence_threshold=threshold,
        margin_threshold=margin_threshold,
    )
    dataset_summary: dict[str, Any] = {}
    summary_path = Path(dataset_root) / "counts.json"
    if summary_path.is_file():
        try:
            raw_summary = json.loads(summary_path.read_text())
            dataset_summary = {
                "split_mode": raw_summary.get("split_mode"),
                "prefer_test_unseen": raw_summary.get("prefer_test_unseen"),
                "stage1_overlap_counts": raw_summary.get("stage1_overlap_counts"),
            }
        except json.JSONDecodeError:
            dataset_summary = {}
    metrics = {
        "schema": "arq.anatomy4_specialists.metrics/v1",
        "target": target.to_dict(),
        "checkpoint": str(Path(checkpoint).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "imgsz": imgsz,
        "device": resolve_device(device),
        "class_order": {
            "model_names": model_names,
            "target_classes": list(target.classes),
            "reordered_to_target": True,
            "assertion": "set(model.names) == set(target.classes)",
        },
        "memory": memory,
        "abstain_selection": {
            "selected_on": "val",
            "target_precision": target_precision,
            "min_coverage": min_coverage,
            "confidence_threshold": threshold,
            "margin_threshold": margin_threshold,
            "target_met": selection["target_met"],
            "chosen": {k: v for k, v in selection.items() if k != "curve"},
            "curve": selection["curve"],
        },
        "requested_split": split,
        "splits_evaluated": ["val", "test"],
        "dataset": dataset_summary,
        "val": val_report,
        "test": test_report,
        "n_val": len(val.labels),
        "n_test": len(test.labels),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
