# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.anatomy4_specialists.contracts import LEAF_SIMPLE_VS_COMPOUND
from experiments.anatomy4_specialists.dataset import build_classification_dataset
from experiments.anatomy4_specialists.evaluate import evaluate_run
from experiments.anatomy4_specialists.splits import assign_splits


def _holdout_rows(count: int = 100, unseen: int = 40):
    rows = []
    for index in range(count):
        label = "Simple" if index % 2 == 0 else "Compound"
        rows.append(
            {
                "crop_id": f"c{index}",
                "observation_id": f"obs-{index}",
                "taxon_name": f"Genus{index} species{index}",
                "target_label": label,
                "in_stage1_train_val": index >= unseen,
            }
        )
    return rows


def test_species_and_observation_splits_are_70_15_15() -> None:
    rows = _holdout_rows()
    for mode in ("species", "observation"):
        report = assign_splits(rows, mode=mode, seed="unit")
        counts = report["counts"]
        assert abs(counts["train"]["rows"] - 70) <= 10
        assert 5 <= counts["val"]["rows"] <= 25
        assert 5 <= counts["test"]["rows"] <= 25
        assert report["isolation"]["observation_disjoint"]


def test_holdout_prefers_stage1_unseen_test_groups() -> None:
    rows = _holdout_rows(count=100, unseen=40)
    plain = assign_splits(rows, mode="species", seed="unit")
    preferred = assign_splits(rows, mode="species", seed="unit", prefer_test_unseen=True)
    assert preferred["prefer_test_unseen"] is True
    assert preferred["isolation"]["ok"]
    assert preferred["stage1_overlap_counts"]["test"].get("unseen", 0) >= plain["stage1_overlap_counts"]["test"].get("unseen", 0)
    assert preferred["stage1_overlap_counts"]["test"].get("unseen", 0) > 0


def _write_split_images(root: Path, split: str, classes: dict[str, int]) -> None:
    for label, count in classes.items():
        directory = root / split / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (directory / f"{split}-{label}-{index}.jpg").write_bytes(b"not-an-image")


def _dataset(tmp_path: Path, split_mode: str) -> Path:
    root = tmp_path / f"dataset-{split_mode}"
    _write_split_images(root, "val", {"Simple": 2, "Compound": 2})
    _write_split_images(root, "test", {"Simple": 2, "Compound": 2})
    (root / "counts.json").write_text(
        json.dumps(
            {
                "split_mode": split_mode,
                "prefer_test_unseen": split_mode == "species",
                "stage1_overlap_counts": {"test": {"unseen": 2}},
            }
        )
    )
    return root


def _fake_predictions(model, paths, **kwargs):  # noqa: ANN001, ANN002
    rows = []
    model_names = ["Compound", "Simple"]  # alphabetical, as Ultralytics builds dirs
    for path in paths:
        label = Path(path).parent.name
        index = model_names.index(label)
        row = np.full(len(LEAF_SIMPLE_VS_COMPOUND.classes), 0.1, dtype=np.float64)
        row[index] = 0.7
        rows.append(row)
    return np.asarray(rows)


@pytest.mark.parametrize("split_mode", ["species", "observation"])
def test_evaluate_emits_heldout_test_metrics_for_both_split_modes(
    tmp_path: Path, monkeypatch, split_mode: str
) -> None:
    import experiments.anatomy4_specialists.device as device_module
    import experiments.anatomy4_specialists.evaluate as evaluate_module

    class FakeModel:
        # Ultralytics class dirs sort alphabetically; target order is Simple, Compound.
        names = {0: "Compound", 1: "Simple"}

    monkeypatch.setattr(evaluate_module, "load_classifier", lambda *a, **k: FakeModel())
    monkeypatch.setattr(evaluate_module, "predict_paths", _fake_predictions)
    monkeypatch.setattr(device_module, "configure_memory", lambda *a, **k: {"device": "cpu", "note": "test"})

    metrics = evaluate_run(
        tmp_path / "best.pt",
        _dataset(tmp_path, split_mode),
        LEAF_SIMPLE_VS_COMPOUND,
        tmp_path / f"out-{split_mode}",
        split="test",
    )
    assert metrics["requested_split"] == "test"
    assert metrics["splits_evaluated"] == ["val", "test"]
    assert metrics["dataset"]["split_mode"] == split_mode
    test = metrics["test"]
    assert test["accuracy"] == pytest.approx(1.0)
    assert test["macro_prf_present_classes"]["f1"] == pytest.approx(1.0)
    assert set(test["per_class_recall"]) == {"Simple", "Compound"}
    assert test["confusion_matrix"]["labels"] == ["Simple", "Compound"]
    assert test["abstention"]["coverage"] == pytest.approx(1.0)
    assert test["abstention"]["confusion_matrix"]["labels"] == ["Simple", "Compound", "abstain"]
    assert (Path(metrics["dataset_root"]) / "..").exists()
    assert (tmp_path / f"out-{split_mode}" / "predictions_test.jsonl").is_file()


def test_build_dataset_records_holdout_flag(tmp_path: Path) -> None:
    rows = []
    for index in range(60):
        label = "Simple" if index < 30 else "Compound"
        path = tmp_path / "crops" / f"c{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        rows.append(
            {
                "crop_id": f"c{index}",
                "observation_id": f"obs-{index}",
                "taxon_name": f"Genus{index} species{index}",
                "image": "https://example.invalid/x.jpg",
                "crop_path": str(path),
                "anatomy_class": "leaf",
                "detector_confidence": 0.9,
                "observation_split": "train",
                "target_label": label,
                "in_stage1_train_val": index >= 30,
            }
        )
    summary = build_classification_dataset(
        rows,
        LEAF_SIMPLE_VS_COMPOUND,
        tmp_path / "dataset",
        split_mode="species",
        min_per_class_per_split=1,
        link_mode="copy",
        prefer_test_unseen=True,
    )
    assert summary["prefer_test_unseen"] is True
    assert summary["stage1_overlap_counts"]["test"].get("unseen", 0) > 0
    counts = json.loads((tmp_path / "dataset" / "counts.json").read_text())
    assert counts["prefer_test_unseen"] is True
