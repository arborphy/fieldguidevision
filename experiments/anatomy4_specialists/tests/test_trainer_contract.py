# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.anatomy4_specialists.contracts import LEAF_SIMPLE_VS_COMPOUND
from experiments.anatomy4_specialists.evaluate import evaluate_run
from experiments.anatomy4_specialists.trainer import (
    bounded_memory_env,
    dry_run,
    preflight,
    training_args,
)


def _config(tmp_path: Path) -> Path:
    config = {
        "contract": "anatomy4-specialists/v1",
        "target": {
            "target_id": "leaf_simple_vs_compound",
            "expert": "leaf",
            "feature_name": "leaf_type",
            "classes": ["Simple", "Compound"],
            "anatomy_classes": ["leaf"],
        },
        "dataset_root": str(tmp_path / "dataset"),
        "output_dir": str(tmp_path / "runs"),
        "training": {"batch": 16, "imgsz": 224},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def _fake_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        for label in ("Simple", "Compound"):
            (root / split / label).mkdir(parents=True, exist_ok=True)
    (root / "labels.json").write_text(
        json.dumps(
            {
                "classes": ["Simple", "Compound"],
                "target": LEAF_SIMPLE_VS_COMPOUND.to_dict(),
            }
        )
    )
    (root / "manifest.jsonl").write_text("")
    (root / "validation.json").write_text(json.dumps({"ok": True, "errors": []}))
    return root


def test_training_args_bounds_and_cache_normalization(tmp_path: Path) -> None:
    config = {"raw": {"training": {"batch": 16, "cache": "false", "imgsz": 224}}, "target": LEAF_SIMPLE_VS_COMPOUND}
    args = training_args(config)
    assert args["batch"] == 16
    assert args["cache"] is False
    assert args["max_vram_fraction"] == 0.30
    with pytest.raises(ValueError):
        training_args(config, {"batch": 0})
    with pytest.raises(ValueError):
        training_args(config, {"max_vram_fraction": 1.5})
    with pytest.raises(ValueError):
        training_args(config, {"cache": "nvme"})


def test_bounded_memory_env_is_advisory_not_a_service_change() -> None:
    env = bounded_memory_env()
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "OMP_NUM_THREADS" in env


def test_preflight_requires_validated_dataset(tmp_path: Path) -> None:
    path = _config(tmp_path)
    from experiments.anatomy4_specialists.contracts import load_run_config

    config = load_run_config(path)
    args = training_args(config)
    with pytest.raises(FileNotFoundError):
        preflight(config, args)

    _fake_dataset(tmp_path)
    result = preflight(config, args)
    assert result["validation"]["ok"] is True
    assert result["labels"]["classes"] == ["Simple", "Compound"]


def test_preflight_rejects_failed_validation(tmp_path: Path) -> None:
    path = _config(tmp_path)
    root = _fake_dataset(tmp_path)
    (root / "validation.json").write_text(json.dumps({"ok": False, "errors": ["bad"]}))
    from experiments.anatomy4_specialists.contracts import load_run_config

    config = load_run_config(path)
    with pytest.raises(RuntimeError):
        preflight(config, training_args(config))


def test_preflight_rejects_target_mismatch(tmp_path: Path) -> None:
    path = _config(tmp_path)
    root = _fake_dataset(tmp_path)
    labels = json.loads((root / "labels.json").read_text())
    labels["classes"] = ["Entire", "Serrate"]
    (root / "labels.json").write_text(json.dumps(labels))
    from experiments.anatomy4_specialists.contracts import load_run_config

    config = load_run_config(path)
    with pytest.raises(ValueError):
        preflight(config, training_args(config))


def test_dry_run_does_not_produce_metrics(tmp_path: Path) -> None:
    path = _config(tmp_path)
    plan = dry_run(path)
    assert plan["target"]["target_id"] == "leaf_simple_vs_compound"
    assert plan["training_args"]["batch"] == 16
    assert "no training" in plan["note"]
    assert not (tmp_path / "runs").exists()


def test_evaluate_refuses_missing_checkpoint(tmp_path: Path) -> None:
    _fake_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluate_run(
            tmp_path / "missing.pt",
            tmp_path / "dataset",
            LEAF_SIMPLE_VS_COMPOUND,
            tmp_path / "out",
        )
