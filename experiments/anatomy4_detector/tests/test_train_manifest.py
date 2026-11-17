# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.anatomy4_detector.manifest import (
    MANIFEST_SCHEMA,
    build_run_manifest,
    git_info,
    sha256_file,
    write_manifest,
)
from experiments.anatomy4_detector.train import (
    build_train_kwargs,
    validate_train_kwargs,
)


def test_train_kwargs_are_explicit_and_detect_only() -> None:
    kwargs = build_train_kwargs(
        data="prepared/data.yaml",
        project="runs",
        name="anatomy4",
        device="0",
    )
    assert kwargs["task"] == "detect"
    assert kwargs["data"] == "prepared/data.yaml"
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == 8
    assert kwargs["workers"] == 4
    assert kwargs["device"] == "0"
    assert kwargs["epochs"] == 80
    assert kwargs["amp"] is True
    assert kwargs["seed"] == 20261112


@pytest.mark.parametrize(
    "override",
    [
        {"epochs": 0},
        {"imgsz": 32},
        {"batch": 0},
        {"workers": -1},
        {"model": "yolo26n-seg.pt"},
    ],
)
def test_train_kwargs_reject_invalid(override: dict) -> None:
    kwargs = build_train_kwargs(
        data="data.yaml", project="runs", device="0", **override
    )
    with pytest.raises(ValueError):
        validate_train_kwargs(kwargs)


def test_manifest_hashes_artifacts_and_condenses_api_registration(tmp_path: Path) -> None:
    onnx_path = tmp_path / "anatomy4.onnx"
    onnx_path.write_bytes(b"onnx-bytes")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}")
    audit = {
        "dataset_root": str(tmp_path / "dataset"),
        "status": "ok",
        "blocking": False,
        "split_isolation": "clean",
        "totals": {"images": 10, "negatives": 2, "boxes": 20},
        "splits": {"train": {"images": 6}, "val": {"images": 2}, "test": {"images": 2}},
        "support": {"boxes_per_class": {"leaf": 5}},
        "labels_sha256": "abc",
        "errors": [],
        "warnings": [],
    }
    evaluation = {
        "split": "test",
        "per_class": {"leaf": {"support": 2, "ap50": 0.5}},
        "macro": {"ap50": 0.5},
        "micro": {"precision": 1.0},
        "counts": {"images": 2},
        "confusion_matrix": {"labels": []},
        "operating_point": {"conf": 0.25, "iou": 0.5},
        "predictions_jsonl": str(tmp_path / "predictions.jsonl"),
        "metrics_json": str(metrics_path),
    }
    manifest = build_run_manifest(
        code_root=Path(__file__).resolve().parents[3],
        dataset_audit=audit,
        dataset_audit_path=None,
        training={"ok": True, "best_weights": str(weights)},
        evaluation=evaluation,
        export={
            "onnx": str(onnx_path),
            "sha256": sha256_file(onnx_path),
            "export_kwargs": {"opset": 17},
        },
        onnx_verification={
            "ok": True,
            "path": str(onnx_path),
            "sha256": sha256_file(onnx_path),
            "input": {"shape": [1, 3, 640, 640]},
            "output": {"shape": [1, 8, 8400]},
            "precision": "fp32",
            "class_names": ["leaf", "needle", "flower", "inflorescence"],
            "end2end": False,
        },
    )
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["status"] == "complete"
    onnx_artifact = next(
        item for item in manifest["artifacts"] if item["kind"] == "onnx"
    )
    assert onnx_artifact["sha256"] == sha256_file(onnx_path)
    assert manifest["api_registration"]["output_shape"] == [1, 8, 8400]
    assert manifest["api_registration"]["classes"] == [
        "leaf",
        "needle",
        "flower",
        "inflorescence",
    ]
    assert manifest["evaluation"]["metrics"]["macro"] == {"ap50": 0.5}
    path = write_manifest(manifest, tmp_path / "run-manifest.json")
    assert json.loads(path.read_text())["contract"] == "anatomy4-v1"


def test_git_info_reads_the_real_worktree() -> None:
    info = git_info(Path(__file__).resolve().parents[3])
    assert info["available"] is True
    assert len(info["revision"]) == 40


def test_manifest_records_val_thresholds_and_leaf_tagged_metrics(tmp_path: Path) -> None:
    audit = {
        "dataset_root": str(tmp_path / "dataset"),
        "status": "ok",
        "blocking": False,
        "split_isolation": "clean",
        "totals": {"images": 10, "negatives": 2, "boxes": 20},
        "splits": {"train": {"images": 6}, "val": {"images": 2}, "test": {"images": 2}},
        "support": {"boxes_per_class": {"leaf": 5}},
        "labels_sha256": "abc",
        "errors": [],
        "warnings": [],
    }
    leaf = {"precision": 0.8, "recall": 0.7, "ap50": 0.75, "support": 4}
    per_class = {name: dict(leaf) for name in ("leaf", "needle", "flower", "inflorescence")}
    test_eval = {
        "split": "test",
        "per_class": per_class,
        "macro": {"f1": 0.7, "ap50": 0.75},
        "micro": {"precision": 0.8},
        "counts": {"images": 2, "gt_boxes": 16},
        "confusion_matrix": {"labels": []},
        "operating_point": {"conf": 0.25, "iou": 0.5},
        "tagged_bbox": {"leaf": leaf, "per_class": per_class},
        "leaf_tagged_bbox": leaf,
        "threshold_sweep": {"thresholds": [0.1, 0.25], "results": {}},
        "selected_threshold": 0.1,
        "tuning": {"method": "val_macro_f1", "selected": 0.1},
        "metrics_json": str(tmp_path / "eval-test" / "metrics.json"),
    }
    val_eval = dict(test_eval, split="val", selected_threshold=0.1)
    val_eval["metrics_json"] = str(tmp_path / "eval-val" / "metrics.json")
    (tmp_path / "eval-test").mkdir()
    (tmp_path / "eval-test" / "metrics.json").write_text("{}")
    (tmp_path / "eval-val").mkdir()
    (tmp_path / "eval-val" / "metrics.json").write_text("{}")
    manifest = build_run_manifest(
        code_root=Path(__file__).resolve().parents[3],
        dataset_audit=audit,
        dataset_audit_path=None,
        training={"ok": True, "best_weights": None},
        evaluation=test_eval,
        evaluation_val=val_eval,
        export=None,
        onnx_verification=None,
    )
    assert manifest["evaluation"]["selected_threshold"] == 0.1
    assert manifest["evaluation"]["leaf_tagged_bbox"]["ap50"] == 0.75
    assert manifest["evaluation"]["metrics"]["tagged_bbox"]["leaf"]["recall"] == 0.7
    assert manifest["evaluation"]["metrics"]["threshold_sweep"]["thresholds"] == [0.1, 0.25]
    assert manifest["evaluation_val"]["split"] == "val"
    assert manifest["evaluation_val"]["leaf_tagged_bbox"]["ap50"] == 0.75


def test_finetune_hyperparameters_are_forwarded() -> None:
    kwargs = build_train_kwargs(
        data="data.yaml",
        project="runs",
        device="0",
        optimizer="AdamW",
        lr0=0.0006,
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=5.0,
        cls=1.5,
        mosaic=0.7,
        close_mosaic=20,
    )
    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["lr0"] == pytest.approx(0.0006)
    assert kwargs["lrf"] == pytest.approx(0.05)
    assert kwargs["cos_lr"] is True
    assert kwargs["warmup_epochs"] == pytest.approx(5.0)
    assert kwargs["cls"] == pytest.approx(1.5)
    assert kwargs["mosaic"] == pytest.approx(0.7)
    assert kwargs["close_mosaic"] == 20
    validate_train_kwargs(kwargs)
