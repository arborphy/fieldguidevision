# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

from pathlib import Path

from experiments.anatomy4_detector.dataset import (
    prepare_ultralytics,
    validate_dataset,
)
from experiments.anatomy4_detector.smoke import (
    build_synthetic_dataset,
    run_smoke,
)


def test_synthetic_dataset_is_valid_and_has_explicit_negatives(tmp_path: Path) -> None:
    build_synthetic_dataset(tmp_path / "dataset")
    audit = validate_dataset(tmp_path / "dataset")
    assert audit["status"] == "ok", audit["errors"]
    assert audit["split_isolation"] == "clean"
    assert audit["totals"]["images"] == 24
    assert audit["totals"]["negatives"] == 3
    assert audit["totals"]["boxes"] > 0
    # One explicit empty label file per split.
    for split in ("train", "val", "test"):
        labels = list((tmp_path / "dataset" / "labels" / split).glob("*.txt"))
        assert labels
        assert any(path.read_text() == "" for path in labels)
    prepared = prepare_ultralytics(audit, tmp_path / "prepared", link_mode="copy")
    empty = [
        path
        for path in (tmp_path / "prepared" / "labels" / "train").glob("*.txt")
        if path.read_text() == ""
    ]
    assert empty, prepared
    assert (tmp_path / "prepared" / "data.yaml").is_file()


def test_smoke_dry_run_writes_a_partial_manifest(tmp_path: Path) -> None:
    manifest = run_smoke(tmp_path / "smoke", dry_run=True, device="cpu")
    assert manifest["synthetic"] is True
    assert manifest["status"] == "dry_run"
    assert manifest["dataset"]["status"] == "ok"
    assert (tmp_path / "smoke" / "run-manifest.json").is_file()
