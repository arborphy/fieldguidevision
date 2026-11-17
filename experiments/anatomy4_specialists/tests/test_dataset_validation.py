# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.anatomy4_specialists.contracts import LEAF_SIMPLE_VS_COMPOUND
from experiments.anatomy4_specialists.dataset import (
    build_classification_dataset,
    load_crop_manifest,
    select_supervised,
    validate_dataset,
)
from experiments.anatomy4_specialists.splits import assign_splits


def _crop(
    tmp_path: Path,
    crop_id: str,
    *,
    observation_id: str,
    taxon: str,
    split: str,
    label: str | None,
    anatomy: str = "leaf",
    eligible: bool | None = True,
):
    path = tmp_path / "crops" / f"{crop_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (100, 120, 80)).save(path)
    row = {
        "crop_id": crop_id,
        "observation_id": observation_id,
        "taxon_name": taxon,
        "image": f"https://example.invalid/{crop_id}.jpg",
        "crop_path": str(path),
        "anatomy_class": anatomy,
        "detector_confidence": 0.9,
        "observation_split": split,
        "split": split,
    }
    if label is not None:
        row["target_label"] = label
    if eligible is not None:
        row["weak_label_eligible"] = eligible
    return row


def _valid_rows(tmp_path: Path):
    rows = []
    for split, suffix in (("train", "tr"), ("val", "va"), ("test", "te")):
        for class_index, (label, genus) in enumerate((("Simple", "Acer"), ("Compound", "Quercus"))):
            for i in range(2):
                rows.append(
                    _crop(
                        tmp_path,
                        crop_id=f"{suffix}-{label}-{i}",
                        observation_id=f"{suffix}-{label}-obs-{i}",
                        taxon=f"{genus}{class_index} species{suffix}{class_index}",
                        split=split,
                        label=label,
                    )
                )
    return rows


def test_valid_dataset_passes_species_disjoint(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    report = validate_dataset(
        rows,
        LEAF_SIMPLE_VS_COMPOUND,
        require_labels=True,
        check_files=True,
        require_species_disjoint=True,
        min_per_class_per_split=2,
    )
    assert report.ok, report.errors
    assert report.counts["supervised_rows"] == 12
    assert report.isolation["ok"]
    assert report.isolation["species_disjoint"]


def test_label_on_mismatched_anatomy_is_a_contract_error(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    rows[0]["anatomy_class"] = "flower"
    report = validate_dataset(rows, LEAF_SIMPLE_VS_COMPOUND, require_labels=True)
    assert not report.ok
    assert any("does not accept" in error for error in report.errors)


def test_missing_label_is_reported(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    rows[0].pop("target_label")
    rows[0].pop("weak_label_eligible", None)
    report = validate_dataset(rows, LEAF_SIMPLE_VS_COMPOUND, require_labels=True)
    assert not report.ok
    assert report.exclusions.get("missing_label") == 1


def test_allow_unlabeled_reports_exclusions_instead_of_failing(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    rows[0].pop("target_label")
    rows[0].pop("weak_label_eligible", None)
    report = validate_dataset(
        rows,
        LEAF_SIMPLE_VS_COMPOUND,
        require_labels=False,
        allow_unlabeled=True,
        min_per_class_per_split=0,
    )
    # 2 per class still present in train/val/test, so the dataset remains valid.
    assert report.exclusions.get("missing_label") == 1


def test_missing_class_in_val_fails(tmp_path: Path) -> None:
    rows = [row for row in _valid_rows(tmp_path) if not (row["observation_split"] == "val" and row["target_label"] == "Compound")]
    report = validate_dataset(rows, LEAF_SIMPLE_VS_COMPOUND, require_labels=True)
    assert not report.ok
    assert any("val split is missing target classes" in error for error in report.errors)


def test_below_min_per_class_fails(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    report = validate_dataset(
        rows, LEAF_SIMPLE_VS_COMPOUND, require_labels=True, min_per_class_per_split=3
    )
    assert not report.ok
    assert any("2 < 3" in error for error in report.errors)


def test_missing_file_fails_when_checked(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    Path(rows[0]["crop_path"]).unlink()
    report = validate_dataset(
        rows, LEAF_SIMPLE_VS_COMPOUND, require_labels=True, check_files=True
    )
    assert not report.ok
    assert any("does not exist" in error for error in report.errors)


def test_duplicate_crop_id_fails(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    rows[1]["crop_id"] = rows[0]["crop_id"]
    report = validate_dataset(rows, LEAF_SIMPLE_VS_COMPOUND, require_labels=True)
    assert not report.ok
    assert any("duplicate crop_id" in error for error in report.errors)


def test_split_isolation_violation_is_caught(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    # Make two crops share one observation, then split that observation across
    # train/test. Observation-level isolation must catch it.
    rows[1]["observation_id"] = rows[0]["observation_id"]
    rows[1]["observation_split"] = "test"
    rows[1]["split"] = "test"
    report = validate_dataset(rows, LEAF_SIMPLE_VS_COMPOUND, require_labels=True)
    assert not report.ok
    assert any("split isolation failed" in error for error in report.errors)


def test_select_supervised_excludes_ineligible_weak_labels(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    rows[0]["weak_label_eligible"] = False
    rows[0]["weak_label_reason"] = "anatomy_mismatch:flower"
    selected, reasons = select_supervised(rows, LEAF_SIMPLE_VS_COMPOUND)
    assert len(selected) == 11
    assert reasons["weak_label_ineligible:anatomy_mismatch:flower"] == 1


def test_build_classification_dataset_writes_tree(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    out = tmp_path / "dataset"
    summary = build_classification_dataset(
        rows,
        LEAF_SIMPLE_VS_COMPOUND,
        out,
        use_manifest_splits=True,
        link_mode="copy",
        min_per_class_per_split=2,
    )
    assert summary["counts"]["total"] == 12
    for split in ("train", "val", "test"):
        for label in LEAF_SIMPLE_VS_COMPOUND.classes:
            assert (out / split / label).is_dir()
            assert len(list((out / split / label).iterdir())) == 2
    labels = json.loads((out / "labels.json").read_text())
    assert labels["classes"] == ["Simple", "Compound"]
    assert labels["label_to_index"] == {"Simple": 0, "Compound": 1}
    assert labels["abstain"]["class"] == "abstain"
    assert (out / "manifest.jsonl").is_file()
    assert json.loads((out / "validation.json").read_text())["ok"] is True


def test_build_classification_dataset_recomputes_splits(tmp_path: Path) -> None:
    # Enough species groups that a species-disjoint recompute can populate
    # both classes in train/val. Each species has one observation and crop.
    rows = []
    for class_index, label in enumerate(("Simple", "Compound")):
        for i in range(20):
            rows.append(
                _crop(
                    tmp_path,
                    crop_id=f"{label}-{i}",
                    observation_id=f"obs-{label}-{i}",
                    taxon=f"Genus{i} species{i}",
                    split="train",
                    label=label,
                )
            )
    out = tmp_path / "dataset"
    summary = build_classification_dataset(
        rows,
        LEAF_SIMPLE_VS_COMPOUND,
        out,
        split_mode="species",
        link_mode="copy",
        min_per_class_per_split=1,
    )
    assert summary["isolation"]["ok"]
    assert summary["isolation"]["species_disjoint"]
    manifest = load_crop_manifest(out / "manifest.jsonl")
    assert len({row["observation_id"] for row in manifest}) == len(manifest)


def test_assign_splits_for_dataset_rows_is_observation_disjoint(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    report = assign_splits(rows, mode="observation", seed="unit")
    assert report["isolation"]["ok"]


def test_invalid_split_value_fails(tmp_path: Path) -> None:
    rows = _valid_rows(tmp_path)
    rows[0]["observation_split"] = "holdout"
    report = validate_dataset(rows, LEAF_SIMPLE_VS_COMPOUND, require_labels=True)
    assert not report.ok
    assert any("invalid observation_split" in error for error in report.errors)
