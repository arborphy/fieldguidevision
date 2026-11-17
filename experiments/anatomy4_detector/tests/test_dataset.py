# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.anatomy4_detector.contracts import ContractError
from experiments.anatomy4_detector.dataset import (
    prepare_ultralytics,
    validate_dataset,
)

from helpers import make_dataset, write_label


def test_valid_dataset_reports_honest_stats(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    audit = validate_dataset(tmp_path)
    assert audit["status"] == "ok", audit["errors"]
    assert audit["blocking"] is False
    assert audit["split_isolation"] == "clean"
    assert audit["totals"]["images"] == 4
    assert audit["totals"]["negatives"] == 1
    assert audit["totals"]["boxes"] == 3
    assert audit["totals"]["boxes_per_class"] == {
        "leaf": 1,
        "needle": 1,
        "flower": 1,
        "inflorescence": 0,
    }
    assert audit["splits"]["train"]["negatives"] == 1
    assert audit["splits"]["test"]["images"] == 1
    assert audit["support"]["test_boxes_per_class"]["flower"] == 1
    assert audit["support"]["test_boxes_per_class"]["leaf"] == 0


def test_missing_label_file_is_an_error_even_for_a_negative(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "neg.txt").unlink()
    audit = validate_dataset(tmp_path)
    codes = {item["code"] for item in audit["errors"]}
    assert "missing_label_file" in codes
    assert audit["blocking"] is True


def test_negative_with_boxes_is_rejected(tmp_path: Path) -> None:
    make_dataset(
        tmp_path,
        entries=[
            {
                "image": "images/train/a.jpg",
                "observation_id": "obs-a",
                "negative": True,
                "boxes": ["0 0.1 0.1 0.5 0.5"],
            },
            {
                "image": "images/val/b.jpg",
                "observation_id": "obs-b",
                "negative": False,
                "boxes": [],
            },
            {
                "image": "images/test/c.jpg",
                "observation_id": "obs-c",
                "negative": False,
                "boxes": [],
            },
        ],
    )
    audit = validate_dataset(tmp_path)
    codes = {item["code"] for item in audit["errors"]}
    assert "negative_has_boxes" in codes


def test_undeclared_negative_is_a_warning_not_an_error(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    metadata = json.loads((tmp_path / "dataset.json").read_text())
    for entry in metadata["images"]:
        if entry["image"].endswith("neg.jpg"):
            entry["negative"] = False
    (tmp_path / "dataset.json").write_text(json.dumps(metadata))
    audit = validate_dataset(tmp_path)
    assert audit["status"] == "ok"
    assert "undeclared_negative" in {item["code"] for item in audit["warnings"]}
    assert audit["splits"]["train"]["negatives"] == 1


def test_observation_leak_across_splits_is_blocking(tmp_path: Path) -> None:
    make_dataset(
        tmp_path,
        entries=[
            {
                "image": "images/train/a.jpg",
                "observation_id": "obs-leak",
                "negative": False,
                "boxes": ["0 0.1 0.1 0.5 0.5"],
            },
            {
                "image": "images/val/b.jpg",
                "observation_id": "obs-b",
                "negative": False,
                "boxes": ["1 0.1 0.1 0.5 0.5"],
            },
            {
                "image": "images/test/c.jpg",
                "observation_id": "obs-leak",
                "negative": False,
                "boxes": ["2 0.1 0.1 0.5 0.5"],
            },
        ],
    )
    audit = validate_dataset(tmp_path)
    assert audit["split_isolation"] == "leaked"
    assert "observation_leak" in {item["code"] for item in audit["errors"]}
    assert audit["blocking"] is True


def test_same_observation_within_one_split_is_legal(tmp_path: Path) -> None:
    make_dataset(
        tmp_path,
        entries=[
            {
                "image": "images/train/a.jpg",
                "observation_id": "obs-shared",
                "negative": False,
                "boxes": ["0 0.1 0.1 0.5 0.5"],
            },
            {
                "image": "images/train/a2.jpg",
                "observation_id": "obs-shared",
                "negative": False,
                "boxes": ["0 0.2 0.2 0.6 0.6"],
            },
            {
                "image": "images/val/b.jpg",
                "observation_id": "obs-b",
                "negative": False,
                "boxes": ["1 0.1 0.1 0.5 0.5"],
            },
            {
                "image": "images/test/c.jpg",
                "observation_id": "obs-c",
                "negative": False,
                "boxes": ["2 0.1 0.1 0.5 0.5"],
            },
        ],
    )
    audit = validate_dataset(tmp_path)
    assert audit["status"] == "ok"
    assert audit["split_isolation"] == "clean"


def test_malformed_labels_are_not_counted_as_negatives(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    # cx cy w h misread as xyxy: x0 > x1, so the label is invalid.
    (tmp_path / "labels" / "train" / "a.txt").write_text(
        "0 0.526808 0.179688 0.293017 0.236328\n"
    )
    audit = validate_dataset(tmp_path)
    assert "label_invalid" in {item["code"] for item in audit["errors"]}
    assert audit["splits"]["train"]["negatives"] == 1  # only neg.jpg
    assert audit["splits"]["train"]["boxes"] == 0
    assert audit["totals"]["negatives"] == 1


def test_invalid_label_and_orphan_label_are_errors(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.9 0.9 0.1 0.1\n")
    write_label(tmp_path / "labels" / "val" / "orphan.txt", ["0 0.1 0.1 0.2 0.2"])
    audit = validate_dataset(tmp_path)
    codes = {item["code"] for item in audit["errors"]}
    assert "label_invalid" in codes
    assert "orphan_label_file" in codes


def test_class_order_mismatch_is_an_error(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    metadata = json.loads((tmp_path / "dataset.json").read_text())
    metadata["classes"] = ["leaf", "flower", "needle", "inflorescence"]
    (tmp_path / "dataset.json").write_text(json.dumps(metadata))
    audit = validate_dataset(tmp_path)
    assert "class_order_mismatch" in {item["code"] for item in audit["errors"]}


def test_dataset_json_declaring_missing_image_is_an_error(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    metadata = json.loads((tmp_path / "dataset.json").read_text())
    metadata["images"].append(
        {"image": "images/test/missing.jpg", "observation_id": "obs-x", "negative": True}
    )
    (tmp_path / "dataset.json").write_text(json.dumps(metadata))
    audit = validate_dataset(tmp_path)
    assert "declared_image_missing" in {item["code"] for item in audit["errors"]}


def test_without_observation_map_isolation_is_unverifiable(tmp_path: Path) -> None:
    make_dataset(tmp_path, write_metadata=False)
    audit = validate_dataset(tmp_path)
    assert audit["status"] == "ok"
    assert audit["split_isolation"] == "unverifiable"
    assert "observation_id_missing" in {item["code"] for item in audit["warnings"]}


def test_prepare_writes_cxcywh_labels_and_explicit_empty_negatives(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    audit = validate_dataset(tmp_path)
    prepared = prepare_ultralytics(audit, tmp_path / "prepared", link_mode="copy")
    label = (tmp_path / "prepared" / "labels" / "train" / "a.txt").read_text().strip()
    fields = [float(value) for value in label.split()]
    assert fields[0] == 0
    assert fields[1:] == pytest.approx([0.3, 0.3, 0.4, 0.4])
    negative = tmp_path / "prepared" / "labels" / "train" / "neg.txt"
    assert negative.is_file()
    assert negative.read_text() == ""
    data_yaml = (tmp_path / "prepared" / "data.yaml").read_text()
    assert '"leaf"' in data_yaml and '"inflorescence"' in data_yaml
    assert (tmp_path / "prepared" / "images" / "train" / "a.jpg").is_file()
    assert prepared["splits"] == {"train": 2, "val": 1, "test": 1}


def test_prepare_refuses_invalid_dataset(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "a.txt").unlink()
    audit = validate_dataset(tmp_path)
    with pytest.raises(ContractError):
        prepare_ultralytics(audit, tmp_path / "prepared")


def test_train_val_dataset_validates_without_test(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "images" / "test" / "c.jpg").unlink()
    (tmp_path / "labels" / "test" / "c.txt").unlink()
    metadata = json.loads((tmp_path / "dataset.json").read_text())
    metadata["images"] = [e for e in metadata["images"] if "images/test/" not in e["image"]]
    (tmp_path / "dataset.json").write_text(json.dumps(metadata))
    audit = validate_dataset(tmp_path, require_test=False)
    assert audit["status"] == "ok", audit["errors"]
    assert audit["splits"]["test"]["images"] == 0
    assert audit["splits"]["train"]["images"] >= 1


def test_assert_disjoint_datasets_detects_each_identity(tmp_path: Path) -> None:
    from experiments.anatomy4_detector.contracts import ContractError
    from experiments.anatomy4_detector.dataset import (
        assert_disjoint_datasets,
        dataset_identity,
    )

    left = tmp_path / "left"
    right = tmp_path / "right"
    make_dataset(left)
    make_dataset(right)
    identity_left = dataset_identity(left)
    assert identity_left["counts"]["observations"] == 4
    with pytest.raises(ContractError, match="isolation violated"):
        assert_disjoint_datasets(left, right)

    # Make the right dataset fully distinct.
    metadata = json.loads((right / "dataset.json").read_text())
    for index, entry in enumerate(metadata["images"]):
        entry["observation_id"] = f"other-{index}"
        entry["photo_id"] = f"other-{index}"
        entry["source_sha256"] = f"sha-other-{index}"
    (right / "dataset.json").write_text(json.dumps(metadata))
    clean = assert_disjoint_datasets(left, right)
    assert clean["total_overlap"] == 0
    assert clean["overlap_counts"] == {
        "observation_ids": 0,
        "photo_ids": 0,
        "sha256": 0,
    }


def test_oversample_writes_train_only_repeats(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    # Make the train image carry a needle so it is oversampled.
    (tmp_path / "labels" / "train" / "a.txt").write_text(
        "0 0.100000 0.100000 0.300000 0.300000\n"
        "1 0.400000 0.100000 0.500000 0.200000\n"
    )
    metadata = json.loads((tmp_path / "dataset.json").read_text())
    for entry in metadata["images"]:
        if entry["image"].endswith("a.jpg"):
            entry["boxes"] = [
                {"class_id": 0, "class_name": "leaf", "x0": 0.1, "y0": 0.1, "x1": 0.3, "y1": 0.3},
                {"class_id": 1, "class_name": "needle", "x0": 0.4, "y0": 0.1, "x1": 0.5, "y1": 0.2},
            ]
    (tmp_path / "dataset.json").write_text(json.dumps(metadata))
    audit = validate_dataset(tmp_path)
    prepared = prepare_ultralytics(
        audit,
        tmp_path / "prepared",
        link_mode="copy",
        oversample_class="needle",
        oversample_factor=3,
    )
    train_images = sorted(p.name for p in (tmp_path / "prepared" / "images" / "train").iterdir())
    assert "a__rep1.jpg" in train_images and "a__rep2.jpg" in train_images
    assert not any("__rep" in name for name in (p.name for p in (tmp_path / "prepared" / "images" / "val").iterdir()))
    assert (tmp_path / "prepared" / "labels" / "train" / "a__rep1.txt").read_text() == (
        tmp_path / "prepared" / "labels" / "train" / "a.txt"
    ).read_text()
    assert prepared["oversample"]["class"] == "needle"
    assert prepared["oversample"]["extra_train_images"] == 2
    assert prepared["splits"]["train"] == audit["splits"]["train"]["images"] + 2
    # Rerun is idempotent (no duplicates).
    again = prepare_ultralytics(
        audit,
        tmp_path / "prepared",
        link_mode="copy",
        oversample_class="needle",
        oversample_factor=3,
    )
    assert again["oversample"]["extra_train_images"] == 2
    assert len(list((tmp_path / "prepared" / "images" / "train").glob("*__rep*"))) == 2
