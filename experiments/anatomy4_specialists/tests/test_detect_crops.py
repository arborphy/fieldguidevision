# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from experiments.anatomy4_specialists.contracts import LEAF_SIMPLE_VS_COMPOUND
from experiments.anatomy4_specialists.dataset import validate_dataset
from experiments.anatomy4_specialists.detect_crops import (
    STAGE1_BOX_SOURCE,
    STAGE1_CLASSES,
    STAGE1_REVIEW_VERDICT,
    crops_from_detections,
    load_dirr_manifest,
    load_stage1_overlap,
    stage1_overlap_flag,
)


def _image(path: Path, size=(200, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 120, 80)).save(path)
    return path


def _detection(class_name: str, confidence: float, bbox=(0.1, 0.1, 0.6, 0.6)):
    return {"class_name": class_name, "confidence": confidence, "bbox": list(bbox)}


def _manifest_rows(tmp_path: Path) -> list[dict]:
    rows = []
    for index, license_value in enumerate(("cc-by-nc", None, "CC0")):
        path = _image(tmp_path / f"img{index}.jpg")
        rows.append(
            {
                "observation_id": f"inat-{index}",
                "taxon_name": "Acer rubrum",
                "photo_id": f"photo-{index}",
                "local_path": str(path),
                "source_url": f"https://example.invalid/{index}.jpg",
                "license": license_value,
            }
        )
    return rows


def test_crops_from_stage1_detections(tmp_path: Path) -> None:
    image = _image(tmp_path / "img.jpg")
    out = tmp_path / "out"
    rows, skipped = crops_from_detections(
        image,
        [
            _detection("leaf", 0.91),
            _detection("flower", 0.72, (0.5, 0.5, 0.9, 0.9)),
            _detection("needle", 0.60),
            _detection("inflorescence", 0.55),
            _detection("bark", 0.99),
            _detection("leaf", 0.40, (0.0, 0.0, 0.2, 0.2)),
        ],
        out,
        observation_id="inat-1",
        taxon_name="Acer rubrum",
        photo_id="p1",
        image_url="https://example.invalid/p1.jpg",
        observation_split_value="train",
        in_stage1_train_val=False,
        max_crops=3,
    )
    assert len(rows) == 3
    assert {row["anatomy_class"] for row in rows} == {"leaf", "flower", "needle"}
    assert skipped["unknown_stage1_class"] == 1
    for row in rows:
        assert row["detector_box_source"] == STAGE1_BOX_SOURCE
        assert row["review_verdict"] == STAGE1_REVIEW_VERDICT
        assert row["in_stage1_train_val"] is False
        assert row["observation_split"] == "train"
        assert Path(row["crop_path"]).is_file()
        assert row["crop_bbox_px"][2] > row["bbox"][0] * 200


def test_max_crops_per_image_takes_highest_confidence(tmp_path: Path) -> None:
    image = _image(tmp_path / "img.jpg")
    rows, _ = crops_from_detections(
        image,
        [_detection("leaf", 0.3), _detection("flower", 0.9)],
        tmp_path / "out",
        observation_id="o",
        taxon_name="Acer rubrum",
        photo_id="p",
        image_url="u",
        max_crops=1,
    )
    assert len(rows) == 1
    assert rows[0]["anatomy_class"] == "flower"


def test_load_dirr_manifest_excludes_null_license(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in _manifest_rows(tmp_path)) + "\n")
    kept, excluded = load_dirr_manifest(path)
    assert len(kept) == 3 and excluded == 0
    kept, excluded = load_dirr_manifest(path, exclude_null_license=True)
    assert len(kept) == 2 and excluded == 1
    assert all(row["license"] for row in kept)


def test_stage1_dataset_overlap_forms(tmp_path: Path) -> None:
    train_dir = tmp_path / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "inat-34975106__55114041.jpg").write_bytes(b"x")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "train": str(train_dir),
                "val": ["inat-999__888"],
                "names": {"0": "leaf"},
            }
        )
    )
    tokens = load_stage1_overlap(dataset)
    row = {"observation_id": "inat-34975106", "photo_id": "55114041", "local_path": "x.jpg"}
    assert stage1_overlap_flag(row, tokens) is True
    assert stage1_overlap_flag({"observation_id": "inat-1", "photo_id": "2"}, tokens) is False
    assert stage1_overlap_flag(row, None) is None

    records = tmp_path / "obs.json"
    records.write_text(json.dumps({"observations": [{"observation_id": "inat-1", "split": "val"}]}))
    tokens = load_stage1_overlap(records)
    assert stage1_overlap_flag({"observation_id": "inat-1", "photo_id": "2"}, tokens) is True

    anatomy_json = tmp_path / "anatomy4-dataset.json"
    anatomy_json.write_text(
        json.dumps(
            {
                "classes": list(STAGE1_CLASSES),
                "images": [
                    {"image": "images/train/tc-1__photo-1.jpg", "observation_id": "inat-1", "photo_id": "photo-1"},
                    {"image": "images/val/tc-2__photo-2.jpg", "observation_id": "inat-2", "photo_id": "photo-2"},
                ],
            }
        )
    )
    tokens = load_stage1_overlap(anatomy_json)
    assert stage1_overlap_flag({"observation_id": "inat-1", "photo_id": "photo-1"}, tokens) is True
    assert stage1_overlap_flag({"observation_id": "inat-2", "photo_id": "photo-2"}, tokens) is True
    assert stage1_overlap_flag({"observation_id": "inat-3", "photo_id": "photo-3"}, tokens) is False


def test_detections_from_result_normalizes_boxes() -> None:
    from experiments.anatomy4_specialists.detect_crops import _detections_from_result

    boxes = SimpleNamespace(
        xyxy=np.asarray([[20.0, 10.0, 100.0, 80.0]]),
        conf=np.asarray([0.8]),
        cls=np.asarray([2.0]),
    )
    result = SimpleNamespace(names={0: "leaf", 1: "needle", 2: "flower", 3: "inflorescence"}, boxes=boxes, orig_shape=(160, 200))
    detections = _detections_from_result(result)
    assert detections == [{"class_name": "flower", "confidence": 0.8, "bbox": [0.1, 0.0625, 0.5, 0.5]}]


def test_validator_accepts_stage1_rows_and_rejects_unknown_source(tmp_path: Path) -> None:
    rows = []
    for split in ("train", "val", "test"):
        for label in ("Simple", "Compound"):
            for index in range(2):
                path = _image(tmp_path / f"{split}-{label}-{index}.jpg")
                rows.append(
                    {
                        "crop_id": f"{split}-{label}-{index}",
                        "observation_id": f"{split}-{label}-obs-{index}",
                        "taxon_name": f"Genus{label} species{split}{index}",
                        "image": "https://example.invalid/x.jpg",
                        "crop_path": str(path),
                        "anatomy_class": "leaf",
                        "detector_confidence": 0.8,
                        "observation_split": split,
                        "target_label": label,
                        "detector_box_source": STAGE1_BOX_SOURCE,
                        "review_verdict": STAGE1_REVIEW_VERDICT,
                    }
                )
    report = validate_dataset(
        rows,
        LEAF_SIMPLE_VS_COMPOUND,
        require_labels=True,
        check_files=True,
        min_per_class_per_split=2,
    )
    assert report.ok, report.errors
    rows[0]["detector_box_source"] = "taxon"
    rows[1]["review_verdict"] = "curator_magic"
    report = validate_dataset(rows, LEAF_SIMPLE_VS_COMPOUND, require_labels=True, min_per_class_per_split=0)
    assert not report.ok
    assert any("unsupported detector_box_source" in error for error in report.errors)
    assert any("unsupported review_verdict" in error for error in report.errors)
