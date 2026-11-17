# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from experiments.anatomy4_detector.blindtest import (
    BlindImportError,
    import_blind_test,
    load_benchmark,
)
from experiments.anatomy4_detector.contracts import CLASS_NAMES, TEST_ONLY_MARKER
from experiments.anatomy4_detector.dataset import (
    prepare_ultralytics,
    validate_dataset,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path: Path, color=(30, 40, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path, quality=95)
    return path


def _label(class_name: str, bbox: list[float]) -> dict:
    return {
        "class": class_name,
        "bbox": bbox,
        "review_state": "accepted",
        "source": "curator",
        "provenance": {
            "annotation_method": "independent_visual_review",
            "independent_of_detector": True,
            "reviewer": "test-reviewer",
        },
    }


def build_benchmark(root: Path) -> dict:
    images_root = root / "blind-test"
    specs = {
        "rv-a": ("batch-1", "inat-a", False, False, [
            _label("leaf", [0.10, 0.10, 0.40, 0.45]),
            _label("flower", [0.55, 0.20, 0.80, 0.50]),
        ]),
        "rv-b": ("batch-1", "inat-b", False, False, [
            _label("needle", [0.20, 0.30, 0.35, 0.70]),
        ]),
        "rv-c": ("batch-1", "inat-c", False, False, [
            _label("inflorescence", [0.30, 0.10, 0.70, 0.40]),
        ]),
        "rv-d": ("batch-1", "inat-d", True, False, []),
        "rv-e": ("batch-1", "inat-e", False, True, [
            _label("leaf", [0.05, 0.05, 0.40, 0.30]),
            _label("leaf", [0.60, 0.55, 0.95, 0.90]),
        ]),
        "rv-f": ("batch-1", "inat-a", False, False, [
            _label("leaf", [0.15, 0.55, 0.50, 0.90]),
        ]),
    }
    annotations = []
    manifest = []
    for image_id, (batch, observation, negative, needs_review, labels) in specs.items():
        path = _image(
            images_root / batch / "images" / f"{image_id}.jpg",
            color=(20 + len(image_id) * 3, 40, 30),
        )
        annotations.append(
            {
                "batch": batch,
                "labels": labels,
                "needs_review": needs_review,
                "needs_review_note": "marginal edge blossom" if needs_review else None,
                "negative": negative,
                "observation_id": observation,
                "photo_url": f"https://example.org/{image_id}.jpg",
                "provenance": {"annotation_method": "independent_visual_review"},
                "review_image_id": image_id,
                "review_state": "accepted",
                "source": "curator",
                "split": "test",
            }
        )
        manifest.append(
            {
                "batch": batch,
                "local_image_path": str(path),
                "observation_id": observation,
                "photo_url": f"https://example.org/{image_id}.jpg",
                "review_image_id": image_id,
                "sha256": _sha(path),
                "split": "test",
            }
        )
    ann_path = root / "annotations.jsonl"
    man_path = root / "manifest.jsonl"
    ann_path.write_text("".join(json.dumps(row) + "\n" for row in annotations))
    man_path.write_text("".join(json.dumps(row) + "\n" for row in manifest))

    boxes_per_class = {name: 0 for name in CLASS_NAMES}
    images_per_class = {name: 0 for name in CLASS_NAMES}
    for row in annotations:
        present = set()
        for label in row["labels"]:
            boxes_per_class[label["class"]] += 1
            present.add(label["class"])
        for name in present:
            images_per_class[name] += 1
    audit = {
        "schema": "arq.anatomy4.blind-audit/v1",
        "frozen": True,
        "frozen_annotations_sha256": _sha(ann_path),
        "frozen_manifest_sha256": _sha(man_path),
        "merged": {
            "rows": len(annotations),
            "boxes_total": sum(boxes_per_class.values()),
            "boxes_by_class": boxes_per_class,
            "images_by_class": images_per_class,
            "negative_rows": sum(1 for row in annotations if row["negative"]),
            "needs_review_rows": sum(1 for row in annotations if row["needs_review"]),
            "positive_rows": sum(1 for row in annotations if not row["negative"]),
            "observations": len({row["observation_id"] for row in annotations}),
        },
    }
    (root / "audit.json").write_text(json.dumps(audit, indent=1))
    return {
        "annotations": ann_path,
        "manifest": man_path,
        "images_root": images_root,
        "audit": root / "audit.json",
    }


def run_import(tmp_path: Path, **overrides) -> dict:
    inputs = build_benchmark(tmp_path / "bench")
    values = {
        "annotations_path": inputs["annotations"],
        "manifest_path": inputs["manifest"],
        "out_dir": tmp_path / "dataset",
        "images_root": inputs["images_root"],
        "overlay_count": 4,
    }
    values.update(overrides)
    return import_blind_test(**values)


def test_import_counts_exclusions_and_layout(tmp_path: Path) -> None:
    result = run_import(tmp_path)
    out = tmp_path / "dataset"
    assert result["status"] == "ok", result["blocking"]
    assert result["test_only"] is True
    assert result["source_counts"]["rows"] == 6
    assert result["source_counts"]["boxes"] == 7
    assert result["source_counts"]["negative_rows"] == 1
    assert result["source_counts"]["needs_review_rows"] == 1
    assert result["source_counts"]["boxes_per_class"] == {
        "leaf": 4,
        "needle": 1,
        "flower": 1,
        "inflorescence": 1,
    }
    assert result["included"] == {
        "images": 5,
        "positive_images": 4,
        "negative_images": 1,
        "boxes": 5,
        "boxes_per_class": {
            "leaf": 2,
            "needle": 1,
            "flower": 1,
            "inflorescence": 1,
        },
        "observations": 4,
    }
    assert result["excluded_needs_review"]["images"] == 1
    assert result["excluded_needs_review"]["boxes"] == 2
    assert result["excluded_needs_review"]["ids"] == ["rv-e"]

    assert (out / "images" / "test" / "rv-a.jpg").is_file()
    assert not (out / "images" / "train").exists()
    assert not (out / "images" / "val").exists()
    assert not (out / "labels" / "test" / "rv-e.txt").exists()
    assert (out / "labels" / "test" / "rv-d.txt").read_text() == ""
    assert (out / "labels" / "test" / "rv-a.txt").read_text().strip().count("\n") == 1
    excluded = [
        json.loads(line)
        for line in (out / "excluded" / "needs_review.jsonl").read_text().splitlines()
    ]
    assert len(excluded) == 1
    assert excluded[0]["included_in_eval"] is False
    assert len(excluded[0]["labels"]) == 2

    dataset = json.loads((out / "dataset.json").read_text())
    assert dataset["test_only"] is True
    assert dataset["excluded"]["needs_review"]["images"] == 1
    by_id = {entry["photo_id"]: entry for entry in dataset["images"]}
    assert "rv-e" not in by_id
    assert by_id["rv-a"]["observation_id"] == "inat-a"
    assert by_id["rv-d"]["negative"] is True
    assert {entry["observation_id"] for entry in dataset["images"]} == {
        "inat-a",
        "inat-b",
        "inat-c",
        "inat-d",
    }

    marker = json.loads((out / TEST_ONLY_MARKER).read_text())
    assert marker["test_only"] is True
    assert marker["allow_train"] is False
    assert marker["counts"]["included"]["boxes"] == 5
    assert marker["counts"]["excluded_needs_review"]["boxes"] == 2

    audit = json.loads((out / "dataset-audit.json").read_text())
    assert audit["status"] == "ok"
    assert audit["test_only"] is True
    assert audit["splits"]["test"]["images"] == 5
    assert audit["splits"]["test"]["boxes"] == 5
    assert audit["splits"]["test"]["negatives"] == 1
    assert "test_only_dataset" in {item["code"] for item in audit["warnings"]}

    for overlay in result["overlays"]:
        path = Path(overlay["overlay"])
        assert path.is_file()
        with Image.open(path) as opened:
            assert opened.size == (64, 48)
    assert result["observation_groups"] == {"inat-a": ["rv-a", "rv-f"]}


def test_boxes_are_preserved_exactly(tmp_path: Path) -> None:
    run_import(tmp_path)
    out = tmp_path / "dataset"
    lines = (out / "labels" / "test" / "rv-a.txt").read_text().splitlines()
    parsed = sorted(
        (int(parts[0]), *(float(value) for value in parts[1:]))
        for parts in (line.split() for line in lines)
    )
    expected = sorted(
        [
            (0, 0.10, 0.10, 0.40, 0.45),
            (2, 0.55, 0.20, 0.80, 0.50),
        ]
    )
    assert len(parsed) == len(expected)
    for got_row, exp_row in zip(parsed, expected):
        assert got_row == pytest.approx(exp_row, abs=1e-6)
    dataset = json.loads((out / "dataset.json").read_text())
    entry = next(item for item in dataset["images"] if item["photo_id"] == "rv-a")
    assert [(box["x0"], box["y0"], box["x1"], box["y1"]) for box in entry["boxes"]] == [
        (0.10, 0.10, 0.40, 0.45),
        (0.55, 0.20, 0.80, 0.50),
    ]


def test_include_needs_review_opt_in_keeps_flags(tmp_path: Path) -> None:
    result = run_import(tmp_path, include_needs_review=True)
    out = tmp_path / "dataset"
    assert result["included"]["images"] == 6
    assert result["included"]["boxes"] == 7
    assert result["excluded_needs_review"]["images"] == 0
    dataset = json.loads((out / "dataset.json").read_text())
    entry = next(item for item in dataset["images"] if item["photo_id"] == "rv-e")
    assert entry["needs_review"] is True
    assert entry["included_in_eval"] is True
    assert len(entry["boxes"]) == 2
    assert (out / "labels" / "test" / "rv-e.txt").read_text().strip().count("\n") == 1
    archived = [
        json.loads(line)
        for line in (out / "excluded" / "needs_review.jsonl").read_text().splitlines()
    ]
    assert len(archived) == 1
    assert archived[0]["included_in_eval"] is True


def test_prepare_refuses_test_only_but_eval_view_is_allowed(tmp_path: Path) -> None:
    run_import(tmp_path)
    out = tmp_path / "dataset"
    audit = validate_dataset(out)
    assert audit["test_only"] is True
    with pytest.raises(Exception, match="test-only"):
        prepare_ultralytics(audit, out / "prepared")
    prepared = prepare_ultralytics(
        audit, out / "prepared-eval", link_mode="copy", allow_test_only=True
    )
    assert prepared["test_only"] is True
    data_yaml = (out / "prepared-eval" / "data.yaml").read_text()
    assert "test: images/test" in data_yaml
    assert "train:" not in data_yaml
    assert "val:" not in data_yaml
    label = (
        out / "prepared-eval" / "labels" / "test" / "rv-a.txt"
    ).read_text().strip().splitlines()[0]
    fields = [float(value) for value in label.split()]
    assert fields[0] == 0.0
    assert fields[1:] == pytest.approx([0.25, 0.275, 0.30, 0.35])


def test_training_guard_rejects_the_prepared_test_only_view(tmp_path: Path) -> None:
    from experiments.anatomy4_detector.train import _assert_trainable_data

    run_import(tmp_path)
    out = tmp_path / "dataset"
    audit = validate_dataset(out)
    prepare_ultralytics(audit, out / "prepared-eval", allow_test_only=True)
    with pytest.raises(ValueError, match="test-only"):
        _assert_trainable_data(out / "prepared-eval" / "data.yaml")


def test_validate_rejects_train_images_in_a_test_only_dataset(tmp_path: Path) -> None:
    run_import(tmp_path)
    out = tmp_path / "dataset"
    shutil.copytree(out, tmp_path / "tampered")
    tampered = tmp_path / "tampered"
    (tampered / "images" / "train").mkdir(parents=True, exist_ok=True)
    shutil.copy2(tampered / "images" / "test" / "rv-a.jpg", tampered / "images" / "train" / "rv-a.jpg")
    (tampered / "labels" / "train").mkdir(parents=True, exist_ok=True)
    shutil.copy2(tampered / "labels" / "test" / "rv-a.txt", tampered / "labels" / "train" / "rv-a.txt")
    audit = validate_dataset(tampered)
    codes = {item["code"] for item in audit["errors"]}
    assert "test_only_has_train" in codes
    assert audit["blocking"] is True


def test_non_test_split_is_rejected(tmp_path: Path) -> None:
    inputs = build_benchmark(tmp_path / "bench")
    rows = [json.loads(line) for line in inputs["annotations"].read_text().splitlines()]
    rows[0]["split"] = "train"
    inputs["annotations"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(BlindImportError, match="is not 'test'"):
        load_benchmark(inputs["annotations"], inputs["manifest"], verify_frozen=False)


def test_frozen_audit_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    inputs = build_benchmark(tmp_path / "bench")
    inputs["annotations"].write_text(inputs["annotations"].read_text() + "\n")
    with pytest.raises(BlindImportError, match="does not match the frozen record"):
        load_benchmark(inputs["annotations"], inputs["manifest"])


def test_manifest_observation_mismatch_is_rejected(tmp_path: Path) -> None:
    inputs = build_benchmark(tmp_path / "bench")
    rows = [json.loads(line) for line in inputs["manifest"].read_text().splitlines()]
    rows[0]["observation_id"] = "inat-somewhere-else"
    inputs["manifest"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(BlindImportError, match="observation_id disagrees"):
        load_benchmark(inputs["annotations"], inputs["manifest"], verify_frozen=False)


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda row: row.update(negative=True), "negative row carries labels"),
        (lambda row: row.update(labels=[]), "no labels and not marked negative"),
        (
            lambda row: row["labels"][0].update(review_state="proposed"),
            "is not accepted/confirmed",
        ),
        (
            lambda row: row["labels"][0].update(bbox=[0.9, 0.9, 0.1, 0.1]),
            "not normalized xyxy",
        ),
        (
            lambda row: row["labels"][0].update({"class": "stem"}),
            "not in",
        ),
        (
            lambda row: row["labels"].append(dict(row["labels"][0])),
            "exact duplicate",
        ),
    ],
)
def test_invalid_annotation_rows_are_rejected(tmp_path: Path, mutate, needle: str) -> None:
    inputs = build_benchmark(tmp_path / "bench")
    rows = [json.loads(line) for line in inputs["annotations"].read_text().splitlines()]
    mutate(rows[0])
    inputs["annotations"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(BlindImportError, match=needle):
        load_benchmark(inputs["annotations"], inputs["manifest"], verify_frozen=False)


def test_image_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    inputs = build_benchmark(tmp_path / "bench")
    image = next((inputs["images_root"] / "batch-1" / "images").glob("rv-a.*"))
    Image.new("RGB", (64, 48), (255, 0, 0)).save(image)
    with pytest.raises(BlindImportError, match="sha256 mismatch"):
        import_blind_test(
            annotations_path=inputs["annotations"],
            manifest_path=inputs["manifest"],
            out_dir=tmp_path / "dataset",
            images_root=inputs["images_root"],
        )


def test_missing_annotation_for_manifest_row_is_rejected(tmp_path: Path) -> None:
    inputs = build_benchmark(tmp_path / "bench")
    rows = [json.loads(line) for line in inputs["annotations"].read_text().splitlines()]
    inputs["annotations"].write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    with pytest.raises(BlindImportError, match="no annotation row"):
        load_benchmark(inputs["annotations"], inputs["manifest"], verify_frozen=False)