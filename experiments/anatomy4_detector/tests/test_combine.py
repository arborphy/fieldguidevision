# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.anatomy4_detector.combine import (
    CombineError,
    _split_for,
    combine_sources,
)
from experiments.anatomy4_detector.dataset import validate_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path: Path, color) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path, quality=95)
    return path


def obs_in(split: str, prefix: str) -> str:
    for index in range(500):
        candidate = f"{prefix}-{index}"
        if _split_for(candidate, "anatomy4-train-v1", 0.85) == split:
            return candidate
    raise AssertionError(f"no observation in {split}")


def label(class_name: str, bbox, *, state="accepted", **extra):
    return {
        "class": class_name,
        "bbox": bbox,
        "review_state": state,
        "source": "curator",
        "provenance": {"annotation_method": "reviewed", **extra},
    }


def build_sources(root: Path) -> dict:
    images_root = root / "train-candidates"
    images = {}
    manifest = []
    specs = {
        "tc-t1a": ("t1", (30, 40, 30)),
        "tc-t1b": ("t1", (35, 45, 35)),
        "tc-t2": ("t2", (40, 50, 40)),
        "tc-t3": ("t3", (45, 55, 45)),
        "tc-t4": ("t4", (50, 60, 50)),
        "tc-v1": ("v1", (55, 65, 55)),
        "tc-v2": ("v2", (60, 70, 60)),
        "tc-v3": ("v3", (65, 75, 65)),
        "tc-x1": ("x1", (70, 80, 70)),
    }
    for image_id, (prefix, color) in specs.items():
        path = _image(images_root / "images" / f"{image_id}.jpg", color)
        images[image_id] = path
        observation = obs_in("train" if prefix.startswith("t") else "val", f"obs-{prefix}")
        manifest.append(
            {
                "image_id": image_id,
                "observation_id": observation,
                "local_image_path": str(path),
                "photo_url": f"https://example.org/{image_id}.jpg",
                "sha256": _sha(path),
            }
        )
    (images_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )
    obs = {row["image_id"]: row["observation_id"] for row in manifest}

    assisted = [
        {
            "image_id": "tc-t1a",
            "observation_id": obs["tc-t1a"],
            "review_state": "accepted",
            "policy": {"name": "assisted-v1", "auto_accept": True},
            "labels": [
                label("leaf", [0.10, 0.10, 0.30, 0.30]),
                label("flower", [0.50, 0.50, 0.70, 0.70]),
                label("needle", [0.20, 0.60, 0.30, 0.80]),
            ],
        },
        {
            "image_id": "tc-t1b",
            "observation_id": obs["tc-t1b"],
            "review_state": "accepted",
            "labels": [label("leaf", [0.20, 0.20, 0.40, 0.40])],
        },
        {
            "image_id": "tc-t3",
            "observation_id": obs["tc-t3"],
            "review_state": "accepted",
            "needs_review": True,
            "labels": [label("leaf", [0.30, 0.30, 0.50, 0.50])],
        },
        {
            "image_id": "tc-t4",
            "observation_id": obs["tc-t4"],
            "review_state": "proposed",
            "labels": [label("leaf", [0.40, 0.40, 0.60, 0.60], state="proposed")],
        },
        {
            "image_id": "tc-v1",
            "observation_id": obs["tc-v1"],
            "review_state": "accepted",
            "labels": [label("inflorescence", [0.15, 0.15, 0.45, 0.45])],
        },
    ]
    assisted_path = images_root / "verified" / "assisted-labels.jsonl"
    assisted_path.parent.mkdir(parents=True, exist_ok=True)
    assisted_path.write_text("".join(json.dumps(row) + "\n" for row in assisted))

    curated = [
        {
            "image_id": "tc-t1a",
            "observation_id": obs["tc-t1a"],
            "review_state": "accepted",
            "labels": [label("needle", [0.25, 0.60, 0.35, 0.80])],
        },
        {
            "review_image_id": "cn2-tc-v3",
            "image_path": str(images["tc-v3"]),
            "observation_id": obs["tc-v3"],
            "review_state": "accepted",
            "labels": [label("needle", [0.20, 0.20, 0.30, 0.40])],
        },
    ]
    curated_dir = images_root / "curated-needle"
    for batch in (1, 2):
        path = curated_dir / f"batch-{batch}" / "annotations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = curated if batch == 1 else []
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    negatives = [
        {
            "image_id": "tc-t1a",
            "observation_id": obs["tc-t1a"],
            "review_state": "accepted",
            "negative": True,
        },
        {
            "image_id": "tc-t2",
            "observation_id": obs["tc-t2"],
            "review_state": "accepted",
            "verdict": "empty_correct",
        },
        {
            "image_id": "tc-v2",
            "observation_id": obs["tc-v2"],
            "review_state": "accepted",
            "negative": True,
        },
        {
            "image_id": "tc-x1",
            "observation_id": obs["tc-x1"],
            "review_state": "accepted",
            "negative": False,
            "reviewer_note": "reviewed exclusion: no needle in frame",
        },
    ]
    negative_dir = images_root / "negative-review"
    negative_dir.mkdir(parents=True, exist_ok=True)
    for batch in (1, 2):
        path = negative_dir / f"batch-{batch}.jsonl"
        rows = negatives if batch == 1 else []
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    review_queue = [
        {
            "image_id": "tc-t1a",
            "observation_id": obs["tc-t1a"],
            "review_state": "accepted",
            "labels": [label("leaf", [0.10, 0.10, 0.30, 0.30])],
        }
    ]
    queue_path = images_root / "review-queue.jsonl"
    queue_path.write_text("".join(json.dumps(row) + "\n" for row in review_queue))

    blind_manifest = images_root / "blind" / "manifest.jsonl"
    blind_annotations = images_root / "blind" / "annotations.jsonl"
    blind_manifest.parent.mkdir(parents=True, exist_ok=True)
    blind_manifest.write_text(
        json.dumps(
            {
                "review_image_id": "rv-blind-1",
                "observation_id": "inat-blind-1",
                "photo_url": "https://example.org/blind.jpg",
                "sha256": "0" * 64,
                "split": "test",
            }
        )
        + "\n"
    )
    blind_annotations.write_text(
        json.dumps(
            {
                "review_image_id": "rv-blind-1",
                "observation_id": "inat-blind-1",
                "photo_url": "https://example.org/blind.jpg",
                "split": "test",
            }
        )
        + "\n"
    )
    return {
        "images_root": images_root,
        "manifest": images_root / "manifest.jsonl",
        "assisted": assisted_path,
        "curated": str(curated_dir / "batch-*" / "annotations.jsonl"),
        "negative": str(negative_dir / "batch-*.jsonl"),
        "review_queue": queue_path,
        "blind_manifest": blind_manifest,
        "blind_annotations": blind_annotations,
        "obs": obs,
    }


def run_combine(tmp_path: Path, **overrides) -> dict:
    inputs = build_sources(tmp_path / "src")
    values = {
        "out_dir": tmp_path / "dataset",
        "assisted": inputs["assisted"],
        "curated_needle_glob": inputs["curated"],
        "negative_glob": inputs["negative"],
        "review_queue": [inputs["review_queue"]],
        "candidate_manifest": inputs["manifest"],
        "images_root": inputs["images_root"],
        "blind_manifest": inputs["blind_manifest"],
        "blind_annotations": inputs["blind_annotations"],
        "overlay_count": 3,
    }
    values.update(overrides)
    return combine_sources(**values)


def test_mixed_sources_merge_counts_and_report(tmp_path: Path) -> None:
    report = run_combine(tmp_path)
    out = tmp_path / "dataset"
    assert report["status"] == "ok", report["blocking"]
    assert report["split_isolation"] == "clean"
    assert report["counts"]["images"] == 6
    assert report["counts"]["positives"] == 4
    assert report["counts"]["negatives"] == 2
    assert report["counts"]["boxes"] == 7
    assert report["counts"]["boxes_per_class"] == {
        "leaf": 2,
        "needle": 3,
        "flower": 1,
        "inflorescence": 1,
    }
    assert report["counts"]["splits"]["train"]["images"] == 3
    assert report["counts"]["splits"]["val"]["images"] == 3
    assert report["counts"]["splits"]["train"]["boxes"] == 5
    assert report["counts"]["splits"]["val"]["boxes"] == 2
    assert report["counts"]["splits"]["train"]["negatives"] == 1
    assert report["counts"]["splits"]["val"]["negatives"] == 1

    assert report["per_source"]["assisted"]["accepted_rows"] == 3
    assert report["per_source"]["assisted"]["boxes"] == 5
    assert report["per_source"]["curated-needle"]["boxes"] == 2
    assert report["per_source"]["negative-review"]["negatives"] == 3
    assert report["per_source"]["review-queue"]["boxes"] == 1

    assert report["dedupe"]["merged_boxes"] == 1
    assert report["conflicts"]["quarantined"] == 1
    assert report["excluded"]["needs_review"] == 1
    assert report["excluded"]["unreviewed"] == 1
    assert report["blind_check"]["matches"] == 0

    audit = json.loads((out / "dataset-audit.json").read_text())
    assert audit["status"] == "ok"
    assert audit["split_isolation"] == "clean"
    assert (out / "merge-report.json").is_file()
    assert (out / "merge-report.md").is_file()
    assert (out / "provenance.jsonl").is_file()
    data_yaml = (out / "data.yaml").read_text()
    assert "train: images/train" in data_yaml and "val: images/val" in data_yaml

    # A review-local id (cn2-tc-v3) canonicalizes to the candidate image id.
    dataset = json.loads((out / "dataset.json").read_text())
    assert "tc-v3" in {entry["photo_id"] for entry in dataset["images"]}
    assert not any(entry["photo_id"].startswith("cn2-") for entry in dataset["images"])

    # Multi-photo observation stays together, and no test rows are produced.
    split_a = _split_for(inputs_obs(report, "tc-t1a"), "anatomy4-train-v1", 0.85)
    assert (out / "images" / split_a / "tc-t1a.jpg").is_file()
    assert (out / "images" / split_a / "tc-t1b.jpg").is_file()
    assert not (out / "images" / "test").exists()


def inputs_obs(report: dict, photo_id: str) -> str:
    data = json.loads(
        (Path(report["audit"]).parent / "dataset.json").read_text()
    )
    return next(entry["observation_id"] for entry in data["images"] if entry["photo_id"] == photo_id)


def test_dedupe_preserves_all_sources(tmp_path: Path) -> None:
    report = run_combine(tmp_path)
    out = tmp_path / "dataset"
    provenance = [
        json.loads(line)
        for line in (out / "provenance.jsonl").read_text().splitlines()
        if line.strip()
    ]
    leaf = next(
        row
        for row in provenance
        if row["photo_id"] == "tc-t1a" and row["class_name"] == "leaf"
    )
    assert len(leaf["sources"]) == 2
    assert {source["source"] for source in leaf["sources"]} == {
        "assisted",
        "review-queue",
    }
    assert all(source["file"] for source in leaf["sources"])
    labels = (out / "labels" / _split_for(inputs_obs(report, "tc-t1a"), "anatomy4-train-v1", 0.85) / "tc-t1a.txt").read_text()
    assert len(labels.splitlines()) == 4  # leaf (once), flower, 2 needles


def test_conflicting_negative_is_quarantined_positive_wins(tmp_path: Path) -> None:
    report = run_combine(tmp_path)
    conflicts = [
        json.loads(line)
        for line in (tmp_path / "dataset" / "quarantine" / "conflicts.jsonl").read_text().splitlines()
    ]
    assert len(conflicts) == 1
    assert conflicts[0]["image_id"] == "tc-t1a"
    assert conflicts[0]["reason"] == "negative_for_positive_image"
    assert conflicts[0]["resolved"] == "positive_wins"
    dataset = json.loads((tmp_path / "dataset" / "dataset.json").read_text())
    entry = next(item for item in dataset["images"] if item["photo_id"] == "tc-t1a")
    assert entry["negative"] is False
    assert len(entry["boxes"]) == 4


def test_explicit_negatives_have_empty_label_files(tmp_path: Path) -> None:
    report = run_combine(tmp_path)
    out = tmp_path / "dataset"
    dataset = json.loads((out / "dataset.json").read_text())
    negatives = [entry for entry in dataset["images"] if entry["negative"]]
    assert {entry["photo_id"] for entry in negatives} == {"tc-t2", "tc-v2"}
    for entry in negatives:
        split = _split_for(entry["observation_id"], "anatomy4-train-v1", 0.85)
        assert (out / "labels" / split / f"{entry['photo_id']}.txt").read_text() == ""


def test_needs_review_and_unreviewed_are_excluded(tmp_path: Path) -> None:
    run_combine(tmp_path)
    out = tmp_path / "dataset"
    needs = (out / "quarantine" / "needs_review.jsonl").read_text().splitlines()
    unreviewed = (out / "quarantine" / "unreviewed.jsonl").read_text().splitlines()
    assert any('"tc-t3"' in line for line in needs)
    assert any('"tc-t4"' in line for line in unreviewed)
    dataset = json.loads((out / "dataset.json").read_text())
    photos = {entry["photo_id"] for entry in dataset["images"]}
    assert "tc-t3" not in photos
    assert "tc-t4" not in photos


def test_class_specific_source_selection(tmp_path: Path) -> None:
    report = run_combine(tmp_path, class_sources=["needle=curated-needle"])
    assert report["counts"]["boxes_per_class"] == {
        "leaf": 2,
        "needle": 2,
        "flower": 1,
        "inflorescence": 1,
    }
    assert report["counts"]["boxes"] == 6


def test_no_negatives_option(tmp_path: Path) -> None:
    report = run_combine(tmp_path, include_negatives=False)
    assert report["counts"]["negatives"] == 0
    assert report["counts"]["images"] == 4


def test_blind_leakage_aborts(tmp_path: Path) -> None:
    inputs = build_sources(tmp_path / "src")
    blind_manifest = inputs["images_root"] / "blind" / "manifest-leak.jsonl"
    blind_manifest.write_text(
        json.dumps(
            {
                "review_image_id": "rv-leak",
                "observation_id": inputs["obs"]["tc-t1a"],
                "photo_url": "https://example.org/leak.jpg",
                "sha256": "f" * 64,
                "split": "test",
            }
        )
        + "\n"
    )
    with pytest.raises(CombineError, match="blind-test leakage"):
        combine_sources(
            out_dir=tmp_path / "dataset",
            assisted=inputs["assisted"],
            curated_needle_glob=inputs["curated"],
            negative_glob=inputs["negative"],
            candidate_manifest=inputs["manifest"],
            images_root=inputs["images_root"],
            blind_manifest=blind_manifest,
            blind_annotations=inputs["blind_annotations"],
        )


def test_missing_blind_denylist_aborts_by_default(tmp_path: Path) -> None:
    inputs = build_sources(tmp_path / "src")
    with pytest.raises(CombineError, match="denylist is unavailable"):
        combine_sources(
            out_dir=tmp_path / "dataset",
            assisted=inputs["assisted"],
            curated_needle_glob=inputs["curated"],
            negative_glob=inputs["negative"],
            candidate_manifest=inputs["manifest"],
            images_root=inputs["images_root"],
            blind_manifest=tmp_path / "missing.jsonl",
            blind_annotations=tmp_path / "missing2.jsonl",
        )


def test_missing_sources_abort_until_allowed(tmp_path: Path) -> None:
    inputs = build_sources(tmp_path / "src")
    with pytest.raises(CombineError, match="expected sources are not ready"):
        combine_sources(
            out_dir=tmp_path / "dataset",
            assisted=inputs["assisted"],
            curated_needle_glob=None,
            negative_glob=inputs["negative"],
            candidate_manifest=inputs["manifest"],
            images_root=inputs["images_root"],
            blind_manifest=inputs["blind_manifest"],
            blind_annotations=inputs["blind_annotations"],
        )
    report = combine_sources(
        out_dir=tmp_path / "dataset2",
        assisted=inputs["assisted"],
        curated_needle_glob=None,
        negative_glob=inputs["negative"],
        candidate_manifest=inputs["manifest"],
        images_root=inputs["images_root"],
        blind_manifest=inputs["blind_manifest"],
        blind_annotations=inputs["blind_annotations"],
        allow_missing_sources=True,
    )
    assert report["counts"]["images"] == 5
    assert report["counts"]["boxes_per_class"]["needle"] == 1


def test_dataset_validates_and_prepares_for_training(tmp_path: Path) -> None:
    from experiments.anatomy4_detector.dataset import prepare_ultralytics

    run_combine(tmp_path)
    out = tmp_path / "dataset"
    audit = validate_dataset(out, require_test=False)
    assert audit["status"] == "ok"
    assert audit["test_only"] is False
    assert audit["split_isolation"] == "clean"
    prepared = prepare_ultralytics(audit, out / "prepared", link_mode="copy")
    assert prepared["splits"]["train"] == 3
    assert prepared["splits"]["val"] == 3
    assert "train: images/train" in (out / "prepared" / "data.yaml").read_text()


def test_overlays_are_written(tmp_path: Path) -> None:
    report = run_combine(tmp_path)
    assert len(report["overlays"]) == 3
    for overlay in report["overlays"]:
        path = Path(overlay["overlay"])
        assert path.is_file()
        with Image.open(path) as image:
            assert image.size == (64, 48)

def test_negative_review_nonnegative_rows_are_excluded_not_invalid(tmp_path: Path) -> None:
    report = run_combine(tmp_path)
    out = tmp_path / "dataset"
    assert report["excluded"]["non_negative"] == 1
    assert report["excluded"]["invalid"] == 0
    assert report["per_source"]["negative-review"]["excluded_non_negative"] == 1
    assert report["per_source"]["negative-review"]["rows"] == 4
    assert report["per_source"]["negative-review"]["negatives"] == 3
    # The reviewed exclusion is not an image, a negative, or a quarantine error.
    dataset = json.loads((out / "dataset.json").read_text())
    assert "tc-x1" not in {entry["photo_id"] for entry in dataset["images"]}
    assert not any(
        "tc-x1" in line
        for line in (out / "quarantine" / "invalid.jsonl").read_text().splitlines()
    )
    excluded = [
        json.loads(line)
        for line in (out / "excluded_non_negative.jsonl").read_text().splitlines()
    ]
    assert len(excluded) == 1
    assert excluded[0]["image_id"] == "tc-x1"
    assert excluded[0]["reason"] == "negative_review_row_negative_false"


def test_multiple_curated_globs_and_flat_assisted_rows(tmp_path: Path) -> None:
    from experiments.anatomy4_detector.combine import _image_id, _labels

    inputs = build_sources(tmp_path / "src")
    curated_dir = inputs["images_root"] / "curated-flower"
    curated_dir.mkdir(parents=True, exist_ok=True)
    extra = curated_dir / "annotations.jsonl"
    extra.write_text(
        json.dumps(
            {
                "image_id": "tc-t2",
                "observation_id": inputs["obs"]["tc-t2"],
                "review_state": "accepted",
                "labels": [label("flower", [0.10, 0.10, 0.40, 0.40])],
            }
        )
        + "\n"
    )
    report = combine_sources(
        out_dir=tmp_path / "dataset",
        assisted=inputs["assisted"],
        curated_needle_glob=[inputs["curated"]],
        curated_globs=[str(curated_dir / "*.jsonl")],
        negative_glob=inputs["negative"],
        review_queue=[inputs["review_queue"]],
        candidate_manifest=inputs["manifest"],
        images_root=inputs["images_root"],
        blind_manifest=inputs["blind_manifest"],
        blind_annotations=inputs["blind_annotations"],
    )
    assert report["per_source"]["curated"]["boxes"] == 1
    assert report["counts"]["boxes_per_class"]["flower"] == 2

    flat = {
        "class": "leaf",
        "bbox_xyxy_norm": [0.1, 0.1, 0.2, 0.2],
        "proposal_id": "a4y-tc-abcdef123456-leaf-leaf-000",
        "review_state": "accepted",
    }
    assert _labels(flat) == [flat]
    manifest = {"tc-abcdef123456": {"image_id": "tc-abcdef123456"}}
    assert _image_id(flat, manifest) == "tc-abcdef123456"


def test_one_negative_and_one_excluded_non_negative_row(tmp_path: Path) -> None:
    inputs = build_sources(tmp_path / "src")
    negative_dir = inputs["images_root"] / "negative-review"
    (negative_dir / "batch-1.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {
                    "image_id": "tc-t2",
                    "observation_id": inputs["obs"]["tc-t2"],
                    "review_state": "accepted",
                    "negative": True,
                },
                {
                    "image_id": "tc-x1",
                    "observation_id": inputs["obs"]["tc-x1"],
                    "review_state": "accepted",
                    "negative": False,
                },
            ]
        )
    )
    (negative_dir / "batch-2.jsonl").write_text("")
    report = combine_sources(
        out_dir=tmp_path / "dataset",
        assisted=inputs["assisted"],
        curated_needle_glob=inputs["curated"],
        negative_glob=str(negative_dir / "batch-*.jsonl"),
        review_queue=[inputs["review_queue"]],
        candidate_manifest=inputs["manifest"],
        images_root=inputs["images_root"],
        blind_manifest=inputs["blind_manifest"],
        blind_annotations=inputs["blind_annotations"],
    )
    assert report["counts"]["negatives"] == 1
    assert report["excluded"]["non_negative"] == 1
    assert report["excluded"]["invalid"] == 0
    assert "tc-x1" not in (
        tmp_path / "dataset" / "quarantine" / "unreviewed.jsonl"
    ).read_text()
    assert "tc-x1" not in (
        tmp_path / "dataset" / "quarantine" / "invalid.jsonl"
    ).read_text()
    assert report["per_source"]["negative-review"]["negatives"] == 1
    assert report["per_source"]["negative-review"]["excluded_non_negative"] == 1
    sidecar = (tmp_path / "dataset" / "excluded_non_negative.jsonl").read_text().splitlines()
    assert len(sidecar) == 1
    assert json.loads(sidecar[0])["image_id"] == "tc-x1"


def test_manifest_image_path_images_subdir_and_no_assisted(tmp_path: Path) -> None:
    root = tmp_path / "train-candidates"
    images = root / "images"
    images.mkdir(parents=True)
    specs = {
        "tc-c1": "train",
        "tc-c2": "train",
        "tc-v1": "val",
        "tc-v2": "val",
        "tc-n1": "train",
    }
    manifest = []
    for index, (image_id, split) in enumerate(specs.items()):
        path = _image(images / f"{image_id}.jpg", (30 + index, 50, 30))
        observation = obs_in(split, f"obs-{image_id}")
        manifest.append(
            {
                "image_id": image_id,
                "observation_id": observation,
                "image_path": f"/out/train-candidates/images/{image_id}.jpg",
                "photo_url": f"https://example.org/{image_id}.jpg",
                "sha256": _sha(path),
            }
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )
    obs = {row["image_id"]: row["observation_id"] for row in manifest}
    curated = root / "curated-needle" / "batch-1" / "annotations.jsonl"
    curated.parent.mkdir(parents=True)
    curated.write_text(
        "".join(
            json.dumps(
                {
                    # Review-local id, no inline image_path: resolution must use
                    # manifest image_path + <root>/images.
                    "review_image_id": f"cn2-{image_id}",
                    "observation_id": obs[image_id],
                    "review_state": "accepted",
                    "labels": [label("needle", [0.10, 0.10, 0.30, 0.30])],
                }
            )
            + "\n"
            for image_id in ("tc-c1", "tc-c2", "tc-v1", "tc-v2")
        )
    )
    negative = root / "negative-review" / "batch-1.jsonl"
    negative.parent.mkdir(parents=True)
    negative.write_text(
        json.dumps(
            {
                "review_image_id": "tc-n1",
                "observation_id": obs["tc-n1"],
                "review_state": "accepted",
                "negative": True,
            }
        )
        + "\n"
    )
    # An invalid assisted file exists but must never be opened in --no-assisted mode.
    assisted = root / "verified" / "assisted-labels.jsonl"
    assisted.parent.mkdir(parents=True)
    assisted.write_text("this is not json\n")
    blind_manifest = root / "blind" / "manifest.jsonl"
    blind_annotations = root / "blind" / "annotations.jsonl"
    blind_manifest.parent.mkdir(parents=True)
    blind_manifest.write_text(
        json.dumps(
            {
                "review_image_id": "rv-blind-x",
                "observation_id": "inat-blind-x",
                "split": "test",
            }
        )
        + "\n"
    )
    blind_annotations.write_text(
        json.dumps(
            {
                "review_image_id": "rv-blind-x",
                "observation_id": "inat-blind-x",
                "split": "test",
            }
        )
        + "\n"
    )

    report = combine_sources(
        out_dir=tmp_path / "dataset",
        assisted=None,
        require_assisted=False,
        curated_needle_glob=str(root / "curated-needle" / "batch-*" / "annotations.jsonl"),
        negative_glob=str(root / "negative-review" / "batch-*.jsonl"),
        candidate_manifest=root / "manifest.jsonl",
        images_root=root,
        blind_manifest=blind_manifest,
        blind_annotations=blind_annotations,
        allow_missing_sources=False,
    )
    assert report["status"] == "ok", report
    assert set(report["per_source"]) == {"curated-needle", "negative-review"}
    assert report["per_source"]["curated-needle"]["images"] == 4
    assert report["per_source"]["curated-needle"]["boxes"] == 4
    assert report["per_source"]["negative-review"]["negatives"] == 1
    assert report["excluded"]["invalid"] == 0
    assert report["counts"]["images"] == 5
    assert report["counts"]["negatives"] == 1
    out = tmp_path / "dataset"
    dataset = json.loads((out / "dataset.json").read_text())
    assert "tc-n1" in {entry["photo_id"] for entry in dataset["images"]}
    assert not any(
        "image not found" in line
        for line in (out / "quarantine" / "invalid.jsonl").read_text().splitlines()
    )
