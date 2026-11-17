# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.anatomy4_detector.contracts import split_for_observation
from experiments.anatomy4_detector.dataset import validate_dataset
from experiments.anatomy4_detector.materialize import (
    MaterializationError,
    materialize,
    proposal_xyxy,
)


def obs_in(split: str, prefix: str) -> str:
    for index in range(500):
        candidate = f"{prefix}-{index}"
        if split_for_observation(candidate) == split:
            return candidate
    raise AssertionError(f"no observation mapped to {split}")


def proposal(
    proposal_id: str,
    photo_id: str,
    observation_id: str,
    class_name: str,
    bbox: list[float],
    score: float = 0.5,
    *,
    detector: str = "yoloe-26s-seg.pt",
) -> dict:
    return {
        "proposal_id": proposal_id,
        "image_id": photo_id,
        "photo_id": photo_id,
        "observation_id": observation_id,
        "image_path": f"/out/images/{photo_id}.jpg",
        "class_name": class_name,
        "class_id": {"leaf": 0, "needle": 1, "flower": 2, "inflorescence": 3}[class_name],
        "prompt": class_name,
        "detector_label": class_name,
        "box_score": score,
        "mask_iou": None,
        "bbox": bbox,
        "detector_bbox": bbox,
        "polygon": [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ],
        "detector_box_area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        "status": "proposed",
        "created_at": "2026-11-12T17:42:05Z",
        "model": {
            "contract": "anatomy4-v1",
            "backend": "yoloe",
            "detector": detector,
            "imgsz": 1024,
        },
    }


def build_audit_inputs(root: Path) -> dict:
    """Representative fixture shaped exactly like the real audit export."""

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    obs = {
        "a": obs_in("train", "obs-a"),
        "b": obs_in("val", "obs-b"),
        "c": obs_in("test", "obs-c"),
        "d": obs_in("train", "obs-d"),
        "e": obs_in("test", "obs-e"),
        "f": obs_in("train", "obs-f"),
        "g": obs_in("val", "obs-g"),
    }
    photo_ids = ["a", "b", "c", "d", "e", "f", "g"]
    for photo_id in photo_ids:
        Image.new("RGB", (64, 48), (30, 40, 30)).save(images_dir / f"{photo_id}.jpg")

    proposals = [
        # a: one accepted leaf, one rejected flower, one unreviewed needle.
        proposal("p1", "a", obs["a"], "leaf", [0.1, 0.2, 0.3, 0.4], 0.51),
        proposal("p2", "a", obs["a"], "flower", [0.5, 0.5, 0.8, 0.9], 0.44),
        proposal("p3", "a", obs["a"], "needle", [0.6, 0.1, 0.7, 0.2], 0.22),
        # b: accepted needle.
        proposal("p4", "b", obs["b"], "needle", [0.4, 0.4, 0.6, 0.7], 0.61),
        # c: accepted inflorescence.
        proposal("p5", "c", obs["c"], "inflorescence", [0.2, 0.1, 0.4, 0.3], 0.71),
        # d: every proposal reviewed and rejected -> reviewed negative.
        proposal("p6", "d", obs["d"], "leaf", [0.2, 0.2, 0.5, 0.5], 0.31),
        proposal("p7", "d", obs["d"], "flower", [0.1, 0.6, 0.4, 0.9], 0.29),
        # e: partially reviewed, nothing accepted -> NOT a negative.
        proposal("p8", "e", obs["e"], "leaf", [0.3, 0.3, 0.5, 0.6], 0.35),
        proposal("p9", "e", obs["e"], "flower", [0.6, 0.6, 0.9, 0.9], 0.25),
        # g: has proposals but never reviewed at all -> excluded.
        proposal("p10", "g", obs["g"], "leaf", [0.1, 0.1, 0.9, 0.9], 0.99),
    ]
    image_records = [
        {
            "image_id": photo_id,
            "observation_id": obs[photo_id],
            "image_path": str(images_dir / f"{photo_id}.jpg"),
            "original_size": [64, 48],
            "download": {"path": str(images_dir / f"{photo_id}.jpg")},
            "proposals": [],
        }
        for photo_id in photo_ids
    ]
    proposals_path = root / "prelabel" / "proposals.json"
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    proposals_path.write_text(
        json.dumps(
            {
                "schema": "arq.anatomy4.prelabel/v1",
                "contract": "anatomy4-v1",
                "provenance": {"backend": "yoloe", "detector": "yoloe-26s-seg.pt"},
                "images": image_records,
                "proposals": proposals,
            },
            indent=1,
        )
    )
    manifest_path = root / "manifest.downloaded.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "arq.anatomy4.manifest/v1",
                "contract": "anatomy4-v1",
                "observations": [],
                "images": image_records,
            },
            indent=1,
        )
    )
    rows = [
        ("p1", "leaf", "tp", "clean leaf"),
        ("p2", "flower", "fp", "petal of same plant"),
        ("p4", "needle", "tp", ""),
        ("p5", "inflorescence", "tp", ""),
        ("p6", "leaf", "fp", "moss on rock"),
        ("p7", "flower", "fp", ""),
        ("p8", "leaf", "fp", ""),
        # p3, p9, p10 intentionally absent (unreviewed)
    ]
    review_path = root / "review.csv"
    with review_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["tile", "proposal_id", "class_name", "box_score", "photo_id", "verdict", "notes"]
        )
        for index, (proposal_id, class_name, verdict, notes) in enumerate(rows):
            writer.writerow([index, proposal_id, class_name, "0.5", "", verdict, notes])
    return {
        "proposals": proposals_path,
        "review": review_path,
        "manifest": manifest_path,
        "images_dir": images_dir,
        "obs": obs,
    }


def test_materialize_maps_accepted_boxes_and_reviewed_negatives(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    out = tmp_path / "dataset"
    result = materialize(
        proposals_path=inputs["proposals"],
        review_path=inputs["review"],
        manifest_path=inputs["manifest"],
        images_dir=inputs["images_dir"],
        out_dir=out,
    )
    counts = result["counts"]
    assert result["status"] == "ok", result["errors"]
    assert result["split_isolation"] == "clean"
    assert counts["images"] == 4  # a, b, c, d (f has no proposals and is not explicit)
    assert counts["positives"] == 3
    assert counts["negatives"] == 1  # reviewed-rejected d
    assert counts["boxes"] == 3
    assert counts["boxes_per_class"] == {
        "leaf": 1,
        "needle": 1,
        "flower": 0,
        "inflorescence": 1,
    }
    # Unreviewed/rejected proposals never become truth.
    assert counts["dropped"]["images_without_accepted_truth"] == 2  # e, g
    assert result["partial_labeling"]["images_with_unreviewed_proposals"] == 1
    assert result["partial_labeling"]["unreviewed_proposals_on_labeled_images"] == 1

    dataset = json.loads((out / "dataset.json").read_text())
    by_photo = {entry["photo_id"]: entry for entry in dataset["images"]}
    assert by_photo["a"]["negative"] is False
    assert by_photo["d"]["negative"] is True
    assert by_photo["d"]["review"]["negative_basis"] == "reviewed_all_rejected"
    assert "e" not in by_photo and "g" not in by_photo

    # Reviewed negatives have explicit empty label files.
    train_labels = out / "labels" / "train"
    assert (train_labels / "d.txt").is_file()
    assert (train_labels / "d.txt").read_text() == ""
    assert (train_labels / "a.txt").read_text().strip().startswith("0 ")

    split = split_for_observation(inputs["obs"]["a"])
    label = (out / "labels" / split / "a.txt").read_text().strip()
    fields = [float(value) for value in label.split()]
    # bbox [0.1, 0.2, 0.3, 0.4] copied as xyxy exactly once.
    assert fields == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])

    provenance = [
        json.loads(line)
        for line in (out / "provenance.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert {row["proposal_id"] for row in provenance} == {"p1", "p4", "p5"}
    leaf = next(row for row in provenance if row["proposal_id"] == "p1")
    assert leaf["box_source"] == "bbox"
    assert leaf["box_score"] == 0.51
    assert leaf["model"]["detector"] == "yoloe-26s-seg.pt"
    assert leaf["review"]["verdict"] == "tp"
    assert leaf["review"]["notes"] == "clean leaf"
    assert leaf["xyxy_norm"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    data_yaml = (out / "data.yaml").read_text()
    assert '"leaf"' in data_yaml and '"inflorescence"' in data_yaml
    assert (out / "materialization.json").is_file()
    assert (out / "dataset-audit.json").is_file()
    audit = validate_dataset(out)
    assert audit["status"] == "ok"


def test_explicit_negative_images_including_no_proposals(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    manifest = json.loads(inputs["manifest"].read_text())
    # Add an image with zero proposals, exactly like the real audit's empty images.
    empty_id = "f"
    assert empty_id in {record["image_id"] for record in manifest["images"]}
    out = tmp_path / "dataset"
    result = materialize(
        proposals_path=inputs["proposals"],
        review_path=inputs["review"],
        manifest_path=inputs["manifest"],
        images_dir=inputs["images_dir"],
        out_dir=out,
        negative_images=["f"],
    )
    assert result["counts"]["images"] == 5
    assert result["counts"]["negatives"] == 2  # d reviewed + f explicit
    dataset = json.loads((out / "dataset.json").read_text())
    by_photo = {entry["photo_id"]: entry for entry in dataset["images"]}
    assert by_photo["f"]["negative"] is True
    assert by_photo["f"]["review"]["negative_basis"] == "explicit"
    split = split_for_observation(inputs["obs"]["f"])
    assert (out / "labels" / split / "f.txt").read_text() == ""


def test_class_filter_retains_only_relevant_reviewed_negatives(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    out = tmp_path / "dataset"
    result = materialize(
        proposals_path=inputs["proposals"],
        review_path=inputs["review"],
        manifest_path=inputs["manifest"],
        images_dir=inputs["images_dir"],
        out_dir=out,
        classes=["leaf"],
        require_test=False,
        require_val=False,
    )
    assert result["counts"]["boxes_per_class"]["leaf"] == 1
    assert result["counts"]["boxes"] == 1
    # a (leaf tp), d (leaf reviewed+rejected) remain; b/c have no selected class.
    dataset = json.loads((out / "dataset.json").read_text())
    photos = {entry["photo_id"] for entry in dataset["images"]}
    assert photos == {"a", "d"}
    assert result["counts"]["positives"] == 1
    assert result["counts"]["negatives"] == 1


def test_unreviewed_is_not_a_negative(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    out = tmp_path / "dataset"
    materialize(
        proposals_path=inputs["proposals"],
        review_path=inputs["review"],
        manifest_path=inputs["manifest"],
        images_dir=inputs["images_dir"],
        out_dir=out,
        negative_policy="explicit",
    )
    photos = {
        entry["photo_id"]
        for entry in json.loads((out / "dataset.json").read_text())["images"]
    }
    assert photos == {"a", "b", "c"}


def test_missing_accepted_image_fails(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    (inputs["images_dir"] / "a.jpg").unlink()
    with pytest.raises(MaterializationError, match="missing"):
        materialize(
            proposals_path=inputs["proposals"],
            review_path=inputs["review"],
            manifest_path=inputs["manifest"],
            images_dir=inputs["images_dir"],
            out_dir=tmp_path / "dataset",
        )


def test_explicit_negative_with_accepted_box_fails(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    with pytest.raises(MaterializationError, match="explicit negative"):
        materialize(
            proposals_path=inputs["proposals"],
            review_path=inputs["review"],
            manifest_path=inputs["manifest"],
            images_dir=inputs["images_dir"],
            out_dir=tmp_path / "dataset",
            negative_images=["a"],
        )


def test_unknown_review_ids_are_reported_not_fatal(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    with inputs["review"].open("a", newline="") as handle:
        handle.write("99,ghost-proposal,leaf,0.5,ghost,tp,\n")
    out = tmp_path / "dataset"
    result = materialize(
        proposals_path=inputs["proposals"],
        review_path=inputs["review"],
        manifest_path=inputs["manifest"],
        images_dir=inputs["images_dir"],
        out_dir=out,
    )
    assert result["review"]["unknown_ids"] == 1
    assert result["review"]["unknown_ids_sample"] == ["ghost-proposal"]


def test_review_without_overlap_fails(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    with inputs["review"].open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tile", "proposal_id", "class_name", "box_score", "photo_id", "verdict", "notes"])
        writer.writerow([0, "other-1", "leaf", "0.5", "a", "tp", ""])
    with pytest.raises(MaterializationError, match="share no proposal_id"):
        materialize(
            proposals_path=inputs["proposals"],
            review_path=inputs["review"],
            manifest_path=inputs["manifest"],
            images_dir=inputs["images_dir"],
            out_dir=tmp_path / "dataset",
        )


def test_conflicting_review_rows_are_flagged_and_last_wins(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    with inputs["review"].open("a", newline="") as handle:
        handle.write("98,p1,leaf,0.51,a,fp,second look\n")
    out = tmp_path / "dataset"
    result = materialize(
        proposals_path=inputs["proposals"],
        review_path=inputs["review"],
        manifest_path=inputs["manifest"],
        images_dir=inputs["images_dir"],
        out_dir=out,
    )
    assert result["review"]["conflicts"] == 1
    # Last verdict is fp, so image a has no accepted truth and is not fully
    # reviewed (p3 unreviewed), so it is excluded.
    dataset = json.loads((out / "dataset.json").read_text())
    assert "a" not in {entry["photo_id"] for entry in dataset["images"]}


def test_cxcywh_style_box_is_rejected_not_converted() -> None:
    with pytest.raises(MaterializationError, match="normalized xyxy"):
        proposal_xyxy({"proposal_id": "x", "bbox": [0.5, 0.5, 0.2, 0.2]})


def test_background_review_marks_empty_and_protects_missed(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    background = tmp_path / "audit" / "review-background.csv"
    background.write_text(
        "tile,photo_id,audit_bucket,verdict,notes\n"
        "0,f,background,empty_correct,\n"
        "1,d,background,missed_organ,leaves present\n"
        "2,ghost,background,empty_correct,from another run\n"
    )
    out = tmp_path / "dataset"
    result = materialize(
        proposals_path=inputs["proposals"],
        review_path=inputs["review"],
        manifest_path=inputs["manifest"],
        images_dir=inputs["images_dir"],
        out_dir=out,
        background_review_path=background,
    )
    assert result["counts"]["images"] == 4  # a, b, c, f
    assert result["counts"]["negatives"] == 1  # f; d is protected by missed_organ
    assert result["counts"]["dropped"]["background_missed_organ"] == 1
    assert result["inputs"]["background_review"]["unknown_ids"] == 1
    dataset = json.loads((out / "dataset.json").read_text())
    by_photo = {entry["photo_id"]: entry for entry in dataset["images"]}
    assert "d" not in by_photo
    assert by_photo["f"]["negative"] is True
    assert by_photo["f"]["review"]["negative_basis"] == "background_empty"


def test_background_empty_contradicting_accepted_boxes_fails(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    background = tmp_path / "audit" / "review-background.csv"
    background.write_text(
        "tile,photo_id,audit_bucket,verdict,notes\n0,a,background,empty_correct,\n"
    )
    with pytest.raises(MaterializationError, match="empty_correct"):
        materialize(
            proposals_path=inputs["proposals"],
            review_path=inputs["review"],
            manifest_path=inputs["manifest"],
            images_dir=inputs["images_dir"],
            out_dir=tmp_path / "dataset",
            background_review_path=background,
        )


def test_multi_photo_observation_stays_in_one_split(tmp_path: Path) -> None:
    root = tmp_path / "audit"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    for photo_id in ("m1", "m2", "t1", "s1"):
        Image.new("RGB", (32, 32), (40, 50, 40)).save(images_dir / f"{photo_id}.jpg")
    obs = {
        "m1": obs_in("val", "obs-multi"),
        "m2": obs_in("val", "obs-multi"),
        "t1": obs_in("train", "obs-train"),
        "s1": obs_in("test", "obs-test"),
    }
    # m1 and m2 are two photos of the SAME observation.
    obs["m2"] = obs["m1"]
    proposals = [
        proposal("q1", "m1", obs["m1"], "leaf", [0.1, 0.1, 0.3, 0.3]),
        proposal("q2", "m2", obs["m2"], "leaf", [0.2, 0.2, 0.5, 0.5]),
        proposal("q3", "t1", obs["t1"], "needle", [0.1, 0.1, 0.2, 0.2]),
        proposal("q4", "s1", obs["s1"], "flower", [0.3, 0.3, 0.6, 0.6]),
    ]
    image_records = [
        {
            "image_id": photo_id,
            "observation_id": obs[photo_id],
            "image_path": str(images_dir / f"{photo_id}.jpg"),
            "download": {"path": str(images_dir / f"{photo_id}.jpg")},
        }
        for photo_id in ("m1", "m2", "t1", "s1")
    ]
    (root / "proposals.json").write_text(
        json.dumps(
            {
                "contract": "anatomy4-v1",
                "images": image_records,
                "proposals": proposals,
            }
        )
    )
    review = root / "review.csv"
    with review.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["proposal_id", "verdict"])
        for proposal_id in ("q1", "q2", "q3", "q4"):
            writer.writerow([proposal_id, "tp"])
    out = tmp_path / "dataset"
    result = materialize(
        proposals_path=root / "proposals.json",
        review_path=review,
        images_dir=images_dir,
        out_dir=out,
    )
    split = split_for_observation(obs["m1"])
    split_dir = out / "images" / split
    assert (split_dir / "m1.jpg").is_file()
    assert (split_dir / "m2.jpg").is_file()
    assert result["split_isolation"] == "clean"
    assert result["counts"]["splits"][split]["observations"] == 1
    assert result["counts"]["splits"][split]["images"] == 2


def test_exact_duplicate_boxes_are_deduped_and_recorded(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    doc = json.loads(inputs["proposals"].read_text())
    duplicate = dict(doc["proposals"][0])
    duplicate["proposal_id"] = "p1b"
    duplicate["box_score"] = 0.41
    doc["proposals"].append(duplicate)
    inputs["proposals"].write_text(json.dumps(doc))
    with inputs["review"].open("a", newline="") as handle:
        handle.write("97,p1b,leaf,0.41,a,tp,duplicate prompt\n")
    out = tmp_path / "dataset"
    result = materialize(
        proposals_path=inputs["proposals"],
        review_path=inputs["review"],
        manifest_path=inputs["manifest"],
        images_dir=inputs["images_dir"],
        out_dir=out,
    )
    assert result["counts"]["boxes"] == 3
    assert result["counts"]["deduped_exact_boxes"] == 1
    dropped = result["dedupe"]["dropped"][0]
    assert dropped["kept_proposal_id"] == "p1"  # higher score wins
    assert dropped["dropped_proposal_id"] == "p1b"
    # The emitted label has exactly one box for that image.
    split = split_for_observation(inputs["obs"]["a"])
    assert len((out / "labels" / split / "a.txt").read_text().strip().splitlines()) == 1


def test_keep_duplicate_boxes_disables_dedupe(tmp_path: Path) -> None:
    inputs = build_audit_inputs(tmp_path / "audit")
    doc = json.loads(inputs["proposals"].read_text())
    duplicate = dict(doc["proposals"][0])
    duplicate["proposal_id"] = "p1b"
    doc["proposals"].append(duplicate)
    inputs["proposals"].write_text(json.dumps(doc))
    with inputs["review"].open("a", newline="") as handle:
        handle.write("97,p1b,leaf,0.41,a,tp,\n")
    result = materialize(
        proposals_path=inputs["proposals"],
        review_path=inputs["review"],
        manifest_path=inputs["manifest"],
        images_dir=inputs["images_dir"],
        out_dir=tmp_path / "dataset",
        dedupe_exact=False,
    )
    assert result["counts"]["boxes"] == 4
    assert result["counts"]["deduped_exact_boxes"] == 0
