# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import json
from pathlib import Path

from experiments.anatomy4_detector.coarse_multi import build_coarse_multi


def _proposal(image_id, class_name, prompt, score, bbox):
    return {
        "proposal_id": f"{image_id}-{class_name}-{prompt}-{score}",
        "image_id": image_id,
        "observation_id": f"obs-{image_id}",
        "class_name": class_name,
        "prompt": prompt,
        "box_score": score,
        "bbox": bbox,
        "model": {"detector": "owlv2" if class_name != "leaf" else "yoloe", "backend": "owlv2" if class_name != "leaf" else None},
    }


def build_fixture(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    images = ["tc-p1", "tc-p2", "tc-p3", "tc-p4", "tc-p5", "tc-p6", "tc-c1", "tc-e1", "tc-n1", "tc-b1"]
    manifest = [
        {
            "image_id": image_id,
            "observation_id": f"obs-{image_id}",
            "image_path": f"/out/train-candidates/images/{image_id}.jpg",
            "photo_url": f"https://example.org/{image_id}.jpg",
        }
        for image_id in images
    ]
    proposals = [
        _proposal("tc-p1", "leaf", "leaf", 0.30, [0.1, 0.1, 0.3, 0.3]),
        _proposal("tc-p1", "flower", "single flower", 0.25, [0.5, 0.5, 0.7, 0.7]),
        _proposal("tc-p2", "inflorescence", "cluster of flowers", 0.29, [0.2, 0.2, 0.5, 0.5]),
        _proposal("tc-p2", "flower", "single flower", 0.25, [0.6, 0.1, 0.8, 0.3]),
        _proposal("tc-p3", "leaf", "leaf", 0.30, [0.1, 0.1, 0.3, 0.3]),
        *[
            _proposal("tc-p3", "leaf", "leaf", 0.10 + index * 0.01, [0.4, 0.4, 0.5, 0.5])
            for index in range(5)
        ],
        *[
            _proposal("tc-p4", "flower", "single flower", 0.30, [0.05 * i, 0.1, 0.05 * i + 0.04, 0.2])
            for i in range(9)
        ],
        _proposal("tc-p5", "leaf", "leaf", 0.30, [0.0, 0.0, 0.8, 0.8]),
        _proposal("tc-p6", "inflorescence", "cluster of flowers", 0.27, [0.2, 0.2, 0.5, 0.5]),
        _proposal("tc-c1", "leaf", "leaf", 0.40, [0.1, 0.1, 0.3, 0.3]),
        _proposal("tc-e1", "leaf", "leaf", 0.40, [0.1, 0.1, 0.3, 0.3]),
        _proposal("tc-n1", "leaf", "leaf", 0.40, [0.1, 0.1, 0.3, 0.3]),
        _proposal("tc-b1", "leaf", "leaf", 0.40, [0.1, 0.1, 0.3, 0.3]),
    ]
    proposals_path = root / "proposals.json"
    proposals_path.write_text(json.dumps({"proposals": proposals}))
    manifest_path = root / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest))

    curated = root / "curated-multi" / "leaf-b1" / "annotations.jsonl"
    curated.parent.mkdir(parents=True)
    curated.write_text(
        json.dumps(
            {
                "review_image_id": "tc-c1",
                "observation_id": "obs-c1",
                "image_path": "/out/train-candidates/images/tc-c1.jpg",
                "labels": [{"class": "leaf", "bbox": [0.1, 0.1, 0.3, 0.3]}],
            }
        )
        + "\n"
    )
    (curated.parent / "excluded.jsonl").write_text(
        json.dumps({"image_id": "tc-e1", "review_image_id": "tc-e1", "review_state": "excluded"}) + "\n"
    )
    needle = root / "curated-needle" / "batch-1" / "annotations.jsonl"
    needle.parent.mkdir(parents=True)
    needle.write_text(
        json.dumps({"review_image_id": "tc-n1", "observation_id": "obs-n1", "labels": []}) + "\n"
    )
    blind_manifest = root / "blind" / "manifest.jsonl"
    blind_manifest.parent.mkdir(parents=True)
    blind_manifest.write_text(
        json.dumps({"review_image_id": "tc-b1", "observation_id": "obs-b1", "split": "test"}) + "\n"
    )
    blind_annotations = root / "blind" / "annotations.jsonl"
    blind_annotations.write_text(
        json.dumps({"review_image_id": "tc-b1", "observation_id": "obs-b1", "split": "test"}) + "\n"
    )
    return {
        "proposals": proposals_path,
        "manifest": manifest_path,
        "curated_globs": [str(root / "curated-multi" / "*" / "annotations.jsonl")],
        "needle_globs": [str(root / "curated-needle" / "batch-*" / "annotations.jsonl")],
        "blind_manifest": blind_manifest,
        "blind_annotations": blind_annotations,
    }


def test_coarse_multi_filters_and_counts(tmp_path: Path) -> None:
    inputs = build_fixture(tmp_path / "src")
    out = tmp_path / "coarse" / "auto-plantae.jsonl"
    report = build_coarse_multi(
        proposals_path=inputs["proposals"],
        manifest_path=inputs["manifest"],
        curated_multi_globs=inputs["curated_globs"],
        curated_needle_globs=inputs["needle_globs"],
        blind_manifest_path=inputs["blind_manifest"],
        blind_annotations_path=inputs["blind_annotations"],
        out_path=out,
        report_path=tmp_path / "coarse-report.json",
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {row["image_id"] for row in rows} == {"tc-p1", "tc-p2"}
    assert report["boxes"]["per_class"] == {
        "leaf": 1,
        "needle": 0,
        "flower": 2,
        "inflorescence": 1,
    }
    assert report["boxes"]["total"] == 4
    assert report["excluded"]["reasons"]["leaf_clutter"] == 1
    assert report["excluded"]["reasons"]["too_many_boxes"] == 1
    assert report["excluded"]["reasons"]["box_too_large"] == 1
    assert report["excluded"]["reasons"]["no_accepted_boxes"] == 1  # p6 cluster .27
    assert report["excluded_sets"]["curated_multi"] == 1
    assert report["excluded_sets"]["curated_multi_excluded"] == 1
    assert report["excluded_sets"]["curated_needle"] == 1
    assert report["excluded_sets"]["blind"] == 1

    row = next(item for item in rows if item["image_id"] == "tc-p1")
    assert row["source"] == "model"
    assert row["review_state"] == "accepted"
    assert row["labels"][0]["source"] == "model"
    assert row["labels"][0]["review_state"] == "accepted"
    assert row["labels"][0]["provenance"]["coarse_policy"] == "arq.anatomy4.coarse-multi/v1"
    assert row["provenance"]["low_conf_leaf_proposals"] == 1
    assert "single_flower" in row["provenance"]["thresholds"]
