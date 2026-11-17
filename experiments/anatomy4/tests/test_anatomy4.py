# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.anatomy4.contracts import (
    map_detector_label,
    polygon_area,
    polygon_to_bbox,
    split_for_observation,
    xyxy_to_yolo,
)
from experiments.anatomy4.labels import write_yolo_dataset
from experiments.anatomy4.manifest import (
    ImageRecord,
    ObservationRecord,
    build_targeted_manifest,
    parse_observations,
    select_sample,
    select_targeted_sample,
)


def test_xyxy_to_yolo_normalizes_and_orders() -> None:
    cx, cy, width, height = xyxy_to_yolo((100, 50, 300, 250), 400, 500)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.3)
    assert width == pytest.approx(0.5)
    assert height == pytest.approx(0.4)
    # Flipped corners are normalized, not trusted.
    assert xyxy_to_yolo((300, 250, 100, 50), 400, 500) == pytest.approx(
        (0.5, 0.3, 0.5, 0.4)
    )


def test_bbox_is_clipped_to_frame_and_polygon_bbox_is_tight() -> None:
    bbox = polygon_to_bbox([(-0.1, 0.2), (0.4, 0.9), (0.8, -0.2)])
    assert bbox == pytest.approx((0.0, 0.0, 0.8, 0.9))
    with pytest.raises(ValueError):
        polygon_to_bbox([])
    assert polygon_area([(0, 0), (1, 0), (1, 1), (0, 1)]) == pytest.approx(1.0)


def test_class_mapping_is_conservative() -> None:
    leaf = ("leaf", "leaves")
    flower = ("flower", "blossom")
    inflorescence = ("inflorescence", "flower cluster", "flower head")
    needle = ("needle", "conifer needle", "pine needle")

    assert map_detector_label("leaf", leaf) == ("leaf", "mapped")
    assert map_detector_label("green leaves", leaf) == ("leaf", "mapped")
    assert map_detector_label("flower cluster", inflorescence) == ("inflorescence", "mapped")
    assert map_detector_label("conifer needle", needle) == ("needle", "mapped")
    # A label from another class is not stolen.
    assert map_detector_label("flower", leaf) == (None, "unmapped")
    # Longest phrase wins: "flower cluster" is inflorescence, not flower.
    assert map_detector_label("flower cluster", flower) == (None, "unmapped")
    assert map_detector_label("flower cluster", inflorescence) == (
        "inflorescence",
        "mapped",
    )
    assert map_detector_label("leaf and flower", leaf)[0] is None
    assert map_detector_label("mushroom", leaf) == (None, "unmapped")


def test_split_is_deterministic_and_observation_level() -> None:
    first = split_for_observation("inat-123")
    assert first in {"train", "val", "test"}
    assert all(split_for_observation("inat-123") == first for _ in range(5))
    # Different observations usually differ, but the property we need is that
    # every image shares its observation's split, which write_yolo_dataset
    # enforces by keying on observation_id.
    splits = {split_for_observation(f"obs-{i}") for i in range(50)}
    assert splits == {"train", "val", "test"}


def _observation(obs_id: str, taxon: str, month: str, iconic: str = "Plantae") -> ObservationRecord:
    return ObservationRecord(
        observation_id=obs_id,
        inat_id=obs_id.split("-")[-1],
        taxon_name=taxon,
        taxon_rank="species",
        common_name=None,
        observed_on=f"2024-{month}-15",
        quality_grade="research",
        iconic_taxon_name=iconic,
        images=[
            ImageRecord(
                photo_id=f"p{obs_id}",
                url="https://example.invalid/p.jpg",
                is_cover=True,
            )
        ],
    )


def test_sampling_is_deterministic_and_marks_one_cover_image() -> None:
    records = [
        _observation(f"obs-{i}", f"Taxon {i}", ["03", "06", "09", "12"][i % 4])
        for i in range(60)
    ] + [
        _observation(f"fungus-{i}", f"Fungus {i}", "07", iconic="Fungi")
        for i in range(20)
    ]
    first = select_sample(records, sample_size=24, seed="test")
    second = select_sample(records, sample_size=24, seed="test")
    assert [r.observation_id for r in first] == [r.observation_id for r in second]
    assert len(first) == 24
    assert len({r.observation_id for r in first}) == 24
    assert all(sum(image.selected_for_audit for image in r.images) == 1 for r in first)


def test_payload_parsing_keeps_photo_license_and_cover() -> None:
    payload = {
        "photos": [
            {
                "id": 111,
                "url": "https://example.invalid/photos/111/square.jpg",
                "license_code": "cc-by-nc",
                "attribution": "A Person, some rights reserved",
                "original_dimensions": {"width": 1024, "height": 768},
            },
            {"id": 222, "url": "https://example.invalid/photos/222/square.jpg"},
        ]
    }
    rows = [
        {
            "arborphy_id": "inat-1",
            "inat_id": "1",
            "taxon_name": "Rhododendron maximum",
            "taxon_rank": "species",
            "common_name": "Great Laurel",
            "observed_on": "2024-06-01",
            "quality_grade": "research",
            "iconic_taxon_name": "Plantae",
            "photo_url": "https://example.invalid/photos/111/square.jpg",
            "source_payload": json.dumps(payload),
        }
    ]
    records = parse_observations(rows)
    assert len(records) == 1
    images = records[0].images
    assert [image.photo_id for image in images] == ["111", "222"]
    assert images[0].is_cover and not images[1].is_cover
    assert images[0].license == "cc-by-nc"
    assert images[0].width == 1024


def _proposal(proposal_id: str, photo_id: str, obs_id: str, class_name: str = "leaf", status: str = "proposed") -> dict:
    return {
        "proposal_id": proposal_id,
        "photo_id": photo_id,
        "observation_id": obs_id,
        "class_name": class_name,
        "class_id": {"leaf": 0, "needle": 1, "flower": 2, "inflorescence": 3}[class_name],
        "status": status,
        "box_score": 0.5,
        "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]],
        "detector_bbox": [0.1, 0.1, 0.4, 0.5],
    }


def test_yolo_writer_consumes_accepted_only(tmp_path: Path) -> None:
    manifest = {
        "contract": "anatomy4-v1",
        "provenance": {"detector": "dino"},
        "images": [
            {"photo_id": "p1", "observation_id": "obs-1", "download": {"path": "/nonexistent/p1.jpg"}},
            {"photo_id": "p2", "observation_id": "obs-1", "download": {"path": "/nonexistent/p2.jpg"}},
            {"photo_id": "p3", "observation_id": "obs-1", "download": {"path": "/nonexistent/p3.jpg"}},
            {"photo_id": "p4", "observation_id": "obs-1", "download": {"path": "/nonexistent/p4.jpg"}},
            {"photo_id": "p5", "observation_id": "obs-1", "download": {"path": "/nonexistent/p5.jpg"}},
        ],
        "proposals": [
            _proposal("keep", "p1", "obs-1"),
            _proposal("reject", "p2", "obs-1"),
            _proposal("unreviewed", "p3", "obs-1"),
            _proposal("maskfail", "p4", "obs-1", status="mask_failed"),
            _proposal("keep2", "p1", "obs-1", class_name="flower"),
        ],
    }
    reviews = {
        "keep": {"proposal_id": "keep", "verdict": "tp"},
        "keep2": {"proposal_id": "keep2", "verdict": "tp"},
        "reject": {"proposal_id": "reject", "verdict": "fp"},
        "maskfail": {"proposal_id": "maskfail", "verdict": "tp"},
    }
    summary = write_yolo_dataset(manifest, tmp_path, reviews=reviews)
    assert summary["accepted_proposals"] == 2
    assert summary["accepted_by_class"] == {"leaf": 1, "flower": 1}
    label_file = next((tmp_path / "labels").rglob("p1.txt"))
    lines = label_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert {line.split()[0] for line in lines} == {"0", "2"}
    # p1 has two accepted boxes but is still one image; p2/p3/p4/p5 have none.
    assert summary["images_written"] == 1
    # Both boxes and both observation-level decisions use the same split.
    split = split_for_observation("obs-1")
    assert (tmp_path / "labels" / split / "p1.txt").is_file()
    assert summary["observations_by_split"] == {split: 1}


def _targeted_observation(
    obs_id: str,
    taxon: str,
    month: str,
    *,
    family: str | None = None,
    iconic: str = "Plantae",
    photos: int = 1,
) -> ObservationRecord:
    return ObservationRecord(
        observation_id=obs_id,
        inat_id=obs_id.split("-")[-1],
        taxon_name=taxon,
        taxon_rank="species",
        common_name=None,
        observed_on=f"2024-{month}-15",
        quality_grade="research",
        iconic_taxon_name=iconic,
        taxon_family=family,
        images=[
            ImageRecord(
                photo_id=f"p{obs_id}-{index}",
                url="https://example.invalid/p.jpg",
                is_cover=index == 0,
                position=index,
            )
            for index in range(photos)
        ],
    )


def test_targeted_sampling_assigns_disjoint_buckets_and_respects_photo_limit() -> None:
    records = [
        _targeted_observation(f"conifer-{i}", "Pinus strobus", "12", family="Pinaceae", photos=3)
        for i in range(20)
    ] + [
        _targeted_observation(f"flower-{i}", f"Flower {i}", "06")
        for i in range(20)
    ] + [
        _targeted_observation(f"leaf-{i}", f"Leaf {i}", "10")
        for i in range(20)
    ] + [
        _targeted_observation(f"animal-{i}", f"Animal {i}", "07", iconic="Animalia")
        for i in range(20)
    ]

    selected = select_targeted_sample(
        records,
        seed="targeted-test",
        needle_observations=8,
        photos_per_needle_observation=2,
        leaf_observations=8,
        flower_observations=8,
        background_observations=8,
    )
    buckets = [record.audit_bucket for record in selected]
    assert buckets.count("needle") == 8
    assert buckets.count("flower") == 8
    assert buckets.count("leaf") == 8
    assert buckets.count("background") == 8
    assert len({record.observation_id for record in selected}) == len(selected)
    for record in selected:
        images = [image for image in record.images if image.selected_for_audit]
        expected = 2 if record.audit_bucket == "needle" else 1
        assert len(images) == expected


def test_targeted_manifest_carries_taxonomy_metadata_only_for_selection() -> None:
    records = [
        _targeted_observation("conifer-1", "Tsuga canadensis", "12", family="Pinaceae", photos=2),
        _targeted_observation("leaf-1", "Acer rubrum", "07"),
        _targeted_observation("flower-1", "Rhododendron maximum", "06"),
        _targeted_observation("animal-1", "Odocoileus virginianus", "06", iconic="Animalia"),
    ]
    manifest = build_targeted_manifest(
        records=records,
        needle_observations=1,
        photos_per_needle_observation=2,
        leaf_observations=1,
        flower_observations=1,
        background_observations=2,
    )
    assert manifest["mode"] == "targeted"
    assert manifest["source_counts"]["selected_images_for_audit"] >= 5
    conifer = next(o for o in manifest["observations"] if o["observation_id"] == "conifer-1")
    assert conifer["audit_bucket"] == "needle"
    assert conifer["taxon_family"] == "Pinaceae"
    assert all(
        image["audit_bucket"] == "needle"
        for image in manifest["images"]
        if image["observation_id"] == "conifer-1" and image["selected_for_audit"]
    )
