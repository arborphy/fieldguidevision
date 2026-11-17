# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from experiments.anatomy4_specialists.contracts import CROP_MANIFEST_SCHEMA, read_jsonl
from experiments.anatomy4_specialists.crops import (
    build_crops,
    crop_id_for,
    load_reviews,
    pad_box_pixels,
)


def _image(path: Path, size=(100, 80), color=(120, 140, 90)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _proposal(photo_id: str, proposal_id: str, **extra):
    base = {
        "proposal_id": proposal_id,
        "photo_id": photo_id,
        "observation_id": "obs-1",
        "class_name": "leaf",
        "status": "proposed",
        "box_score": 0.82,
        "detector_bbox": [0.2, 0.2, 0.6, 0.6],
    }
    base.update(extra)
    return base


def _manifest(image_path: Path, proposals):
    return {
        "images": [
            {
                "photo_id": "p1",
                "url": "https://example.invalid/p1.jpg",
                "observation_id": "obs-1",
                "taxon_name": "Quercus rubra",
                "download": {"path": str(image_path)},
            }
        ],
        "proposals": proposals,
    }


def test_pad_box_pixels_relative_and_clamped() -> None:
    assert pad_box_pixels((10, 10, 50, 50), 100, 100, padding_frac=0.5) == (0, 0, 70, 70)
    assert pad_box_pixels((0, 0, 20, 20), 100, 80, padding_frac=1.0) == (0, 0, 40, 40)
    # Absolute mode pads by a fraction of the shorter image side.
    assert pad_box_pixels((40, 30, 60, 50), 100, 80, padding_frac=0.1, padding_mode="absolute") == (32, 22, 68, 58)


def test_pad_box_pixels_grows_to_minimum() -> None:
    x0, y0, x1, y1 = pad_box_pixels((49, 39, 51, 41), 100, 80, padding_frac=0.0, min_crop_px=20)
    assert x1 - x0 >= 20
    assert y1 - y0 >= 20


def test_crop_id_is_deterministic() -> None:
    assert crop_id_for("p1", [0.2, 0.2, 0.6, 0.6], "leaf") == crop_id_for("p1", [0.2, 0.2, 0.6, 0.6], "leaf")
    assert crop_id_for("p1", [0.2, 0.2, 0.6, 0.6], "leaf") != crop_id_for("p1", [0.2, 0.2, 0.6, 0.6], "flower")


def test_build_crops_materializes_padded_crop_and_manifest(tmp_path: Path) -> None:
    image = _image(tmp_path / "source.jpg")
    manifest = _manifest(image, [_proposal("p1", "ok")])
    report = build_crops(
        manifest,
        tmp_path / "out",
        reviews={"ok": {"proposal_id": "ok", "verdict": "tp"}},
        padding_frac=0.25,
    )
    assert report.written == 1
    row = read_jsonl(tmp_path / "out" / "crop_manifest.jsonl")[0]
    assert row["schema"] == CROP_MANIFEST_SCHEMA
    assert row["observation_id"] == "obs-1"
    assert row["taxon_name"] == "Quercus rubra"
    assert row["image"] == "https://example.invalid/p1.jpg"
    assert row["anatomy_class"] == "leaf"
    assert row["detector_confidence"] == pytest.approx(0.82)
    assert row["observation_split"] in {"train", "val", "test"}
    assert row["detector_box_source"] == "detector"
    crop_path = Path(row["crop_path"])
    assert crop_path.is_file()
    with Image.open(crop_path) as crop:
        # Box is 40x32 px; 25% relative padding on each side -> 60x48.
        assert crop.size == (60, 48)


def test_build_crops_rejects_taxon_boxes_noboxes_and_unreviewed(tmp_path: Path) -> None:
    image = _image(tmp_path / "source.jpg")
    manifest = _manifest(
        image,
        [
            _proposal("p1", "taxon", box_source="taxon"),
            _proposal("p1", "nobox", detector_bbox=None),
            _proposal("p1", "unreviewed"),
        ],
    )
    report = build_crops(
        manifest,
        tmp_path / "out",
        reviews={"taxon": {"verdict": "tp"}, "nobox": {"verdict": "tp"}},
    )
    assert report.written == 0
    assert report.skip_reasons["taxon_derived_box"] == 1
    assert report.skip_reasons["missing_detector_box"] == 1
    assert report.skip_reasons["review:missing"] == 1


def test_build_crops_allows_unreviewed_only_on_request(tmp_path: Path) -> None:
    image = _image(tmp_path / "source.jpg")
    manifest = _manifest(image, [_proposal("p1", "unreviewed")])
    assert build_crops(manifest, tmp_path / "a").written == 0
    assert (
        build_crops(manifest, tmp_path / "b", allow_unreviewed=True).written == 1
    )


def test_rejected_and_non_proposed_are_skipped(tmp_path: Path) -> None:
    image = _image(tmp_path / "source.jpg")
    manifest = _manifest(
        image,
        [
            _proposal("p1", "fp"),
            _proposal("p1", "maskfail", status="mask_failed"),
        ],
    )
    report = build_crops(
        manifest,
        tmp_path / "out",
        reviews={"fp": {"verdict": "fp"}, "maskfail": {"verdict": "tp"}},
    )
    assert report.written == 0
    assert report.skip_reasons["review:fp"] == 1
    assert report.skip_reasons["proposal_status_not_proposed"] == 1


def test_coverage_floor_is_a_skip_reason(tmp_path: Path) -> None:
    image = _image(tmp_path / "source.jpg")
    manifest = _manifest(image, [_proposal("p1", "low", box_score=0.1)])
    report = build_crops(
        manifest,
        tmp_path / "out",
        reviews={"low": {"verdict": "tp"}},
        min_detector_confidence=0.5,
    )
    assert report.skip_reasons["below_detector_confidence_floor"] == 1


def test_build_crops_uses_observation_lookup_for_taxon_and_url(tmp_path: Path) -> None:
    image = _image(tmp_path / "source.jpg")
    manifest = _manifest(image, [_proposal("p1", "ok")])
    manifest["images"][0].pop("taxon_name")
    manifest["images"][0].pop("url")
    lookup = {
        "taxon_by_observation": {"obs-1": "Forsythia xintermedia"},
        "url_by_photo": {"p1": "https://example.invalid/looked-up.jpg"},
    }
    report = build_crops(
        manifest,
        tmp_path / "out",
        reviews={"ok": {"verdict": "tp"}},
        observations=lookup,
    )
    assert report.written == 1
    row = read_jsonl(tmp_path / "out" / "crop_manifest.jsonl")[0]
    assert row["taxon_name"] == "Forsythia xintermedia"
    assert row["image"] == "https://example.invalid/looked-up.jpg"


def test_build_crops_skips_rows_without_taxon_or_url(tmp_path: Path) -> None:
    image = _image(tmp_path / "source.jpg")
    manifest = _manifest(image, [_proposal("p1", "no-taxon"), _proposal("p1", "no-url")])
    manifest["images"][0].pop("taxon_name")
    manifest["images"][0].pop("url")
    report = build_crops(
        manifest,
        tmp_path / "out",
        reviews={"no-taxon": {"verdict": "tp"}, "no-url": {"verdict": "tp"}},
        observations={"taxon_by_observation": {"obs-1": "Acer rubrum"}},
    )
    assert report.written == 0
    assert report.skip_reasons["missing_image_url"] == 2


def test_load_observation_lookup_from_manifest_and_jsonl(tmp_path: Path) -> None:
    from experiments.anatomy4_specialists.crops import load_observation_lookup

    manifest = {
        "observations": [{"observation_id": "obs-1", "taxon_name": "Acer rubrum"}],
        "images": [{"photo_id": "p1", "url": "https://example.invalid/p1.jpg"}],
    }
    lookup = load_observation_lookup(manifest)
    assert lookup["taxon_by_observation"] == {"obs-1": "Acer rubrum"}
    assert lookup["url_by_photo"] == {"p1": "https://example.invalid/p1.jpg"}
    path = tmp_path / "obs.jsonl"
    path.write_text(
        '{"observation_id": "obs-2", "taxon_name": "Quercus rubra"}\n'
        '{"photo_id": "p2", "url": "https://example.invalid/p2.jpg"}\n'
    )
    lookup = load_observation_lookup(path)
    assert lookup["taxon_by_observation"]["obs-2"] == "Quercus rubra"
    assert lookup["url_by_photo"]["p2"] == "https://example.invalid/p2.jpg"


def test_load_reviews_csv(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    path.write_text("proposal_id,verdict\nok,tp\nbad,fp\n")
    reviews = load_reviews(path)
    assert reviews["ok"]["verdict"] == "tp"
    assert reviews["bad"]["verdict"] == "fp"
