# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from experiments.anatomy4_detector.onnx_eval import (
    decode_detections,
    letterbox_rgb,
    nms_indices,
    orient_output,
)


def test_letterbox_pads_centered_and_tracks_meta() -> None:
    image = Image.new("RGB", (100, 50), (255, 0, 0))
    tensor, meta = letterbox_rgb(image, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert meta["orig_w"] == 100 and meta["orig_h"] == 50
    assert meta["new_w"] == 640 and meta["new_h"] == 320
    assert meta["pad_x"] == 0 and meta["pad_y"] == 160
    assert meta["scale_x"] == pytest.approx(6.4)
    assert meta["scale_y"] == pytest.approx(6.4)


def test_orient_output_accepts_both_layouts() -> None:
    raw = np.zeros((1, 8, 10), dtype=np.float32)
    assert orient_output(raw, 4).shape == (10, 8)
    raw_t = np.zeros((1, 10, 8), dtype=np.float32)
    assert orient_output(raw_t, 4).shape == (10, 8)


def test_decode_filters_nms_and_maps_to_original_pixels() -> None:
    # 640 letterbox of a 100x50 image (scale 6.4, pad_y 160).
    meta = {
        "orig_w": 100.0,
        "orig_h": 50.0,
        "input_size": 640.0,
        "pad_x": 0.0,
        "pad_y": 160.0,
        "scale_x": 6.4,
        "scale_y": 6.4,
    }
    raw = np.zeros((1, 8, 4), dtype=np.float32)
    # channels: cx cy w h leaf needle flower inflorescence
    raw[0, :4, 0] = [320, 320, 64, 64]
    raw[0, 4, 0] = 0.9  # leaf
    # duplicate overlapping leaf, should be NMS-suppressed
    raw[0, :4, 1] = [322, 322, 64, 64]
    raw[0, 4, 1] = 0.8
    # low-confidence flower, filtered at 0.25
    raw[0, :4, 2] = [100, 200, 32, 32]
    raw[0, 6, 2] = 0.2
    # second flower above threshold
    raw[0, :4, 3] = [200, 200, 32, 32]
    raw[0, 6, 3] = 0.7
    detections = decode_detections(raw, meta, ["leaf", "needle", "flower", "inflorescence"], threshold=0.25)
    assert [d["class"] for d in detections] == ["leaf", "flower"]
    leaf = detections[0]
    # cx 320 -> (320 - 32)/6.4 = 45, w 64/6.4 = 10 -> 45..55 px
    assert leaf["bbox_pixels"] == pytest.approx([45.0, 20.0, 55.0, 30.0], abs=0.02)
    assert leaf["bbox"] == pytest.approx([0.45, 0.4, 0.55, 0.6], abs=1e-3)


def test_nms_keeps_highest_score() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5], dtype=np.float32)
    assert nms_indices(boxes, scores, 0.5) == [1, 2]


def test_cross_stack_golden_vectors_match_arq_api_kernel() -> None:
    import json
    from pathlib import Path

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "yolo26_onnx_golden_vectors.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    meta = fixture["meta"]
    classes = fixture["classes"]
    expected = fixture["expected_detections"]

    img = Image.new("RGB", (int(meta["orig_w"]), int(meta["orig_h"])), (80, 120, 60))
    _tensor, computed_meta = letterbox_rgb(img, int(meta["input_size"]))
    for key, val in meta.items():
        assert computed_meta[key] == pytest.approx(val)

    anchors_rm = np.asarray(fixture["anchors_row_major"], dtype=np.float32)
    for raw in (anchors_rm.T[None, :, :], anchors_rm[None, :, :]):
        dets = decode_detections(
            raw,
            meta,
            classes,
            threshold=fixture["threshold"],
            iou_threshold=fixture["iou_threshold"],
        )
        assert len(dets) == len(expected)
        for det, exp in zip(dets, expected):
            assert det["class"] == exp["class"]
            assert det["class_id"] == exp["class_id"]
            assert det["confidence"] == pytest.approx(exp["confidence"], abs=1e-6)
            assert det["bbox"] == pytest.approx(exp["bbox"], abs=1e-6)
            assert det["bbox_pixels"] == pytest.approx(exp["bbox_pixels"], abs=1e-2)
