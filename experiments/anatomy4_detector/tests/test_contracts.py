# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import pytest

from experiments.anatomy4_detector.contracts import (
    CLASS_NAMES,
    ContractError,
    cxcywh_to_xyxy,
    iou_xyxy,
    observation_leaks,
    parse_label_line,
    parse_label_text,
    split_for_observation,
    xyxy_to_cxcywh,
)


def test_parse_label_line_reads_normalized_xyxy() -> None:
    box = parse_label_line("3 0.100000 0.200000 0.400000 0.900000")
    assert box is not None
    assert box.class_id == 3
    assert box.class_name == "inflorescence"
    assert box.as_tuple() == pytest.approx((0.1, 0.2, 0.4, 0.9))
    assert parse_label_line("   ") is None
    assert parse_label_text("0 0.1 0.1 0.2 0.2\n\n2 0.5 0.5 0.6 0.6\n")[1].class_id == 2


@pytest.mark.parametrize(
    "line",
    [
        "0 0.1 0.1 0.2",  # four fields
        "0 0.1 0.1 0.2 0.3 0.4",  # six fields
        "4 0.1 0.1 0.2 0.2",  # class out of range
        "-1 0.1 0.1 0.2 0.2",
        "0 0.5 0.1 0.2 0.2",  # x0 > x1
        "0 0.1 0.5 0.2 0.2",  # y0 > y1
        "0 -0.1 0.1 0.2 0.2",  # not normalized
        "0 0.1 0.1 1.2 0.2",
        "0 nan 0.1 0.2 0.2",
        "leaf 0.1 0.1 0.2 0.2",
    ],
)
def test_parse_label_line_rejects_invalid(line: str) -> None:
    with pytest.raises(ContractError):
        parse_label_line(line)


def test_cxcywh_round_trip() -> None:
    box = (0.1, 0.2, 0.4, 0.9)
    cx, cy, width, height = xyxy_to_cxcywh(box)
    assert (cx, cy, width, height) == pytest.approx((0.25, 0.55, 0.3, 0.7), abs=1e-6)
    assert cxcywh_to_xyxy((cx, cy, width, height)) == pytest.approx(box)


def test_iou_is_pixel_space_and_handles_disjoint() -> None:
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou_xyxy((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(0.3333, abs=1e-3)


def test_split_is_deterministic_and_covers_all_splits() -> None:
    first = split_for_observation("obs-1")
    assert first in {"train", "val", "test"}
    assert all(split_for_observation("obs-1") == first for _ in range(5))
    assert {split_for_observation(f"obs-{index}") for index in range(80)} == {
        "train",
        "val",
        "test",
    }


def test_observation_leaks_detects_cross_split_observation() -> None:
    leaks = observation_leaks(
        {"train": ["a", "b"], "val": ["c"], "test": ["b"]}
    )
    assert leaks == {"b": ["test", "train"]}
    assert observation_leaks({"train": ["a"], "val": ["b"], "test": ["c"]}) == {}


def test_class_order_is_the_contract() -> None:
    assert CLASS_NAMES == ("leaf", "needle", "flower", "inflorescence")
