# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from experiments.anatomy4_detector.contracts import CLASS_NAMES
from experiments.anatomy4_detector.export import (
    EXPORT_OPSET,
    build_export_kwargs,
    assert_export_kwargs,
    parse_names_metadata,
    verify_onnx,
)


def test_export_arguments_exactly_match_the_api_contract() -> None:
    kwargs = build_export_kwargs()
    assert kwargs == {
        "format": "onnx",
        "imgsz": 640,
        "batch": 1,
        "dynamic": False,
        "opset": 17,
        "simplify": False,
        "device": "cpu",
        "half": False,
    }
    # The load-bearing rule: nms must never be passed at all.
    assert "nms" not in kwargs
    assert EXPORT_OPSET == 17


@pytest.mark.parametrize(
    "override",
    [
        {"nms": False},
        {"nms": True},
        {"imgsz": 320},
        {"batch": 2},
        {"dynamic": True},
        {"opset": 19},
        {"simplify": True},
        {"device": "0"},
        {"half": True},
        {"format": "engine"},
    ],
)
def test_export_argument_violations_are_rejected(override: dict) -> None:
    kwargs = build_export_kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError):
        assert_export_kwargs(kwargs)


def test_parse_names_metadata_accepts_python_and_json_forms() -> None:
    python_form = "{0: 'leaf', 1: 'needle', 2: 'flower', 3: 'inflorescence'}"
    assert parse_names_metadata(python_form) == dict(enumerate(CLASS_NAMES))
    json_form = '{"0": "leaf", "1": "needle", "2": "flower", "3": "inflorescence"}'
    assert parse_names_metadata(json_form) == dict(enumerate(CLASS_NAMES))


def _write_fake_onnx(
    path: Path,
    *,
    output_shape: tuple[int, ...],
    names: str,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
) -> None:
    values = numpy_helper.from_array(
        np.zeros((int(np.prod(output_shape)),), dtype=np.float32), name="values"
    )
    shape = numpy_helper.from_array(
        np.asarray(output_shape, dtype=np.int64), name="shape"
    )
    node = helper.make_node("Reshape", ["values", "shape"], ["output0"])
    graph = helper.make_graph(
        [node],
        "fake",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(output_shape))],
        initializer=[values, shape],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 10
    model.metadata_props.add(key="names", value=names)
    model.metadata_props.add(key="task", value="detect")
    onnx.save(model, path)


def test_verify_accepts_contract_shapes_and_class_order(tmp_path: Path) -> None:
    path = tmp_path / "good.onnx"
    _write_fake_onnx(
        path,
        output_shape=(1, 8, 8400),
        names=str(dict(enumerate(CLASS_NAMES))),
    )
    result = verify_onnx(path)
    assert result["ok"] is True, result["checks"]
    assert result["input"]["shape"] == [1, 3, 640, 640]
    assert result["output"]["shape"] == [1, 8, 8400]
    assert result["output"]["runtime_shape"] == [1, 8, 8400]
    assert result["precision"] == "fp32"
    assert result["end2end"] is False
    assert result["class_names"] == list(CLASS_NAMES)


def test_verify_rejects_end2end_output(tmp_path: Path) -> None:
    path = tmp_path / "bad.onnx"
    _write_fake_onnx(
        path,
        output_shape=(1, 300, 6),
        names=str(dict(enumerate(CLASS_NAMES))),
    )
    result = verify_onnx(path)
    assert result["ok"] is False
    assert result["end2end"] is True
    codes = {item["name"] for item in result["checks"] if not item["ok"]}
    assert "output_shape" in codes


def test_verify_rejects_wrong_class_order(tmp_path: Path) -> None:
    path = tmp_path / "order.onnx"
    _write_fake_onnx(
        path,
        output_shape=(1, 8, 8400),
        names=str({0: "needle", 1: "leaf", 2: "flower", 3: "inflorescence"}),
    )
    result = verify_onnx(path)
    assert result["ok"] is False
    assert "class_order" in {item["name"] for item in result["checks"] if not item["ok"]}
