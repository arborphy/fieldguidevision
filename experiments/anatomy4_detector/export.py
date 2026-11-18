# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""ONNX export contract for arq-api.

Exact export arguments (anatomy4-v1):

    imgsz=640, batch=1, dynamic=False, opset=17, simplify=False,
    device=cpu, FP32 (half=False, quantize unset)

**Never pass ``nms``.** Ultralytics 8.4.190 treats ``nms=False`` as an
end-to-end export and emits ``[1, 300, 6]``, which arq-api cannot decode. The
contract output is raw ``[1, 8, 8400]`` (4 box channels + 4 classes).

Verification uses onnxruntime on real inputs and asserts shape, dtype, and the
embedded class-name order.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import CLASS_NAMES, CONTRACT

EXPORT_IMGSZ = 640
EXPORT_BATCH = 1
EXPORT_OPSET = 17


def anchor_count(imgsz: int = EXPORT_IMGSZ) -> int:
    """YOLO anchor count for a square input: 8400 at 640, 2100 at 320."""

    size = int(imgsz)
    return (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2


def expected_output_shape(
    *, imgsz: int = EXPORT_IMGSZ, classes: Sequence[str] = CLASS_NAMES
) -> list[int]:
    return [1, 4 + len(classes), anchor_count(imgsz)]


def expected_input_shape(imgsz: int = EXPORT_IMGSZ) -> list[int]:
    return [1, 3, int(imgsz), int(imgsz)]


EXPECTED_INPUT_SHAPE = expected_input_shape()
EXPECTED_OUTPUT_CHANNELS = 4 + len(CLASS_NAMES)  # 8
EXPECTED_OUTPUT_ANCHORS = anchor_count()
EXPECTED_OUTPUT_SHAPE = expected_output_shape()

EXPORT_SPEC: dict[str, Any] = {
    "format": "onnx",
    "imgsz": EXPORT_IMGSZ,
    "batch": EXPORT_BATCH,
    "dynamic": False,
    "opset": EXPORT_OPSET,
    "simplify": False,
    "device": "cpu",
    "half": False,
}

# Keys the exporter accepts that would silently break the contract.
FORBIDDEN_EXPORT_KEYS: frozenset[str] = frozenset({"nms", "end2end", "int8", "quantize"})


def build_export_kwargs(*, imgsz: int = EXPORT_IMGSZ) -> dict[str, Any]:
    """Return a fresh, validated kwargs dict for ``YOLO.export``."""

    kwargs = dict(EXPORT_SPEC)
    kwargs["imgsz"] = int(imgsz)
    assert_export_kwargs(kwargs, imgsz=imgsz)
    return kwargs


def assert_export_kwargs(
    kwargs: Mapping[str, Any], *, imgsz: int = EXPORT_IMGSZ
) -> None:
    forbidden = FORBIDDEN_EXPORT_KEYS.intersection(kwargs)
    if forbidden:
        raise ValueError(
            "anatomy4-v1 export forbids these keys: "
            f"{sorted(forbidden)} (nms=False produces an incompatible "
            "[1,300,6] end-to-end output)"
        )
    expected = {
        "format": "onnx",
        "imgsz": int(imgsz),
        "batch": 1,
        "dynamic": False,
        "opset": 17,
        "simplify": False,
        "device": "cpu",
        "half": False,
    }
    for key, value in expected.items():
        if kwargs.get(key) != value:
            raise ValueError(
                f"export {key}={kwargs.get(key)!r} violates anatomy4-v1 "
                f"(expected {value!r})"
            )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_onnx(
    weights: str | Path,
    out_path: str | Path | None = None,
    *,
    contract: str = CONTRACT,
    classes: Sequence[str] = CLASS_NAMES,
    imgsz: int = EXPORT_IMGSZ,
) -> dict[str, Any]:
    from ultralytics import YOLO

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    model = YOLO(str(weights_path))
    kwargs = build_export_kwargs(imgsz=imgsz)
    exported = Path(model.export(**kwargs)).expanduser().resolve()
    target = (
        Path(out_path).expanduser().resolve()
        if out_path is not None
        else weights_path.with_suffix(".onnx")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if exported != target:
        target.write_bytes(exported.read_bytes())
        if exported.parent != target.parent:
            try:
                exported.unlink()
            except OSError:
                pass
    return {
        "schema": "arq.anatomy4.export/v1",
        "contract": contract,
        "class_names": list(classes),
        "weights": str(weights_path),
        "onnx": str(target),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "export_kwargs": kwargs,
        "passed_nms": False,
        "precision": "fp32",
        "quantize": None,
    }


def parse_names_metadata(value: str) -> dict[int, str] | str:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(parsed, dict):
        return {int(key): str(name) for key, name in parsed.items()}
    if isinstance(parsed, list):
        return {index: str(name) for index, name in enumerate(parsed)}
    return value


def verify_onnx(
    path: str | Path,
    *,
    expected_names: Sequence[str] = CLASS_NAMES,
    classes: Sequence[str] | None = None,
    contract: str = CONTRACT,
    imgsz: int = EXPORT_IMGSZ,
    run_inference: bool = True,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    names_expected = list(classes) if classes is not None else list(expected_names)
    input_shape_expected = expected_input_shape(imgsz)
    output_shape_expected = expected_output_shape(imgsz=imgsz, classes=names_expected)
    model_path = Path(path).expanduser().resolve()
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    metadata = session.get_modelmeta().custom_metadata_map or {}
    raw_names = metadata.get("names")
    names = parse_names_metadata(raw_names) if raw_names is not None else None
    ordered_names = (
        [names[index] for index in sorted(names)]
        if isinstance(names, dict)
        else None
    )

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    input_shape = list(inputs[0].shape)
    output_shape = list(outputs[0].shape)
    check("single_input", len(inputs) == 1, [item.name for item in inputs])
    check("single_output", len(outputs) == 1, [item.name for item in outputs])
    check("input_shape", input_shape == input_shape_expected, input_shape)
    check("output_shape", output_shape == output_shape_expected, output_shape)
    check("input_dtype_fp32", inputs[0].type == "tensor(float)", inputs[0].type)
    check("output_dtype_fp32", outputs[0].type == "tensor(float)", outputs[0].type)
    check(
        "class_order",
        ordered_names == names_expected,
        ordered_names,
    )
    check("not_end2end", output_shape != [1, 300, 6], output_shape)
    check(
        "metadata_task_detect",
        metadata.get("task", "detect") == "detect",
        metadata.get("task"),
    )

    runtime_output_shape: list[int] | None = None
    if run_inference:
        dummy = np.zeros(input_shape_expected, dtype=np.float32)
        result = session.run([outputs[0].name], {inputs[0].name: dummy})[0]
        runtime_output_shape = list(result.shape)
        check("runtime_output_shape", runtime_output_shape == output_shape_expected, runtime_output_shape)
        check("runtime_dtype_fp32", str(result.dtype) == "float32", str(result.dtype))
        check("runtime_finite", bool(np.isfinite(result).all()), None)

    ok = all(item["ok"] for item in checks)
    return {
        "schema": "arq.anatomy4.onnx-verify/v1",
        "contract": contract,
        "ok": ok,
        "path": str(model_path),
        "sha256": sha256_file(model_path),
        "bytes": model_path.stat().st_size,
        "input": {
            "name": inputs[0].name,
            "shape": input_shape,
            "type": inputs[0].type,
        },
        "output": {
            "name": outputs[0].name,
            "shape": output_shape,
            "type": outputs[0].type,
            "runtime_shape": runtime_output_shape,
        },
        "class_names": ordered_names,
        "expected_class_names": names_expected,
        "metadata": metadata,
        "checks": checks,
        "end2end": output_shape == [1, 300, 6],
        "precision": "fp32" if inputs[0].type == "tensor(float)" else inputs[0].type,
    }
