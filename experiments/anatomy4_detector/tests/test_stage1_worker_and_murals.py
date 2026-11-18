# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-STAGE1-PULL-WORKER role=test
"""Stage-1 generalization: dynamic classes, visual samples, and the worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from experiments.anatomy4_detector.export import verify_onnx
from experiments.anatomy4_detector.metrics import PixelBox, PixelDetection
from experiments.anatomy4_detector.onnx_eval import decode_raw_output, evaluate_onnx
from experiments.anatomy4_detector.publish import AuthMaterial
from experiments.anatomy4_detector.render import (
    build_eval_samples_manifest,
    classify_eval_record,
    render_stratified_murals,
)
from experiments.anatomy4_detector.worker import (
    Stage1Worker,
    WorkerConfig,
    _api_metrics,
    _image_entries,
    materialize_dataset,
    run_once,
)

SPIDER = ["cephalothorax", "abdomen", "eye_arrangement", "pedipalp", "orb_web"]
THREE = ["bud", "stem", "tendril"]


def _write_fake_onnx(
    path: Path,
    *,
    classes: list[str],
    imgsz: int = 640,
) -> Path:
    anchors = (imgsz // 8) ** 2 + (imgsz // 16) ** 2 + (imgsz // 32) ** 2
    output_shape = (1, 4 + len(classes), anchors)
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
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(output_shape))],
        initializer=[values, shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    model.metadata_props.add(key="names", value=str(dict(enumerate(classes))))
    model.metadata_props.add(key="task", value="detect")
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def _image(path: Path, size: tuple[int, int] = (80, 60), color=(40, 120, 60)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _record(
    key: str,
    image: Path,
    ground_truth: list[tuple[int, float, float, float, float]],
    detections: list[tuple[int, float, float, float, float, float]],
) -> dict[str, Any]:
    return {
        "item_key": key,
        "stem": key,
        "observation_id": f"obs-{key}",
        "photo_url": f"https://example.test/{key}.jpg",
        "image": str(image),
        "prepared_image": str(image),
        "width": 80,
        "height": 60,
        "ground_truth": [PixelBox(*box) for box in ground_truth],
        "detections": [PixelDetection(*det) for det in detections],
    }


# ── 1. Dynamic contracts ────────────────────────────────────────────────────


def test_verify_onnx_accepts_five_class_spider_contract_at_320(tmp_path: Path) -> None:
    path = _write_fake_onnx(tmp_path / "spider.onnx", classes=SPIDER, imgsz=320)
    result = verify_onnx(
        path, classes=SPIDER, contract="arborphy-spider-v1", imgsz=320
    )
    assert result["ok"] is True, result["checks"]
    assert result["class_names"] == SPIDER
    assert result["contract"] == "arborphy-spider-v1"
    assert result["output"]["shape"] == [1, 9, 2100]


def test_verify_onnx_accepts_three_class_stage1_contract(tmp_path: Path) -> None:
    path = _write_fake_onnx(tmp_path / "three.onnx", classes=THREE)
    result = verify_onnx(path, classes=THREE, contract="stage1-tendril-v1")
    assert result["ok"] is True, result["checks"]
    assert result["output"]["shape"] == [1, 7, 8400]
    assert result["expected_class_names"] == THREE


def test_decode_raw_output_uses_arbitrary_class_names() -> None:
    meta = {
        "orig_w": 100.0,
        "orig_h": 100.0,
        "input_size": 640.0,
        "pad_x": 0.0,
        "pad_y": 0.0,
        "scale_x": 6.4,
        "scale_y": 6.4,
    }
    raw = np.zeros((1, 4 + len(SPIDER), 3), dtype=np.float32)
    raw[0, :4, 0] = [320.0, 320.0, 64.0, 64.0]
    raw[0, 4 + 2, 0] = 0.91  # eye_arrangement
    rows = decode_raw_output(raw, classes=SPIDER, meta=meta, threshold=0.25)
    assert [row["class"] for row in rows] == ["eye_arrangement"]
    assert rows[0]["class_id"] == 2


def test_evaluate_onnx_and_manifest_are_class_list_agnostic(tmp_path: Path) -> None:
    path = _write_fake_onnx(tmp_path / "three.onnx", classes=THREE)
    image = _image(tmp_path / "img.jpg")
    records = [
        {
            "item_key": "a",
            "observation_id": "obs-a",
            "photo_url": "https://example.test/a.jpg",
            "image": str(image),
            "prepared_image": str(image),
            "width": 80,
            "height": 60,
            "ground_truth": [
                {"class_id": 1, "bbox": [0.1, 0.1, 0.5, 0.5]},
            ],
        }
    ]
    evaluation = evaluate_onnx(
        onnx_path=path,
        records=records,
        classes=THREE,
        contract="stage1-tendril-v1",
        split="test",
    )
    assert evaluation["class_names"] == THREE
    assert set(evaluation["metrics"]["per_class"]) == set(THREE)
    assert evaluation["metrics"]["contract"] == "stage1-tendril-v1"
    assert evaluation["records"][0]["item_key"] == "a"
    manifest = build_eval_samples_manifest(
        evaluation["records"], contract="stage1-tendril-v1", classes=THREE
    )
    assert manifest["class_names"] == THREE
    assert manifest["bucket_counts"]["total_images"] == 1
    assert manifest["bucket_counts"]["fn_images"] == 1
    assert manifest["samples"][0]["classes_fn"] == ["stem"]


def test_render_stratified_murals_supports_arbitrary_classes(tmp_path: Path) -> None:
    image = _image(tmp_path / "img.jpg")
    records = [
        _record("tp", image, [(2, 5, 5, 40, 40)], [(2, 6, 6, 41, 41, 0.9)]),
        _record("fp", image, [], [(0, 5, 5, 30, 30, 0.8)]),
        _record("fn", image, [(1, 10, 10, 50, 50)], []),
        _record("tn", image, [], []),
    ]
    murals = render_stratified_murals(
        records, tmp_path / "murals", contract="stage1-tendril-v1", classes=THREE
    )
    names = {mural["filename"] for mural in murals}
    assert "mural-overview.jpg" in names
    assert "mural-errors.jpg" in names
    assert "mural-stem.jpg" in names
    assert "mural-orb-web.jpg" not in names
    for mural in murals:
        assert Path(mural["path"]).is_file()
        assert Path(mural["path"]).stat().st_size < 4 * 1024 * 1024


# ── 2. Classification + samples ─────────────────────────────────────────────


def test_classify_eval_record_tp_fp_fn_confusion_and_tn(tmp_path: Path) -> None:
    image = _image(tmp_path / "img.jpg")
    record = _record(
        "mixed",
        image,
        ground_truth=[(0, 5, 5, 40, 40), (1, 45, 5, 70, 40)],
        detections=[
            (0, 6, 6, 39, 39, 0.9),   # TP leaf
            (2, 46, 6, 69, 39, 0.8),  # FP flower over needle GT -> confusion
        ],
    )
    result = classify_eval_record(record, classes=["leaf", "needle", "flower", "inflorescence"])
    assert result["tp"] == 1 and result["fp"] == 1 and result["fn"] == 1
    assert result["buckets"] == ["tp", "fp", "fn"]
    assert result["classes_tp"] == ["leaf"]
    assert result["classes_fp"] == ["flower"]
    assert result["classes_fn"] == ["needle"]
    assert result["confusion_pairs"] == [["needle", "flower"]]
    confusion = [d for d in result["detections"] if d["match"] == "fp"][0]
    assert confusion["confused_with"] == "needle"
    assert result["ground_truth"][0]["bbox"] == pytest.approx(
        [0.0625, 0.083333, 0.5, 0.666667], abs=1e-4
    )

    clean = classify_eval_record(
        _record("clean", image, [], []), classes=["leaf", "needle"]
    )
    assert clean["tn"] == 1
    assert clean["buckets"] == ["tn"]


def test_bucket_counts_include_all_records_not_just_samples(tmp_path: Path) -> None:
    image = _image(tmp_path / "img.jpg")
    records = [
        _record(f"tp-{index}", image, [(0, 5, 5, 30, 30)], [(0, 5, 5, 30, 30, 0.9)])
        for index in range(30)
    ] + [_record("tn", image, [], [])]
    manifest = build_eval_samples_manifest(records, classes=["leaf"], sample_per_bucket=4)
    counts = manifest["bucket_counts"]
    assert counts["total_images"] == 31
    assert counts["tp_images"] == 30
    assert counts["tn_images"] == 1
    assert counts["by_class"]["leaf"]["tp"] == 30
    assert counts["sampled_images"] == len(manifest["samples"]) <= 5
    assert manifest["schema"] == "arq.stage1.eval_samples/v1"


def test_manifest_caps_500_records_and_is_deterministic(tmp_path: Path) -> None:
    image = _image(tmp_path / "img.jpg")
    records = []
    for index in range(500):
        if index % 4 == 0:
            records.append(_record(f"r{index:03d}", image, [], []))
        elif index % 4 == 1:
            records.append(
                _record(f"r{index:03d}", image, [(0, 5, 5, 30, 30)], [(0, 5, 5, 30, 30, 0.9)])
            )
        elif index % 4 == 2:
            records.append(
                _record(f"r{index:03d}", image, [], [(1, 5, 5, 30, 30, 0.8)])
            )
        else:
            records.append(_record(f"r{index:03d}", image, [(0, 5, 5, 30, 30)], []))
    first = build_eval_samples_manifest(records, classes=["leaf", "needle"])
    second = build_eval_samples_manifest(records, classes=["leaf", "needle"])
    assert first["samples"] == second["samples"]
    assert len(first["samples"]) <= 240
    assert first["bucket_counts"]["total_images"] == 500
    keys = [sample["item_key"] for sample in first["samples"]]
    assert keys == sorted(keys)


def test_render_murals_are_single_sheet_and_bounded(tmp_path: Path) -> None:
    image = _image(tmp_path / "img.jpg")
    records = [
        _record(
            f"r{index:03d}",
            image,
            [(index % 2, 5, 5, 30, 30)],
            [(index % 2, 5, 5, 30, 30, 0.9)] if index % 3 else [],
        )
        for index in range(200)
    ]
    murals = render_stratified_murals(records, tmp_path / "murals", classes=["leaf", "needle"])
    for mural in murals:
        assert Path(mural["path"]).stat().st_size < 4 * 1024 * 1024
        assert mural["images"] <= 12
    assert len([m for m in murals if m["bucket"] == "all"]) == 1


# ── 3. Worker ───────────────────────────────────────────────────────────────


class _Api:
    def __init__(self, *, cancel_training: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.cancel_training = cancel_training
        self.media_requests = 0
        self.completes: list[dict[str, Any]] = []
        self.artifacts: list[str] = []
        self.model_payload: dict[str, Any] | None = None
        self.metrics_payload: dict[str, Any] | None = None
        self.heartbeats: list[dict[str, Any]] = []
        self.job: dict[str, Any] = {
            "job_id": "job-1",
            "contract": "arborphy-spider-v1",
            "class_names": SPIDER,
            "dataset_id": "ds-spider",
            "model_name": "spider-stage1",
            "epochs": 2,
            "imgsz": 640,
            "base_weights": "model:base-spider",
        }
        self.export = {
            "class_names": SPIDER,
            "images": [
                {
                    "item_key": "train-a",
                    "split": "train",
                    "url": "http://arq.test/media/a.jpg",
                    "photo_url": "http://arq.test/media/a.jpg",
                    "observation_id": "obs-a",
                    "labels": [{"class_id": 0, "bbox": [0.1, 0.1, 0.4, 0.4]}],
                },
                {
                    "item_key": "val-a",
                    "split": "val",
                    "url": "http://arq.test/media/b.jpg",
                    "photo_url": "http://arq.test/media/b.jpg",
                    "observation_id": "obs-b",
                    "labels": [{"class_id": 1, "bbox": [0.2, 0.2, 0.5, 0.5]}],
                },
                {
                    "item_key": "test-a",
                    "split": "test",
                    "url": "http://arq.test/media/c.jpg",
                    "photo_url": "http://arq.test/media/c.jpg",
                    "observation_id": "obs-c",
                    "labels": [],
                },
            ],
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        self.calls.append((method, path))
        if path.endswith("/training-jobs/lease"):
            return httpx.Response(200, json={"job": self.job})
        if path.endswith("/dataset-export"):
            return httpx.Response(200, json=self.export)
        if path.endswith("/base-weights"):
            return httpx.Response(200, content=b"fake-pt-weights")
        if path.endswith("/heartbeat"):
            payload = json.loads(request.content or b"{}")
            self.heartbeats.append(payload)
            cancel = self.cancel_training and payload.get("status") == "training"
            return httpx.Response(200, json={"cancel_requested": cancel})
        if path.startswith("/api/media/") or path.startswith("/media/"):
            self.media_requests += 1
            return httpx.Response(200, content=_jpeg_bytes())
        if path == "/api/fieldvision/anatomy/models" and method == "POST":
            self.model_payload = json.loads(request.content)
            return httpx.Response(200, json={"model_id": "model-1"})
        if path.endswith("/artifacts"):
            self.artifacts.append(request.url.params.get("kind", ""))
            return httpx.Response(200, json={"bytes": 10, "sha256": "x", "storage": "gcs"})
        if path.endswith("/metrics") and method == "POST":
            self.metrics_payload = json.loads(request.content)
            return httpx.Response(200, json={"model_id": "model-1"})
        if path.endswith("/training-jobs/job-1/complete"):
            self.completes.append(json.loads(request.content))
            return httpx.Response(200, json={"job_id": "job-1", "status": "succeeded"})
        return httpx.Response(404, json={"detail": f"unexpected {method} {path}"})


def _jpeg_bytes(size: tuple[int, int] = (64, 48)) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", size, (30, 90, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _trainer(tmp_path: Path, *, fail_with: str | None = None):
    def backend(job, yolo_dir, base_weights_path, heartbeat):
        if fail_with is not None:
            raise RuntimeError(fail_with)
        assert (yolo_dir / "data.yaml").is_file()
        assert (yolo_dir / "labels" / "test").is_dir()
        assert (yolo_dir / "images" / "train").is_dir()
        heartbeat(1, {"box_loss": 0.1})
        onnx_path = _write_fake_onnx(
            tmp_path / "export" / "model.onnx",
            classes=list(job["class_names"]),
        )
        best = tmp_path / "export" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"best-weights")
        return best, onnx_path, {"epochs": int(job["epochs"])}

    return backend


def _config(tmp_path: Path, **overrides) -> WorkerConfig:
    values = dict(
        base="http://arq.test",
        worker_id="test-worker",
        work_dir=tmp_path / "jobs",
        media_cache_dir=tmp_path / "cache",
    )
    values.update(overrides)
    return WorkerConfig(**values)


def _worker(tmp_path: Path, api: _Api, **kwargs) -> Stage1Worker:
    transport = httpx.MockTransport(api.handler)
    return Stage1Worker(
        _config(tmp_path),
        auth=AuthMaterial(api_key="SECRET", bearer="TOKEN"),
        transport=transport,
        trainer_backend=_trainer(tmp_path, fail_with=kwargs.get("fail_with")),
        sleep=lambda _: None,
    )


def test_worker_empty_queue_is_a_noop(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/training-jobs/lease")
        return httpx.Response(200, json={"job": None})

    result = run_once(
        _config(tmp_path),
        auth=AuthMaterial(api_key="SECRET", bearer="TOKEN"),
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    assert result is None


def test_worker_full_cycle_publishes_and_caches_media(tmp_path: Path) -> None:
    api = _Api()
    worker = _worker(tmp_path, api)
    report = worker.run_once()
    assert report is not None and report["status"] == "succeeded"
    assert report["model_id"] == "model-1"
    assert api.model_payload is not None
    assert api.model_payload["class_names"] == SPIDER
    assert api.model_payload["contract"] == "arborphy-spider-v1"
    assert api.model_payload["training_job_id"] == "job-1"
    assert api.model_payload["provenance"]["eval_split_used"] == "test"
    assert api.metrics_payload is not None
    assert api.completes[-1]["outcome"] == "succeeded"
    assert api.completes[-1]["onnx_parity_passed"] is True
    assert {"onnx", "weights", "metrics", "manifest", "preview"}.issubset(set(api.artifacts))
    statuses = [payload.get("status") for payload in api.heartbeats]
    assert "preparing" in statuses
    assert "training" in statuses
    assert "evaluating" in statuses
    assert "publishing" in statuses
    manifest_path = tmp_path / "jobs" / "job-1" / "eval_samples.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "arq.stage1.eval_samples/v1"
    assert manifest["split"] == "test"
    assert manifest["murals"]
    first_run_media = api.media_requests
    assert first_run_media == 3

    # Second run leases the same job; every image must come from the media cache.
    api.calls.clear()
    second = worker.run_once()
    assert second is not None and second["status"] == "succeeded"
    assert api.media_requests == first_run_media


def test_worker_cancel_mid_training_exits_cleanly(tmp_path: Path) -> None:
    api = _Api(cancel_training=True)
    worker = _worker(tmp_path, api)
    report = worker.run_once()
    assert report is not None and report["status"] == "cancelled"
    assert not any(call[1].endswith("/models") for call in api.calls)
    assert api.completes == []


def test_worker_trainer_failure_reports_scrubbed_error(tmp_path: Path) -> None:
    api = _Api()
    worker = _worker(tmp_path, api, fail_with="exploded for SECRET and TOKEN")
    report = worker.run_once()
    assert report is not None and report["status"] == "failed"
    assert api.completes[-1]["outcome"] == "failed"
    error_text = api.completes[-1]["error_text"]
    assert "SECRET" not in error_text
    assert "TOKEN" not in error_text
    assert "[REDACTED]" in error_text
    assert not any(call[1].endswith("/models") for call in api.calls)


class _StaticMediaClient:
    def get_bytes(self, url, *, stage="media"):
        return _jpeg_bytes()


def test_export_pairs_supply_labels_and_materialize_writes_yolo_cxcywh(tmp_path: Path) -> None:
    export = {
        "class_names": THREE,
        "pairs": [
            {
                "item_id": "p1",
                "split": "train",
                "image": {
                    "item_key": "p1",
                    "split": "train",
                    "url": "http://arq.test/media/p1.jpg",
                    "observation_id": "obs-p1",
                },
                "label": {"lines": ["0 0.200000 0.200000 0.200000 0.200000"]},
            }
        ],
    }
    entries = _image_entries(export)
    assert entries[0]["labels"] == [
        {"class_id": 0, "bbox": pytest.approx([0.1, 0.1, 0.3, 0.3])}
    ]

    result = materialize_dataset(
        _StaticMediaClient(),
        export,
        tmp_path / "yolo",
        classes=THREE,
        cache_dir=None,
    )
    label_files = sorted((tmp_path / "yolo" / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 1
    line = label_files[0].read_text().split()
    assert [int(line[0]), *(float(value) for value in line[1:])] == pytest.approx(
        [0, 0.2, 0.2, 0.2, 0.2]
    )
    assert result["records"]["train"][0]["ground_truth"][0]["x0"] == pytest.approx(0.1)


def test_api_metrics_fills_undefined_values_and_trims_confusion_background() -> None:
    payload = {
        "class_names": THREE,
        "per_class": {
            "bud": {"precision": 1.0, "recall": 1.0, "ap50": 1.0, "ap50_95": 1.0},
            "stem": {"precision": None, "recall": 0.0, "ap50": 0.0, "ap50_95": None},
            "tendril": {"precision": 0.5, "recall": None, "ap50": 0.0, "ap50_95": 0.0},
        },
        "confusion_matrix": {
            "labels": [*THREE, "background"],
            "matrix": [
                [3, 0, 0, 0],
                [0, 2, 1, 4],
                [0, 0, 0, 2],
                [0, 0, 0, 0],
            ],
        },
    }
    out = _api_metrics(payload, THREE)
    assert out["per_class"]["stem"]["precision"] == 0.0
    assert out["per_class"]["stem"]["ap50_95"] == 0.0
    assert out["per_class"]["tendril"]["recall"] == 0.0
    assert out["confusion"] == [[3, 0, 0], [0, 2, 1], [0, 0, 0]]
    assert out["confusion_classes"] == THREE
