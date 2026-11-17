# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from PIL import Image

from experiments.anatomy4_detector.contracts import CLASS_NAMES, CONTRACT, split_for_observation
from experiments.anatomy4_detector.publish import (
    plan_publish,
    ARTIFACT_LIMITS,
    ArqClient,
    AuthMaterial,
    ItemPlan,
    PublishError,
    PublishOptions,
    auth_from_env,
    build_dataset_payload,
    build_items,
    build_metrics,
    collect_artifacts,
    describe_plan,
    publish,
    resolve_base,
    validate_infer_response,
)

MODEL_ID = "anat-model-a4-test-0001"
DATASET_ID = "anat-ds-a4-test-0001"
API_KEY = "test-api-key-DO-NOT-LEAK"
TOKEN = "test-bearer-token-DO-NOT-LEAK"


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path: Path, size=(64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (30, 40, 30)).save(path)
    return path


def make_dataset(root: Path, *, extra_boxes: int = 0) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    images = {
        "train": _image(root / "images" / "train" / "a.jpg"),
        "val": _image(root / "images" / "val" / "b.jpg"),
        "test": _image(root / "images" / "test" / "c.jpg"),
    }
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    leaf_boxes = [
        {
            "class_id": 0,
            "class_name": "leaf",
            "x0": 0.1,
            "y0": 0.1,
            "x1": 0.3,
            "y1": 0.3,
        }
    ]
    for index in range(extra_boxes):
        leaf_boxes.append(
            {
                "class_id": 0,
                "class_name": "leaf",
                "x0": 0.4 + index * 0.001,
                "y0": 0.1,
                "x1": 0.5 + index * 0.001,
                "y1": 0.2,
            }
        )
    (root / "labels" / "train" / "a.txt").write_text(
        "".join(
            f"{box['class_id']} {box['x0']:.6f} {box['y0']:.6f} "
            f"{box['x1']:.6f} {box['y1']:.6f}\n"
            for box in leaf_boxes
        )
    )
    (root / "labels" / "val" / "b.txt").write_text("")
    (root / "labels" / "test" / "c.txt").write_text(
        "1 0.200000 0.200000 0.500000 0.500000\n"
    )

    entries = [
        {
            "image": "images/train/a.jpg",
            "photo_id": "a",
            "observation_id": "inat-a",
            "negative": False,
            "model": "yoloe-26s-seg.pt",
            "source_sha256": "aaa",
            "review": {
                "reviewed_proposals": 2,
                "accepted": 1,
                "rejected": 1,
                "undecided": 1,
                "negative_basis": None,
            },
            "boxes": leaf_boxes,
        },
        {
            "image": "images/val/b.jpg",
            "photo_id": "b",
            "observation_id": "inat-b",
            "negative": True,
            "model": "yoloe-26s-seg.pt",
            "source_sha256": "bbb",
            "review": {
                "reviewed_proposals": 1,
                "accepted": 0,
                "rejected": 1,
                "undecided": 0,
                "negative_basis": "reviewed_all_rejected",
            },
            "boxes": [],
        },
        {
            "image": "images/test/c.jpg",
            "photo_id": "c",
            "observation_id": "inat-c",
            "negative": False,
            "model": "yoloe-26s-seg.pt",
            "source_sha256": "ccc",
            "review": {
                "reviewed_proposals": 1,
                "accepted": 1,
                "rejected": 0,
                "undecided": 0,
                "negative_basis": None,
            },
            "boxes": [
                {
                    "class_id": 1,
                    "class_name": "needle",
                    "x0": 0.2,
                    "y0": 0.2,
                    "x1": 0.5,
                    "y1": 0.5,
                }
            ],
        },
    ]
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schema": "arq.anatomy4.dataset/v1",
                "contract": "anatomy4-v1",
                "classes": list(CLASS_NAMES),
                "images": entries,
            },
            indent=1,
        )
    )
    provenance_rows = [
        {
            "schema": "arq.anatomy4.proposal-provenance/v1",
            "proposal_id": "p1",
            "image": "images/train/a.jpg",
            "photo_id": "a",
            "observation_id": "inat-a",
            "split": "train",
            "class_id": 0,
            "class_name": "leaf",
            "xyxy_norm": [0.1, 0.1, 0.3, 0.3],
            "box_source": "bbox",
            "box_score": 0.81,
            "prompt": "leaf",
            "detector_label": "leaf",
            "mask_iou": None,
            "model": {"detector": "yoloe-26s-seg.pt", "backend": "yoloe"},
            "review": {
                "verdict": "tp",
                "raw_verdict": "tp",
                "notes": "clean leaf",
                "sheet": "review.csv",
                "row": 2,
            },
        }
    ]
    with (root / "provenance.jsonl").open("w") as handle:
        for row in provenance_rows:
            handle.write(json.dumps(row) + "\n")

    proposals_path = root / "audit" / "proposals.json"
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    proposals_path.write_text(
        json.dumps(
            {
                "contract": "anatomy4-v1",
                "images": [
                    {
                        "image_id": photo_id,
                        "observation_id": f"inat-{photo_id}",
                        "download": {"source_ref": f"https://example.org/{photo_id}.jpg"},
                    }
                    for photo_id in ("a", "b", "c", "pending")
                ],
                "proposals": [
                    {
                        "proposal_id": "p1",
                        "photo_id": "a",
                        "observation_id": "inat-a",
                        "class_name": "leaf",
                        "box_score": 0.81,
                        "bbox": [0.1, 0.1, 0.3, 0.3],
                        "model": {"detector": "yoloe-26s-seg.pt"},
                    },
                    {
                        "proposal_id": "p2",
                        "photo_id": "pending",
                        "observation_id": "inat-pending",
                        "class_name": "flower",
                        "box_score": 0.35,
                        "bbox": [0.2, 0.2, 0.4, 0.4],
                        "model": {"detector": "yoloe-26s-seg.pt"},
                    },
                    {
                        "proposal_id": "p3",
                        "photo_id": "b",
                        "observation_id": "inat-b",
                        "class_name": "flower",
                        "box_score": 0.22,
                        "bbox": [0.3, 0.3, 0.4, 0.4],
                        "model": {"detector": "yoloe-26s-seg.pt"},
                    },
                ],
            }
        )
    )
    review_path = root / "audit" / "review.csv"
    review_path.write_text(
        "proposal_id,verdict,notes\n"
        "p1,tp,clean leaf\n"
        "p3,fp,petal of the same plant\n"
    )
    (root / "materialization.json").write_text(
        json.dumps(
            {
                "schema": "arq.anatomy4.materialization/v1",
                "contract": "anatomy4-v1",
                "selection": {"seed": "anatomy4-v1", "classes": list(CLASS_NAMES)},
                "inputs": {
                    "proposals": {"path": str(proposals_path), "sha256": _sha(proposals_path)},
                    "review": {"path": str(review_path), "sha256": _sha(review_path)},
                },
                "review": {"rows": 2, "accepted": 1, "rejected": 1},
                "counts": {"images": 3},
            },
            indent=1,
        )
    )
    return {"root": root, "images": images}


def make_run(tmp_path: Path, *, zero_support: bool = False, extra_boxes: int = 0) -> dict[str, Any]:
    dataset = make_dataset(tmp_path / "dataset", extra_boxes=extra_boxes)
    root = dataset["root"]
    onnx = _write(tmp_path / "export" / "anatomy4.onnx", b"ONNX-FAKE-BYTES" * 64)
    weights = _write(tmp_path / "runs" / "weights" / "best.pt", b"BEST-WEIGHTS" * 64)
    previews = [
        _write(tmp_path / "eval" / "contact" / "contact-leaf-00.jpg", b"\xff\xd8\xff\xe0PREVIEW-A"),
        _write(tmp_path / "eval" / "overlays" / "c.jpg", b"\xff\xd8\xff\xe0PREVIEW-B"),
    ]
    predictions = tmp_path / "eval" / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"split": "test", "stem": "c", "class_name": "needle", "confidence": 0.9}) + "\n"
    )
    per_class = {
        cls: {
            "class_id": index,
            "support": 0 if (zero_support and cls == "needle") else 2,
            "precision": None if (zero_support and cls == "needle") else 0.8,
            "recall": None if (zero_support and cls == "needle") else 0.7,
            "ap50": None if (zero_support and cls == "needle") else 0.75,
            "ap50_95": None if (zero_support and cls == "needle") else 0.55,
        }
        for index, cls in enumerate(CLASS_NAMES)
    }
    metrics_json = tmp_path / "eval" / "metrics.json"
    metrics_json.write_text(
        json.dumps(
            {
                "split": "test",
                "per_class": per_class,
                "macro": {"ap50": 0.75},
                "micro": {"precision": 0.8},
                "counts": {
                    "images": 3,
                    "gt_boxes": 6 if zero_support else 8,
                    "negative_images": 1,
                },
                "operating_point": {"conf": 0.25, "iou": 0.5},
                "confusion_matrix": {
                    "labels": list(CLASS_NAMES) + ["background"],
                    "matrix": [
                        [2, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 3],
                    ],
                },
                "predictions_jsonl": str(predictions),
            },
            indent=1,
        )
    )
    manifest = {
        "schema": "arq.anatomy4.detector-run/v1",
        "contract": "anatomy4-v1",
        "generated_at": "2026-11-12T00:00:00+00:00",
        "synthetic": False,
        "status": "complete",
        "git": {"available": True, "revision": "a" * 40, "branch": "anatomy4-detector", "dirty": False},
        "versions": {"torch": "2.14.0+cu130", "ultralytics": "8.4.190"},
        "dataset": {
            "root": str(root),
            "status": "ok",
            "blocking": False,
            "split_isolation": "clean",
            "totals": {"images": 3, "negatives": 1, "boxes": 8},
            "splits": {"train": {"images": 1}, "val": {"images": 1}, "test": {"images": 1}},
            "support": {"boxes_per_class": {cls: 2 for cls in CLASS_NAMES}},
            "labels_sha256": "labels-hash",
            "errors": [],
            "warnings": [],
        },
        "training": {
            "trainer": "ultralytics",
            "model": "yolo26n.pt",
            "task": "detect",
            "best_weights": str(weights),
            "device": "0",
            "parameters": 2572280,
            "hyperparameters": {"epochs": 80, "imgsz": 640, "batch": 8, "seed": 20261112},
            "elapsed_seconds": 1200.0,
        },
        "evaluation": {
            "split": "test",
            "metrics_json": str(metrics_json),
            "metrics": {
                "split": "test",
                "per_class": per_class,
                "macro": {"ap50": 0.75},
                "micro": {"precision": 0.8},
                "counts": {
                    "images": 3,
                    "gt_boxes": 6 if zero_support else 8,
                    "negative_images": 1,
                },
                "operating_point": {"conf": 0.25, "iou": 0.5},
                "confusion_matrix": {
                    "labels": list(CLASS_NAMES) + ["background"],
                    "matrix": [
                        [2, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 3],
                    ],
                },
                "predictions_jsonl": str(predictions),
            },
        },
        "export": {
            "onnx": str(onnx),
            "sha256": _sha(onnx),
            "export_kwargs": {
                "format": "onnx",
                "imgsz": 640,
                "batch": 1,
                "dynamic": False,
                "opset": 17,
                "simplify": False,
                "device": "cpu",
                "half": False,
            },
            "verification": {
                "ok": True,
                "input_shape": [1, 3, 640, 640],
                "output_shape": [1, 8, 8400],
                "precision": "fp32",
                "class_names": list(CLASS_NAMES),
                "end2end": False,
            },
        },
        "previews": [str(path) for path in previews],
    }
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "dataset_root": root,
        "onnx": onnx,
        "weights": weights,
        "previews": previews,
        "metrics_json": metrics_json,
    }


def make_test_dataset(root: Path) -> dict[str, Any]:
    """Blind-style test-only dataset: 5 images, 8 boxes, 1 explicit negative."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)
    entries = []
    specs = [
        ("t-1", ["leaf", "needle"]),
        ("t-2", ["flower", "inflorescence"]),
        ("t-3", ["leaf", "needle"]),
        ("t-4", ["flower", "inflorescence"]),
        ("t-5", []),
    ]
    for index, (photo_id, classes) in enumerate(specs):
        image = root / "images" / "test" / f"{photo_id}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 48), (20 + index * 10, 40, 30)).save(image)
        lines = []
        boxes = []
        for position, class_name in enumerate(classes):
            x0 = 0.1 + position * 0.2
            y0 = 0.1 + index * 0.01
            box = [x0, y0, x0 + 0.15, y0 + 0.15]
            class_id = CLASS_NAMES.index(class_name)
            lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in box))
            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "x0": box[0],
                    "y0": box[1],
                    "x1": box[2],
                    "y1": box[3],
                }
            )
        (root / "labels" / "test" / f"{photo_id}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else ""
        )
        entries.append(
            {
                "image": f"images/test/{photo_id}.jpg",
                "photo_id": photo_id,
                "observation_id": f"blind-{photo_id}",
                "negative": not boxes,
                "source_sha256": f"sha-{photo_id}",
                "boxes": boxes,
            }
        )
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schema": "arq.anatomy4.dataset/v1",
                "contract": "anatomy4-v1",
                "classes": list(CLASS_NAMES),
                "test_only": True,
                "images": entries,
            },
            indent=1,
        )
    )
    (root / "FROZEN_TEST_ONLY.json").write_text(
        json.dumps(
            {
                "schema": "arq.anatomy4.frozen-test-only/v1",
                "contract": "anatomy4-v1",
                "test_only": True,
                "allow_train": False,
                "splits": ["test"],
            },
            indent=1,
        )
    )
    return {"root": root, "boxes": 8, "items": len(entries)}


def strip_test_split(run: dict[str, Any]) -> dict[str, Any]:
    path = run["dataset_root"] / "dataset.json"
    doc = json.loads(path.read_text())
    doc["images"] = [
        entry for entry in doc["images"] if not str(entry["image"]).startswith("images/test/")
    ]
    path.write_text(json.dumps(doc))
    return run


def options(run: dict[str, Any], **overrides: Any) -> PublishOptions:
    values = {
        "manifest_path": run["manifest_path"],
        "dataset_root": run["dataset_root"],
        "dataset_id": DATASET_ID,
        "model_id": MODEL_ID,
        "name": "anatomy4 test",
        "max_previews": 4,
    }
    values.update(overrides)
    return PublishOptions(**values)


class Recorder:
    def __init__(self, *, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.calls: list[dict[str, Any]] = []
        self.overrides: dict[str, list[tuple[int, Any]]] = {}

    def add(self, method: str, path: str, *responses: tuple[int, Any]) -> None:
        self.overrides.setdefault(f"{method} {path}", []).extend(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        suffix = path[len("/api") :] if path.startswith("/api") else path
        key = f"{request.method} {suffix}"
        self.calls.append(
            {
                "method": request.method,
                "path": suffix,
                "key": key,
                "content": request.content,
                "headers": dict(request.headers),
                "url": str(request.url),
            }
        )
        if key in self.overrides and self.overrides[key]:
            status, payload = self.overrides[key].pop(0)
            return httpx.Response(status, json=payload)
        if request.method == "GET" and suffix.startswith("/fieldvision/anatomy/datasets/"):
            return httpx.Response(404, json={"detail": "not found"})
        if request.method == "GET" and suffix.startswith("/fieldvision/anatomy/models/"):
            return httpx.Response(404, json={"detail": "not found"})
        if request.method == "POST" and suffix == "/fieldvision/anatomy/datasets":
            return httpx.Response(201, json={"dataset_id": DATASET_ID})
        if request.method == "POST" and suffix.endswith("/items"):
            payload = json.loads(request.content or b"{}")
            return httpx.Response(
                200, json={"inserted": len(payload.get("items") or []), "updated": 0}
            )
        if request.method == "POST" and suffix == "/fieldvision/anatomy/models":
            return httpx.Response(201, json={"model_id": self.model_id})
        if request.method == "POST" and suffix.endswith("/artifacts"):
            return httpx.Response(201, json={"sha256": "server-sha", "storage": "db"})
        if request.method == "POST" and suffix.endswith("/metrics"):
            return httpx.Response(200, json={"model_id": self.model_id})
        if request.method == "POST" and suffix.endswith("/activate"):
            return httpx.Response(
                200,
                json={
                    "model_id": self.model_id,
                    "is_active": True,
                    "preflight": {"source": "db:anatomy4.onnx"},
                },
            )
        if request.method == "POST" and suffix == "/fieldvision/anatomy/infer":
            return httpx.Response(
                200,
                json={
                    "contract": "anatomy4-v1",
                    "model_id": self.model_id,
                    "class_names": list(CLASS_NAMES),
                    "width": 64,
                    "height": 48,
                    "input_size": 640,
                    "infer_ms": 12.5,
                    "threshold": 0.25,
                    "iou_threshold": 0.45,
                    "detection_count": 1,
                    "detections": [
                        {
                            "class": "needle",
                            "class_id": 1,
                            "bbox": [0.2, 0.2, 0.5, 0.5],
                            "confidence": 0.91,
                        }
                    ],
                    "source": "onnx",
                },
            )
        raise AssertionError(f"unexpected request {key}")


def test_auth_repr_and_env_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    auth = AuthMaterial(api_key=API_KEY, bearer=TOKEN)
    text = f"{auth!r} {auth}"
    assert API_KEY not in text and TOKEN not in text
    assert "api_key=set" in text and "bearer=set" in text

    env = {"ARQ_API_KEY": API_KEY, "ARQ_USER_TOKEN": TOKEN, "GOOGLE_ID_TOKEN": "other"}
    resolved = auth_from_env(env)
    assert resolved.api_key == API_KEY
    assert resolved.bearer == TOKEN  # ARQ_USER_TOKEN wins
    assert auth_from_env({"GOOGLE_ID_TOKEN": TOKEN}).bearer == TOKEN
    assert auth_from_env({}).secrets == ()


def test_resolve_base_variants() -> None:
    assert resolve_base("http://localhost:8000") == "http://localhost:8000/api"
    assert resolve_base("https://dev.arborphy.ai/") == "https://dev.arborphy.ai/api"
    assert resolve_base("https://dev.arborphy.ai/api") == "https://dev.arborphy.ai/api"
    with pytest.raises(PublishError):
        resolve_base("dev.arborphy.ai")


def test_dataset_payload_includes_manifest_and_provenance(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    materialization = json.loads((run["dataset_root"] / "materialization.json").read_text())
    payload = build_dataset_payload(
        run["manifest"],
        dataset_id=DATASET_ID,
        name="anatomy4 test",
        manifest_path=run["manifest_path"],
        materialization=materialization,
    )
    assert payload["contract"] == "anatomy4-v1"
    assert payload["class_names"] == list(CLASS_NAMES)
    assert payload["dataset_id"] == DATASET_ID
    assert payload["split_config"]["train"] == pytest.approx(0.70)
    assert payload["split_config"]["seed"] == "anatomy4-v1"
    assert payload["manifest"]["run_manifest_sha256"] == _sha(run["manifest_path"])
    assert payload["manifest"]["export"]["verification"]["output_shape"] == [1, 8, 8400]
    assert payload["manifest"]["trainer"]["hyperparameters"]["epochs"] == 80
    assert payload["provenance"]["inputs"]["proposals"]["sha256"]
    assert payload["provenance"]["git"]["revision"] == "a" * 40


def test_items_mark_positives_negatives_and_pending(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    materialization = json.loads((run["dataset_root"] / "materialization.json").read_text())
    plan = build_items(dataset_root=run["dataset_root"], materialization=materialization)
    assert (plan.positives, plan.negatives, plan.pending) == (2, 1, 1)
    by_key = {item["item_key"]: item for item in plan.items}
    positive = by_key["img:a"]
    assert positive["split"] == "train"
    assert positive["observation_id"] == "inat-a"
    assert positive["review_state"] == "accepted"
    assert positive["source"] == "human"
    assert positive["photo_url"] == "https://example.org/a.jpg"
    label = positive["labels"][0]
    assert label["class"] == "leaf"
    assert label["bbox"] == [0.1, 0.1, 0.3, 0.3]
    assert label["source"] == "human"
    assert label["review_state"] == "accepted"
    assert label["confidence"] == pytest.approx(0.81)
    assert label["model_id"] == "yoloe-26s-seg.pt"
    assert label["provenance"]["proposal_id"] == "p1"
    assert label["provenance"]["review"]["notes"] == "clean leaf"

    negative = by_key["img:b"]
    assert negative["negative"] is True
    assert negative["labels"] == []
    assert negative["review_state"] == "accepted"
    assert negative["provenance"]["negative_basis"] == "reviewed_all_rejected"

    pending = by_key["img:pending"]
    assert pending["labels"] == []
    assert pending["review_state"] == "proposed"
    assert pending["provenance"]["status"] == "pending_review"
    assert pending["provenance"]["proposal_ids"] == ["p2"]
    assert pending["split"] == split_for_observation("inat-pending", seed="anatomy4-v1")
    # Rejected proposal p3 never becomes a label or a pending item.
    assert all("p3" not in json.dumps(item) for item in plan.items)


def test_items_can_skip_pending(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    materialization = json.loads((run["dataset_root"] / "materialization.json").read_text())
    plan = build_items(
        dataset_root=run["dataset_root"], materialization=materialization, include_pending=False
    )
    assert plan.pending == 0
    assert len(plan.items) == 3


def test_items_read_boxes_from_labels_when_dataset_json_omits_them(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    dataset_path = run["dataset_root"] / "dataset.json"
    doc = json.loads(dataset_path.read_text())
    for entry in doc["images"]:
        entry.pop("boxes", None)
    dataset_path.write_text(json.dumps(doc))
    plan = build_items(
        dataset_root=run["dataset_root"], materialization=None, include_pending=False
    )
    assert (plan.positives, plan.negatives) == (2, 1)
    by_key = {item["item_key"]: item for item in plan.items}
    assert by_key["img:a"]["labels"][0]["class"] == "leaf"
    assert by_key["img:c"]["labels"][0]["class"] == "needle"


def test_items_refuse_dataset_json_label_disagreement(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    dataset_path = run["dataset_root"] / "dataset.json"
    doc = json.loads(dataset_path.read_text())
    doc["images"][0]["boxes"][0]["x0"] = 0.15
    dataset_path.write_text(json.dumps(doc))
    with pytest.raises(PublishError, match="disagree"):
        build_items(
            dataset_root=run["dataset_root"], materialization=None, include_pending=False
        )


def test_batching_is_bounded_and_ordered() -> None:
    plan = ItemPlan(items=[{"i": index} for index in range(1203)])
    assert [len(batch) for batch in plan.batches(500)] == [500, 500, 203]
    assert plan.batches(500)[0][0]["i"] == 0
    assert plan.batches(500)[-1][-1]["i"] == 1202
    with pytest.raises(PublishError):
        plan.batches(501)
    with pytest.raises(PublishError):
        plan.batches(0)


def test_label_limit_is_enforced(tmp_path: Path) -> None:
    run = make_run(tmp_path, extra_boxes=64)  # 65 boxes on one image
    materialization = json.loads((run["dataset_root"] / "materialization.json").read_text())
    with pytest.raises(PublishError, match="per-item limit"):
        build_items(dataset_root=run["dataset_root"], materialization=materialization)


def test_metrics_mapping_and_confusion_submatrix(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    metrics = build_metrics(run["manifest"], model_id=MODEL_ID, dataset_id=DATASET_ID)
    assert metrics["test_count"] == 8
    assert metrics["per_class"]["leaf"]["ap50"] == pytest.approx(0.75)
    assert metrics["confusion"] == [
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert "background row/column excluded" in metrics["provenance"]["confusion_note"]
    assert metrics["eval_dataset_id"] == DATASET_ID
    assert metrics["model_id"] == MODEL_ID


def test_metrics_refuse_undefined_zero_support_by_default(tmp_path: Path) -> None:
    run = make_run(tmp_path, zero_support=True)
    with pytest.raises(PublishError, match="zero-support"):
        build_metrics(run["manifest"], model_id=MODEL_ID, dataset_id=DATASET_ID)
    allowed = build_metrics(
        run["manifest"],
        model_id=MODEL_ID,
        dataset_id=DATASET_ID,
        allow_zero_support=True,
    )
    assert allowed["per_class"]["needle"]["ap50"] == 0.0
    assert allowed["provenance"]["undefined_zero_support"] == {
        "needle": ["precision", "recall", "ap50", "ap50_95"]
    }
    assert allowed["provenance"]["undefined_policy"].startswith("substituted 0.0")


def test_artifact_collection_kinds_and_preview_metadata(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    artifacts = collect_artifacts(run["manifest"], model_id=MODEL_ID)
    by_name = {artifact.filename: artifact for artifact in artifacts}
    assert by_name["anatomy4.onnx"].kind == "onnx"
    assert by_name["best.pt"].kind == "weights"
    assert by_name["metrics.json"].kind == "metrics"
    assert sum(1 for artifact in artifacts if artifact.kind == "preview") == 2
    preview_detections = json.loads(by_name["preview_detections.json"].data)
    assert preview_detections["model_id"] == MODEL_ID
    assert preview_detections["class_order"] == list(CLASS_NAMES)
    assert preview_detections["predictions_sample"][0]["class_name"] == "needle"
    assert by_name["anatomy4.onnx"].sha256 == _sha(run["onnx"])


def test_artifact_limit_is_checked_before_upload(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    run["onnx"].write_bytes(b"x" * (ARTIFACT_LIMITS["onnx"] + 1))
    run["manifest"]["export"]["sha256"] = _sha(run["onnx"])
    run["manifest_path"].write_text(json.dumps(run["manifest"]))
    with pytest.raises(PublishError, match="above the API limit"):
        collect_artifacts(run["manifest"], model_id=MODEL_ID)


def test_dry_run_makes_no_requests_and_never_prints_secrets(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run = make_run(tmp_path)
    recorder = Recorder()

    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError("dry-run must not send requests")

    caplog.set_level(logging.DEBUG)
    result = publish(
        options(run, dry_run=True),
        base="https://dev.arborphy.ai",
        auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
        transport=httpx.MockTransport(explode),
        sleep=lambda _: None,
    )
    assert result["dry_run"] is True
    assert recorder.calls == []
    text = json.dumps(result) + caplog.text
    assert API_KEY not in text
    assert TOKEN not in text
    assert result["auth"] == {"api_key": True, "bearer": True}
    assert result["base"] == "https://dev.arborphy.ai/api"
    assert result["items"]["positives"] == 2
    assert result["items"]["negatives"] == 1
    assert result["items"]["pending"] == 1
    assert [artifact["kind"] for artifact in result["artifacts"]].count("preview") == 2


def test_missing_auth_fails_before_network(tmp_path: Path) -> None:
    run = make_run(tmp_path)

    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request without auth")

    with pytest.raises(PublishError, match="ARQ_API_KEY"):
        publish(
            options(run),
            base="http://localhost:8000",
            auth=AuthMaterial(),
            transport=httpx.MockTransport(explode),
        )
    with pytest.raises(PublishError, match="ARQ_USER_TOKEN"):
        publish(
            options(run),
            base="http://localhost:8000",
            auth=AuthMaterial(api_key=API_KEY),
            transport=httpx.MockTransport(explode),
        )


def test_full_publish_call_order_payloads_and_report(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    recorder = Recorder()
    report_path = tmp_path / "publish-report.json"
    report = publish(
        options(run, report_out=report_path),
        base="http://localhost:8000",
        auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
        transport=httpx.MockTransport(recorder),
        sleep=lambda _: None,
    )
    keys = [call["key"] for call in recorder.calls]
    assert keys == [
        f"GET /fieldvision/anatomy/datasets/{DATASET_ID}",
        "POST /fieldvision/anatomy/datasets",
        "POST /fieldvision/anatomy/datasets/%s/items" % DATASET_ID,
        f"GET /fieldvision/anatomy/models/{MODEL_ID}",
        "POST /fieldvision/anatomy/models",
        "POST /fieldvision/anatomy/models/%s/artifacts" % MODEL_ID,
        "POST /fieldvision/anatomy/models/%s/artifacts" % MODEL_ID,
        "POST /fieldvision/anatomy/models/%s/artifacts" % MODEL_ID,
        "POST /fieldvision/anatomy/models/%s/artifacts" % MODEL_ID,
        "POST /fieldvision/anatomy/models/%s/artifacts" % MODEL_ID,
        "POST /fieldvision/anatomy/models/%s/artifacts" % MODEL_ID,
        "POST /fieldvision/anatomy/models/%s/metrics" % MODEL_ID,
        "POST /fieldvision/anatomy/models/%s/activate" % MODEL_ID,
        "POST /fieldvision/anatomy/infer",
    ]
    item_payload = json.loads(recorder.calls[2]["content"])
    assert len(item_payload["items"]) == 4
    artifact_call = recorder.calls[5]
    assert artifact_call["url"].endswith("kind=onnx&filename=anatomy4.onnx")
    metrics_payload = json.loads(recorder.calls[11]["content"])
    assert metrics_payload["metrics"]["per_class"]["leaf"]["ap50"] == pytest.approx(0.75)
    assert metrics_payload["metrics"]["test_count"] == 8
    infer_content = recorder.calls[-1]["content"]
    assert b'name="model_id"' in infer_content
    assert MODEL_ID.encode() in infer_content
    assert b'name="threshold"' in infer_content

    assert report["dataset_action"] == "created"
    assert report["model_action"] == "created"
    assert report["activated"] is True
    assert report["infer"]["detection_count"] == 1
    assert report["infer"]["class_counts"]["needle"] == 1
    stored = json.loads(report_path.read_text())
    assert stored["dataset_id"] == DATASET_ID
    assert API_KEY not in report_path.read_text()
    assert TOKEN not in report_path.read_text()


def test_skip_items_avoids_item_calls(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    recorder = Recorder()
    report = publish(
        options(run, skip_items=True),
        base="http://localhost:8000",
        auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
        transport=httpx.MockTransport(recorder),
        sleep=lambda _: None,
    )
    keys = [call["key"] for call in recorder.calls]
    assert not any("/items" in key for key in keys)
    assert report["items"] == {"skipped": True}


def test_existing_dataset_and_model_are_reused(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    recorder = Recorder()
    recorder.add("GET", f"/fieldvision/anatomy/datasets/{DATASET_ID}", (200, {"dataset_id": DATASET_ID}))
    recorder.add("GET", f"/fieldvision/anatomy/models/{MODEL_ID}", (200, {"model_id": MODEL_ID}))
    report = publish(
        options(run, skip_items=True),
        base="http://localhost:8000",
        auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
        transport=httpx.MockTransport(recorder),
        sleep=lambda _: None,
    )
    keys = [call["key"] for call in recorder.calls]
    assert "POST /fieldvision/anatomy/datasets" not in keys
    assert "POST /fieldvision/anatomy/models" not in keys
    assert report["dataset_action"] == "reused"
    assert report["model_action"] == "reused"


@pytest.mark.parametrize(
    "status,needle",
    [
        (401, "authentication failed"),
        (403, "arborphy.ai account"),
        (422, "validation failed"),
        (413, "too large"),
    ],
)
def test_error_messages_are_clear_and_scrubbed(
    tmp_path: Path, status: int, needle: str
) -> None:
    run = make_run(tmp_path)
    recorder = Recorder()
    recorder.add(
        "POST",
        "/fieldvision/anatomy/datasets",
        (status, {"detail": {"errors": [f"bad input {API_KEY} {TOKEN}"]}}),
    )
    with pytest.raises(PublishError) as excinfo:
        publish(
            options(run, skip_items=True),
            base="http://localhost:8000",
            auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
            transport=httpx.MockTransport(recorder),
            sleep=lambda _: None,
        )
    message = str(excinfo.value)
    assert needle in message
    assert excinfo.value.status == status
    assert API_KEY not in message
    assert TOKEN not in message


def test_retries_5xx_then_succeeds_and_does_not_retry_422(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    recorder = Recorder()
    recorder.add(
        "POST",
        "/fieldvision/anatomy/datasets",
        (503, {"detail": "warming"}),
        (503, {"detail": "warming"}),
        (201, {"dataset_id": DATASET_ID}),
    )
    sleeps: list[float] = []
    publish(
        options(run, skip_items=True, max_previews=0, allow_no_previews=True),
        base="http://localhost:8000",
        auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
        transport=httpx.MockTransport(recorder),
        sleep=sleeps.append,
    )
    dataset_posts = [call for call in recorder.calls if call["key"] == "POST /fieldvision/anatomy/datasets"]
    assert len(dataset_posts) == 3
    assert sleeps[:2] == [0.5, 1.0]

    recorder2 = Recorder()
    recorder2.add("POST", "/fieldvision/anatomy/datasets", (422, {"detail": "bad"}))
    sleeps2: list[float] = []
    with pytest.raises(PublishError):
        publish(
            options(run, skip_items=True),
            base="http://localhost:8000",
            auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
            transport=httpx.MockTransport(recorder2),
            sleep=sleeps2.append,
        )
    assert sleeps2 == []
    assert len([c for c in recorder2.calls if c["key"] == "POST /fieldvision/anatomy/datasets"]) == 1


def test_timeout_is_retried_with_a_clear_error(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        raise httpx.ConnectTimeout("connect timed out", request=request)

    sleeps: list[float] = []
    with pytest.raises(PublishError, match="connect timed out"):
        publish(
            options(run, skip_items=True),
            base="http://localhost:8000",
            auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
            transport=httpx.MockTransport(handler),
            sleep=sleeps.append,
            retries=2,
        )
    assert attempts["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_infer_smoke_validates_onnx_provenance() -> None:
    good = {
        "model_id": MODEL_ID,
        "class_names": list(CLASS_NAMES),
        "width": 64,
        "height": 48,
        "detection_count": 1,
        "detections": [{"class": "leaf", "bbox": [0.1, 0.1, 0.2, 0.2]}],
        "source": "onnx",
    }
    summary = validate_infer_response(good, model_id=MODEL_ID)
    assert summary["class_counts"]["leaf"] == 1

    with pytest.raises(PublishError, match="source='onnx'"):
        validate_infer_response({**good, "source": "stub"}, model_id=MODEL_ID)
    with pytest.raises(PublishError, match="expected"):
        validate_infer_response(good, model_id="other-model")
    with pytest.raises(PublishError, match="class order"):
        validate_infer_response({**good, "class_names": ["needle"]}, model_id=MODEL_ID)
    with pytest.raises(PublishError, match="bbox"):
        validate_infer_response(
            {**good, "detections": [{"class": "leaf", "bbox": [1, 2, 3, 4]}]},
            model_id=MODEL_ID,
        )


def test_synthetic_manifest_is_refused(tmp_path: Path) -> None:
    from experiments.anatomy4_detector.publish import plan_publish

    run = make_run(tmp_path)
    run["manifest"]["synthetic"] = True
    run["manifest_path"].write_text(json.dumps(run["manifest"]))
    with pytest.raises(PublishError, match="synthetic"):
        plan_publish(options(run))


def test_client_never_logs_auth_headers(caplog: pytest.LogCaptureFixture) -> None:
    recorder = Recorder()
    caplog.set_level(logging.DEBUG)
    client = ArqClient(
        "http://localhost:8000",
        AuthMaterial(api_key=API_KEY, bearer=TOKEN),
        transport=httpx.MockTransport(recorder),
        sleep=lambda _: None,
    )
    try:
        client.get_optional(f"/fieldvision/anatomy/datasets/{DATASET_ID}", stage="lookup")
    finally:
        client.close()
    assert API_KEY not in caplog.text
    assert TOKEN not in caplog.text
    assert "Authorization" not in caplog.text
    assert "X-API-Key" not in caplog.text

def test_contract_parity_with_the_arq_api_validators(tmp_path: Path) -> None:
    """Paylods must satisfy the actual API validators when their tree exists."""

    api_root = Path("/Users/valkyrie/dev/arborphy-bench-yolo-pipeline/arq-api")
    if not (api_root / "fieldvision" / "anatomy_contract.py").is_file():
        pytest.skip("arq-api anatomy router worktree not present")
    import sys

    if str(api_root) not in sys.path:
        sys.path.insert(0, str(api_root))
    try:
        from fieldvision import anatomy_contract as contract
    except ImportError:
        pytest.skip("arq-api anatomy contract is not importable")

    run = make_run(tmp_path)
    materialization = json.loads((run["dataset_root"] / "materialization.json").read_text())
    plan = build_items(dataset_root=run["dataset_root"], materialization=materialization)
    classes = contract.validate_class_names(list(CLASS_NAMES), CONTRACT)
    for item in plan.items:
        if item.get("labels"):
            contract.normalize_labels(
                item["labels"],
                class_names=classes,
                default_source=item.get("source"),
                default_model_id=item.get("model_id"),
                default_provenance=item.get("provenance"),
                default_review_state=item.get("review_state"),
            )
        elif item.get("negative"):
            contract.normalize_negative_item(item)
        else:
            contract.normalize_staged_empty_item(item)
    payload = build_dataset_payload(
        run["manifest"],
        dataset_id=DATASET_ID,
        name="parity",
        manifest_path=run["manifest_path"],
        materialization=materialization,
    )
    contract.validate_class_names(payload["class_names"], payload["contract"])
    contract.normalize_split_config(payload["split_config"])
    metrics = build_metrics(run["manifest"], model_id=MODEL_ID, dataset_id=DATASET_ID)
    validated = contract.validate_metrics(metrics, classes)
    assert validated["test_count"] == 8
    assert len(validated["confusion"]) == 4
    assert contract.yolo_lines(plan.items[0]["labels"], classes)


def test_items_refuse_negative_flag_with_boxes_and_unmarked_empty(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    dataset_path = run["dataset_root"] / "dataset.json"
    doc = json.loads(dataset_path.read_text())
    doc["images"][0]["negative"] = True
    dataset_path.write_text(json.dumps(doc))
    with pytest.raises(PublishError, match="marked negative"):
        build_items(
            dataset_root=run["dataset_root"], materialization=None, include_pending=False
        )

    run2 = make_run(tmp_path / "second")
    dataset_path = run2["dataset_root"] / "dataset.json"
    doc = json.loads(dataset_path.read_text())
    doc["images"][1]["negative"] = False
    dataset_path.write_text(json.dumps(doc))
    with pytest.raises(PublishError, match="not marked as a reviewed negative"):
        build_items(
            dataset_root=run2["dataset_root"],
            materialization=None,
            include_pending=False,
        )


def test_reuse_refuses_class_order_mismatch(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    recorder = Recorder()
    recorder.add(
        "GET",
        f"/fieldvision/anatomy/datasets/{DATASET_ID}",
        (200, {"dataset_id": DATASET_ID, "class_names": ["needle", "leaf", "flower", "inflorescence"]}),
    )
    with pytest.raises(PublishError, match="already exists with classes"):
        publish(
            options(run, skip_items=True),
            base="http://localhost:8000",
            auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
            transport=httpx.MockTransport(recorder),
            sleep=lambda _: None,
        )

    run2 = make_run(tmp_path / "second")
    recorder2 = Recorder()
    recorder2.add("GET", f"/fieldvision/anatomy/datasets/{DATASET_ID}", (200, {"dataset_id": DATASET_ID}))
    recorder2.add(
        "GET",
        f"/fieldvision/anatomy/models/{MODEL_ID}",
        (200, {"model_id": MODEL_ID, "class_names": ["leaf"]}),
    )
    with pytest.raises(PublishError, match="model .* already exists with classes"):
        publish(
            options(run2, skip_items=True),
            base="http://localhost:8000",
            auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
            transport=httpx.MockTransport(recorder2),
            sleep=lambda _: None,
        )


def test_publish_merges_test_dataset_with_explicit_test_split(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    strip_test_split(run)
    test = make_test_dataset(tmp_path / "blind")
    options_ = options(run, test_dataset_root=test["root"])
    plan = plan_publish(options_)
    assert plan["test_items"] is not None
    assert len(plan["test_items"].items) == 5
    assert all(item["split"] == "test" for item in plan["test_items"].items)
    assert plan["test_summary"]["counts"]["labels"] == 8
    assert plan["test_summary"]["counts"]["negatives"] == 1
    assert plan["test_summary"]["isolation"]["total_overlap"] == 0
    assert plan["metrics"]["test_count"] == 8
    assert plan["metrics"]["provenance"]["test_dataset_boxes"] == 8
    assert plan["dataset_payload"]["manifest"]["evaluation_dataset"]["counts"]["labels"] == 8

    recorder = Recorder()
    report = publish(
        options_,
        base="http://localhost:8000",
        auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
        transport=httpx.MockTransport(recorder),
        sleep=lambda _: None,
    )
    item_payload = json.loads(recorder.calls[2]["content"])
    splits = [item["split"] for item in item_payload["items"]]
    assert splits.count("test") == 5  # only the blind test items
    blind_items = [item for item in item_payload["items"] if item["item_key"].startswith("img:t-")]
    assert len(blind_items) == 5
    assert all(item["split"] == "test" for item in blind_items)
    assert sum(len(item.get("labels") or []) for item in blind_items) == 8
    assert report["test_items"]["labels"] == 8
    assert report["metrics_recorded"] is True


def test_publish_rejects_observation_overlap_with_test_dataset(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    strip_test_split(run)
    test = make_test_dataset(tmp_path / "blind")
    doc = json.loads((test["root"] / "dataset.json").read_text())
    doc["images"][0]["observation_id"] = "inat-a"
    (test["root"] / "dataset.json").write_text(json.dumps(doc))
    with pytest.raises(PublishError, match="isolation violated"):
        plan_publish(options(run, test_dataset_root=test["root"]))


def test_publish_rejects_test_box_count_mismatch(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    strip_test_split(run)
    test = make_test_dataset(tmp_path / "blind")
    (test["root"] / "labels" / "test" / "t-1.txt").write_text("")
    doc = json.loads((test["root"] / "dataset.json").read_text())
    doc["images"][0]["boxes"] = []
    doc["images"][0]["negative"] = True
    (test["root"] / "dataset.json").write_text(json.dumps(doc))
    with pytest.raises(PublishError, match="tagged boxes"):
        plan_publish(options(run, test_dataset_root=test["root"]))


def test_publish_dry_run_reports_test_items_without_requests(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    strip_test_split(run)
    test = make_test_dataset(tmp_path / "blind")
    result = publish(
        options(
            run,
            test_dataset_root=test["root"],
            dry_run=True,
        ),
        base="https://dev.arborphy.ai",
        auth=AuthMaterial(api_key=API_KEY, bearer=TOKEN),
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("no requests"))
        ),
    )
    assert result["dry_run"] is True
    assert result["test_items"]["labels"] == 8
    assert result["items"]["splits"]["test"] == 5
    assert result["metrics"]["test_count"] == 8


def test_publish_rejects_train_dataset_with_test_items_when_separate(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    test = make_test_dataset(tmp_path / "blind")
    with pytest.raises(PublishError, match="train.val only"):
        plan_publish(options(run, test_dataset_root=test["root"]))
