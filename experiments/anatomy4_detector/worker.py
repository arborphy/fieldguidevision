# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-STAGE1-PULL-WORKER role=canonical
"""Stage-1 pull-queue worker.

Leases a training job from arq-api, materializes the exported dataset locally,
trains (pluggable backend), verifies the exported ONNX contract, evaluates on
the held-out split, renders stratified murals, and publishes the model plus
artifacts back to the API.

The worker is deliberately small and dependency-injected: tests supply an
``httpx.MockTransport`` and a deterministic ONNX trainer, so the full lease ->
dataset -> train -> evaluate -> publish loop runs without CUDA or Ultralytics.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import CLASS_NAMES, CONTRACT
from .export import EXPORT_IMGSZ, export_onnx, verify_onnx
from .onnx_eval import evaluate_onnx
from .publish import ArqClient, AuthMaterial, auth_from_env, resolve_base
from .render import build_eval_samples_manifest, render_stratified_murals

LEASE_PATH = "/fieldvision/anatomy/training-jobs/lease"
JOB_PATH = "/fieldvision/anatomy/training-jobs"
MODELS_PATH = "/fieldvision/anatomy/models"

MAX_WEIGHTS_BYTES = 48 * 1024 * 1024


class JobCancelledError(RuntimeError):
    """The server asked the worker to stop; it already marked the job cancelled."""


class WorkerError(RuntimeError):
    pass


@dataclass
class WorkerConfig:
    base: str = "http://localhost:8000"
    worker_id: str = field(default_factory=socket.gethostname)
    work_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "arq-stage1-jobs")
    media_cache_dir: Path | None = None
    contracts: tuple[str, ...] = ()
    device: str = "auto"
    poll_seconds: float = 10.0
    once: bool = False


# ── Trainer backend seam ────────────────────────────────────────────────────


class TrainerBackend(Protocol):
    def __call__(
        self,
        job: Mapping[str, Any],
        yolo_dir: Path,
        base_weights_path: Path | None,
        heartbeat: Callable[[int, Mapping[str, Any] | None], None],
    ) -> tuple[Path | None, Path, dict[str, Any]]: ...


def ultralytics_trainer(
    job: Mapping[str, Any],
    yolo_dir: Path,
    base_weights_path: Path | None,
    heartbeat: Callable[[int, Mapping[str, Any] | None], None],
) -> tuple[Path | None, Path, dict[str, Any]]:
    """Default production backend: Ultralytics train -> best.pt -> ONNX."""

    from ultralytics import YOLO

    classes = list(job.get("class_names") or CLASS_NAMES)
    imgsz = int(job.get("imgsz") or EXPORT_IMGSZ)
    model = YOLO(str(base_weights_path) if base_weights_path else (job.get("base_model") or "yolo26n.pt"))

    def _on_epoch(trainer: Any) -> None:
        metrics: dict[str, Any] = {}
        try:
            metrics = {
                key: float(value)
                for key, value in dict(getattr(trainer, "metrics", {}) or {}).items()
                if isinstance(value, (int, float))
            }
        except Exception:
            metrics = {}
        heartbeat(int(getattr(trainer, "epoch", 0)) + 1, metrics)

    model.add_callback("on_train_epoch_end", _on_epoch)
    project = yolo_dir / "runs"
    results = model.train(
        data=str(yolo_dir / "data.yaml"),
        epochs=int(job.get("epochs") or 50),
        imgsz=imgsz,
        batch=int(job.get("batch") or 8),
        device=_resolve_device(str(job.get("device") or "auto")),
        project=str(project),
        name="stage1",
        exist_ok=True,
        seed=int(job.get("seed") or 0),
        verbose=False,
    )
    trainer = getattr(model, "trainer", None)
    best = Path(getattr(trainer, "best", "") or "")
    if not best.is_file():
        save_dir = Path(getattr(trainer, "save_dir", project / "stage1"))
        best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise WorkerError("training finished without a best.pt checkpoint")
    export = export_onnx(
        best,
        classes=classes,
        contract=str(job.get("contract") or CONTRACT),
        imgsz=imgsz,
    )
    summary = {
        "epochs": int(job.get("epochs") or 50),
        "imgsz": imgsz,
        "best_weights": str(best),
        "results_dir": str(getattr(results, "save_dir", project / "stage1")),
    }
    return best, Path(export["onnx"]), summary


def _resolve_device(value: str) -> str:
    text = str(value or "auto").strip().lower()
    if text in ("auto", ""):
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return text


# ── Media + dataset helpers ─────────────────────────────────────────────────


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _suffix_for(url: str) -> str:
    match = re.search(r"\.([A-Za-z0-9]{2,5})(?:\?|$)", str(url))
    if match:
        suffix = match.group(1).lower()
        if suffix in {"jpg", "jpeg", "png", "webp", "bmp"}:
            return "." + suffix
    return ".jpg"


def _safe_name(text: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "-" for char in str(text))
    return safe.strip("-.") or "image"


def fetch_media(
    client: ArqClient,
    url: str,
    cache_dir: Path | None,
    *,
    stage: str = "media",
) -> bytes:
    """Fetch image bytes, caching by URL hash so repeat runs cost no network."""

    cache_key = _sha256_text(url)
    suffix = _suffix_for(url)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{cache_key}{suffix}"
        if cached.is_file():
            return cached.read_bytes()
    data = client.get_bytes(url, stage=stage)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{cache_key}{suffix}"
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)
    return data


def _normalized_box(box: Any, classes: Sequence[str]) -> tuple[int, float, float, float, float]:
    if hasattr(box, "class_id") and hasattr(box, "x0"):
        return (
            int(box.class_id),
            float(box.x0),
            float(box.y0),
            float(box.x1),
            float(box.y1),
        )
    if isinstance(box, Mapping):
        class_id = box.get("class_id")
        if class_id is None and box.get("class") is not None:
            class_id = list(classes).index(str(box["class"]))
        bbox = box.get("bbox")
        if bbox is not None:
            x0, y0, x1, y1 = (float(value) for value in bbox)
        else:
            x0, y0, x1, y1 = (
                float(box["x0"]),
                float(box["y0"]),
                float(box["x1"]),
                float(box["y1"]),
            )
        return (int(class_id), x0, y0, x1, y1)
    raise WorkerError(f"unsupported label box: {box!r}")


def _yolo_line_boxes(lines: Any) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for line in lines or []:
        parts = str(line).strip().split()
        if len(parts) != 5:
            continue
        class_id, cx, cy, width, height = parts
        cx, cy = float(cx), float(cy)
        width, height = float(width), float(height)
        boxes.append({
            "class_id": int(class_id),
            "bbox": [
                cx - width / 2.0,
                cy - height / 2.0,
                cx + width / 2.0,
                cy + height / 2.0,
            ],
        })
    return boxes


def _image_entries(export: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs = export.get("pairs") or []
    if pairs:
        normalized: list[dict[str, Any]] = []
        for pair in pairs:
            image = dict(pair.get("image") or {})
            label = pair.get("label") or {}
            image.setdefault("item_key", pair.get("item_id"))
            image.setdefault("split", pair.get("split"))
            image["labels"] = _yolo_line_boxes(label.get("lines"))
            normalized.append(image)
        return normalized

    entries = export.get("images") or export.get("items") or []
    normalized = []
    if isinstance(entries, Mapping):
        # {"train": [...], "val": [...], "test": [...]}
        for split, items in entries.items():
            for item in items:
                normalized.append({**dict(item), "split": item.get("split") or split})
    else:
        for item in entries:
            normalized.append(dict(item))

    labels_by_item = {
        str(row.get("item_id")): row
        for row in (export.get("files") or [])
        if row.get("item_id")
    }
    for item in normalized:
        if item.get("labels"):
            continue
        key = str(item.get("item_key") or item.get("item_id") or "")
        label_row = labels_by_item.get(key)
        if label_row:
            item["labels"] = _yolo_line_boxes(label_row.get("lines"))
    return normalized


def materialize_dataset(
    client: ArqClient,
    export: Mapping[str, Any],
    yolo_dir: Path,
    *,
    classes: Sequence[str],
    cache_dir: Path | None,
) -> dict[str, Any]:
    """Write images/{split}, labels/{split}, and data.yaml from a dataset export."""

    for split in ("train", "val", "test"):
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for index, entry in enumerate(_image_entries(export)):
        split = str(entry.get("split") or "train")
        if split not in records:
            raise WorkerError(f"unsupported split {split!r}")
        url = (
            entry.get("url")
            or entry.get("image_url")
            or entry.get("photo_url")
            or entry.get("image")
            or entry.get("path")
        )
        if not url:
            raise WorkerError(f"dataset image {index} has no URL/path")
        data = fetch_media(client, str(url), cache_dir, stage=f"dataset.image[{split}]")
        item_key = str(entry.get("item_key") or entry.get("stem") or f"{split}-{index}")
        suffix = _suffix_for(str(url))
        filename = f"{_safe_name(item_key)[:80]}-{_sha256_text(str(url))[:10]}{suffix}"
        image_path = yolo_dir / "images" / split / filename
        image_path.write_bytes(data)

        labels = entry.get("labels") or entry.get("boxes") or entry.get("annotations") or []
        lines: list[str] = []
        normalized_boxes: list[dict[str, float | int]] = []
        for box in labels:
            class_id, x0, y0, x1, y1 = _normalized_box(box, classes)
            width = max(0.0, x1 - x0)
            height = max(0.0, y1 - y0)
            lines.append(
                f"{class_id} {(x0 + x1) / 2.0:.6f} {(y0 + y1) / 2.0:.6f} "
                f"{width:.6f} {height:.6f}"
            )
            normalized_boxes.append(
                {"class_id": class_id, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
            )
        label_path = yolo_dir / "labels" / split / (image_path.stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        records[split].append(
            {
                "item_key": item_key,
                "stem": image_path.stem,
                "observation_id": entry.get("observation_id"),
                "photo_url": entry.get("photo_url") or url,
                "image": str(image_path),
                "prepared_image": str(image_path),
                "split": split,
                "width": entry.get("width"),
                "height": entry.get("height"),
                "ground_truth": normalized_boxes,
            }
        )

    names = "\n".join(
        f"  {index}: {json.dumps(str(name))}" for index, name in enumerate(classes)
    )
    (yolo_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {json.dumps(str(yolo_dir))}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                names,
                "",
            ]
        )
    )
    return {
        "root": str(yolo_dir),
        "data_yaml": str(yolo_dir / "data.yaml"),
        "counts": {split: len(items) for split, items in records.items()},
        "records": records,
    }


def _api_metrics(payload: Mapping[str, Any], classes: Sequence[str]) -> dict[str, Any]:
    out = dict(payload)
    per_class: dict[str, Any] = {}
    for name in classes:
        entry = dict((payload.get("per_class") or {}).get(name) or {})
        for key in ("precision", "recall", "f1", "ap50", "ap50_95"):
            value = entry.get(key)
            entry[key] = (
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else 0.0
            )
        per_class[name] = entry
    out["per_class"] = per_class

    confusion = payload.get("confusion_matrix")
    if isinstance(confusion, Mapping):
        matrix = confusion.get("matrix")
        labels = confusion.get("labels")
        if isinstance(matrix, list) and isinstance(labels, list):
            index = {str(label): position for position, label in enumerate(labels)}
            if all(name in index for name in classes):
                out["confusion"] = [
                    [int(matrix[index[truth]][index[predicted]]) for predicted in classes]
                    for truth in classes
                ]
                out["confusion_classes"] = list(classes)
    return out


# ── Worker ──────────────────────────────────────────────────────────────────

class Stage1Worker:
    def __init__(
        self,
        config: WorkerConfig,
        *,
        client: ArqClient | None = None,
        trainer_backend: TrainerBackend | None = None,
        auth: AuthMaterial | None = None,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.auth = auth or auth_from_env()
        self.client = client or ArqClient(
            resolve_base(config.base), self.auth, transport=transport, sleep=sleep
        )
        self.trainer_backend: TrainerBackend = trainer_backend or ultralytics_trainer
        self.sleep = sleep
        self._log = log or (lambda message: None)

    # -- helpers ------------------------------------------------------------

    def _heartbeat(
        self, job_id: str, status: str, extra: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"worker_id": self.config.worker_id, "status": status}
        if extra:
            payload.update(extra)
        response = self.client.post_json(
            f"{JOB_PATH}/{job_id}/heartbeat", payload, stage=f"job.heartbeat[{status}]"
        )
        job = response.get("job") if isinstance(response.get("job"), Mapping) else {}
        if response.get("cancel_requested") or job.get("cancel_requested"):
            raise JobCancelledError(f"job {job_id} was cancelled by the server")
        return response

    def _lease_job(self) -> dict[str, Any] | None:
        payload: dict[str, Any] = {"worker_id": self.config.worker_id}
        if self.config.contracts:
            payload["contracts"] = list(self.config.contracts)
        response = self.client.post_json(LEASE_PATH, payload, stage="job.lease")
        job = response.get("job")
        if job is None:
            return None
        if not isinstance(job, Mapping):
            raise WorkerError(f"lease returned malformed job: {job!r}")
        return dict(job)

    def _upload(self, model_id: str, path: Path, kind: str, content_type: str) -> dict[str, Any]:
        return self.client.post_multipart(
            f"{MODELS_PATH}/{model_id}/artifacts",
            filename=path.name,
            data=path.read_bytes(),
            content_type=content_type,
            params={"kind": kind, "filename": path.name},
            stage=f"artifact.{kind}",
        )

    def _scrub(self, text: str) -> str:
        scrubber = getattr(self.client, "_scrub", None)
        if callable(scrubber):
            return str(scrubber(text))
        for secret in self.auth.secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    # -- main loop ----------------------------------------------------------

    def run_once(self) -> dict[str, Any] | None:
        job = self._lease_job()
        if job is None:
            return None
        job_id = str(job.get("job_id") or job.get("id") or "")
        if not job_id:
            raise WorkerError("leased job has no job_id")
        classes = tuple(job.get("class_names") or CLASS_NAMES)
        contract = str(job.get("contract") or CONTRACT)
        imgsz = int(job.get("imgsz") or EXPORT_IMGSZ)
        job_dir = Path(self.config.work_dir).expanduser() / job_id
        yolo_dir = job_dir / "dataset"
        job_dir.mkdir(parents=True, exist_ok=True)

        report: dict[str, Any] = {
            "schema": "arq.stage1.worker-report/v1",
            "job_id": job_id,
            "worker_id": self.config.worker_id,
            "contract": contract,
            "class_names": list(classes),
        }
        try:
            self._heartbeat(job_id, "preparing")
            export = self.client.get(
                f"{JOB_PATH}/{job_id}/dataset-export?worker_id={self.config.worker_id}",
                stage="job.dataset-export",
            )
            dataset = materialize_dataset(
                self.client,
                export,
                yolo_dir,
                classes=classes,
                cache_dir=self.config.media_cache_dir,
            )
            report["dataset"] = {
                "counts": dataset["counts"],
                "data_yaml": dataset["data_yaml"],
            }

            base_weights_path: Path | None = None
            base_weights = str(job.get("base_weights") or "")
            if base_weights.startswith("model:"):
                data = self.client.get_bytes(
                    f"{JOB_PATH}/{job_id}/base-weights?worker_id={self.config.worker_id}",
                    stage="job.base-weights",
                )
                base_weights_path = job_dir / "warm_start.pt"
                base_weights_path.write_bytes(data)
            report["base_weights"] = base_weights or None

            self._heartbeat(job_id, "training")
            best_pt, onnx_path, train_summary = self.trainer_backend(
                job,
                yolo_dir,
                base_weights_path,
                lambda epoch, metrics=None: self._heartbeat(
                    job_id, "training", {"epoch": epoch, **(dict(metrics or {}))}
                ),
            )
            report["train"] = train_summary

            self._heartbeat(job_id, "evaluating")
            verification = verify_onnx(
                onnx_path, classes=classes, contract=contract, imgsz=imgsz
            )
            if not verification.get("ok"):
                failed = [
                    item for item in verification.get("checks", []) if not item.get("ok")
                ]
                raise WorkerError(f"ONNX verification failed: {failed}")

            eval_split = "test" if dataset["counts"].get("test") else "val"
            eval_records = dataset["records"].get(eval_split) or dataset["records"]["train"]
            if not eval_records:
                raise WorkerError("dataset export contains no evaluable images")
            evaluation = evaluate_onnx(
                onnx_path=onnx_path,
                records=eval_records,
                classes=classes,
                contract=contract,
                split=eval_split,
                imgsz=imgsz,
                conf_threshold=float(job.get("conf_threshold") or 0.25),
            )
            murals = render_stratified_murals(
                evaluation["records"],
                job_dir / "murals",
                contract=contract,
                classes=classes,
                conf_threshold=float(job.get("conf_threshold") or 0.25),
            )
            metrics_payload = _api_metrics(evaluation["metrics"], classes)
            metrics_payload["test_count"] = len(eval_records)
            metrics_payload["eval_split_used"] = eval_split
            manifest = build_eval_samples_manifest(
                evaluation["records"],
                contract=contract,
                model_id=job.get("model_name"),
                dataset_id=job.get("dataset_id"),
                split=eval_split,
                conf_threshold=float(job.get("conf_threshold") or 0.25),
                classes=classes,
                murals=murals,
            )
            metrics_path = job_dir / "metrics.json"
            manifest_path = job_dir / "eval_samples.json"
            metrics_path.write_text(json.dumps(metrics_payload, indent=1) + "\n")
            manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
            report["evaluation"] = {
                "split": eval_split,
                "macro": metrics_payload.get("macro"),
                "micro": metrics_payload.get("micro"),
                "sampled_images": manifest["bucket_counts"]["sampled_images"],
            }
            report["onnx"] = {
                "path": str(onnx_path),
                "sha256": verification.get("sha256"),
                "ok": True,
            }

            self._heartbeat(job_id, "publishing")
            model_response = self.client.post_json(
                MODELS_PATH,
                {
                    "name": job.get("model_name") or f"stage1-{job_id}",
                    "contract": contract,
                    "class_names": list(classes),
                    "dataset_id": job.get("dataset_id"),
                    "training_job_id": job_id,
                    "provenance": {
                        "worker_id": self.config.worker_id,
                        "contract": contract,
                        "class_names": list(classes),
                        "eval_split_used": eval_split,
                        "imgsz": imgsz,
                        "onnx_sha256": verification.get("sha256"),
                        "dataset_counts": dataset["counts"],
                        "train": train_summary,
                        "eval_macro": metrics_payload.get("macro"),
                        "eval_micro": metrics_payload.get("micro"),
                    },
                },
                stage="model.create",
            )
            model_id = str(
                model_response.get("model_id")
                or model_response.get("id")
                or job.get("model_id")
                or f"model-{job_id}"
            )
            report["model_id"] = model_id

            uploads = [
                self._upload(model_id, Path(onnx_path), "onnx", "application/onnx")
            ]
            if best_pt is not None:
                best = Path(best_pt)
                if best.is_file() and best.stat().st_size <= MAX_WEIGHTS_BYTES:
                    uploads.append(self._upload(model_id, best, "weights", "application/octet-stream"))
            uploads.append(self._upload(model_id, metrics_path, "metrics", "application/json"))
            uploads.append(self._upload(model_id, manifest_path, "manifest", "application/json"))
            for mural in murals:
                uploads.append(
                    self._upload(model_id, Path(mural["path"]), "preview", "image/jpeg")
                )
            report["uploads"] = [
                {"filename": item.get("filename"), "kind": item.get("kind")}
                for item in uploads
            ]

            self.client.post_json(
                f"{MODELS_PATH}/{model_id}/metrics",
                {"metrics": metrics_payload},
                stage="model.metrics",
            )
            complete = self.client.post_json(
                f"{JOB_PATH}/{job_id}/complete",
                {
                    "worker_id": self.config.worker_id,
                    "outcome": "succeeded",
                    "result_model_id": model_id,
                    "onnx_parity_passed": True,
                    "summary": {
                        "eval_split_used": eval_split,
                        "macro": metrics_payload.get("macro"),
                        "micro": metrics_payload.get("micro"),
                        "sampled_images": manifest["bucket_counts"]["sampled_images"],
                        "murals": len(murals),
                    },
                },
                stage="job.complete",
            )
            report["status"] = "succeeded"
            report["complete"] = complete
            return report
        except JobCancelledError:
            report["status"] = "cancelled"
            return report
        except Exception as exc:  # noqa: BLE001 - worker must report, never leak secrets
            message = self._scrub(f"{type(exc).__name__}: {exc}")
            report["status"] = "failed"
            report["error"] = message
            try:
                self.client.post_json(
                    f"{JOB_PATH}/{job_id}/complete",
                    {
                        "worker_id": self.config.worker_id,
                        "outcome": "failed",
                        "error_text": message,
                    },
                    stage="job.complete.failed",
                )
            except Exception as report_exc:  # noqa: BLE001
                report["complete_error"] = self._scrub(str(report_exc))
            return report

    def run_loop(self) -> int:
        while True:
            result = self.run_once()
            if self.config.once:
                return 0
            if result is None:
                self.sleep(max(0.0, float(self.config.poll_seconds)))


def run_once(
    config: WorkerConfig,
    *,
    client: ArqClient | None = None,
    trainer_backend: TrainerBackend | None = None,
    auth: AuthMaterial | None = None,
    transport: Any = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any] | None:
    """Convenience wrapper for one lease/train/publish cycle."""

    worker = Stage1Worker(
        config,
        client=client,
        trainer_backend=trainer_backend,
        auth=auth,
        transport=transport,
        sleep=sleep,
    )
    return worker.run_once()
