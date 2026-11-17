# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Single run manifest: the artifact arq-api registers.

One JSON file carries the dataset audit, per-class held-out metrics, weights and
ONNX paths with SHA-256, previews, trainer/device/hyperparameters, and the git
revision. Paths are absolute so the manifest can be read from any host.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import CLASS_NAMES, CONTRACT

MANIFEST_SCHEMA = "arq.anatomy4.detector-run/v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_info(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    revision = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    top = run("rev-parse", "--show-toplevel")
    if revision is None:
        return {"available": False}
    return {
        "available": True,
        "revision": revision,
        "branch": branch,
        "dirty": bool(status),
        "dirty_entries": len(status.splitlines()) if status else 0,
        "root": top,
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}

    def version(module: str, attribute: str = "__version__") -> str | None:
        try:
            imported = __import__(module)
            return str(getattr(imported, attribute, None))
        except Exception:
            return None

    versions["torch"] = version("torch")
    versions["ultralytics"] = version("ultralytics")
    versions["onnxruntime"] = version("onnxruntime")
    versions["onnx"] = version("onnx")
    versions["numpy"] = version("numpy")
    versions["pillow"] = version("PIL")
    return versions


def _evaluation_metrics(
    evaluation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not evaluation or not evaluation.get("per_class"):
        return None
    return {
        "split": evaluation.get("split"),
        "per_class": evaluation["per_class"],
        "macro": evaluation.get("macro"),
        "micro": evaluation.get("micro"),
        "counts": evaluation.get("counts"),
        "confusion_matrix": evaluation.get("confusion_matrix"),
        "operating_point": evaluation.get("operating_point"),
        "predictions_jsonl": evaluation.get("predictions_jsonl"),
        "tagged_bbox": evaluation.get("tagged_bbox"),
        "leaf_tagged_bbox": evaluation.get("leaf_tagged_bbox"),
        "threshold_sweep": evaluation.get("threshold_sweep"),
        "selected_threshold": evaluation.get("selected_threshold"),
        "tuning": evaluation.get("tuning"),
    }


def _artifact(kind: str, path: str | Path, **extra: Any) -> dict[str, Any]:
    target = Path(path)
    payload: dict[str, Any] = {
        "kind": kind,
        "path": str(target.resolve()),
        "exists": target.is_file(),
        "bytes": target.stat().st_size if target.is_file() else None,
        "sha256": sha256_file(target) if target.is_file() else None,
    }
    payload.update(extra)
    return payload


def build_run_manifest(
    *,
    code_root: str | Path,
    dataset_audit: Mapping[str, Any],
    dataset_audit_path: str | Path | None,
    training: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    evaluation_val: Mapping[str, Any] | None = None,
    evaluation_onnx: Mapping[str, Any] | None = None,
    export: Mapping[str, Any] | None,
    onnx_verification: Mapping[str, Any] | None,
    prepared: Mapping[str, Any] | None = None,
    extra_artifacts: Iterable[tuple[str, str | Path]] = (),
    synthetic: bool = False,
    status: str | None = None,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    previews: list[str] = []

    if dataset_audit_path:
        artifacts.append(_artifact("dataset_audit", dataset_audit_path))
    if prepared and prepared.get("data_yaml"):
        artifacts.append(_artifact("data_yaml", prepared["data_yaml"]))
    if training:
        for key in ("best_weights", "last_weights"):
            if training.get(key):
                artifacts.append(_artifact(key, training[key]))
    for named_evaluation in (evaluation, evaluation_val, evaluation_onnx):
        if not named_evaluation:
            continue
        for key in ("metrics_json", "predictions_jsonl", "ground_truth_jsonl"):
            if named_evaluation.get(key):
                artifacts.append(_artifact(key, named_evaluation[key]))
        if named_evaluation is evaluation:
            for overlay in named_evaluation.get("overlays") or []:
                if Path(overlay["overlay"]).is_file():
                    previews.append(str(Path(overlay["overlay"]).resolve()))
            for sheet in named_evaluation.get("contact_sheets") or []:
                if Path(sheet["path"]).is_file():
                    previews.append(str(Path(sheet["path"]).resolve()))
    if export and export.get("onnx"):
        artifacts.append(
            _artifact(
                "onnx",
                export["onnx"],
                export_kwargs=export.get("export_kwargs"),
                passed_nms=False,
            )
        )
    for kind, path in extra_artifacts:
        artifacts.append(_artifact(kind, path))

    test_metrics = _evaluation_metrics(evaluation)
    val_metrics = _evaluation_metrics(evaluation_val)
    onnx_metrics = _evaluation_metrics(evaluation_onnx)

    onnx_summary = None
    if onnx_verification:
        onnx_summary = {
            "ok": onnx_verification.get("ok"),
            "path": onnx_verification.get("path"),
            "sha256": onnx_verification.get("sha256"),
            "input_shape": onnx_verification.get("input", {}).get("shape"),
            "output_shape": onnx_verification.get("output", {}).get("shape"),
            "precision": onnx_verification.get("precision"),
            "class_names": onnx_verification.get("class_names"),
            "end2end": onnx_verification.get("end2end"),
        }

    registration = {
        "contract": CONTRACT,
        "task": "detect",
        "classes": list(CLASS_NAMES),
        "model_family": "yolo26n",
        "onnx": (export or {}).get("onnx"),
        "onnx_sha256": (export or {}).get("sha256"),
        "imgsz": 640,
        "batch": 1,
        "input_shape": [1, 3, 640, 640],
        "output_shape": [1, 8, 8400],
        "precision": "fp32",
        "test_metrics": {
            key: (test_metrics or {}).get(key)
            for key in ("macro", "micro", "counts")
        }
        if test_metrics
        else None,
        "artifact_count": len(artifacts),
        "synthetic": synthetic,
    }

    completed = bool(training and training.get("ok") and evaluation and export)
    return {
        "schema": MANIFEST_SCHEMA,
        "contract": CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": synthetic,
        "status": status or ("complete" if completed else "partial"),
        "git": git_info(code_root),
        "versions": package_versions(),
        "dataset": {
            "root": dataset_audit.get("dataset_root"),
            "status": dataset_audit.get("status"),
            "blocking": dataset_audit.get("blocking"),
            "split_isolation": dataset_audit.get("split_isolation"),
            "totals": dataset_audit.get("totals"),
            "splits": dataset_audit.get("splits"),
            "support": dataset_audit.get("support"),
            "labels_sha256": dataset_audit.get("labels_sha256"),
            "errors": dataset_audit.get("errors"),
            "warnings": dataset_audit.get("warnings"),
            "audit_path": str(dataset_audit_path) if dataset_audit_path else None,
            "prepared": dict(prepared) if prepared else None,
        },
        "training": dict(training) if training else None,
        "evaluation": {
            "split": (evaluation or {}).get("split"),
            "metrics": test_metrics,
            "metrics_json": (evaluation or {}).get("metrics_json"),
            "threshold_sweep": (evaluation or {}).get("threshold_sweep"),
            "selected_threshold": (evaluation or {}).get("selected_threshold"),
            "tuning": (evaluation or {}).get("tuning"),
            "leaf_tagged_bbox": (evaluation or {}).get("leaf_tagged_bbox"),
        }
        if evaluation
        else None,
        "evaluation_val": {
            "split": (evaluation_val or {}).get("split"),
            "metrics": val_metrics,
            "metrics_json": (evaluation_val or {}).get("metrics_json"),
            "leaf_tagged_bbox": (evaluation_val or {}).get("leaf_tagged_bbox"),
        }
        if evaluation_val
        else None,
        "evaluation_onnx": {
            "split": (evaluation_onnx or {}).get("split"),
            "backend": (evaluation_onnx or {}).get("backend"),
            "onnx": (evaluation_onnx or {}).get("onnx"),
            "metrics": onnx_metrics,
            "metrics_json": (evaluation_onnx or {}).get("metrics_json"),
            "leaf_tagged_bbox": (evaluation_onnx or {}).get("leaf_tagged_bbox"),
        }
        if evaluation_onnx
        else None,
        "export": {
            "onnx": (export or {}).get("onnx"),
            "sha256": (export or {}).get("sha256"),
            "export_kwargs": (export or {}).get("export_kwargs"),
            "verification": onnx_summary,
        }
        if export
        else None,
        "artifacts": artifacts,
        "previews": previews,
        "api_registration": registration,
    }


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=1) + "\n")
    return target
