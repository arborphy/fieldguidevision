# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Publish an anatomy4 detector run to arq-api.

Implements the anatomy4 sprint seam in ``arq-api/routers/fieldvision_anatomy.py``
(contract ``anatomy4-v1``): create a dataset with its manifest/provenance,
upsert reviewed items (accepted positives, accepted negatives, and pending
unreviewed items), register a model, upload ONNX / best.pt / metrics / preview
artifacts plus preview-detection metadata, record per-class metrics, activate,
and run one REAL inference smoke test on a held-out image.

Auth (read from the environment, never printed):

* ``ARQ_BASE`` — default ``http://localhost:8000``; may be
  ``https://dev.arborphy.ai`` or already end in ``/api``.
* ``ARQ_API_KEY`` — client gate (``X-API-Key``).
* ``ARQ_USER_TOKEN`` or ``GOOGLE_ID_TOKEN`` — ``Authorization: Bearer``.

Model writes/activation/inference require an arborphy.ai account (403
otherwise). Secrets are held in :class:`AuthMaterial`, whose ``repr`` is
redacted; every error path scrubs secret values before formatting a message.

Nothing here sends requests unless :func:`publish` is called without
``dry_run``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    CLASS_NAMES,
    CONTRACT,
    ContractError,
    parse_label_text,
    split_for_observation,
)
from .dataset import assert_disjoint_datasets

log = logging.getLogger(__name__)

API_PREFIX = "/api"
CONTRACT_ID = CONTRACT

# Limits mirrored from arq-api/fieldvision/anatomy_store.py (never exceed them;
# the client refuses locally before uploading so a failure is cheap and clear).
MAX_BATCH_ITEMS = 500
MAX_LABELS_PER_ITEM = 64
ARTIFACT_LIMITS: dict[str, int] = {
    "onnx": 48 * 1024 * 1024,
    "weights": 48 * 1024 * 1024,
    "preview": 4 * 1024 * 1024,
    "manifest": 4 * 1024 * 1024,
    "metrics": 4 * 1024 * 1024,
    "artifact": 8 * 1024 * 1024,
}
MAX_INFER_BYTES = 40 * 1024 * 1024
DEFAULT_BATCH_SIZE = 200
DEFAULT_MAX_PREVIEWS = 8
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


class PublishError(RuntimeError):
    """A clear, non-secret-bearing failure from planning or publishing."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        status: int | None = None,
        detail: Any = None,
    ) -> None:
        self.stage = stage
        self.status = status
        self.detail = detail
        super().__init__(message)


# ── Auth ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthMaterial:
    api_key: str | None = None
    bearer: str | None = None

    def __repr__(self) -> str:  # never print secret values
        return (
            "AuthMaterial(api_key="
            + ("set" if self.api_key else "unset")
            + ", bearer="
            + ("set" if self.bearer else "unset")
            + ")"
        )

    __str__ = __repr__

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(value for value in (self.api_key, self.bearer) if value)

    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.bearer:
            headers["Authorization"] = f"Bearer {self.bearer}"
        return headers

    def require(self) -> None:
        if not self.api_key:
            raise PublishError(
                "ARQ_API_KEY is not set (required as the X-API-Key client gate)",
                stage="auth",
            )
        allow_unsigned = os.environ.get("ARQ_ALLOW_UNSIGNED_LOCAL") == "1"
        if not self.bearer and not allow_unsigned:
            raise PublishError(
                "No user token: set ARQ_USER_TOKEN or GOOGLE_ID_TOKEN "
                "(sent as Authorization: Bearer)",
                stage="auth",
            )


def auth_from_env(env: Mapping[str, str] | None = None) -> AuthMaterial:
    values = os.environ if env is None else env
    bearer = values.get("ARQ_USER_TOKEN") or values.get("GOOGLE_ID_TOKEN")
    return AuthMaterial(
        api_key=(values.get("ARQ_API_KEY") or "").strip() or None,
        bearer=(bearer or "").strip() or None,
    )


def resolve_base(raw: str | None) -> str:
    base = (raw or "http://localhost:8000").strip().rstrip("/")
    if not re.match(r"^https?://", base):
        raise PublishError(
            f"ARQ_BASE must start with http:// or https:// (got {base!r})",
            stage="config",
        )
    if base.endswith(API_PREFIX):
        return base
    return base + API_PREFIX


def auth_summary(auth: AuthMaterial) -> dict[str, bool]:
    return {"api_key": bool(auth.api_key), "bearer": bool(auth.bearer)}


# ── HTTP client ─────────────────────────────────────────────────────────────


class ArqClient:
    """httpx client with explicit timeouts, bounded retries, clear errors."""

    RETRY_STATUSES = frozenset({429, 502, 503, 504})
    MAX_DETAIL_CHARS = 800

    def __init__(
        self,
        base_url: str,
        auth: AuthMaterial,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff: float = 0.5,
    ) -> None:
        import httpx

        self.base_url = resolve_base(base_url)
        self.auth = auth
        self.retries = max(0, int(retries))
        self.sleep = sleep
        self.backoff = backoff
        self._httpx = httpx
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ArqClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- public verbs -------------------------------------------------------

    def get_optional(self, path: str, *, stage: str) -> dict[str, Any] | None:
        response = self._request("GET", path, stage=stage, allow_404=True)
        if response.status_code == 404:
            return None
        return self._json(response, stage)

    def get(self, path: str, *, stage: str) -> dict[str, Any]:
        return self._json(self._request("GET", path, stage=stage), stage)

    def post_json(
        self, path: str, payload: Mapping[str, Any], *, stage: str
    ) -> dict[str, Any]:
        return self._json(
            self._request("POST", path, json=payload, stage=stage), stage
        )

    def post_multipart(
        self,
        path: str,
        *,
        filename: str,
        data: bytes,
        content_type: str,
        params: Mapping[str, Any] | None = None,
        form: Mapping[str, Any] | None = None,
        stage: str,
    ) -> dict[str, Any]:
        files = {"file": (filename, data, content_type)}
        return self._json(
            self._request(
                "POST",
                path,
                files=files,
                params=params,
                data=form,
                stage=stage,
            ),
            stage,
        )

    # -- internals ----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        stage: str,
        json: Mapping[str, Any] | None = None,
        files: Any = None,
        data: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Any:
        import httpx

        url = path if path.startswith("/") else f"/{path}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=self.auth.headers(),
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self.retries:
                    self._wait(attempt)
                    continue
                raise PublishError(
                    self._scrub(
                        f"{stage}: {method} {self.base_url}{url} failed after "
                        f"{self.retries + 1} attempt(s): {type(exc).__name__}: {exc}"
                    ),
                    stage=stage,
                ) from exc
            log.info(
                "arq %s %s -> %s (%s)", method, url, response.status_code, stage
            )
            if response.status_code in self.RETRY_STATUSES and attempt < self.retries:
                self._wait(attempt, response)
                continue
            if response.status_code == 404 and allow_404:
                return response
            if response.status_code >= 400:
                raise self._error(response, method=method, url=url, stage=stage)
            return response
        raise PublishError(
            self._scrub(f"{stage}: request failed: {last_error}"), stage=stage
        )

    def _wait(self, attempt: int, response: Any = None) -> None:
        delay = self.backoff * (2**attempt)
        if response is not None:
            header = response.headers.get("retry-after")
            if header:
                try:
                    delay = max(delay, min(float(header), 30.0))
                except ValueError:
                    pass
        self.sleep(delay)

    def _json(self, response: Any, stage: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            raise PublishError(
                self._scrub(
                    f"{stage}: expected JSON from {response.request.method} "
                    f"{response.request.url}, got "
                    f"{response.headers.get('content-type') or 'no content-type'}"
                ),
                stage=stage,
                status=response.status_code,
            )
        if not isinstance(payload, dict):
            return {"data": payload}
        return payload

    def _error(self, response: Any, *, method: str, url: str, stage: str) -> PublishError:
        status = response.status_code
        detail = self._detail(response)
        if status == 401:
            message = (
                f"{stage}: authentication failed (401). Set ARQ_API_KEY (X-API-Key "
                "client gate) and ARQ_USER_TOKEN or GOOGLE_ID_TOKEN (Bearer user "
                "token). Token values are never logged."
            )
        elif status == 403:
            message = (
                f"{stage}: permission denied (403). Model registration, artifact "
                "upload, activation, and inference require an arborphy.ai account "
                "or employee role."
            )
        elif status == 422:
            message = f"{stage}: validation failed (422): {detail}"
        elif status == 413:
            message = f"{stage}: artifact/request too large (413): {detail}"
        elif status == 404:
            message = f"{stage}: not found (404) for {method} {url}: {detail}"
        else:
            message = f"{stage}: HTTP {status} from {method} {url}: {detail}"
        return PublishError(
            self._scrub(message), stage=stage, status=status, detail=detail
        )

    def _detail(self, response: Any) -> str:
        text = ""
        try:
            payload = response.json()
            if isinstance(payload, Mapping):
                detail = payload.get("detail", payload)
                if isinstance(detail, Mapping) and detail.get("errors"):
                    text = "; ".join(str(item) for item in detail["errors"])
                else:
                    text = json.dumps(detail, ensure_ascii=False)
            else:
                text = json.dumps(payload, ensure_ascii=False)
        except ValueError:
            text = (response.text or "").strip()
        text = " ".join(text.split())
        return self._scrub(text)[: self.MAX_DETAIL_CHARS] or "(no detail)"

    def _scrub(self, text: str) -> str:
        for secret in self.auth.secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text


# ── Plan builders ───────────────────────────────────────────────────────────


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishError(f"cannot read JSON {path}: {exc}", stage="plan") from exc
    if not isinstance(data, dict):
        raise PublishError(f"{path} must contain a JSON object", stage="plan")
    return data


def _slug(text: str, max_len: int = 28) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return (cleaned or "anatomy4")[:max_len].strip("-") or "anatomy4"


def stable_dataset_id(manifest: Mapping[str, Any], name: str) -> str:
    audit = manifest.get("dataset") or {}
    key = f"{audit.get('labels_sha256') or ''}:{audit.get('root') or ''}:{name}"
    return f"anat-ds-a4-{_slug(name, 18)}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def stable_model_id(manifest: Mapping[str, Any], name: str) -> str:
    export = manifest.get("export") or {}
    key = f"{export.get('sha256') or ''}:{name}"
    return f"anat-model-a4-{_slug(name, 14)}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def build_dataset_payload(
    manifest: Mapping[str, Any],
    *,
    dataset_id: str,
    name: str,
    manifest_path: str | Path | None = None,
    materialization: Mapping[str, Any] | None = None,
    test_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit = manifest.get("dataset") or {}
    split_config = {"train": 0.70, "val": 0.15, "test": 0.15, "seed": CONTRACT_ID}
    if materialization:
        seed = (materialization.get("selection") or {}).get("seed")
        if seed:
            split_config["seed"] = str(seed)
    manifest_doc = {
        "schema": "arq.anatomy4.publisher-manifest/v1",
        "contract": CONTRACT_ID,
        "source": "experiments.anatomy4_detector.publish",
        "run_manifest_schema": manifest.get("schema"),
        "run_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path else None
        ),
        "generated_at": manifest.get("generated_at"),
        "git": manifest.get("git"),
        "versions": manifest.get("versions"),
        "dataset": {
            "root": audit.get("root"),
            "status": audit.get("status"),
            "split_isolation": audit.get("split_isolation"),
            "totals": audit.get("totals"),
            "splits": audit.get("splits"),
            "support": audit.get("support"),
            "labels_sha256": audit.get("labels_sha256"),
            "warnings": [item.get("code") for item in (audit.get("warnings") or [])],
            "errors": [item.get("code") for item in (audit.get("errors") or [])],
        },
        "materialization": _materialization_summary(materialization),
        "evaluation_dataset": dict(test_summary) if test_summary else None,
        "class_order": list(CLASS_NAMES),
        "trainer": _trainer_summary(manifest),
        "export": _export_summary(manifest),
        "evaluation": _evaluation_summary(manifest),
    }
    return {
        "dataset_id": dataset_id,
        "name": name,
        "contract": CONTRACT_ID,
        "class_names": list(CLASS_NAMES),
        "manifest": manifest_doc,
        "split_config": split_config,
        "provenance": {
            "publisher": "experiments.anatomy4_detector.publish",
            "adapter": (materialization or {}).get("schema"),
            "inputs": ((materialization or {}).get("inputs") or {}),
            "selection": ((materialization or {}).get("selection") or {}),
            "review": ((materialization or {}).get("review") or {}),
            "evaluation_dataset": dict(test_summary) if test_summary else None,
            "git": manifest.get("git"),
            "synthetic": bool(manifest.get("synthetic")),
        },
        "status": "draft",
    }


def _materialization_summary(materialization: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not materialization:
        return None
    return {
        "schema": materialization.get("schema"),
        "counts": materialization.get("counts"),
        "partial_labeling": materialization.get("partial_labeling"),
        "review": materialization.get("review"),
        "selection": materialization.get("selection"),
        "model_ids": materialization.get("model_ids"),
        "inputs": {
            key: (
                {"sha256": value.get("sha256"), "path": value.get("path")}
                if isinstance(value, Mapping)
                else value
            )
            for key, value in (materialization.get("inputs") or {}).items()
        },
    }


def _trainer_summary(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    training = manifest.get("training") or {}
    if not training:
        return None
    return {
        "trainer": training.get("trainer"),
        "model": training.get("model"),
        "task": training.get("task"),
        "device": training.get("device"),
        "parameters": training.get("parameters"),
        "elapsed_seconds": training.get("elapsed_seconds"),
        "hyperparameters": training.get("hyperparameters"),
        "vram": training.get("vram"),
    }


def _export_summary(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    export = manifest.get("export") or {}
    if not export:
        return None
    verification = export.get("verification") or {}
    return {
        "onnx": export.get("onnx"),
        "sha256": export.get("sha256"),
        "export_kwargs": export.get("export_kwargs"),
        "verification": {
            "ok": verification.get("ok"),
            "input_shape": verification.get("input_shape"),
            "output_shape": verification.get("output_shape"),
            "precision": verification.get("precision"),
            "class_names": verification.get("class_names"),
            "end2end": verification.get("end2end"),
        },
    }


def _evaluation_summary(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    onnx_evaluation = manifest.get("evaluation_onnx") or {}
    evaluation = onnx_evaluation or (manifest.get("evaluation") or {})
    metrics = evaluation.get("metrics") or {}
    metrics_source = "onnx" if onnx_evaluation else "pytorch"
    if not evaluation:
        return None
    return {
        "split": evaluation.get("split"),
        "counts": metrics.get("counts"),
        "macro": metrics.get("macro"),
        "micro": metrics.get("micro"),
        "operating_point": metrics.get("operating_point"),
    }


def build_model_payload(
    manifest: Mapping[str, Any],
    *,
    model_id: str,
    name: str,
    dataset_id: str,
) -> dict[str, Any]:
    training = manifest.get("training") or {}
    return {
        "model_id": model_id,
        "name": name,
        "contract": CONTRACT_ID,
        "class_names": list(CLASS_NAMES),
        "dataset_id": dataset_id,
        "status": "registered",
        "artifact_meta": {
            "publisher": "experiments.anatomy4_detector.publish",
            "run_manifest_schema": manifest.get("schema"),
            "trainer": training.get("trainer"),
            "model": training.get("model"),
            "task": training.get("task"),
        },
        "provenance": {
            "publisher": "experiments.anatomy4_detector.publish",
            "git": manifest.get("git"),
            "versions": manifest.get("versions"),
            "training": _trainer_summary(manifest),
            "export": _export_summary(manifest),
            "synthetic": bool(manifest.get("synthetic")),
        },
    }


# ── Items ───────────────────────────────────────────────────────────────────


@dataclass
class ItemPlan:
    items: list[dict[str, Any]] = field(default_factory=list)
    positives: int = 0
    negatives: int = 0
    pending: int = 0
    rejected_proposals: int = 0
    skipped_images: list[str] = field(default_factory=list)
    pending_source: str | None = None

    def batches(self, size: int) -> list[list[dict[str, Any]]]:
        size = int(size)
        if size < 1 or size > MAX_BATCH_ITEMS:
            raise PublishError(
                f"batch size must be 1..{MAX_BATCH_ITEMS} (got {size})", stage="items"
            )
        return [
            self.items[index : index + size]
            for index in range(0, len(self.items), size)
        ]

    def summary(self) -> dict[str, Any]:
        splits: Counter[str] = Counter(str(item.get("split") or "train") for item in self.items)
        return {
            "items": len(self.items),
            "positives": self.positives,
            "negatives": self.negatives,
            "pending": self.pending,
            "rejected_proposals_excluded": self.rejected_proposals,
            "skipped_images": self.skipped_images,
            "pending_source": self.pending_source,
            "labels": sum(len(item.get("labels") or []) for item in self.items),
            "splits": dict(sorted(splits.items())),
        }


def _split_from_image_path(relative: str) -> str:
    parts = Path(relative).parts
    if len(parts) >= 3 and parts[0] == "images" and parts[1] in {"train", "val", "test"}:
        return parts[1]
    raise PublishError(
        f"dataset image {relative!r} is not under images/<train|val|test>/",
        stage="items",
    )


def _boxes_from_labels(root: Path, relative: str, split: str) -> list[Any]:
    stem = Path(relative).stem
    path = root / "labels" / split / f"{stem}.txt"
    if not path.is_file():
        raise PublishError(
            f"missing label file for {relative} ({path}); an absent file is not "
            "an implicit negative",
            stage="items",
        )
    try:
        return parse_label_text(path.read_text(), source=str(path))
    except ContractError as exc:
        raise PublishError(f"invalid label file: {exc}", stage="items") from exc


def _entry_boxes(
    entry: Mapping[str, Any], root: Path, relative: str, split: str
) -> list[Any]:
    """Boxes from the label file (truth); dataset.json boxes are cross-checked."""
    parsed = _boxes_from_labels(root, relative, split)
    declared = entry.get("boxes")
    if declared:
        declared_keys = sorted(
            (
                int(box["class_id"]),
                round(float(box["x0"]), 6),
                round(float(box["y0"]), 6),
                round(float(box["x1"]), 6),
                round(float(box["y1"]), 6),
            )
            for box in declared
        )
        parsed_keys = sorted(
            (box.class_id, *(round(value, 6) for value in box.as_tuple()))
            for box in parsed
        )
        if declared_keys != parsed_keys:
            raise PublishError(
                f"dataset.json boxes and labels/{split}/{Path(relative).stem}.txt "
                f"disagree for {relative}; refusing to guess which is truth",
                stage="items",
            )
    return list(parsed)


def _photo_id(entry: Mapping[str, Any], relative: str) -> str:
    if entry.get("photo_id"):
        return str(entry["photo_id"])
    return Path(relative).stem


def _provenance_index(dataset_root: Path) -> dict[tuple[str, int, tuple[float, ...]], dict[str, Any]]:
    """Index provenance.jsonl by (image relpath, class id, rounded bbox)."""
    path = dataset_root / "provenance.jsonl"
    index: dict[tuple[str, int, tuple[float, ...]], dict[str, Any]] = {}
    if not path.is_file():
        return index
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        bbox = tuple(round(float(value), 6) for value in row.get("xyxy_norm") or [])
        key = (str(row.get("image")), int(row.get("class_id", -1)), bbox)
        index[key] = row
    return index


def _source_urls(proposals_path: Path | None) -> dict[str, str]:
    """Map photo id -> original source URL from the prelabel image records."""
    if proposals_path is None or not proposals_path.is_file():
        return {}
    try:
        doc = json.loads(proposals_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    urls: dict[str, str] = {}
    for record in doc.get("images") or []:
        if not isinstance(record, Mapping):
            continue
        photo_id = str(record.get("image_id") or record.get("photo_id") or "")
        download = record.get("download") or {}
        url = download.get("source_ref") or record.get("url")
        if photo_id and isinstance(url, str) and url.startswith(("http://", "https://")):
            urls[photo_id] = url
    return urls


def _label_from_entry(
    entry: Mapping[str, Any],
    box: Mapping[str, Any],
    provenance_row: Mapping[str, Any] | None,
    detector: str | None,
) -> dict[str, Any]:
    bbox = [
        round(float(box["x0"]), 6),
        round(float(box["y0"]), 6),
        round(float(box["x1"]), 6),
        round(float(box["y1"]), 6),
    ]
    review = (provenance_row or {}).get("review") or {}
    provenance = {
        "schema": "arq.anatomy4.published-label/v1",
        "proposal_id": (provenance_row or {}).get("proposal_id"),
        "detector": ((provenance_row or {}).get("model") or {}).get("detector") or detector,
        "box_score": (provenance_row or {}).get("box_score"),
        "prompt": (provenance_row or {}).get("prompt"),
        "detector_label": (provenance_row or {}).get("detector_label"),
        "review": {
            "verdict": review.get("verdict") or "tp",
            "raw_verdict": review.get("raw_verdict") or "tp",
            "notes": review.get("notes") or "",
            "sheet": review.get("sheet"),
            "row": review.get("row"),
        },
        "materialized_from": "dino-yoloe-audit",
    }
    label: dict[str, Any] = {
        "class": str(box["class_name"]),
        "bbox": bbox,
        "source": "human",
        "review_state": "accepted",
        "provenance": provenance,
    }
    confidence = (provenance_row or {}).get("box_score")
    if isinstance(confidence, (int, float)):
        label["confidence"] = round(float(confidence), 6)
    model_id = provenance["detector"]
    if model_id:
        label["model_id"] = str(model_id)
    return label


def build_items(
    *,
    dataset_root: str | Path,
    materialization: Mapping[str, Any] | None = None,
    include_pending: bool = True,
) -> ItemPlan:
    root = Path(dataset_root).expanduser().resolve()
    dataset = _load_json(root / "dataset.json")
    images = dataset.get("images") or []
    if not isinstance(images, list):
        raise PublishError("dataset.json images must be a list", stage="items")
    provenance_index = _provenance_index(root)
    inputs = (materialization or {}).get("inputs") or {}
    proposals_path = Path(inputs["proposals"]["path"]) if inputs.get("proposals") else None
    urls = _source_urls(proposals_path)

    plan = ItemPlan()
    plan.rejected_proposals = int(
        ((materialization or {}).get("review") or {}).get("rejected") or 0
    )
    entries_by_photo: dict[str, dict[str, Any]] = {}
    for entry in images:
        relative = str(entry.get("image") or "")
        split = _split_from_image_path(relative)
        photo_id = _photo_id(entry, relative)
        entries_by_photo[photo_id] = entry
        observation_id = entry.get("observation_id")
        if not observation_id:
            raise PublishError(
                f"dataset image {relative!r} has no observation_id; the API "
                "cannot guarantee observation-safe splits",
                stage="items",
            )
        detectors = {
            str((row.get("model") or {}).get("detector"))
            for row in provenance_index.values()
            if row.get("image") == relative and (row.get("model") or {}).get("detector")
        }
        detector = (entry.get("model") or (sorted(detectors)[0] if detectors else None))
        src_meta = entry.get("source") if isinstance(entry.get("source"), Mapping) else {}
        raw_purl = (
            entry.get("photo_url")
            or (src_meta.get("photo_url") if src_meta else None)
            or urls.get(photo_id)
        )
        resolved_purl = str(raw_purl).strip() if raw_purl else ""
        if resolved_purl.startswith("http://10.0.0.245:55434/photos/"):
            resolved_purl = "https://inaturalist-open-data.s3.amazonaws.com/photos/" + resolved_purl[len("http://10.0.0.245:55434/photos/"):]
        provenance = {
            "adapter": "dino-yoloe-audit",
            "source_image": entry.get("source_image"),
            "source_sha256": entry.get("source_sha256") or (src_meta.get("source_sha256") if src_meta else None),
            "review": entry.get("review"),
            "detector": detector,
            "materialization": (materialization or {}).get("schema"),
            "photo_url": resolved_purl or None,
        }
        item: dict[str, Any] = {
            "observation_id": str(observation_id),
            "item_key": f"img:{photo_id}",
            "split": split,
            "review_state": "accepted",
            "source": "human",
            "provenance": provenance,
        }
        if resolved_purl:
            item["photo_url"] = resolved_purl
        if detector:
            item["model_id"] = str(detector)
        boxes = _entry_boxes(entry, root, relative, split)
        if entry.get("negative") is True and boxes:
            raise PublishError(
                f"{relative} is marked negative but its label file has "
                f"{len(boxes)} box(es)",
                stage="items",
            )
        if not boxes and entry.get("negative") is not True:
            raise PublishError(
                f"{relative} has an empty label file but is not marked as a "
                "reviewed negative; a pending empty image cannot be training truth",
                stage="items",
            )
        if boxes:
            if len(boxes) > MAX_LABELS_PER_ITEM:
                raise PublishError(
                    f"image {relative} has {len(boxes)} boxes, above the API "
                    f"per-item limit of {MAX_LABELS_PER_ITEM}",
                    stage="items",
                )
            labels = []
            for box in boxes:
                bbox = [round(value, 6) for value in box.as_tuple()]
                key = (relative, int(box.class_id), tuple(bbox))
                labels.append(
                    _label_from_entry(
                        entry,
                        {
                            "class_id": box.class_id,
                            "class_name": box.class_name,
                            "x0": box.x0,
                            "y0": box.y0,
                            "x1": box.x1,
                            "y1": box.y1,
                        },
                        provenance_index.get(key),
                        detector,
                    )
                )
            item["labels"] = labels
            plan.positives += 1
        else:
            item["labels"] = []
            item["negative"] = True
            item["provenance"] = {
                **provenance,
                "negative_basis": (entry.get("review") or {}).get("negative_basis"),
                "reviewed_proposals": (entry.get("review") or {}).get("reviewed_proposals"),
            }
            plan.negatives += 1
        plan.items.append(item)

    if include_pending and materialization:
        plan.pending_source = "materialization.inputs"
        pending = _pending_items(
            materialization,
            existing_photo_ids=set(entries_by_photo),
            urls=urls,
        )
        plan.pending = len(pending)
        plan.items.extend(pending)
    return plan


def _pending_items(
    materialization: Mapping[str, Any],
    *,
    existing_photo_ids: set[str],
    urls: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Staged items for images with unreviewed proposals.

    Rejected proposals are excluded from truth and cannot be stored as empty
    "rejected" items by the API contract (empty items must be either reviewed
    negatives or ``review_state=proposed``), so only undecided images become
    pending queue entries.
    """

    inputs = materialization.get("inputs") or {}
    proposals_ref = (inputs.get("proposals") or {}).get("path")
    review_ref = (inputs.get("review") or {}).get("path")
    if not proposals_ref or not review_ref:
        raise PublishError(
            "materialization.json does not record proposals/review paths; "
            "run with --no-pending or re-materialize",
            stage="items",
        )
    proposals_path = Path(proposals_ref)
    review_path = Path(review_ref)
    if not proposals_path.is_file() or not review_path.is_file():
        raise PublishError(
            "materialization input files are not readable on this host "
            f"({proposals_path}, {review_path}); use --no-pending",
            stage="items",
        )
    from .materialize import load_reviews

    reviews = load_reviews(review_path)
    doc = json.loads(proposals_path.read_text())
    proposals = doc.get("proposals") or []
    images = {
        str(record.get("image_id") or record.get("photo_id") or ""): record
        for record in doc.get("images") or []
    }
    seed = ((materialization.get("selection") or {}).get("seed")) or CONTRACT_ID
    by_image: dict[str, list[dict[str, Any]]] = {}
    for proposal in proposals:
        photo_id = str(proposal.get("photo_id") or proposal.get("image_id") or "")
        if photo_id:
            by_image.setdefault(photo_id, []).append(proposal)

    items: list[dict[str, Any]] = []
    for photo_id in sorted(by_image):
        if photo_id in existing_photo_ids:
            continue
        undecided = [
            proposal
            for proposal in by_image[photo_id]
            if (reviews.entries.get(str(proposal.get("proposal_id"))) or None) is None
            or reviews.entries[str(proposal.get("proposal_id"))].verdict == "undecided"
        ]
        if not undecided:
            continue
        record = images.get(photo_id) or {}
        observation_id = str(
            record.get("observation_id")
            or undecided[0].get("observation_id")
            or ""
        )
        if not observation_id:
            continue
        detector = None
        for proposal in undecided:
            model = proposal.get("model") or {}
            if isinstance(model, Mapping) and model.get("detector"):
                detector = str(model["detector"])
                break
        item: dict[str, Any] = {
            "observation_id": observation_id,
            "item_key": f"img:{photo_id}",
            "split": split_for_observation(observation_id, seed=seed),
            "review_state": "proposed",
            "source": "model" if detector else "import",
            "labels": [],
            "provenance": {
                "adapter": "dino-yoloe-audit",
                "status": "pending_review",
                "photo_id": photo_id,
                "proposal_ids": [str(p.get("proposal_id")) for p in undecided],
                "proposal_count": len(undecided),
                "detector": detector,
            },
        }
        if detector:
            item["model_id"] = detector
        if photo_id in urls:
            item["photo_url"] = urls[photo_id]
        items.append(item)
    return items


# ── Metrics ─────────────────────────────────────────────────────────────────


def build_metrics(
    manifest: Mapping[str, Any],
    *,
    model_id: str,
    dataset_id: str,
    allow_zero_support: bool = False,
) -> dict[str, Any]:
    onnx_evaluation = manifest.get("evaluation_onnx") or {}
    evaluation = onnx_evaluation or (manifest.get("evaluation") or {})
    metrics = evaluation.get("metrics") or {}
    metrics_source = "onnx" if onnx_evaluation else "pytorch"
    per_class = metrics.get("per_class") or {}
    counts = metrics.get("counts") or {}
    operating = metrics.get("operating_point") or {}
    if not per_class:
        raise PublishError(
            "manifest has no held-out per-class metrics to record", stage="metrics"
        )

    undefined: dict[str, list[str]] = {}
    clean: dict[str, dict[str, float]] = {}
    for cls in CLASS_NAMES:
        entry = per_class.get(cls)
        if not isinstance(entry, Mapping):
            raise PublishError(
                f"metrics are missing class {cls!r}", stage="metrics"
            )
        values: dict[str, float] = {}
        missing: list[str] = []
        for key in ("precision", "recall", "ap50", "ap50_95"):
            value = entry.get(key)
            if value is None:
                missing.append(key)
                values[key] = 0.0
            else:
                values[key] = round(float(value), 6)
        if missing:
            undefined[cls] = missing
        clean[cls] = values
    if undefined and not allow_zero_support:
        detail = ", ".join(
            f"{cls} ({'/'.join(keys)})" for cls, keys in sorted(undefined.items())
        )
        raise PublishError(
            "held-out metrics are undefined (zero test support or no predictions): "
            f"{detail}. The API requires a number for every class. Re-run "
            "evaluation with test support for those classes, or pass "
            "--allow-zero-support to record 0.0 with an explicit provenance note.",
            stage="metrics",
        )

    # AnatomyLabPanel derives per-class support by counting labels on test items
    # and only trusts it when the total equals metrics.test_count. Use tagged GT
    # box support (not image count).
    test_count = sum(int(per_class[cls].get("support") or 0) for cls in CLASS_NAMES)
    declared_boxes = counts.get("gt_boxes")
    if isinstance(declared_boxes, int) and declared_boxes != test_count:
        raise PublishError(
            f"metrics support totals {test_count} but counts.gt_boxes is "
            f"{declared_boxes}; the test metrics are internally inconsistent",
            stage="metrics",
        )

    payload: dict[str, Any] = {
        "test_count": int(test_count),
        "per_class": clean,
        "eval_dataset_id": dataset_id,
        "model_id": model_id,
        "notes": (
            f"held-out {evaluation.get('split') or 'test'} split; "
            "AP50 is IoU 0.5, AP50-95 is the COCO 101-point mean"
        ),
        "provenance": {
            "publisher": "experiments.anatomy4_detector.publish",
            "evaluation_split": evaluation.get("split"),
            "metrics_source": metrics_source,
            "metrics_backend": evaluation.get("backend"),
            "test_images": counts.get("images"),
            "test_boxes": counts.get("gt_boxes"),
            "negative_images": counts.get("negative_images"),
            "operating_point": operating,
            "macro": metrics.get("macro"),
            "micro": metrics.get("micro"),
            "tagged_bbox": metrics.get("tagged_bbox"),
            "leaf_tagged_bbox": metrics.get("leaf_tagged_bbox"),
            "threshold_sweep": metrics.get("threshold_sweep"),
            "selected_threshold": evaluation.get("selected_threshold"),
            "tuning": evaluation.get("tuning"),
            "test_count_convention": (
                "sum of per-class test support (tagged GT boxes); matches the "
                "AnatomyLabPanel derived-support reconciliation over test items"
            ),
            "undefined_zero_support": undefined,
            "undefined_policy": (
                "substituted 0.0 with this note" if undefined else "none"
            ),
        },
    }

    confusion = metrics.get("confusion_matrix") or {}
    labels = confusion.get("labels") or []
    matrix = confusion.get("matrix") or []
    if matrix and all(cls in labels for cls in CLASS_NAMES):
        indexes = [labels.index(cls) for cls in CLASS_NAMES]
        payload["confusion"] = [
            [int(matrix[row][column]) for column in indexes] for row in indexes
        ]
        payload["provenance"]["confusion_note"] = (
            "4x4 class-vs-class submatrix; background row/column excluded. "
            "The full 5x5 matrix is in the uploaded metrics.json artifact."
        )
        payload["provenance"]["confusion_labels"] = list(CLASS_NAMES)
    return payload


# ── Artifacts ───────────────────────────────────────────────────────────────


@dataclass
class Artifact:
    filename: str
    kind: str
    content_type: str
    data: bytes
    source_path: str | None = None

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.data)

    @property
    def bytes(self) -> int:
        return len(self.data)

    def summary(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "kind": self.kind,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "source_path": self.source_path,
        }


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", Path(name).name).strip("-")
    if not cleaned or cleaned in {".", ".."}:
        raise PublishError(f"unsafe artifact filename {name!r}", stage="artifacts")
    return cleaned


def _check_limit(kind: str, filename: str, size: int) -> None:
    limit = ARTIFACT_LIMITS[kind]
    if size <= 0:
        raise PublishError(f"{kind} artifact {filename} is empty", stage="artifacts")
    if size > limit:
        raise PublishError(
            f"{kind} artifact {filename} is {size} bytes, above the API limit of "
            f"{limit} bytes",
            stage="artifacts",
        )


def _read_artifact(
    path: str | Path, *, kind: str, filename: str, content_type: str
) -> Artifact:
    source = Path(path)
    if not source.is_file():
        raise PublishError(f"artifact not found: {source}", stage="artifacts")
    data = source.read_bytes()
    _check_limit(kind, filename, len(data))
    return Artifact(
        filename=_safe_filename(filename),
        kind=kind,
        content_type=content_type,
        data=data,
        source_path=str(source.resolve()),
    )


def build_preview_detections(
    manifest: Mapping[str, Any],
    *,
    model_id: str | None = None,
    max_predictions: int = 200,
) -> Artifact:
    onnx_evaluation = manifest.get("evaluation_onnx") or {}
    evaluation = onnx_evaluation or (manifest.get("evaluation") or {})
    metrics = evaluation.get("metrics") or {}
    metrics_source = "onnx" if onnx_evaluation else "pytorch"
    payload: dict[str, Any] = {
        "schema": "arq.anatomy4.preview-detections/v1",
        "contract": CONTRACT_ID,
        "model_id": model_id,
        "split": evaluation.get("split"),
        "operating_point": metrics.get("operating_point"),
        "class_order": list(CLASS_NAMES),
        "overlays": [],
        "contact_sheets": [],
        "predictions_sample": [],
        "note": (
            "TP/FP/FN counts in overlays are computed by the trainer at the "
            "recorded operating point; predictions_sample is a bounded sample "
            "of predictions.jsonl for audit."
        ),
    }
    metrics_path = evaluation.get("metrics_json")
    if metrics_path and Path(metrics_path).is_file():
        try:
            metrics_doc = json.loads(Path(metrics_path).read_text())
        except (OSError, json.JSONDecodeError):
            metrics_doc = {}
        payload["overlays"] = (metrics_doc.get("overlays") or [])[:64]
        payload["contact_sheets"] = (metrics_doc.get("contact_sheets") or [])[:32]
    predictions_path = metrics.get("predictions_jsonl")
    if predictions_path and Path(predictions_path).is_file():
        rows = []
        for line in Path(predictions_path).read_text().splitlines():
            if len(rows) >= max_predictions:
                break
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        payload["predictions_sample"] = rows
    data = json.dumps(payload).encode("utf-8")
    if len(data) > ARTIFACT_LIMITS["manifest"]:
        # Predictions dominate; keep the overlay audit and drop the sample.
        payload["predictions_sample"] = []
        payload["predictions_sample_truncated"] = True
        data = json.dumps(payload).encode("utf-8")
    _check_limit("manifest", "preview_detections.json", len(data))
    return Artifact(
        filename="preview_detections.json",
        kind="manifest",
        content_type="application/json",
        data=data,
        source_path=predictions_path,
    )


def collect_artifacts(
    manifest: Mapping[str, Any],
    *,
    model_id: str | None = None,
    metrics_artifact: Mapping[str, Any] | None = None,
    previews: Sequence[str] | None = None,
    max_previews: int = DEFAULT_MAX_PREVIEWS,
    allow_no_previews: bool = False,
) -> list[Artifact]:
    export = manifest.get("export") or {}
    training = manifest.get("training") or {}
    evaluation = manifest.get("evaluation") or {}
    artifacts: list[Artifact] = []

    onnx_path = export.get("onnx")
    if not onnx_path:
        raise PublishError("manifest has no export.onnx path", stage="artifacts")
    onnx = _read_artifact(
        onnx_path,
        kind="onnx",
        filename="anatomy4.onnx",
        content_type="application/octet-stream",
    )
    expected = export.get("sha256")
    if expected and onnx.sha256 != expected:
        raise PublishError(
            "ONNX sha256 does not match the manifest; refusing to upload a "
            f"different artifact ({onnx.sha256} != {expected})",
            stage="artifacts",
        )
    artifacts.append(onnx)

    weights_path = training.get("best_weights")
    if not weights_path:
        raise PublishError("manifest has no training.best_weights", stage="artifacts")
    weights = _read_artifact(
        weights_path,
        kind="weights",
        filename="best.pt",
        content_type="application/octet-stream",
    )
    artifacts.append(weights)

    metrics_path = evaluation.get("metrics_json")
    if metrics_path and Path(metrics_path).is_file():
        artifacts.append(
            _read_artifact(
                metrics_path,
                kind="metrics",
                filename="metrics.json",
                content_type="application/json",
            )
        )
    elif metrics_artifact is not None:
        data = json.dumps(metrics_artifact, indent=1).encode("utf-8")
        _check_limit("metrics", "metrics.json", len(data))
        artifacts.append(
            Artifact(
                filename="metrics.json",
                kind="metrics",
                content_type="application/json",
                data=data,
            )
        )
    else:
        raise PublishError(
            "no evaluation metrics JSON available to upload", stage="artifacts"
        )

    preview_paths = list(previews if previews is not None else manifest.get("previews") or [])
    seen: set[str] = set()
    uploaded = 0
    for index, path in enumerate(preview_paths):
        source = Path(path)
        if not source.is_file():
            continue
        if source.stat().st_size > ARTIFACT_LIMITS["preview"]:
            log.warning(
                "skipping preview %s: %s bytes exceeds the %s-byte limit",
                source.name,
                source.stat().st_size,
                ARTIFACT_LIMITS["preview"],
            )
            continue
        filename = _safe_filename(f"preview-{index + 1:02d}-{source.name}")
        if filename in seen:
            continue
        seen.add(filename)
        artifacts.append(
            _read_artifact(
                source,
                kind="preview",
                filename=filename,
                content_type=(
                    "image/png" if source.suffix.lower() == ".png" else "image/jpeg"
                ),
            )
        )
        uploaded += 1
        if uploaded >= max(0, int(max_previews)):
            break
    if uploaded == 0 and not allow_no_previews:
        raise PublishError(
            "no preview images available to upload (pass --no-previews to skip)",
            stage="artifacts",
        )
    artifacts.append(build_preview_detections(manifest, model_id=model_id))
    return artifacts


# ── Smoke inference ─────────────────────────────────────────────────────────


def choose_test_image(
    dataset_root: str | Path, override: str | Path | None = None
) -> Path:
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise PublishError(f"--test-image not found: {path}", stage="infer")
        return path
    root = Path(dataset_root).expanduser().resolve()
    dataset = _load_json(root / "dataset.json")
    candidates = [
        root / entry["image"]
        for entry in dataset.get("images") or []
        if str(entry.get("image") or "").startswith("images/test/")
    ]
    if not candidates:
        raise PublishError(
            "dataset has no test-split image for the /infer smoke test",
            stage="infer",
        )
    return sorted(candidates)[0]


def validate_infer_response(
    response: Mapping[str, Any],
    *,
    model_id: str,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    if response.get("source") != "onnx":
        raise PublishError(
            f"/infer did not report source='onnx' (got {response.get('source')!r}); "
            "refusing to treat a non-ONNX result as a smoke pass",
            stage="infer",
        )
    if response.get("model_id") != model_id:
        raise PublishError(
            f"/infer used model {response.get('model_id')!r}, expected {model_id!r}",
            stage="infer",
        )
    if list(response.get("class_names") or []) != list(class_names):
        raise PublishError(
            f"/infer class order {response.get('class_names')} != {list(class_names)}",
            stage="infer",
        )
    detections = response.get("detections")
    if not isinstance(detections, list):
        raise PublishError("/infer returned no detections array", stage="infer")
    if response.get("detection_count") != len(detections):
        raise PublishError(
            "/infer detection_count disagrees with detections length", stage="infer"
        )
    if not (isinstance(response.get("width"), int) and response["width"] > 0):
        raise PublishError("/infer returned an invalid width", stage="infer")
    if not (isinstance(response.get("height"), int) and response["height"] > 0):
        raise PublishError("/infer returned an invalid height", stage="infer")
    for detection in detections:
        if detection.get("class") not in class_names:
            raise PublishError(
                f"/infer detection has unknown class {detection.get('class')!r}",
                stage="infer",
            )
        bbox = detection.get("bbox")
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(0.0 <= float(value) <= 1.0 for value in bbox)
        ):
            raise PublishError(
                f"/infer detection bbox is not normalized xyxy: {bbox!r}",
                stage="infer",
            )
    return {
        "model_id": model_id,
        "width": response["width"],
        "height": response["height"],
        "infer_ms": response.get("infer_ms"),
        "detection_count": len(detections),
        "class_counts": {
            cls: sum(1 for det in detections if det.get("class") == cls)
            for cls in class_names
        },
        "threshold": response.get("threshold"),
        "iou_threshold": response.get("iou_threshold"),
        "source": response.get("source"),
    }


# ── Orchestration ───────────────────────────────────────────────────────────


@dataclass
class PublishOptions:
    manifest_path: Path
    dataset_root: Path | None = None
    test_dataset_root: Path | None = None
    materialization_path: Path | None = None
    dataset_id: str | None = None
    model_id: str | None = None
    name: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    skip_items: bool = False
    include_pending: bool = True
    max_previews: int = DEFAULT_MAX_PREVIEWS
    allow_no_previews: bool = False
    allow_synthetic: bool = False
    allow_zero_support: bool = False
    test_image: Path | None = None
    threshold: float = 0.25
    iou_threshold: float = 0.45
    dry_run: bool = False
    report_out: Path | None = None


def plan_publish(options: PublishOptions) -> dict[str, Any]:
    manifest = _load_json(options.manifest_path)
    if manifest.get("contract") != CONTRACT_ID:
        raise PublishError(
            f"manifest contract {manifest.get('contract')!r} != {CONTRACT_ID!r}",
            stage="manifest",
        )
    if manifest.get("synthetic") is True and not options.allow_synthetic:
        raise PublishError(
            "refusing to publish a synthetic smoke run; its metrics are not a "
            "detector claim (use --allow-synthetic only for API plumbing tests)",
            stage="manifest",
        )
    dataset_root = (
        Path(options.dataset_root).expanduser().resolve()
        if options.dataset_root is not None
        else Path(str((manifest.get("dataset") or {}).get("root") or "")).expanduser().resolve()
    )
    if not (dataset_root / "dataset.json").is_file():
        raise PublishError(
            f"dataset root does not contain dataset.json: {dataset_root}",
            stage="dataset",
        )
    materialization_path = options.materialization_path
    if materialization_path is None:
        candidate = dataset_root / "materialization.json"
        materialization_path = candidate if candidate.is_file() else None
    materialization = (
        _load_json(materialization_path)
        if materialization_path is not None and Path(materialization_path).is_file()
        else None
    )
    name = options.name or (
        f"anatomy4 yolo26n {manifest.get('generated_at') or ''}".strip()
    )
    dataset_id = options.dataset_id or stable_dataset_id(manifest, name)
    model_id = options.model_id or stable_model_id(manifest, name)

    items = None
    test_items = None
    test_summary = None
    isolation = None
    if not options.skip_items:
        items = build_items(
            dataset_root=dataset_root,
            materialization=materialization,
            include_pending=options.include_pending,
        )
        if options.test_dataset_root is not None:
            non_train_val = sorted(
                {
                    str(item.get("split"))
                    for item in items.items
                    if item.get("split") not in {"train", "val"}
                }
            )
            if non_train_val:
                raise PublishError(
                    "--dataset must be train+val only when --test-dataset is "
                    f"given (found splits {non_train_val}); the frozen test "
                    "belongs in --test-dataset",
                    stage="items",
                )
            test_root = Path(options.test_dataset_root).expanduser().resolve()
            test_items = build_items(
                dataset_root=test_root, materialization=None, include_pending=False
            )
            bad_splits = sorted(
                {
                    str(item.get("split"))
                    for item in test_items.items
                    if item.get("split") != "test"
                }
            )
            if bad_splits:
                raise PublishError(
                    f"--test-dataset items must all be split=test (found {bad_splits})",
                    stage="items",
                )
            try:
                isolation = assert_disjoint_datasets(dataset_root, test_root)
            except ContractError as exc:
                raise PublishError(str(exc), stage="items") from exc
            items.items.extend(test_items.items)
            items.positives += test_items.positives
            items.negatives += test_items.negatives
            test_summary = {
                "root": str(test_root),
                "test_only": True,
                "counts": test_items.summary(),
                "isolation": isolation,
            }
    metrics = build_metrics(
        manifest,
        model_id=model_id,
        dataset_id=dataset_id,
        allow_zero_support=options.allow_zero_support,
    )
    if test_items is not None:
        test_boxes = sum(len(item.get("labels") or []) for item in test_items.items)
        if test_boxes != metrics["test_count"]:
            raise PublishError(
                f"test dataset supplies {test_boxes} tagged boxes but the held-out "
                f"metrics report test_count={metrics['test_count']}; the manifest "
                "was not evaluated on this test dataset",
                stage="metrics",
            )
        metrics["provenance"]["test_dataset_boxes"] = test_boxes
        metrics["provenance"]["test_dataset_items"] = len(test_items.items)
    dataset_payload = build_dataset_payload(
        manifest,
        dataset_id=dataset_id,
        name=name,
        manifest_path=options.manifest_path,
        materialization=materialization,
        test_summary=test_summary,
    )
    model_payload = build_model_payload(
        manifest, model_id=model_id, name=name, dataset_id=dataset_id
    )
    artifacts = collect_artifacts(
        manifest,
        model_id=model_id,
        metrics_artifact=metrics,
        max_previews=options.max_previews,
        allow_no_previews=options.allow_no_previews,
    )
    test_image = choose_test_image(
        options.test_dataset_root or dataset_root, options.test_image
    )
    return {
        "manifest": manifest,
        "materialization": materialization,
        "test_items": test_items,
        "test_summary": test_summary,
        "name": name,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "dataset_payload": dataset_payload,
        "items": items,
        "test_items": test_items,
        "test_summary": test_summary,
        "metrics": metrics,
        "model_payload": model_payload,
        "artifacts": artifacts,
        "test_image": test_image,
        "dataset_root": str(dataset_root),
    }


def describe_plan(
    plan: Mapping[str, Any],
    *,
    auth: AuthMaterial,
    base: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = 0.25,
) -> dict[str, Any]:
    items = plan.get("items")
    return {
        "dry_run": True,
        "base": base,
        "auth": auth_summary(auth),
        "dataset": {
            "dataset_id": plan["dataset_id"],
            "name": plan["name"],
            "payload_bytes": len(
                json.dumps(plan["dataset_payload"]).encode("utf-8")
            ),
            "manifest_sections": sorted(plan["dataset_payload"]["manifest"].keys()),
        },
        "items": items.summary() if items is not None else {"skipped": True},
        "test_items": (
            plan["test_items"].summary() if plan.get("test_items") is not None else None
        ),
        "test_dataset": plan.get("test_summary"),
        "item_batches": (
            [len(batch) for batch in items.batches(batch_size)]
            if items is not None
            else []
        ),
        "model": {
            "model_id": plan["model_id"],
            "classes": plan["model_payload"]["class_names"],
            "dataset_id": plan["dataset_payload"]["dataset_id"],
        },
        "artifacts": [artifact.summary() for artifact in plan["artifacts"]],
        "metrics": {
            "test_count": plan["metrics"]["test_count"],
            "per_class": plan["metrics"]["per_class"],
            "confusion": "4x4" if "confusion" in plan["metrics"] else None,
            "zero_support_substituted": plan["metrics"]["provenance"].get(
                "undefined_zero_support"
            ),
        },
        "activate": True,
        "infer": {
            "test_image": str(plan["test_image"]),
            "bytes": plan["test_image"].stat().st_size,
            "threshold": threshold,
        },
    }


def publish(
    options: PublishOptions,
    *,
    base: str,
    auth: AuthMaterial,
    transport: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    plan = plan_publish(options)
    resolved_base = resolve_base(base)
    description = describe_plan(
        plan,
        auth=auth,
        base=resolved_base,
        batch_size=options.batch_size,
        threshold=options.threshold,
    )
    if options.dry_run:
        return description
    auth.require()

    report: dict[str, Any] = {
        "schema": "arq.anatomy4.publish-report/v1",
        "contract": CONTRACT_ID,
        "base": resolved_base,
        "auth": auth_summary(auth),
        "dry_run": False,
        "dataset_id": plan["dataset_id"],
        "model_id": plan["model_id"],
        "items": plan["items"].summary() if plan["items"] is not None else {"skipped": True},
        "test_items": (
            plan["test_items"].summary() if plan.get("test_items") is not None else None
        ),
        "test_dataset": plan.get("test_summary"),
        "artifacts": [artifact.summary() for artifact in plan["artifacts"]],
    }
    client = ArqClient(
        resolved_base,
        auth,
        timeout=timeout,
        retries=retries,
        transport=transport,
        sleep=sleep,
    )
    try:
        existing_dataset = client.get_optional(
            f"/fieldvision/anatomy/datasets/{plan['dataset_id']}",
            stage="dataset.lookup",
        )
        if existing_dataset is not None:
            existing_classes = list(existing_dataset.get("class_names") or [])
            if existing_classes and existing_classes != list(CLASS_NAMES):
                raise PublishError(
                    f"dataset {plan['dataset_id']} already exists with classes "
                    f"{existing_classes}, not {list(CLASS_NAMES)}",
                    stage="dataset.lookup",
                )
            report["dataset_action"] = "reused"
        else:
            created = client.post_json(
                "/fieldvision/anatomy/datasets",
                plan["dataset_payload"],
                stage="dataset.create",
            )
            report["dataset_action"] = "created"
            report["dataset_response_id"] = created.get("dataset_id")

        if plan["items"] is not None:
            batches = plan["items"].batches(options.batch_size)
            batch_reports = []
            for index, batch in enumerate(batches):
                response = client.post_json(
                    f"/fieldvision/anatomy/datasets/{plan['dataset_id']}/items",
                    {"items": batch},
                    stage=f"items.batch[{index}]",
                )
                batch_reports.append(
                    {
                        "batch": index,
                        "items": len(batch),
                        "inserted": response.get("inserted"),
                        "updated": response.get("updated"),
                    }
                )
            report["item_batches"] = batch_reports
        else:
            report["item_batches"] = []

        existing_model = client.get_optional(
            f"/fieldvision/anatomy/models/{plan['model_id']}", stage="model.lookup"
        )
        if existing_model is not None:
            existing_classes = list(existing_model.get("class_names") or [])
            if existing_classes and existing_classes != list(CLASS_NAMES):
                raise PublishError(
                    f"model {plan['model_id']} already exists with classes "
                    f"{existing_classes}, not {list(CLASS_NAMES)}",
                    stage="model.lookup",
                )
            report["model_action"] = "reused"
        else:
            created = client.post_json(
                "/fieldvision/anatomy/models",
                plan["model_payload"],
                stage="model.create",
            )
            report["model_action"] = "created"
            report["model_response_id"] = created.get("model_id")

        uploads = []
        chunk_max = 20 * 1024 * 1024
        for artifact in plan["artifacts"]:
            response: dict[str, Any] = {}
            total_len = len(artifact.data)
            for offset in range(0, total_len, chunk_max):
                slice_bytes = artifact.data[offset : offset + chunk_max]
                params: dict[str, str] = {
                    "kind": artifact.kind,
                    "filename": artifact.filename,
                }
                if offset > 0:
                    params["append"] = "true"
                response = client.post_multipart(
                    f"/fieldvision/anatomy/models/{plan['model_id']}/artifacts",
                    filename=artifact.filename,
                    data=slice_bytes,
                    content_type=artifact.content_type,
                    params=params,
                    stage=f"artifact.{artifact.kind}",
                )
            uploads.append(
                {
                    "filename": artifact.filename,
                    "kind": artifact.kind,
                    "bytes": response.get("bytes") or artifact.bytes,
                    "sha256": response.get("sha256") or artifact.sha256,
                    "storage": response.get("storage"),
                }
            )
        report["uploads"] = uploads

        metrics_response = client.post_json(
            f"/fieldvision/anatomy/models/{plan['model_id']}/metrics",
            {"metrics": plan["metrics"]},
            stage="model.metrics",
        )
        report["metrics_recorded"] = True
        report["metrics_response_model"] = metrics_response.get("model_id")

        activation = client.post_json(
            f"/fieldvision/anatomy/models/{plan['model_id']}/activate",
            {},
            stage="model.activate",
        )
        report["activated"] = bool(activation.get("is_active"))
        report["activation_preflight"] = activation.get("preflight")

        test_image = plan["test_image"]
        image_bytes = test_image.read_bytes()
        if len(image_bytes) > MAX_INFER_BYTES:
            raise PublishError(
                f"/infer smoke image is {len(image_bytes)} bytes, above the API "
                f"limit of {MAX_INFER_BYTES}",
                stage="infer",
            )
        infer_response = client.post_multipart(
            "/fieldvision/anatomy/infer",
            filename=_safe_filename(test_image.name),
            data=image_bytes,
            content_type=(
                "image/png" if test_image.suffix.lower() == ".png" else "image/jpeg"
            ),
            params=None,
            form={
                "model_id": plan["model_id"],
                "contract": CONTRACT_ID,
                "threshold": str(options.threshold),
                "iou_threshold": str(options.iou_threshold),
            },
            stage="infer",
        )
        report["infer"] = validate_infer_response(
            infer_response, model_id=plan["model_id"]
        )
        report["infer"]["test_image"] = str(test_image)
    finally:
        client.close()

    if options.report_out:
        target = Path(options.report_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=1) + "\n")
    return report