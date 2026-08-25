"""Local agentic-tagging runner server.

Holds the Gemma + Grounding-DINO/SAM models loaded once, and serves run requests
from the workbench UI: POST a statement + tunable knobs, get a job id, then poll
for live progress as each photo resolves. Stdlib ``http.server`` + a worker
thread — no new dependencies, and the models stay warm across iterations.

The workbench Vite proxy forwards ``/fgv/*`` here (default port 8790), so the UI
never touches the CLI.

Endpoints
---------
GET  /fgv/health                     -> {"status": "ok", "models_loaded": bool}
POST /fgv/runs                       -> start a run; body = RunRequest JSON
GET  /fgv/runs/{id}                  -> {"status", "progress", "records": [...]}
GET  /fgv/options                    -> available devos, strictness levels, scope labels

RunRequest:
    {
      "statement": str,
      "devo": "dirr",
      "taxon_scope": ["Tree", "Shrub"],
      "limit": 40,
      "strictness": "strict" | "balanced" | "permissive",
      "max_turns": 8,
      "min_organ_area": 0.05,           # optional organ-band override
      "min_organ_confidence": 0.30,     # optional organ-band override
      "research_only": true
    }
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

logger = logging.getLogger("fgv_runner")

DEFAULT_SOURCE = ROOT / "observations_pound_ridge" / "ward_pound_ridge_species.csv"
RUNS_ROOT = ROOT / "scratch" / "p17-runs"

# ---------------------------------------------------------------------------
# Lazy model holder — loaded on first run, kept warm. Rebuilt if the advanced
# detector/segmenter knobs change between runs (they are model-construction args).
# ---------------------------------------------------------------------------

# Advanced (model-construction) knobs. Changing any of these rebuilds the models.
_MODEL_KNOBS = ("detector_model", "segmenter_model", "polygon_epsilon", "multimask_output")

_MODEL_DEFAULTS = {
    "detector_model": "IDEA-Research/grounding-dino-tiny",
    "segmenter_model": "facebook/sam2.1-hiera-small",
    "box_threshold": 0.25,
    "text_threshold": 0.20,
    "iou_threshold": 0.30,
    "polygon_epsilon": 0.003,
    "multimask_output": False,
}


class _Models:
    def __init__(self) -> None:
        self.loaded = False
        self._vlm = None
        self._segmenter = None
        self._knob_signature: tuple | None = None
        self._lock = threading.Lock()

    def _make_segment(self, segmenter) -> "object":
        """Return a detect-style callable that surfaces the full candidate signal."""
        return lambda image, prompts: segmenter.detect_image(image, list(prompts))

    def ensure(self, knobs: dict[str, Any] | None = None) -> None:
        knobs = {**_MODEL_DEFAULTS, **(knobs or {})}
        signature = tuple(knobs.get(k) for k in _MODEL_KNOBS)
        with self._lock:
            if self.loaded and signature == self._knob_signature:
                self._apply_runtime_knobs(knobs)
                return
            logger.info("Loading Gemma 4 12B + Grounding-DINO/SAM (knobs=%s)...", signature)
            from segmentation.local_vlm_evaluator import LocalVlmEvaluator
            from segmentation.grounded_sam2 import GroundedSam2Segmenter

            vlm_eval = LocalVlmEvaluator(device="mps")
            segmenter = GroundedSam2Segmenter(
                device="mps",
                detector_model=knobs["detector_model"],
                segmenter_model=knobs["segmenter_model"],
                box_threshold=knobs["box_threshold"],
                text_threshold=knobs["text_threshold"],
                iou_threshold=knobs["iou_threshold"],
                polygon_epsilon=knobs["polygon_epsilon"],
                multimask_output=knobs["multimask_output"],
            )
            self._vlm = lambda image, prompt: vlm_eval.generate_text(image, prompt)
            self._segmenter = segmenter
            self._knob_signature = signature
            self.loaded = True
            logger.info("Models loaded and warm.")

    def _apply_runtime_knobs(self, knobs: dict[str, Any]) -> None:
        """Thresholds are read per-call, so they can be tuned without a rebuild."""
        if not self._segmenter:
            return
        self._segmenter._box_threshold = knobs["box_threshold"]
        self._segmenter._text_threshold = knobs["text_threshold"]
        self._segmenter._iou_threshold = knobs["iou_threshold"]

    @property
    def segment(self):
        return self._make_segment(self._segmenter) if self._segmenter else None


MODELS = _Models()

# In-memory run registry: run_id -> {"status", "progress", "records", "error", ...}
RUNS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Run execution (worker thread)
# ---------------------------------------------------------------------------


def _load_image(url: str):
    from io import BytesIO
    from urllib.request import Request, urlopen

    from PIL import Image

    if url.startswith("file://"):
        return Image.open(Path(url.removeprefix("file://"))).convert("RGB")
    req = Request(url, headers={"User-Agent": "ArborphyFGV/0.1"})
    with urlopen(req, timeout=30) as resp:
        return Image.open(BytesIO(resp.read())).convert("RGB")


def _thumb_data_uri(url: str, max_px: int = 240) -> str | None:
    try:
        import base64
        from io import BytesIO

        img = _load_image(url)
        img.thumbnail((max_px, max_px))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".bmp"}

# The area pool comes from the local arq-api (not a static CSV). The runner reads
# the live catalogue for the chosen area instance and runs the agentic loop over
# every observation with a photo.
ARQ_API_BASE = "http://localhost:8000"


def _arq_get(path: str) -> Any:
    import json as _json
    from urllib.request import Request, urlopen

    req = Request(ARQ_API_BASE + path, headers={"User-Agent": "ArborphyFGV/0.1"})
    with urlopen(req, timeout=120) as resp:
        return _json.load(resp)


def list_area_instances() -> list[dict[str, Any]]:
    """Area instances from arq-api for the picker (id, name, bbox, obs count).

    The count comes from the catalogue summary so the UI can show pool size
    without fetching full catalogues.
    """
    areas: list[dict[str, Any]] = []
    offset = 0
    while True:
        d = _arq_get(f"/api/instances?limit=500&offset={offset}")
        rows = d.get("data", [])
        if not rows:
            break
        for r in rows:
            if r.get("instance_type") == "area" and not r.get("deleted_at"):
                areas.append({
                    "instance_id": r["instance_id"],
                    "name": r.get("name") or r["instance_id"],
                    "bbox": [r.get("bbox_west"), r.get("bbox_south"), r.get("bbox_east"), r.get("bbox_north")],
                })
        offset += len(rows)
        if len(rows) < 500 or offset >= (d.get("total") or 0):
            break
    # Enrich with observation counts (best-effort; a slow summary should not
    # block the picker).
    for a in areas:
        try:
            s = _arq_get(f"/api/instances/{a['instance_id']}/catalogue/summary")
            a["observations"] = (s.get("summary") or {}).get("observations")
        except Exception:
            a["observations"] = None
    return areas


def _upgrade_inat_photo(url: str) -> str:
    """Catalogue photo_url is the small `square.jpg`; detection wants `medium`."""
    if "inaturalist-open-data" in url and url.endswith("square.jpg"):
        return url[: -len("square.jpg")] + "medium.jpg"
    return url


def _build_area_frames(req: dict[str, Any]) -> tuple:
    """Build CaptureFrames from every photo observation in a chosen area.

    Reads the live catalogue from arq-api. The candidate-taxon scope still gates
    which observations become frames (the statement's taxon breadth), unless the
    caller passes an empty scope — then ALL photo observations in the area run,
    which is the "run it on all observations in an area" case.
    """
    from segmentation.pipeline import CaptureFrame

    area_id = req.get("area_instance_id")
    if not area_id:
        raise ValueError("area_instance_id is required for the area pool source")

    catalogue = _arq_get(f"/api/instances/{area_id}/catalogue")
    observations = catalogue.get("observations", [])
    research_only = bool(req.get("research_only", True))
    limit = int(req.get("limit", 40))
    scope = {s.strip().lower() for s in (req.get("taxon_scope") or []) if str(s).strip()}

    frames = []
    for obs in observations:
        photo = obs.get("photo_url")
        if not photo:
            continue
        if research_only and obs.get("quality_grade") != "research":
            continue
        if scope:
            name = (obs.get("taxon_name") or "").lower()
            # Broad scope gate: keep the obs if its taxon name mentions any scope
            # term. (The catalogue has no Dirr value labels, so scope matches on
            # name text — the agent still verifies the features from the image.)
            if not any(term in name for term in scope):
                continue
        frames.append(
            CaptureFrame(
                observation_id=obs.get("arborphy_id") or f"obs-{len(frames)}",
                image_url=_upgrade_inat_photo(photo),
                subject_id=obs.get("taxon_name") or "unknown",
                species_name=obs.get("taxon_name"),
                photo_id=None,
                license=obs.get("license"),
                captured_at=obs.get("observed_on"),
                frame_index=0,
            )
        )
        if len(frames) >= limit:
            break
    return tuple(frames)


def _build_custom_frames(req: dict[str, Any]):
    """Build CaptureFrames from a user-supplied photo set.

    Accepts ``photo_dir`` (a folder scanned for image files) and/or ``photos``
    (a list of local file paths or http(s) URLs). Local paths become ``file://``
    URLs. For arbitrary photos the taxon is unknown — the agent infers the subject
    during resolution, which is the point of an open-arity statement.
    """
    from segmentation.pipeline import CaptureFrame

    urls: list[str] = []

    photo_dir = req.get("photo_dir")
    if photo_dir:
        d = Path(photo_dir).expanduser()
        if not d.is_dir():
            raise ValueError(f"photo_dir is not a directory: {photo_dir}")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file():
                urls.append("file://" + str(p.resolve()))

    for entry in req.get("photos") or []:
        entry = str(entry).strip()
        if not entry:
            continue
        if entry.startswith(("http://", "https://", "file://")):
            urls.append(entry)
        else:
            p = Path(entry).expanduser()
            if not p.is_file():
                raise ValueError(f"photo not found: {entry}")
            urls.append("file://" + str(p.resolve()))

    limit = int(req.get("limit", 40))
    urls = urls[:limit]

    frames = []
    for i, url in enumerate(urls):
        stem = url.rsplit("/", 1)[-1].rsplit(".", 1)[0] or f"photo-{i}"
        frames.append(
            CaptureFrame(
                observation_id=f"custom-{i:04d}-{stem}",
                image_url=url,
                subject_id="unknown",
                species_name=None,
                frame_index=0,
            )
        )
    return tuple(frames)


def _execute_run(run_id: str, req: dict[str, Any]) -> None:
    from segmentation.agentic_resolver import (
        AgenticSceneResolver,
        resolution_to_trace_record,
    )
    from segmentation.devo_adapter import DirrDevoAdapter
    from segmentation.pipeline import CaptureFrame
    from segmentation.statement_compiler import StatementCompiler
    from scripts.run_agentic_tagging import load_subset

    run = RUNS[run_id]
    try:
        # Model-construction + detector knobs (advanced). ensure() rebuilds only
        # when the construction signature changed; thresholds apply per-call.
        model_knobs = {k: req[k] for k in (
            "detector_model", "segmenter_model", "box_threshold", "text_threshold",
            "iou_threshold", "polygon_epsilon", "multimask_output",
        ) if k in req}
        MODELS.ensure(model_knobs)

        devo = DirrDevoAdapter()
        compiler = StatementCompiler(devo)
        # Open-arity compilation: let Gemma translate the free-form statement into the
        # DeVo vocabulary. Fall back to keyword compilation if the VLM path fails.
        try:
            plan = compiler.compile_with_vlm(req["statement"], MODELS._vlm)
            run["compile_mode"] = "vlm"
        except Exception as exc:
            logger.warning("VLM compile failed (%s); falling back to keyword compile", exc)
            plan = compiler.compile(req["statement"])
            run["compile_mode"] = "keyword"

        # Optional organ-band overrides (degrees of freedom for tuning).
        min_area = req.get("min_organ_area")
        min_conf = req.get("min_organ_confidence")
        if min_area is not None or min_conf is not None:
            from segmentation.statement_compiler import OrganRequirement

            plan = type(plan)(
                statement=plan.statement,
                devo_name=plan.devo_name,
                tag_targets=plan.tag_targets,
                organs=tuple(
                    OrganRequirement(
                        o.organ,
                        o.required,
                        float(min_area) if min_area is not None else o.min_area_fraction,
                        float(min_conf) if min_conf is not None else o.min_detection_confidence,
                    )
                    for o in plan.organs
                ),
                prefer_whole_specimen=plan.prefer_whole_specimen,
                alias_notes=plan.alias_notes,
            )

        # Photo source priority: an explicit photo set (folder/list) wins; then a
        # chosen area pool from arq-api; then the WPR iNat subset by taxon scope.
        if req.get("photo_dir") or req.get("photos"):
            frames = _build_custom_frames(req)
            run["source"] = "custom"
        elif req.get("area_instance_id"):
            frames = _build_area_frames(req)
            run["source"] = "area-pool"
            run["area_instance_id"] = req.get("area_instance_id")
        else:
            scope = req.get("taxon_scope") or ["Tree", "Shrub"]
            candidate_species: set[str] = set()
            for lbl in scope:
                candidate_species.update(
                    t.scientific_name for t in devo.resolve_taxa_by_feature_values([lbl])
                )
            frames = load_subset(
                DEFAULT_SOURCE,
                candidate_species=candidate_species,
                limit=int(req.get("limit", 40)),
                research_only=bool(req.get("research_only", True)),
            )
            run["source"] = "wpr-inat"
        run["total"] = len(frames)
        run["plan"] = {
            "tag_targets": [
                {"feature_name": t.feature_name, "value_label": t.value_label, "organ": t.required_organ}
                for t in plan.tag_targets
            ],
            "organs": [
                {"organ": o.organ, "required": o.required,
                 "min_area": o.min_area_fraction, "min_conf": o.min_detection_confidence}
                for o in plan.organs
            ],
            "alias_notes": [{"phrase": p, "resolved": v} for p, v in plan.alias_notes],
            "compile_mode": run.get("compile_mode"),
        }
        run["status"] = "running"

        resolver = AgenticSceneResolver(
            vlm=MODELS._vlm, segment=MODELS.segment,
            max_turns=int(req.get("max_turns", 8)),
            strictness=req.get("strictness", "strict"),
            min_organ_area=min_area,
            min_organ_confidence=min_conf,
        )

        for i, frame in enumerate(frames):
            # Kill switch: the cancel endpoint sets this flag; stop between photos.
            if run.get("cancel_requested"):
                run["status"] = "cancelled"
                run["progress"] = f"cancelled after {i}/{len(frames)} photos"
                logger.info("run %s cancelled by user after %d/%d photos", run_id, i, len(frames))
                return
            label = frame.species_name or frame.image_url.rsplit("/", 1)[-1]
            run["progress"] = f"{i + 1}/{len(frames)} {label}"
            try:
                image = _load_image(frame.image_url)
                res = resolver.resolve(image, plan, frame)
            except Exception as exc:
                logger.exception("run %s photo %s failed", run_id, frame.observation_id)
                run["records"].append({
                    "observation_id": frame.observation_id, "species_name": frame.species_name,
                    "image_url": frame.image_url, "outcome": "error", "error": str(exc),
                    "turns_used": 0, "feature_tags": [], "grounded_organs": [], "turns": [],
                })
                continue
            record = resolution_to_trace_record(res, frame, plan)
            record["thumb"] = _thumb_data_uri(frame.image_url)
            # Attach the grounded polygon (normalized) for the first accepted organ.
            if res.grounded_segments:
                seg = res.grounded_segments[0]
                record["polygon"] = [list(p) for p in seg.polygon]
            # Flatten every grounding-turn candidate for the overlay + verdict panel.
            record["candidates"] = [
                c for t in record.get("turns", []) for c in (t.get("candidates") or [])
            ]
            run["records"].append(record)
        run["status"] = "done"
        run["progress"] = f"done ({len(frames)} photos)"
    except Exception as exc:
        logger.exception("run %s failed", run_id)
        run["status"] = "error"
        run["error"] = str(exc)
    finally:
        _persist_run(run_id)


def _persist_run(run_id: str) -> None:
    """Write a run's full record to disk so it survives restarts and can be reopened."""
    run = RUNS.get(run_id)
    if not run:
        return
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "status": run.get("status"),
        "started_at": run.get("started_at"),
        "request": run.get("request"),
        "plan": run.get("plan"),
        "compile_mode": run.get("compile_mode"),
        "source": run.get("source"),
        "total": run.get("total"),
        "records": run.get("records", []),
        "error": run.get("error"),
    }
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2))
    # A flat trace for external review tooling.
    with (run_dir / "iteration_trace.jsonl").open("w") as fh:
        for rec in run.get("records", []):
            fh.write(json.dumps(rec) + "\n")
    logger.info("Persisted run %s -> %s", run_id, run_dir)


def _list_runs() -> list[dict[str, Any]]:
    """Merge in-memory runs with any persisted on disk (disk survives restarts)."""
    by_id: dict[str, dict[str, Any]] = {}
    if RUNS_ROOT.is_dir():
        for run_dir in sorted(RUNS_ROOT.iterdir()):
            meta = run_dir / "run.json"
            if meta.is_file():
                try:
                    data = json.loads(meta.read_text())
                    by_id[data["run_id"]] = {
                        "run_id": data["run_id"],
                        "status": data.get("status"),
                        "started_at": data.get("started_at"),
                        "statement": (data.get("request") or {}).get("statement"),
                        "source": data.get("source"),
                        "total": data.get("total"),
                        "records": len(data.get("records", [])),
                    }
                except Exception:
                    continue
    # In-memory runs win (they're live).
    for rid, run in RUNS.items():
        by_id[rid] = {
            "run_id": rid,
            "status": run.get("status"),
            "started_at": run.get("started_at"),
            "statement": (run.get("request") or {}).get("statement"),
            "source": run.get("source"),
            "total": run.get("total"),
            "records": len(run.get("records", [])),
        }
    return sorted(by_id.values(), key=lambda r: r.get("started_at") or 0, reverse=True)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def _options_payload() -> dict[str, Any]:
    from segmentation.agentic_resolver import STRICTNESS_LEVELS
    from segmentation.devo_adapter import DirrDevoAdapter

    devo = DirrDevoAdapter()
    habit_labels = sorted({v.value_label for v in devo.list_features()
                           if v.feature_name in ("growth_habit_types", "phenology_types")})

    # Full tunable parameter surface. Every numeric knob that reaches the loop or
    # the detector is declared here so the UI renders it and the run carries it.
    # `advanced` knobs are model-construction / run-breaking — group them behind a
    # collapsible in the UI. type: "number" | "text" | "bool".
    params = [
        # --- Basic: the acceptance path (which candidates survive) ---
        {"key": "limit", "label": "Photos", "type": "number", "default": 40, "min": 1, "max": 200, "step": 1, "advanced": False},
        {"key": "max_turns", "label": "Max turns", "type": "number", "default": 8, "min": 1, "max": 20, "step": 1, "advanced": False},
        {"key": "min_organ_area", "label": "Min organ area", "type": "number", "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01, "advanced": False},
        {"key": "min_organ_confidence", "label": "Min organ conf", "type": "number", "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": False},
        # --- Advanced: detector / segmenter (rebuild or run-breaking) ---
        {"key": "box_threshold", "label": "DINO box threshold", "type": "number", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True},
        {"key": "text_threshold", "label": "DINO text threshold", "type": "number", "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True},
        {"key": "iou_threshold", "label": "NMS IoU threshold", "type": "number", "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True},
        {"key": "polygon_epsilon", "label": "Polygon epsilon", "type": "number", "default": 0.003, "min": 0.0, "max": 0.02, "step": 0.001, "advanced": True},
        {"key": "multimask_output", "label": "SAM multimask", "type": "bool", "default": False, "advanced": True},
        {"key": "detector_model", "label": "Detector model", "type": "text", "default": "IDEA-Research/grounding-dino-tiny", "advanced": True},
        {"key": "segmenter_model", "label": "Segmenter model", "type": "text", "default": "facebook/sam2.1-hiera-small", "advanced": True},
    ]
    return {
        "devos": ["dirr"],
        "strictness_levels": list(STRICTNESS_LEVELS),
        "scope_labels": habit_labels,
        "params": params,
        "defaults": {
            "statement": "all observations which show a deciduous tree "
                         "(highest candidate is one that shows the entire specimen)",
            "taxon_scope": ["Tree", "Shrub"],
            "strictness": "strict",
            **{p["key"]: p["default"] for p in params},
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet
        pass

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/fgv/health":
            self._send(200, {"status": "ok", "models_loaded": MODELS.loaded})
        elif self.path == "/fgv/options":
            self._send(200, _options_payload())
        elif self.path == "/fgv/areas":
            try:
                self._send(200, {"areas": list_area_instances()})
            except Exception as exc:
                self._send(502, {"error": f"could not list areas from arq-api: {exc}"})
        elif self.path == "/fgv/runs":
            self._send(200, {"runs": _list_runs()})
        elif self.path.startswith("/fgv/runs/"):
            run_id = self.path.rsplit("/", 1)[-1]
            run = RUNS.get(run_id)
            if not run:
                # Fall back to the persisted record (survives restarts).
                meta = RUNS_ROOT / run_id / "run.json"
                if meta.is_file():
                    data = json.loads(meta.read_text())
                    run = {
                        "run_id": data["run_id"], "status": data.get("status"),
                        "progress": f"{data.get('status')} ({data.get('total')} photos)",
                        "total": data.get("total", 0), "records": data.get("records", []),
                        "plan": data.get("plan"), "compile_mode": data.get("compile_mode"),
                        "request": data.get("request"), "error": data.get("error"),
                    }
            if not run:
                self._send(404, {"error": "unknown run"})
            else:
                self._send(200, run)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        # Kill switch: cancel an in-flight run (the worker checks the flag between photos).
        if self.path.startswith("/fgv/runs/") and self.path.endswith("/cancel"):
            run_id = self.path.split("/")[-2]
            run = RUNS.get(run_id)
            if not run:
                self._send(404, {"error": "unknown run"})
                return
            if run.get("status") in ("done", "error", "cancelled"):
                self._send(409, {"error": f"run already {run.get('status')}"})
                return
            run["cancel_requested"] = True
            self._send(202, {"run_id": run_id, "status": "cancelling",
                             "note": "takes effect after the current photo finishes"})
            return

        if self.path != "/fgv/runs":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            if not req.get("statement"):
                self._send(422, {"error": "statement is required"})
                return
        except json.JSONDecodeError:
            self._send(422, {"error": "invalid JSON"})
            return
        run_id = "run-" + uuid.uuid4().hex[:12]
        RUNS[run_id] = {
            "run_id": run_id, "status": "queued", "progress": "queued",
            "records": [], "total": 0, "request": req, "error": None,
            "started_at": time.time(),
        }
        threading.Thread(target=_execute_run, args=(run_id, req), daemon=True).start()
        self._send(202, {"run_id": run_id, "status": "queued"})


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Local agentic-tagging runner server.")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("FGV runner listening on http://%s:%d (models load lazily on first run)",
                args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
