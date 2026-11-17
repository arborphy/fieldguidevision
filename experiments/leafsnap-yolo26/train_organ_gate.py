# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""E3 organ gate: fine-tune YOLO26n top-level organ detector (proof run).

Scope
-----
ONLY the six-class top-level organ detector is trained here
(leaf / bud / flower / twig / branch / entire_plant). The lower-level leaf-type
specialist is intentionally NOT trained: E3 gates it, so it must not run before
the gate is reviewed.

What this script does
---------------------
1. Preflight: dataset/manifest integrity, hash of the dataset inputs, pretrained
   ``yolo26n.pt`` (detection, not YOLOE) resolution, MPS availability.
2. Deterministic fine-tune on MPS: imgsz 320, batch 16, ~16-24 epochs with
   early stopping, fixed seed, workers=0.
3. Persist training parameters, per-epoch history, and the best checkpoint.
4. Evaluate the best checkpoint on the held-out test split:
   * standard Ultralytics box mAP50 / mAP50-95 and per-class P/R,
   * a transparent per-source evaluation (``evaluator.py``) that separates
     LeafSnap FIELD leaf boxes (real segmentation-derived localization) from
     BioImages weak full-frame organ boxes,
   * the confidence threshold is selected on VALIDATION ONLY.
5. Export the best checkpoint to ONNX opset 17, static 320x320, batch 1, FP16
   when the FP16 graph keeps the raw detection output layout (verified against
   the FP32 graph with onnxruntime); otherwise FP32 with the reason recorded.
6. Write ``ORGAN_GATE.md`` + ``metrics.json`` (E3 pass/fail + caveats).

Usage
-----
    .venv/bin/python experiments/leafsnap-yolo26/train_organ_gate.py --smoke
    .venv/bin/python experiments/leafsnap-yolo26/train_organ_gate.py
    .venv/bin/python experiments/leafsnap-yolo26/train_organ_gate.py --skip-train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import evaluator as ev  # noqa: E402

DETECTOR_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "detector"
DATASET_YAML = DETECTOR_ROOT / "dataset.yaml"
MANIFEST = DETECTOR_ROOT / "manifest.jsonl"
COUNTS = DETECTOR_ROOT / "counts.json"
OUT_ROOT = ROOT / "scratch" / "leafsnap-yolo26" / "organ-gate"

IMGSZ = 320
BATCH = 16
SEED = 0
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 5
PRETRAINED = "yolo26n.pt"
STANDARD_OPERATING_CONF = 0.25  # Ultralytics' default prediction confidence
GATE_LEAF_RECALL = 0.80
GATE_LEAF_MAP50 = 0.80


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is a hard dep in practice
        pass
    import torch

    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def package_versions() -> dict[str, str]:
    import torch
    import ultralytics

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    try:
        import onnx

        versions["onnx"] = onnx.__version__
    except Exception:
        versions["onnx"] = "missing"
    try:
        import onnxruntime

        versions["onnxruntime"] = onnxruntime.__version__
    except Exception:
        versions["onnxruntime"] = "missing"
    return versions


def resolve_pretrained(dest_dir: Path) -> Path:
    """Download/copy pretrained YOLO26n detection weights into the run dir."""
    from ultralytics.utils.downloads import attempt_download_asset

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / PRETRAINED
    if not target.exists():
        downloaded = Path(attempt_download_asset(str(dest_dir / PRETRAINED)))
        if downloaded.resolve() != target.resolve():
            shutil.copy(downloaded, target)
    if not target.exists() or target.stat().st_size < 1_000_000:
        raise RuntimeError(f"could not resolve pretrained weights at {target}")
    return target


def dataset_fingerprint() -> dict[str, Any]:
    train_images = sorted((DETECTOR_ROOT / "images" / "train").iterdir())
    return {
        "dataset_yaml": str(DATASET_YAML),
        "manifest": str(MANIFEST),
        "sha256": {
            "dataset.yaml": sha256_file(DATASET_YAML),
            "manifest.jsonl": sha256_file(MANIFEST),
            "counts.json": sha256_file(COUNTS),
        },
        "train_images": len(train_images),
        "manifest_rows": len(MANIFEST.read_text().splitlines()),
    }


def parse_results_csv(path: Path) -> list[dict[str, Any]]:
    import csv

    history: list[dict[str, Any]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clean: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                key = key.strip()
                value = (value or "").strip()
                try:
                    clean[key] = float(value)
                except ValueError:
                    clean[key] = value
            history.append(clean)
    return history


def build_smoke_dataset(out_dir: Path, n_train: int = 24, n_val: int = 12) -> Path:
    """Tiny symlinked dataset for a 1-epoch device/format smoke test."""
    root = out_dir / "smoke-dataset"
    if root.exists():
        shutil.rmtree(root)
    counts = json.loads(COUNTS.read_text())
    per_class_train = {i: max(1, counts["by_split_class"]["train"][ev.CLASS_NAMES[i]] // 120) for i in range(6)}
    per_class_val = {i: max(1, counts["by_split_class"]["val"][ev.CLASS_NAMES[i]] // 60) for i in range(6)}
    rows = {r["uid"]: r for r in ev.load_manifest(MANIFEST)}
    picked = {"train": 0, "val": 0}
    for split, target, per_class in (("train", n_train, per_class_train), ("val", n_val, per_class_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for class_id, take in per_class.items():
            chosen = [
                r for r in rows.values()
                if r["split"] == split and CLASS_IDS[r["class_name"]] == class_id
            ][:take]
            for row in chosen:
                src_img = DETECTOR_ROOT / row["image"]
                src_lbl = DETECTOR_ROOT / row["label"]
                (root / "images" / split / src_img.name).symlink_to(src_img)
                (root / "labels" / split / src_lbl.name).symlink_to(src_lbl)
                picked[split] += 1
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        f"nc: 6\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(ev.CLASS_NAMES))
    )
    return yaml_path


CLASS_IDS = ev.CLASS_IDS


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    data_yaml = DATASET_YAML
    run_name = "train"
    project = out_dir / "runs"
    if args.smoke:
        data_yaml = build_smoke_dataset(out_dir)
        project = out_dir / "smoke-runs"
        run_name = "smoke"

    weights = resolve_pretrained(out_dir / "pretrained")
    device = "mps" if torch.backends.mps.is_available() and not args.cpu else "cpu"
    params = {
        "experiment": "leafsnap-yolo26 organ gate (E3)",
        "task": "detect",
        "model": str(weights),
        "model_name": PRETRAINED,
        "device": device,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "seed": SEED,
        "deterministic": True,
        "workers": 0,
        "amp": False,
        "cache": False,
        "optimizer": "auto (Ultralytics default for fine-tuning)",
        "data": str(data_yaml),
        "dataset": dataset_fingerprint(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed_everything": True,
        "packages": package_versions(),
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "device": device,
            "torch_mps": bool(torch.backends.mps.is_available()),
        },
    }
    write_json(out_dir / "training_params.json", params)

    seed_everything(SEED)
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=0,
        seed=SEED,
        deterministic=True,
        amp=False,
        cache=False,
        project=str(project),
        name=run_name,
        exist_ok=True,
        plots=False,
        verbose=True,
        val=True,
        save=True,
        save_period=-1,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    run_dir = project / run_name
    results_csv = run_dir / "results.csv"
    history = parse_results_csv(results_csv) if results_csv.exists() else []
    best_src = run_dir / "weights" / "best.pt"
    last_src = run_dir / "weights" / "last.pt"
    (out_dir / "history.csv").write_text(results_csv.read_text() if results_csv.exists() else "")
    write_json(out_dir / "history.json", history)
    if best_src.exists():
        shutil.copy(best_src, out_dir / "best.pt")
    if last_src.exists():
        shutil.copy(last_src, out_dir / "last.pt")

    epoch_seconds = []
    for prev, cur in zip(history, history[1:]):
        try:
            epoch_seconds.append(float(cur["time"]) - float(prev["time"]))
        except (KeyError, TypeError, ValueError):
            pass

    params.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "epochs_run": len(history),
            "wall_clock_minutes": round(elapsed / 60.0, 2),
            "mean_epoch_seconds": round(statistics.mean(epoch_seconds), 1) if epoch_seconds else None,
            "best_checkpoint": str(out_dir / "best.pt") if (out_dir / "best.pt").exists() else None,
            "ultralytics_run_dir": str(run_dir),
            "early_stopped": len(history) < args.epochs,
        }
    )
    write_json(out_dir / "training_params.json", params)
    return params | {"_elapsed_s": elapsed, "_run_dir": str(run_dir)}


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def standard_test_metrics(best_pt: Path, out_dir: Path) -> dict[str, Any]:
    """Ultralytics' own val on the held-out test split (threshold-free mAP)."""
    from ultralytics import YOLO

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(best_pt))
    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False,
        project=str(out_dir / "runs"),
        name="standard-val-test",
        exist_ok=True,
    )
    box = metrics.box
    ap_index = getattr(box, "ap_class_index", [])
    nt_per_class = getattr(metrics, "nt_per_class", None)
    per_class = {}
    for pos, class_id in enumerate(ap_index):
        class_id = int(class_id)
        per_class[ev.CLASS_NAMES[class_id]] = {
            "class_id": class_id,
            "precision": round(float(box.p[pos]), 4),
            "recall": round(float(box.r[pos]), 4),
            "f1": round(float(box.f1[pos]), 4)
            if getattr(box, "f1", None) is not None
            else None,
            "ap50": round(float(box.ap50[pos]), 4),
            "ap50_95": round(float(box.ap[pos]), 4),
            "n_gt": int(nt_per_class[class_id]) if nt_per_class is not None else None,
        }
    payload = {
        "split": "test",
        "imgsz": IMGSZ,
        "conf_floor": 0.001,
        "nms_iou": 0.7,
        "box": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
        },
        "per_class": per_class,
        "note": (
            "Ultralytics report. P/R are the library's per-class values at its "
            "internal max-F1 operating point; mAP50/mAP50-95 are threshold-free. "
            "BioImages weak boxes are included, so this aggregate is not a pure "
            "localization number."
        ),
        "speed_ms_per_image": {
            k: round(float(v), 3) for k, v in metrics.speed.items()
        },
    }
    write_json(out_dir / "standard_test_val.json", payload)
    return payload


def predict_records(
    best_pt: Path,
    split: str,
    rows: Sequence[dict[str, Any]],
    out_dir: Path,
    conf_floor: float = 0.001,
) -> list[dict[str, Any]]:
    """Run inference once at a low conf floor; threshold later in the evaluator."""
    from ultralytics import YOLO

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(best_pt))
    paths = [str(DETECTOR_ROOT / r["image"]) for r in rows]
    by_path = {str((DETECTOR_ROOT / r["image"]).resolve()): r for r in rows}
    by_name = {Path(r["image"]).name: r for r in rows}

    records: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()
    results = model.predict(
        source=paths,
        imgsz=IMGSZ,
        conf=conf_floor,
        iou=0.7,
        max_det=100,
        batch=BATCH,
        device=device,
        stream=True,
        verbose=False,
        half=False,
    )
    for result in results:
        key = str(Path(result.path).resolve())
        row = by_path.get(key) or by_name.get(Path(result.path).name)
        if row is None:
            raise RuntimeError(f"unmapped prediction path {result.path}")
        boxes = result.boxes
        preds = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().tolist()
            conf = boxes.conf.detach().cpu().tolist()
            cls = boxes.cls.detach().cpu().tolist()
            for b, s, c in zip(xyxy, conf, cls):
                preds.append({"class_id": int(c), "confidence": float(s), "bbox_xyxy": b})
        records[row["uid"]] = ev.make_record(row, preds)

    ordered = [records[r["uid"]] for r in rows]
    elapsed = time.perf_counter() - t0
    out_path = out_dir / f"predictions_{split}.jsonl"
    with out_path.open("w") as fh:
        for rec in ordered:
            fh.write(json.dumps(rec) + "\n")
    write_json(
        out_dir / f"predictions_{split}_summary.json",
        {
            "split": split,
            "images": len(ordered),
            "conf_floor": conf_floor,
            "nms_iou": 0.7,
            "imgsz": IMGSZ,
            "elapsed_s": round(elapsed, 1),
            "mean_ms_per_image": round(1000.0 * elapsed / max(1, len(ordered)), 2),
            "total_predictions": sum(len(r["preds"]) for r in ordered),
        },
    )
    return ordered


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------
def _describe_onnx(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    inputs = []
    for inp in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        inputs.append(
            {
                "name": inp.name,
                "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
                "shape": dims,
            }
        )
    outputs = []
    for out in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        outputs.append(
            {
                "name": out.name,
                "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
                "shape": dims,
            }
        )
    return {
        "path": str(path),
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in model.opset_import],
        "ir_version": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "inputs": inputs,
        "outputs": outputs,
        "bytes": path.stat().st_size,
    }


def _finalize_onnx(path: Path) -> dict[str, Any]:
    """Topologically sort node order and enforce onnx.checker validity.

    Ultralytics' FP16 ONNX conversion leaves the inserted input-cast node out of
    topological order; onnxruntime tolerates it but ``onnx.checker`` rejects the
    graph, and stricter runtimes may too. Reordering nodes is output-preserving
    (verified by the parity probe), so we sort and then require the checker.
    """
    import onnx

    model = onnx.load(str(path))
    producer = {}
    for idx, node in enumerate(model.graph.node):
        for out in node.output:
            producer[out] = idx
    available = {t.name for t in model.graph.input} | {t.name for t in model.graph.initializer}
    remaining = set(range(len(model.graph.node)))
    ordered = []
    while remaining:
        progressed = False
        for idx in sorted(remaining):
            node = model.graph.node[idx]
            if all(dep in available or dep == "" for dep in node.input):
                ordered.append(node)
                available.update(node.output)
                remaining.discard(idx)
                progressed = True
        if not progressed:
            raise RuntimeError("ONNX graph has a cyclic/unresolvable dependency")
    reordered = [n.name for n in model.graph.node] != [n.name for n in ordered]
    if reordered:
        del model.graph.node[:]
        model.graph.node.extend(ordered)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    info = {"onnx_checker": "pass", "topologically_sorted": bool(reordered), "nodes": len(ordered)}
    return info


def _onnx_forward(path: Path, arr: "Any") -> "Any":
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    return inp.name, session.run(None, {inp.name: arr})


def _probe_tensor() -> "Any":
    """First held-out test image, letterboxed exactly like Ultralytics does."""
    import numpy as np
    from PIL import Image
    from ultralytics.data.augment import LetterBox

    first = next(
        row for row in ev.load_manifest(MANIFEST) if row["split"] == "test"
    )
    image = np.asarray(Image.open(DETECTOR_ROOT / first["image"]).convert("RGB"))
    letterbox = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False)
    boxed = letterbox(image=image)[..., ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(boxed, dtype=np.float32) / 255.0


def export_onnx(best_pt: Path, out_dir: Path) -> dict[str, Any]:
    """Export opset17 / static 320 / batch1; prefer FP16 when layout preserved."""
    import numpy as np
    from ultralytics import YOLO

    results: dict[str, Any] = {"attempts": {}}

    def export_one(tag: str, half: bool) -> Path:
        model = YOLO(str(best_pt))
        produced = Path(
            model.export(
                format="onnx",
                imgsz=IMGSZ,
                opset=17,
                half=half,
                dynamic=False,
                simplify=False,
                batch=1,
                device="cpu",
            )
        )
        if not produced.exists():
            produced = Path(str(best_pt.with_suffix(".onnx")))
        target = out_dir / f"organ_gate_{tag}.onnx"
        shutil.copy(produced, target)
        if produced.resolve() != target.resolve() and produced.suffix == ".onnx":
            produced.unlink(missing_ok=True)
        validity = _finalize_onnx(target)
        results["attempts"][tag] = {"half": half, "path": str(target), "ok": True, **validity}
        return target

    fp32 = export_one("fp32", half=False)
    fp32_info = _describe_onnx(fp32)

    # Parity probe uses a real held-out image (deterministic, letterboxed to
    # 320x320) rather than noise, so box-coordinate agreement is meaningful.
    probe = _probe_tensor()
    _, out32 = _onnx_forward(fp32, probe)
    results["fp32_smoke"] = {"shapes": [list(o.shape) for o in out32]}

    chosen = fp32
    chosen_kind = "fp32 (float32 I/O and weights)"
    fp16_info = None
    try:
        fp16 = export_one("fp16", half=True)
        fp16_info = _describe_onnx(fp16)
        in_dtype = fp16_info["inputs"][0]["dtype"]
        probe16 = probe.astype(np.float16) if in_dtype == "FLOAT16" else probe
        _, out16 = _onnx_forward(fp16, probe16)
        layout_match = (
            len(out16) == len(out32)
            and all(tuple(a.shape) == tuple(b.shape) for a, b in zip(out16, out32))
            and [o["name"] for o in fp16_info["outputs"]] == [o["name"] for o in fp32_info["outputs"]]
        )
        a32 = np.asarray(out32[0], dtype=np.float32)
        a16 = np.asarray(out16[0], dtype=np.float32)
        diff = np.abs(a32 - a16)
        nc = len(ev.CLASS_NAMES)
        box_diff = diff[:, :4] if diff.shape[1] == 4 + nc else diff
        score_diff = diff[:, 4:] if diff.shape[1] == 4 + nc else diff
        max_box = float(box_diff.max()) if box_diff.size else 0.0
        max_score = float(score_diff.max()) if score_diff.size else 0.0
        # FP16 must keep the raw output layout AND the detection-relevant numbers:
        # class scores (what thresholds/NMS consume) within 1e-2, box coordinates
        # (pixels at 320) within 2 px.
        numeric_ok = max_score <= 1e-2 and max_box <= 2.0
        results["fp16_parity"] = {
            "layout_matches_fp32": bool(layout_match),
            "numeric_ok": bool(numeric_ok),
            "probe": "first held-out test image, ultralytics LetterBox 320x320, RGB 0-1",
            "input_dtype": in_dtype,
            "output_dtype": fp16_info["outputs"][0]["dtype"],
            "output_shapes": [list(o.shape) for o in out16],
            "max_abs_diff_all": round(float(diff.max()), 6) if diff.size else 0.0,
            "mean_abs_diff_all": round(float(diff.mean()), 6) if diff.size else 0.0,
            "max_abs_diff_box_px": round(max_box, 6),
            "max_abs_diff_class_scores": round(max_score, 8),
            "acceptance": "layout identical; class-score diff <= 1e-2; box diff <= 2 px",
        }
        if layout_match and numeric_ok:
            chosen, chosen_kind = fp16, "fp16"
        else:
            results["fp16_rejection"] = "layout or numeric parity failed"
    except Exception as exc:  # FP16 ONNX export can fail on some stacks
        results["fp16_rejection"] = f"{type(exc).__name__}: {exc}"
        if (out_dir / "organ_gate_fp16.onnx").exists():
            (out_dir / "organ_gate_fp16.onnx").unlink()

    final = out_dir / "organ_gate.onnx"
    shutil.copy(chosen, final)
    info = _describe_onnx(final)
    info["variant"] = chosen_kind
    out_info = info["outputs"][0]
    shape = out_info["shape"]
    if isinstance(shape[-1], int) and len(shape) == 3 and shape[1] == 4 + len(ev.CLASS_NAMES):
        layout = "[batch, 4 + nc, anchors]: cx, cy, w, h, class scores (no objectness)"
    elif isinstance(shape[-1], int) and len(shape) == 3 and shape[-1] == 6:
        layout = "[batch, N, 6]: x1, y1, x2, y2, confidence, class_id (NMS-free end-to-end)"
    else:
        layout = "unexpected layout: see output shape"

    labels = {
        "dataset": "leafsnap-yolo26 detector (E3 organ gate)",
        "task": "detect",
        "model": "yolo26n (fine-tuned)",
        "class_order": {str(i): name for i, name in enumerate(ev.CLASS_NAMES)},
        "names": {name: i for i, name in enumerate(ev.CLASS_NAMES)},
        "nc": len(ev.CLASS_NAMES),
        "input": {
            "name": info["inputs"][0]["name"],
            "shape": info["inputs"][0]["shape"],
            "dtype": info["inputs"][0]["dtype"],
            "layout": "NCHW",
            "preprocessing": "RGB, letterboxed to 320x320, scaled to 0-1, no mean/std normalization",
        },
        "output": {
            "name": out_info["name"],
            "shape": out_info["shape"],
            "dtype": out_info["dtype"],
            "layout": layout,
            "coordinate_space": "input pixels (320x320 letterboxed image)",
        },
        "onnx": {
            "opset": info["opset"],
            "ir_version": info["ir_version"],
            "producer": info["producer"],
            "variant": chosen_kind,
            "io_dtype_note": (
                "FP16 exports keep float32 input/output tensors (standard "
                "Ultralytics ONNX fp16 conversion); weights/compute are FP16."
                if chosen_kind == "fp16"
                else "float32 weights and I/O."
            ),
            "dynamic": False,
            "batch": 1,
            "imgsz": IMGSZ,
            "onnx_checker": "pass",
            "node_order": "topologically sorted (Ultralytics FP16 export leaves the input cast node out of order)",
            "fp16_parity": results.get("fp16_parity"),
            "file": "organ_gate.onnx",
        },
    }
    write_json(out_dir / "organ_gate.labels.json", labels)
    results.update({"chosen": chosen_kind, "chosen_info": info, "fp32_info": fp32_info, "labels": labels})
    # Keep exactly one deliverable graph (organ_gate.onnx). Variant copies were
    # only used to verify parity; onnx_export.json records their metadata.
    (out_dir / "organ_gate_fp32.onnx").unlink(missing_ok=True)
    (out_dir / "organ_gate_fp16.onnx").unlink(missing_ok=True)
    results["retained_files"] = ["organ_gate.onnx", "organ_gate.labels.json"]
    write_json(out_dir / "onnx_export.json", results)
    return results


# --------------------------------------------------------------------------
# E3 gate + report
# --------------------------------------------------------------------------
def onnx_fidelity(onnx_path: Path, out_dir: Path) -> dict[str, Any]:
    """Validate the exported ONNX graph through Ultralytics on the test split.

    This is the end-to-end proof that the FP16 graph still detects: same data,
    same imgsz, mAP computed by the library on the ONNX model.
    """
    from ultralytics import YOLO

    model = YOLO(str(onnx_path), task="detect")
    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device="cpu",
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False,
        project=str(out_dir / "runs"),
        name="onnx-val-test",
        exist_ok=True,
    )
    box = metrics.box
    nt = getattr(metrics, "nt_per_class", None)
    per_class = {}
    for pos, class_id in enumerate(getattr(box, "ap_class_index", [])):
        cid = int(class_id)
        per_class[ev.CLASS_NAMES[cid]] = {
            "precision": round(float(box.p[pos]), 4),
            "recall": round(float(box.r[pos]), 4),
            "ap50": round(float(box.ap50[pos]), 4),
            "ap50_95": round(float(box.ap[pos]), 4),
            "n_gt": int(nt[cid]) if nt is not None else None,
        }
    payload = {
        "onnx": str(onnx_path),
        "split": "test",
        "imgsz": IMGSZ,
        "box": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
        },
        "per_class": per_class,
    }
    write_json(out_dir / "onnx_test_val.json", payload)
    return payload


def summarize_gate(
    standard: dict[str, Any],
    test_selected: dict[str, Any],
    test_standard_conf: dict[str, Any],
) -> dict[str, Any]:
    leaf_std = standard["per_class"].get("leaf", {})
    leaf_map50 = leaf_std.get("ap50")
    leaf_recall_std_conf = test_standard_conf["sources"]["leafsnap_field_leaf"]["recall"]
    leaf_recall_selected = test_selected["sources"]["leafsnap_field_leaf"]["recall"]
    leaf_map50_custom = test_selected["per_class"]["leaf"]["ap50"]

    checks = {
        "leaf_map50_ge_0.80": {
            "value": leaf_map50,
            "threshold": GATE_LEAF_MAP50,
            "pass": leaf_map50 is not None and leaf_map50 >= GATE_LEAF_MAP50,
            "source": "Ultralytics standard val on the held-out test split (threshold-free box mAP50 for class leaf)",
        },
        "leafsnap_field_leaf_recall_ge_0.80": {
            "value": leaf_recall_std_conf,
            "threshold": GATE_LEAF_RECALL,
            "pass": leaf_recall_std_conf >= GATE_LEAF_RECALL,
            "source": f"transparent evaluator, LeafSnap field test subset, confidence={STANDARD_OPERATING_CONF} (model default), IoU>=0.5",
        },
    }
    primary_pass = all(c["pass"] for c in checks.values())
    return {
        "criteria_pre_registered": {
            "primary_gate": (
                "LeafSnap FIELD leaf recall >= 0.80 AND leaf mAP50 >= 0.80, measured at "
                "the model's standard operating point (Ultralytics default conf=0.25 for "
                "recall; threshold-free box mAP50 for localization)."
            ),
            "leaf_recall_threshold": GATE_LEAF_RECALL,
            "leaf_map50_threshold": GATE_LEAF_MAP50,
            "operating_point": {
                "confidence": STANDARD_OPERATING_CONF,
                "iou": ev.IOU_MATCH,
                "selection": "library default; not tuned on test",
            },
            "secondary_reported_not_gating": [
                "nonleaf macro recall (bud/flower/twig/branch/entire_plant)",
                "BioImages weak-box organ metrics (weak labels, not localization quality)",
                "metrics at the validation-selected confidence threshold",
            ],
            "explicitly_not_used_to_hide_leaf_failure": (
                "Nonleaf macro recall and BioImages weak-box numbers are reported "
                "separately and never averaged into the leaf gate."
            ),
        },
        "checks": checks,
        "status": "PASS" if primary_pass else "FAIL",
        "secondary": {
            "leaf_map50_custom_evaluator": leaf_map50_custom,
            "leaf_recall_at_val_selected_conf": leaf_recall_selected,
            "val_selected_conf": test_selected["confidence_threshold"],
            "leafsnap_field_leaf_precision_at_std_conf": test_standard_conf["sources"]["leafsnap_field_leaf"]["precision"],
            "nonleaf_macro_recall_at_std_conf": test_standard_conf["nonleaf_macro_recall"],
            "nonleaf_macro_recall_at_selected_conf": test_selected["nonleaf_macro_recall"],
            "bioimages_weakbox_macro_recall_at_std_conf": test_standard_conf["sources"]["bioimages_weak_box"]["macro_recall"],
            "bioimages_weakbox_map50_at_selected_conf": test_selected["sources"]["bioimages_weak_box"]["map50_iou0.5"],
            "secondary_pass_if_val_selected_conf_used": bool(
                leaf_map50 is not None
                and leaf_map50 >= GATE_LEAF_MAP50
                and leaf_recall_selected >= GATE_LEAF_RECALL
            ),
        },
    }


def write_report(
    out_dir: Path,
    params: dict[str, Any],
    standard: dict[str, Any],
    gate: dict[str, Any],
    test_standard_conf: dict[str, Any],
    test_selected: dict[str, Any],
    val_report: dict[str, Any],
    selection: dict[str, Any],
    onnx: dict[str, Any],
    fidelity: dict[str, Any] | None,
) -> None:
    leaf = standard["per_class"].get("leaf", {})
    nonleaf_rows = "\n".join(
        f"| {name} | {standard['per_class'].get(name, {}).get('n_gt', 0)} | "
        f"{standard['per_class'].get(name, {}).get('precision', float('nan')):.3f} | "
        f"{standard['per_class'].get(name, {}).get('recall', float('nan')):.3f} | "
        f"{standard['per_class'].get(name, {}).get('ap50', float('nan')):.3f} | "
        f"{standard['per_class'].get(name, {}).get('ap50_95', float('nan')):.3f} |"
        for name in ev.NONLEAF_CLASSES
    )
    bio = test_selected["sources"]["bioimages_weak_box"]["per_class"]
    bio_rows = "\n".join(
        f"| {name} | {b['n_gt']} | {b['precision']:.3f} | {b['recall']:.3f} | "
        f"{(b['ap50'] if b['ap50'] is not None else float('nan')):.3f} |"
        for name, b in bio.items()
    )
    epochs_run = params.get("epochs_run")
    minutes = params.get("wall_clock_minutes")
    chosen = onnx.get("chosen")
    out_shape = onnx["labels"]["output"]["shape"]
    report = f"""# E3 — LeafSnap organ gate (YOLO26n proof run)

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by
`experiments/leafsnap-yolo26/train_organ_gate.py`.
Only the **top-level six-class organ detector** was trained. The lower-level
leaf specialist was not trained; E3 gates it.

## E3 criteria (pre-registered before results)

Primary gate — **both** must hold at the model's standard operating point:

1. **LeafSnap FIELD leaf recall >= {GATE_LEAF_RECALL:.2f}** on the held-out test
   split (LeafSnap field images only, IoU >= 0.5, model default confidence
   `{STANDARD_OPERATING_CONF}`).
2. **Leaf mAP50 >= {GATE_LEAF_MAP50:.2f}** on the held-out test split
   (threshold-free Ultralytics box mAP50 for class `leaf`).

Rules:

- The operating point is the library default (`conf={STANDARD_OPERATING_CONF}`);
  the gate is never tuned on test.
- The transparent evaluator's confidence threshold is selected on **validation
  only** and reported as a secondary number.
- Nonleaf macro recall is reported separately and is **not** part of the gate;
  it cannot offset or hide a leaf failure.
- BioImages boxes are weak full-frame / near-full-frame labels. Their metrics
  are a sanity check, **not** localization quality.
- LeafSnap boxes are segmentation-derived, not human-drawn; they inherit
  segmentation error.

## Training

| item | value |
|---|---|
| base model | `{PRETRAINED}` (YOLO26 detection, not YOLOE) |
| device | `{params.get('device')}` (Apple Silicon MPS) |
| imgsz / batch | {IMGSZ} / {BATCH} |
| epochs run / requested | {epochs_run} / {params.get('epochs_requested')} (patience {params.get('patience')}) |
| seed / workers | {SEED} / 0 (deterministic) |
| wall clock | {minutes} min |
| mean epoch | {params.get('mean_epoch_seconds')} s |
| early stopped | {params.get('early_stopped')} |
| train images | {params['dataset']['train_images']} |

Dataset fingerprint (sha256):

- `dataset.yaml` `{params['dataset']['sha256']['dataset.yaml'][:16]}…`
- `manifest.jsonl` `{params['dataset']['sha256']['manifest.jsonl'][:16]}…`

## Standard Ultralytics test metrics (held-out test split)

| metric | value |
|---|---|
| box mAP50 | **{standard['box']['map50']:.4f}** |
| box mAP50-95 | {standard['box']['map50_95']:.4f} |
| precision (all) | {standard['box']['precision']:.4f} |
| recall (all) | {standard['box']['recall']:.4f} |

Per-class (Ultralytics: P/R at its internal max-F1 point, mAP threshold-free):

| class | GT | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| leaf | {leaf.get('n_gt', 0)} | {leaf.get('precision', float('nan')):.3f} | {leaf.get('recall', float('nan')):.3f} | {leaf.get('ap50', float('nan')):.3f} | {leaf.get('ap50_95', float('nan')):.3f} |
{nonleaf_rows}

### Validation context (selection split, not part of the gate)

Validation mixes LeafSnap lab + field images and different species than test;
it is shown only to make the threshold choice auditable.

| metric | val @ conf {selection['confidence']} |
|---|---|
| leaf P / R / F1 | {val_report['per_class']['leaf']['precision']:.3f} / {val_report['per_class']['leaf']['recall']:.3f} / {val_report['per_class']['leaf']['f1']:.3f} |
| leaf AP50 | {val_report['per_class']['leaf']['ap50']} |
| nonleaf macro recall | {val_report['nonleaf_macro_recall']} |
| BioImages weak-box macro recall | {val_report['sources']['bioimages_weak_box']['macro_recall']} |

## E3 result

- Leaf mAP50 (standard, test): **{gate['checks']['leaf_map50_ge_0.80']['value']}** (threshold {GATE_LEAF_MAP50}) → {'PASS' if gate['checks']['leaf_map50_ge_0.80']['pass'] else 'FAIL'}
- LeafSnap field leaf recall @ conf {STANDARD_OPERATING_CONF}: **{gate['checks']['leafsnap_field_leaf_recall_ge_0.80']['value']}** (threshold {GATE_LEAF_RECALL}) → {'PASS' if gate['checks']['leafsnap_field_leaf_recall_ge_0.80']['pass'] else 'FAIL'}

### **E3: {gate['status']}**

Secondary (reported, not gating):

| metric | value |
|---|---|
| leaf recall @ val-selected conf {selection['confidence']} | {gate['secondary']['leaf_recall_at_val_selected_conf']} |
| nonleaf macro recall @ conf {STANDARD_OPERATING_CONF} | {gate['secondary']['nonleaf_macro_recall_at_std_conf']} |
| nonleaf macro recall @ val-selected conf | {gate['secondary']['nonleaf_macro_recall_at_selected_conf']} |
| BioImages weak-box macro recall @ conf {STANDARD_OPERATING_CONF} | {gate['secondary']['bioimages_weakbox_macro_recall_at_std_conf']} |
| would pass if val-selected conf were used | {gate['secondary']['secondary_pass_if_val_selected_conf_used']} |

## Source-split evaluation (transparent evaluator, IoU 0.5)

Threshold `{selection['confidence']}` selected on **validation** (rule:
{selection['rule']}).

### LeafSnap FIELD leaf (real boxes)

| metric | value |
|---|---|
| images | {test_selected['sources']['leafsnap_field_leaf']['n_images']} |
| precision | {test_selected['sources']['leafsnap_field_leaf']['precision']:.4f} |
| recall | {test_selected['sources']['leafsnap_field_leaf']['recall']:.4f} |
| F1 | {test_selected['sources']['leafsnap_field_leaf']['f1']:.4f} |
| AP50 (threshold-free, IoU 0.5) | {test_selected['sources']['leafsnap_field_leaf']['ap50']} |

Cross-class firings across all test images, including weak-box BioImages
(GT→pred class, overlapping box):
{json.dumps(test_selected['cross_class_matches'], ensure_ascii=False) if test_selected['cross_class_matches'] else 'none'}

### BioImages organ (WEAK boxes — sanity only, NOT localization quality)

| class | GT | P | R | AP50 |
|---|---:|---:|---:|---:|
{bio_rows}

Weak-box macro recall: {test_selected['sources']['bioimages_weak_box']['macro_recall']} ·
weak-box mAP50: {test_selected['sources']['bioimages_weak_box']['map50_iou0.5']}

## ONNX export

- `organ_gate.onnx` — opset **17**, static **320x320**, batch **1**, **{chosen.upper()}**,
  `onnx.checker` pass (node order sorted after the FP16 conversion)
- input: `{onnx['labels']['input']['name']}` {onnx['labels']['input']['shape']} `{onnx['labels']['input']['dtype']}`
- output: `{onnx['labels']['output']['name']}` `{out_shape}` `{onnx['labels']['output']['dtype']}`
- output layout: {onnx['labels']['output']['layout']}
- FP16 parity vs FP32: {json.dumps(onnx.get('fp16_parity'))}
- ONNX test mAP50 / mAP50-95 (Ultralytics on the exported graph, CPU):
  **{fidelity['box']['map50'] if fidelity else 'n/a'}** / {fidelity['box']['map50_95'] if fidelity else 'n/a'}
  (leaf mAP50 {fidelity['per_class']['leaf']['ap50'] if fidelity else 'n/a'} vs
  PyTorch {leaf.get('ap50')})

## Caveats

- **Bud is data-starved**: 49 source images total (train 31 / val 10 / test 8);
  its per-class numbers are noisy and a single miss moves recall by ~12 points.
- **BioImages has no boxes**: near-full-frame weak labels for organs and a full
  frame for `entire_plant`. High BioImages scores are an upper bound on
  "did it fire on the right image with roughly the right class", never a
  localization claim.
- **LeafSnap boxes are segmentation-derived** (mask foreground bbox), not
  human-drawn; they inherit mask errors.
- **LeafSnap field images are curated single-leaf photographs**: the leaf
  dominates the frame. The gate's recall is evidence that this detector finds
  the curated leaf, NOT evidence for multi-leaf / cluttered-scene detection.
- **Leaf mAP50-95 is lower than mAP50** ({leaf.get('ap50_95', float('nan')):.3f}
  vs {leaf.get('ap50', float('nan')):.3f}): boxes are close but not pixel-tight;
  the gate is specified at IoU 0.5 only.
- **One GT box per image** by construction, so the evaluator's greedy
  IoU-0.5 matching is exact for this dataset; it is not a general detector
  benchmark.
- Splits are species-disjoint (LeafSnap) / individual-disjoint (BioImages), so
  test measures generalization to unseen species/individuals.
- Proof-run budget: {epochs_run} epochs, no hyperparameter sweep. A FAIL is
  evidence about this budget and data, not a proof about the architecture.

## Artifacts

- `best.pt`, `last.pt` — checkpoints
- `history.csv`, `history.json` — per-epoch training history
- `training_params.json` — full training parameters + dataset hashes
- `standard_test_val.json` — Ultralytics test metrics
- `predictions_val.jsonl`, `predictions_test.jsonl` — raw predictions at conf floor 0.001
- `metrics.json` — machine-readable results + E3 verdict
- `organ_gate.onnx`, `organ_gate.labels.json`, `onnx_export.json`,
  `onnx_test_val.json` (ONNX test fidelity)
"""
    (out_dir / "ORGAN_GATE.md").write_text(report)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    parser.add_argument("--skip-train", action="store_true", help="reuse best.pt and only evaluate/export")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="1-epoch tiny-dataset device/format check")
    parser.add_argument("--cpu", action="store_true", help="force CPU (debug only)")
    args = parser.parse_args(argv)

    if args.smoke:
        args.epochs = 1
        args.patience = 1
        args.out = OUT_ROOT / "smoke"
        args.skip_onnx = True

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DATASET_YAML.exists() or not MANIFEST.exists():
        raise SystemExit(f"missing dataset artifacts under {DETECTOR_ROOT}")
    validation = json.loads((DETECTOR_ROOT.parent / "validation.json").read_text())
    if validation.get("status") != "pass":
        raise SystemExit("dataset validation.json is not pass; refusing to train")

    manifest_rows = ev.load_manifest(MANIFEST)
    by_split = {split: [r for r in manifest_rows if r["split"] == split] for split in ("train", "val", "test")}
    print(
        f"dataset ok: train={len(by_split['train'])} val={len(by_split['val'])} test={len(by_split['test'])}",
        flush=True,
    )

    if args.skip_train:
        best_pt = out_dir / "best.pt"
        if not best_pt.exists():
            raise SystemExit(f"--skip-train requires {best_pt}")
        params = json.loads((out_dir / "training_params.json").read_text())
    else:
        params = train(args, out_dir)
        best_pt = out_dir / "best.pt"
        if not best_pt.exists():
            raise SystemExit(f"training finished without a best checkpoint at {best_pt}")

    if args.smoke:
        print("smoke run complete:", params.get("wall_clock_minutes"), "min", flush=True)
        return 0

    print("standard Ultralytics test evaluation...", flush=True)
    standard = standard_test_metrics(best_pt, out_dir)

    print("predicting val/test at conf floor 0.001...", flush=True)
    val_records = predict_records(best_pt, "val", by_split["val"], out_dir)
    test_records = predict_records(best_pt, "test", by_split["test"], out_dir)

    selection = ev.select_confidence(val_records)
    write_json(out_dir / "confidence_selection.json", selection)
    print(f"validation-selected confidence: {selection['confidence']} (macro F1 {selection['macro_f1']})", flush=True)

    val_selected = ev.evaluate(val_records, selection["confidence"], label="val_selected_conf")
    test_selected = ev.evaluate(test_records, selection["confidence"], label="test_val_selected_conf")
    test_standard_conf = ev.evaluate(
        test_records, STANDARD_OPERATING_CONF, label="test_standard_operating_conf"
    )
    write_json(out_dir / "source_split_eval.json", {
        "val": val_selected,
        "test_at_val_selected_conf": test_selected,
        "test_at_standard_conf": test_standard_conf,
    })

    gate = summarize_gate(standard, test_selected, test_standard_conf)

    onnx: dict[str, Any] = {"chosen": "skipped", "labels": {"output": {"shape": [], "name": "", "dtype": "", "layout": ""},
                                                           "input": {"shape": [], "name": "", "dtype": ""}},
                            "fp16_parity": None}
    fidelity: dict[str, Any] | None = None
    if not args.skip_onnx:
        print("exporting ONNX...", flush=True)
        onnx = export_onnx(best_pt, out_dir)
        print("validating ONNX on the test split...", flush=True)
        fidelity = onnx_fidelity(out_dir / "organ_gate.onnx", out_dir)

    metrics = {
        "experiment": "leafsnap-yolo26 organ gate (E3 proof run)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "top-level 6-class organ detector only; lower-level leaf specialist NOT trained",
        "training": {k: v for k, v in params.items() if not k.startswith("_")},
        "standard_test_val": standard,
        "confidence_selection_val_only": {
            "confidence": selection["confidence"],
            "macro_f1": selection["macro_f1"],
            "rule": selection["rule"],
            "grid": selection["grid"],
            "curve": selection["curve"],
        },
        "source_split": {
            "val_at_selected_conf": val_selected,
            "test_at_val_selected_conf": test_selected,
            "test_at_standard_conf": test_standard_conf,
        },
        "e3": gate,
        "onnx": {
            "chosen": onnx.get("chosen"),
            "file": "organ_gate.onnx",
            "labels_file": "organ_gate.labels.json",
            "info": onnx.get("chosen_info"),
            "fp16_parity": onnx.get("fp16_parity"),
            "fp16_rejection": onnx.get("fp16_rejection"),
            "test_val_fidelity": fidelity,
        },
        "caveats": [
            "bud: only 49 source images (train 31 / val 10 / test 8); per-class bud metrics are high-variance",
            "BioImages has no boxes: organ boxes are near-full-frame weak labels, entire_plant is full frame; metrics are a sanity check, not localization quality",
            "LeafSnap boxes are segmentation-derived, not human-drawn; they inherit mask error",
            "one GT box per image by dataset construction; greedy IoU-0.5 matching is exact here, not a general benchmark",
            "proof-run budget only (no hyperparameter sweep)",
        ],
    }
    write_json(out_dir / "metrics.json", metrics)
    write_report(
        out_dir, params, standard, gate, test_standard_conf, test_selected,
        val_selected, selection, onnx, fidelity,
    )
    (out_dir / "EVALUATION_CRITERIA.md").write_text(
        "# E3 criteria (pre-registered)\n\n"
        + json.dumps(gate["criteria_pre_registered"], indent=2)
        + "\n"
    )

    print(
        f"E3 {gate['status']}: leaf mAP50={gate['checks']['leaf_map50_ge_0.80']['value']} "
        f"leaf_recall@{STANDARD_OPERATING_CONF}={gate['checks']['leafsnap_field_leaf_recall_ge_0.80']['value']} "
        f"(val-selected conf {selection['confidence']}: recall={gate['secondary']['leaf_recall_at_val_selected_conf']})",
        flush=True,
    )
    print(f"artifacts in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
