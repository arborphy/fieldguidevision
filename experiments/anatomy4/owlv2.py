# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""OWLv2 open-vocabulary *box* proposal backend (HF transformers).

Architecture-independent comparison point: a ViT image tower with a CLIP text
tower and image-conditioned object queries. Box-only, one forward pass per
image with every class prompt in the same query batch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from .altdet import (
    Timer,
    build_output_manifest,
    dedupe_same_class,
    make_box_proposal,
    prompt_class_index,
    prompts_for,
    select_images,
    utc_now,
)
from .contracts import AnatomyConfig

LOGGER = logging.getLogger("anatomy4.owlv2")


def run_owlv2(
    manifest: dict[str, Any],
    config: AnatomyConfig,
    out_dir: str | Path,
    *,
    model_name: str = "google/owlv2-base-patch16-ensemble",
    device: str = "cuda",
    score_threshold: float = 0.02,
    iou: float = 0.5,
    half: bool = True,
    max_per_class: int = 10,
    bucket: str | None = None,
    limit: int | None = None,
    audit_buckets: Sequence[str] | None = None,
    prompts_override: Mapping[str, Sequence[str]] | None = None,
    keep_duplicates: bool = False,
) -> dict[str, Any]:
    """Run OWLv2 box proposals.

    ``prompts_override`` and ``audit_buckets`` let a caller apply a reviewed
    per-class operating point over a chosen scope (e.g. needle prompts on every
    plant stratum) without building a synthetic config per scope.
    ``keep_duplicates`` stores every raw candidate; provenance-preserving dedupe
    is then the caller's job (``hybrid.merge_prompt_duplicates``).
    """
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    images = select_images(manifest, bucket=bucket, limit=limit)
    if audit_buckets is not None:
        allowed = set(audit_buckets)
        images = [image for image in images if image.get("audit_bucket") in allowed]
    prompts = (
        {name: list(values) for name, values in prompts_override.items()}
        if prompts_override
        else prompts_for(config)
    )
    flat = prompt_class_index(prompts)
    phrases = [prompt for prompt, _ in flat]

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()
    if half and device.startswith("cuda"):
        model = model.half()
    if hasattr(model, "config"):
        model.config.use_cache = False
    provenance = {
        "contract": config.contract,
        "config_fingerprint": config.fingerprint(),
        "backend": "owlv2",
        "detector": model_name,
        "prompts": {name: list(values) for name, values in prompts.items()},
        "query_order": phrases,
        "score_threshold": score_threshold,
        "nms_iou": iou,
        "max_per_class": max_per_class,
        "device": device,
        "half": bool(half and device.startswith("cuda")),
        "keep_duplicates": bool(keep_duplicates),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "created_at": utc_now(),
    }

    records: list[dict[str, Any]] = []
    wall = Timer()
    raw_detections = 0
    for index, image in enumerate(images, start=1):
        photo_id = str(image.get("photo_id") or image.get("image_id"))
        image_path = image.get("download", {}).get("path")
        if not image_path or not Path(image_path).is_file():
            continue
        started = Timer()
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            inputs = processor(text=[phrases], images=rgb, return_tensors="pt")
        inputs = {
            key: value.to(device)
            if not torch.is_floating_point(value)
            else value.to(device=device, dtype=torch.float16 if provenance["half"] else torch.float32)
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_threshold,
        )[0]
        candidates: list[dict[str, Any]] = []
        for rank, (box, score, label) in enumerate(
            zip(
                results["boxes"].tolist(),
                results["scores"].tolist(),
                results["labels"].tolist(),
            )
        ):
            class_index = int(label)
            if class_index < 0 or class_index >= len(flat):
                continue
            prompt, class_name = flat[class_index]
            if float(score) < config.class_by_name[class_name].box_threshold:
                continue
            raw_detections += 1
            candidates.append(
                make_box_proposal(
                    photo_id=photo_id,
                    observation_id=str(image.get("observation_id") or ""),
                    image_path=image_path,
                    audit_bucket=image.get("audit_bucket"),
                    class_name=class_name,
                    prompt=prompt,
                    score=float(score),
                    box=box,
                    width=width,
                    height=height,
                    provenance=provenance,
                    detector_rank=rank,
                    slug="o",
                    config=config,
                )
            )
        proposals = (
            candidates
            if keep_duplicates
            else dedupe_same_class(candidates, iou_threshold=iou, max_per_class=max_per_class)
        )
        record = {
            "image_id": photo_id,
            "photo_id": photo_id,
            "observation_id": str(image.get("observation_id") or ""),
            "audit_bucket": image.get("audit_bucket"),
            "selected_for_audit": True,
            "image_path": image_path,
            "download": image.get("download", {}),
            "original_size": [width, height],
            "inference_size": None,
            "elapsed_seconds": round(started.elapsed, 3),
            "proposals": proposals,
            "memory": {
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if torch.cuda.is_available()
                else 0.0,
            },
            "model": provenance,
        }
        records.append(record)
        if index % 25 == 0:
            LOGGER.info(
                "[%d/%d] %s proposals=%d", index, len(images), photo_id, len(proposals)
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_manifest = build_output_manifest(
        records,
        config=config,
        backend="owlv2",
        provenance=provenance,
        audit_bucket=bucket,
        total_runtime_seconds=wall.elapsed,
        prompts=prompts,
    )
    result_manifest["summary"]["runtime_seconds_per_image"] = round(
        wall.elapsed / max(1, len(records)), 4
    )
    result_manifest["summary"]["raw_detections_above_floor"] = raw_detections
    (output / "proposals.json").write_text(json.dumps(result_manifest, indent=1))
    return result_manifest
