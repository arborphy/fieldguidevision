"""Optional Grounding DINO + SAM 2.1 plant-organ segmentation backend.

Dependencies are intentionally optional so canonical adapter users do not need
PyTorch. Install the ``segmentation`` extra before using this backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from .pipeline import CaptureFrame, RawSegment, _bbox, _polygon_area, canonical_organ_label


@dataclass(frozen=True)
class DetectionCandidate:
    """One detector box carried through NMS, SAM, and the acceptance band.

    This is the full per-candidate signal the experiment needs — nothing is
    dropped between Grounding-DINO emitting a box and the resolver accepting or
    rejecting it. ``bbox`` is normalized (0-1) ``(x0, y0, x1, y1)`` so the UI can
    overlay it directly on the source image.
    """

    prompt: str  # the REC phrase the agent wrote for this detection call
    detector_label: str  # the label Grounding-DINO attached to this box
    organ: str  # canonical organ label (mapping of detector_label)
    box_score: float  # Grounding-DINO box confidence (post box/text threshold)
    bbox: tuple[float, float, float, float]  # normalized (x0, y0, x1, y1)
    nms_kept: bool  # survived batched NMS (False = suppressed as a duplicate)
    mask_quality: float | None = None  # SAM predicted-IoU for this box's mask
    polygon: tuple[tuple[float, float], ...] = ()  # normalized SAM polygon
    area_fraction: float | None = None  # polygon area as a fraction of the frame
    accepted_organ: str | None = None  # set by the resolver when it accepts this candidate
    verdict: str = ""  # resolver's accept/reject reason (filled downstream)


@dataclass(frozen=True)
class DetectionEvent:
    """The complete detector output for one grounding turn (one REC call)."""

    rec_phrase: str
    candidates: tuple[DetectionCandidate, ...]
    detector_model: str
    segmenter_model: str
    box_threshold: float
    text_threshold: float
    iou_threshold: float | None


class GroundedSam2Segmenter:
    model_name = "grounding-dino+sam2.1"

    def __init__(
        self,
        *,
        detector_model: str = "IDEA-Research/grounding-dino-tiny",
        segmenter_model: str = "facebook/sam2.1-hiera-small",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        iou_threshold: float | None = 0.3,
        polygon_epsilon: float = 0.003,
        multimask_output: bool = False,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Sam2Model,
                Sam2Processor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Grounded SAM 2 dependencies are missing; install with "
                "`uv sync --extra segmentation`"
            ) from exc

        self._torch = torch
        self._device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
            "cpu"
        )
        self._detector_processor = AutoProcessor.from_pretrained(detector_model)
        self._detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_model).to(self._device)
        self._sam_processor = Sam2Processor.from_pretrained(segmenter_model)
        self._sam = Sam2Model.from_pretrained(segmenter_model).to(self._device)
        self._detector.eval()
        self._sam.eval()
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._iou_threshold = iou_threshold
        self._polygon_epsilon = polygon_epsilon
        self._multimask_output = multimask_output
        self.model_version = f"{detector_model}|{segmenter_model}"

    @staticmethod
    def _load_image(url: str):
        from PIL import Image

        if url.startswith("file://"):
            return Image.open(Path(url.removeprefix("file://"))).convert("RGB")
        request = Request(url, headers={"User-Agent": "ArborphyFieldguidevision/0.1"})
        with urlopen(request, timeout=30) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")

    def _mask_to_polygon(self, mask, width: int, height: int) -> tuple[tuple[float, float], ...]:
        import cv2
        import numpy as np

        binary = np.asarray(mask, dtype=np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return ()
        contour = max(contours, key=cv2.contourArea)
        epsilon = self._polygon_epsilon * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        return tuple((float(x) / width, float(y) / height) for x, y in simplified)

    def segment(self, frame: CaptureFrame, prompts) -> tuple[RawSegment, ...]:
        from PIL import Image

        image = Image.open(frame.local_path).convert("RGB") if frame.local_path else self._load_image(frame.image_url)
        return self.segment_image(image, prompts)

    def segment_image(self, image, prompts) -> tuple[RawSegment, ...]:
        """Backward-compatible path: accepted segments only, no signal capture."""
        segments, _event = self.detect_image(image, prompts)
        return segments

    def detect_image(self, image, prompts) -> tuple[tuple[RawSegment, ...], DetectionEvent]:
        """Run detector + SAM, capturing the FULL candidate signal.

        Returns ``(accepted_segments, event)`` where ``accepted_segments`` is the
        same tuple ``segment_image`` historically produced (NMS survivors with a
        usable polygon) and ``event`` records every detector box — including the
        ones NMS suppressed or whose mask failed — with per-candidate scores,
        verdicts, and normalized bboxes/polygons for the experiment UI.
        """
        from .gpu_lock import MPS_LOCK

        prompts = list(prompts)
        with MPS_LOCK:
            detector_inputs = self._detector_processor(
                images=image, text=prompts, return_tensors="pt"
            ).to(self._device)
            with self._torch.inference_mode():
                detector_outputs = self._detector(**detector_inputs)
        detected = self._detector_processor.post_process_grounded_object_detection(
            detector_outputs,
            detector_inputs.input_ids,
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = detected["boxes"]
        if len(boxes) == 0:
            return (), DetectionEvent(
                rec_phrase="; ".join(prompts),
                candidates=(),
                detector_model=self.model_version.split("|")[0],
                segmenter_model=self.model_version.split("|")[1],
                box_threshold=self._box_threshold,
                text_threshold=self._text_threshold,
                iou_threshold=self._iou_threshold,
            )

        raw_labels = detected.get("text_labels") or detected.get("labels")
        raw_scores = detected["scores"]
        W, H = image.width, image.height

        # Batched NMS per label — record which boxes survive vs. are suppressed.
        if self._iou_threshold is not None:
            import torchvision

            label_list = list(raw_labels)
            label_to_idx = {lbl: idx for idx, lbl in enumerate(set(label_list))}
            class_idxs = self._torch.tensor([label_to_idx[lbl] for lbl in label_list], device=boxes.device)
            keep_indices = set(torchvision.ops.batched_nms(boxes, raw_scores, class_idxs, iou_threshold=self._iou_threshold).cpu().tolist())
        else:
            keep_indices = set(range(len(boxes)))

        # SAM runs only on the NMS survivors (the masks we'd actually use); the
        # suppressed boxes are still recorded (without a mask) so the UI can show
        # what the detector saw before dedup.
        keep_sorted = sorted(keep_indices)
        kept_boxes = boxes[keep_sorted] if keep_sorted else boxes[:0]
        sam_masks = None
        sam_iou = None
        if len(kept_boxes) > 0:
            with MPS_LOCK:
                sam_inputs = self._sam_processor(
                    images=image,
                    input_boxes=[kept_boxes.detach().cpu().tolist()],
                    return_tensors="pt",
                ).to(self._device)
                with self._torch.inference_mode():
                    sam_outputs = self._sam(**sam_inputs, multimask_output=self._multimask_output)
            sam_masks = self._sam_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"]
            )[0]
            sam_iou = sam_outputs.iou_scores[0].cpu()  # [num_kept, num_masks]

        # Index the SAM outputs by their original (pre-NMS) box position.
        sam_by_orig: dict[int, tuple[object, float]] = {}
        if sam_masks is not None:
            for kept_pos, orig_idx in enumerate(keep_sorted):
                sam_by_orig[orig_idx] = (sam_masks[kept_pos], float(sam_iou[kept_pos, 0]))

        candidates: list[DetectionCandidate] = []
        accepted: list[RawSegment] = []
        for orig_idx in range(len(boxes)):
            box = boxes[orig_idx].detach().cpu().tolist()  # xyxy absolute px
            norm_bbox = (box[0] / W, box[1] / H, box[2] / W, box[3] / H)
            label = str(raw_labels[orig_idx])
            score = float(raw_scores[orig_idx])
            nms_kept = orig_idx in keep_indices

            polygon: tuple[tuple[float, float], ...] = ()
            mask_quality: float | None = None
            area: float | None = None
            if nms_kept and orig_idx in sam_by_orig:
                mask, mask_quality = sam_by_orig[orig_idx]
                polygon = self._mask_to_polygon(mask[0].numpy(), W, H)
                if len(polygon) >= 3:
                    area = _polygon_area(polygon)
                else:
                    polygon = ()

            candidates.append(
                DetectionCandidate(
                    prompt="; ".join(prompts),
                    detector_label=label,
                    organ=canonical_organ_label(label, prompts),
                    box_score=score,
                    bbox=norm_bbox,
                    nms_kept=nms_kept,
                    mask_quality=mask_quality,
                    polygon=polygon,
                    area_fraction=area,
                )
            )
            if nms_kept and len(polygon) >= 3:
                accepted.append(
                    RawSegment(
                        label=canonical_organ_label(label, prompts),
                        polygon=polygon,
                        detection_confidence=score,
                        mask_quality=float(mask_quality) if mask_quality is not None else 0.0,
                        prompt=label,
                        detector_label=label,
                    )
                )

        event = DetectionEvent(
            rec_phrase="; ".join(prompts),
            candidates=tuple(candidates),
            detector_model=self.model_version.split("|")[0],
            segmenter_model=self.model_version.split("|")[1],
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            iou_threshold=self._iou_threshold,
        )
        return tuple(accepted), event
