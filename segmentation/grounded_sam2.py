"""Optional Grounding DINO + SAM 2.1 plant-organ segmentation backend.

Dependencies are intentionally optional so canonical adapter users do not need
PyTorch. Install the ``segmentation`` extra before using this backend.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from .pipeline import CaptureFrame, RawSegment, canonical_organ_label


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
        from .gpu_lock import MPS_LOCK

        with MPS_LOCK:
            detector_inputs = self._detector_processor(
                images=image, text=labels, return_tensors="pt"
            ).to(self._device)
            with torch.inference_mode():
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
            return ()
            
        import torchvision
        
        if self._iou_threshold is not None:
            import torchvision
            label_list = detected.get("text_labels") or detected.get("labels")
            scores = detected["scores"]
            label_to_idx = {lbl: idx for idx, lbl in enumerate(set(label_list))}
            class_idxs = torch.tensor([label_to_idx[lbl] for lbl in label_list], device=boxes.device)
            keep_indices = torchvision.ops.batched_nms(boxes, scores, class_idxs, iou_threshold=self._iou_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            text_labels = [label_list[i] for i in keep_indices.cpu().tolist()]
        else:
            text_labels = detected.get("text_labels") or detected.get("labels")
            scores = detected["scores"]

        with MPS_LOCK:
            sam_inputs = self._sam_processor(
                images=image,
                input_boxes=[boxes.detach().cpu().tolist()],
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                sam_outputs = self._sam(**sam_inputs, multimask_output=False)
        masks = self._sam_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"]
        )[0]
        
        results: list[RawSegment] = []
        for index, (label, detection_score) in enumerate(zip(text_labels, scores)):
            mask = masks[index, 0].numpy()
            polygon = self._mask_to_polygon(mask, image.width, image.height)
            if len(polygon) < 3:
                continue
            detector_label = str(label)
            results.append(
                RawSegment(
                    label=canonical_organ_label(detector_label, prompts),
                    polygon=polygon,
                    detection_confidence=float(detection_score),
                    mask_quality=float(sam_outputs.iou_scores[0, index, 0].cpu()),
                    prompt=detector_label,
                    detector_label=detector_label,
                )
            )
        return tuple(results)
