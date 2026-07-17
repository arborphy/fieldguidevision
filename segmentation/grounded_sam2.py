"""Optional Grounding DINO + SAM 2.1 plant-organ segmentation backend.

Dependencies are intentionally optional so canonical adapter users do not need
PyTorch. Install the ``segmentation`` extra before using this backend.
"""

from __future__ import annotations

from io import BytesIO
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
        self._polygon_epsilon = polygon_epsilon
        self.model_version = f"{detector_model}|{segmenter_model}"

    @staticmethod
    def _load_image(url: str):
        from PIL import Image

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
        torch = self._torch
        image = self._load_image(frame.image_url)
        labels = [[str(prompt) for prompt in prompts]]
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
        text_labels = detected.get("text_labels") or detected.get("labels")
        results: list[RawSegment] = []
        for index, (label, detection_score) in enumerate(zip(text_labels, detected["scores"])):
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
