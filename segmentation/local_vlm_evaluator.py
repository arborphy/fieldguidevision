"""Local Vision-Language Model Evaluator for Field Guide Vision.

Loads a local VLM (like Gemma 4 12B) into VRAM via transformers on MPS/CUDA.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from PIL import Image

from .semantic_context_payload import SemanticContextPayload
from .semantic_anchor import SemanticAnchorDecision

logger = logging.getLogger(__name__)


class LocalVlmEvaluator:
    """Evaluates scenes natively using Hugging Face transformers."""

    def __init__(self, model_id: str = "google/gemma-4-12b-it", device: str | None = None) -> None:
        self.model_id = model_id
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as e:
            raise RuntimeError("transformers and torch must be installed to run local VLM.") from e

        self._torch = torch
        self._device = device or (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = self._torch.bfloat16 if self._device == "mps" else self._torch.float16
        logger.info("Loading Local VLM (%s) on %s in %s...", self.model_id, self._device, self._dtype)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype,
        ).to(self._device)
        self.model.eval()
        
        logger.info("Local VLM loaded successfully.")

    def __call__(self, image: Image.Image, payload: SemanticContextPayload) -> SemanticAnchorDecision:
        prompt_text = payload.format_vlm_prompt()
        response_text = self.generate_text(image, prompt_text)
        return self._parse_decision(response_text)

    def generate_text(self, image: Image.Image, prompt_text: str) -> str:
        """Run one VLM turn and return the raw response text (for agent loops)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # Shortcut thinking channel if model uses it
        if "<|turn>model" in text and "<|channel>thought" not in text:
            text += "<|channel>thought\nDirect JSON response requested.\n<channel|>\n"

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )
        
        from .gpu_lock import MPS_LOCK

        with MPS_LOCK:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
                
            with self._torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            
        full_text = self.processor.decode(output_ids[0], skip_special_tokens=False)
        full_text = full_text.replace("<pad>", "")  # Strip MPS padding artifacts
        logger.info("VLM Full Decode (len=%d): %r", len(full_text), full_text[:1000])  # Log first 1000 chars for debugging
        
        if "<channel|>" in full_text:
            response_text = full_text.split("<channel|>")[-1]
        elif "<|turn>model" in full_text:
            response_text = full_text.split("<|turn>model")[-1]
        elif "<start_of_turn>model" in full_text:
            response_text = full_text.split("<start_of_turn>model")[-1]
        else:
            # Fallback for models that don't use Gemma chat template or are encoder-decoder
            if output_ids.shape[1] > inputs["input_ids"].shape[1]:
                generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            else:
                response_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
                
        response_text = response_text.replace("<end_of_turn>", "").replace("<eos>", "").replace("<|end|>", "").strip()
        logger.info("VLM Parsed Response (len=%d):\n%s", len(response_text), response_text)
        return response_text

    def _parse_decision(self, text: str) -> SemanticAnchorDecision:
        """Robustly parse JSON schema output from the VLM."""
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start:end+1]
            else:
                json_str = text
                
        try:
            data = json.loads(json_str)
            box = data.get("focus_bounding_box")
            box_tuple = tuple(float(x) for x in box) if isinstance(box, list) and len(box) == 4 else None
            
            return SemanticAnchorDecision(
                scene_classification=data.get("scene_classification", "Parse Error"),
                confidence=float(data.get("confidence", 0.0)),
                is_target_valid=bool(data.get("is_target_valid", False)),
                focal_directives=tuple(data.get("focal_directives", [])),
                micro_directives=tuple(),  # Handled by logic
                focus_bounding_box=box_tuple,
                rationale=data.get("rationale", text[:200]),
            )
        except Exception as e:
            logger.error("Failed to parse VLM output: %s\nRaw output: %s", e, text)
            return SemanticAnchorDecision(
                scene_classification="VLM Parse Error",
                confidence=0.0,
                is_target_valid=False,
                focal_directives=(),
                micro_directives=(),
                rationale=f"Parse Error: {e}",
            )
