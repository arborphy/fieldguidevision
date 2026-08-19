"""Asynchronous Semantic Anchoring Agent for Field Guide Vision.

Maintains scene-stable contextual classification in the background (0.5–1 FPS)
and outputs focal Grounding DINO detection prompts and search bounding boxes.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from PIL import Image

from .semantic_context_payload import SemanticContextPayload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticAnchorDecision:
    scene_classification: str
    confidence: float
    is_target_valid: bool
    focal_directives: tuple[str, ...]
    micro_directives: tuple[str, ...] = ()
    focus_bounding_box: tuple[float, float, float, float] | None = None  # (ymin, xmin, ymax, xmax)
    rationale: str = ""
    latency_ms: float = 0.0
    model_name: str = "gemma-anchor"
    timestamp: float = 0.0


class AsyncSemanticAnchor:
    """Asynchronous worker that evaluates scene snapshots with ecological priors."""

    def __init__(
        self,
        *,
        model_name: str = "mock-gemma-4",
        api_key: str | None = None,
        custom_evaluator: Callable[[Image.Image, SemanticContextPayload], SemanticAnchorDecision] | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.custom_evaluator = custom_evaluator
        
        self._request_queue: queue.Queue[tuple[Image.Image, SemanticContextPayload, float]] = queue.Queue(maxsize=2)
        self._latest_decision: SemanticAnchorDecision | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._decision_callbacks: list[Callable[[SemanticAnchorDecision], None]] = []
        self._start_time: float = time.time()

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="SemanticAnchorWorker")
        self._thread.start()

    def stop(self) -> None:
        """Stop the background worker thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def register_callback(self, callback: Callable[[SemanticAnchorDecision], None]) -> None:
        self._decision_callbacks.append(callback)

    def submit_frame(self, image: Image.Image, payload: SemanticContextPayload) -> bool:
        """Non-blocking submit of the latest snapshot to the async anchor queue."""
        if not self._running:
            return False
        
        # If queue is full, drop the stale frame and put the freshest one
        try:
            self._request_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._request_queue.put_nowait((image.copy(), payload, time.time()))
            return True
        except queue.Full:
            return False

    def get_latest_decision(self) -> SemanticAnchorDecision | None:
        """Get the most recent decision without waiting."""
        with self._lock:
            return self._latest_decision

    def _worker_loop(self) -> None:
        while self._running:
            try:
                item = self._request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            image, payload, submit_time = item
            start_t = time.perf_counter()

            decision = self._evaluate_snapshot(image, payload)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            
            decision = SemanticAnchorDecision(
                scene_classification=decision.scene_classification,
                confidence=decision.confidence,
                is_target_valid=decision.is_target_valid,
                focal_directives=decision.focal_directives,
                micro_directives=decision.micro_directives,
                focus_bounding_box=decision.focus_bounding_box,
                rationale=decision.rationale,
                latency_ms=elapsed_ms,
                model_name=self.model_name,
                timestamp=time.time(),
            )

            with self._lock:
                self._latest_decision = decision

            for cb in self._decision_callbacks:
                try:
                    cb(decision)
                except Exception as e:
                    logger.warning("Error in semantic anchor callback: %s", e)

            self._request_queue.task_done()

    def _evaluate_snapshot(self, image: Image.Image, payload: SemanticContextPayload) -> SemanticAnchorDecision:
        """Execute evaluation using custom evaluator or GoBotany context prior logic."""
        if self.custom_evaluator:
            return self.custom_evaluator(image, payload)

        # If Knowledge Graph is attached, extract slots & directives from graph hierarchy
        if payload.knowledge_graph:
            kg = payload.knowledge_graph
            
            # Simulate a 5-second "empty scene" gate before Gemma successfully recognizes the object
            is_valid = (time.time() - self._start_time) >= 5.0
            
            if not is_valid:
                return SemanticAnchorDecision(
                    scene_classification="Empty scene or target not found",
                    confidence=0.90,
                    is_target_valid=False,
                    focal_directives=(),
                    micro_directives=(),
                    focus_bounding_box=None,
                    rationale="I do not see the target object in this scan.",
                )
                
            rec_prompt = f"the {kg.entity_name.lower()} in the foreground"
            
            # Level 2 child slots
            child_slots = kg.get_level2_micro_slots()
            child_prompts = []
            for s in child_slots:
                if s.prompts:
                    child_prompts.extend(s.prompts[:2])
                else:
                    child_prompts.append(s.label)

            return SemanticAnchorDecision(
                scene_classification=kg.entity_name,
                confidence=0.94,
                is_target_valid=True,
                focal_directives=(rec_prompt,),
                micro_directives=tuple(child_prompts),
                focus_bounding_box=(0.1, 0.1, 0.9, 0.9),
                rationale=f"Evaluated against Anatomical Knowledge Graph '{kg.entity_name}' using REC prompt '{rec_prompt}'.",
            )

        # Default fast rule/prior based evaluator: selects most relevant GoBotany candidate
        candidates = payload.candidate_taxa
        if not candidates:
            return SemanticAnchorDecision(
                scene_classification="Unknown target",
                confidence=0.5,
                is_target_valid=True,
                focal_directives=("object", "part"),
                micro_directives=(),
                focus_bounding_box=(0.1, 0.1, 0.9, 0.9),
                rationale="No knowledge graph or candidate taxa in context payload.",
            )

        # Pick top candidate or prior match
        top = candidates[0]
        prompts = top.prompts if top.prompts else ("leaf", "bark", "branch", "whole plant")
        macro_prompts = tuple(prompts[:2])
        micro_prompts = tuple(prompts[2:]) if len(prompts) > 2 else ("leaf lobe", "margin")
        
        return SemanticAnchorDecision(
            scene_classification=f"{top.scientific_name} ({top.common_name or 'N/A'})",
            confidence=0.88,
            is_target_valid=True,
            focal_directives=macro_prompts,
            micro_directives=micro_prompts,
            focus_bounding_box=(0.05, 0.05, 0.95, 0.95),
            rationale=f"Evaluated against Henrico, VA late-summer GoBotany priors; active prompts: {', '.join(prompts[:4])}",
        )
