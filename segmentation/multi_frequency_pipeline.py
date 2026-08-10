"""Master Multi-Frequency Planar Filter Pipeline Coordinator.

Decouples real-time optical flow tracking (30–60 FPS), spatial feature issuance
(5–10 FPS Grounding DINO + SAM 2.1), and low-frequency contextual VLM anchoring
(0.5–1 FPS Gemma Agent) into an asynchronous lock-free engine on Apple Silicon.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence
import cv2
import numpy as np
from PIL import Image

from .gobotany_devo_adapter import GoBotanyDevoAdapter
from .pipeline import CaptureFrame, RawSegment
from .planar_tracker import PlanarMeshTracker, TrackedPolygon, TrackingUpdate
from .recursive_subdivider import HierarchicalPolygon, RecursiveDepthResolver
from .semantic_anchor import AsyncSemanticAnchor, SemanticAnchorDecision
from .semantic_context_payload import SemanticContextPayload, SemanticPayloadFactory, SensorTelemetry

logger = logging.getLogger(__name__)


@dataclass
class PipelineSnapshot:
    """Immutable real-time frame packet for HUD and UI rendering."""
    frame: np.ndarray
    polygons: list[TrackedPolygon]
    fps: float
    tracking_latency_ms: float
    spatial_latency_ms: float
    semantic_decision: SemanticAnchorDecision | None
    current_payload: SemanticContextPayload | None
    tracking_drift: float = 0.0
    level1_count: int = 0
    level2_count: int = 0


class MultiFrequencyPipeline:
    """Coordinates high-frequency tracking, mid-frequency SAM, and low-frequency Gemma."""

    def __init__(
        self,
        *,
        spatial_segmenter: Any | None = None,  # GroundedSam2Segmenter or Mock
        semantic_anchor: AsyncSemanticAnchor | None = None,
        tracker: PlanarMeshTracker | None = None,
        devo_adapter: GoBotanyDevoAdapter | None = None,
        knowledge_graph: Any | None = None,
        enable_recursive_subdivision: bool = True,
    ) -> None:
        self.devo = devo_adapter or GoBotanyDevoAdapter()
        self.payload_factory = SemanticPayloadFactory(self.devo)
        self.knowledge_graph = knowledge_graph
        self.spatial_segmenter = spatial_segmenter
        self.semantic_anchor = semantic_anchor or AsyncSemanticAnchor()
        self.tracker = tracker or PlanarMeshTracker()
        self.subdivider = RecursiveDepthResolver()
        self.enable_recursive_subdivision = enable_recursive_subdivision

        self._running = False
        self._spatial_queue: queue.Queue[tuple[np.ndarray, list[str], list[str]]] = queue.Queue(maxsize=1)
        self._spatial_thread: threading.Thread | None = None
        self._latest_spatial_latency_ms: float = 0.0
        
        self._last_semantic_submit: float = 0.0
        self._semantic_interval_sec: float = 1.0  # Run Gemma every 1 sec
        self._latest_payload: SemanticContextPayload | None = None
        self._lock = threading.Lock()

        # Connect semantic anchor callback
        self.semantic_anchor.register_callback(self._on_semantic_decision)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.semantic_anchor.start()
        self._spatial_thread = threading.Thread(
            target=self._spatial_worker_loop, daemon=True, name="SpatialFeatureIssuerWorker"
        )
        self._spatial_thread.start()

    def stop(self) -> None:
        self._running = False
        self.semantic_anchor.stop()
        if self._spatial_thread and self._spatial_thread.is_alive():
            self._spatial_thread.join(timeout=1.0)

    def _on_semantic_decision(self, decision: SemanticAnchorDecision) -> None:
        """Called when Gemma emits a new scene classification and focal directive."""
        if not decision.is_target_valid:
            logger.info(
                "Semantic Anchor updated: Target Not Present (is_target_valid=False)"
            )
        else:
            logger.info(
                "Semantic Anchor updated: %s (directives: %s)",
                decision.scene_classification,
                decision.focal_directives,
            )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        telemetry: SensorTelemetry | None = None,
        force_spatial_reissue: bool = False,
    ) -> PipelineSnapshot:
        """High-frequency camera frame processor running at 30-60 FPS."""
        # 1. Optical Flow Temporal Tracking Step (<10ms)
        tracking_update = self.tracker.update_frame(frame_bgr)

        # 2. Check if Spatial Feature Issuer needs a new detection pass
        if tracking_update.requires_reissue or force_spatial_reissue:
            self._trigger_spatial_reissue(frame_bgr)

        # 3. Check if Semantic Anchor (Gemma) should receive a background snapshot (~1 FPS)
        now = time.time()
        if now - self._last_semantic_submit > self._semantic_interval_sec:
            self._last_semantic_submit = now
            self._trigger_semantic_anchor(frame_bgr, telemetry)

        # 4. Assemble real-time snapshot
        decision = self.semantic_anchor.get_latest_decision()
        polys = self.tracker.current_polygons

        lvl1 = sum(1 for p in polys if p.level == 1)
        lvl2 = sum(1 for p in polys if p.level == 2)

        return PipelineSnapshot(
            frame=frame_bgr,
            polygons=polys,
            fps=tracking_update.fps,
            tracking_latency_ms=tracking_update.latency_ms,
            spatial_latency_ms=self._latest_spatial_latency_ms,
            semantic_decision=decision,
            current_payload=self._latest_payload,
            tracking_drift=tracking_update.motion_magnitude,
            level1_count=lvl1,
            level2_count=lvl2,
        )

    def _trigger_spatial_reissue(self, frame_bgr: np.ndarray) -> None:
        """Non-blocking dispatch of frame to Grounding DINO + SAM thread."""
        decision = self.semantic_anchor.get_latest_decision()
        
        # Lazy Gate: Gemma determined target is missing, or is still initializing. Save compute and clear meshes.
        if not decision or not decision.is_target_valid:
            self.tracker.set_polygons([])
            return
            
        macro_prompts = list(decision.focal_directives) if decision and decision.focal_directives else ["leaf", "bark", "branch"]
        micro_prompts = list(decision.micro_directives) if decision and decision.micro_directives else ["leaf lobe", "margin"]

        try:
            self._spatial_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._spatial_queue.put_nowait((frame_bgr.copy(), macro_prompts, micro_prompts))
        except queue.Full:
            pass

    def _trigger_semantic_anchor(self, frame_bgr: np.ndarray, telemetry: SensorTelemetry | None) -> None:
        """Non-blocking dispatch to async Gemma VLM thread."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if frame_bgr.ndim == 3 else frame_bgr
        pil_img = Image.fromarray(rgb)

        decision = self.semantic_anchor.get_latest_decision()
        prev_class = decision.scene_classification if decision else None

        if self.knowledge_graph and getattr(self.knowledge_graph, "category", "") == "handheld_input_device":
            payload = self.payload_factory.build_xbox_controller_payload(
                telemetry=telemetry,
                previous_classification=prev_class,
            )
        else:
            payload = self.payload_factory.build_henrico_yard_payload(
                telemetry=telemetry,
                previous_classification=prev_class,
            )
            if self.knowledge_graph:
                payload = replace(payload, knowledge_graph=self.knowledge_graph) if hasattr(payload, "__dataclass_fields__") else payload

        self._latest_payload = payload
        self.semantic_anchor.submit_frame(pil_img, payload)

    def _spatial_worker_loop(self) -> None:
        """Mid-frequency worker loop executing Grounding DINO + SAM 2.1."""
        while self._running:
            try:
                frame_bgr, macro_prompts, micro_prompts = self._spatial_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            start_t = time.perf_counter()
            try:
                raw_segments = self._run_spatial_inference(frame_bgr, macro_prompts)
                
                # Transform to Hierarchical Polygons
                lvl1_polys = self.subdivider.build_level1_polygons(raw_segments)
                all_polys: list[tuple[str, int, str, Sequence[tuple[float, float]], str | None, str | None]] = []

                for p in lvl1_polys:
                    all_polys.append((p.polygon_id, p.level, p.label, p.polygon, p.detector_label, p.parent_id))

                # If recursive subdivision enabled, generate Level-2 micro-features
                if self.enable_recursive_subdivision and lvl1_polys and micro_prompts:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    crops = self.subdivider.prepare_subdivision_crops(pil_img, lvl1_polys)
                    
                    for parent_poly, crop_img, bounds in crops[:2]:  # Subdivide top 2 regions
                        micro_segments = self._run_crop_inference(crop_img, micro_prompts)
                        micro_hier = self.subdivider.transform_micro_segments(parent_poly, micro_segments, bounds)
                        for mp in micro_hier:
                            all_polys.append((mp.polygon_id, mp.level, mp.label, mp.polygon, mp.detector_label, mp.parent_id))

                # Update tracker with newly issued mesh
                self.tracker.set_polygons(all_polys)

            except Exception as e:
                logger.error("Spatial worker exception: %s", e)

            self._latest_spatial_latency_ms = (time.perf_counter() - start_t) * 1000.0
            self._spatial_queue.task_done()

    def _run_spatial_inference(self, frame_bgr: np.ndarray, prompts: list[str]) -> tuple[RawSegment, ...]:
        """Execute GroundedSam2Segmenter or fallback mock segmenter."""
        if self.spatial_segmenter is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # Create in-memory capture frame
            c_frame = CaptureFrame(
                observation_id="live_stream",
                image_url="",
                subject_id="webcam",
                local_path=None,
            )
            return self.spatial_segmenter.segment_image(pil_img, prompts)

        # Deterministic synthetic segmenter fallback for fast simulation
        h, w = frame_bgr.shape[:2]
        # Return synthetic structural polygon in center
        poly_center = (
            (0.3, 0.2), (0.7, 0.2), (0.8, 0.7), (0.6, 0.85), (0.4, 0.85), (0.2, 0.7)
        )
        return (
            RawSegment(
                label=prompts[0] if prompts else "leaf",
                polygon=poly_center,
                detection_confidence=0.92,
                mask_quality=0.88,
                stability_score=0.9,
                prompt=prompts[0] if prompts else "leaf",
                detector_label="structural organ",
            ),
        )

    def _run_crop_inference(self, crop_img: Image.Image, prompts: list[str]) -> tuple[RawSegment, ...]:
        """Run localized micro-feature segmentation on ROI crop."""
        if self.spatial_segmenter is not None and hasattr(self.spatial_segmenter, "segment_image"):
            return self.spatial_segmenter.segment_image(crop_img, prompts)

        # Micro-feature synthetic segment in crop space
        poly_micro = ((0.2, 0.2), (0.5, 0.15), (0.6, 0.5), (0.3, 0.6))
        return (
            RawSegment(
                label=prompts[0] if prompts else "micro-feature",
                polygon=poly_micro,
                detection_confidence=0.85,
                mask_quality=0.80,
                prompt=prompts[0] if prompts else "micro-feature",
                detector_label="micro-feature",
            ),
        )
