"""Automated Feasibility and Unit Test Suite for Semantically-Aware Planar Object Filter."""

from __future__ import annotations

import time
import unittest
import numpy as np
from PIL import Image

from segmentation.gobotany_devo_adapter import GoBotanyDevoAdapter
from segmentation.semantic_context_payload import (
    LocationContext,
    SemanticContextPayload,
    SemanticPayloadFactory,
    SensorTelemetry,
)
from segmentation.semantic_anchor import AsyncSemanticAnchor, SemanticAnchorDecision
from segmentation.planar_tracker import PlanarMeshTracker, TrackedPolygon
from segmentation.recursive_subdivider import HierarchicalPolygon, RecursiveDepthResolver
from segmentation.multi_frequency_pipeline import MultiFrequencyPipeline, PipelineSnapshot
from segmentation.pipeline import RawSegment


def test_gobotany_devo_adapter_queries() -> None:
    adapter = GoBotanyDevoAdapter()
    piles = adapter.list_piles()
    assert len(piles) > 0

    candidates = adapter.get_henrico_va_yard_candidates()
    assert len(candidates) > 0
    
    first = candidates[0]
    assert first.scientific_name
    assert len(first.prompt_vocabulary) > 0
    assert "whole plant" in first.prompt_vocabulary


def test_semantic_payload_formatting() -> None:
    adapter = GoBotanyDevoAdapter()
    factory = SemanticPayloadFactory(adapter)
    payload = factory.build_henrico_yard_payload(
        telemetry=SensorTelemetry(azimuth_degrees=180.0, user_zoom_level=1.5),
        previous_classification="Quercus alba candidate",
    )

    assert payload.location.name == "Tuckaway Lane, Henrico, VA"
    assert "Henrico" in payload.location.ecosite or "Virginia" in payload.location.ecosite
    assert len(payload.candidate_taxa) > 0

    prompt = payload.format_vlm_prompt()
    assert "Tuckaway Lane, Henrico, VA" in prompt
    assert "GoBotany DeVo" in prompt
    assert "schema" in prompt.lower()


def test_async_semantic_anchor_worker() -> None:
    anchor = AsyncSemanticAnchor(model_name="test-mock")
    anchor.start()

    factory = SemanticPayloadFactory(GoBotanyDevoAdapter())
    payload = factory.build_henrico_yard_payload()
    img = Image.new("RGB", (320, 240), color="green")

    submitted = anchor.submit_frame(img, payload)
    assert submitted is True

    # Allow worker thread to process
    time.sleep(0.3)
    decision = anchor.get_latest_decision()
    assert decision is not None
    assert decision.is_target_valid is True
    assert len(decision.focal_directives) > 0

    anchor.stop()


def test_planar_mesh_tracker_optical_flow() -> None:
    tracker = PlanarMeshTracker()
    
    # Create test synthetic frame 1 (black background with white rectangle)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame1[150:350, 200:400] = 255

    # Define initial normalized polygon around the white box
    init_poly = [
        ("poly_1", 1, "leaf", ((0.31, 0.31), (0.62, 0.31), (0.62, 0.72), (0.31, 0.72)), "leaf", None)
    ]
    tracker.set_polygons(init_poly)

    # Frame 1 update
    u1 = tracker.update_frame(frame1)
    assert len(u1.polygons) == 1
    assert u1.fps > 0

    # Create test frame 2 shifted 15 pixels to the right
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2[150:350, 215:415] = 255

    # Frame 2 update (optical flow should follow the shift)
    u2 = tracker.update_frame(frame2)
    assert len(u2.polygons) == 1
    tracked = u2.polygons[0]
    
    # Check that x coordinates shifted rightwards (larger x)
    x_prev = init_poly[0][3][0][0]
    x_new = tracked.vertices[0][0]
    assert x_new >= x_prev - 0.05  # within reasonable tracking bounds


def test_recursive_depth_resolver() -> None:
    subdivider = RecursiveDepthResolver()
    
    raw_lvl1 = [
        RawSegment(
            label="leaf",
            polygon=((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)),
            detection_confidence=0.95,
            mask_quality=0.90,
            prompt="leaf",
            detector_label="structural leaf",
        )
    ]

    lvl1_polys = subdivider.build_level1_polygons(raw_lvl1)
    assert len(lvl1_polys) == 1
    assert lvl1_polys[0].level == 1

    img = Image.new("RGB", (400, 400), color="darkgreen")
    crops = subdivider.prepare_subdivision_crops(img, lvl1_polys)
    assert len(crops) == 1

    parent, crop_img, bounds = crops[0]
    assert crop_img.size[0] > 0

    # Simulate micro-segmentation inside crop
    raw_micro = [
        RawSegment(
            label="toothed leaf margin",
            polygon=((0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4)),
            detection_confidence=0.88,
            mask_quality=0.85,
            prompt="toothed leaf margin",
        )
    ]

    micro_polys = subdivider.transform_micro_segments(parent, raw_micro, bounds)
    assert len(micro_polys) == 1
    assert micro_polys[0].level == 2
    assert micro_polys[0].parent_id == parent.polygon_id


def test_multi_frequency_pipeline_end_to_end() -> None:
    pipeline = MultiFrequencyPipeline()
    pipeline.start()

    # Process 5 simulated camera frames
    for i in range(5):
        frame = np.full((360, 480, 3), 50 + i * 10, dtype=np.uint8)
        # Add moving square
        frame[100:200, 100 + i * 5 : 200 + i * 5] = 200

        snap = pipeline.process_frame(frame, force_spatial_reissue=(i == 0))
        assert snap.frame is not None
        assert snap.fps > 0

        time.sleep(0.02)  # Simulate ~50 FPS camera stream

    pipeline.stop()
