#!/usr/bin/env python3
"""Interactive Live Demo for Semantically-Aware Planar Object Filter with Knowledge Graphs.

Demonstrates hierarchical anatomical object decomposition and real-time optical flow
tracking on Apple Silicon (MacBook M5 Pro). Supports live webcam, phone video, static photos,
or animated rotation of physical objects (e.g. Xbox Controller or Botanical Flora).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure workspace root is on PYTHONPATH
workspace_dir = Path(__file__).resolve().parent.parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from segmentation.gobotany_devo_adapter import GoBotanyDevoAdapter
from segmentation.knowledge_graph_schema import (
    AnatomicalKnowledgeGraph,
    get_botanical_plant_knowledge_graph,
    get_xbox_controller_knowledge_graph,
)
from segmentation.multi_frequency_pipeline import MultiFrequencyPipeline, PipelineSnapshot
from segmentation.planar_tracker import PlanarMeshTracker
from segmentation.semantic_anchor import AsyncSemanticAnchor, SemanticAnchorDecision
from segmentation.semantic_context_payload import (
    LocationContext,
    SemanticContextPayload,
    SemanticPayloadFactory,
    SensorTelemetry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveWebcamFilterDemo")


def draw_hud(
    frame: np.ndarray,
    snapshot: PipelineSnapshot,
    location_name: str,
    knowledge_graph: AnatomicalKnowledgeGraph | None = None,
    show_side_panel: bool = True,
    zoom_level: float = 1.0,
) -> np.ndarray:
    """Render real-time polygon meshes, telemetry HUD, and Knowledge Graph decomposition panel."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Apply synthetic digital zoom if zoom_level > 1.0
    if zoom_level > 1.0:
        cw, ch = int(w / zoom_level), int(h / zoom_level)
        cx, cy = w // 2, h // 2
        crop = out[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2]
        out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. Render Tracked Polygons
    # Level 1 (Macro Chassis / Structural Silhouette): Cyan (255, 255, 0 in BGR)
    # Level 2 (Micro-Feature / Buttons / Thumbsticks / Lobes): Lime Green (0, 255, 128 in BGR)
    for poly in snapshot.polygons:
        pts_px = poly.to_pixel_coords(w, h)
        if len(pts_px) < 3:
            continue

        if poly.level == 1:
            color = (255, 255, 0)  # Cyan
            thickness = 2
        else:
            color = (0, 255, 128)  # Lime Green
            thickness = 2

        # Draw semi-transparent filled polygon
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts_px], color)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        # Draw solid contour boundary
        cv2.polylines(out, [pts_px], isClosed=True, color=color, thickness=thickness)

        # Label tag near centroid
        cx, cy = np.mean(pts_px, axis=0).astype(int)
        
        if poly.level == 1 and snapshot.semantic_decision and snapshot.semantic_decision.focal_directives:
            lbl = f"L1: {snapshot.semantic_decision.focal_directives[0]}"
        else:
            lbl = f"L{poly.level}: {poly.label}"
        cv2.putText(out, lbl, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, lbl, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 2. Top Telemetry Banner
    banner_h = 60
    banner_bg = out[:banner_h, :].copy()
    cv2.rectangle(banner_bg, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(banner_bg, 0.75, out[:banner_h, :], 0.25, 0, out[:banner_h, :])

    # Telemetry Text
    fps_str = f"FPS: {snapshot.fps:4.1f}"
    track_str = f"Flow Tracker: {snapshot.tracking_latency_ms:4.1f}ms"
    spatial_str = f"SAM / Spatial: {snapshot.spatial_latency_ms:4.1f}ms"
    
    dec = snapshot.semantic_decision
    gemma_status = dec.scene_classification if dec else "Gemma: Initializing..."
    gemma_lat = f"Gemma Latency: {dec.latency_ms:4.0f}ms" if dec else "Async VLM"

    cv2.putText(out, "SEMANTICALLY-AWARE PLANAR FILTER (M5 PRO MPS)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(out, f"{fps_str}  |  {track_str}  |  {spatial_str}  |  {gemma_lat}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 3. Side Panel (Knowledge Graph Anatomy & Context)
    if show_side_panel and w >= 640:
        panel_w = 300
        panel_x = w - panel_w
        panel_bg = out[banner_h:, panel_x:].copy()
        cv2.rectangle(panel_bg, (0, 0), (panel_w, h - banner_h), (15, 25, 20), -1)
        cv2.addWeighted(panel_bg, 0.80, out[banner_h:, panel_x:], 0.20, 0, out[banner_h:, panel_x:])

        px = panel_x + 12
        py = banner_h + 24
        cv2.putText(out, "ANATOMICAL KNOWLEDGE GRAPH", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
        py += 20
        entity_title = knowledge_graph.entity_name if knowledge_graph else location_name
        cv2.putText(out, entity_title[:32], (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
        py += 18
        category_str = f"Category: {knowledge_graph.category}" if knowledge_graph else "Piedmont Hardwood Yard"
        cv2.putText(out, category_str[:32], (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)
        
        py += 26
        cv2.putText(out, "SCENE CLASSIFICATION", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
        py += 20
        cv2.putText(out, gemma_status[:32], (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        py += 28
        cv2.putText(out, "GRAPH SLOTS & DIRECTIVES", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
        py += 18
        
        if knowledge_graph:
            for s in knowledge_graph.slots[:6]:
                lvl_tag = "L1" if s.level == 1 else "L2"
                cv2.putText(out, f"[{lvl_tag}] {s.label}", (px + 4, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 128) if s.level == 2 else (255, 255, 0), 1)
                py += 16
        else:
            directives = dec.focal_directives if dec and dec.focal_directives else ["leaf", "bark", "branch"]
            for d in directives[:5]:
                cv2.putText(out, f"• {d}", (px + 6, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 128), 1)
                py += 16

        py += 18
        cv2.putText(out, f"Meshes: L1={snapshot.level1_count}  L2={snapshot.level2_count}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
        py += 24
        cv2.putText(out, "CONTROLS: [Q]uit  [G]emma  [Z]oom  [S]ave", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1)

    return out


def open_camera(requested_index: int) -> tuple[cv2.VideoCapture | None, int | None]:
    """Auto-probe camera devices on macOS using AVFoundation and standard backends."""
    probe_indices = [requested_index]
    for idx in [0, 1, 2, 3]:
        if idx not in probe_indices:
            probe_indices.append(idx)

    for idx in probe_indices:
        logger.info("Probing camera device #%d with CAP_AVFOUNDATION...", idx)
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                logger.info("Successfully opened camera #%d (resolution: %dx%d)", idx, frame.shape[1], frame.shape[0])
                return cap, idx
            cap.release()

        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                logger.info("Successfully opened camera #%d with default backend (resolution: %dx%d)", idx, frame.shape[1], frame.shape[0])
                return cap, idx
            cap.release()

    return None, None


def render_synthetic_controller_frame(t: float, angle_deg: float) -> np.ndarray:
    """Render a synthetic handheld Xbox Controller that rotates in 2D/3D space."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (25, 25, 28)  # Dark desktop background

    # Simulate empty scene for the first 5 seconds to test Gemma's lazy gating
    if t < 5.0:
        return frame

    center = (320 + int(40 * math.sin(t * 0.7)), 240 + int(20 * math.cos(t * 0.5)))
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def rot(x: float, y: float) -> tuple[int, int]:
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        return (int(center[0] + rx), int(center[1] + ry))

    # Level 1: Chassis (outer dual-grip body)
    chassis_pts = [
        rot(-140, -60), rot(140, -60), rot(160, 40), rot(120, 90),
        rot(70, 70), rot(0, 50), rot(-70, 70), rot(-120, 90), rot(-160, 40)
    ]
    cv2.fillPoly(frame, [np.array(chassis_pts, dtype=np.int32)], (60, 60, 65))
    cv2.polylines(frame, [np.array(chassis_pts, dtype=np.int32)], True, (90, 90, 95), 2)

    # Level 2: Left Thumbstick (upper left)
    lt_center = rot(-75, -20)
    cv2.circle(frame, lt_center, 22, (30, 30, 35), -1)
    cv2.circle(frame, lt_center, 16, (80, 80, 85), -1)
    cv2.circle(frame, lt_center, 8, (45, 45, 50), -1)

    # Level 2: Right Thumbstick (lower center-right)
    rt_center = rot(35, 15)
    cv2.circle(frame, rt_center, 22, (30, 30, 35), -1)
    cv2.circle(frame, rt_center, 16, (80, 80, 85), -1)
    cv2.circle(frame, rt_center, 8, (45, 45, 50), -1)

    # Level 2: D-Pad (lower left)
    dp_center = rot(-40, 20)
    cv2.circle(frame, dp_center, 18, (35, 35, 40), -1)
    cv2.rectangle(frame, (dp_center[0] - 12, dp_center[1] - 4), (dp_center[0] + 12, dp_center[1] + 4), (90, 90, 95), -1)
    cv2.rectangle(frame, (dp_center[0] - 4, dp_center[1] - 12), (dp_center[0] + 4, dp_center[1] + 12), (90, 90, 95), -1)

    # Level 2: ABXY Buttons (upper right)
    abxy_center = rot(75, -20)
    # A (green, bottom), B (red, right), X (blue, left), Y (yellow, top)
    cv2.circle(frame, rot(75, -8), 6, (0, 180, 0), -1)    # A Button
    cv2.circle(frame, rot(87, -20), 6, (0, 0, 200), -1)   # B Button
    cv2.circle(frame, rot(63, -20), 6, (200, 0, 0), -1)   # X Button
    cv2.circle(frame, rot(75, -32), 6, (0, 200, 200), -1) # Y Button

    return frame


def run_live_demo(
    webcam_index: int = 0,
    video_path: str | None = None,
    image_path: str | None = None,
    mode: str = "controller",  # controller | plant | custom
    kg_path: str | None = None,
    location_name: str = "Tuckaway Ln, Henrico, VA",
    use_synthetic: bool = False,
    max_frames: int | None = None,
    mock_vlm: bool = False,
) -> None:
    logger.info("Setting up Knowledge Graph for mode: %s...", mode)
    
    if kg_path and Path(kg_path).exists():
        raw_kg = json.loads(Path(kg_path).read_text())
        knowledge_graph = AnatomicalKnowledgeGraph.from_dict(raw_kg)
    elif mode == "controller":
        knowledge_graph = get_xbox_controller_knowledge_graph()
    else:
        knowledge_graph = get_botanical_plant_knowledge_graph("Quercus alba", "White Oak")

    devo = GoBotanyDevoAdapter()
    
    spatial_segmenter = None
    try:
        from segmentation.grounded_sam2 import GroundedSam2Segmenter
        logger.info("Initializing Grounded SAM 2.1 on MPS...")
        spatial_segmenter = GroundedSam2Segmenter(box_threshold=0.7)
    except Exception as e:
        logger.info("Running spatial issuer with built-in high-performance simulator (%s)", e)

    custom_evaluator = None
    if not mock_vlm:
        try:
            from segmentation.local_vlm_evaluator import LocalVlmEvaluator
            custom_evaluator = LocalVlmEvaluator(model_id="google/gemma-4-12b-it")
        except Exception as e:
            logger.error("Failed to load local VLM: %s", e)

    pipeline = MultiFrequencyPipeline(
        spatial_segmenter=spatial_segmenter,
        devo_adapter=devo,
        knowledge_graph=knowledge_graph,
        enable_recursive_subdivision=True,
    )
    if custom_evaluator:
        pipeline.semantic_anchor.custom_evaluator = custom_evaluator
        pipeline.semantic_anchor.model_name = "gemma-4-12b-it"

    pipeline.start()

    cap: cv2.VideoCapture | None = None
    static_img: np.ndarray | None = None

    if image_path and Path(image_path).exists():
        logger.info("Loading input image: %s", image_path)
        static_img = cv2.imread(image_path)
        if static_img is None:
            logger.error("Failed to load image at %s", image_path)

    elif video_path and Path(video_path).exists():
        logger.info("Opening video source: %s", video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Could not open video file %s", video_path)
            cap = None

    elif not use_synthetic:
        cap, active_idx = open_camera(webcam_index)
        if cap is None:
            logger.warning(
                "Could not open camera device #%d.\n"
                "  -> macOS Camera Permission Note: Ensure your Terminal / IDE is granted Camera access in:\n"
                "     System Settings -> Privacy & Security -> Camera\n"
                "  -> Falling back to rotating synthetic demo."
            )

    zoom_level = 1.0
    show_hud_panel = True
    frame_count = 0
    start_time = time.time()

    logger.info("Starting live processing loop. Controls: [Q]uit  [G]emma  [Z]oom  [H]UD  [S]ave")

    try:
        while True:
            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            t = time.time() - start_time

            if static_img is not None:
                ih, iw = static_img.shape[:2]
                cw, ch = int(iw * 0.8), int(ih * 0.8)
                max_dx = (iw - cw) // 2
                max_dy = (ih - ch) // 2
                dx = int(max_dx * math.sin(t * 0.5))
                dy = int(max_dy * math.cos(t * 0.3))
                cx = iw // 2 + dx
                cy = ih // 2 + dy
                crop = static_img[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2]
                frame = cv2.resize(crop, (640, 480), interpolation=cv2.INTER_LINEAR)

            elif cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    if video_path:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                if frame.shape[1] > 1280:
                    scale = 1280 / frame.shape[1]
                    frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)))

            else:
                # Synthetic stream: if controller mode, render rotating controller
                if mode == "controller":
                    angle = 35.0 * math.sin(t * 1.2)  # Rotating back and forth by +/- 35 degrees
                    frame = render_synthetic_controller_frame(t, angle)
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    frame[:, :] = (35, 60, 30)
                    cx = int(320 + 80 * math.sin(t * 0.8))
                    cy = int(240 + 30 * math.cos(t * 0.5))
                    cv2.rectangle(frame, (cx - 30, cy), (cx + 30, 480), (45, 65, 85), -1)
                    cv2.circle(frame, (cx, cy - 60), 110, (40, 140, 60), -1)
                    cv2.circle(frame, (cx - 60, cy - 40), 70, (45, 160, 70), -1)
                    cv2.circle(frame, (cx + 60, cy - 40), 75, (35, 130, 55), -1)

            telemetry = SensorTelemetry(
                azimuth_degrees=float((frame_count * 2) % 360),
                user_zoom_level=zoom_level,
            )
            snapshot = pipeline.process_frame(frame, telemetry=telemetry)

            rendered = draw_hud(
                frame,
                snapshot,
                location_name=location_name,
                knowledge_graph=knowledge_graph,
                show_side_panel=show_hud_panel,
                zoom_level=zoom_level,
            )

            cv2.imshow("Semantically-Aware Planar Filter Demo", rendered)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break
            elif key == ord("g"):
                logger.info("Forcing Gemma semantic re-anchor...")
                pipeline._trigger_semantic_anchor(frame, telemetry)
            elif key == ord("z"):
                zoom_level = 1.8 if zoom_level == 1.0 else 1.0
                logger.info("Toggled zoom level to %1.1fx", zoom_level)
            elif key == ord("h"):
                show_hud_panel = not show_hud_panel
            elif key == ord("s"):
                save_path = workspace_dir / f"capture_{mode}_{int(time.time())}.png"
                cv2.imwrite(str(save_path), rendered)
                logger.info("Saved capture to %s", save_path)

    finally:
        pipeline.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Live demo stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantically-Aware Planar Filter Live Demo")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--mode", type=str, default="controller", choices=["controller", "plant", "custom"], help="Anatomical knowledge graph target mode")
    parser.add_argument("--demo-controller", action="store_true", help="Convenience alias to run in Xbox controller mode")
    parser.add_argument("--demo-plant", action="store_true", help="Convenience alias to run in botanical plant mode")
    parser.add_argument("--kg", type=str, default=None, help="Path to custom knowledge graph JSON file")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (.mp4, .mov)")
    parser.add_argument("--image", type=str, default=None, help="Path to static image file (.jpg, .png)")
    parser.add_argument("--location", type=str, default="Tuckaway Ln, Henrico, VA", help="Field location name")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic test stream")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for benchmarking)")
    parser.add_argument("--mock-vlm", action="store_true", help="Use mock VLM evaluator for fast UI debugging instead of loading Local VLM")
    args = parser.parse_args()

    mode = "plant" if args.demo_plant else ("controller" if args.demo_controller else args.mode)

    run_live_demo(
        webcam_index=args.webcam,
        mode=mode,
        kg_path=args.kg,
        video_path=args.video,
        image_path=args.image,
        location_name=args.location,
        use_synthetic=args.synthetic,
        max_frames=args.max_frames,
        mock_vlm=args.mock_vlm,
    )


if __name__ == "__main__":
    main()
