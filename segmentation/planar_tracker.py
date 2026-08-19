"""High-Frequency Temporal Planar Mesh Tracker for Field Guide Vision.

Propagates polygon mesh vertices frame-to-frame across high-frequency camera
frames (30–60 FPS / ~16ms budget) using lightweight Lucas-Kanade optical flow,
decoupling visual tracking from deep neural inference.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence
import cv2
import numpy as np


@dataclass
class TrackedPolygon:
    polygon_id: str
    level: int  # 1: Structural Mesh, 2: Recursive Micro-Feature
    label: str
    vertices: np.ndarray  # Shape (N, 2) in normalized [0, 1] coordinates (x, y)
    confidence: float = 1.0
    age_frames: int = 0
    drift_score: float = 0.0
    detector_label: str | None = None
    parent_id: str | None = None

    def to_pixel_coords(self, width: int, height: int) -> np.ndarray:
        scaled = self.vertices * np.array([width, height], dtype=np.float32)
        return scaled.astype(np.int32)

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) normalized."""
        if len(self.vertices) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        xmin, ymin = np.min(self.vertices, axis=0)
        xmax, ymax = np.max(self.vertices, axis=0)
        return (float(xmin), float(ymin), float(xmax), float(ymax))


@dataclass
class TrackingUpdate:
    polygons: list[TrackedPolygon]
    fps: float
    latency_ms: float
    requires_reissue: bool
    motion_magnitude: float


class PlanarMeshTracker:
    """Optical flow polygon mesh propagator running at 30-60 FPS."""

    def __init__(
        self,
        *,
        max_drift_threshold: float = 0.35,
        max_points_per_poly: int = 40,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
    ) -> None:
        self.max_drift_threshold = max_drift_threshold
        self.max_points_per_poly = max_points_per_poly
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self._prev_gray: np.ndarray | None = None
        self._tracked_polygons: list[TrackedPolygon] = []
        self._last_timestamp: float = time.perf_counter()
        self._fps: float = 30.0
        import threading
        self._lock = threading.Lock()

    @property
    def current_polygons(self) -> list[TrackedPolygon]:
        with self._lock:
            return list(self._tracked_polygons)

    def set_polygons(self, raw_polygons: Sequence[tuple[str, int, str, Sequence[tuple[float, float]], str | None, str | None]]) -> None:
        """Assign newly detected polygons from Spatial Feature Issuer (SAM / Grounding DINO).
        
        Tuple format: (polygon_id, level, label, vertices[(x,y)], detector_label, parent_id)
        """
        updated = []
        for p_id, level, label, verts, det_lbl, parent_id in raw_polygons:
            if not verts:
                continue
            v_arr = np.array(verts, dtype=np.float32)
            # Subsample or interpolate if too many or too few points
            if len(v_arr) > self.max_points_per_poly:
                indices = np.linspace(0, len(v_arr) - 1, self.max_points_per_poly, dtype=int)
                v_arr = v_arr[indices]
            updated.append(
                TrackedPolygon(
                    polygon_id=p_id,
                    level=level,
                    label=label,
                    vertices=v_arr,
                    confidence=1.0,
                    age_frames=0,
                    drift_score=0.0,
                    detector_label=det_lbl,
                    parent_id=parent_id,
                )
            )
        with self._lock:
            self._tracked_polygons = updated

    def update_frame(self, frame: np.ndarray) -> TrackingUpdate:
        """Track polygon vertices into the new camera frame."""
        start_t = time.perf_counter()
        
        # Calculate instantaneous FPS
        now = time.perf_counter()
        dt = now - self._last_timestamp
        self._last_timestamp = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = 0.9 * self._fps + 0.1 * instant_fps

        h, w = frame.shape[:2]
        if frame.ndim == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame

        with self._lock:
            tracked_polys_snapshot = list(self._tracked_polygons)

        if self._prev_gray is None or not tracked_polys_snapshot:
            self._prev_gray = curr_gray
            return TrackingUpdate(
                polygons=tracked_polys_snapshot,
                fps=self._fps,
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
                requires_reissue=True if not self._tracked_polygons else False,
                motion_magnitude=0.0,
            )

        # Collect all active points across all polygons
        all_pts_list = []
        poly_slices = []
        offset = 0

        for poly in tracked_polys_snapshot:
            pts_px = poly.vertices * np.array([w, h], dtype=np.float32)
            all_pts_list.append(pts_px)
            poly_slices.append((offset, offset + len(pts_px)))
            offset += len(pts_px)

        if not all_pts_list:
            self._prev_gray = curr_gray
            return TrackingUpdate(
                polygons=[],
                fps=self._fps,
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
                requires_reissue=True,
                motion_magnitude=0.0,
            )

        all_pts_prev = np.vstack(all_pts_list).reshape(-1, 1, 2).astype(np.float32)

        # Calculate optical flow
        all_pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_gray, all_pts_prev, None, **self.lk_params
        )

        self._prev_gray = curr_gray
        status_flat = status.ravel() == 1

        motion_vectors = all_pts_curr[status_flat] - all_pts_prev[status_flat]
        total_motion = np.mean(np.linalg.norm(motion_vectors, axis=1)) if len(motion_vectors) > 0 else 0.0

        surviving_polygons: list[TrackedPolygon] = []
        reissue_needed = False

        for i, poly in enumerate(tracked_polys_snapshot):
            s_start, s_end = poly_slices[i]
            p_prev = all_pts_prev[s_start:s_end].reshape(-1, 2)
            p_curr = all_pts_curr[s_start:s_end].reshape(-1, 2)
            p_status = status_flat[s_start:s_end]

            surviving_ratio = np.sum(p_status) / len(p_status) if len(p_status) > 0 else 0.0

            if surviving_ratio < 0.5:
                # Lost too many vertices -> flag reissue
                reissue_needed = True
                continue

            # Estimate affine or rigid transformation to preserve polygon shape
            valid_prev = p_prev[p_status]
            valid_curr = p_curr[p_status]

            if len(valid_prev) >= 3:
                # Estimate Partial 2D Affine (translation + rotation + uniform scale)
                transform_matrix, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
                if transform_matrix is not None:
                    # Warp entire original polygon with estimated rigid transform to avoid vertex degeneration
                    ones = np.ones((len(p_prev), 1), dtype=np.float32)
                    homo_prev = np.hstack([p_prev, ones])
                    warped_px = (homo_prev @ transform_matrix.T)
                else:
                    warped_px = p_curr
            else:
                # Translation fallback
                shift = np.mean(valid_curr - valid_prev, axis=0)
                warped_px = p_prev + shift

            # Normalize back to [0, 1]
            warped_norm = warped_px / np.array([w, h], dtype=np.float32)
            warped_norm = np.clip(warped_norm, 0.0, 1.0)

            # Compute drift score
            centroid_prev = np.mean(poly.vertices, axis=0)
            centroid_curr = np.mean(warped_norm, axis=0)
            drift = float(np.linalg.norm(centroid_curr - centroid_prev))

            poly.vertices = warped_norm.astype(np.float32)
            poly.age_frames += 1
            poly.drift_score += drift
            poly.confidence = float(surviving_ratio * math.exp(-0.01 * poly.age_frames))

            if poly.drift_score > self.max_drift_threshold:
                reissue_needed = True

            surviving_polygons.append(poly)

        with self._lock:
            # Only update if set_polygons hasn't injected a brand new mesh while we were processing
            if self._tracked_polygons == tracked_polys_snapshot:
                self._tracked_polygons = surviving_polygons
            else:
                # A new spatial inference arrived during our optical flow calc; prefer the new spatial mesh.
                surviving_polygons = self._tracked_polygons

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return TrackingUpdate(
            polygons=surviving_polygons,
            fps=self._fps,
            latency_ms=elapsed_ms,
            requires_reissue=reissue_needed or len(surviving_polygons) == 0,
            motion_magnitude=float(total_motion),
        )
