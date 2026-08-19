"""Recursive Depth Resolution Engine for Field Guide Vision.

Subdivides coarse structural polygons (Level 1: canopy, trunk, whole plant)
into sharp botanical micro-features (Level 2: leaf lobes, serrated margins,
bark furrows, petiole base) without needing full VLM re-classification.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Sequence
from PIL import Image
import numpy as np

from .pipeline import CaptureFrame, RawSegment


@dataclass(frozen=True)
class HierarchicalPolygon:
    polygon_id: str
    level: int  # 1: Structural Mesh, 2: Micro-Feature
    label: str
    polygon: tuple[tuple[float, float], ...]  # Normalized (x, y) [0.0, 1.0]
    parent_id: str | None = None
    confidence: float = 1.0
    detector_label: str | None = None
    bounding_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # (xmin, ymin, xmax, ymax)


class RecursiveDepthResolver:
    """Manages hierarchical spatial polygon refinement between Level 1 and Level 2."""

    def __init__(self, default_micro_prompts: Sequence[str] | None = None) -> None:
        self.default_micro_prompts = default_micro_prompts or (
            "leaf lobe",
            "toothed leaf margin",
            "leaf petiole",
            "furrowed bark",
            "scaly bark",
            "fruit",
            "flower",
        )

    def build_level1_polygons(
        self,
        raw_segments: Sequence[RawSegment],
    ) -> list[HierarchicalPolygon]:
        """Convert initial Grounding DINO + SAM segments into Level 1 structural polygons."""
        polys = []
        for seg in raw_segments:
            if not seg.polygon:
                continue
            poly_id = f"lvl1_{uuid.uuid4().hex[:8]}"
            pts = np.array(seg.polygon)
            xmin, ymin = np.min(pts, axis=0)
            xmax, ymax = np.max(pts, axis=0)
            
            polys.append(
                HierarchicalPolygon(
                    polygon_id=poly_id,
                    level=1,
                    label=seg.label,
                    polygon=seg.polygon,
                    parent_id=None,
                    confidence=seg.detection_confidence * seg.mask_quality,
                    detector_label=seg.detector_label or seg.label,
                    bounding_box=(float(xmin), float(ymin), float(xmax), float(ymax)),
                )
            )
        return polys

    def prepare_subdivision_crops(
        self,
        image: Image.Image,
        parent_polygons: Sequence[HierarchicalPolygon],
        padding_ratio: float = 0.05,
    ) -> list[tuple[HierarchicalPolygon, Image.Image, tuple[float, float, float, float]]]:
        """Extract ROI image crops and their coordinate bounding transforms for Level 2 refinement."""
        crops: list[tuple[HierarchicalPolygon, Image.Image, tuple[float, float, float, float]]] = []
        w, h = image.size

        for parent in parent_polygons:
            xmin, ymin, xmax, ymax = parent.bounding_box
            # Add padding
            pw = (xmax - xmin) * padding_ratio
            ph = (ymax - ymin) * padding_ratio

            c_xmin = max(0.0, xmin - pw)
            c_ymin = max(0.0, ymin - ph)
            c_xmax = min(1.0, xmax + pw)
            c_ymax = min(1.0, ymax + ph)

            crop_box_px = (
                int(c_xmin * w),
                int(c_ymin * h),
                int(c_xmax * w),
                int(c_ymax * h),
            )

            if crop_box_px[2] - crop_box_px[0] < 20 or crop_box_px[3] - crop_box_px[1] < 20:
                continue

            crop_img = image.crop(crop_box_px)
            crops.append((parent, crop_img, (c_xmin, c_ymin, c_xmax, c_ymax)))

        return crops

    def transform_micro_segments(
        self,
        parent: HierarchicalPolygon,
        raw_micro_segments: Sequence[RawSegment],
        crop_bounds: tuple[float, float, float, float],
    ) -> list[HierarchicalPolygon]:
        """Project ROI crop micro-segments back into full-frame normalized coordinates."""
        c_xmin, c_ymin, c_xmax, c_ymax = crop_bounds
        crop_w = c_xmax - c_xmin
        crop_h = c_ymax - c_ymin

        micro_polys: list[HierarchicalPolygon] = []

        for seg in raw_micro_segments:
            if not seg.polygon:
                continue
            # Translate vertices from crop space to full image space
            translated_pts = []
            for px, py in seg.polygon:
                gx = c_xmin + px * crop_w
                gy = c_ymin + py * crop_h
                translated_pts.append((float(gx), float(gy)))

            poly_id = f"lvl2_{uuid.uuid4().hex[:8]}"
            pts_arr = np.array(translated_pts, dtype=np.float32)
            
            # Ensure micro-feature centroid is actually inside the L1 parent polygon
            import cv2
            centroid = np.mean(pts_arr, axis=0)
            parent_pts = np.array(parent.polygon, dtype=np.float32)
            if cv2.pointPolygonTest(parent_pts, (float(centroid[0]), float(centroid[1])), False) < 0:
                continue
                
            xmin, ymin = np.min(pts_arr, axis=0)
            xmax, ymax = np.max(pts_arr, axis=0)

            micro_polys.append(
                HierarchicalPolygon(
                    polygon_id=poly_id,
                    level=2,
                    label=seg.label,
                    polygon=tuple(translated_pts),
                    parent_id=parent.polygon_id,
                    confidence=seg.detection_confidence * seg.mask_quality,
                    detector_label=seg.detector_label or seg.label,
                    bounding_box=(float(xmin), float(ymin), float(xmax), float(ymax)),
                )
            )

        return micro_polys
