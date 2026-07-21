#!/usr/bin/env python3
"""Serve real Grounding-DINO + SAM 2.1 organ gate evaluations.

This local/LAN development service evaluates one DeVo noticing category per
sampled camera frame. It returns geometry/confidence only and never emits a
taxon or source-vocabulary assertion.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmentation.grounded_sam2 import GroundedSam2Segmenter  # noqa: E402
from segmentation.pipeline import CaptureFrame  # noqa: E402


class GrowthFormGate:
    def __init__(self, *, device: str, detector_model: str, segmenter_model: str):
        self.segmenter = GroundedSam2Segmenter(
            device=device, detector_model=detector_model, segmenter_model=segmenter_model,
        )
        self.telemetry: list[dict] = []

    def record_event(self, event: dict) -> None:
        """Keep a bounded local trace for the active field-debug session."""
        self.telemetry.append({"received_at": datetime.now(timezone.utc).isoformat(), **event})
        del self.telemetry[:-200]

    def evaluate(self, image_base64: str, threshold: float, layer: str) -> dict:
        prompts = {
            "growth_form": "whole plant", "leaf": "leaf", "bark": "bark",
            "stem": "stem", "bud": "bud", "flower": "flower", "fruit": "fruit",
        }
        if layer not in prompts:
            raise ValueError(f"unsupported organ layer: {layer}")
        encoded = image_base64.split(",", 1)[-1]
        payload = base64.b64decode(encoded)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            image.write(payload)
            image.flush()
            frame = CaptureFrame(
                observation_id=f"live-{layer}-gate", image_url="live://camera",
                local_path=image.name, subject_id=f"live-{layer}",
            )
            segments = self.segmenter.segment(frame, (prompts[layer],))
        proposals = []
        for index, segment in enumerate(segments):
            confidence = round(0.5 * segment.detection_confidence + 0.5 * segment.mask_quality, 6)
            polygon = [[x, y] for x, y in segment.polygon]
            if polygon and polygon[0] != polygon[-1]:
                polygon.append(polygon[0])
            proposals.append({
                "proposal_id": f"live-{layer}-{index}", "category": prompts[layer],
                "polygon": polygon, "confidence": confidence,
                "detector_confidence": segment.detection_confidence,
                "mask_quality": segment.mask_quality, "prompt": segment.prompt,
            })
        proposals.sort(key=lambda item: item["confidence"], reverse=True)
        best = proposals[0] if proposals else None
        return {
            "provider": {"name": "grounding-dino+sam2.1", "version": self.segmenter.model_version},
            "layer": layer, "threshold": threshold,
            "confidence": best["confidence"] if best else 0.0,
            "gate_open": bool(best and best["confidence"] >= threshold),
            "proposal": best, "proposal_count": len(proposals),
            "taxon_bets": {"state": "blocked", "reason": "Organ geometry is not a taxonomic claim."},
        }


def handler_for(gate: GrowthFormGate):
    # Baked-client-key auth: when GATE_API_KEY is set, every endpoint requires
    # the X-API-Key header. Deployed APKs carry the key; revocation is a
    # secret rotation + app rebuild. When unset (local dev), the gate is open.
    required_key = os.environ.get("GATE_API_KEY") or None

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            if required_key is None:
                return True
            presented = self.headers.get("X-API-Key") or ""
            return hmac.compare_digest(presented.encode(), required_key.encode())

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.end_headers()

        def do_POST(self):  # noqa: N802
            if self.path not in {"/growth-form-gate", "/growth-form-gate/telemetry"}:
                self._json(404, {"detail": "not found"})
                return
            if not self._authorized():
                self._json(401, {"detail": "invalid or missing X-API-Key"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(size))
                if self.path.endswith("/telemetry"):
                    gate.record_event(request)
                    self._json(202, {"accepted": True, "events": len(gate.telemetry)})
                    return
                threshold = float(request.get("threshold", 0.75))
                if not 0 <= threshold <= 1:
                    raise ValueError("threshold must be between 0 and 1")
                self._json(200, gate.evaluate(request["image_base64"], threshold, request.get("layer", "growth_form")))
            except Exception as error:
                self._json(422, {"detail": str(error)})

        def do_GET(self):  # noqa: N802
            if self.path == "/growth-form-gate/telemetry":
                if not self._authorized():
                    self._json(401, {"detail": "invalid or missing X-API-Key"})
                    return
                self._json(200, {"events": gate.telemetry, "count": len(gate.telemetry)})
                return
            self._json(404, {"detail": "not found"})

        def log_message(self, format, *args):  # noqa: A003
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8011, type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--segmenter-model", default="facebook/sam2.1-hiera-tiny")
    args = parser.parse_args()
    gate = GrowthFormGate(device=args.device, detector_model=args.detector_model, segmenter_model=args.segmenter_model)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(gate))
    print(f"Growth-form gate listening on http://{args.host}:{args.port}/growth-form-gate")
    server.serve_forever()


if __name__ == "__main__":
    main()
