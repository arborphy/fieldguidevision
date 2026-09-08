"""Offline checks for the enclosed scratch demo; no models, keys, or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

DEMO_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fgv-demo.py"
SPEC = importlib.util.spec_from_file_location("fgv_demo", DEMO_PATH)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image_path = Path(self.temp.name) / "test.png"
        image = np.zeros((20, 30, 3), np.uint8)
        image[:, :15] = (0, 255, 0)
        image[:, 15:] = (0, 0, 255)
        cv2.imwrite(str(self.image_path), image)
        self.input = {"image": str(self.image_path), "tags": ["leaf"], "regions": [
            {"id": "leaf-1", "organ": "leaf", "bbox": [0, 0, 15, 20], "subject_hint": "maybe A"}]}

    def test_repeatability_and_region_histogram(self):
        a, _, _ = demo.classify_photo(self.input)
        b, _, _ = demo.classify_photo(self.input)
        self.assertEqual(a, b)
        stats = a["measurements"]["leaf-1"]
        self.assertEqual(stats["pixel_count"], 300)
        self.assertEqual(stats["mean_rgb"], [0, 255, 0])
        for hist in stats["rgb_histogram"].values():
            self.assertEqual(sum(hist), 300)
        self.assertIsNone(a["taxon_assignment"])
        self.assertEqual(a["confirmed_assertions"], [])

    def test_tag_only_does_not_invent_regions(self):
        data, _, masks = demo.classify_photo({"image": str(self.image_path), "tags": ["leaf", "trunk", "trunk"]})
        self.assertEqual(data["regions"], [])
        self.assertEqual(list(masks), ["scene"])
        self.assertEqual(data["organ_tags"], ["leaf", "trunk"])

    def test_polygon_and_invalid_geometry(self):
        mask, geometry = demo.region_mask({"polygon": [[1, 1], [10, 1], [1, 10]]}, (20, 30, 3))
        self.assertGreater(np.count_nonzero(mask), 0)
        self.assertEqual(geometry, {"polygon": [[1, 1], [10, 1], [1, 10]]})
        for region in ({"bbox": [-1, 0, 5, 5]}, {"bbox": [0, 0, 31, 20]},
                       {"bbox": [1, 1, 1, 2]}, {"bbox": [0.5, 0, 5, 5]},
                       {"polygon": [[0, 0], [1, 1], [2, 2]]}, {},
                       {"bbox": [0, 0, 2, 2], "polygon": [[0, 0], [1, 0], [1, 1]]}):
            with self.subTest(region=region), self.assertRaises(ValueError):
                demo.region_mask(region, (20, 30, 3))

    def test_duplicate_region_is_rejected(self):
        self.input["regions"] *= 2
        with self.assertRaises(ValueError):
            demo.classify_photo(self.input)

    def test_canonical_provenance_and_unknown_organs(self):
        fixture = demo.ROOT / "tests/fixtures/gobotany_canonical_release.json"
        data, _, _ = demo.classify_photo(self.input, [fixture])
        item = data["inspection_checklist"][0]
        self.assertEqual(item["vocabulary"], "gobotany")
        self.assertEqual(item["feature_id"], "leaf_arrangement_wa")
        self.assertEqual(item["state"], "not_assessed")
        self.assertEqual(item["allowed_values"][0]["source_value_id"], "leaf_arrangement_wa:0")
        _, checks = demo.devo_checklist([fixture], ["unmapped organ"])
        self.assertEqual(checks, [])

    def test_model_proposals_are_validated_and_never_confirmed(self):
        import sys
        sys.path.insert(0, str(demo.ROOT / "scripts/adapters"))
        data, image, masks = demo.classify_photo(self.input)
        valid = {"notices": [{"region_id": "leaf-1", "description": "Green surface",
                             "uncertainty": "Margin partly hidden", "check_ids": []}], "next_capture": []}
        original = json.dumps(data, sort_keys=True)
        with patch("openrouter_command_adapter.call_openrouter", return_value=json.dumps(valid)):
            result = demo.organ_classification_agent(data, image, masks, "mock-vision")
        self.assertEqual(result["notices"][0]["review_state"], "proposed")
        self.assertFalse(result["deterministic"])
        self.assertEqual(json.dumps(data, sort_keys=True), original)
        valid["notices"][0]["check_ids"] = ["invented-id"]
        with patch("openrouter_command_adapter.call_openrouter", return_value=json.dumps(valid)):
            result = demo.organ_classification_agent(data, image, masks, "mock-vision")
        self.assertEqual(result["state"], "invalid_response")
        self.assertEqual(result["notices"], [])
        self.assertIn("invented-id", result["raw_response_text"])

    def test_report_escapes_untrusted_text(self):
        data, image, masks = demo.classify_photo(self.input)
        report = demo.render_report(data, image, masks, {"text": "<script>alert(1)</script>"})
        self.assertNotIn("<script>", report)
        self.assertIn("&lt;script&gt;", report)
        self.assertIn("RGB histogram", report)

    def test_cli_report_and_no_overwrite(self):
        out = Path(self.temp.name) / "run"
        command = [sys.executable, str(DEMO_PATH),
                   "--image", str(self.image_path), "--tags", "leaf", "--devo",
                   str(demo.ROOT / "tests/fixtures/gobotany_canonical_release.json"),
                   "--output-dir", str(out)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = (out / "evidence.json").read_bytes()
        evidence = json.loads(before)
        self.assertEqual(evidence["measurements"]["scene"]["pixel_count"], 600)
        self.assertTrue(evidence["inspection_checklist"])
        self.assertEqual(json.loads((out / "proposals.json").read_text())["state"], "not_requested")
        self.assertIn("data:image/png;base64,", (out / "report.html").read_text())
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual((out / "evidence.json").read_bytes(), before)
        repeated = Path(self.temp.name) / "repeat"
        third = subprocess.run(command[:-1] + [str(repeated)], capture_output=True, text=True)
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual((repeated / "evidence.json").read_bytes(), before)
        self.assertEqual((repeated / "report.html").read_bytes(), (out / "report.html").read_bytes())

    def test_cli_relative_input_and_remote_failure_preserve_evidence(self):
        input_path = Path(self.temp.name) / "input.json"
        input_path.write_text(json.dumps(self.input | {"image": "test.png"}))
        out = Path(self.temp.name) / "failed-model-run"
        result = subprocess.run([
            sys.executable, str(DEMO_PATH),
            "--input", str(input_path), "--output-dir", str(out),
            "--openrouter-model", "mock-vision",
        ], capture_output=True, text=True, env=dict(os.environ, OPENROUTER_API_KEY=""))
        self.assertEqual(result.returncode, 2, result.stderr)
        evidence = json.loads((out / "evidence.json").read_text())
        self.assertEqual(evidence["measurements"]["leaf-1"]["pixel_count"], 300)
        self.assertEqual(json.loads((out / "proposals.json").read_text())["state"], "error")
        self.assertTrue((out / "report.html").is_file())


if __name__ == "__main__":
    unittest.main()
