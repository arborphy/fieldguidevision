from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from segmentation.pipeline import (
    CaptureFrame,
    RawSegment,
    build_notice_batch,
    canonical_organ_label,
    load_inaturalist_frames,
)
from scripts.select_notice_gallery import select_gallery


class FakeSegmenter:
    model_name = "fake-sam"
    model_version = "test-1"

    def segment(self, frame, prompts):
        score = 0.95 if frame.frame_index == 1 else 0.55
        return (
            RawSegment(
                label="leaf",
                polygon=((0.2, 0.2), (0.8, 0.2), (0.75, 0.75), (0.25, 0.8)),
                detection_confidence=score,
                mask_quality=score,
                stability_score=0.9,
            ),
            RawSegment(
                label="flower",
                polygon=((0.0, 0.0), (0.01, 0.0), (0.01, 0.01)),
                detection_confidence=0.2,
                mask_quality=0.3,
            ),
        )


class FailingSegmenter(FakeSegmenter):
    def segment(self, frame, prompts):
        if frame.observation_id == "bad-obs":
            raise TimeoutError("image download timed out")
        return super().segment(frame, prompts)


class SegmentationPipelineTests(unittest.TestCase):
    def test_retains_all_proposals_and_selects_top_k_per_subject_category(self):
        frames = (
            CaptureFrame("obs-1", "https://example/1.jpg", "plant-1", frame_index=0),
            CaptureFrame("obs-2", "https://example/2.jpg", "plant-1", frame_index=1),
        )
        batch = build_notice_batch(
            frames, FakeSegmenter(), top_k=1, selection_scope="subject-category"
        )
        leaves = [p for p in batch.proposals if p.category_concept_id == "category:leaf"]
        self.assertEqual(len(leaves), 2)
        self.assertEqual(sum(p.selected_for_review for p in leaves), 1)
        selected = next(p for p in leaves if p.selected_for_review)
        self.assertEqual(selected.observation_id, "obs-2")
        self.assertEqual(selected.representative_rank, 1)

    def test_category_scope_selects_top_k_across_subjects(self):
        frames = (
            CaptureFrame("obs-1", "https://example/1.jpg", "plant-1", frame_index=0),
            CaptureFrame("obs-2", "https://example/2.jpg", "plant-2", frame_index=1),
        )
        batch = build_notice_batch(frames, FakeSegmenter(), top_k=1)
        leaves = [p for p in batch.proposals if p.category_concept_id == "category:leaf"]
        self.assertEqual(sum(p.selected_for_review for p in leaves), 1)
        self.assertEqual(next(p for p in leaves if p.selected_for_review).observation_id, "obs-2")

    def test_low_quality_edge_segment_is_preserved_as_training_signal(self):
        frame = CaptureFrame("obs-1", "https://example/1.jpg", "plant-1")
        batch = build_notice_batch((frame,), FakeSegmenter())
        flower = next(p for p in batch.proposals if p.category_concept_id == "category:flower")
        self.assertIn("small_region", flower.quality.flags)
        self.assertIn("touches_frame_edge", flower.quality.flags)
        self.assertEqual(flower.review_state, "proposed")

    def test_export_has_bbox_polygon_model_and_ranking_fields(self):
        frame = CaptureFrame("obs-1", "https://example/1.jpg", "plant-1", license="CC-BY")
        payload = build_notice_batch((frame,), FakeSegmenter()).to_dict()
        annotation = payload["observations"][0]["annotations"][0]
        self.assertEqual(payload["schema_version"], "arq.notice-proposal-batch/v1")
        self.assertIn("polygon", annotation)
        self.assertIn("bbox_x0", annotation)
        self.assertIn("quality", annotation)
        self.assertEqual(annotation["model"]["name"], "fake-sam")

    def test_batch_artifacts_are_immutable(self):
        frame = CaptureFrame("obs-1", "https://example/1.jpg", "plant-1")
        batch = build_notice_batch((frame,), FakeSegmenter())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.json"
            batch.write(path)
            self.assertEqual(json.loads(path.read_text())["summary"]["frames"], 1)
            with self.assertRaises(FileExistsError):
                batch.write(path)

    def test_one_bad_frame_does_not_abort_stream(self):
        frames = (
            CaptureFrame("good-obs", "https://example/good.jpg", "plant-1"),
            CaptureFrame("bad-obs", "https://example/bad.jpg", "plant-2"),
        )
        batch = build_notice_batch(frames, FailingSegmenter())
        self.assertEqual(len(batch.failures), 1)
        self.assertEqual(batch.failures[0].observation_id, "bad-obs")
        self.assertTrue(batch.proposals)

    def test_loads_first_100_photo_rows_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text(
                "id,image_url,scientific_name,license,observed_on\n"
                "1,https://example/1.jpg,Acer rubrum,CC-BY,2026-01-01\n"
                "2,,Quercus alba,CC-BY,2026-01-02\n"
                "3,https://example/3.jpg,Quercus alba,CC-BY-NC,2026-01-03\n"
            )
            frames = load_inaturalist_frames(path, limit=2)
        self.assertEqual([frame.observation_id for frame in frames], ["1", "3"])
        self.assertEqual(frames[1].subject_id, "Quercus alba")

    def test_gallery_selects_only_top_k_per_category_without_mutating_batch(self):
        frames = (
            CaptureFrame("obs-1", "https://example/1.jpg", "plant-1", frame_index=0),
            CaptureFrame("obs-2", "https://example/2.jpg", "plant-2", frame_index=1),
        )
        original = build_notice_batch(frames, FakeSegmenter()).to_dict()
        gallery = select_gallery(original, top_k=1, scope="category")
        leaves = [item for item in gallery["annotations"] if item["notice_category_id"] == "category:leaf"]
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0]["observation_id"], "obs-2")
        self.assertEqual(len(original["observations"]), 2)

    def test_composite_detector_label_is_not_silently_assigned_to_one_organ(self):
        self.assertEqual(canonical_organ_label("leaf"), "leaf")
        self.assertEqual(canonical_organ_label("fruit bud"), "ambiguous organ")


if __name__ == "__main__":
    unittest.main()
