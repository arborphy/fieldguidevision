from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


FIELDGUIDEVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIELDGUIDEVISION_ROOT))

from canonical_adapter import load_canonical_release  # noqa: E402
from hierarchical_pipeline import (  # noqa: E402
    CuratedWorkflow,
    FeatureEvidence,
    WorkflowEvidence,
    WorkflowNode,
    recommend_instance,
)
from segmentation.pipeline import CaptureFrame, RawSegment, build_notice_batch  # noqa: E402


FIXTURE = Path(__file__).parent / "fixtures" / "gobotany_canonical_release.json"


class HierarchicalPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.release = load_canonical_release(FIXTURE)
        self.workflow = CuratedWorkflow((
            WorkflowNode("kingdom:plantae", "plant", "taxon", taxon_rank="kingdom"),
            WorkflowNode("notice:leaf", "leaf", "notice", "kingdom:plantae"),
            WorkflowNode(
                "feature:leaf-arrangement",
                "leaf arrangement",
                "feature",
                "notice:leaf",
                source_feature_id="leaf_arrangement_wa",
            ),
        ))

    def test_only_confirmed_parent_unlocks_more_specific_work(self) -> None:
        self.assertEqual(
            [node.node_id for node in self.workflow.next_tasks(())],
            ["kingdom:plantae"],
        )
        self.assertEqual(
            [node.node_id for node in self.workflow.next_tasks((
                WorkflowEvidence("kingdom:plantae", "confirmed"),
            ))],
            ["notice:leaf"],
        )
        self.assertEqual(
            [node.node_id for node in self.workflow.next_tasks((
                WorkflowEvidence("kingdom:plantae", "confirmed"),
                WorkflowEvidence("notice:leaf", "confirmed", "notice-proposal-1"),
            ))],
            ["feature:leaf-arrangement"],
        )

    def test_rejected_parent_does_not_unlock_children(self) -> None:
        self.assertEqual(
            self.workflow.next_tasks((WorkflowEvidence("kingdom:plantae", "rejected"),)),
            (),
        )

    def test_curator_confirmed_source_feature_recommends_a_species(self) -> None:
        recommendation = recommend_instance((self.release,), (
            FeatureEvidence(
                release_id=self.release.vocabulary_release_id,
                feature_id="leaf_arrangement_wa",
                value_id="leaf_arrangement_wa-2_leaves_per_node",
                notice_proposal_id="notice-proposal-1",
                annotation_id="annotation-1",
                confidence=0.82,
            ),
        ))
        self.assertEqual(recommendation.status, "recommended")
        self.assertEqual(recommendation.taxon_name, "Acer campestre")
        self.assertEqual(recommendation.taxon_rank, "species")
        self.assertEqual(recommendation.evidence[0].notice_proposal_id, "notice-proposal-1")
        self.assertEqual(recommendation.release_ids, (self.release.vocabulary_release_id,))

    def test_unknown_or_mismatched_source_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not belong"):
            recommend_instance((self.release,), (
                FeatureEvidence(
                    release_id=self.release.vocabulary_release_id,
                    feature_id="not_leaf_arrangement",
                    value_id="leaf_arrangement_wa-2_leaves_per_node",
                    notice_proposal_id="notice-proposal-1",
                ),
            ))

    def test_curator_confirmation_is_required_for_taxon_narrowing(self) -> None:
        with self.assertRaisesRegex(ValueError, "curator-confirmed"):
            FeatureEvidence(
                release_id=self.release.vocabulary_release_id,
                feature_id="leaf_arrangement_wa",
                value_id="leaf_arrangement_wa-2_leaves_per_node",
                notice_proposal_id="notice-proposal-1",
                review_state="proposed",
            )

    def test_multiple_candidates_can_stop_at_shared_taxonomic_rank(self) -> None:
        expanded_release = replace(
            self.release,
            taxa={
                **self.release.taxa,
                "Acer platanoides": {"taxon_name": "Acer platanoides"},
            },
            claims=(*self.release.claims, {
                "devo_name": "gobotany",
                "taxon_name": "Acer platanoides",
                "feature_name": "leaf_arrangement_wa",
                "vocab_id": "leaf_arrangement_wa-2_leaves_per_node",
                "assertion_source": "test",
            }),
        )
        recommendation = recommend_instance(
            (expanded_release,),
            (FeatureEvidence(
                release_id=expanded_release.vocabulary_release_id,
                feature_id="leaf_arrangement_wa",
                value_id="leaf_arrangement_wa-2_leaves_per_node",
                notice_proposal_id="notice-proposal-1",
            ),),
            {"Acer campestre": {"genus": "Acer"}, "Acer platanoides": {"genus": "Acer"}},
        )
        self.assertEqual(recommendation.status, "ambiguous")
        self.assertEqual(recommendation.taxon_rank, "genus")
        self.assertEqual(recommendation.taxon_name, "Acer")

    def test_polygon_proposal_can_carry_reviewed_feature_evidence_to_recommendation(self) -> None:
        class LeafSegmenter:
            model_name = "test-segmenter"
            model_version = "test-1"

            def segment(self, frame, prompts):
                return (RawSegment(
                    label="leaf",
                    polygon=((0.2, 0.2), (0.8, 0.2), (0.7, 0.7)),
                    detection_confidence=0.9,
                    mask_quality=0.9,
                ),)

        batch = build_notice_batch((
            CaptureFrame("obs-1", "https://example.test/leaf.jpg", "subject-1"),
        ), LeafSegmenter())
        proposal = batch.proposals[0]
        recommendation = recommend_instance((self.release,), (
            FeatureEvidence(
                release_id=self.release.vocabulary_release_id,
                feature_id="leaf_arrangement_wa",
                value_id="leaf_arrangement_wa-2_leaves_per_node",
                notice_proposal_id=proposal.proposal_id,
            ),
        ))
        self.assertEqual(proposal.category_concept_id, "category:leaf")
        self.assertEqual(recommendation.status, "recommended")
        self.assertEqual(recommendation.evidence[0].notice_proposal_id, proposal.proposal_id)


if __name__ == "__main__":
    unittest.main()
