from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


FIELDGUIDEVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIELDGUIDEVISION_ROOT))

from canonical_adapter import (  # noqa: E402
    TrialProvenance,
    TrialRecord,
    classification_metrics,
    load_canonical_release,
)
from canonical_adapter.trials import write_trial_records  # noqa: E402


FIXTURE = Path(__file__).parent / "fixtures" / "gobotany_canonical_release.json"
P4_RELEASE = FIELDGUIDEVISION_ROOT.parent / "arq-refdata" / "gobotany_extract" / "outputs" / "api_pilot" / "gobotany_release.json"


class CanonicalReleaseTests(unittest.TestCase):
    def test_reader_preserves_release_and_canonical_identifiers(self) -> None:
        release = load_canonical_release(FIXTURE)

        case = release.experiment_cases_for_taxon("Acer campestre")[0]

        self.assertEqual(release.vocabulary_name, "gobotany")
        self.assertEqual(case.vocabulary_release_id, release.vocabulary_release_id)
        self.assertEqual(case.canonical_taxon_id, "Acer campestre")
        self.assertEqual(case.canonical_feature_id, "leaf_arrangement_wa")
        self.assertEqual(case.canonical_value_id, "leaf_arrangement_wa-2_leaves_per_node")
        self.assertEqual(case.source_taxon_id, "1520")
        self.assertEqual(case.source_value_id, "leaf_arrangement_wa:0")

    def test_reader_consumes_approved_p4_release(self) -> None:
        release = load_canonical_release(P4_RELEASE)

        self.assertEqual(release.contract_version, "1.0.0-rc.2")
        self.assertEqual(release.vocabulary_name, "gobotany")
        self.assertEqual(len(release.taxa), 546)
        self.assertEqual(len(release.claims), 38643)
        self.assertTrue(release.data_hash.startswith("sha256:"))


class TrialMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.release = load_canonical_release(FIXTURE)
        self.case = self.release.experiment_cases_for_taxon("Acer campestre")[0]
        self.provenance = TrialProvenance.for_release(
            trial_id="p5b-test",
            model="mock-model",
            adapter_version="p5b",
            release=self.release,
            started_at="2026-07-16T12:00:00+00:00",
        )

    def test_abstention_is_excluded_from_classification_accuracy(self) -> None:
        records = [
            TrialRecord(self.provenance, self.case, "leaf_arrangement_wa-2_leaves_per_node", "classified", self.case.canonical_value_id),
            TrialRecord(self.provenance, self.case, "cannot determine", "abstained", None),
            TrialRecord(self.provenance, self.case, "other", "classified", "leaf_arrangement_wa-1_leaf_per_node"),
        ]

        metrics = classification_metrics(records)

        self.assertEqual(metrics.total_records, 3)
        self.assertEqual(metrics.classified_records, 2)
        self.assertEqual(metrics.abstained_records, 1)
        self.assertEqual(metrics.correct_classifications, 1)
        self.assertEqual(metrics.classification_accuracy, 0.5)
        self.assertEqual(metrics.classification_coverage, 2 / 3)

    def test_trial_artifact_retains_raw_response_and_release_provenance(self) -> None:
        record = TrialRecord(self.provenance, self.case, "full unmodified model response", "abstained", None)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trial.json"
            write_trial_records(path, [record])
            payload = json.loads(path.read_text())

        serialized = payload["records"][0]
        self.assertEqual(serialized["raw_response"], "full unmodified model response")
        self.assertEqual(serialized["provenance"]["release_id"], self.release.vocabulary_release_id)
        self.assertEqual(serialized["case"]["canonical_value_id"], self.case.canonical_value_id)
        self.assertEqual(payload["metrics"]["classification_accuracy"], None)

    def test_trial_rejects_release_provenance_mismatch(self) -> None:
        mismatch = TrialProvenance(
            **{**self.provenance.__dict__, "release_id": "sha256:" + "0" * 64}
        )
        with self.assertRaisesRegex(ValueError, "release ID"):
            TrialRecord(mismatch, self.case, "answer", "abstained", None)


if __name__ == "__main__":
    unittest.main()
