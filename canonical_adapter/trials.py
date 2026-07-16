"""Immutable trial records and metrics that separate abstention from accuracy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Literal

from .release import ExperimentCase


ParsedOutcome = Literal["classified", "abstained", "invalid"]


@dataclass(frozen=True)
class TrialProvenance:
    trial_id: str
    started_at: str
    model: str
    adapter_version: str
    release_id: str
    release_data_hash: str
    vocabulary_name: str
    contract_version: str

    @classmethod
    def for_release(
        cls,
        *,
        trial_id: str,
        model: str,
        adapter_version: str,
        release,
        started_at: str | None = None,
    ) -> "TrialProvenance":
        return cls(
            trial_id=trial_id,
            started_at=started_at or datetime.now().astimezone().isoformat(timespec="seconds"),
            model=model,
            adapter_version=adapter_version,
            release_id=release.vocabulary_release_id,
            release_data_hash=release.data_hash,
            vocabulary_name=release.vocabulary_name,
            contract_version=release.contract_version,
        )


@dataclass(frozen=True)
class TrialRecord:
    provenance: TrialProvenance
    case: ExperimentCase
    raw_response: str
    parsed_outcome: ParsedOutcome
    predicted_canonical_value_id: str | None

    def __post_init__(self) -> None:
        if self.case.vocabulary_release_id != self.provenance.release_id:
            raise ValueError("Trial case release ID must match trial provenance")
        if self.case.vocabulary_name != self.provenance.vocabulary_name:
            raise ValueError("Trial case vocabulary must match trial provenance")
        if self.parsed_outcome == "classified" and not self.predicted_canonical_value_id:
            raise ValueError("Classified trial records require a canonical prediction")
        if self.parsed_outcome != "classified" and self.predicted_canonical_value_id is not None:
            raise ValueError("Abstained or invalid records cannot contain a classification")

    @property
    def is_correct_classification(self) -> bool | None:
        if self.parsed_outcome != "classified":
            return None
        return self.predicted_canonical_value_id == self.case.canonical_value_id


@dataclass(frozen=True)
class ClassificationMetrics:
    total_records: int
    classified_records: int
    abstained_records: int
    invalid_records: int
    correct_classifications: int
    classification_accuracy: float | None
    classification_coverage: float | None


def classification_metrics(records: list[TrialRecord] | tuple[TrialRecord, ...]) -> ClassificationMetrics:
    """Calculate accuracy only among actual classifications, never abstentions."""
    classified = [record for record in records if record.parsed_outcome == "classified"]
    abstained = sum(record.parsed_outcome == "abstained" for record in records)
    invalid = sum(record.parsed_outcome == "invalid" for record in records)
    correct = sum(record.is_correct_classification is True for record in classified)
    total = len(records)
    return ClassificationMetrics(
        total_records=total,
        classified_records=len(classified),
        abstained_records=abstained,
        invalid_records=invalid,
        correct_classifications=correct,
        classification_accuracy=correct / len(classified) if classified else None,
        classification_coverage=len(classified) / total if total else None,
    )


def write_trial_records(path: str | Path, records: list[TrialRecord] | tuple[TrialRecord, ...]) -> None:
    """Write raw responses and immutable release provenance as one JSON artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": [asdict(record) for record in records],
        "metrics": asdict(classification_metrics(records)),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
