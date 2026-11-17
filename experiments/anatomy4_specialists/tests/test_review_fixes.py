# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from experiments.anatomy4_specialists.contact_sheet import _load_font, load_sheet_rows
from experiments.anatomy4_specialists.contracts import (
    FLOWER_INFLORESCENCE_TYPE,
    LEAF_SIMPLE_VS_COMPOUND,
)
from experiments.anatomy4_specialists.dataset import validate_dataset
from experiments.anatomy4_specialists.device import configure_memory, vram_requirement
from experiments.anatomy4_specialists.dirr_join import Claim, ClaimsIndex, join_weak_labels, load_claims_db
from experiments.anatomy4_specialists.evaluate import (
    classifier_names,
    probabilities_from_output,
    reorder_probabilities,
)
from experiments.anatomy4_specialists.metrics import abstention_report
from experiments.anatomy4_specialists.splits import assign_splits
from experiments.anatomy4_specialists.trainer import resolve_resume


def _crop(crop_id: str, taxon: str, anatomy: str):
    return {
        "crop_id": crop_id,
        "observation_id": f"obs-{crop_id}",
        "taxon_name": taxon,
        "anatomy_class": anatomy,
        "image": f"https://example.invalid/{crop_id}.jpg",
        "crop_path": f"/tmp/{crop_id}.jpg",
        "detector_confidence": 0.9,
        "observation_split": "train",
    }


# --- P0.2: no double softmax -------------------------------------------------
def test_probabilities_are_not_softmaxed_twice() -> None:
    probs = np.asarray([[0.7, 0.3], [0.2, 0.8]])
    assert np.allclose(probabilities_from_output(probs), probs)


def test_logits_are_softmaxed_once() -> None:
    logits = np.asarray([[2.0, 1.0]])
    expected = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert np.allclose(probabilities_from_output(logits), expected)
    # A probability matrix returned by the model must pass through unchanged.
    assert np.allclose(probabilities_from_output(expected), expected)


# --- P0.3: model order -> target order ---------------------------------------
def test_classifier_names_and_reorder() -> None:
    class Fake:
        names = {0: "Corymb", 1: "Cyme", 2: "Panicle", 3: "Raceme"}

    names = classifier_names(Fake())
    assert names == ["Corymb", "Cyme", "Panicle", "Raceme"]
    probs = np.asarray([[0.10, 0.20, 0.30, 0.40]])
    reordered = reorder_probabilities(probs, names, FLOWER_INFLORESCENCE_TYPE.classes)
    assert np.allclose(reordered, [[0.40, 0.10, 0.20, 0.30]])


def test_reorder_rejects_class_set_mismatch() -> None:
    with pytest.raises(ValueError, match="do not match"):
        reorder_probabilities([[0.5, 0.5]], ["Simple", "Other"], LEAF_SIMPLE_VS_COMPOUND.classes)


# --- P1.4: hybrid marker collision ------------------------------------------
def test_hybrid_marker_forms_collide() -> None:
    claims = [Claim("Forsythia xintermedia", "leaf_type", "Simple")]
    joined = join_weak_labels(
        [_crop("crop", "Forsythia × intermedia", "leaf")], claims, LEAF_SIMPLE_VS_COMPOUND
    )["rows"][0]
    assert joined["target_label"] == "Simple"

    reverse = [Claim("Forsythia × intermedia", "leaf_type", "Simple")]
    joined = join_weak_labels(
        [_crop("crop", "Forsythia xintermedia", "leaf")], reverse, LEAF_SIMPLE_VS_COMPOUND
    )["rows"][0]
    assert joined["target_label"] == "Simple"


# --- P1.5: any out-of-target in-feature value is ambiguous -------------------
def test_mixed_in_and_out_of_target_values_is_ambiguous() -> None:
    claims = [
        Claim("Mixed leaf", "leaf_type", "Simple"),
        Claim("Mixed leaf", "leaf_type", "Needlelike"),
    ]
    joined = join_weak_labels(
        [_crop("crop", "Mixed leaf", "leaf")], claims, LEAF_SIMPLE_VS_COMPOUND
    )["rows"][0]
    assert joined["weak_label_reason"] == "ambiguous_claim"
    assert joined["claim_out_of_target_values"] == ["Needlelike"]
    assert "target_label" not in joined


# --- P1.6: class-aware seed search ------------------------------------------
def _split_rows() -> list[dict]:
    rows = []
    for i in range(12):
        label = "Simple" if i < 6 else "Compound"
        for j in range(3):
            rows.append(
                {
                    "crop_id": f"{label}-{i}-{j}",
                    "observation_id": f"obs-{label}-{i}-{j}",
                    "taxon_name": f"Genus{i} species{i}",
                    "target_label": label,
                }
            )
    return rows


def test_seed_search_finds_class_support_in_train_and_val() -> None:
    report = assign_splits(
        _split_rows(),
        mode="species",
        seed="unit",
        required_classes=("Simple", "Compound"),
        required_min_per_class=2,
    )
    assert report["seed_search"]["satisfied"] is True
    for split in ("train", "val"):
        assert set(report["class_counts"][split]) == {"Simple", "Compound"}
    assert report["isolation"]["ok"]
    assert assign_splits(
        _split_rows(), mode="species", seed="unit", required_classes=("Simple", "Compound")
    )["assignments"] == report["assignments"]


def test_seed_search_reports_impossible_support() -> None:
    rows = [row for row in _split_rows() if row["target_label"] == "Simple"]
    report = assign_splits(
        rows, mode="species", seed="unit", required_classes=("Simple", "Compound"), max_seed_attempts=5
    )
    assert report["seed_search"]["satisfied"] is False
    assert "Compound" in report["seed_search"]["missing"]["val"]


# --- P1.7: VRAM budget + margin ---------------------------------------------
def test_vram_requirement_includes_budget_plus_margin() -> None:
    req = vram_requirement(8.0, 0.30, 0.5)
    assert req["process_budget_gb"] == pytest.approx(2.4)
    assert req["required_free_gb"] == pytest.approx(2.9)
    with pytest.raises(ValueError):
        vram_requirement(8.0, 0.0, 0.5)


def test_device_index_aliases_normalize_to_cuda() -> None:
    from experiments.anatomy4_specialists.device import resolve_device

    assert resolve_device("0") == "cuda:0"
    assert resolve_device("1") == "cuda:1"
    assert resolve_device("cpu") == "cpu"


def test_configure_memory_cpu_does_not_cap() -> None:
    info = configure_memory("cpu", max_vram_fraction=0.3, min_free_vram_gb=0.5)
    assert info["device"] == "cpu"
    assert "VRAM cap not applied" in info["note"]


# --- P1.8: pinned train extra ------------------------------------------------
def test_train_extra_is_pinned() -> None:
    data = tomllib.loads(Path(__file__).resolve().parents[3].joinpath("pyproject.toml").read_text())
    train = data["project"]["optional-dependencies"]["train"]
    assert "ultralytics==8.4.190" in train
    assert any(dep.startswith("pytest") for dep in train)


# --- P1.9: real resume -------------------------------------------------------
def test_resolve_resume_contract(tmp_path: Path) -> None:
    assert resolve_resume(tmp_path / "run", False) is None
    with pytest.raises(FileNotFoundError):
        resolve_resume(tmp_path / "run", True)
    last = tmp_path / "run" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"checkpoint")
    assert resolve_resume(tmp_path / "run", True) == last


# --- P1.10: evaluate defaults from config -----------------------------------
def test_evaluate_parser_defaults_are_config_driven() -> None:
    from experiments.anatomy4_specialists.cli import build_parser

    args = build_parser().parse_args(["evaluate", "--config", "c.json", "--out", "out"])
    assert args.target_precision is None
    assert args.min_coverage is None
    assert args.imgsz is None
    assert args.device is None


def test_evaluate_run_applies_vram_cap(monkeypatch, tmp_path: Path) -> None:
    import experiments.anatomy4_specialists.device as device_module
    import experiments.anatomy4_specialists.evaluate as evaluate_module

    called: dict = {}

    def fake_configure(device, **kwargs):  # noqa: ANN001
        called.update({"device": device, **kwargs})
        return {"device": "cpu", "note": "fake"}

    def stop(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("stop after cap")

    monkeypatch.setattr(device_module, "configure_memory", fake_configure)
    monkeypatch.setattr(evaluate_module, "load_classifier", stop)
    with pytest.raises(RuntimeError, match="stop after cap"):
        evaluate_module.evaluate_run(
            tmp_path / "checkpoint.pt",
            tmp_path / "dataset",
            LEAF_SIMPLE_VS_COMPOUND,
            tmp_path / "out",
            max_vram_fraction=0.31,
            min_free_vram_gb=0.6,
        )
    assert called == {
        "device": "auto",
        "max_vram_fraction": 0.31,
        "min_free_vram_gb": 0.6,
    }


def test_cmd_evaluate_uses_config_abstain_targets(monkeypatch, tmp_path: Path) -> None:
    import experiments.anatomy4_specialists.evaluate as evaluate_module
    from experiments.anatomy4_specialists.cli import build_parser, cmd_evaluate

    config = {
        "target": {
            "target_id": "leaf_simple_vs_compound",
            "expert": "leaf",
            "feature_name": "leaf_type",
            "classes": ["Simple", "Compound"],
            "anatomy_classes": ["leaf"],
        },
        "dataset_root": str(tmp_path / "dataset"),
        "output_dir": str(tmp_path / "runs"),
        "training": {
            "target_precision": 0.91,
            "min_coverage": 0.22,
            "imgsz": 320,
            "batch": 12,
            "device": "cpu",
            "max_vram_fraction": 0.30,
            "min_free_vram_gb": 0.5,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    captured: dict = {}
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_run",
        lambda *args, **kwargs: captured.update(kwargs) or {},
    )
    args = build_parser().parse_args(["evaluate", "--config", str(path), "--out", str(tmp_path / "out")])
    cmd_evaluate(args)
    assert captured["target_precision"] == pytest.approx(0.91)
    assert captured["min_coverage"] == pytest.approx(0.22)
    assert captured["imgsz"] == 320
    assert captured["batch_size"] == 12
    assert captured["device"] == "cpu"
    assert captured["max_vram_fraction"] == pytest.approx(0.30)
    assert captured["min_free_vram_gb"] == pytest.approx(0.5)


# --- P1.11: union wide fallback ---------------------------------------------
def test_claims_table_and_wide_table_are_unioned(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_psql(command, query):  # noqa: ANN001
        calls.append(query)
        if "devo_taxon_claims" in query:
            return [
                {
                    "devo_name": "dirr",
                    "taxon_name": "Quercus rubra",
                    "feature_name": "leaf_type",
                    "value_label": "simple",
                    "provenance": "claims",
                }
            ]
        if "information_schema.columns" in query:
            return [{"column_name": "inflorescences-raceme"}]
        return [{"taxon_name": "Quercus rubra", "inflorescences-raceme": "1"}]

    import experiments.anatomy4_specialists.dirr_join as module

    monkeypatch.setattr(module, "_run_psql", fake_run_psql)
    claims = load_claims_db(features=["leaf_type", "inflorescences"])
    assert {(c.feature_name, c.value_label) for c in claims} == {
        ("leaf_type", "simple"),
        ("inflorescences", "raceme"),
    }
    assert len(calls) == 3


# --- P1.12: explicit claim source -------------------------------------------
def test_join_labels_requires_explicit_claim_source() -> None:
    from experiments.anatomy4_specialists.cli import build_parser

    parser = build_parser()
    base = ["join-labels", "--crops", "c.jsonl", "--target-config", "t.json", "--out", "o.jsonl"]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--db", "--claims-json", "c.json"])
    assert parser.parse_args([*base, "--db"]).db is True


# --- P1.13: contact sheets use dataset split + portable font ----------------
def test_contact_sheet_rows_use_dataset_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.jsonl").write_text(
        json.dumps(
            {
                "crop_id": "c1",
                "observation_id": "o1",
                "taxon_name": "Acer rubrum",
                "image": "https://example.invalid/c1.jpg",
                "crop_path": "/tmp/c1.jpg",
                "anatomy_class": "leaf",
                "detector_confidence": 0.9,
                "observation_split": "test",
                "target_label": "Simple",
            }
        )
        + "\n"
    )
    rows = load_sheet_rows(dataset_root=dataset)
    assert rows[0]["observation_split"] == "test"
    assert _load_font(11) is not None


# --- P1.14: inflorescence only ----------------------------------------------
def test_flower_crop_is_not_inflorescence_evidence() -> None:
    claims = [Claim("Cornus florida", "inflorescences", "Cyme")]
    row = join_weak_labels(
        [_crop("c", "Cornus florida", "flower")], claims, FLOWER_INFLORESCENCE_TYPE
    )["rows"][0]
    assert row["weak_label_reason"] == "anatomy_mismatch:flower"
    assert "target_label" not in row


# --- P1.15: recall semantics + clean observation ids ------------------------
def test_recall_semantics_are_explicit() -> None:
    report = abstention_report(
        [0, 0, 1, 1],
        [[0.9, 0.1], [0.4, 0.6], [0.55, 0.45], [0.1, 0.9]],
        ["Simple", "Compound"],
        confidence_threshold=0.56,
    )
    assert report["recall_semantics"]["all_rows"].startswith("abstained")
    assert report["per_class_recall_covered_only"] == report["per_class_recall_covered"]
    assert report["per_class_recall_all_rows"]["Compound"] == 0.5


def test_observation_id_must_be_trimmed_and_nonempty(tmp_path: Path) -> None:
    row = {
        "crop_id": "c1",
        "observation_id": " obs-1",
        "taxon_name": "Acer rubrum",
        "image": "https://example.invalid/c1.jpg",
        "crop_path": str(tmp_path / "c1.jpg"),
        "anatomy_class": "leaf",
        "detector_confidence": 0.9,
        "observation_split": "train",
        "target_label": "Simple",
    }
    Path(row["crop_path"]).write_bytes(b"x")
    report = validate_dataset([row], LEAF_SIMPLE_VS_COMPOUND, require_labels=True, min_per_class_per_split=0)
    assert not report.ok
    assert any("observation_id" in error for error in report.errors)
