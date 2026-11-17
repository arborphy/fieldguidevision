# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.anatomy4_specialists.contracts import (
    FLOWER_INFLORESCENCE_TYPE,
    LEAF_SIMPLE_VS_COMPOUND,
    canonical_binomial,
    canonical_label,
)
from experiments.anatomy4_specialists.dirr_join import (
    Claim,
    ClaimsIndex,
    _claims_from_payload,
    join_weak_labels,
    load_claims,
    load_claims_db,
)


def _crop(crop_id: str, taxon: str, anatomy: str, **extra):
    return {
        "crop_id": crop_id,
        "observation_id": f"obs-{crop_id}",
        "taxon_name": taxon,
        "anatomy_class": anatomy,
        "crop_path": f"/tmp/{crop_id}.jpg",
        "detector_confidence": 0.9,
        "observation_split": "train",
        **extra,
    }


CLAIMS = [
    Claim("Quercus rubra L.", "leaf_type", "Simple", provenance="devo_taxa"),
    Claim("Acer negundo", "leaf_type", "Compound"),
    Claim("Cornus florida", "inflorescences", "Cyme"),
    Claim("Cercis canadensis", "inflorescences", "Raceme"),
    Claim("Ambiguous tree", "inflorescences", "Raceme"),
    Claim("Ambiguous tree", "inflorescences", "Corymb"),
    Claim("Mixed leaf", "leaf_type", "Simple"),
    Claim("Mixed leaf", "leaf_type", "Compound"),
    Claim("Needle pine", "leaf_type", "Needlelike"),
]


def test_canonical_helpers() -> None:
    assert canonical_binomial("Quercus rubra L.") == "quercus rubra"
    assert canonical_binomial("Quercus rubra (L.) Pursh") == "quercus rubra"
    assert canonical_binomial("Quercus rubra var. borealis") == "quercus rubra"
    assert canonical_binomial("Quercus") is None
    assert canonical_binomial("Something sp.") is None
    assert canonical_binomial("Abelia × grandiflora") == "abelia grandiflora"
    assert canonical_binomial("×Chitalpa tashkentensis") == "chitalpa tashkentensis"
    assert canonical_binomial("Abelia xgrandiflora") == "abelia xgrandiflora"
    assert canonical_label("simple_leaf", LEAF_SIMPLE_VS_COMPOUND) == "Simple"
    assert canonical_label("CORYMB", FLOWER_INFLORESCENCE_TYPE) == "Corymb"
    assert canonical_label("Needlelike", LEAF_SIMPLE_VS_COMPOUND) is None


def test_leaf_claim_only_attaches_to_leaf_crop() -> None:
    joined = join_weak_labels(
        [
            _crop("leaf-q", "Quercus rubra", "leaf"),
            _crop("flower-q", "Quercus rubra", "flower"),
            _crop("needle-q", "Quercus rubra", "needle"),
        ],
        CLAIMS,
        LEAF_SIMPLE_VS_COMPOUND,
    )["rows"]
    assert joined[0]["weak_label_eligible"] is True
    assert joined[0]["target_label"] == "Simple"
    assert joined[1]["weak_label_eligible"] is False
    assert joined[1]["weak_label_reason"] == "anatomy_mismatch:flower"
    assert "target_label" not in joined[1]
    assert joined[2]["weak_label_reason"] == "anatomy_mismatch:needle"


def test_inflorescence_claim_attaches_only_to_inflorescence_crops() -> None:
    joined = join_weak_labels(
        [
            _crop("flower-c", "Cornus florida", "flower"),
            _crop("inflor-c", "Cornus florida", "inflorescence"),
            _crop("leaf-c", "Cornus florida", "leaf"),
        ],
        CLAIMS,
        FLOWER_INFLORESCENCE_TYPE,
    )["rows"]
    # An isolated flower does not carry arrangement evidence.
    assert joined[0]["weak_label_reason"] == "anatomy_mismatch:flower"
    assert "target_label" not in joined[0]
    assert joined[1]["target_label"] == "Cyme"
    assert joined[2]["weak_label_reason"] == "anatomy_mismatch:leaf"
    assert "target_label" not in joined[2]


def test_leaf_claim_is_not_used_on_flower_crop_even_if_flower_target_absent() -> None:
    joined = join_weak_labels(
        [_crop("flower-c", "Quercus rubra", "flower")],
        CLAIMS,
        FLOWER_INFLORESCENCE_TYPE,
    )["rows"]
    assert joined[0]["weak_label_eligible"] is False
    assert joined[0]["claim_values"] == []
    assert "target_label" not in joined[0]


def test_ambiguous_and_mixed_claims_are_excluded() -> None:
    rows = [
        _crop("amb", "Ambiguous tree", "inflorescence"),
        _crop("mixed", "Mixed leaf", "leaf"),
        _crop("outside", "Needle pine", "leaf"),
        _crop("missing", "Unknown sylvatica", "leaf"),
    ]
    flower = join_weak_labels(rows[:1], CLAIMS, FLOWER_INFLORESCENCE_TYPE)["rows"][0]
    leaf = join_weak_labels(rows[1:2], CLAIMS, LEAF_SIMPLE_VS_COMPOUND)["rows"][0]
    outside = join_weak_labels(rows[2:3], CLAIMS, LEAF_SIMPLE_VS_COMPOUND)["rows"][0]
    missing = join_weak_labels(rows[3:4], CLAIMS, LEAF_SIMPLE_VS_COMPOUND)["rows"][0]
    assert flower["weak_label_reason"] == "ambiguous_claim"
    assert leaf["weak_label_reason"] == "ambiguous_claim"
    assert outside["weak_label_reason"] == "claim_outside_target"
    assert missing["weak_label_reason"] == "no_claim"
    for row in (flower, leaf, outside, missing):
        assert "target_label" not in row


def test_audit_counts_and_anatomy_gate() -> None:
    joined = join_weak_labels(
        [
            _crop("a", "Quercus rubra", "leaf"),
            _crop("b", "Quercus rubra", "flower"),
            _crop("c", "Acer negundo", "leaf"),
        ],
        CLAIMS,
        LEAF_SIMPLE_VS_COMPOUND,
    )
    audit = joined["audit"]
    assert audit["rows_eligible"] == 2
    assert audit["rows_ineligible"] == 1
    assert audit["eligible_by_label"] == {"Compound": 1, "Simple": 1}
    assert audit["anatomy_gate"] == {"eligible": 2, "rejected": 1}


def test_claims_index_skips_non_species_rows() -> None:
    index = ClaimsIndex([Claim("Quercus", "leaf_type", "Simple"), Claim("Quercus rubra", "leaf_type", "Simple")])
    assert index.n_claims == 1
    assert index.n_skipped == 1


def test_load_claims_json_shapes(tmp_path: Path) -> None:
    payload = {"claims": [{"taxon_name": "Quercus rubra", "feature_name": "leaf_type", "value_label": "Simple"}]}
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(payload))
    claims = load_claims(claims_json=path)
    assert len(claims) == 1 and claims[0].species_key == "quercus rubra"
    assert _claims_from_payload([{"taxon_name": "Acer negundo", "feature": "leaf_type", "value": "Compound"}])[0].value_label == "Compound"


def test_wide_devo_taxa_fallback(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_psql(command, query):  # noqa: ANN001
        calls.append(query)
        if "devo_taxon_claims" in query:
            return []
        if "information_schema.columns" in query:
            return [
                {"column_name": "leaf_type-simple"},
                {"column_name": "leaf_type-compound"},
                {"column_name": "inflorescences-raceme"},
            ]
        return [
            {"taxon_name": "Quercus rubra", "leaf_type-simple": "1", "leaf_type-compound": "0", "inflorescences-raceme": "0"},
            {"taxon_name": "Acer negundo", "leaf_type-simple": "0", "leaf_type-compound": "1", "inflorescences-raceme": "1"},
        ]

    import experiments.anatomy4_specialists.dirr_join as module

    monkeypatch.setattr(module, "_run_psql", fake_run_psql)
    claims = load_claims_db(features=["leaf_type", "inflorescences"])
    pairs = {(c.taxon_name, c.feature_name, c.value_label) for c in claims}
    assert ("Quercus rubra", "leaf_type", "simple") in pairs
    assert ("Acer negundo", "leaf_type", "compound") in pairs
    assert ("Acer negundo", "inflorescences", "raceme") in pairs
    assert len(calls) == 3


def test_anatomy_gate_can_be_disabled_only_explicitly() -> None:
    joined = join_weak_labels(
        [_crop("flower-q", "Quercus rubra", "flower")],
        CLAIMS,
        LEAF_SIMPLE_VS_COMPOUND,
        require_anatomy_match=False,
    )["rows"]
    assert joined[0]["target_label"] == "Simple"
    assert joined[0]["anatomy_matches_claim"] is False
