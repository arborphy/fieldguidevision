#!/usr/bin/env python3
# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Self-checks for eda.py: entropy, candidate sets, probe model, and the join.

The math checks are hand-computed on a six-species toy so they need no data.
An optional block validates the generated artifacts when they are present.

Run either way:

    .venv/bin/python experiments/leafsnap-yolo26/test_eda.py
    .venv/bin/python -m pytest experiments/leafsnap-yolo26/test_eda.py -q
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eda  # noqa: E402


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# Six species, uniform prior: H(Y) = log2(6).
SPECIES = [f"s{i}" for i in range(1, 7)]
TOY = {
    "s1": {"A": frozenset({"a1"}), "B": frozenset({"b1"}), "C": frozenset({"c1"}), "E": frozenset({"e1", "e2"})},
    "s2": {"A": frozenset({"a1"}), "B": frozenset({"b2"}), "C": frozenset({"c2"}), "E": frozenset({"e1"})},
    "s3": {"A": frozenset({"a2"}), "B": frozenset({"b2"}), "C": frozenset({"c3"}), "E": frozenset({"e2"})},
    "s4": {"A": frozenset({"a2"}), "B": frozenset({"b2"}), "C": frozenset({"c4"})},
    "s5": {"A": frozenset({"a3"}), "B": frozenset({"b2"}), "C": frozenset({"c5"}), "D": frozenset({"d1"})},
    "s6": {"A": frozenset({"a3"}), "B": frozenset({"b2"}), "C": frozenset({"c6"}), "D": frozenset({"d2"})},
}


def toy_meta(feature: str, values: list[str]) -> dict:
    return {
        "label": feature,
        "section": "TEST",
        "required_organ": "leaf blade",
        "views": ["leaf_obverse"],
        "n_values": len(values),
        "value_columns": [{"column": v, "value_label": v, "definition": ""} for v in values],
    }


def test_entropy() -> None:
    check(approx(eda.entropy([1.0]), 0.0), "one cell has zero entropy")
    check(approx(eda.entropy([1.0, 1.0]), 1.0), "two equal cells = 1 bit")
    check(approx(eda.entropy([1.0, 1.0, 1.0, 1.0]), 2.0), "four equal cells = 2 bits")
    check(approx(eda.entropy([3.0, 1.0]), 0.8112781244591328), "3:1 entropy")
    check(approx(eda.entropy([0.0, 0.0]), 0.0), "zero weight total is 0, not NaN")


def test_information_gain_and_nmi() -> None:
    h_y = math.log2(6.0)
    cells_a = [TOY[name]["A"] for name in SPECIES]
    a = eda.partition_metrics(cells_a, [1.0] * 6)
    check(approx(a["entropy_taxon"], h_y), "H(Y) = log2 6")
    check(approx(a["entropy_feature"], math.log2(3.0)), "H(F) = log2 3")
    check(approx(a["conditional_entropy"], 1.0), "H(Y|A) = log2 2")
    check(approx(a["information_gain"], math.log2(3.0)), "IG(A) = log2 3")
    expected_nmi = math.log2(3.0) / math.sqrt(math.log2(3.0) * h_y)
    check(approx(a["nmi"], expected_nmi), "NMI = IG / sqrt(H(F)H(Y))")

    # IG equals the mean log2(N / cell size) under a uniform prior.
    check(
        approx(a["information_gain"], sum(math.log2(6 / 2) for _ in SPECIES) / 6),
        "IG = mean log2(N/cell)",
    )

    cells_b = [TOY[name]["B"] for name in SPECIES]
    b = eda.partition_metrics(cells_b, [1.0] * 6)
    expected_b = h_y - (5.0 / 6.0) * math.log2(5.0)
    check(approx(b["information_gain"], expected_b), "IG(B) closed form")

    cells_c = [TOY[name]["C"] for name in SPECIES]
    c = eda.partition_metrics(cells_c, [1.0] * 6)
    check(approx(c["information_gain"], h_y), "a unique feature explains all of Y")
    check(approx(c["conditional_entropy"], 0.0), "H(Y|C) = 0")
    check(approx(c["nmi"], 1.0), "NMI = 1 for a perfect feature")


def test_cell_stats_and_coverage() -> None:
    cells_a = [TOY[name]["A"] for name in SPECIES]
    stats = eda.cell_stats(cells_a, [True] * 6, 6)
    check(stats["median_candidate_set_covered"] == 2.0, "median candidate set = 2")
    check(approx(stats["resolution_rate_covered"], 0.0), "no species is alone")
    check(approx(stats["resolution_rate_all"], 0.0), "no species is alone (all)")

    cells_c = [TOY[name]["C"] for name in SPECIES]
    stats_c = eda.cell_stats(cells_c, [True] * 6, 6)
    check(stats_c["median_candidate_set_covered"] == 1.0, "unique cells median 1")
    check(approx(stats_c["resolution_rate_covered"], 1.0), "every species resolved")

    # D covers s5, s6 only; unobserved species fall back to the full space.
    cells_d = [TOY[name].get("D", frozenset()) for name in SPECIES]
    covered = [bool(TOY[name].get("D")) for name in SPECIES]
    stats_d = eda.cell_stats(cells_d, covered, 6)
    check(stats_d["median_candidate_set_covered"] == 1.0, "two singleton D cells")
    check(stats_d["median_candidate_set_all"] == 6.0, "unobserved -> full space (N=6)")
    check(approx(stats_d["resolution_rate_all"], 2 / 6), "only covered singletons resolve")


def test_multi_value_probe_optimism() -> None:
    """The exact cell says unique; a single observed value says otherwise."""
    cells_e = [TOY[name].get("E", frozenset()) for name in SPECIES]
    covered = [bool(TOY[name].get("E")) for name in SPECIES]
    exact = eda.cell_stats(cells_e, covered, 6)
    check(approx(exact["resolution_rate_covered"], 1.0), "exact cells are unique")

    index = eda.build_value_index(TOY, SPECIES, "E")
    probe = eda.probe_stats(cells_e, index, 6)
    check(probe["median_probe_candidate_set_covered"] == 2.0, "probe set = 2")
    check(
        approx(probe["probe_confirmed_unique_rate_covered"], 0.0),
        "no species is uniquely resolved by any single value",
    )

    # Hand-computed single-observation MI. e1 = {s1 (1/2), s2 (1)}, e2 =
    # {s1 (1/2), s3 (1)}, unobserved = {s4, s5, s6}.
    probe = eda.probe_partition_metrics(SPECIES, TOY, ("E",), [1.0] * 6)
    expected = math.log2(6.0) - (
        0.25 * 0.9182958340544894 * 2 + 0.5 * math.log2(3.0)
    )
    check(approx(probe["information_gain"], expected, 1e-9), "probe IG closed form")
    check(approx(expected, 4.0 / 3.0, 1e-9), "probe IG is 4/3 bits")


def test_probe_model_is_monotone() -> None:
    """Joint probe information can never be below either marginal."""
    single = {
        f: eda.probe_partition_metrics(SPECIES, TOY, (f,), [1.0] * 6)["information_gain"]
        for f in ("A", "B", "E")
    }
    for a, b in (("A", "B"), ("A", "E"), ("B", "E")):
        joint = eda.probe_partition_metrics(
            SPECIES, TOY, (a, b), [1.0] * 6
        )["information_gain"]
        check(joint + 1e-9 >= single[a], f"{a}+{b} joint >= IG({a})")
        check(joint + 1e-9 >= single[b], f"{a}+{b} joint >= IG({b})")


def test_evaluate_feature_toy() -> None:
    meta_a = toy_meta("A", ["a1", "a2", "a3"])
    result = eda.evaluate_feature("A", SPECIES, TOY, meta_a)
    check(result["coverage"]["n_species"] == 6, "A covers all six")
    check(approx(result["information_gain"], math.log2(3.0), 1e-6), "exact IG")
    check(approx(result["probe_information_gain"], math.log2(3.0), 1e-6), "probe IG equal when single-valued")
    check(result["multi_valued"]["n_species"] == 0, "A is single-valued")

    meta_e = toy_meta("E", ["e1", "e2"])
    multi = eda.evaluate_feature("E", SPECIES, TOY, meta_e)
    check(multi["coverage"]["n_species"] == 3, "E covers three")
    check(multi["coverage"]["n_uncovered"] == 3, "three unobserved")
    check(multi["probe_information_gain"] < multi["information_gain"], "probe < exact for multi-valued")
    check(approx(multi["probe_information_gain"], 4.0 / 3.0, 1e-6), "probe IG = 4/3 bits")
    check(
        approx(
            multi["probe_conditional_entropy_ratio"],
            multi["probe_conditional_entropy"] / math.log2(6.0),
            1e-6,
        ),
        "probe conditional entropy ratio",
    )


def test_weighted_identity() -> None:
    """Equal weights reproduce the unweighted metrics exactly."""
    cells = [TOY[name]["A"] for name in SPECIES]
    unweighted = eda.partition_metrics(cells, [1.0] * 6)
    weighted = eda.partition_metrics(cells, [0.4] * 6)
    check(approx(unweighted["information_gain"], weighted["information_gain"]), "scale invariance")


def test_edit_distance() -> None:
    check(eda.within_one_edit("pinuspeucea", "pinuspeuce"), "one deletion")
    check(eda.within_one_edit("aesculuspavi", "aesculuspavia"), "one insertion")
    check(not eda.within_one_edit("aesculushippocastamon", "aesculushippocastanum"), "three edits is not one")
    check(not eda.within_one_edit("abc", "abc"), "identical is not an edit")


def test_join_ladder() -> None:
    dirr = [
        "Acer rubrum",
        "Acer xfreemanii",
        "Spiraea alba var. latifolia",
        "Robinia pseudoacacia",
        "Cladrastis kentukea",
        "Aesculus pavia",
        "Broussonetia papyrifera",
        "Acer rubrum var. drummondii",
        "Acer rubrum var. trilobum",
    ]
    leafsnap = [
        "Acer rubrum",
        "Acer freemanii",
        "Spiraea alba",
        "Robinia pseudo-acacia",
        "Cladrastis lutea",
        "Aesculus pavi",
        "Broussonettia papyrifera",
        "Acer rubrum var. drummondii",
        "Imaginary plant",
    ]
    result = eda.resolve_species_names(leafsnap, dirr)
    matches = {m["leafsnap_species"]: m for m in result["matches"]}

    check(matches["Acer rubrum"]["dirr_species"] == "Acer rubrum", "exact wins over varieties")
    check(matches["Acer rubrum"]["rule"] == "exact", "exact rule")
    check(matches["Acer freemanii"]["rule"] == "hybrid_name_form", "hybrid marker")
    check(matches["Spiraea alba"]["rule"] == "infraspecific_parent", "parent species -> variety")
    check(matches["Robinia pseudo-acacia"]["rule"] == "orthographic_variant", "hyphen variant")
    check(matches["Cladrastis lutea"]["dirr_species"] == "Cladrastis kentukea", "curated synonym")
    check(matches["Aesculus pavi"]["dirr_species"] == "Aesculus pavia", "truncated/typo epithet")
    check(matches["Broussonettia papyrifera"]["dirr_species"] == "Broussonetia papyrifera", "one-character typo")
    check(matches["Acer rubrum var. drummondii"]["dirr_species"] == "Acer rubrum var. drummondii", "variety exact")
    check(
        [u["leafsnap_species"] for u in result["unmatched"]] == ["Imaginary plant"],
        "unmatched reported, never guessed",
    )
    check(result["summary"]["n_ambiguous"] == 0, "no ambiguity in this fixture")


def test_join_ambiguity() -> None:
    dirr = ["Acer rubrum var. drummondii", "Acer rubrum var. trilobum"]
    result = eda.resolve_species_names(["Acer rubrum"], dirr)
    check(result["summary"]["n_ambiguous"] == 1, "two varieties for one binomial is ambiguous")
    check(result["summary"]["n_matched"] == 0, "ambiguous is not matched")
    check(result["ambiguous"][0]["candidates"] == dirr, "both candidates reported")


def test_artifacts_if_present() -> None:
    results_path = HERE / "eda_results.json"
    assertions_path = HERE / "feature_assertions.json"
    if not results_path.exists() or not assertions_path.exists():
        return

    results = json.loads(results_path.read_text(encoding="utf-8"))
    assertions = json.loads(assertions_path.read_text(encoding="utf-8"))

    summary = results["name_resolution"]["summary"]
    check(summary["n_matched"] + summary["n_unmatched"] + summary["n_ambiguous"] == 185, "185 accounted for")
    check(
        len(assertions["species"]) == summary["n_matched"],
        "feature_assertions species == matched count",
    )
    empty = [s for s in assertions["species"] if not s["features"]]
    check(
        assertions["meta"]["n_matched_species_with_no_assertions"] == len(empty),
        "empty Dirr rows are counted, not hidden",
    )
    check(
        assertions["meta"]["matched_species_with_no_assertions"]
        == sorted(s["leafsnap_species"] for s in empty),
        "empty-row species listed in the artifact",
    )
    check(
        len(results["pairs"]["all"]) == results["pairs"]["n_eligible"],
        "artifact retains every eligible pair",
    )
    check(
        len(results["triples"]["all"]) == results["triples"]["n_evaluated"],
        "artifact retains every evaluated triple",
    )
    pairs = results["pairs"]["all"]
    seen = set()
    for pair in pairs:
        key = (pair["feature_a"], pair["feature_b"])
        if key in seen:
            continue
        seen.add(key)
        check(
            pair["joint_probe_ig"] + 1e-9 >= max(pair["probe_ig_a"], pair["probe_ig_b"]),
            f"joint probe >= marginals for {key}",
        )
        check(pair["probe_synergy"] >= -1e-9, f"probe synergy non-negative for {key}")
        check(0.0 <= pair["joint_nmi"] <= 1.0 + 1e-9, f"NMI in [0,1] for {key}")
    check(results["triples"]["n_fully_non_local"] >= 0, "triple counter present")


def main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} self-checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
