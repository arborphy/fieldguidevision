#!/usr/bin/env python3
# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Checks for the E3 organ-gate proof run and its evaluator.

Data/math checks need no training artifacts; the artifact block is skipped when
``scratch/leafsnap-yolo26/organ-gate`` has not been produced yet.

    .venv/bin/python experiments/leafsnap-yolo26/test_organ_gate.py
    .venv/bin/python -m pytest experiments/leafsnap-yolo26/test_organ_gate.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluator as ev  # noqa: E402

ROOT = HERE.parents[1]
DETECTOR = ROOT / "scratch" / "leafsnap-yolo26" / "detector"
OUT = ROOT / "scratch" / "leafsnap-yolo26" / "organ-gate"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_evaluator_self_checks() -> None:
    ev._self_test()


def test_manifest_matches_counts() -> None:
    rows = ev.load_manifest(DETECTOR / "manifest.jsonl")
    counts = json.loads((DETECTOR / "counts.json").read_text())
    check(len(rows) == counts["total_images"], "manifest row count")
    check(len({r["uid"] for r in rows}) == len(rows), "uids unique")

    by_split_class: dict[str, dict[str, int]] = {}
    for r in rows:
        by_split_class.setdefault(r["split"], {}).setdefault(r["class_name"], 0)
        by_split_class[r["split"]][r["class_name"]] += 1
    for split, expected in counts["by_split_class"].items():
        got = by_split_class.get(split, {})
        for cls, n in expected.items():
            check(got.get(cls, 0) == n, f"{split}/{cls}: {got.get(cls, 0)} != {n}")

    leaf = [r for r in rows if r["source"] == "leafsnap" and r["split"] == "test"]
    check(all(r["domain"] == "field" for r in leaf), "LeafSnap test is field-only")
    check(all(r["weak_box"] is False for r in leaf), "LeafSnap boxes are not weak")
    check(
        all(r["weak_box"] for r in rows if r["source"] == "bioimages"),
        "BioImages boxes are weak by construction",
    )


def test_gate_definition_is_leaf_primary() -> None:
    # The gate must be a conjunction over leaf metrics only, with nonleaf
    # recall explicitly secondary.
    from train_organ_gate import GATE_LEAF_MAP50, GATE_LEAF_RECALL, STANDARD_OPERATING_CONF

    check(GATE_LEAF_RECALL == 0.80 and GATE_LEAF_MAP50 == 0.80, "pre-registered thresholds")
    check(STANDARD_OPERATING_CONF == 0.25, "standard operating point is the library default")


def test_artifacts_consistent() -> None:
    if not (OUT / "metrics.json").exists():
        print("artifacts missing; skipping artifact checks")
        return
    metrics = json.loads((OUT / "metrics.json").read_text())
    gate = metrics["e3"]
    checks = gate["checks"]
    expected = all(c["pass"] for c in checks.values())
    check(gate["status"] == ("PASS" if expected else "FAIL"), "gate status matches its checks")
    check(set(checks) == {"leaf_map50_ge_0.80", "leafsnap_field_leaf_recall_ge_0.80"}, "leaf-only gate")

    counts = json.loads((DETECTOR / "counts.json").read_text())
    for cls, row in metrics["standard_test_val"]["per_class"].items():
        check(row["ap50"] <= 1.0 + 1e-9, f"{cls} mAP50 <= 1")
        check(row["ap50_95"] <= 1.0 + 1e-9, f"{cls} mAP50-95 <= 1")
        check(row["n_gt"] == counts["by_split_class"]["test"][cls], f"{cls} GT count")

    custom = metrics["source_split"]["test_at_val_selected_conf"]
    check(custom["per_class"]["leaf"]["ap50"] <= 1.0 + 1e-9, "leaf AP <= 1")
    check(custom["sources"]["leafsnap_field_leaf"]["n_images"] == counts["by_split_source"]["test"]["leafsnap"], "leafsnap test count")
    check(custom["sources"]["bioimages_weak_box"]["n_images"] == counts["by_split_source"]["test"]["bioimages"], "bioimages test count")

    train = metrics["training"]
    history = json.loads((OUT / "history.json").read_text())
    check(len(history) == train["epochs_run"], "history length == epochs run")
    check((OUT / "best.pt").exists(), "best checkpoint saved")

    labels = json.loads((OUT / "organ_gate.labels.json").read_text())
    check(labels["class_order"] == {str(i): n for i, n in enumerate(ev.CLASS_NAMES)}, "label class order")
    check(labels["onnx"]["opset"][0]["version"] == 17, "opset 17")
    check(labels["onnx"]["dynamic"] is False, "static graph")
    check(labels["input"]["shape"] == [1, 3, 320, 320], "static 320 input, batch 1")
    out_shape = labels["output"]["shape"]
    check(out_shape[0] == 1 and out_shape[1] == 4 + len(ev.CLASS_NAMES), "raw [1, 4+nc, anchors] output")
    check((OUT / "organ_gate.onnx").exists(), "onnx exported")

    onnx_val = json.loads((OUT / "onnx_test_val.json").read_text())
    check(
        abs(onnx_val["per_class"]["leaf"]["ap50"] - metrics["standard_test_val"]["per_class"]["leaf"]["ap50"]) < 0.01,
        "ONNX leaf mAP50 within 1 point of PyTorch",
    )


if __name__ == "__main__":
    test_evaluator_self_checks()
    test_manifest_matches_counts()
    test_gate_definition_is_leaf_primary()
    test_artifacts_consistent()
    print("organ-gate checks: OK")
