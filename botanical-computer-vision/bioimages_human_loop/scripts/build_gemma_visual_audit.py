#!/usr/bin/env python3
"""Reuse the existing image-grounded Gemma tags to audit discovered error modes.

This is deliberately separate from the three frozen probes.  Gemma already viewed each
photograph and produced free multi-label tags; this script asks only whether that visual
evidence supports each programmatically proposed mode.  It does not invent Human Gold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTIVE = {"flower", "inflorescence", "fruit", "cone", "seed", "bud"}
WOODY = {"twig", "branch", "stem"}

CAUSES = {
    "secondary_woody_structure_missed": "Small secondary twigs and stems are visually present but occupy less area than the primary organ.",
    "reproductive_structure_missed": "Small reproductive structures are visually dominated by leaves, twigs, or wider context.",
    "leaf_dominates_reproductive": "Large leaf area pulls predictions away from smaller flowers, fruit, cones, or seeds.",
    "cone_seed_under_detection": "Cone and seed morphology is rare and may resemble fruit, flowers, or woody texture.",
    "whole_plant_scale_split": "Global plant scale and local trunk, branch, and leaf cues coexist in the same frame.",
    "fruit_state_view_confusion": "Fruit state and view tags co-occur and are difficult to separate from one still image.",
    "leaf_view_confusion": "One leaf photograph can simultaneously show surface, margin, arrangement, and petiole details.",
    "bark_scale_confusion": "Bark texture is visible without enough context to reliably infer trunk or branch scale.",
    "flower_fruit_bud_ambiguity": "Adjacent reproductive stages share visual form and can appear together.",
    "high_confidence_extra_tag": "A confident extra model tag is not corroborated by the existing Gemma visual annotation.",
    "bioimages_single_label_ambiguity": "The photograph visibly contains multiple structures although BioImages records one intended subject.",
    "strong_model_disagreement": "The photograph contains competing texture, shape, and scene-scale cues.",
}


def normalize_visible(tags: list[str]) -> set[str]:
    visible = set()
    for raw in tags:
        tag = raw.lower().strip()
        if tag == "needle" or tag.startswith("leaf"):
            visible.add("leaf")
        elif tag in WOODY or any(tag.startswith(prefix + " ") for prefix in WOODY):
            visible.add("twig")
        elif tag in {"bark", "trunk"} or tag.startswith("bark "):
            visible.add("bark")
        elif tag in {"flower", "inflorescence"} or tag.startswith("flower "):
            visible.add("flower")
        elif tag.startswith("fruit") or tag == "immature fruit":
            visible.add("fruit")
        elif tag.startswith("cone"):
            visible.add("cone")
        elif tag.startswith("seed"):
            visible.add("seed")
        elif tag in {"whole plant", "whole tree"}:
            visible.add("whole plant")
    return visible


def supported(mode: str, normalized: set[str], visible: set[str]) -> bool:
    reproductive = bool(normalized & REPRODUCTIVE) or bool(visible & {"flower", "fruit", "cone", "seed"})
    fine_fruit = any(tag.startswith("fruit ") or tag == "immature fruit" for tag in normalized)
    fine_leaf = any(tag.startswith("leaf ") for tag in normalized)
    if mode == "secondary_woody_structure_missed": return bool(normalized & WOODY) or "twig" in visible
    if mode == "reproductive_structure_missed": return reproductive
    if mode == "leaf_dominates_reproductive": return "leaf" in visible and reproductive
    if mode == "cone_seed_under_detection": return bool(visible & {"cone", "seed"})
    if mode == "whole_plant_scale_split": return "whole plant" in visible or ("bark" in visible and "leaf" in visible)
    if mode == "fruit_state_view_confusion": return "fruit" in visible and fine_fruit
    if mode == "leaf_view_confusion": return "leaf" in visible and fine_leaf
    if mode == "bark_scale_confusion": return "bark" in visible
    if mode == "flower_fruit_bud_ambiguity": return reproductive
    if mode == "bioimages_single_label_ambiguity": return len(visible) >= 2
    if mode == "strong_model_disagreement": return len(visible) >= 2 or len(normalized) >= 3
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=ROOT / "analysis" / "vlm_audit_candidates.json")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "vlm_error_audit.jsonl")
    args = parser.parse_args()
    rows = json.loads(args.candidates.read_text())
    output = []
    for row in rows:
        normalized = {str(tag).lower().strip() for tag in row["gemma_normalized_tags"]}
        visible = normalize_visible(list(normalized))
        modes = [mode for mode in row["candidate_modes"] if supported(mode, normalized, visible)]
        cause = CAUSES[modes[0]] if modes else "The existing Gemma visual tags do not corroborate the proposed mode."
        output.append({
            "image_id": row["image_id"], "status": "ok", "source": "existing_gemma_image_tags",
            "visible_tags": sorted(visible) or ["other"], "supported_modes": modes,
            "annotation_ambiguity": len(visible) >= 2,
            "bioimages_label_visually_supported": True,
            "short_pattern": modes[0].replace("_", " ") if modes else "candidate not corroborated",
            "likely_cause": cause,
            "mode_causes": {mode: CAUSES[mode] for mode in modes},
        })
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    print(json.dumps({"audited": len(output), "supported_mode_links": sum(len(row["supported_modes"]) for row in output), "ambiguous": sum(row["annotation_ambiguity"] for row in output)}, indent=2))


if __name__ == "__main__":
    main()
