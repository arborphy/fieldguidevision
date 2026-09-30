#!/usr/bin/env python3
"""Discover recurring multi-label error modes and build a diverse human set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("dinov3", "bioclip", "efficientnet_b0")
MODEL_NAMES = {
    "dinov3": "DINOv3",
    "bioclip": "BioCLIP 2.5",
    "efficientnet_b0": "EfficientNet-B0",
}
STRUCTURES = {
    "whole plant", "trunk", "bark", "branch", "stem", "twig", "bud",
    "leaf", "needle", "flower", "inflorescence", "fruit", "cone", "seed",
}
REPRODUCTIVE = {"flower", "inflorescence", "fruit", "cone", "seed"}
FRUIT_VIEWS = {"immature fruit", "fruit borne on plant", "fruit close-up", "opened or sectioned fruit"}
LEAF_VIEWS = {"leaf upper surface", "leaf margin", "leaf arrangement on twig", "petiole arrangement", "needle attachment"}
BARK_VIEWS = {"large-tree bark", "medium-tree or large-branch bark", "small-tree or branch bark"}
HUMAN_TAGS = ("leaf", "twig", "bark", "flower", "fruit", "cone", "seed", "whole plant", "other")


MODE_DEFINITIONS = (
    {
        "id": "secondary_woody_structure_missed",
        "name": "Secondary woody structures are missed",
        "description": "Twig, branch, or stem is visible with the main subject but at least two models omit it.",
        "cause": "Small woody structures occupy less area than the primary organ and are weakly represented by single-subject training labels.",
    },
    {
        "id": "reproductive_structure_missed",
        "name": "Reproductive structures disappear in context",
        "description": "Flower, fruit, cone, seed, or inflorescence is present but at least two models miss it.",
        "cause": "Small reproductive structures are visually dominated by leaves, twigs, or the wider plant view.",
    },
    {
        "id": "leaf_dominates_reproductive",
        "name": "Leaf dominates flower or fruit",
        "description": "A reproductive structure is present, yet one or more models lead with leaf and omit reproductive tags.",
        "cause": "Large high-contrast foliage overwhelms smaller flowers or fruits in the pooled image representation.",
    },
    {
        "id": "cone_seed_under_detection",
        "name": "Cone and seed under-detection",
        "description": "Cone or seed is supported by BioImages or Gemma but is absent from one or more model tag sets.",
        "cause": "These are rare classes and often resemble fruit, flowers, buds, or background texture.",
    },
    {
        "id": "whole_plant_scale_split",
        "name": "Whole-plant views split into trunk, branch, or leaf",
        "description": "Models disagree on scene scale, especially between whole plant and its dominant local structure.",
        "cause": "Global and local cues coexist; different backbones emphasize canopy texture, trunk geometry, or individual leaves.",
    },
    {
        "id": "fruit_state_view_confusion",
        "name": "Fruit state and view tags are unstable",
        "description": "Models disagree about immature, attached, close-up, or opened fruit views.",
        "cause": "Fine fruit states have limited support and can co-occur, making a single view label visually ambiguous.",
    },
    {
        "id": "leaf_view_confusion",
        "name": "Leaf view tags are over-interpreted",
        "description": "Upper surface, margin, arrangement, and petiole view tags are missed or added inconsistently.",
        "cause": "The same photograph can legitimately show several leaf details, while BioImages names only one primary view.",
    },
    {
        "id": "bark_scale_confusion",
        "name": "Bark scale is confused",
        "description": "Large-trunk, medium-branch, and small-branch bark views are interchanged or missed.",
        "cause": "Crop scale and lack of an absolute size reference make neighboring bark categories difficult to separate.",
    },
    {
        "id": "flower_fruit_bud_ambiguity",
        "name": "Flower, fruit, and bud stages overlap",
        "description": "Predictions cross between flower, inflorescence, bud, immature fruit, and fruit.",
        "cause": "Closely related developmental stages share shape and color, and several may be visible together.",
    },
    {
        "id": "high_confidence_extra_tag",
        "name": "High-confidence extra tags",
        "description": "A model adds a tag above 0.85 confidence that Gemma does not include.",
        "cause": "Some are genuine false positives; others expose incomplete model-generated or canonical annotations.",
    },
    {
        "id": "bioimages_single_label_ambiguity",
        "name": "BioImages single-label conflicts may be false alarms",
        "description": "Models agree well with Gemma but miss the one BioImages canonical tag.",
        "cause": "The canonical label describes the intended subject, while other structures can dominate what is visibly present.",
    },
    {
        "id": "strong_model_disagreement",
        "name": "Backbones make fundamentally different decisions",
        "description": "The three model tag sets have low overlap or disagree on their top tag.",
        "cause": "DINOv3, BioCLIP, and EfficientNet emphasize different morphology, semantic, and texture cues.",
    },
)


def load_site_data(path: Path) -> dict:
    raw = path.read_text()
    return json.loads(raw.removeprefix("window.BIOIMAGES_DATA=").removesuffix(";\n"))


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def prediction_sets(image: dict) -> tuple[dict[str, set[str]], dict[str, dict[str, float]]]:
    sets = {}
    scores = {}
    for model in MODELS:
        tags = image["tagging"]["models"][model]["tags"]
        sets[model] = {item["tag"] for item in tags}
        scores[model] = {item["tag"]: float(item["confidence"]) for item in tags}
    return sets, scores


def signals_for(image: dict) -> dict:
    tagging = image["tagging"]
    reference = set(tagging["gemma_normalized_tags"])
    canonical = tagging.get("bioimages_canonical_tag")
    sets, scores = prediction_sets(image)
    tops = {model: tagging["models"][model]["tags"][0]["tag"] for model in MODELS}
    misses = {tag: [model for model in MODELS if tag not in sets[model]] for tag in reference}
    false_positive = {
        model: {tag: score for tag, score in scores[model].items() if tag not in reference}
        for model in MODELS
    }
    secondary_woody = {
        tag for tag in reference & {"twig", "branch", "stem"}
        if tag != canonical and len(misses[tag]) >= 2
    }
    reproductive_missed = {tag for tag in reference & REPRODUCTIVE if len(misses[tag]) >= 2}
    leaf_dominates = [
        model for model in MODELS
        if tops[model] in LEAF_VIEWS | {"leaf"}
        and reference & REPRODUCTIVE
        and not (sets[model] & REPRODUCTIVE)
    ]
    rare_reference = bool((reference | ({canonical} if canonical else set())) & {"cone", "seed"})
    rare_missing_models = [
        model for model in MODELS
        if any(tag not in sets[model] for tag in (reference | ({canonical} if canonical else set())) & {"cone", "seed"})
    ]
    whole_context = "whole plant" in reference or canonical == "whole plant"
    scale_models = [
        model for model in MODELS
        if whole_context and ("whole plant" not in sets[model] or tops[model] in {"trunk", "branch", "twig", "leaf", "bark"})
    ]
    fruit_view_models = [
        model for model in MODELS
        if any((tag in reference) != (tag in sets[model]) for tag in FRUIT_VIEWS)
    ] if reference & (FRUIT_VIEWS | {"fruit"}) else []
    leaf_view_models = [
        model for model in MODELS
        if any((tag in reference) != (tag in sets[model]) for tag in LEAF_VIEWS)
    ] if reference & (LEAF_VIEWS | {"leaf", "needle"}) else []
    bark_view_models = [
        model for model in MODELS
        if any((tag in reference) != (tag in sets[model]) for tag in BARK_VIEWS)
    ] if reference & (BARK_VIEWS | {"bark"}) else []
    developmental = {"flower", "inflorescence", "bud", "fruit", "immature fruit"}
    developmental_models = [
        model for model in MODELS
        if (sets[model] & developmental) != (reference & developmental)
    ] if reference & developmental else []
    high_conf = {
        model: {tag: score for tag, score in false_positive[model].items() if score >= 0.85}
        for model in MODELS
    }
    good_gemma_models = [model for model in MODELS if tagging["models"][model]["gemma_f1"] >= 0.67]
    bio_miss_models = [model for model in MODELS if tagging["models"][model]["bioimages_match"] is False]
    bio_ambiguity_models = sorted(set(good_gemma_models) & set(bio_miss_models))
    pairwise = [jaccard(sets[a], sets[b]) for a, b in ((MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]), (MODELS[1], MODELS[2]))]
    strong_disagreement = min(pairwise) < 0.25 or len(set(tops.values())) == 3

    modes: dict[str, list[str]] = {}
    if secondary_woody:
        modes["secondary_woody_structure_missed"] = sorted({m for tag in secondary_woody for m in misses[tag]})
    if reproductive_missed:
        modes["reproductive_structure_missed"] = sorted({m for tag in reproductive_missed for m in misses[tag]})
    if leaf_dominates:
        modes["leaf_dominates_reproductive"] = leaf_dominates
    if rare_reference and rare_missing_models:
        modes["cone_seed_under_detection"] = rare_missing_models
    if scale_models:
        modes["whole_plant_scale_split"] = scale_models
    if len(fruit_view_models) >= 2:
        modes["fruit_state_view_confusion"] = fruit_view_models
    if len(leaf_view_models) >= 2:
        modes["leaf_view_confusion"] = leaf_view_models
    if len(bark_view_models) >= 2:
        modes["bark_scale_confusion"] = bark_view_models
    if len(developmental_models) >= 2 and any(sets[m] & developmental != reference & developmental for m in developmental_models):
        modes["flower_fruit_bud_ambiguity"] = developmental_models
    high_models = [model for model in MODELS if high_conf[model]]
    if high_models:
        modes["high_confidence_extra_tag"] = high_models
    if bio_ambiguity_models:
        modes["bioimages_single_label_ambiguity"] = bio_ambiguity_models
    if strong_disagreement:
        modes["strong_model_disagreement"] = list(MODELS)

    return {
        "modes": modes,
        "reference": sorted(reference),
        "canonical": canonical,
        "tops": tops,
        "misses": misses,
        "high_confidence_extra": high_conf,
        "bio_mismatch_models": bio_miss_models,
        "gemma_disagreement_models": [m for m in MODELS if tagging["models"][m]["gemma_jaccard"] < 0.33],
        "mean_pairwise_jaccard": float(tagging["mean_pairwise_jaccard"]),
    }


def diverse_pick(candidates: list[dict], limit: int, seed: str) -> list[dict]:
    chosen = []
    seen_species = Counter()
    seen_individuals = Counter()
    remaining = list(candidates)
    while remaining and len(chosen) < limit:
        def rank(item: dict) -> tuple:
            image = item["image"]
            audit_bonus = 3 if item.get("vlm_support") else 0
            severity = item.get("severity", 0)
            diversity = -(seen_species[image["species"]] * 2 + seen_individuals[image["individual_id"]] * 5)
            return audit_bonus + severity + diversity, stable(seed + "|" + image["id"])
        best = max(remaining, key=rank)
        remaining.remove(best)
        chosen.append(best)
        seen_species[best["image"]["species"]] += 1
        seen_individuals[best["image"]["individual_id"]] += 1
    return chosen


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def human_categories(image: dict, signal: dict) -> set[str]:
    categories = set(signal["modes"])
    if signal["bio_mismatch_models"]:
        categories.add("bioimages_mismatch")
    if signal["gemma_disagreement_models"]:
        categories.add("gemma_disagreement")
    if image["tagging"]["model_disagreement"]:
        categories.add("model_disagreement")
    if any(signal["high_confidence_extra"].values()):
        categories.add("high_confidence_error")
    reference = set(signal["reference"])
    if reference & {"cone", "seed", "needle"}:
        categories.add("rare_class")
    if (
        not signal["bio_mismatch_models"]
        and not signal["gemma_disagreement_models"]
        and not image["tagging"]["model_disagreement"]
        and all(image["tagging"]["models"][m]["gemma_f1"] >= 0.8 for m in MODELS)
    ):
        categories.add("easy_case")
    return categories


def build_human_set(images: list[dict], signals: dict[str, dict], top_mode_ids: list[str]) -> list[dict]:
    quotas = {
        **{f"mode:{mode}": 7 for mode in top_mode_ids[:8]},
        "bioimages_mismatch": 20,
        "gemma_disagreement": 20,
        "model_disagreement": 20,
        "high_confidence_error": 15,
        "rare_class": 15,
        "easy_case": 12,
        **{f"tag:{tag}": value for tag, value in {
            "leaf": 12, "twig": 10, "bark": 8, "flower": 10, "fruit": 10,
            "cone": 8, "seed": 8, "whole plant": 8, "other": 4,
        }.items()},
    }
    records = []
    for image in images:
        signal = signals[image["id"]]
        categories = human_categories(image, signal)
        coverage = set(categories)
        coverage.update(f"mode:{mode}" for mode in signal["modes"])
        reference = set(signal["reference"])
        canonical = signal["canonical"]
        for tag in HUMAN_TAGS[:-1]:
            if tag in reference or canonical == tag:
                coverage.add(f"tag:{tag}")
        if not any(f"tag:{tag}" in coverage for tag in HUMAN_TAGS[:-1]):
            coverage.add("tag:other")
        records.append({"image": image, "signal": signal, "coverage": coverage})

    chosen = []
    counts = Counter()
    species = Counter()
    individuals = Counter()
    remaining = records[:]
    while remaining and len(chosen) < 100:
        def score(record: dict) -> tuple:
            gain = sum(max(0, quotas[key] - counts[key]) / quotas[key] for key in record["coverage"] if key in quotas)
            image = record["image"]
            diversity = 2.5 * (species[image["species"]] == 0) + 1.5 * (individuals[image["individual_id"]] == 0)
            penalty = species[image["species"]] * 0.18 + individuals[image["individual_id"]] * 0.8
            return gain + diversity - penalty, stable("human-set-v2|" + image["id"])
        best = max(remaining, key=score)
        remaining.remove(best)
        chosen.append(best)
        counts.update(best["coverage"])
        species[best["image"]["species"]] += 1
        individuals[best["image"]["individual_id"]] += 1

    output = []
    for index, record in enumerate(chosen, 1):
        image = record["image"]
        relevant = sorted(key for key in record["coverage"] if key in quotas)
        reasons = []
        for key in relevant:
            if key.startswith("mode:"):
                reasons.append("covers " + key.removeprefix("mode:").replace("_", " "))
            elif key.startswith("tag:"):
                reasons.append("covers tag " + key.removeprefix("tag:"))
            else:
                reasons.append(key.replace("_", " "))
        output.append({
            "review_index": index,
            "image_id": image["id"],
            "species": image["species"],
            "individual_id": image["individual_id"],
            "selection_categories": relevant,
            "selection_reason": "; ".join(reasons),
            "error_modes": sorted(record["signal"]["modes"]),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-data", type=Path, default=ROOT / "data" / "site-data.js")
    parser.add_argument("--vlm-audit", type=Path, default=ROOT / "analysis" / "vlm_error_audit.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()

    data = load_site_data(args.site_data)
    images = data["images"]
    by_id = {image["id"]: image for image in images}
    signals = {image["id"]: signals_for(image) for image in images}

    vlm = {}
    if args.vlm_audit.exists():
        for line in args.vlm_audit.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    vlm[row["image_id"]] = row

    mode_outputs = []
    member_rows = []
    for definition in MODE_DEFINITIONS:
        mode_id = definition["id"]
        members = []
        model_counts = Counter()
        for image in images:
            involved = signals[image["id"]]["modes"].get(mode_id)
            if not involved:
                continue
            audit = vlm.get(image["id"], {})
            support = mode_id in audit.get("supported_modes", [])
            severity = len(involved) + len(signals[image["id"]]["gemma_disagreement_models"]) + len(signals[image["id"]]["bio_mismatch_models"])
            members.append({"image": image, "models": involved, "severity": severity, "vlm_support": support, "audit": audit})
            model_counts.update(involved)
            member_rows.append({
                "mode_id": mode_id, "image_id": image["id"], "species": image["species"],
                "individual_id": image["individual_id"], "models": ";".join(involved),
                "vlm_reviewed": bool(audit), "vlm_supported": support,
            })
        representatives = diverse_pick(members, 12, "mode-representative-v2|" + mode_id)
        reviewed = [item for item in members if item["audit"]]
        causes = Counter(
            item["audit"].get("mode_causes", {}).get(mode_id, item["audit"].get("likely_cause", ""))
            for item in reviewed if item["vlm_support"]
        )
        causes.pop("", None)
        mode_outputs.append({
            **definition,
            "image_count": len(members),
            "model_counts": {MODEL_NAMES[m]: model_counts[m] for m in MODELS},
            "vlm_reviewed": len(reviewed),
            "vlm_supported": sum(item["vlm_support"] for item in reviewed),
            "vlm_annotation_ambiguity": sum(item["audit"].get("annotation_ambiguity", False) for item in reviewed),
            "vlm_common_cause": causes.most_common(1)[0][0] if causes else "",
            "representative_image_ids": [item["image"]["id"] for item in representatives],
            "member_image_ids": [item["image"]["id"] for item in members],
            "member_models": {item["image"]["id"]: item["models"] for item in members},
        })

    mode_outputs.sort(key=lambda row: (-row["image_count"], row["name"]))
    top_mode_ids = [row["id"] for row in mode_outputs]
    human_set = build_human_set(images, signals, top_mode_ids)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "error_modes.json").write_text(json.dumps(mode_outputs, indent=2, ensure_ascii=False) + "\n")
    write_csv(output / "error_modes.csv", [{
        "mode_id": row["id"], "name": row["name"], "image_count": row["image_count"],
        "dinov3": row["model_counts"]["DINOv3"], "bioclip": row["model_counts"]["BioCLIP 2.5"],
        "efficientnet_b0": row["model_counts"]["EfficientNet-B0"],
        "vlm_reviewed": row["vlm_reviewed"], "vlm_supported": row["vlm_supported"],
        "vlm_annotation_ambiguity": row["vlm_annotation_ambiguity"],
    } for row in mode_outputs])
    write_csv(output / "error_mode_members.csv", member_rows)
    (output / "human_review_set.json").write_text(json.dumps(human_set, indent=2, ensure_ascii=False) + "\n")
    write_csv(output / "human_review_set.csv", [{**row, "selection_categories": ";".join(row["selection_categories"]), "error_modes": ";".join(row["error_modes"])} for row in human_set])

    # High-signal, diverse VLM audit set drawn from every discovered mode.
    audit_ids = []
    for mode in mode_outputs:
        candidates = []
        for image in images:
            involved = signals[image["id"]]["modes"].get(mode["id"])
            if involved:
                candidates.append({"image": image, "severity": len(involved) + len(signals[image["id"]]["gemma_disagreement_models"])})
        for item in diverse_pick(candidates, 26, "vlm-candidate-v2|" + mode["id"]):
            if item["image"]["id"] not in audit_ids:
                audit_ids.append(item["image"]["id"])
    audit_records = []
    for image_id in audit_ids:
        image = by_id[image_id]
        audit_records.append({
            "image_id": image_id, "file": image["file"], "species": image["species"],
            "bioimages_label": image["tagging"]["bioimages_label"],
            "gemma_tags": image["tagging"]["gemma_tags"],
            "gemma_normalized_tags": image["tagging"]["gemma_normalized_tags"],
            "model_tags": {model: image["tagging"]["models"][model]["tags"] for model in MODELS},
            "candidate_modes": sorted(signals[image_id]["modes"]),
        })
    (output / "vlm_audit_candidates.json").write_text(json.dumps(audit_records, indent=2, ensure_ascii=False) + "\n")

    payload = {
        "generated_from_images": len(images),
        "vlm_audited_images": len(vlm),
        "human_tags": list(HUMAN_TAGS),
        "models": MODEL_NAMES,
        "modes": mode_outputs,
        "human_set": human_set,
        "mode_definitions": list(MODE_DEFINITIONS),
    }
    (ROOT / "data" / "human-loop-data.js").write_text(
        "window.BIOIMAGES_HUMAN_LOOP=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    )
    print(json.dumps({
        "images": len(images), "modes": len(mode_outputs), "vlm_candidates": len(audit_records),
        "vlm_audited": len(vlm), "human_set": len(human_set),
        "human_species": len({row["species"] for row in human_set}),
        "human_individuals": len({row["individual_id"] for row in human_set}),
        "top_modes": [{"id": row["id"], "images": row["image_count"]} for row in mode_outputs[:8]],
    }, indent=2))


if __name__ == "__main__":
    main()
