# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Joint detector-score × verifier accept policies, fitted to the human audit.

The verifier alone gives a per-crop judgement. This module asks the next
question: given the detector's own confidence and the verifier's verdict, what
per-class rule reaches a stated measured accept precision, and how much of the
human-accepted pool does that rule actually recover?

Two label lanes are kept strictly separate:

- **assisted-training labels** — machine-accepted proposals marked with model
  provenance. A proposal may only be auto-accepted here when it belongs to a
  training split.
- **human-reviewed test truth** — the test split is never auto-accepted; its
  labels are human decisions and the policy is applied to it for audit only.

Everything here is pure: it re-scores stored verdicts and never calls a model.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import CLASS_NAMES, split_for_observation
from .verify import POLICY_PROFILES, accept_decision, prompt_hash, wilson_interval

POLICY_SCHEMA = "arq.anatomy4.verify-policy/v1"
DEFAULT_TARGET_PRECISION = 0.90
DEFAULT_MIN_ACCEPTS = 5
DEFAULT_LOO_MIN_PRECISION = 0.85
ASSIST_SPLIT = "train"
TEST_SPLIT = "test"
REVIEW_SPLITS: tuple[str, ...] = ("val", "test")
REJECTED_VERDICTS: tuple[str, ...] = ("not_target", "ambiguous")

# Verifier requirement variants enumerated by the fit. `balanced` is the
# measured recommendation; `strict` is kept so the data can overrule it.
VERIFIER_VARIANTS: dict[str, dict[str, Any]] = {
    "balanced": {
        "profile": "balanced",
        "require_resolved": True,
        "accepted_tightness": ("keep", "too_tight", "too_broad"),
        "min_confidence": 0.0,
    },
    "balanced_resolved_only": {
        "profile": None,
        "require_resolved": True,
        "accepted_tightness": ("keep", "too_tight", "too_broad"),
        "min_confidence": 0.0,
    },
    "strict": {
        "profile": "strict",
        "require_resolved": True,
        "accepted_tightness": ("keep",),
        "min_confidence": 0.7,
    },
    "triage": {
        "profile": "triage",
        "require_resolved": False,
        "accepted_tightness": ("keep", "too_tight", "too_broad"),
        "min_confidence": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_proposals(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return {str(p["proposal_id"]): p for p in data.get("proposals") or []}


def load_review(path: str | Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            proposal_id = (row.get("proposal_id") or "").strip()
            if proposal_id:
                rows[proposal_id] = {key: (value or "").strip() for key, value in row.items()}
    return rows


def load_effective_verdicts(path: str | Path) -> dict[str, dict[str, Any]]:
    """Latest pass wins, matching the verifier report."""

    rows: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("proposal_id"):
            rows[str(row["proposal_id"])] = row
    return rows


def build_rows(
    *,
    proposals_path: str | Path,
    verdicts_path: str | Path,
    review_path: str | Path,
) -> list[dict[str, Any]]:
    """Join detector score + verifier verdict + human verdict + split."""

    proposals = load_proposals(proposals_path)
    review = load_review(review_path)
    verdicts = load_effective_verdicts(verdicts_path)
    rows: list[dict[str, Any]] = []
    for proposal_id, verdict_row in verdicts.items():
        proposal = proposals.get(proposal_id)
        if proposal is None or proposal_id not in review:
            continue
        observation_id = str(
            proposal.get("observation_id")
            or verdict_row.get("photo_id")
            or proposal.get("photo_id")
            or proposal_id
        )
        rows.append(
            {
                "proposal_id": proposal_id,
                "class": str(verdict_row.get("proposed_class") or proposal.get("class_name")),
                "detector_score": float(proposal.get("box_score") or 0.0),
                "human_accept": (review[proposal_id].get("verdict") or "").lower() == "tp",
                "human_notes": review[proposal_id].get("notes") or "",
                "split": split_for_observation(observation_id),
                "verdict": verdict_row.get("verdict") or {},
                "pass": int(verdict_row.get("pass") or 1),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


def default_verifier_requirement(variant: str = "balanced") -> dict[str, Any]:
    spec = VERIFIER_VARIANTS[variant]
    return {
        "variant": variant,
        "profile": spec["profile"],
        "require_resolved": bool(spec["require_resolved"]),
        "accepted_tightness": list(spec["accepted_tightness"]),
        "min_confidence": float(spec["min_confidence"]),
    }


def verdict_passes(verdict: Mapping[str, Any], proposed_class: str, requirement: Mapping[str, Any]) -> bool:
    return bool(
        accept_decision(
            verdict,
            proposed_class,
            min_confidence=float(requirement["min_confidence"]),
            accepted_tightness=tuple(requirement["accepted_tightness"]),
            require_resolved=bool(requirement["require_resolved"]),
            profile=requirement.get("profile"),
        )["accept"]
    )


def rule_accepts(row: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    # Round both sides: detector scores are floats and a policy threshold must
    # not exclude a row because 0.60 - 9*0.03 is 0.32999999999999996.
    threshold = round(float(rule.get("detector_score_min") or 0.0), 6)
    if round(float(row["detector_score"]), 6) < threshold:
        return False
    return verdict_passes(row["verdict"], str(row["class"]), rule["verifier"])


def score_rule(rows: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    """Counts + honest intervals for one rule applied to one class's rows."""

    accepted = [row for row in rows if rule_accepts(row, rule)]
    tp = sum(1 for row in accepted if row["human_accept"])
    fp = len(accepted) - tp
    human_positives = sum(1 for row in rows if row["human_accept"])
    fn = human_positives - tp
    precision = tp / len(accepted) if accepted else None
    recall = tp / human_positives if human_positives else None
    precision_low, precision_high = wilson_interval(tp, len(accepted))
    recall_low, recall_high = wilson_interval(tp, human_positives)
    # How many additional false accepts the rule could absorb before dropping
    # below target precision? (0 means one more FP breaks it.)
    tolerance = 0
    if accepted and tp:
        tolerance = max(0, int(tp / DEFAULT_TARGET_PRECISION) - len(accepted))
    return {
        "n_class": len(rows),
        "human_positives": human_positives,
        "accepts": len(accepted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "precision_ci95": [precision_low, precision_high],
        "recall_ci95": [recall_low, recall_high],
        "yield_of_human_accepts": round(len(accepted) / len(rows), 4) if rows else None,
        "extra_fp_tolerated_at_target": tolerance,
        "accepted_ids": [row["proposal_id"] for row in accepted],
    }


def split_support(rows: Sequence[Mapping[str, Any]], rule: Mapping[str, Any], split: str) -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split]
    metrics = score_rule(subset, rule)
    return metrics


def _threshold_grid(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    scores = sorted({round(float(row["detector_score"]), 6) for row in rows}, reverse=True)
    return [0.0] + scores


def candidate_rules(
    rows: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[str] = ("balanced", "balanced_resolved_only", "strict", "triage"),
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for threshold in _threshold_grid(rows):
        for variant in variants:
            rules.append(
                {
                    "detector_score_min": threshold,
                    "verifier": default_verifier_requirement(variant),
                }
            )
    return rules


def _fit_objective(metrics: Mapping[str, Any], assisted: Mapping[str, Any]) -> tuple:
    # Highest usable assisted yield, then precision, then lower threshold.
    return (
        int(assisted["tp"]),
        float(metrics["precision"] or 0.0),
        int(metrics["accepts"]),
    )


def leave_one_out(
    rows: Sequence[Mapping[str, Any]],
    *,
    select,
) -> dict[str, Any]:
    """Apply the *selection procedure* leave-one-out (selection optimism check)."""

    predictions: list[dict[str, Any]] = []
    for index, held_out in enumerate(rows):
        train = [row for offset, row in enumerate(rows) if offset != index]
        rule = select(train) if len(train) >= 4 else None
        accepted = bool(rule and rule_accepts(held_out, rule))
        predictions.append({"human_accept": held_out["human_accept"], "accept": accepted})
    tp = sum(1 for row in predictions if row["accept"] and row["human_accept"])
    fp = sum(1 for row in predictions if row["accept"] and not row["human_accept"])
    fn = sum(1 for row in predictions if not row["accept"] and row["human_accept"])
    precision_low, precision_high = wilson_interval(tp, tp + fp)
    recall_low, recall_high = wilson_interval(tp, tp + fn)
    return {
        "accepts": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision_ci95": [precision_low, precision_high],
        "recall_ci95": [recall_low, recall_high],
    }


def fit_class(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_accepts: int = DEFAULT_MIN_ACCEPTS,
    min_assisted_support: int = DEFAULT_MIN_ACCEPTS,
    min_assisted_precision: float = DEFAULT_TARGET_PRECISION,
    loo_min_precision: float = DEFAULT_LOO_MIN_PRECISION,
    validation_rows: Sequence[Mapping[str, Any]] | None = None,
    validation_min_precision: float = 0.80,
    validation_min_accepts: int = 3,
    background_rows: Sequence[Mapping[str, Any]] | None = None,
    variants: Sequence[str] = ("balanced", "balanced_resolved_only", "strict", "triage"),
    allow_verifier_variant: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one class. Returns the chosen rule plus every rejected alternative."""

    class_name = str(rows[0]["class"]) if rows else "unknown"
    rules = candidate_rules(rows, variants=variants)
    evaluated: list[dict[str, Any]] = []
    for rule in rules:
        metrics = score_rule(rows, rule)
        assisted = split_support(rows, rule, ASSIST_SPLIT)
        evaluated.append({"rule": rule, "all": metrics, "assisted": assisted})

    def select_entries(pool: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        candidates = [
            entry
            for entry in pool
            if entry["all"]["precision"] is not None
            and entry["all"]["precision"] >= target_precision
            and entry["all"]["accepts"] >= min_accepts
            and entry["assisted"]["accepts"] >= min_assisted_support
            # The lane that receives labels must itself meet the bar: an
            # overall-clean rule with a known train false accept cannot be
            # shipped just because other splits dilute the precision.
            and entry["assisted"]["precision"] is not None
            and entry["assisted"]["precision"] >= min_assisted_precision
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda entry: _fit_objective(entry["all"], entry["assisted"]))

    def select_on_rows(pool: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        evaluated_pool = []
        for rule in candidate_rules(pool, variants=variants):
            evaluated_pool.append(
                {
                    "rule": rule,
                    "all": score_rule(pool, rule),
                    "assisted": split_support(pool, rule, ASSIST_SPLIT),
                }
            )
        entry = select_entries(evaluated_pool)
        return None if entry is None else dict(entry["rule"])

    chosen_entry = select_entries(evaluated)
    chosen_rule = None if chosen_entry is None else chosen_entry["rule"]
    loo = leave_one_out(rows, select=select_on_rows)

    loo_precision = loo["precision"]
    baseline_rule = {"detector_score_min": 0.0, "verifier": default_verifier_requirement("balanced")}
    effective_rule = chosen_entry["rule"] if chosen_entry is not None else baseline_rule
    validation_support = (
        None if not validation_rows else score_rule(validation_rows, effective_rule)
    )
    background_support = (
        None if not background_rows else score_rule(background_rows, effective_rule)
    )
    validation_ok = True
    validation_note = None
    if validation_support and validation_support["accepts"] >= validation_min_accepts:
        if (validation_support["precision"] or 0.0) < validation_min_precision:
            validation_ok = False
            validation_note = (
                f"independent validation precision {validation_support['precision']:.3f} on "
                f"{validation_support['accepts']} accepts is below {validation_min_precision:.2f}"
            )
    validated = bool(
        chosen_entry is not None
        and loo_precision is not None
        and loo_precision >= loo_min_precision
        and validation_ok
    )
    action = "auto_assist" if validated else "review"
    baseline = score_rule(rows, baseline_rule)

    alternatives = sorted(
        [
            entry
            for entry in evaluated
            if entry is not chosen_entry
            and entry["all"]["precision"] is not None
            and entry["all"]["precision"] >= target_precision
        ],
        key=lambda entry: (
            entry["assisted"]["tp"],
            entry["all"]["precision"] or 0.0,
        ),
        reverse=True,
    )[:5]

    if validated:
        confidence = "validated_measured"
    elif chosen_entry is not None:
        confidence = "unvalidated_fit"
    else:
        confidence = "insufficient_support"

    if validated:
        decision_reason = (
            f"validated: apparent precision {chosen_entry['all']['precision']:.3f} "
            f">= {target_precision:.2f}, LOO {loo['precision']:.3f} >= {loo_min_precision:.2f}, "
            f"{chosen_entry['assisted']['accepts']} train accepts"
        )
    elif chosen_entry is not None and not validation_ok and validation_note:
        decision_reason = f"apparent fit meets the target but {validation_note}"
    elif chosen_entry is not None:
        decision_reason = (
            f"apparent precision {chosen_entry['all']['precision']:.3f} meets the target but "
            f"LOO selection precision {loo['precision']} is below {loo_min_precision:.2f}"
        )
    else:
        decision_reason = (
            f"no rule reaches {target_precision:.2f} precision with >= {min_accepts} accepts "
            f"and >= {min_assisted_support} train accepts"
        )

    result: dict[str, Any] = {
        "class": class_name,
        "action": action,
        "decision_reason": decision_reason,
        "confidence": confidence,
        "policy": None
        if not validated
        else {
            "detector_score_min": chosen_entry["rule"]["detector_score_min"],
            "verifier": chosen_entry["rule"]["verifier"],
            "applies_to_splits": [ASSIST_SPLIT],
            "never_auto_accept_splits": list(REVIEW_SPLITS),
            "rejects_verdict_classes": list(REJECTED_VERDICTS),
            "audit": {
                "sample_fraction": 0.10,
                "min_sample": 5,
                "reason": (
                    "small-n fit; audit a random 10% (at least 5) of auto-accepted "
                    "labels per class until 50 labels have been checked"
                ),
            },
        },
        "measured_support": {
            "support_ref": (
                "shipped"
                if validated
                else ("unvalidated_candidate" if chosen_entry is not None else "baseline")
            ),
            "baseline_no_score_floor": baseline,
            "all_reviewed": chosen_entry["all"] if chosen_entry is not None else baseline,
            "assisted_train": chosen_entry["assisted"]
            if chosen_entry is not None
            else split_support(rows, baseline_rule, ASSIST_SPLIT),
            "human_reviewed_test": split_support(
                rows, chosen_entry["rule"] if chosen_entry is not None else baseline_rule, TEST_SPLIT
            ),
            "held_out_validation": loo,
            "independent_validation": validation_support,
            "background_negatives": background_support,
        },
        "validated": validated,
        "caveats": _class_caveats(
            class_name, rows, chosen_entry, baseline, loo, validated
        ),
        "rejected_best_alternatives": [
            {
                "detector_score_min": entry["rule"]["detector_score_min"],
                "verifier_variant": entry["rule"]["verifier"]["variant"],
                "apparent": {
                    "accepts": entry["all"]["accepts"],
                    "tp": entry["all"]["tp"],
                    "fp": entry["all"]["fp"],
                    "precision": entry["all"]["precision"],
                    "recall": entry["all"]["recall"],
                },
                "assisted_train": {
                    "accepts": entry["assisted"]["accepts"],
                    "tp": entry["assisted"]["tp"],
                    "fp": entry["assisted"]["fp"],
                },
                "note": "meets target on the fitting data but is not the chosen highest-yield rule",
            }
            for entry in alternatives
        ],
    }
    if chosen_entry is not None and not validated:
        result["unvalidated_candidate"] = {
            "detector_score_min": chosen_entry["rule"]["detector_score_min"],
            "verifier_variant": chosen_entry["rule"]["verifier"]["variant"],
            "apparent": chosen_entry["all"],
            "held_out": loo,
            "why_not_shipped": (
                f"apparent precision meets the target but leave-one-out selection "
                f"precision is {loo['precision']}, below the {loo_min_precision} floor"
            ),
        }
    if chosen_entry is None:
        best = max(
            evaluated,
            key=lambda entry: (
                entry["all"]["precision"] or 0.0,
                entry["assisted"]["tp"],
            ),
        )
        result["best_effort"] = {
            "detector_score_min": best["rule"]["detector_score_min"],
            "verifier_variant": best["rule"]["verifier"]["variant"],
            **best["all"],
            "why_not_shipped": _why_not_shipped(
                best,
                target_precision=target_precision,
                min_accepts=min_accepts,
                min_assisted_support=min_assisted_support,
                min_assisted_precision=min_assisted_precision,
            ),
        }
    if allow_verifier_variant:
        result["measured_alternative_verifier"] = allow_verifier_variant
    return result


def _why_not_shipped(
    entry: Mapping[str, Any],
    *,
    target_precision: float,
    min_accepts: int,
    min_assisted_support: int,
    min_assisted_precision: float,
) -> str:
    overall = entry["all"]
    assisted = entry["assisted"]
    if (overall["precision"] or 0.0) < target_precision:
        return (
            f"best apparent precision {_fmt(overall['precision'])} on "
            f"{overall['accepts']} accepts is below the {target_precision:.2f} target"
        )
    if overall["accepts"] < min_accepts:
        return f"meets the target but has only {overall['accepts']} accepts (< {min_accepts})"
    if assisted["accepts"] < min_assisted_support:
        return (
            f"meets the target but has only {assisted['accepts']} assisted-train "
            f"accepts (< {min_assisted_support})"
        )
    if (assisted["precision"] or 0.0) < min_assisted_precision:
        return (
            f"overall precision meets the target but the assisted train lane is "
            f"{_fmt(assisted['precision'])} (< {min_assisted_precision:.2f})"
        )
    return "meets the apparent gates but fails held-out validation"


def _class_caveats(
    class_name: str,
    rows: Sequence[Mapping[str, Any]],
    chosen_entry: Mapping[str, Any] | None,
    baseline: Mapping[str, Any],
    loo: Mapping[str, Any],
    validated: bool,
) -> list[str]:
    positives = baseline["human_positives"]
    notes: list[str] = [
        f"{len(rows)} reviewed proposals, {positives} human-accepted — every interval is wide at this n."
    ]
    if chosen_entry is not None:
        support = chosen_entry["all"]
        train = chosen_entry["assisted"]
        test = split_support(rows, chosen_entry["rule"], TEST_SPLIT)
        if support["fp"] and not (test["fp"] and not train["fp"]):
            notes.append(
                f"{support['fp']} false accept(s) remain across all reviewed splits."
            )
        if test["fp"] and not train["fp"]:
            notes.append(
                f"the {support['accepts']}-accept rule is clean on train only because its "
                f"{support['fp']} known false accept landed in the human-reviewed test split; "
                "use the all-reviewed precision as the conservative estimate."
            )
        if (train["accepts"] or 0) < 10:
            notes.append(
                f"assisted-train support is {train['accepts']} labels; treat precision as provisional."
            )
    if validated:
        notes.append("probationary: audit 10% of auto-accepted labels until 50 are checked.")
    elif chosen_entry is not None:
        notes.append(
            "apparent fit meets the target but does not survive leave-one-out selection; "
            "do not auto-assist this class."
        )
    else:
        notes.append(
            "no rule reaches the precision target with the minimum support; "
            "the verifier still triages (its rejects can be deprioritized)."
        )
    return notes


def fit_policies(
    *,
    proposals_path: str | Path,
    verdicts_path: str | Path,
    review_path: str | Path,
    model: str | None = None,
    detector: str | None = None,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_accepts: int = DEFAULT_MIN_ACCEPTS,
    min_assisted_support: int = DEFAULT_MIN_ACCEPTS,
    min_assisted_precision: float = DEFAULT_TARGET_PRECISION,
    loo_min_precision: float = DEFAULT_LOO_MIN_PRECISION,
    validation: Mapping[str, Any] | None = None,
    background: Mapping[str, Any] | None = None,
    validation_min_precision: float = 0.80,
    validation_min_accepts: int = 3,
    verifier_alternatives: Mapping[str, Mapping[str, Any]] | None = None,
    decision_out: str | Path | None = None,
) -> dict[str, Any]:
    rows = build_rows(
        proposals_path=proposals_path,
        verdicts_path=verdicts_path,
        review_path=review_path,
    )
    validation_rows: list[dict[str, Any]] = []
    if validation:
        validation_rows = build_rows(
            proposals_path=proposals_path,
            verdicts_path=validation["verdicts"],
            review_path=validation["review"],
        )
    background_rows: list[dict[str, Any]] = []
    if background:
        background_rows = build_rows(
            proposals_path=proposals_path,
            verdicts_path=background["verdicts"],
            review_path=background["review"],
        )
    by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_NAMES}
    for row in rows:
        if row["class"] in by_class:
            by_class[row["class"]].append(row)

    per_class: dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        class_rows = by_class[class_name]
        alternative = (verifier_alternatives or {}).get(class_name)
        per_class[class_name] = (
            fit_class(
                class_rows,
                target_precision=target_precision,
                min_accepts=min_accepts,
                min_assisted_support=min_assisted_support,
                min_assisted_precision=min_assisted_precision,
                loo_min_precision=loo_min_precision,
                validation_rows=[row for row in validation_rows if row["class"] == class_name],
                validation_min_precision=validation_min_precision,
                validation_min_accepts=validation_min_accepts,
                background_rows=[row for row in background_rows if row["class"] == class_name],
                allow_verifier_variant=alternative,
            )
            if class_rows
            else {
                "class": class_name,
                "action": "review",
                "confidence": "insufficient",
                "policy": None,
                "measured_support": None,
                "validated": False,
            }
        )

    # Fallbacks for proposals the class policy does not accept.
    fallback = {
        "verifier_reject": {
            "action": "deprioritize",
            "condition": "verifier verdict class is not_target or ambiguous, or the verdict is invalid",
        },
        "verifier_accept_but_policy_reject": {
            "action": "review",
            "condition": "verifier accepted but the class policy (score floor / support gate) did not",
        },
        "non_train_split": {
            "action": "review",
            "condition": "split is val or test — human review is authoritative, never auto-accept",
        },
    }

    policy = {
        "schema": POLICY_SCHEMA,
        "contract": "anatomy4-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "proposals": str(proposals_path),
            "verdicts": str(verdicts_path),
            "review": str(review_path),
            "validation_verdicts": str(validation["verdicts"]) if validation else None,
            "validation_review": str(validation["review"]) if validation else None,
            "background_verdicts": str(background["verdicts"]) if background else None,
            "background_review": str(background["review"]) if background else None,
            "model": model,
            "detector": detector,
            "prompt_hash": prompt_hash(),
            "n_proposals": len(rows),
        },
        "targets": {
            "accept_precision": target_precision,
            "min_accepts_per_class": min_accepts,
            "min_assisted_train_accepts": min_assisted_support,
            "min_assisted_train_precision": min_assisted_precision,
            "loo_min_precision": loo_min_precision,
            "validation_min_precision": validation_min_precision,
            "validation_min_accepts": validation_min_accepts,
        },
        "lanes": {
            "assisted_training": {
                "splits": [ASSIST_SPLIT],
                "requires_provenance": True,
                "description": (
                    "machine-accepted labels written only with detector score, verifier "
                    "verdict, model id, prompt hash and policy id recorded per label"
                ),
            },
            "human_reviewed_truth": {
                "splits": list(REVIEW_SPLITS),
                "auto_accept": False,
                "description": (
                    "val/test labels are human decisions; the policy is scored on test "
                    "for audit only and must never fill it"
                ),
            },
        },
        "fallback_actions": fallback,
        "per_class": per_class,
        "provenance_requirements": {
            "required_fields": [
                "proposal_id",
                "photo_id",
                "class_name",
                "detector_score",
                "verifier_verdict_class",
                "verifier_confidence",
                "verifier_bbox_tightness",
                "verifier_resolved_enough",
                "model",
                "prompt_hash",
                "policy_schema",
                "policy_action",
            ],
            "note": "auto-accepted labels without these fields must be re-queued for review",
        },
    }

    import hashlib

    # Stable id: exclude the timestamp so a re-run on the same evidence matches.
    hashed = {key: value for key, value in policy.items() if key != "created_at"}
    policy["operational_summary"] = _operational_summary(rows, per_class)
    policy["policy_id"] = hashlib.sha256(
        json.dumps(hashed, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    if decision_out is not None:
        write_decisions(Path(decision_out), rows, per_class)
    return policy


def _operational_summary(
    rows: Sequence[Mapping[str, Any]],
    per_class: Mapping[str, Any],
) -> dict[str, Any]:
    """What the policy does to the review queue, measured on the audited set."""

    def bucket(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts = {"n": len(items), "auto_assist": 0, "review": 0, "deprioritize": 0}
        accepted = {"auto_assist": 0, "review": 0, "deprioritize": 0}
        false_accepts = {"auto_assist": 0, "review": 0, "deprioritize": 0}
        human_positives = {"auto_assist": 0, "review": 0, "deprioritize": 0}
        for row in items:
            action = decision_for_row(row, per_class.get(row["class"]) or {})["action"]
            counts[action] += 1
            if row["human_accept"]:
                human_positives[action] += 1
            else:
                false_accepts[action] += 1
            accepted[action] += 1
        return {
            **counts,
            "human_positives": human_positives,
            "non_human_positives": false_accepts,
            "auto_assist_precision": round(
                human_positives["auto_assist"] / accepted["auto_assist"], 4
            )
            if accepted["auto_assist"]
            else None,
        }

    train = bucket([row for row in rows if row["split"] == ASSIST_SPLIT])
    non_train = bucket([row for row in rows if row["split"] != ASSIST_SPLIT])
    reviewed = len(rows)
    manual_baseline = reviewed
    manual_after = train["review"] + non_train["review"] + non_train["auto_assist"]
    return {
        "reviewed_proposals": reviewed,
        "assisted_split": train,
        "human_reviewed_splits": non_train,
        "manual_review_items_before": manual_baseline,
        "manual_review_items_after": manual_after,
        "manual_review_reduction_pct": round(
            (manual_baseline - manual_after) / manual_baseline, 4
        )
        if manual_baseline
        else None,
        "assisted_split_review_before": train["n"],
        "assisted_split_review_after": train["review"],
        "assisted_split_review_reduction_pct": round(
            (train["n"] - train["review"]) / train["n"], 4
        )
        if train["n"]
        else None,
        "human_positives_auto_assisted": train["human_positives"]["auto_assist"],
        "human_positives_left_for_review": train["human_positives"]["review"]
        + non_train["human_positives"]["review"],
        "human_positives_deprioritized": train["human_positives"]["deprioritize"],
        "note": (
            "deprioritize means LAST in the review queue, not deleted; "
            "it exists so the measured misses stay recoverable"
        ),
    }


def decision_for_row(row: Mapping[str, Any], class_policy: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the action for one proposal under its class policy."""

    policy = class_policy.get("policy")
    if row["split"] != ASSIST_SPLIT:
        return {"action": "review", "reason": "non_train_split"}
    verdict_class = str((row["verdict"] or {}).get("verdict_class") or "")
    if verdict_class in REJECTED_VERDICTS or not (row["verdict"] or {}).get("valid", False):
        return {"action": "deprioritize", "reason": f"verifier_{verdict_class or 'invalid'}"}
    if policy is None:
        return {"action": "review", "reason": "class_policy_review_only"}
    if rule_accepts(row, policy):
        return {"action": "auto_assist", "reason": "policy_accept"}
    return {"action": "review", "reason": "policy_reject"}


def write_decisions(path: Path, rows: Sequence[Mapping[str, Any]], per_class: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in sorted(rows, key=lambda item: (item["class"], -item["detector_score"])):
            class_policy = per_class.get(row["class"]) or {}
            action = decision_for_row(row, class_policy)
            handle.write(
                json.dumps(
                    {
                        "proposal_id": row["proposal_id"],
                        "class": row["class"],
                        "split": row["split"],
                        "detector_score": row["detector_score"],
                        "human_accept": row["human_accept"],
                        "verifier_verdict_class": (row["verdict"] or {}).get("verdict_class"),
                        "action": action["action"],
                        "action_reason": action["reason"],
                        "class_action": class_policy.get("action"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_policy_markdown(policy: Mapping[str, Any]) -> str:
    lines = ["# Anatomy4 joint detector/verifier policy", ""]
    detector = (policy.get("source") or {}).get("detector")
    if detector:
        lines.append(f"- proposal source: `{detector}`")
    targets = policy["targets"]
    lines.append(
        f"- target accept precision: {targets['accept_precision']:.2f}; "
        f"min accepts/class: {targets['min_accepts_per_class']}; "
        f"min assisted-train accepts: {targets['min_assisted_train_accepts']}; "
        f"held-out precision floor: {targets['loo_min_precision']:.2f}"
    )
    lines.append(
        "- assisted lane: train only, provenance required. val/test are human "
        "truth and are never auto-accepted."
    )
    lines.append("")
    lines.append(
        "| class | action | rule | detector score ≥ | verifier | accepts | TP | FP | "
        "precision (95% CI) | recall | assisted-train TP | LOO precision |"
    )
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for class_name, entry in policy["per_class"].items():
        support = entry.get("measured_support") or {}
        loo = support.get("held_out_validation") or {}
        rule = entry.get("policy") or {}
        if rule:
            display = rule
            overall = support.get("all_reviewed") or {}
            assisted = support.get("assisted_train") or {}
            label = "shipped"
        elif entry.get("unvalidated_candidate"):
            candidate = entry["unvalidated_candidate"]
            display = {
                "detector_score_min": candidate["detector_score_min"],
                "verifier": {"variant": candidate["verifier_variant"]},
            }
            overall = candidate["apparent"]
            assisted = {"tp": None}
            label = "candidate (not shipped)"
        else:
            display = {"detector_score_min": None, "verifier": {"variant": "balanced"}}
            overall = support.get("baseline_no_score_floor") or {}
            assisted = {"tp": None}
            label = "baseline verifier only"
        precision_ci = overall.get("precision_ci95") or [None, None]
        assisted_tp = "—" if assisted.get("tp") is None else str(assisted.get("tp"))
        if entry.get("action") == "auto_assist":
            loo_precision = loo.get("precision")
        else:
            loo_precision = (
                (entry.get("unvalidated_candidate") or {}).get("held_out") or loo
            ).get("precision")
        lines.append(
            f"| {class_name} | {entry.get('action')} | {label} | "
            f"{_num(display.get('detector_score_min'))} | "
            f"{(display.get('verifier') or {}).get('variant') or '—'} | {overall.get('accepts', 0)} | "
            f"{overall.get('tp', 0)} | {overall.get('fp', 0)} | "
            f"{_fmt(overall.get('precision'))} ({_fmt(precision_ci[0])}–{_fmt(precision_ci[1])}) | "
            f"{_fmt(overall.get('recall'))} | {assisted_tp} | "
            f"{_fmt(loo_precision)} |"
        )
    lines.append("")
    for class_name, entry in policy["per_class"].items():
        support = entry.get("measured_support") or {}
        text = f"## {class_name}: {entry.get('action')} ({entry.get('confidence')})"
        if entry.get("action") == "auto_assist":
            assisted = support.get("assisted_train") or {}
            all_reviewed = support.get("all_reviewed") or {}
            test = support.get("human_reviewed_test") or {}
            text += (
                f"\n- train labels auto-accepted: {assisted.get('accepts')} "
                f"({assisted.get('tp')} human-accepted, {assisted.get('fp')} false accepts)\n"
                f"- assisted precision {_fmt(assisted.get('precision'))}, "
                f"assisted recall {_fmt(assisted.get('recall'))}\n"
                f"- all reviewed: P {_fmt(all_reviewed.get('precision'))} "
                f"R {_fmt(all_reviewed.get('recall'))}, "
                f"{all_reviewed.get('accepts')} accepts / {all_reviewed.get('fp')} FP; "
                f"held-out (LOO) P {_fmt((support.get('held_out_validation') or {}).get('precision'))}\n"
                f"- test split audited: {test.get('accepts')} accepts, "
                f"{test.get('fp')} false accepts — human truth, never labels"
            )
        else:
            best = entry.get("best_effort")
            candidate = entry.get("unvalidated_candidate")
            if candidate:
                text += (
                    f"\n- unvalidated candidate: score ≥ {candidate['detector_score_min']:.4f}, "
                    f"{candidate['apparent']['accepts']} accepts, "
                    f"P {_fmt(candidate['apparent']['precision'])}, "
                    f"R {_fmt(candidate['apparent']['recall'])}; "
                    f"held-out P {_fmt(candidate['held_out']['precision'])} — {candidate['why_not_shipped']}"
                )
            if best:
                text += (
                    f"\n- best effort: score ≥ {best['detector_score_min']:.4f}, "
                    f"{best['accepts']} accepts, P {_fmt(best['precision'])}, "
                    f"R {_fmt(best['recall'])} — {best['why_not_shipped']}"
                )
        alternative = entry.get("measured_alternative_verifier")
        if alternative:
            measured = alternative.get("measured") or {}
            text += (
                f"\n- measured verifier alternative `{alternative.get('variant')}`: "
                f"{measured.get('accepts')} accepts, {measured.get('tp')} TP, "
                f"{measured.get('fp')} FP, P {_fmt(measured.get('precision'))}, "
                f"R {_fmt(measured.get('recall'))} — {measured.get('note', '')}"
            )
        for caveat in entry.get("caveats") or []:
            text += f"\n- caveat: {caveat}"
        lines.append(text + "\n")
    independent_rows: list[str] = []
    for class_name, entry in policy["per_class"].items():
        support = entry.get("measured_support") or {}
        validation = support.get("independent_validation")
        background = support.get("background_negatives")
        if not validation and not background:
            continue
        independent_rows.append(
            f"| {class_name} | {_group(validation)} | {_group(background)} |"
        )
    if independent_rows:
        lines.append("## Independent checks (not used to fit the rule)")
        lines.append("")
        lines.append(
            "`validation` is the separate broad-sample class sheets; `background` is the "
            "background/negative sheets (not clean negatives — the reviewer records real organs as tp)."
        )
        lines.append("")
        lines.append(
            "| class | validation: accepts / TP / FP / precision | "
            "background: accepts / TP / FP / precision |"
        )
        lines.append("| --- | --- | --- |")
        lines.extend(independent_rows)
        lines.append("")
    lines.append("## Fallback actions\n")
    for name, fallback in policy["fallback_actions"].items():
        lines.append(f"- `{name}` → **{fallback['action']}**: {fallback['condition']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _group(metrics: Mapping[str, Any] | None) -> str:
    if not metrics:
        return "—"
    return (
        f"{metrics.get('accepts', 0)} / {metrics.get('tp', 0)} / "
        f"{metrics.get('fp', 0)} / {_fmt(metrics.get('precision'))}"
    )


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _num(value: Any) -> str:
    return "0 (none)" if not value else f"{float(value):.4f}"
