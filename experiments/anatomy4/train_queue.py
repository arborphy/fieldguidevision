# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Bounded production verification and assisted/review queue export.

This module consumes explicit train-candidate proposal pools only. It has no
path or branch for the blind-test artifacts. Model accepts are not human truth:
labels carry model provenance and val/test are never eligible in this command.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .policy import POLICY_SCHEMA
from .verify import (
    SCHEMA as VERIFIER_SCHEMA,
    OpenRouterSettings,
    accept_decision,
    build_batch_messages,
    call_with_retries,
    find_image,
    make_crop,
    extract_json_payload,
    parse_verdict_item,
    prompt_hash,
)

RUN_SCHEMA = "arq.anatomy4.train-verify-run/v1"
ASSISTED_SOURCE = "model"
ASSISTED_REVIEW_STATE = "accepted"
DEFAULT_SEED = "anatomy4-production-train-verify-v1"
YOLOE_SCORE_BINS = (
    (0.095076, 0.20, 80),
    (0.20, 0.30, 100),
    (0.30, 0.45, 100),
    (0.45, 1.000001, 120),
)
INFLO_SCORE_BINS = (
    (0.05, 0.20, 80),
    (0.20, 0.30, 100),
    (0.30, 0.45, 100),
    (0.45, 1.000001, 120),
)


def _stable_rank(seed: str, proposal_id: str) -> str:
    return hashlib.sha256(f"{seed}:{proposal_id}".encode()).hexdigest()


def _is_new_train_candidate(proposal: Mapping[str, Any]) -> bool:
    if proposal.get("status") != "proposed":
        return False
    if proposal.get("proposal_state") not in (None, "proposed", "proposal_only"):
        return False
    if proposal.get("review_status") not in (None, "unreviewed", "candidate"):
        return False
    if proposal.get("origin") == "development":
        return False
    return True


def _take_bins(
    proposals: Sequence[Mapping[str, Any]],
    *,
    bins: Sequence[tuple[float, float, int]],
    target: int,
    seed: str,
    per_photo_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_photo: dict[str, int] = {}
    bin_counts: dict[str, int] = {}

    def eligible_for(score: float, low: float, high: float) -> bool:
        return low <= score < high

    for low, high, quota in bins:
        label = f"{low:g}-{high:g}"
        candidates = [
            proposal
            for proposal in proposals
            if eligible_for(float(proposal.get("box_score") or 0.0), low, high)
            and str(proposal.get("proposal_id")) not in selected_ids
            and per_photo.get(str(proposal.get("photo_id")), 0) < per_photo_cap
        ]
        candidates.sort(key=lambda p: _stable_rank(seed, str(p.get("proposal_id"))))
        taken = 0
        for proposal in candidates:
            if len(selected) >= target or taken >= quota:
                break
            pid = str(proposal.get("proposal_id") or "")
            photo_id = str(proposal.get("photo_id") or proposal.get("image_id") or pid)
            if not pid or per_photo.get(photo_id, 0) >= per_photo_cap:
                continue
            selected.append(dict(proposal))
            selected_ids.add(pid)
            per_photo[photo_id] = per_photo.get(photo_id, 0) + 1
            taken += 1
        bin_counts[label] = taken

    # Reallocate shortages without changing the score emphasis: take highest
    # score strata first, still enforcing the per-image cap.
    if len(selected) < target:
        remaining = [
            proposal
            for proposal in proposals
            if str(proposal.get("proposal_id")) not in selected_ids
            and per_photo.get(str(proposal.get("photo_id")), 0) < per_photo_cap
        ]
        remaining.sort(
            key=lambda p: (
                -float(p.get("box_score") or 0.0),
                _stable_rank(seed, str(p.get("proposal_id"))),
            )
        )
        for proposal in remaining:
            if len(selected) >= target:
                break
            pid = str(proposal.get("proposal_id") or "")
            photo_id = str(proposal.get("photo_id") or proposal.get("image_id") or pid)
            if not pid or per_photo.get(photo_id, 0) >= per_photo_cap:
                continue
            selected.append(dict(proposal))
            selected_ids.add(pid)
            per_photo[photo_id] = per_photo.get(photo_id, 0) + 1
            score = float(proposal.get("box_score") or 0.0)
            label = next(
                (f"{low:g}-{high:g}" for low, high, _ in bins if low <= score < high),
                "reallocated",
            )
            bin_counts[label] = bin_counts.get(label, 0) + 1
    return selected, bin_counts


def select_production_candidates(
    *,
    yoloe_proposals: Sequence[Mapping[str, Any]],
    owlv2_proposals: Sequence[Mapping[str, Any]],
    leaf_count: int = 400,
    inflorescence_count: int = 400,
    flower_count: int = 400,
    seed: str = DEFAULT_SEED,
    max_per_photo: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic score-stratified selection from train-candidate pools."""

    yoloe_leaf = [
        p
        for p in yoloe_proposals
        if p.get("class_name") == "leaf"
        and _is_new_train_candidate(p)
        and (p.get("model") or {}).get("backend", "yoloe") in {"yoloe", None}
    ]
    yoloe_inflorescence = [
        p
        for p in yoloe_proposals
        if p.get("class_name") == "inflorescence"
        and _is_new_train_candidate(p)
        and (p.get("model") or {}).get("backend", "yoloe") in {"yoloe", None}
    ]
    owlv2_flower = [
        p
        for p in owlv2_proposals
        if p.get("class_name") == "flower"
        and _is_new_train_candidate(p)
        and float(p.get("box_score") or 0.0) >= 0.15
        and (
            (p.get("model") or {}).get("backend") == "owlv2"
            or "owlv2" in str((p.get("model") or {}).get("detector") or "").lower()
        )
    ]

    leaf, leaf_bins = _take_bins(
        yoloe_leaf,
        bins=YOLOE_SCORE_BINS,
        target=leaf_count,
        seed=f"{seed}:leaf",
        per_photo_cap=max_per_photo,
    )
    inflorescence, inflo_bins = _take_bins(
        yoloe_inflorescence,
        bins=INFLO_SCORE_BINS,
        target=inflorescence_count,
        seed=f"{seed}:inflorescence",
        per_photo_cap=max_per_photo,
    )

    # OWLv2 flower: round-robin over shuffled image groups, then proposals
    # within an image. Capping at three ensures broad image coverage.
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for proposal in owlv2_flower:
        photo_id = str(proposal.get("photo_id") or proposal.get("image_id") or "")
        grouped.setdefault(photo_id, []).append(proposal)
    rng = random.Random(f"{seed}:owlv2-flower")
    photo_ids = sorted(grouped)
    rng.shuffle(photo_ids)
    for photo_id in photo_ids:
        grouped[photo_id].sort(key=lambda p: _stable_rank(seed, str(p.get("proposal_id"))))
    flower: list[dict[str, Any]] = []
    for round_index in range(max_per_photo):
        order = list(photo_ids)
        rng.shuffle(order)
        for photo_id in order:
            candidates = grouped[photo_id]
            if round_index < len(candidates) and len(flower) < flower_count:
                flower.append(dict(candidates[round_index]))
        if len(flower) >= flower_count:
            break

    selected = leaf + inflorescence + flower
    if len(leaf) != leaf_count or len(inflorescence) != inflorescence_count or len(flower) != flower_count:
        raise ValueError(
            "candidate pools cannot satisfy requested sample counts under per-photo cap: "
            f"leaf={len(leaf)}/{leaf_count}, inflorescence={len(inflorescence)}/{inflorescence_count}, "
            f"flower={len(flower)}/{flower_count}"
        )
    metadata = {
        "seed": seed,
        "target_counts": {
            "leaf_yoloe": leaf_count,
            "inflorescence_yoloe": inflorescence_count,
            "flower_owlv2": flower_count,
        },
        "candidate_counts": {
            "leaf_yoloe": len(yoloe_leaf),
            "inflorescence_yoloe": len(yoloe_inflorescence),
            "flower_owlv2_score_ge_0_15": len(owlv2_flower),
        },
        "score_strata_selected": {
            "leaf_yoloe": leaf_bins,
            "inflorescence_yoloe": inflo_bins,
        },
        "unique_photos": {
            "leaf_yoloe": len({str(p.get("photo_id")) for p in leaf}),
            "inflorescence_yoloe": len({str(p.get("photo_id")) for p in inflorescence}),
            "flower_owlv2": len({str(p.get("photo_id")) for p in flower}),
        },
        "max_proposals_per_photo": max_per_photo,
        "flower_min_detector_score": 0.15,
    }
    return selected, metadata


def load_manifest_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            observation_id = str(row.get("observation_id") or "")
            photo_id = str(row.get("photo_id") or "")
            for key in (observation_id, photo_id):
                if key:
                    metadata.setdefault(key, row)
    return metadata


def select_photo_url(
    proposal: Mapping[str, Any],
    manifest: Mapping[str, Mapping[str, Any]],
) -> str | None:
    for key in (
        str(proposal.get("observation_id") or ""),
        str(proposal.get("photo_id") or ""),
        str(proposal.get("image_id") or ""),
    ):
        if key and key in manifest:
            row = manifest[key]
            photo_url = row.get("photo_url") or row.get("source_url")
            if photo_url:
                return str(photo_url)
    return proposal.get("photo_url")


def _policy_accepts(
    proposal: Mapping[str, Any],
    verdict: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    class_name = str(proposal.get("class_name") or "")
    class_policy = (policy.get("per_class") or {}).get(class_name) or {}
    if class_name == "flower":
        return False, {"accept": False, "reasons": ["class_policy_review_only"]}
    if class_policy.get("action") != "auto_assist" or not class_policy.get("policy"):
        return False, None
    if "train" not in (class_policy["policy"].get("applies_to_splits") or []):
        return False, {"accept": False, "reasons": ["policy_not_train_eligible"]}
    rule = class_policy["policy"]
    score_min = float(rule.get("detector_score_min") or 0.0)
    detector_score = float(proposal.get("box_score") or 0.0)
    if detector_score + 1e-9 < score_min:
        return False, {"accept": False, "reasons": [f"detector_score<{score_min:.6f}"]}
    requirement = rule.get("verifier") or {}
    decision = accept_decision(
        verdict,
        class_name,
        min_confidence=float(requirement.get("min_confidence") or 0.0),
        accepted_tightness=tuple(requirement.get("accepted_tightness") or ("keep",)),
        require_resolved=bool(requirement.get("require_resolved", True)),
        profile=requirement.get("profile"),
    )
    return bool(decision["accept"]), decision


def _source_url(proposal: Mapping[str, Any], image_path: Path) -> str | None:
    from PIL import Image

    photo_url = proposal.get("photo_url") or proposal.get("source_url")
    if photo_url:
        return str(photo_url)
    # Find saved metadata alongside the candidate pool when available.
    candidates = [
        image_path.parent.parent / "manifest.jsonl",
        image_path.parent.parent / "manifest.csv",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".jsonl":
            for line in candidate.read_text().splitlines():
                if not line:
                    continue
                row = json.loads(line)
                if str(row.get("photo_id")) == str(proposal.get("photo_id")) or str(
                    row.get("observation_id")
                ) == str(proposal.get("observation_id")):
                    return row.get("photo_url") or row.get("source_url")
        else:
            with candidate.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("photo_id")) == str(proposal.get("photo_id")) or str(
                        row.get("observation_id")
                    ) == str(proposal.get("observation_id")):
                        return row.get("photo_url") or row.get("source_url")
    return None


def _wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n <= 0:
        return [None, None]
    from .verify import wilson_interval

    low, high = wilson_interval(k, n, z)
    return [low, high]


def run_train_verification(
    *,
    proposals: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    images_dir: str | Path,
    policies: Mapping[str, Mapping[str, Any]],
    settings: OpenRouterSettings,
    out_dir: str | Path,
    requested_policies: Mapping[str, str],
    selection_metadata: Mapping[str, Any] | None = None,
    batch_size: int = 4,
    max_workers: int = 4,
    max_crops: int = 1200,
    audit_fraction: float = 0.10,
    audit_seed: str = "anatomy4-curator-audit-v1",
    max_retries: int = 3,
) -> dict[str, Any]:
    """Verify selected crops in bounded parallel batches and write queues."""

    if len(proposals) > max_crops:
        raise ValueError(f"selected {len(proposals)} crops exceeds max_crops={max_crops}")
    output = Path(out_dir)
    crops_dir = output / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for proposal in proposals:
        proposal = dict(proposal)
        class_name = str(proposal.get("class_name") or "")
        detector = str(requested_policies.get(class_name) or "")
        if not detector:
            skipped.append({"proposal_id": proposal.get("proposal_id"), "reason": "class_not_enabled"})
            continue
        proposal["_detector_key"] = detector
        proposal["_photo_url"] = select_photo_url(proposal, manifest)
        local_image = find_candidate_image(proposal, images_dir)
        proposal["_local_image"] = str(local_image) if local_image else None
        if not proposal["_local_image"] or not Path(proposal["_local_image"]).exists():
            raise FileNotFoundError(f"image missing for selected proposal {proposal.get('proposal_id')}")
        if not proposal.get("_photo_url"):
            raise ValueError(f"photo_url missing for selected proposal {proposal.get('proposal_id')}")
        selected.append(proposal)

    raw_path = output / "raw_calls.jsonl"
    verdict_path = output / "verdicts.jsonl"
    if verdict_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {verdict_path}")
    raw_calls: list[dict[str, Any]] = []
    if raw_path.exists():
        raw_calls = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    covered_ids: set[str] = set()
    for call in raw_calls:
        if not call.get("ok") or not call.get("content"):
            continue
        try:
            payload = extract_json_payload(str(call["content"]))
            returned = payload.get("verdicts") if isinstance(payload, dict) else payload
            expected = len(call.get("proposal_ids") or [])
            if isinstance(returned, list) and len(returned) == expected:
                covered_ids.update(str(proposal_id) for proposal_id in call.get("proposal_ids") or [])
        except Exception:
            continue
    to_process = [proposal for proposal in selected if str(proposal.get("proposal_id")) not in covered_ids]
    batches: list[list[dict[str, Any]]] = []
    for start in range(0, len(to_process), max(1, batch_size)):
        batches.append(to_process[start : start + max(1, batch_size)])

    def prepare_batch(index: int, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        crop_items: list[dict[str, Any]] = []
        for proposal in batch:
            proposal_id = str(proposal["proposal_id"])
            crop_path = crops_dir / f"{proposal_id}.jpg"
            crop_metadata = make_crop(
                proposal["_local_image"],
                proposal["detector_bbox"],
                crop_path,
                margin=0.18,
                max_side=900,
            )
            crop_items.append(
                {
                    "proposal": proposal,
                    "proposal_id": proposal_id,
                    "proposed_class": proposal["class_name"],
                    "crop_path": str(crop_path),
                    "crop_meta": crop_metadata,
                }
            )
        messages = build_batch_messages(crop_items, variant="v1")
        return {
            "index": index,
            "items": crop_items,
            "messages": messages,
            "settings": settings,
        }

    prepared = [prepare_batch(index, batch) for index, batch in enumerate(batches)]

    def send_batch(task: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        result = call_with_retries(
            task["settings"],
            task["messages"],
            attempts=max_retries,
        )
        return int(task["index"]), result

    def process_call_record(call_record: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> None:
        if not call_record.get("ok") or not call_record.get("content"):
            for item in items:
                proposal = item["proposal"]
                public_proposal = {key: value for key, value in proposal.items() if not str(key).startswith("_")}
                verdicts.append(
                    {
                        "schema": "arq.anatomy4.train-verdict/v1",
                        "proposal_id": item["proposal_id"],
                        "observation_id": str(proposal.get("observation_id") or ""),
                        "photo_id": str(proposal.get("photo_id") or proposal.get("image_id") or ""),
                        "photo_url": proposal["_photo_url"],
                        "class": item["proposed_class"],
                        "split": "train",
                        "human_truth": False,
                        "verification_status": "request_failed",
                        "verification_error": call_record.get("error"),
                        "detector": {
                            "detector": proposal["_detector_key"],
                            "model": (proposal.get("model") or {}).get("detector"),
                            "proposal_id": proposal["proposal_id"],
                            "detector_score": float(proposal.get("box_score") or 0.0),
                            "detector_bbox_xyxy_norm": [float(value) for value in proposal["detector_bbox"]],
                        },
                        "verifier": None,
                        "decision": {"action": "review", "reason": "verifier_request_failed"},
                        "model_provenance": {
                            "verifier_model": model_id,
                            "verifier_prompt": "anatomy4-verify-crop-v1",
                            "verifier_prompt_hash": prompt_id,
                            "policy_id": policies[proposal["_detector_key"]].get("policy_id"),
                        },
                        "crop": item["crop_meta"],
                        "latency_ms": call_record.get("latency_ms"),
                        "usage": call_record.get("usage") or {},
                        "original_proposal": public_proposal,
                    }
                )
            return
        try:
            raw_payload = extract_json_payload(str(call_record["content"]))
            raw_items = raw_payload.get("verdicts") if isinstance(raw_payload, dict) else raw_payload
            if not isinstance(raw_items, list) or len(raw_items) != len(items):
                raise ValueError("model returned a mismatched number of verdicts")
        except Exception as exc:
            for item in items:
                proposal = item["proposal"]
                public_proposal = {key: value for key, value in proposal.items() if not str(key).startswith("_")}
                verdicts.append(
                    {
                        "schema": "arq.anatomy4.train-verdict/v1",
                        "proposal_id": item["proposal_id"],
                        "observation_id": str(proposal.get("observation_id") or ""),
                        "photo_id": str(proposal.get("photo_id") or proposal.get("image_id") or ""),
                        "photo_url": proposal["_photo_url"],
                        "class": item["proposed_class"],
                        "split": "train",
                        "human_truth": False,
                        "verification_status": "invalid_response",
                        "verification_error": f"{type(exc).__name__}: {exc}",
                        "detector": {
                            "detector": proposal["_detector_key"],
                            "model": (proposal.get("model") or {}).get("detector"),
                            "proposal_id": proposal["proposal_id"],
                            "detector_score": float(proposal.get("box_score") or 0.0),
                            "detector_bbox_xyxy_norm": [float(value) for value in proposal["detector_bbox"]],
                        },
                        "verifier": None,
                        "decision": {"action": "review", "reason": "verifier_response_invalid"},
                        "model_provenance": {
                            "verifier_model": model_id,
                            "verifier_prompt": "anatomy4-verify-crop-v1",
                            "verifier_prompt_hash": prompt_id,
                            "policy_id": policies[proposal["_detector_key"]].get("policy_id"),
                        },
                        "crop": item["crop_meta"],
                        "latency_ms": call_record.get("latency_ms"),
                        "usage": call_record.get("usage") or {},
                        "raw_verdict": None,
                        "original_proposal": public_proposal,
                    }
                )
            return

        for item, raw_item in zip(items, raw_items):
            proposal = item["proposal"]
            public_proposal = {key: value for key, value in proposal.items() if not str(key).startswith("_")}
            manifest_row = manifest.get(str(proposal.get("observation_id") or ""), {})
            known_names = [
                str(proposal.get(key) or manifest_row.get(key) or "")
                for key in ("taxon_name", "common_name")
            ]
            detector_key = str(proposal["_detector_key"])
            policy = policies[detector_key]
            try:
                verdict = parse_verdict_item(
                    raw_item,
                    proposed_class=item["proposed_class"],
                    known_names=known_names,
                )
            except Exception as exc:
                verdicts.append(
                    {
                        "schema": "arq.anatomy4.train-verdict/v1",
                        "proposal_id": item["proposal_id"],
                        "observation_id": str(proposal.get("observation_id") or ""),
                        "photo_id": str(proposal.get("photo_id") or proposal.get("image_id") or ""),
                        "photo_url": proposal["_photo_url"],
                        "class": item["proposed_class"],
                        "split": "train",
                        "human_truth": False,
                        "verification_status": "invalid_response",
                        "verification_error": f"{type(exc).__name__}: {exc}",
                        "detector": {
                            "detector": detector_key,
                            "model": (proposal.get("model") or {}).get("detector"),
                            "proposal_id": proposal["proposal_id"],
                            "detector_score": float(proposal.get("box_score") or 0.0),
                            "detector_bbox_xyxy_norm": [float(value) for value in proposal["detector_bbox"]],
                        },
                        "verifier": None,
                        "decision": {"action": "review", "reason": "verifier_response_invalid"},
                        "model_provenance": {
                            "verifier_model": model_id,
                            "verifier_prompt": "anatomy4-verify-crop-v1",
                            "verifier_prompt_hash": prompt_id,
                            "policy_id": policy.get("policy_id"),
                        },
                        "crop": item["crop_meta"],
                        "latency_ms": call_record.get("latency_ms"),
                        "usage": call_record.get("usage") or {},
                        "raw_verdict": raw_item,
                        "original_proposal": public_proposal,
                    }
                )
                continue
            auto_assist, policy_decision = _policy_accepts(proposal, verdict, policy)
            detector_provenance = {
                "detector": detector_key,
                "model": (proposal.get("model") or {}).get("detector"),
                "proposal_id": proposal["proposal_id"],
                "detector_score": float(proposal.get("box_score") or 0.0),
                "detector_bbox_xyxy_norm": [float(value) for value in proposal["detector_bbox"]],
            }
            verdicts.append(
                {
                    "schema": "arq.anatomy4.train-verdict/v1",
                    "proposal_id": item["proposal_id"],
                    "observation_id": str(proposal.get("observation_id") or ""),
                    "photo_id": str(proposal.get("photo_id") or proposal.get("image_id") or ""),
                    "photo_url": proposal["_photo_url"],
                    "class": item["proposed_class"],
                    "split": "train",
                    "human_truth": False,
                    "verification_status": "completed",
                    "detector": detector_provenance,
                    "verifier": {
                        "class": verdict["verdict_class"],
                        "resolved_enough": verdict["resolved_enough"],
                        "bbox_tightness": verdict["bbox_tightness"],
                        "reason": verdict["reason"],
                        "confidence": verdict["confidence"],
                        "valid": verdict["valid"],
                        "invalid_reason": verdict["invalid_reason"],
                        "taxon_evidence": verdict["taxon_evidence"],
                        "invented_box_fields": verdict["invented_box_fields"],
                    },
                    "decision": {
                        "action": "auto_assist" if auto_assist else "review",
                        "policy_decision": policy_decision,
                        "policy_id": policy.get("policy_id"),
                    },
                    "model_provenance": {
                        "verifier_model": model_id,
                        "verifier_prompt": "anatomy4-verify-crop-v1",
                        "verifier_prompt_hash": prompt_id,
                        "verifier_schema": VERIFIER_SCHEMA,
                        "policy_id": policy.get("policy_id"),
                        "policy_schema": POLICY_SCHEMA,
                    },
                    "crop": item["crop_meta"],
                    "latency_ms": call_record.get("latency_ms"),
                    "usage": call_record.get("usage") or {},
                    "raw_verdict": raw_item,
                    "original_proposal": public_proposal,
                }
            )

    verdicts: list[dict[str, Any]] = []
    model_id = settings.model
    prompt_id = prompt_hash("v1")
    request_started = time.monotonic()
    proposal_by_id = {str(proposal["proposal_id"]): proposal for proposal in selected}
    for call in raw_calls:
        prior_items = []
        for item_meta in call.get("items") or []:
            proposal = proposal_by_id.get(str(item_meta.get("proposal_id")))
            if proposal is None:
                continue
            crop_meta = item_meta.get("crop") or {}
            prior_items.append(
                {
                    "proposal": proposal,
                    "proposal_id": str(item_meta["proposal_id"]),
                    "proposed_class": str(item_meta.get("proposed_class") or proposal.get("class_name")),
                    "crop_meta": crop_meta,
                    "crop_path": crop_meta.get("crop_path"),
                }
            )
        if prior_items:
            process_call_record(call, prior_items)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(send_batch, task): task for task in prepared}
        for future in as_completed(futures):
            task = futures[future]
            batch_index = int(task["index"])
            items = task["items"]
            try:
                _, result = future.result()
            except Exception as exc:
                result = {
                    "ok": False,
                    "status": None,
                    "latency_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                    "raw": None,
                    "content": None,
                    "usage": {},
                    "attempts": [],
                }
            call_record: dict[str, Any] = {
                "schema": VERIFIER_SCHEMA,
                "batch_index": batch_index,
                "crop_count": len(items),
                "proposal_ids": [item["proposal_id"] for item in items],
                "items": [
                    {
                        "proposal_id": item["proposal_id"],
                        "photo_id": str(item["proposal"].get("photo_id") or item["proposal"].get("image_id")),
                        "proposed_class": item["proposed_class"],
                        "detector_score": item["proposal"].get("box_score"),
                        "detector": item["proposal"]["_detector_key"],
                        "crop": item["crop_meta"],
                    }
                    for item in items
                ],
                "model": model_id,
                "prompt_version": "anatomy4-verify-crop-v1",
                "prompt_hash": prompt_id,
                "ok": bool(result.get("ok")),
                "status": result.get("status"),
                "latency_ms": result.get("latency_ms"),
                "attempts": result.get("attempts") or [],
                "usage": result.get("usage") or {},
                "request_id": result.get("request_id"),
                "error": result.get("error"),
                "raw": result.get("raw"),
                "content": result.get("content"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            raw_calls.append(call_record)
            with raw_path.open("a") as handle:
                handle.write(json.dumps(call_record, sort_keys=True) + "\n")

            process_call_record(call_record, items)

    # If a request failed and was retried on resume, the later result wins.
    effective_verdicts: dict[str, dict[str, Any]] = {}
    for row in verdicts:
        effective_verdicts[str(row["proposal_id"])] = row
    verdicts = list(effective_verdicts.values())
    verdicts.sort(key=lambda row: (row["class"], row["proposal_id"]))
    rng = random.Random(audit_seed)
    assisted_candidates: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    for record in verdicts:
        proposal = record["original_proposal"]
        detector_key = record["detector"]["detector"]
        policy = policies[detector_key]
        auto_assist = record["decision"]["action"] == "auto_assist"
        if auto_assist:
            label = {
                "schema": "arq.anatomy4.assisted-label/v1",
                "proposal_id": record["proposal_id"],
                "observation_id": record["observation_id"],
                "photo_url": record["photo_url"],
                "class": record["class"],
                "bbox_xyxy_norm": [float(value) for value in proposal["detector_bbox"]],
                "source": ASSISTED_SOURCE,
                "review_state": ASSISTED_REVIEW_STATE,
                "human_truth": False,
                "audit_sample": False,
                "model_id": model_id,
                "prompt": "anatomy4-verify-crop-v1",
                "policy_id": policy.get("policy_id"),
                "detector_score": record["detector"]["detector_score"],
                "detector_provenance": record["detector"],
                "verifier_provenance": record["model_provenance"],
                "verifier_verdict": record["verifier"],
                "policy_action": "auto_assist",
                "split": "train",
            }
            assisted_candidates.append(label)
        else:
            review_queue.append(
                {
                    "schema": "arq.anatomy4.review-queue-item/v1",
                    "proposal_id": record["proposal_id"],
                    "observation_id": record["observation_id"],
                    "photo_url": record["photo_url"],
                    "class": record["class"],
                    "action": "review",
                    "human_truth": False,
                    "verification_status": record.get("verification_status", "completed"),
                    "original_proposal": proposal,
                    "verifier_verdict": record["verifier"],
                    "verifier_decision": record["decision"],
                    "model_provenance": record["model_provenance"],
                }
            )

    audit_count = math.ceil(len(assisted_candidates) * max(0.0, min(1.0, audit_fraction)))
    audit_indices = set(rng.sample(range(len(assisted_candidates)), audit_count)) if audit_count else set()
    for index, label in enumerate(assisted_candidates):
        label["audit_sample"] = index in audit_indices
        if label["audit_sample"]:
            label["audit_sample_seed"] = audit_seed

    def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    write_jsonl(verdict_path, verdicts)
    write_jsonl(output / "assisted-labels.jsonl", assisted_candidates)
    write_jsonl(output / "review-queue.jsonl", review_queue)

    counts: dict[str, Any] = {}
    for class_name in sorted({row["class"] for row in verdicts}):
        class_rows = [row for row in verdicts if row["class"] == class_name]
        assisted = [row for row in assisted_candidates if row["class"] == class_name]
        reviewed = [row for row in review_queue if row["class"] == class_name]
        tp_estimate = 0
        support = policies[class_rows[0]["detector"]["detector"]]["per_class"][class_name].get("measured_support") or {}
        train_support = support.get("assisted_train") or {}
        p = train_support.get("precision")
        ci = train_support.get("precision_ci95") or [None, None]
        tp_estimate = len(assisted) * float(p or 0.0)
        counts[class_name] = {
            "selected": len(class_rows),
            "assisted_label_candidates": len(assisted),
            "review_queue": len(reviewed),
            "detector_score_min": min(
                float(row["detector"]["detector_score"]) for row in class_rows
            ),
            "detector_score_max": max(
                float(row["detector"]["detector_score"]) for row in class_rows
            ),
            "assisted_train_precision_point_estimate": p,
            "assisted_train_precision_ci95": ci,
            "estimated_assisted_true_yield_point": round(tp_estimate, 2),
            "estimated_assisted_true_yield_ci95": [
                None if ci[0] is None else round(len(assisted) * ci[0], 2),
                None if ci[1] is None else round(len(assisted) * ci[1], 2),
            ],
            "audit_sample": sum(1 for label in assisted if label["audit_sample"]),
            "policy_id": policies[class_rows[0]["detector"]["detector"]].get("policy_id"),
        }

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
    latency_values: list[float] = []
    for call in raw_calls:
        usage = call.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total_usage[key] += int(usage.get(key) or 0)
        total_usage["cost"] += float(usage.get("cost") or 0.0)
        if call.get("latency_ms") is not None:
            latency_values.append(float(call["latency_ms"]))

    stats = {
        "schema": RUN_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_metadata": dict(selection_metadata or {}),
        "selection_seed": (selection_metadata or {}).get("seed"),
        "blind_test_touched": False,
        "human_truth_written": False,
        "concurrency": {"max_workers": max_workers, "batch_size": batch_size},
        "model": settings.public(),
        "verifier_schema": VERIFIER_SCHEMA,
        "verifier_prompt": "anatomy4-verify-crop-v1",
        "verifier_prompt_hash": prompt_id,
        "policy_ids": {
            detector: policy.get("policy_id") for detector, policy in policies.items()
        },
        "selected": len(proposals),
        "verified": len(verdicts),
        "skipped": skipped,
        "calls": len(raw_calls),
        "crop_requests": sum(int(call.get("crop_count") or 0) for call in raw_calls),
        "failed_calls": sum(1 for call in raw_calls if not call.get("ok")),
        "invalid_response_proposals": sum(
            1 for row in verdicts if row.get("verification_status") == "invalid_response"
        ),
        "retries": sum(max(0, len(call.get("attempts") or []) - 1) for call in raw_calls),
        "latency_ms": {
            "mean": round(sum(latency_values) / len(latency_values), 1) if latency_values else None,
            "p95": round(sorted(latency_values)[int(0.95 * (len(latency_values) - 1))], 1)
            if latency_values
            else None,
            "max": max(latency_values) if latency_values else None,
            "elapsed_wall_seconds": round(time.monotonic() - request_started, 1),
        },
        "usage": total_usage,
        "per_class": counts,
        "assisted_label_candidates": len(assisted_candidates),
        "review_queue_items": len(review_queue),
        "curator_audit_sample": {
            "seed": audit_seed,
            "fraction": audit_fraction,
            "size": audit_count,
            "proposal_ids": [label["proposal_id"] for label in assisted_candidates if label["audit_sample"]],
            "human_truth": False,
        },
        "artifacts": {
            "verdicts": str(verdict_path),
            "raw_calls": str(raw_path),
            "assisted_labels": str(output / "assisted-labels.jsonl"),
            "review_queue": str(output / "review-queue.jsonl"),
        },
    }
    (output / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def find_candidate_image(proposal: Mapping[str, Any], images_dir: str | Path) -> Path | None:
    root = Path(images_dir)
    candidates = [str(proposal.get("image_path") or "")]
    candidates.extend(
        [
            f"{proposal.get('image_id')}.jpg",
            f"{proposal.get('image_id')}.jpeg",
            f"{proposal.get('photo_id')}.jpg",
            f"{proposal.get('photo_id')}.jpeg",
        ]
    )
    for raw in candidates:
        if not raw:
            continue
        candidate = Path(raw)
        local = root / candidate.name
        if local.exists():
            return local
        if candidate.is_file():
            return candidate
    return None
