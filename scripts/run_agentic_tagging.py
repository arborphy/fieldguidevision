"""Agentic statement-tagging batch runner.

Compiles a logical statement against a DeVo source (Dirr primary), runs the
agentic scene-resolution loop (Gemma ↔ Grounding-DINO) over a Ward Pound Ridge
observation subset, and emits:

  1. an ``arq.notice-proposal-batch/v1`` artifact (one proposal per grounded organ),
  2. a feature-tags JSON (DeVo tags with annotator=model provenance),
  3. an ``iteration_trace.jsonl`` — the experiment's primary evidence.

Examples
--------
Dry-run with a mock agent (no GPU) to validate the loop and compiler::

    uv run python scripts/run_agentic_tagging.py \
        --statement "all observations which show a deciduous tree (highest candidate is one that shows the entire specimen)" \
        --devo dirr --limit 12 --mock-agent --output-dir scratch/p17

Real run on MPS with local Gemma 4 12B::

    uv run python scripts/run_agentic_tagging.py \
        --statement "..." --devo dirr --limit 40 --output-dir scratch/p17
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmentation.agentic_resolver import (  # noqa: E402
    AgenticSceneResolver,
    OUTCOME_RESOLVED,
    resolution_to_trace_record,
)
from segmentation.devo_adapter import DirrDevoAdapter  # noqa: E402
from segmentation.pipeline import (  # noqa: E402
    BatchResult,
    CaptureFrame,
    NoticeProposal,
    ProposalFailure,
    ProposalQuality,
    _bbox,
    _category_id,
    _proposal_id,
    _quality,
    _rank_top_k,
    _valid_polygon,
    load_inaturalist_frames,
)
from segmentation.statement_compiler import StatementCompiler  # noqa: E402

logger = logging.getLogger("run_agentic_tagging")

DEFAULT_SOURCE = ROOT / "observations_pound_ridge" / "ward_pound_ridge_species.csv"
MODEL_NAME = "grounding-dino+sam2.1"


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def load_subset(
    source: Path,
    *,
    candidate_species: set[str],
    limit: int,
    research_only: bool = True,
    obs_id_prefix: str = "inat-",
) -> tuple[CaptureFrame, ...]:
    """Deterministic subset: research-grade obs whose iNat taxon is a candidate
    species. The candidate set is the statement's *taxon scope* — broad enough that
    the agent still has to verify the tag targets from the image (e.g. all woody
    species, so deciduousness is assessed, not pre-filtered).

    ``observation_id`` is emitted with ``obs_id_prefix`` so it matches the local
    DB's ``arborphy_id`` convention (``inat-{id}``) — the batch import validates
    that every referenced observation already exists under that id.
    """
    frames: list[CaptureFrame] = []
    import csv

    with source.open(newline="") as fh:
        for row in csv.DictReader(fh):
            image_url = (row.get("image_url") or "").strip()
            species = (row.get("scientific_name") or row.get("species_guess") or "").strip()
            raw_id = (row.get("id") or row.get("observation_id") or "").strip()
            if not image_url or not species or species not in candidate_species:
                continue
            if research_only and (row.get("quality_grade") or "").strip() != "research":
                continue
            obs_id = raw_id if raw_id.startswith(obs_id_prefix) else f"{obs_id_prefix}{raw_id}"
            frames.append(
                CaptureFrame(
                    observation_id=obs_id,
                    photo_id=(row.get("photo_id") or "").strip() or None,
                    image_url=image_url,
                    subject_id=species,
                    species_name=species,
                    license=(row.get("license") or "").strip() or None,
                    captured_at=(row.get("observed_on") or "").strip() or None,
                    frame_index=0,
                )
            )
            if len(frames) >= limit:
                break
    return tuple(frames)


def _hash_image(image) -> str:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "sha256:" + hashlib.sha256(buf.getvalue()).hexdigest()


def _load_image(url: str):
    """Load an image from a URL or file:// path without the torch stack."""
    from io import BytesIO
    from urllib.request import Request, urlopen

    from PIL import Image

    if url.startswith("file://"):
        return Image.open(Path(url.removeprefix("file://"))).convert("RGB")
    request = Request(url, headers={"User-Agent": "ArborphyFieldguidevision/0.1"})
    with urlopen(request, timeout=30) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


# ---------------------------------------------------------------------------
# Agent + segmenter construction
# ---------------------------------------------------------------------------


def build_real_agent(device: str | None):
    """Local Gemma 4 12B VLM + Grounding-DINO/SAM segmenter."""
    from segmentation.local_vlm_evaluator import LocalVlmEvaluator
    from segmentation.grounded_sam2 import GroundedSam2Segmenter

    vlm_eval = LocalVlmEvaluator(device=device)
    segmenter = GroundedSam2Segmenter(device=device)

    def vlm(image, prompt_text: str) -> str:
        return vlm_eval.generate_text(image, prompt_text)

    def segment(image, prompts):
        return segmenter.segment_image(image, prompts)

    return vlm, segment


def build_mock_agent():
    """Deterministic mock agent for GPU-free loop validation."""
    from segmentation.pipeline import RawSegment

    def vlm(image, prompt_text: str) -> str:
        if "Decide ONLY whether it plausibly shows" in prompt_text:
            return json.dumps({"action": "ground", "is_target_plausible": True,
                               "rationale": "mock: plausible"})
        if "directing an object-detection tool" in prompt_text:
            return json.dumps({"action": "ground", "rec_phrase": "whole plant",
                               "rationale": "mock: ground whole plant"})
        # assess
        return json.dumps({
            "action": "resolve",
            "feature_assessments": [
                {"feature_name": "growth_habit_types", "value_label": "Tree",
                 "confidence": 0.9, "abstain": False, "rationale": "mock"},
                {"feature_name": "phenology_types", "value_label": "Deciduous",
                 "confidence": 0.85, "abstain": False, "rationale": "mock"},
            ],
            "rationale": "mock: resolved",
        })

    def segment(image, prompts):
        # A centered polygon covering the middle 60% of the frame.
        poly = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))
        return (RawSegment(label=prompts[0], polygon=poly, detection_confidence=0.5,
                           mask_quality=0.7, prompt=prompts[0], detector_label=prompts[0]),)

    return vlm, segment


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic statement-tagging batch runner.")
    parser.add_argument("--statement", required=True)
    parser.add_argument("--devo", default="dirr", choices=("dirr",))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--mock-agent", action="store_true", help="No GPU; validate the loop.")
    parser.add_argument("--any-quality", action="store_true", help="Include non-research obs.")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--area-instance-id",
        default=None,
        help="Target area instance for the notice-batch input_manifest (required for "
             "POST /fieldguidevision/batches/import). The artifact is rejected if this "
             "does not match the import's area_instance_id.",
    )
    parser.add_argument(
        "--taxon-scope",
        default="Tree,Shrub",
        help="Comma-separated Dirr value labels defining the candidate taxon scope "
             "(OR-matched). Broad scope keeps the agent honest: it verifies the tag "
             "targets from the image rather than pre-filtering to known matches.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    devo = DirrDevoAdapter()
    compiler = StatementCompiler(devo)
    plan = compiler.compile(args.statement)
    logger.info("Compiled plan: tag_targets=%s organs=%s",
                [(t.feature_name, t.value_label) for t in plan.tag_targets],
                [(o.organ, o.required) for o in plan.organs])

    # Candidate taxon scope: species asserting ANY of the scope value labels (OR).
    # Broad by design so the agent verifies the tag targets from the image.
    scope_labels = [s.strip() for s in args.taxon_scope.split(",") if s.strip()]
    candidate_species: set[str] = set()
    for lbl in scope_labels:
        candidate_species.update(t.scientific_name for t in devo.resolve_taxa_by_feature_values([lbl]))
    frames = load_subset(
        args.source, candidate_species=candidate_species,
        limit=args.limit, research_only=not args.any_quality,
    )
    logger.info("Subset: %d research-grade observations (scope=%s -> %d candidate species)",
                len(frames), scope_labels, len(candidate_species))
    if not frames:
        logger.error("Empty subset — check value labels / quality filter.")
        sys.exit(1)

    if args.mock_agent:
        vlm, segment = build_mock_agent()
    else:
        vlm, segment = build_real_agent(args.device)

    resolver = AgenticSceneResolver(vlm=vlm, segment=segment, max_turns=args.max_turns)

    proposals: list[NoticeProposal] = []
    failures: list[ProposalFailure] = []
    tag_records: list[dict] = []
    image_hashes: dict[str, str] = {}
    trace_path = args.output_dir / "iteration_trace.jsonl"

    from PIL import Image  # noqa: F401

    with trace_path.open("w") as trace_fh:
        for i, frame in enumerate(frames):
            logger.info("[%d/%d] %s (%s)", i + 1, len(frames), frame.observation_id, frame.species_name)
            try:
                image = _load_image(frame.image_url)
            except Exception as exc:
                failures.append(ProposalFailure(frame.observation_id, frame.image_url,
                                                MODEL_NAME, type(exc).__name__, str(exc)))
                continue
            try:
                res = resolver.resolve(image, plan, frame)
            except Exception as exc:  # never let one photo kill the batch
                logger.exception("Resolver failed on %s", frame.observation_id)
                failures.append(ProposalFailure(frame.observation_id, frame.image_url,
                                                MODEL_NAME, type(exc).__name__, str(exc)))
                continue

            trace_fh.write(json.dumps(resolution_to_trace_record(res, frame, plan)) + "\n")
            trace_fh.flush()
            logger.info("  -> %s in %d turns; organs=%s tags=%d",
                        res.outcome, res.turns_used, [s.label for s in res.grounded_segments],
                        len(res.feature_tags))

            # Notice proposals from grounded organs.
            sha = _hash_image(image)
            image_hashes[frame.observation_id] = sha
            for idx, seg in enumerate(res.grounded_segments):
                if not _valid_polygon(seg.polygon):
                    continue
                proposals.append(NoticeProposal(
                    proposal_id=_proposal_id(frame, seg, idx),
                    observation_id=frame.observation_id,
                    subject_id=frame.subject_id,
                    image_url=frame.image_url,
                    photo_id=frame.photo_id,
                    species_name=frame.species_name,
                    license=frame.license,
                    frame_index=frame.frame_index,
                    category_concept_id=_category_id(seg.label),
                    category_label=seg.label,
                    polygon=seg.polygon,
                    bbox=_bbox(seg.polygon),
                    quality=_quality(seg),
                    model_name=MODEL_NAME,
                    model_version="agentic|gemma-4-12b|grounding-dino+sam2.1",
                    prompt=seg.prompt or seg.label,
                    detector_label=seg.detector_label,
                ))

            # Feature tags (only on resolved photos).
            if res.outcome == OUTCOME_RESOLVED:
                for tag in res.feature_tags:
                    tag_records.append({
                        "observation_id": frame.observation_id,
                        "species_name": frame.species_name,
                        "devo_name": plan.devo_name,
                        "feature_name": tag.feature_name,
                        "value_label": tag.value_label,
                        "confidence": tag.confidence,
                        "organ": tag.organ,
                        "annotator": "model",
                        "model": "gemma-4-12b-it",
                        "rationale": tag.rationale,
                        "image_sha256": sha,
                    })

    # Attach image hashes (the batch import requires every observation hashed).
    hashed_frames = tuple(
        CaptureFrame(
            observation_id=f.observation_id, image_url=f.image_url, subject_id=f.subject_id,
            image_sha256=image_hashes.get(f.observation_id, ""), local_path=f.local_path,
            species_name=f.species_name, photo_id=f.photo_id, license=f.license,
            captured_at=f.captured_at, frame_index=f.frame_index,
        )
        for f in frames
    )

    batch = BatchResult(
        source_name=str(args.source),
        model_name=MODEL_NAME,
        model_version="agentic|gemma-4-12b|grounding-dino+sam2.1",
        prompts=plan.seed_prompts(),
        selection_scope="category",
        frames=hashed_frames,
        proposals=_rank_top_k(tuple(proposals), top_k=5, selection_scope="category"),
        failures=tuple(failures),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    artifact_path = args.output_dir / "notice_batch.json"
    batch.write(artifact_path)

    # Inject the input_manifest the batch import requires (source area + per-frame
    # image hashes). BatchResult.to_dict() is deliberately minimal; the manifest is
    # an import-time concern, so it is layered on here rather than in the schema.
    if args.area_instance_id:
        artifact = json.loads(artifact_path.read_text())
        artifact["input_manifest"] = {
            "source_area": {"instance_id": args.area_instance_id},
            "frames": [
                {"observation_id": f.observation_id, "image_sha256": f.image_sha256}
                for f in hashed_frames
                if f.image_sha256
            ],
        }
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        logger.info("Wrote input_manifest for area %s (%d hashed frames)",
                    args.area_instance_id, len(artifact["input_manifest"]["frames"]))
    else:
        logger.warning("No --area-instance-id: artifact lacks input_manifest and will "
                       "be rejected by batches/import. Re-run with --area-instance-id to import.")

    tags_path = args.output_dir / "feature_tags.json"
    tags_path.write_text(json.dumps({
        "statement": args.statement,
        "devo": plan.devo_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tags": tag_records,
    }, indent=2))

    summary = batch.to_dict()["summary"]
    print(
        f"\nDone. {summary['proposals']} organ proposals, {summary['failed_frames']} failed frames.\n"
        f"Feature tags: {len(tag_records)} across {len(frames)} photos.\n"
        f"Artifacts in {args.output_dir}:\n"
        f"  - {artifact_path.name}\n  - {tags_path.name}\n  - {trace_path.name}"
    )


if __name__ == "__main__":
    main()
