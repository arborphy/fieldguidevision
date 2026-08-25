"""Agentic scene-resolution loop for Field Guide Vision batch tagging.

Gemma is the scene-resolution agent; Grounding-DINO is its tool. Per photo:

1. **Cheap look.** Gemma decides whether the photo plausibly contains the target
   *before* the detector is fired. Non-starters are discarded early.
2. **Gemma writes the REC phrase.** Grounding-DINO runs only on Gemma's
   referring-expression directive.
3. **Null ≠ negative.** An empty detection is a poor term↔REC fit, so Gemma
   rephrases and retries. There is no fixed iteration count — Gemma halts when
   the scene is resolved, when it abstains (with a rationale), or when it
   contradicts the statement. A generous turn ceiling is a safety valve only.
4. On resolution, the same agent assesses the plan's tag targets against the
   grounded organ crops, emitting (feature, value, confidence) or abstaining per
   feature.

Every turn is logged to an iteration trace — the experiment's primary evidence.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from PIL import Image

from .pipeline import CaptureFrame, RawSegment, _polygon_area
from .statement_compiler import CompiledPlan, OrganRequirement

logger = logging.getLogger(__name__)

# Outcome states for one photo.
OUTCOME_RESOLVED = "resolved"
OUTCOME_DISCARDED = "discarded"  # cheap look: target not plausibly present
OUTCOME_ABSTAIN = "abstain"  # Gemma: scene can't support the statement
OUTCOME_CONTRADICT = "contradict"  # Gemma: statement contradicted by the scene
OUTCOME_CEILING = "turn_ceiling"  # safety valve hit (recorded)


@dataclass(frozen=True)
class FeatureTag:
    """One DeVo (feature, value) tag emitted by the agent for a photo."""

    feature_name: str
    value_label: str
    confidence: float
    organ: str
    rationale: str = ""


@dataclass(frozen=True)
class TurnRecord:
    """One agent turn in the iteration trace."""

    turn: int
    phase: str  # "cheap_look" | "ground" | "rephrase" | "assess"
    rec_phrase: str | None
    grounded_organs: tuple[str, ...]
    detector_confidences: tuple[float, ...]
    gemma_rationale: str
    latency_ms: float
    # Full detector signal for grounding turns (None on cheap_look/assess turns).
    # Serialized as a list of per-candidate dicts; see DetectionCandidate.
    candidates: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SceneResolution:
    """The terminal result of the agent loop for one photo."""

    observation_id: str
    outcome: str
    turns: tuple[TurnRecord, ...]
    grounded_segments: tuple[RawSegment, ...]  # organs that grounded (accepted band)
    feature_tags: tuple[FeatureTag, ...]
    rationale: str
    turns_used: int
    hit_turn_ceiling: bool = False


# ---------------------------------------------------------------------------
# Agent decision parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentDecision:
    """Structured output parsed from one Gemma turn."""

    action: str  # "discard" | "ground" | "rephrase" | "resolve" | "abstain" | "contradict"
    rationale: str
    rec_phrase: str | None = None
    target_organ: str | None = None  # which plan organ this REC is grounding
    is_target_plausible: bool = False
    feature_assessments: tuple[dict[str, Any], ...] = ()


def _parse_agent_decision(text: str) -> AgentDecision:
    """Robustly parse a JSON agent decision from Gemma output."""
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        json_str = text[start : end + 1] if start != -1 and end != -1 else text
    try:
        data = json.loads(json_str)
    except Exception:
        return AgentDecision(action="abstain", rationale=f"Parse error: {text[:200]}")
    return AgentDecision(
        action=str(data.get("action", "abstain")).lower(),
        rationale=str(data.get("rationale", "")),
        rec_phrase=data.get("rec_phrase"),
        target_organ=data.get("target_organ"),
        is_target_plausible=bool(data.get("is_target_plausible", False)),
        feature_assessments=tuple(data.get("feature_assessments", []) or ()),
    )


# ---------------------------------------------------------------------------
# The agentic resolver
# ---------------------------------------------------------------------------

# Callable signature for the VLM: (image, prompt_text) -> raw text.
VlmCallable = Callable[[Image.Image, str], str]
# Callable signature for the segmenter. Returns EITHER a bare sequence of
# accepted RawSegments (legacy/mock path) OR a (segments, event) tuple from
# GroundedSam2Segmenter.detect_image — the resolver detects which it got and
# captures the full candidate signal when available.
SegmentCallable = Callable[[Image.Image, Sequence[str]], Any]


# Strictness presets for the assess prompt. The experiment showed this one knob is
# the agentic boundary: permissive over-claims (tags unverifiable features), strict
# abstains on anything not clearly demonstrated.
STRICTNESS_LEVELS = ("permissive", "balanced", "strict")

_ASSESS_DIRECTIVES = {
    "permissive": (
        "Assess each target feature value against the visible organ(s). Confirm a "
        "value when the photo is consistent with it; abstain only when the organ is "
        "not visible at all."
    ),
    "balanced": (
        "Assess each target feature value against the visible organ(s), applying each "
        "value's definition. Confirm a value when the photo reasonably demonstrates its "
        "defining criterion; abstain when the view is genuinely insufficient."
    ),
    "strict": (
        "Assess each target feature value against the visible organ(s), applying each "
        "value's definition STRICTLY and conservatively. Confirm a value ONLY when the "
        "photo clearly demonstrates its defining criterion; otherwise abstain. Key "
        "cautions:\n"
        "- Growth habit: a 'tree' has a single central axis branching well above head "
        "height. A multi-stemmed or low woody plant is a shrub, not a tree — do not "
        "call it a tree unless the single tall trunk is clearly visible.\n"
        "- Phenology (deciduous vs evergreen): a single photo of a plant IN LEAF cannot "
        "establish deciduousness — evergreen broadleaves look identical in leaf. Abstain "
        "unless leafless stems, bud/leaf arrangement, or seasonal context make it "
        "determinable."
    ),
}


@dataclass(frozen=True)
class ResolverConfig:
    """Tunable knobs for the agentic resolver (the experiment's degrees of freedom).

    Every number the loop consults is here so a job can carry it end-to-end.
    Detector/segmenter thresholds live on the segmenter itself; these are the
    loop- and acceptance-level knobs.
    """

    max_turns: int = 8  # safety valve only; Gemma decides termination
    strictness: str = "strict"  # one of STRICTNESS_LEVELS
    vlm_temperature: float | None = None  # None = model default
    # Acceptance band overrides applied to every organ (None = keep the plan's
    # per-organ band from the compiler).
    min_organ_area: float | None = None
    min_organ_confidence: float | None = None


class AgenticSceneResolver:
    """Drives the Gemma ↔ Grounding-DINO loop for one photo at a time."""

    def __init__(
        self,
        *,
        vlm: VlmCallable,
        segment: SegmentCallable,
        max_turns: int = 8,  # safety valve only; Gemma decides termination
        strictness: str = "strict",
        min_organ_area: float | None = None,
        min_organ_confidence: float | None = None,
    ) -> None:
        if strictness not in STRICTNESS_LEVELS:
            raise ValueError(f"strictness must be one of {STRICTNESS_LEVELS}")
        self.vlm = vlm
        self.segment = segment
        self.max_turns = max_turns
        self.strictness = strictness
        self.min_organ_area = min_organ_area
        self.min_organ_confidence = min_organ_confidence

    # -- detector call -------------------------------------------------------

    def _run_segmenter(
        self, image: Image.Image, prompts: Sequence[str]
    ) -> tuple[list[RawSegment], tuple[dict[str, Any], ...]]:
        """Call the segment callable and normalize to (segments, candidate dicts).

        Supports both the legacy ``segments`` return and the signal-rich
        ``(segments, DetectionEvent)`` return from ``GroundedSam2Segmenter``.
        """
        out = self.segment(image, prompts)
        if isinstance(out, tuple) and len(out) == 2 and hasattr(out[1], "candidates"):
            segments, event = out
            cand_dicts = tuple(
                {
                    "detector_label": c.detector_label,
                    "organ": c.organ,
                    "box_score": round(c.box_score, 4),
                    "bbox": [round(v, 4) for v in c.bbox],
                    "nms_kept": c.nms_kept,
                    "mask_quality": round(c.mask_quality, 4) if c.mask_quality is not None else None,
                    "area_fraction": round(c.area_fraction, 4) if c.area_fraction is not None else None,
                    "polygon": [[round(x, 4), round(y, 4)] for x, y in c.polygon],
                    "verdict": c.verdict,
                }
                for c in event.candidates
            )
            return list(segments), cand_dicts
        # Legacy / mock path: bare segments, no candidate signal.
        return list(out), ()

    # -- payload builders ---------------------------------------------------

    def _cheap_look_prompt(self, plan: CompiledPlan, frame: CaptureFrame) -> str:
        targets = ", ".join(f"{t.value_label} ({t.feature_label})" for t in plan.tag_targets)
        species = frame.species_name or frame.subject_id
        return (
            "You are the scene-resolution agent for a botanical photo triage system.\n"
            f"Logical statement: \"{plan.statement}\"\n"
            f"Feature values to assess: {targets}\n"
            f"Reported subject (from the observation record): {species}\n\n"
            "Look at this photo. Decide ONLY whether it plausibly shows the target "
            "subject in a view where those features could be assessed. Do NOT assess "
            "the features yet. Reply with strictly valid JSON:\n"
            "{\n"
            '  "action": "ground" | "discard",\n'
            '  "is_target_plausible": true|false,\n'
            '  "rationale": "<one sentence of visual evidence>"\n'
            "}\n"
            "Use \"discard\" only when the photo clearly cannot show the target "
            "(wrong subject, microscopic view, landscape with no specimen, etc.)."
        )

    def _ground_prompt(
        self,
        plan: CompiledPlan,
        frame: CaptureFrame,
        prior_attempts: Sequence[str],
        grounded_so_far: Sequence[str],
    ) -> str:
        targets = ", ".join(t.value_label for t in plan.tag_targets)
        organs = ", ".join(o.organ for o in plan.organs)
        attempts = (
            "Prior REC phrases that yielded no usable detection: " + "; ".join(prior_attempts) + ". "
            if prior_attempts
            else ""
        )
        grounded = (
            "Already grounded organs: " + ", ".join(grounded_so_far) + ". "
            if grounded_so_far
            else ""
        )
        return (
            "You are directing an object-detection tool (Grounding DINO) by writing "
            "referring-expression (REC) phrases.\n"
            f"Logical statement: \"{plan.statement}\"\n"
            f"Feature values to locate: {targets}\n"
            f"Organs to ground: {organs}\n"
            f"{attempts}{grounded}"
            "A null detection means a poor fit between the term and the REC phrase — "
            "the target may still be present. Write ONE new REC phrase for the next "
            "organ to ground.\n"
            "IMPORTANT detector contract: Grounding DINO detects SHORT noun phrases "
            "best (1-4 words, e.g. \"tree\", \"whole plant\", \"leaf\", \"bark\"). Long "
            "descriptive sentences fail. Put your rich visual reasoning in `rationale`, "
            "but keep `rec_phrase` a SHORT noun phrase. If a short phrase failed, try a "
            "different short synonym or a more specific organ (e.g. \"crown\", \"trunk\", "
            "\"lobed leaf\") rather than a longer sentence.\n"
            "If the scene genuinely cannot support the statement, abstain. Reply with "
            "strictly valid JSON:\n"
            "{\n"
            '  "action": "ground" | "rephrase" | "abstain" | "contradict",\n'
            f'  "target_organ": "<one of: {organs}>",\n'
            '  "rec_phrase": "<SHORT noun phrase, or null>",\n'
            '  "rationale": "<one sentence>"\n'
            "}"
        )

    def _assess_prompt(
        self,
        plan: CompiledPlan,
        frame: CaptureFrame,
        grounded_organs: Sequence[str],
    ) -> str:
        # Give the agent the value definitions as grounding references.
        lines = []
        for t in plan.tag_targets:
            defn = f" — {t.value_definition}" if t.value_definition else ""
            lines.append(f'  - feature "{t.feature_name}" value "{t.value_label}"{defn}')
        target_block = "\n".join(lines)
        directive = _ASSESS_DIRECTIVES[self.strictness]
        return (
            "The following organs are now grounded and visible: "
            + ", ".join(grounded_organs)
            + ".\n"
            + directive
            + "\nReply with strictly valid JSON:\n"
            "{\n"
            '  "action": "resolve",\n'
            '  "feature_assessments": [\n'
            '    {"feature_name": "<name>", "value_label": "<value>", '
            '"confidence": 0.0-1.0, "abstain": false, "rationale": "<short>"}\n'
            "  ],\n"
            '  "rationale": "<one sentence overall>"\n'
            "}\n"
            f"Target feature values:\n{target_block}"
        )

    # -- grounding gate ------------------------------------------------------

    def _accepted_organs(
        self,
        segments: Sequence[RawSegment],
        plan: CompiledPlan,
        target_organ: str | None = None,
    ) -> dict[str, RawSegment]:
        """Map organ -> the best segment meeting its quality band.

        A segment is attributed to an organ two ways: (1) its canonical label
        matches a plan organ exactly, or (2) it grounded from a REC phrase the
        agent wrote *for* ``target_organ`` — the agent grounds one organ per call,
        so a detection from that call counts toward the intended organ even when
        the phrase canonicalizes to "ambiguous organ" (multi-word RECs).

        The resolver-level ``min_organ_area`` / ``min_organ_confidence`` overrides
        (when set) replace the plan's per-organ band — the job's tunable knobs win.
        """
        best: dict[str, RawSegment] = {}
        reqs = {o.organ: o for o in plan.organs}
        for seg in segments:
            organ = seg.label if seg.label in reqs else target_organ
            if organ is None or organ not in reqs:
                continue
            req = reqs[organ]
            min_area = self.min_organ_area if self.min_organ_area is not None else req.min_area_fraction
            min_conf = (
                self.min_organ_confidence
                if self.min_organ_confidence is not None
                else req.min_detection_confidence
            )
            area = _polygon_area(seg.polygon)
            if area < min_area or seg.detection_confidence < min_conf:
                continue
            if organ not in best or seg.detection_confidence > best[organ].detection_confidence:
                # Relabel to the resolved organ so downstream (notice category, tag
                # organ) carries the intent, not the canonicalization fallback.
                best[organ] = RawSegment(
                    label=organ,
                    polygon=seg.polygon,
                    detection_confidence=seg.detection_confidence,
                    mask_quality=seg.mask_quality,
                    stability_score=seg.stability_score,
                    prompt=seg.prompt,
                    detector_label=seg.detector_label,
                )
        return best

    def _required_satisfied(self, accepted: dict[str, RawSegment], plan: CompiledPlan) -> bool:
        return all(req.organ in accepted for req in plan.required_organs)

    def _verdicts_for(
        self,
        cand_dicts: tuple[dict[str, Any], ...],
        plan: CompiledPlan,
        target_organ: str | None,
        newly: dict[str, RawSegment],
        segments: Sequence[RawSegment],
    ) -> tuple[dict[str, Any], ...]:
        """Annotate each candidate with the reason it was accepted or rejected.

        Verdict ladder (first match wins):
          - ``nms_suppressed`` — Grounding-DINO emitted it, NMS deduped it.
          - ``no_usable_mask`` — survived NMS but SAM produced no valid polygon.
          - ``organ_mismatch`` — canonical label matches no plan organ and the
            candidate wasn't attributed to the turn's target organ.
          - ``below_band`` — failed the organ's area/confidence acceptance band.
          - ``accepted`` / ``accepted:<organ>`` — became the organ's best segment.
          - ``outcompeted`` — met the band but lost the organ to a higher-conf box.
        """
        reqs = {o.organ: o for o in plan.organs}
        # Accepted candidates are identifiable by their polygon matching an
        # accepted segment's polygon (the resolver relabels the winner to the organ).
        accepted_keys = {(round(s.detection_confidence, 4), s.detector_label) for s in segments}
        winner_keys = {(round(s.detection_confidence, 4), s.detector_label) for s in newly.values()}

        out: list[dict[str, Any]] = []
        for c in cand_dicts:
            verdict = c.get("verdict") or ""
            if not c.get("nms_kept"):
                verdict = "nms_suppressed"
            elif not c.get("polygon"):
                verdict = "no_usable_mask"
            else:
                organ = c.get("organ") if c.get("organ") in reqs else target_organ
                if organ is None or organ not in reqs:
                    verdict = "organ_mismatch"
                else:
                    req = reqs[organ]
                    min_area = self.min_organ_area if self.min_organ_area is not None else req.min_area_fraction
                    min_conf = (
                        self.min_organ_confidence
                        if self.min_organ_confidence is not None
                        else req.min_detection_confidence
                    )
                    area = c.get("area_fraction") or 0.0
                    conf = c.get("box_score") or 0.0
                    if area < min_area:
                        verdict = f"below_band:area {area:.3f}<{min_area:.3f}"
                    elif conf < min_conf:
                        verdict = f"below_band:conf {conf:.3f}<{min_conf:.3f}"
                    else:
                        key = (round(conf, 4), c.get("detector_label"))
                        verdict = f"accepted:{organ}" if key in winner_keys else "outcompeted"
            out.append({**c, "verdict": verdict})
        return tuple(out)

    # -- the loop ------------------------------------------------------------

    def resolve(self, image: Image.Image, plan: CompiledPlan, frame: CaptureFrame) -> SceneResolution:
        turns: list[TurnRecord] = []
        prior_attempts: list[str] = []
        accepted: dict[str, RawSegment] = {}
        hit_ceiling = False

        # Turn 0: cheap look — discard before the detector is ever fired.
        t0 = time.perf_counter()
        decision = _parse_agent_decision(self.vlm(image, self._cheap_look_prompt(plan, frame)))
        turns.append(
            TurnRecord(
                turn=0,
                phase="cheap_look",
                rec_phrase=None,
                grounded_organs=(),
                detector_confidences=(),
                gemma_rationale=decision.rationale,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        )
        if decision.action == "discard" or not decision.is_target_plausible:
            return SceneResolution(
                observation_id=frame.observation_id,
                outcome=OUTCOME_DISCARDED,
                turns=tuple(turns),
                grounded_segments=(),
                feature_tags=(),
                rationale=decision.rationale,
                turns_used=1,
            )

        # Grounding turns: Gemma writes/rephrases the REC until resolved/abstain.
        for turn_idx in range(1, self.max_turns + 1):
            tN = time.perf_counter()
            ground_decision = _parse_agent_decision(
                self.vlm(image, self._ground_prompt(plan, frame, prior_attempts, list(accepted)))
            )
            action = ground_decision.action

            if action in ("abstain", "contradict"):
                turns.append(
                    TurnRecord(turn_idx, "assess", None, tuple(accepted), (),
                               ground_decision.rationale, (time.perf_counter() - tN) * 1000)
                )
                return SceneResolution(
                    frame.observation_id, OUTCOME_ABSTAIN if action == "abstain" else OUTCOME_CONTRADICT,
                    tuple(turns), tuple(accepted.values()), (), ground_decision.rationale, len(turns),
                )

            rec = ground_decision.rec_phrase or next(
                (o.organ for o in plan.organs if o.organ not in accepted), plan.organs[0].organ
            )
            # Which organ is the agent grounding? Prefer its explicit choice; else the
            # first ungrounded organ (required before optional).
            target_organ = ground_decision.target_organ
            if target_organ not in {o.organ for o in plan.organs}:
                target_organ = next(
                    (o.organ for o in plan.required_organs if o.organ not in accepted),
                    next((o.organ for o in plan.organs if o.organ not in accepted), plan.organs[0].organ),
                )
            segments, cand_dicts = self._run_segmenter(image, (rec,))
            newly = self._accepted_organs(segments, plan, target_organ=target_organ)
            accepted.update(newly)
            prior_attempts.append(rec)
            # Stamp per-candidate verdicts so the UI can show why each box was
            # accepted, NMS-suppressed, or rejected by the organ band.
            cand_dicts = self._verdicts_for(cand_dicts, plan, target_organ, newly, segments)
            turns.append(
                TurnRecord(
                    turn_idx,
                    "ground" if not prior_attempts[:-1] or rec not in prior_attempts[:-1] else "rephrase",
                    rec,
                    tuple(accepted),
                    tuple(round(s.detection_confidence, 3) for s in newly.values()),
                    ground_decision.rationale,
                    (time.perf_counter() - tN) * 1000,
                    cand_dicts,
                )
            )

            if self._required_satisfied(accepted, plan):
                break
        else:
            hit_ceiling = True

        if not self._required_satisfied(accepted, plan):
            # Loop exhausted without satisfying the required organs.
            return SceneResolution(
                frame.observation_id,
                OUTCOME_CEILING if hit_ceiling else OUTCOME_ABSTAIN,
                tuple(turns),
                tuple(accepted.values()),
                (),
                "Required organs did not ground within the turn budget.",
                len(turns),
                hit_turn_ceiling=hit_ceiling,
            )

        # Assess tag targets against the grounded organs.
        tA = time.perf_counter()
        assess = _parse_agent_decision(
            self.vlm(image, self._assess_prompt(plan, frame, list(accepted)))
        )
        turns.append(
            TurnRecord(len(turns), "assess", None, tuple(accepted), (),
                       assess.rationale, (time.perf_counter() - tA) * 1000)
        )
        tags = self._feature_tags_from(assess, plan)
        return SceneResolution(
            frame.observation_id,
            OUTCOME_RESOLVED,
            tuple(turns),
            tuple(accepted.values()),
            tags,
            assess.rationale,
            len(turns),
        )

    def _feature_tags_from(self, assess: AgentDecision, plan: CompiledPlan) -> tuple[FeatureTag, ...]:
        valid = {(t.feature_name, t.value_label): t for t in plan.tag_targets}
        tags: list[FeatureTag] = []
        for fa in assess.feature_assessments:
            key = (str(fa.get("feature_name")), str(fa.get("value_label")))
            if key not in valid:
                continue
            if fa.get("abstain"):
                continue  # abstention is the absence of a tag
            target = valid[key]
            tags.append(
                FeatureTag(
                    feature_name=target.feature_name,
                    value_label=target.value_label,
                    confidence=float(fa.get("confidence", 0.0)),
                    organ=target.required_organ,
                    rationale=str(fa.get("rationale", "")),
                )
            )
        return tuple(tags)


# ---------------------------------------------------------------------------
# Trace serialization (the experiment's primary evidence)
# ---------------------------------------------------------------------------


def resolution_to_trace_record(res: SceneResolution, frame: CaptureFrame, plan: CompiledPlan) -> dict:
    return {
        "observation_id": res.observation_id,
        "species_name": frame.species_name,
        "image_url": frame.image_url,
        "statement": plan.statement,
        "devo": plan.devo_name,
        "outcome": res.outcome,
        "turns_used": res.turns_used,
        "hit_turn_ceiling": res.hit_turn_ceiling,
        "grounded_organs": [s.label for s in res.grounded_segments],
        "feature_tags": [asdict(t) for t in res.feature_tags],
        "rationale": res.rationale,
        "turns": [asdict(t) for t in res.turns],
    }
