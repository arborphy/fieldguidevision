"""Statement compiler for agentic Field Guide Vision tagging.

Turns a logical statement over taxa / features / observations into a
``CompiledPlan`` — the target organs to ground, the DeVo feature values to tag,
and the accept/reject coverage rule. The compiler is DeVo-agnostic: it reads only
``ResolvedFeatureValue.required_organ`` from the active adapter, never a
source-specific schema.

A statement names feature *values* (e.g. "deciduous tree"); the compiler resolves
them through the adapter to the feature-value targets, the organs required to
assess them, and which organs are required vs. optional for the accept gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .devo_adapter import (
    DeVoAdapter,
    ORGAN_WHOLE_PLANT,
    ResolvedFeatureValue,
)


@dataclass(frozen=True)
class OrganRequirement:
    """An organ to ground, and whether it gates acceptance."""

    organ: str
    required: bool  # required organs gate the accept rule; optional ones extend tagging
    min_area_fraction: float = 0.02  # polygon area band for "usable" grounding
    min_detection_confidence: float = 0.30


@dataclass(frozen=True)
class CompiledPlan:
    """The executable form of a logical statement."""

    statement: str
    devo_name: str
    tag_targets: tuple[ResolvedFeatureValue, ...]  # feature values to assert on accept
    organs: tuple[OrganRequirement, ...]  # required + optional organs to ground
    prefer_whole_specimen: bool = False  # rank higher when whole-plant organ grounds
    alias_notes: tuple[tuple[str, str], ...] = ()  # (phrase, "value + value") resolutions

    @property
    def required_organs(self) -> tuple[OrganRequirement, ...]:
        return tuple(o for o in self.organs if o.required)

    @property
    def optional_organs(self) -> tuple[OrganRequirement, ...]:
        return tuple(o for o in self.organs if not o.required)

    def seed_prompts(self) -> tuple[str, ...]:
        """Initial Grounding-DINO REC seeds, one per organ to ground."""
        return tuple(o.organ for o in self.organs)


# Conservative default organ requirements. Leaf-level features need a clearer,
# larger, better-lit organ than habit-level ones, hence the tighter bands.
_DEFAULT_ORGAN_BANDS: dict[str, tuple[float, float]] = {
    ORGAN_WHOLE_PLANT: (0.05, 0.30),  # habit: needs a substantial specimen in frame
    "leaf blade": (0.03, 0.35),  # leaf detail: a clear single leaf suffices
    "flower": (0.02, 0.30),
}


# Words that, when they immediately follow a value label, indicate the label is
# being used as an organ-scope modifier (not a botanical feature value). "Entire
# specimen/plant" is about the whole organism's extent, not the leaf-margin value
# "Entire". Habit nouns (tree/shrub) are NOT followers: "deciduous tree" is a real
# phenology claim about the tree.
_ORGAN_FOLLOWERS = ("specimen", "plant", "organism")


class StatementCompiler:
    """Compiles logical statements against a DeVo adapter."""

    def __init__(self, adapter: DeVoAdapter) -> None:
        self.adapter = adapter
        # Index feature values by normalized label for statement lookup.
        self._by_value: dict[str, ResolvedFeatureValue] = {
            self._norm(v.value_label): v for v in adapter.list_features()
        }

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _match_values(self, statement: str) -> tuple[ResolvedFeatureValue, ...]:
        """Exact-label matching for the deterministic ``compile`` fallback.

        Matches Dirr value labels used as standalone descriptors; a label is skipped
        when it modifies an organ word (``entire specimen`` ≠ the leaf-margin value
        ``Entire``). Open-arity natural-language statements go through
        ``compile_with_vlm`` instead — this path is the fast deterministic fallback.
        """
        norm_stmt = f" {self._norm(statement)} "
        matched: list[ResolvedFeatureValue] = []
        for norm_label, fv in self._by_value.items():
            if not norm_label:
                continue
            for m in re.finditer(re.escape(norm_label), norm_stmt):
                end = m.end()
                tail = norm_stmt[end : end + 16].strip()
                next_word = tail.split(" ", 1)[0] if tail else ""
                if next_word in _ORGAN_FOLLOWERS:
                    continue  # label is modifying an organ word, not a feature value
                matched.append(fv)
                break

        # Deduplicate by (feature, value) preserving order.
        seen: set[tuple[str, str]] = set()
        out: list[ResolvedFeatureValue] = []
        for fv in matched:
            key = (fv.feature_name, fv.value_label)
            if key not in seen:
                seen.add(key)
                out.append(fv)
        return tuple(out)

    def compile(self, statement: str) -> CompiledPlan:
        """Compile a natural-language logical statement into an executable plan."""
        tag_targets = self._match_values(statement)
        if not tag_targets:
            # Help the user toward a statable statement: list a sample of the values
            # the active DeVo actually knows, grouped by feature.
            by_feature: dict[str, list[str]] = {}
            for fv in self.adapter.list_features():
                by_feature.setdefault(fv.feature_name, []).append(fv.value_label)
            sample = "; ".join(
                f"{feat} ({', '.join(sorted(set(vals))[:6])}…)" if len(set(vals)) > 6
                else f"{feat} ({', '.join(sorted(set(vals)))})"
                for feat, vals in list(by_feature.items())[:6]
            )
            raise ValueError(
                f"The statement matched no {self.adapter.devo_name} feature values. "
                f"This DeVo only knows these feature values — name one or more of them. "
                f"Available: {sample}. (Common phrasings like \"leaves of three\" or "
                f"\"compound leaves\" also work.)"
            )

        prefer_whole = bool(
            re.search(r"(entire|whole|full)\s+(specimen|plant)|habit", statement, re.IGNORECASE)
        )

        # Organs required to assess the tag targets. If any target is a
        # classification feature (whole plant), that organ is required; leaf/floral
        # features make their organs required too unless the statement prefers the
        # whole specimen (then leaf is optional-extends-tagging).
        organs: dict[str, OrganRequirement] = {}
        for fv in tag_targets:
            organ = fv.required_organ
            min_area, min_conf = _DEFAULT_ORGAN_BANDS.get(organ, (0.02, 0.30))
            required = True
            if prefer_whole and organ != ORGAN_WHOLE_PLANT:
                # Whole-specimen statements gate on the habit organ; finer organs
                # become optional so their absence doesn't reject a good habit shot.
                required = False
            if organ in organs:
                # A required organ requirement wins over an optional one.
                existing = organs[organ]
                if required and not existing.required:
                    organs[organ] = OrganRequirement(organ, True, min_area, min_conf)
            else:
                organs[organ] = OrganRequirement(organ, required, min_area, min_conf)

        # Whole-specimen preference always grounds the habit organ even when the
        # tag targets are leaf-level (a habit shot is the ranking preference).
        if prefer_whole and ORGAN_WHOLE_PLANT not in organs:
            min_area, min_conf = _DEFAULT_ORGAN_BANDS[ORGAN_WHOLE_PLANT]
            organs[ORGAN_WHOLE_PLANT] = OrganRequirement(
                ORGAN_WHOLE_PLANT, True, min_area, min_conf
            )

        return CompiledPlan(
            statement=statement,
            devo_name=self.adapter.devo_name,
            tag_targets=tag_targets,
            organs=tuple(organs.values()),
            prefer_whole_specimen=prefer_whole,
        )

    # -- Open-arity compilation via a VLM --------------------------------------

    def _vocabulary_prompt_block(self) -> str:
        """Render the active DeVo's feature catalog as VLM context."""
        by_feature: dict[str, list[tuple[str, str]]] = {}
        for fv in self.adapter.list_features():
            by_feature.setdefault(fv.feature_name, []).append((fv.value_label, fv.required_organ))
        lines = []
        for feature_name, vals in sorted(by_feature.items()):
            val_str = ", ".join(f"{v} [{organ}]" for v, organ in sorted(set(vals)))
            lines.append(f"- {feature_name}: {val_str}")
        return "\n".join(lines)

    def compile_with_vlm(self, statement: str, vlm, image=None) -> CompiledPlan:
        """Open-arity compilation: let Gemma translate a free-form logic phrase into
        the DeVo's feature values + the organs to ground.

        ``vlm`` is a callable ``(image, prompt_text) -> raw_text``; ``image`` may be
        None (compilation is text-only). Gemma is given the full feature catalog and
        asked to emit structured JSON. The keyword ``compile`` path remains as a fast
        deterministic fallback for exact-label statements and for tests.
        """
        import json

        # Compilation is text-only; supply a neutral blank image for the multimodal
        # processor when none is given.
        if image is None:
            from PIL import Image

            image = Image.new("RGB", (8, 8), (240, 240, 240))

        catalog = self._vocabulary_prompt_block()
        prompt = (
            "You are the compilation agent for a botanical photo-tagging system. A user "
            "writes a free-form logical statement about which plant observations to find "
            "and what to tag. Translate it into the controlled vocabulary below.\n\n"
            f"User statement: \"{statement}\"\n\n"
            f"Controlled vocabulary ({self.adapter.devo_name}). Each value is listed with "
            f"the organ required to assess it in brackets:\n{catalog}\n\n"
            "Decide:\n"
            "1. tag_targets — the feature VALUES to assert. Use ONLY value labels from the "
            "catalog above (exact strings). If the statement implies a concept the catalog "
            "can't express (e.g. \"leaves of three\" → Compound + Palmate), pick the closest "
            "catalog values and say so.\n"
            "2. organs — which plant organs must be grounded to assess those values, and "
            "whether each is required (gates acceptance) or optional (extends tagging).\n"
            "3. prefer_whole_specimen — true if the statement favors whole-plant views.\n"
            "4. notes — a one-line plain-language reading of what will be tagged, including "
            "any approximation you made.\n\n"
            "Reply with strictly valid JSON:\n"
            "{\n"
            '  "tag_targets": [{"feature_name": "...", "value_label": "..."}],\n'
            '  "organs": [{"organ": "whole plant"|"leaf blade"|"flower", "required": true|false}],\n'
            '  "prefer_whole_specimen": true|false,\n'
            '  "notes": "..."\n'
            "}"
        )
        raw = vlm(image, prompt)
        return self._plan_from_vlm_json(statement, raw)

    def _plan_from_vlm_json(self, statement: str, raw: str) -> CompiledPlan:
        import json
        import re as _re

        m = _re.search(r"```json\s*(.*?)\s*```", raw, _re.DOTALL)
        text = m.group(1) if m else raw
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        data = json.loads(text)

        # Resolve each VLM-chosen (feature, value) to the adapter's canonical
        # ResolvedFeatureValue (which carries the organ + definition). Skip unknowns.
        catalog = {(fv.feature_name, fv.value_label): fv for fv in self.adapter.list_features()}
        by_value = {fv.value_label: fv for fv in self.adapter.list_features()}
        tag_targets: list[ResolvedFeatureValue] = []
        for t in data.get("tag_targets", []):
            key = (str(t.get("feature_name")), str(t.get("value_label")))
            fv = catalog.get(key) or by_value.get(str(t.get("value_label")))
            if fv is not None:
                tag_targets.append(fv)
        if not tag_targets:
            raise ValueError(f"VLM compilation produced no resolvable feature values: {raw[:200]}")

        organs: list[OrganRequirement] = []
        for o in data.get("organs", []):
            organ = str(o.get("organ", ORGAN_WHOLE_PLANT))
            min_area, min_conf = _DEFAULT_ORGAN_BANDS.get(organ, (0.02, 0.30))
            organs.append(OrganRequirement(organ, bool(o.get("required", True)), min_area, min_conf))
        if not organs:
            # Default: the organs implied by the tag targets, all required.
            organs = [
                OrganRequirement(fv.required_organ, True, *_DEFAULT_ORGAN_BANDS.get(fv.required_organ, (0.02, 0.30)))
                for fv in tag_targets
            ]

        notes = str(data.get("notes", ""))
        return CompiledPlan(
            statement=statement,
            devo_name=self.adapter.devo_name,
            tag_targets=tuple(tag_targets),
            organs=tuple(organs),
            prefer_whole_specimen=bool(data.get("prefer_whole_specimen", False)),
            alias_notes=(("statement", notes),) if notes else (),
        )
