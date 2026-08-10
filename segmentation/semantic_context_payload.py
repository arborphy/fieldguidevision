"""Contextual Semantic Payload Builder for Field Guide Vision.

Assembles environmental priors (Location: Tuckaway Ln, Henrico, VA, Phenology,
Azimuth, Camera telemetry), GoBotany DeVo candidate trees of life, and
Anatomical Knowledge Graphs into a structured prompt payload for the async Gemma / VLM agent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from .gobotany_devo_adapter import GoBotanyDevoAdapter, GoBotanyTaxonCandidate
from .knowledge_graph_schema import (
    AnatomicalKnowledgeGraph,
    get_botanical_plant_knowledge_graph,
    get_xbox_controller_knowledge_graph,
)


@dataclass(frozen=True)
class SensorTelemetry:
    azimuth_degrees: float = 0.0
    camera_pitch: float = 0.0
    user_zoom_level: float = 1.0
    camera_motion_delta: str = "stationary"  # stationary | panning | zooming


@dataclass(frozen=True)
class LocationContext:
    name: str = "Tuckaway Lane, Henrico, VA"
    latitude: float = 37.58
    longitude: float = -77.53
    ecosite: str = "Virginia Piedmont Hardwood Edge / Residential Yard"
    fallback_source: str = "Shenandoah National Park / Central VA Flora"


@dataclass(frozen=True)
class TaxonPriorSummary:
    scientific_name: str
    common_name: str | None
    family: str | None
    pile_slug: str | None
    discriminative_characters: tuple[dict[str, str], ...]
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class SemanticContextPayload:
    timestamp: str
    phenology_season: str
    location: LocationContext
    telemetry: SensorTelemetry
    candidate_taxa: tuple[TaxonPriorSummary, ...]
    knowledge_graph: AnatomicalKnowledgeGraph | None = None
    previous_classification: str | None = None
    target_genus_filter: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "phenology_season": self.phenology_season,
            "location": asdict(self.location),
            "telemetry": asdict(self.telemetry),
            "candidate_taxa": [
                {
                    "scientific_name": c.scientific_name,
                    "common_name": c.common_name,
                    "family": c.family,
                    "pile_slug": c.pile_slug,
                    "discriminative_characters": list(c.discriminative_characters),
                    "prompts": list(c.prompts),
                }
                for c in self.candidate_taxa
            ],
            "knowledge_graph": self.knowledge_graph.to_dict() if self.knowledge_graph else None,
            "previous_classification": self.previous_classification,
            "target_genus_filter": self.target_genus_filter,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_vlm_prompt(self) -> str:
        """Format the system and user instructions for the Gemma VLM agent."""
        kg_str = ""
        if self.knowledge_graph:
            kg_slots_str = "\n".join([
                f"  - Node [{s.slot_id}] (Level {s.level}): '{s.label}' | Quadrant: {s.relative_quadrant} | Prompts: {list(s.prompts)}"
                for s in self.knowledge_graph.slots
            ])
            kg_str = (
                f"Anatomical Knowledge Graph: {self.knowledge_graph.entity_name} ({self.knowledge_graph.category})\n"
                f"Slots & Containment:\n{kg_slots_str}\n\n"
            )

        candidates_str = "\n".join([
            f"- {c.scientific_name} ({c.common_name or 'N/A'}, {c.family or 'N/A'}): "
            f"Key Characters: {', '.join([f'{d['friendly_name']}={d['value_label']}' for d in c.discriminative_characters[:4]])}; "
            f"Available Prompts: {', '.join(c.prompts)}"
            for c in self.candidate_taxa
        ]) if self.candidate_taxa else "None (Using Knowledge Graph directly)"

        return (
            f"You are the Semantic Anchor Agent for real-time planar object filtering.\n"
            f"Location / Context: {self.location.name} ({self.location.ecosite})\n"
            f"Season / Phenology: {self.phenology_season} (Timestamp: {self.timestamp})\n"
            f"Camera Azimuth: {self.telemetry.azimuth_degrees}°, Zoom: {self.telemetry.user_zoom_level}x\n\n"
            f"{kg_str}"
            f"Locally Relevant Candidate Species:\n{candidates_str}\n\n"
            f"Previous Scene Classification: {self.previous_classification or 'None (initial frame)'}\n\n"
            f"Instructions:\n"
            f"1. Examine the image snapshot in the context of the knowledge graph and priors.\n"
            f"2. Confirm if the target object is present in the frame (even if held by a hand or partially occluded), identify its 3D orientation/pose, and list active visible slots.\n"
            f"3. If the object IS present in the frame: set `is_target_valid` to true, provide confidence > 0.5, and issue a single, highly specific Referring Expression Comprehension (REC) prompt (e.g. 'the black xbox controller being held in the foreground') in `focal_directives`.\n"
            f"4. If the object is DEFINITELY NOT present: set `is_target_valid` to false, set `scene_classification` to 'None', set `confidence` to 0.0, and leave `focal_directives` empty.\n"
            f"5. Output strictly valid JSON matching this schema:\n"
            f"{{\n"
            f'  "scene_classification": "<identified target or None>",\n'
            f'  "confidence": 0.95,\n'
            f'  "is_target_valid": true,\n'
            f'  "focal_directives": ["<REC phrase describing target in scene>"],\n'
            f'  "active_slot_ids": ["controller_chassis", "left_thumbstick"],\n'
            f'  "focus_bounding_box": [ymin, xmin, ymax, xmax],\n'
            f'  "rationale": "<short visual evidence summary>"\n'
            f"}}"
        )


class SemanticPayloadFactory:
    """Constructs SemanticContextPayload instances for botanical or physical object graphs."""

    def __init__(self, devo_adapter: GoBotanyDevoAdapter | None = None) -> None:
        self.devo = devo_adapter or GoBotanyDevoAdapter()

    def build_henrico_yard_payload(
        self,
        telemetry: SensorTelemetry | None = None,
        previous_classification: str | None = None,
        pile_slug: str | None = "woody-angiosperms",
        limit_candidates: int = 10,
    ) -> SemanticContextPayload:
        now = datetime.now(timezone.utc)
        month = now.month
        if month in (12, 1, 2):
            season = "winter (dormant buds, bark, silhouette)"
        elif month in (3, 4, 5):
            season = "spring (leaf emergence, early flowering)"
        elif month in (6, 7, 8):
            season = "late summer (mature vegetative foliage, developing fruit/bark)"
        else:
            season = "autumn (fall color, mature fruit, leaf drop)"

        raw_candidates = self.devo.get_henrico_va_yard_candidates()
        if not raw_candidates:
            raw_candidates = self.devo.get_candidate_taxa(pile_slug=pile_slug, limit=limit_candidates)

        summaries = tuple([
            TaxonPriorSummary(
                scientific_name=c.scientific_name,
                common_name=c.common_name,
                family=c.family,
                pile_slug=c.pile_slug,
                discriminative_characters=tuple([
                    {
                        "character_name": char.character_name,
                        "friendly_name": char.friendly_name,
                        "value_label": char.value_label,
                        "friendly_text": char.friendly_text or "",
                    }
                    for char in c.discriminative_characters
                ]),
                prompts=c.prompt_vocabulary,
            )
            for c in raw_candidates[:limit_candidates]
        ])

        # Botanical knowledge graph for default candidate
        top_name = summaries[0].scientific_name if summaries else "Quercus alba"
        top_common = summaries[0].common_name if summaries else "White Oak"
        kg = get_botanical_plant_knowledge_graph(top_name, top_common or "White Oak")

        return SemanticContextPayload(
            timestamp=now.isoformat(),
            phenology_season=season,
            location=LocationContext(
                name="Tuckaway Lane, Henrico, VA",
                latitude=37.58,
                longitude=-77.53,
                ecosite="Virginia Piedmont Mixed Hardwood / Residential Yard",
                fallback_source="Shenandoah National Park / Central VA Flora",
            ),
            telemetry=telemetry or SensorTelemetry(),
            candidate_taxa=summaries,
            knowledge_graph=kg,
            previous_classification=previous_classification,
        )

    def build_xbox_controller_payload(
        self,
        telemetry: SensorTelemetry | None = None,
        previous_classification: str | None = None,
    ) -> SemanticContextPayload:
        """Construct payload specifically for Xbox Controller desk demonstration."""
        now = datetime.now(timezone.utc)
        kg = get_xbox_controller_knowledge_graph()

        return SemanticContextPayload(
            timestamp=now.isoformat(),
            phenology_season="indoor / desktop lighting",
            location=LocationContext(
                name="Desk Environment",
                latitude=37.58,
                longitude=-77.53,
                ecosite="Indoor Desk / Handheld Controller View",
                fallback_source="Anatomical Knowledge Graph: Xbox Wireless Controller",
            ),
            telemetry=telemetry or SensorTelemetry(),
            candidate_taxa=(),
            knowledge_graph=kg,
            previous_classification=previous_classification,
        )
