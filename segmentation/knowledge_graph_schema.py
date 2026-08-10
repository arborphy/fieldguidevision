"""Anatomical Knowledge Graph & Hierarchical Decomposition Schema.

Defines the graphical data structure representing compositional containment,
spatial slots, topological expectations, and pose-dependent visibility
for both botanical organisms and structured interactive physical objects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class AnatomicalSlot:
    """A node in the anatomical knowledge graph representing a physical component or organ."""
    slot_id: str
    label: str
    level: int  # 1: Structural macro-boundary, 2: Micro-feature / organ detail
    parent_slot_id: str | None = None
    relative_quadrant: str | None = None  # e.g., "upper_left", "center", "lower_base", "top_margin"
    expected_shape: str | None = None  # e.g., "circular disk", "lobed polygon", "curved fissure"
    prompts: tuple[str, ...] = ()
    is_active: bool = True
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PoseCondition:
    """Orientation or view condition gating sub-component visibility."""
    pose_name: str
    description: str
    active_slots: tuple[str, ...]
    occluded_slots: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnatomicalKnowledgeGraph:
    """A compositional knowledge graph defining hierarchical object decomposition."""
    graph_id: str
    entity_name: str
    category: str  # e.g., "botanical_taxon", "handheld_input_device"
    root_slot_id: str
    slots: tuple[AnatomicalSlot, ...]
    pose_conditions: tuple[PoseCondition, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_slot(self, slot_id: str) -> AnatomicalSlot | None:
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def get_children(self, parent_slot_id: str) -> list[AnatomicalSlot]:
        return [s for s in self.slots if s.parent_slot_id == parent_slot_id]

    def get_level1_roots(self) -> list[AnatomicalSlot]:
        return [s for s in self.slots if s.level == 1]

    def get_level2_micro_slots(self, parent_slot_id: str | None = None) -> list[AnatomicalSlot]:
        if parent_slot_id:
            return [s for s in self.slots if s.level == 2 and s.parent_slot_id == parent_slot_id]
        return [s for s in self.slots if s.level == 2]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnatomicalKnowledgeGraph:
        slots = tuple([
            AnatomicalSlot(
                slot_id=s["slot_id"],
                label=s["label"],
                level=s["level"],
                parent_slot_id=s.get("parent_slot_id"),
                relative_quadrant=s.get("relative_quadrant"),
                expected_shape=s.get("expected_shape"),
                prompts=tuple(s.get("prompts", ())),
                is_active=s.get("is_active", True),
                tags=tuple(s.get("tags", ())),
            )
            for s in data["slots"]
        ])
        poses = tuple([
            PoseCondition(
                pose_name=p["pose_name"],
                description=p["description"],
                active_slots=tuple(p["active_slots"]),
                occluded_slots=tuple(p.get("occluded_slots", ())),
            )
            for p in data.get("pose_conditions", [])
        ])
        return cls(
            graph_id=data["graph_id"],
            entity_name=data["entity_name"],
            category=data["category"],
            root_slot_id=data["root_slot_id"],
            slots=slots,
            pose_conditions=poses,
            metadata=data.get("metadata", {}),
        )


# Built-in Reference Knowledge Graphs

def get_xbox_controller_knowledge_graph() -> AnatomicalKnowledgeGraph:
    """Knowledge graph for desk demonstration: Xbox Wireless Controller."""
    slots = (
        AnatomicalSlot(
            slot_id="controller_chassis",
            label="Controller Body",
            level=1,
            parent_slot_id=None,
            relative_quadrant="full_view",
            expected_shape="ergonomic dual-grip chassis",
            prompts=("video game controller body", "xbox controller chassis", "gamepad"),
            tags=("chassis", "macro_structure"),
        ),
        AnatomicalSlot(
            slot_id="left_thumbstick",
            label="Left Analog Stick",
            level=2,
            parent_slot_id="controller_chassis",
            relative_quadrant="upper_left",
            expected_shape="concave circular rubber disc",
            prompts=("left analog stick", "left thumbstick cap", "analog joystick"),
            tags=("input", "thumbstick"),
        ),
        AnatomicalSlot(
            slot_id="right_thumbstick",
            label="Right Analog Stick",
            level=2,
            parent_slot_id="controller_chassis",
            relative_quadrant="lower_center",
            expected_shape="concave circular rubber disc",
            prompts=("right analog stick", "right thumbstick cap", "analog stick"),
            tags=("input", "thumbstick"),
        ),
        AnatomicalSlot(
            slot_id="dpad",
            label="Directional Pad",
            level=2,
            parent_slot_id="controller_chassis",
            relative_quadrant="lower_left",
            expected_shape="faceted cross or circular d-pad disc",
            prompts=("directional pad", "cross d-pad", "d-pad"),
            tags=("input", "dpad"),
        ),
        AnatomicalSlot(
            slot_id="abxy_cluster",
            label="ABXY Action Buttons",
            level=2,
            parent_slot_id="controller_chassis",
            relative_quadrant="upper_right",
            expected_shape="diamond cluster of four round colored buttons",
            prompts=("A button", "B button", "X button", "Y button", "colored action buttons"),
            tags=("input", "action_buttons"),
        ),
        AnatomicalSlot(
            slot_id="bumpers_triggers",
            label="Shoulder Bumpers & Triggers",
            level=2,
            parent_slot_id="controller_chassis",
            relative_quadrant="top_shoulder",
            expected_shape="curved plastic shoulder tabs",
            prompts=("left bumper", "right bumper", "shoulder trigger", "top bumper"),
            tags=("input", "shoulder_buttons"),
        ),
    )

    poses = (
        PoseCondition(
            pose_name="faceplate_front",
            description="Front faceplate visible, grips held toward user",
            active_slots=("controller_chassis", "left_thumbstick", "right_thumbstick", "dpad", "abxy_cluster"),
            occluded_slots=("bumpers_triggers",),
        ),
        PoseCondition(
            pose_name="top_shoulder_tilted",
            description="Controller pitched forward toward camera showing top shoulder edge",
            active_slots=("controller_chassis", "bumpers_triggers", "left_thumbstick", "abxy_cluster"),
            occluded_slots=("dpad", "right_thumbstick"),
        ),
    )

    return AnatomicalKnowledgeGraph(
        graph_id="kg_xbox_controller",
        entity_name="Xbox Wireless Controller",
        category="handheld_input_device",
        root_slot_id="controller_chassis",
        slots=slots,
        pose_conditions=poses,
        metadata={"vendor": "Microsoft Xbox", "interface": "ABXY Diamond / Asymmetric Staggered Sticks"},
    )


def get_botanical_plant_knowledge_graph(
    scientific_name: str = "Quercus alba",
    common_name: str = "White Oak",
    gobotany_characters: Sequence[dict[str, str]] | None = None,
) -> AnatomicalKnowledgeGraph:
    """Generates botanical decomposition knowledge graph tied to GoBotany character vocabulary."""
    slots = [
        AnatomicalSlot(
            slot_id="plant_silhouette",
            label="Canopy / Habit Silhouette",
            level=1,
            parent_slot_id=None,
            relative_quadrant="full_frame",
            expected_shape="broad rounded canopy or structural branches",
            prompts=("whole plant", "tree canopy", "branch cluster"),
            tags=("macro_habit", "structural"),
        ),
        AnatomicalSlot(
            slot_id="trunk_bark",
            label="Trunk & Bark Surface",
            level=1,
            parent_slot_id=None,
            relative_quadrant="lower_vertical",
            expected_shape="vertical cylindrical column with fissured bark",
            prompts=("tree trunk", "bark", "furrowed bark", "scaly bark"),
            tags=("trunk", "bark"),
        ),
        AnatomicalSlot(
            slot_id="leaf_blade",
            label="Leaf Blade",
            level=2,
            parent_slot_id="plant_silhouette",
            relative_quadrant="foliage_cluster",
            expected_shape="lobed or toothed planar lamina",
            prompts=("leaf", "leaf blade", "lobed leaf blade"),
            tags=("foliage", "leaf"),
        ),
        AnatomicalSlot(
            slot_id="leaf_lobe_margin",
            label="Leaf Lobe & Margin Teeth",
            level=2,
            parent_slot_id="leaf_blade",
            relative_quadrant="leaf_perimeter",
            expected_shape="rounded sinuses or serrated teeth",
            prompts=("leaf lobe", "rounded leaf lobe", "toothed leaf margin", "leaf sinus"),
            tags=("margin", "micro_feature"),
        ),
        AnatomicalSlot(
            slot_id="leaf_petiole",
            label="Petiole Attachment",
            level=2,
            parent_slot_id="leaf_blade",
            relative_quadrant="leaf_base",
            expected_shape="slender cylindrical stalk connecting blade to twig",
            prompts=("leaf petiole", "petiole stalk"),
            tags=("petiole", "micro_feature"),
        ),
        AnatomicalSlot(
            slot_id="reproductive_structure",
            label="Fruit / Acorn / Flower",
            level=2,
            parent_slot_id="plant_silhouette",
            relative_quadrant="branch_tips",
            expected_shape="woody cupule with nut or floral cluster",
            prompts=("fruit", "acorn", "acorn cup", "flower", "nut"),
            tags=("reproductive", "micro_feature"),
        ),
    ]

    return AnatomicalKnowledgeGraph(
        graph_id=f"kg_{scientific_name.lower().replace(' ', '_')}",
        entity_name=f"{scientific_name} ({common_name})",
        category="botanical_taxon",
        root_slot_id="plant_silhouette",
        slots=tuple(slots),
        metadata={"scientific_name": scientific_name, "common_name": common_name},
    )
