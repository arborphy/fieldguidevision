"""Auditable visual tag vocabulary shared by probes, Gemma, and the report."""

from __future__ import annotations


VOCABULARY = [
    # Visible structures. These are allowed to co-occur.
    {"tag": "whole plant", "kind": "structure", "description": "Most or all of the tree, shrub, vine, or herb is visible."},
    {"tag": "trunk", "kind": "structure", "description": "A main woody trunk is visibly prominent."},
    {"tag": "bark", "kind": "structure", "description": "Bark texture or bark surface is visibly prominent."},
    {"tag": "branch", "kind": "structure", "description": "A woody branch larger than a twig is visibly prominent."},
    {"tag": "stem", "kind": "structure", "description": "A non-trunk stem is visibly prominent."},
    {"tag": "twig", "kind": "structure", "description": "A current or small woody twig is visibly prominent."},
    {"tag": "bud", "kind": "structure", "description": "One or more vegetative or reproductive buds are clearly visible."},
    {"tag": "leaf", "kind": "structure", "description": "One or more broad leaves or leaf blades are visible."},
    {"tag": "needle", "kind": "structure", "description": "Needle-like or scale-like conifer leaves are visible."},
    {"tag": "flower", "kind": "structure", "description": "One or more flowers or visibly flower-like reproductive units are present."},
    {"tag": "inflorescence", "kind": "structure", "description": "A flower cluster, catkin, or whole inflorescence is visible."},
    {"tag": "fruit", "kind": "structure", "description": "Fruit or a fruiting body is visible."},
    {"tag": "cone", "kind": "structure", "description": "A seed or pollen cone is visible."},
    {"tag": "seed", "kind": "structure", "description": "A seed is visibly isolated or is the main subject."},
    # View and condition tags retained only when visually meaningful and supported.
    {"tag": "view up trunk", "kind": "view", "description": "The camera looks upward along the trunk into the crown."},
    {"tag": "winter habit", "kind": "view", "description": "A leaf-off whole plant or crown is shown in winter condition."},
    {"tag": "large-tree bark", "kind": "view", "description": "Bark is shown on a large mature trunk."},
    {"tag": "medium-tree or large-branch bark", "kind": "view", "description": "Bark is shown on a medium trunk or large branch."},
    {"tag": "small-tree or branch bark", "kind": "view", "description": "Bark is shown on a small trunk or branch."},
    {"tag": "leaf upper surface", "kind": "view", "description": "A whole leaf is presented mainly from its upper surface."},
    {"tag": "leaf margin", "kind": "view", "description": "The leaf edge or margin is the emphasized subject."},
    {"tag": "leaf arrangement on twig", "kind": "view", "description": "Leaves are shown in their visible arrangement along a twig."},
    {"tag": "petiole arrangement", "kind": "view", "description": "Petiole attachment or orientation on a twig is emphasized."},
    {"tag": "needle attachment", "kind": "view", "description": "Needle attachment to a twig or shoot is emphasized."},
    {"tag": "winter twig", "kind": "view", "description": "A leafless twig is shown in winter or dormant condition."},
    {"tag": "leaf scar", "kind": "view", "description": "A leaf scar is clearly shown in close-up."},
    {"tag": "terminal bud", "kind": "view", "description": "A terminal bud is clearly shown in close-up."},
    {"tag": "immature fruit", "kind": "view", "description": "Fruit is visibly immature or developing."},
    {"tag": "fruit borne on plant", "kind": "view", "description": "Fruit is shown attached in its natural position on the plant."},
    {"tag": "fruit close-up", "kind": "view", "description": "Fruit is the close-up subject without a more specific condition."},
    {"tag": "opened or sectioned fruit", "kind": "view", "description": "Fruit is open, split, or sectioned so its interior is visible."},
    {"tag": "flower close-up", "kind": "view", "description": "A flower or reproductive unit is shown at close range."},
    {"tag": "flower interior", "kind": "view", "description": "The interior of a flower is the emphasized subject."},
    {"tag": "frontal flower view", "kind": "view", "description": "A flower is viewed front-on."},
    {"tag": "lateral flower view", "kind": "view", "description": "A flower is viewed from the side."},
    {"tag": "male reproductive structure", "kind": "view", "description": "A visibly male flower, inflorescence, or pollen cone is shown."},
    {"tag": "female reproductive structure", "kind": "view", "description": "A visibly female flower, inflorescence, or seed cone is shown."},
    {"tag": "closed cone", "kind": "view", "description": "A closed or immature seed cone is shown."},
    {"tag": "open cone", "kind": "view", "description": "A mature open seed cone is shown."},
    {"tag": "receptive cone", "kind": "view", "description": "A receptive female cone is shown."},
]

TAGS = [item["tag"] for item in VOCABULARY]
STRUCTURE_TAGS = [item["tag"] for item in VOCABULARY if item["kind"] == "structure"]


def canonical_tag(organ_tag: str) -> str | None:
    """Map BioImages' single coarse label to the comparable visible-structure tag."""
    tag = (organ_tag or "").strip().lower()
    return {
        "whole plant": "whole plant",
        "bark": "bark",
        "twig": "twig",
        "stem": "stem",
        "leaf": "leaf",
        "needle": "needle",
        "flower": "flower",
        "fruit": "fruit",
        "cone": "cone",
        "seed": "seed",
        "unspecified": None,
    }.get(tag)


def weak_tags(organ_tag: str, organ_category: str, subview: str) -> list[str]:
    """Translate one BioImages label into auditable multi-label weak targets.

    This mapping is for probe training and audit only. It does not claim that
    unlisted structures are absent from the image.
    """
    coarse = (organ_tag or "").strip().lower()
    category = (organ_category or "").strip().lower()
    view = (subview or "").strip().lower()
    tags: set[str] = set()

    base = canonical_tag(coarse)
    if base:
        tags.add(base)

    if category in {"whole tree", "whole tree (or vine)", "whole plant"}:
        tags.add("whole plant")
        if view == "view up trunk":
            tags.update({"trunk", "view up trunk"})
        if view == "winter":
            tags.add("winter habit")
        if view == "in flower - general view":
            tags.add("flower")

    if category == "bark":
        tags.add("bark")
        if view == "of a large tree":
            tags.update({"trunk", "large-tree bark"})
        elif view == "of a medium tree or large branch":
            tags.update({"branch", "medium-tree or large-branch bark"})
        elif view == "of a small tree or small branch":
            tags.update({"branch", "small-tree or branch bark"})

    if category == "twig":
        tags.add("twig")
        if view == "winter overall":
            tags.update({"winter twig", "bud"})
        elif view == "close-up winter leaf scar/bud":
            tags.update({"winter twig", "leaf scar", "bud"})
        elif view == "close-up winter terminal bud":
            tags.update({"winter twig", "terminal bud", "bud"})
        elif view == "orientation of petioles":
            tags.update({"leaf", "petiole arrangement"})
        elif view == "showing attachment of needles":
            tags.update({"needle", "needle attachment"})

    if category == "leaf":
        if coarse == "needle" or view == "entire needle":
            tags.update({"leaf", "needle"})
        else:
            tags.add("leaf")
        if view == "whole upper surface":
            tags.add("leaf upper surface")
        elif view == "margin of upper + lower surface":
            tags.add("leaf margin")
        elif view == "showing orientation on twig":
            tags.update({"twig", "leaf arrangement on twig"})

    if category == "inflorescence":
        tags.update({"flower", "inflorescence"})
        if view in {"close-up of flower interior", "frontal view of flower", "lateral view of flower", "ventral view of flower + perianth"}:
            tags.add("flower close-up")
        if view == "close-up of flower interior":
            tags.add("flower interior")
        elif view in {"frontal view of flower", "ventral view of flower + perianth"}:
            tags.add("frontal flower view")
        elif view == "lateral view of flower":
            tags.add("lateral flower view")
        elif view == "whole - male":
            tags.add("male reproductive structure")
        elif view == "whole - female":
            tags.add("female reproductive structure")

    if category == "fruit":
        tags.add("fruit")
        if view == "immature":
            tags.add("immature fruit")
        elif view == "as borne on the plant":
            tags.add("fruit borne on plant")
        elif view == "lateral or general close-up":
            tags.add("fruit close-up")
        elif view == "section or open":
            tags.add("opened or sectioned fruit")

    if category == "cone":
        tags.add("cone")
        if view == "male":
            tags.add("male reproductive structure")
        else:
            tags.add("female reproductive structure")
        if view == "female - closed":
            tags.add("closed cone")
        elif view == "female - mature open":
            tags.add("open cone")
        elif view == "female - receptive":
            tags.add("receptive cone")

    if category == "seed":
        tags.add("seed")
    if category == "stem":
        tags.add("stem")

    return [tag for tag in TAGS if tag in tags]
