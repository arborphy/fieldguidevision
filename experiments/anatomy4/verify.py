# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""Crop-level VLM verification for anatomy4 proposals.

The verifier is deliberately independent of the detector and of taxonomy:
it sees the *proposed crop pixels* (never the whole frame unless the box is
already large), the detector's proposed class, and a strict JSON schema. It
must not invent boxes and must not use taxon names as evidence.

External spend is bounded. The command records every raw response, latency,
token usage and model provenance so the audit is reproducible.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import CLASS_NAMES, clip_bbox

SCHEMA = "arq.anatomy4.verify/v1"
VERDICT_CLASSES: tuple[str, ...] = (
    "leaf",
    "needle",
    "flower",
    "inflorescence",
    "not_target",
    "ambiguous",
)
TIGHTNESS_VALUES: tuple[str, ...] = ("keep", "too_broad", "too_tight")
DEFAULT_MODEL = "openai/gpt-4o-mini"

_CLASS_SYNONYMS: dict[str, str] = {
    "leaf": "leaf",
    "leaves": "leaf",
    "leaf blade": "leaf",
    "leaf blades": "leaf",
    "foliage": "leaf",
    "needle": "needle",
    "needles": "needle",
    "conifer needle": "needle",
    "conifer needles": "needle",
    "pine needle": "needle",
    "pine needles": "needle",
    "evergreen needle": "needle",
    "evergreen needles": "needle",
    "flower": "flower",
    "flowers": "flower",
    "blossom": "flower",
    "blossoms": "flower",
    "single flower": "flower",
    "single flowers": "flower",
    "inflorescence": "inflorescence",
    "inflorescences": "inflorescence",
    "flower cluster": "inflorescence",
    "flower clusters": "inflorescence",
    "flower head": "inflorescence",
    "flower heads": "inflorescence",
    "cluster of flowers": "inflorescence",
    "clusters of flowers": "inflorescence",
    "not_target": "not_target",
    "not target": "not_target",
    "not a target": "not_target",
    "other": "not_target",
    "none": "not_target",
    "background": "not_target",
    "ambiguous": "ambiguous",
    "uncertain": "ambiguous",
    "unclear": "ambiguous",
    "unsure": "ambiguous",
}

_TIGHTNESS_SYNONYMS: dict[str, str] = {
    "keep": "keep",
    "good": "keep",
    "tight": "too_tight",
    "too tight": "too_tight",
    "too_tight": "too_tight",
    "broad": "too_broad",
    "too broad": "too_broad",
    "too_broad": "too_broad",
    "loose": "too_broad",
}

_BOX_KEYS = ("bbox", "box", "boxes", "coordinates", "polygon", "points", "x0", "y0", "x1", "y1")

_BINOMIAL_RE = re.compile(r"\b[A-Z][a-z]{2,}\s+[a-z]{2,}\b")
_FAMILY_RE = re.compile(r"\b[A-Z][a-z]{2,}aceae\b")
_ORDER_RE = re.compile(r"\b[A-Z][a-z]{2,}ales\b")

# Sentence-initial capitalised words that look like genus names to the generic
# regex but are ordinary English. Without this, "The crop shows bark" reads as
# a binomial and every such reason is wrongly invalidated.
_NON_TAXON_CAPITALS: frozenset[str] = frozenset(
    {
        "the", "this", "that", "these", "those", "there", "here", "it", "its",
        "a", "an", "no", "not", "yes", "both", "one", "two", "three", "some",
        "all", "any", "only", "most", "more", "less", "several", "single",
        "multiple", "crop", "image", "photo", "picture", "box", "boxes",
        "region", "area", "frame", "view", "subject", "main", "visible",
        "shows", "show", "showing", "appears", "looks", "seems", "contains",
        "leaf", "leaves", "needle", "needles", "flower", "flowers", "bloom",
        "blooms", "blossom", "blossoms", "inflorescence", "cluster", "clusters",
        "head", "heads", "bud", "buds", "stem", "stems", "twig", "twigs",
        "branch", "branches", "bark", "trunk", "root", "roots", "fruit",
        "fruits", "seed", "seeds", "conifer", "evergreen", "pine", "grass",
        "moss", "fern", "fern", "fungus", "mushroom", "lichen", "insect",
        "insects", "beetle", "beetles", "ant", "ants", "tree", "trees",
        "plant", "plants", "shrub", "vine", "white", "green", "yellow",
        "purple", "pink", "red", "brown", "orange", "blue", "dark", "light",
        "small", "large", "tiny", "long", "narrow", "broad", "flat", "thin",
        "not_target", "ambiguous", "background", "litter", "soil", "dirt",
        "wood", "stone", "rock", "hand", "finger", "fingers", "sky", "water",
    }
)


class VerdictParseError(ValueError):
    """The model response could not be read as a strict verdict payload."""


# ---------------------------------------------------------------------------
# Pure parsing / validation
# ---------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_json_payload(content: str) -> Any:
    """Extract the first JSON object/array from a model response.

    Tolerates code fences and leading prose (a cheap model sometimes adds a
    sentence). Raises VerdictParseError when no JSON can be decoded.
    """

    if not isinstance(content, str) or not content.strip():
        raise VerdictParseError("empty model response")
    value = strip_code_fences(content)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", value):
        try:
            payload, _ = decoder.raw_decode(value[match.start() :])
            return payload
        except json.JSONDecodeError:
            continue
    raise VerdictParseError(f"no JSON payload in response: {content[:300]!r}")


def normalize_class(value: Any) -> str:
    key = " ".join(str(value or "").lower().replace("_", " ").split())
    if key in _CLASS_SYNONYMS:
        return _CLASS_SYNONYMS[key]
    raise VerdictParseError(f"unknown verdict class: {value!r}")


def normalize_tightness(value: Any) -> str:
    key = " ".join(str(value or "").lower().replace("_", " ").split())
    if key in _TIGHTNESS_SYNONYMS:
        return _TIGHTNESS_SYNONYMS[key]
    raise VerdictParseError(f"unknown bbox_tightness: {value!r}")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise VerdictParseError(f"not a boolean: {value!r}")


def coerce_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VerdictParseError(f"not a confidence: {value!r}") from exc
    if 1.0 < number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        raise VerdictParseError(f"confidence outside [0, 1]: {value!r}")
    return round(number, 4)


def invented_box_fields(item: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    for key in _BOX_KEYS:
        if key in item and item[key] not in (None, "", [], {}):
            found.append(key)
    return sorted(found)


def detect_taxon_evidence(
    text: str | None,
    known_names: Iterable[str] = (),
) -> list[str]:
    """Find taxon names in a reason/evidence string.

    The verifier must decide from visible anatomy. Latin binomials, family
    names, order names, and any caller-supplied names (from the manifest, used
    for auditing only — never sent to the model) count as taxon evidence.
    """

    if not text:
        return []
    found: list[str] = []
    known = {name.strip().lower() for name in known_names if name and len(name.strip()) > 3}
    lowered = text.lower()
    for name in known:
        if name in lowered:
            found.append(name)
    for match in _BINOMIAL_RE.finditer(text):
        first = match.group(0).split()[0].lower()
        if first in _NON_TAXON_CAPITALS:
            continue
        found.append(match.group(0))
    for regex in (_FAMILY_RE, _ORDER_RE):
        found.extend(match.group(0) for match in regex.finditer(text))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def parse_verdict_item(
    item: Mapping[str, Any],
    *,
    proposed_class: str | None = None,
    known_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate one raw verdict object into the canonical verdict shape."""

    if not isinstance(item, Mapping):
        raise VerdictParseError("verdict item is not an object")
    verdict_class = normalize_class(item.get("class", item.get("verdict_class")))
    reason = str(item.get("reason") or "").strip()
    if not reason:
        raise VerdictParseError("missing reason")
    verdict = {
        "verdict_class": verdict_class,
        "resolved_enough": coerce_bool(item.get("resolved_enough")),
        "bbox_tightness": normalize_tightness(item.get("bbox_tightness")),
        "reason": reason[:600],
        "confidence": coerce_confidence(item.get("confidence")),
        "invented_box_fields": invented_box_fields(item),
        "taxon_evidence": detect_taxon_evidence(reason, known_names),
    }
    if proposed_class is not None:
        verdict["proposed_class"] = proposed_class
    if verdict["taxon_evidence"]:
        verdict["valid"] = False
        verdict["invalid_reason"] = "taxon_evidence"
    elif verdict["invented_box_fields"]:
        verdict["valid"] = False
        verdict["invalid_reason"] = "invented_box"
    else:
        verdict["valid"] = True
        verdict["invalid_reason"] = None
    return verdict


def parse_verdicts(
    content: str,
    *,
    expected: int,
    proposed_classes: Sequence[str | None] | None = None,
    known_names: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Parse a batch response into ``expected`` canonical verdicts."""

    payload = extract_json_payload(content)
    if isinstance(payload, Mapping) and isinstance(payload.get("verdicts"), list):
        items = payload["verdicts"]
    elif isinstance(payload, Mapping) and "class" in payload:
        items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise VerdictParseError("response has no verdicts array")

    if len(items) == 1 and expected == 1 and isinstance(items[0], Mapping) and isinstance(
        items[0].get("verdicts"), list
    ):
        items = items[0]["verdicts"]
    if len(items) != expected:
        raise VerdictParseError(f"expected {expected} verdicts, got {len(items)}")

    verdicts: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        proposed = None
        if proposed_classes is not None:
            proposed = proposed_classes[index]
        verdicts.append(parse_verdict_item(item, proposed_class=proposed, known_names=known_names))
    return verdicts


# Accept profiles. ``strict`` is the conservative default; the measured sweep
# showed the model's self-reported confidence is bimodal (mostly 0.0 or 1.0) and
# flat as a gate, so ``balanced`` ignores it and keeps tightness as a flag.
POLICY_PROFILES: dict[str, dict[str, Any]] = {
    "strict": {
        "min_confidence": 0.7,
        "accepted_tightness": ("keep",),
        "require_resolved": True,
        "description": "class match + resolved + tight box + confidence >= 0.70",
    },
    "balanced": {
        "min_confidence": 0.0,
        "accepted_tightness": ("keep", "too_tight", "too_broad"),
        "require_resolved": True,
        "description": (
            "class match + resolved; self-reported confidence and tightness are "
            "recorded but do not gate (tightness stays a review flag)"
        ),
    },
    "triage": {
        "min_confidence": 0.0,
        "accepted_tightness": ("keep", "too_tight", "too_broad"),
        "require_resolved": False,
        "description": (
            "reject only not_target/ambiguous/invalid; everything else goes to "
            "human review — never an unattended gate"
        ),
    },
}


def accept_decision(
    verdict: Mapping[str, Any],
    proposed_class: str,
    *,
    min_confidence: float = 0.7,
    accepted_tightness: Sequence[str] = ("keep",),
    require_resolved: bool = True,
    profile: str | None = None,
) -> dict[str, Any]:
    """Accept policy for turning a proposal into a training label.

    A proposal is accepted only when the verifier independently names the same
    class, considers the crop resolved, and (per profile) the box is tight and
    the self-reported confidence clears a floor.
    """

    if profile is not None:
        if profile not in POLICY_PROFILES:
            raise ValueError(f"unknown policy profile: {profile}")
        spec = POLICY_PROFILES[profile]
        min_confidence = float(spec["min_confidence"])
        accepted_tightness = tuple(spec["accepted_tightness"])
        require_resolved = bool(spec["require_resolved"])

    reasons: list[str] = []
    flags: list[str] = []
    if not verdict.get("valid", False):
        reasons.append(str(verdict.get("invalid_reason") or "invalid"))
    if verdict.get("verdict_class") != proposed_class:
        reasons.append(f"class_mismatch:{verdict.get('verdict_class')}")
    if verdict.get("verdict_class") in {"not_target", "ambiguous"}:
        reasons.append(f"verdict_{verdict.get('verdict_class')}")
    if require_resolved and verdict.get("resolved_enough") is not True:
        reasons.append("not_resolved")
    tightness = verdict.get("bbox_tightness")
    if tightness not in accepted_tightness:
        reasons.append(f"bbox_{tightness}")
    elif tightness != "keep":
        flags.append(f"bbox_{tightness}")
    confidence = float(verdict.get("confidence") or 0.0)
    if confidence < float(min_confidence):
        reasons.append(f"low_confidence:{confidence:.2f}<{min_confidence:.2f}")
    return {
        "accept": not reasons,
        "reasons": reasons,
        "flags": flags,
        "min_confidence": float(min_confidence),
        "accepted_tightness": list(accepted_tightness),
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# Crop preparation
# ---------------------------------------------------------------------------


def crop_pixels(
    bbox_norm: Sequence[float],
    width: int,
    height: int,
    *,
    margin: float = 0.18,
) -> tuple[int, int, int, int]:
    """Pixel crop box (left, top, right, bottom) around a normalized bbox."""

    x0, y0, x1, y1 = clip_bbox(bbox_norm)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    box_w = (x1 - x0) * width
    box_h = (y1 - y0) * height
    pad = margin * max(box_w, box_h)
    if pad > 0:
        pad = max(4.0, pad)
    left = max(0, int(round(x0 * width - pad)))
    top = max(0, int(round(y0 * height - pad)))
    right = min(width, int(round(x1 * width + pad)))
    bottom = min(height, int(round(y1 * height + pad)))
    if right - left < 2 or bottom - top < 2:
        raise ValueError("degenerate crop box")
    return left, top, right, bottom


def make_crop(
    image_path: str | Path,
    bbox_norm: Sequence[float],
    out_path: str | Path,
    *,
    margin: float = 0.18,
    max_side: int = 900,
    min_side: int = 224,
    quality: int = 88,
) -> dict[str, Any]:
    """Write a crop of the real pixels and return its metadata."""

    from PIL import Image, ImageOps

    source = Path(image_path)
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        left, top, right, bottom = crop_pixels(bbox_norm, image.width, image.height, margin=margin)
        crop = image.crop((left, top, right, bottom))
        scale = 1.0
        longest = max(crop.width, crop.height)
        if longest > max_side:
            scale = max_side / longest
        elif longest < min_side:
            scale = min_side / longest
        if scale != 1.0:
            crop = crop.resize(
                (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
                Image.LANCZOS,
            )
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        crop.save(target, format="JPEG", quality=quality, optimize=True)
        return {
            "source_path": str(source),
            "crop_path": str(target),
            "source_size": [image.width, image.height],
            "crop_pixels": [left, top, right, bottom],
            "crop_size": [crop.width, crop.height],
            "margin": margin,
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }


def image_data_url(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

PROMPT_VERSION = "anatomy4-verify-crop-v1"

_SYSTEM_PROMPT = (
    "You verify plant anatomy crops for a vision dataset. Judge only what is "
    "visible in the pixels. Do not identify or name species, genera, families, "
    "or any taxon, and never use a taxon name as evidence. Do not guess or "
    "output bounding boxes, coordinates, or polygons. Answer with strict JSON "
    "only, no prose."
)

_CLASS_DEFINITIONS = (
    "Class definitions (visual only):\n"
    "- leaf: a broad, flat leaf blade (including a leaflet of a compound leaf). "
    "Grass blades, ferns, moss, and leaf litter are not 'leaf'.\n"
    "- needle: a long, narrow, evergreen conifer needle or needle fascicle. "
    "Pine candles are not needles.\n"
    "- flower: a single flower (one bloom, one head). A cluster of several "
    "blooms is inflorescence, not flower.\n"
    "- inflorescence: a cluster, spike, umbel, raceme, or head of multiple "
    "flowers.\n"
    "- not_target: the crop does not show the proposed class as its main "
    "subject (background, bark, trunk, moss, fungus, grass, litter, insect, "
    "or a different organ).\n"
    "- ambiguous: the crop cannot be decided confidently from these pixels."
)

_OUTPUT_SPEC = (
    "Return JSON exactly in this shape:\n"
    '{"verdicts":[{"class":"leaf|needle|flower|inflorescence|not_target|ambiguous",'
    '"resolved_enough":true,"bbox_tightness":"keep|too_broad|too_tight",'
    '"reason":"short visual reason, no taxon names","confidence":0.0}]}\n'
    "Rules: one object per crop, in crop order; resolved_enough is true only "
    "when the proposed class is clearly visible; bbox_tightness describes the "
    "crop as given (keep, too_broad, or too_tight); confidence is 0.0-1.0. "
    "No extra keys."
)

_MAIN_SUBJECT_RULE = (
    "Main-subject rule: accept the proposed class only when it is the MAIN "
    "SUBJECT of the crop and occupies a substantial visible area. If it is "
    "background, a few distant branches at a frame edge, or incidental to a "
    "wider scene (forest floor, trunk, ground, hand, sky), answer not_target "
    "even when some of the organ is technically visible. Set resolved_enough "
    "false in that case."
)

_FOCUSED_NEEDLE_PROMPT = (
    "Second pass for a low-confidence needle proposal. Use the wider context. "
    "Ask only whether long, narrow evergreen conifer needles (not grass, fern, "
    "moss, daylily leaves, pine cones, or bare twigs) are visibly the main "
    "subject of the proposed region. Then give the same strict JSON verdict."
)


PROMPT_VARIANTS: tuple[str, ...] = ("v1", "main_subject")


def build_batch_messages(
    crops: Sequence[Mapping[str, Any]],
    *,
    focused: str | None = None,
    variant: str = "v1",
) -> list[dict[str, Any]]:
    """Build OpenRouter chat messages for one batch of crops."""

    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant: {variant}")
    content: list[dict[str, Any]] = [{"type": "text", "text": _SYSTEM_PROMPT}]
    content.append({"type": "text", "text": _CLASS_DEFINITIONS})
    if variant == "main_subject":
        content.append({"type": "text", "text": _MAIN_SUBJECT_RULE})
    if focused:
        content.append({"type": "text", "text": focused})
    for index, crop in enumerate(crops, start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"Crop {index}: detector proposed class = "
                    f"{crop['proposed_class']}."
                ),
            }
        )
        content.append({"type": "image_url", "image_url": {"url": image_data_url(crop["crop_path"])}})
    content.append({"type": "text", "text": _OUTPUT_SPEC})
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def prompt_hash(variant: str = "v1") -> str:
    payload = {
        "version": PROMPT_VERSION,
        "system": _SYSTEM_PROMPT,
        "classes": _CLASS_DEFINITIONS,
        "output": _OUTPUT_SPEC,
        "focused_needle": _FOCUSED_NEEDLE_PROMPT,
        "main_subject": _MAIN_SUBJECT_RULE if variant == "main_subject" else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Credentials / OpenRouter client
# ---------------------------------------------------------------------------


def read_env_file(path: str | Path) -> dict[str, str]:
    """Read KEY=VALUE pairs without exporting or logging values."""

    values: dict[str, str] = {}
    file_path = Path(path)
    if not file_path.exists():
        return values
    for raw_line in file_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def discover_env_file(
    explicit: str | Path | None = None,
    *,
    extra_roots: Sequence[str | Path] = (),
) -> Path | None:
    """Find a .env containing OPENROUTER_API_KEY without printing it."""

    if explicit:
        return Path(explicit).expanduser()
    candidates: list[Path] = []
    for root in (Path.cwd(), Path(__file__).resolve().parents[2], *map(Path, extra_roots)):
        current = root.resolve()
        for _ in range(5):
            candidates.append(current / ".env")
            if current.parent == current:
                break
            current = current.parent
    for candidate in candidates:
        if not candidate.exists():
            continue
        values = read_env_file(candidate)
        if values.get("OPENROUTER_API_KEY", "").strip():
            return candidate
    return None


@dataclass
class OpenRouterSettings:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 1200
    referer: str = "https://local.codex"
    title: str = "Anatomy4 crop verification"
    env_file: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "referer": self.referer,
            "title": self.title,
            "env_file": self.env_file,
        }


def resolve_model(model: str) -> str:
    try:
        from scripts.adapters.openrouter_models import resolve_model_alias
    except ImportError:  # pragma: no cover - only when run outside the repo
        return model
    return resolve_model_alias(model, require_image=True)


def load_openrouter_settings(
    *,
    env_file: str | Path | None = None,
    model: str | None = None,
) -> OpenRouterSettings:
    env = dict(os.environ)
    chosen = discover_env_file(env_file)
    if chosen is not None:
        file_values = read_env_file(chosen)
        if env_file is not None:
            # An explicit file is authoritative: do not let a stale shell key win.
            env.update(file_values)
        else:
            for key, value in file_values.items():
                env.setdefault(key, value)
    api_key = env.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required (root .env or environment)")
    model_id = (model or env.get("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    model_id = resolve_model(model_id)
    return OpenRouterSettings(
        api_key=api_key,
        model=model_id,
        base_url=env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        timeout=float(env.get("OPENROUTER_TIMEOUT", "120")),
        temperature=float(env.get("OPENROUTER_TEMPERATURE", "0")),
        max_tokens=int(env.get("OPENROUTER_MAX_TOKENS", "1200")),
        referer=env.get("OPENROUTER_REFERER", "https://local.codex"),
        title=env.get("OPENROUTER_TITLE", "Anatomy4 crop verification"),
        env_file=str(chosen) if chosen is not None else None,
    )


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def post_chat(
    settings: OpenRouterSettings,
    messages: Sequence[Mapping[str, Any]],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """One HTTP call. Returns raw response metadata; never logs the key."""

    payload = {
        "model": settings.model,
        "messages": list(messages),
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{settings.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.referer,
            "X-Title": settings.title,
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=float(timeout or settings.timeout)) as response:
            raw = json.loads(response.read().decode("utf-8"))
            status = response.status
            headers = dict(response.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        latency_ms = round((time.monotonic() - started) * 1000.0, 1)
        return {
            "ok": False,
            "status": exc.code,
            "latency_ms": latency_ms,
            "error": body[:2000],
            "raw": None,
            "headers": dict(exc.headers) if exc.headers else {},
        }
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        ssl.SSLError,
        ConnectionError,
        OSError,
    ) as exc:
        # TLS resets (bad record mac), connection drops and timeouts are all
        # retryable; a long batch run must not die on one of them.
        latency_ms = round((time.monotonic() - started) * 1000.0, 1)
        return {
            "ok": False,
            "status": None,
            "latency_ms": latency_ms,
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "raw": None,
            "headers": {},
        }

    latency_ms = round((time.monotonic() - started) * 1000.0, 1)
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = None
    return {
        "ok": content is not None,
        "status": status,
        "latency_ms": latency_ms,
        "raw": raw,
        "content": content,
        "usage": raw.get("usage") or {},
        "request_id": raw.get("id"),
        "headers": {k: v for k, v in headers.items() if k.lower() in {"x-request-id", "cf-ray"}},
        "error": None if content is not None else json.dumps(raw)[:2000],
    }


def call_with_retries(
    settings: OpenRouterSettings,
    messages: Sequence[Mapping[str, Any]],
    *,
    attempts: int = 3,
    backoff: float = 1.0,
    sleep=time.sleep,
) -> dict[str, Any]:
    """HTTP call with bounded exponential backoff on retryable failures."""

    history: list[dict[str, Any]] = []
    for attempt in range(1, max(1, attempts) + 1):
        result = post_chat(settings, messages)
        history.append(
            {
                "attempt": attempt,
                "ok": result["ok"],
                "status": result["status"],
                "latency_ms": result["latency_ms"],
                "error": result["error"],
            }
        )
        retryable = result["status"] in _RETRYABLE_STATUS or result["status"] is None
        if result["ok"] or not retryable or attempt >= attempts:
            result = dict(result)
            result["attempts"] = history
            return result
        sleep(backoff * (2 ** (attempt - 1)))
    return result


# ---------------------------------------------------------------------------
# Verification run
# ---------------------------------------------------------------------------


def load_review_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("proposal_id") or "").strip():
                rows.append({key: (value or "").strip() for key, value in row.items()})
    return rows


def load_known_names(manifest_path: str | Path) -> dict[str, list[str]]:
    """Map photo_id -> taxon names (audit guard only; never sent to the model)."""

    data = json.loads(Path(manifest_path).read_text())
    names: dict[str, list[str]] = {}
    observations = data.get("observations") or []
    for observation in observations:
        taxon = str(observation.get("taxon_name") or "").strip()
        common = str(observation.get("common_name") or "").strip()
        values = [value for value in (taxon, common) if value]
        for image in observation.get("images") or []:
            photo_id = str(image.get("photo_id") or "")
            if photo_id:
                names[photo_id] = values
    return names


def sample_evenly_per_class(
    proposals: Sequence[Mapping[str, Any]],
    per_class: int,
) -> list[dict[str, Any]]:
    """Deterministically spread ``per_class`` proposals across each class.

    Even spacing (not the first N) keeps the detector's confidence range in the
    sample instead of only its highest-scoring proposals.
    """

    by_class: dict[str, list[Mapping[str, Any]]] = {}
    for proposal in proposals:
        by_class.setdefault(str(proposal.get("class_name")), []).append(proposal)
    picked: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        items = by_class.get(class_name, [])
        if not items or per_class <= 0:
            continue
        if len(items) <= per_class:
            picked.extend(dict(item) for item in items)
            continue
        if per_class == 1:
            picked.append(dict(items[len(items) // 2]))
            continue
        # Spread includes both endpoints so the lowest-scoring proposals (the
        # ones an accept gate most needs to see) are never skipped.
        for index in range(per_class):
            position = round(index * (len(items) - 1) / float(per_class - 1))
            picked.append(dict(items[min(len(items) - 1, position)]))
    return picked


def find_image(images_dir: str | Path, photo_id: str) -> Path | None:
    root = Path(images_dir)
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = root / f"{photo_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def select_proposals(
    proposals: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    reviewed = {row["proposal_id"]: row for row in review_rows}
    selected: list[dict[str, Any]] = []
    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id") or "")
        if proposal_id not in reviewed:
            continue
        selected.append({**dict(proposal), "review": reviewed[proposal_id]})
    return selected


def proposal_bbox(proposal: Mapping[str, Any]) -> tuple[float, float, float, float]:
    bbox = proposal.get("detector_bbox")
    if not bbox:
        raise ValueError(f"{proposal.get('proposal_id')} has no detector_bbox")
    return clip_bbox(bbox)


def render_disagreement_cards(
    *,
    verdict_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, str]],
    images_dir: str | Path,
    out_dir: str | Path,
    min_confidence: float = 0.7,
    max_cards: int = 40,
) -> dict[str, Any]:
    """Render one evidence card per disagreement (full frame with the box + crop)."""

    from PIL import Image, ImageDraw, ImageFont

    review_by_id = {row["proposal_id"]: row for row in review_rows}
    effective: dict[str, Mapping[str, Any]] = {}
    for row in verdict_rows:
        effective[row["proposal_id"]] = row
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    cards: list[dict[str, Any]] = []
    for proposal_id, row in sorted(effective.items()):
        review = review_by_id.get(proposal_id) or {}
        human_accept = (review.get("verdict") or "").lower() == "tp"
        decision = accept_decision(
            row["verdict"], row["proposed_class"], min_confidence=min_confidence
        )
        if decision["accept"] == human_accept:
            continue
        photo_id = str(row.get("photo_id") or "")
        image_path = find_image(images_dir, photo_id)
        if image_path is None:
            continue
        box = row["crop"]["crop_pixels"]
        with Image.open(image_path) as raw:
            full = raw.convert("RGB").copy()
        scale = min(1.0, 420.0 / max(full.width, full.height))
        full = full.resize((max(1, int(full.width * scale)), max(1, int(full.height * scale))), Image.LANCZOS)
        draw = ImageDraw.Draw(full)
        draw.rectangle(
            [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale],
            outline=(255, 40, 40),
            width=3,
        )
        crop = Image.open(row["crop"]["crop_path"]).convert("RGB")
        crop.thumbnail((260, 260), Image.LANCZOS)

        width = 720
        header = 116
        card = Image.new("RGB", (width, max(full.height, crop.height) + header), (255, 255, 255))
        cd = ImageDraw.Draw(card)
        cd.text(
            (10, 8),
            f"{proposal_id}  proposed={row['proposed_class']}  human={'accept' if human_accept else 'reject'}"
            f"  verifier={'accept' if decision['accept'] else 'reject'} ({row['verdict']['verdict_class']},"
            f" conf {row['verdict']['confidence']:.2f}, {row['verdict']['bbox_tightness']})",
            fill=(0, 0, 0),
            font=font,
        )
        cd.text((10, 32), f"verifier: {row['verdict']['reason']}"[:110], fill=(20, 60, 20), font=small)
        cd.text(
            (10, 50),
            f"human note: {(review.get('notes') or '(none)')[:110]}",
            fill=(90, 20, 20),
            font=small,
        )
        if decision["reasons"]:
            cd.text((10, 68), f"policy: {', '.join(decision['reasons'])[:110]}", fill=(70, 70, 70), font=small)
        cd.text((10, 88), "red = proposed box on the source frame (crop margin included)", fill=(90, 90, 90), font=small)
        card.paste(full, (10, header))
        card.paste(crop, (450, header))
        name = f"{proposal_id}.jpg"
        card.save(output / name, quality=88)
        cards.append(
            {
                "proposal_id": proposal_id,
                "proposed_class": row["proposed_class"],
                "human_accept": human_accept,
                "verifier_accept": decision["accept"],
                "verifier_class": row["verdict"]["verdict_class"],
                "verifier_confidence": row["verdict"]["confidence"],
                "verifier_reason": row["verdict"]["reason"],
                "human_notes": review.get("notes") or "",
                "card": str(output / name),
            }
        )
        if len(cards) >= max_cards:
            break

    if cards:
        thumbs = [Image.open(card["card"]).convert("RGB") for card in cards]
        thumb_w = 360
        scaled = []
        for thumb in thumbs:
            factor = thumb_w / thumb.width
            scaled.append(thumb.resize((thumb_w, max(1, int(thumb.height * factor))), Image.LANCZOS))
        cols = 2
        rows_count = (len(scaled) + cols - 1) // cols
        row_heights = [
            max(scaled[index].height for index in range(row * cols, min(len(scaled), (row + 1) * cols)))
            for row in range(rows_count)
        ]
        sheet = Image.new("RGB", (thumb_w * cols, sum(row_heights)), (255, 255, 255))
        y = 0
        for row_index, height in enumerate(row_heights):
            for col in range(cols):
                index = row_index * cols + col
                if index >= len(scaled):
                    continue
                sheet.paste(scaled[index], (col * thumb_w, y))
            y += height
        sheet.save(output / "disagreements-sheet.jpg", quality=86)

    (output / "disagreements.json").write_text(json.dumps(cards, indent=2))
    return {"cards": len(cards), "dir": str(output)}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_verification(
    *,
    proposals_path: str | Path,
    images_dir: str | Path,
    review_path: str | Path,
    manifest_path: str | Path | None,
    out_dir: str | Path,
    settings: OpenRouterSettings,
    batch_size: int = 4,
    margin: float = 0.18,
    max_calls: int = 200,
    min_confidence: float = 0.7,
    attempts: int = 3,
    second_pass: bool = True,
    second_pass_margin: float = 0.55,
    second_pass_confidence: float = 0.7,
    limit: int | None = None,
    sample_per_class: int | None = None,
    proposal_ids: Sequence[str] | None = None,
    prompt_variant: str = "v1",
    crop_max_side: int = 900,
    accepted_tightness: Sequence[str] = ("keep",),
    sleep=time.sleep,
) -> dict[str, Any]:
    """Crop, verify, and write verdicts + report artifacts."""

    output = Path(out_dir)
    crops_dir = output / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    proposals_doc = json.loads(Path(proposals_path).read_text())
    review_rows = load_review_rows(review_path)
    selected = select_proposals(proposals_doc.get("proposals") or [], review_rows)
    if proposal_ids is not None:
        wanted = {str(item) for item in proposal_ids}
        selected = [item for item in selected if str(item.get("proposal_id")) in wanted]
    if sample_per_class is not None:
        selected = sample_evenly_per_class(selected, sample_per_class)
    if limit is not None:
        selected = selected[:limit]
    known_names = load_known_names(manifest_path) if manifest_path else {}

    call_budget = int(max_calls)
    raw_calls: list[dict[str, Any]] = []
    raw_calls_path = output / "raw_calls.jsonl"
    if raw_calls_path.exists():
        raw_calls = [
            json.loads(line)
            for line in raw_calls_path.read_text().splitlines()
            if line.strip()
        ]
    covered: dict[int, set[str]] = {1: set(), 2: set()}
    for call in raw_calls:
        if call.get("ok"):
            covered.setdefault(int(call.get("pass") or 1), set()).update(
                str(pid) for pid in call.get("proposal_ids") or []
            )
    calls_used = len(raw_calls)
    verdict_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    primary = [
        item
        for item in selected
        if item.get("class_name") in CLASS_NAMES
        and str(item.get("proposal_id")) not in covered[1]
    ]
    for proposal_id in sorted(covered[1] & {str(item.get("proposal_id")) for item in selected}):
        skipped.append({"proposal_id": proposal_id, "reason": "already_verified"})

    def verify_batch(
        batch: Sequence[Mapping[str, Any]],
        *,
        pass_number: int,
        margin_value: float,
        focused: str | None,
    ) -> None:
        nonlocal calls_used
        crops: list[dict[str, Any]] = []
        for item in batch:
            proposal_id = str(item["proposal_id"])
            image_path = find_image(images_dir, str(item.get("photo_id") or item.get("image_id")))
            if image_path is None:
                skipped.append({"proposal_id": proposal_id, "reason": "image_missing"})
                continue
            crop_path = crops_dir / f"{proposal_id}-p{pass_number}.jpg"
            try:
                crop_meta = make_crop(
                    image_path,
                    proposal_bbox(item),
                    crop_path,
                    margin=margin_value,
                    max_side=crop_max_side,
                )
            except Exception as exc:  # pragma: no cover - corrupted image path
                skipped.append({"proposal_id": proposal_id, "reason": f"crop_failed:{exc}"})
                continue
            crops.append(
                {
                    "proposal": item,
                    "proposal_id": proposal_id,
                    "proposed_class": str(item["class_name"]),
                    "crop_path": crop_meta["crop_path"],
                    "crop_meta": crop_meta,
                    "pass": pass_number,
                }
            )
        if not crops:
            return
        if calls_used + 1 > call_budget:
            for crop in crops:
                skipped.append({"proposal_id": crop["proposal_id"], "reason": "call_budget"})
            return

        messages = build_batch_messages(crops, focused=focused, variant=prompt_variant)
        result = call_with_retries(settings, messages, attempts=attempts, sleep=sleep)
        calls_used += 1
        record = (
            {
                "schema": SCHEMA,
                "pass": pass_number,
                "crop_count": len(crops),
                "proposal_ids": [crop["proposal_id"] for crop in crops],
                "items": [
                    {
                        "proposal_id": crop["proposal_id"],
                        "photo_id": str(
                            crop["proposal"].get("photo_id")
                            or crop["proposal"].get("image_id")
                        ),
                        "proposed_class": crop["proposed_class"],
                        "proposal_box_score": crop["proposal"].get("box_score"),
                        "review_verdict": (
                            (crop["proposal"].get("review") or {}).get("verdict") or ""
                        ).lower(),
                        "review_notes": (crop["proposal"].get("review") or {}).get("notes") or "",
                        "crop": crop["crop_meta"],
                    }
                    for crop in crops
                ],
                "model": settings.model,
                "prompt_hash": prompt_hash(prompt_variant),
                "ok": result["ok"],
                "status": result["status"],
                "latency_ms": result["latency_ms"],
                "attempts": result["attempts"],
                "usage": result.get("usage") or {},
                "request_id": result.get("request_id"),
                "error": result.get("error"),
                "raw": result.get("raw"),
                "content": result.get("content"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        raw_calls.append(record)
        # Persist each call immediately: a dropped connection or a killed run
        # must not lose already-paid responses.
        with raw_calls_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        if not result["ok"] or not result.get("content"):
            for crop in crops:
                skipped.append({"proposal_id": crop["proposal_id"], "reason": "request_failed"})
            return
        try:
            parsed = parse_verdicts(
                str(result["content"]),
                expected=len(crops),
                proposed_classes=[crop["proposed_class"] for crop in crops],
                known_names=[
                    name
                    for crop in crops
                    for name in known_names.get(str(crop["proposal"].get("photo_id")), [])
                ],
            )
        except VerdictParseError as exc:
            for crop in crops:
                skipped.append(
                    {"proposal_id": crop["proposal_id"], "reason": f"parse_failed:{exc}"}
                )
            return

        payload = extract_json_payload(str(result["content"]))
        raw_items = payload.get("verdicts") if isinstance(payload, Mapping) else payload
        for index, (crop, verdict) in enumerate(zip(crops, parsed)):
            proposal = crop["proposal"]
            decision = accept_decision(
                verdict,
                crop["proposed_class"],
                min_confidence=min_confidence,
            )
            review = proposal.get("review") or {}
            verdict_rows.append(
                {
                    "schema": SCHEMA,
                    "proposal_id": crop["proposal_id"],
                    "photo_id": str(proposal.get("photo_id") or proposal.get("image_id")),
                    "pass": pass_number,
                    "proposed_class": crop["proposed_class"],
                    "proposal_box_score": proposal.get("box_score"),
                    "review_verdict": (review.get("verdict") or "").lower(),
                    "review_notes": review.get("notes") or "",
                    "model": settings.model,
                    "prompt_hash": prompt_hash(prompt_variant),
                    "verdict": verdict,
                    "raw_verdict": raw_items[index],
                    "decision": decision,
                    "crop": crop["crop_meta"],
                    "latency_ms": result["latency_ms"],
                    "usage": result.get("usage") or {},
                }
            )

    try:
        for start in range(0, len(primary), max(1, batch_size)):
            verify_batch(primary[start : start + max(1, batch_size)], pass_number=1, margin_value=margin, focused=None)
    finally:
        pass

    # Second pass: low-confidence / not-accepted needle proposals with wider context.
    if second_pass and calls_used < call_budget:
        primary_by_id = {row["proposal_id"]: row for row in verdict_rows if row["pass"] == 1}
        needle_candidates = [
            item
            for item in primary
            if str(item.get("class_name")) == "needle"
            and (
                verdict_rows
                and (
                    primary_by_id.get(str(item["proposal_id"]), {}).get("decision", {}).get("accept") is False
                    or float(
                        primary_by_id.get(str(item["proposal_id"]), {})
                        .get("verdict", {})
                        .get("confidence", 0.0)
                    )
                    < second_pass_confidence
                )
            )
        ]
        # Only rerun proposals that actually received a first-pass verdict.
        needle_candidates = [
            item
            for item in needle_candidates
            if str(item["proposal_id"]) in primary_by_id
            and str(item["proposal_id"]) not in covered[2]
        ]
        for start in range(0, len(needle_candidates), max(1, batch_size)):
            verify_batch(
                needle_candidates[start : start + max(1, batch_size)],
                pass_number=2,
                margin_value=second_pass_margin,
                focused=_FOCUSED_NEEDLE_PROMPT,
            )

    complete_verdicts = reparse_raw_calls(
        raw_calls_path=raw_calls_path,
        verdicts_path=output / "verdicts.jsonl" if (output / "verdicts.jsonl").exists() else None,
        known_names=known_names,
    )
    if complete_verdicts:
        verdict_rows = complete_verdicts
    _write_jsonl(output / "verdicts.jsonl", verdict_rows)
    report = build_report(
        review_rows=review_rows,
        verdict_rows=verdict_rows,
        settings=settings,
        review_path=str(review_path),
        proposals_path=str(proposals_path),
        crops_dir=str(crops_dir),
        calls_used=calls_used,
        skipped=skipped,
        min_confidence=min_confidence,
        batch_size=batch_size,
        accepted_tightness=accepted_tightness,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (output / "report.md").write_text(render_report_markdown(report))
    return report


# ---------------------------------------------------------------------------
# Evaluation against the human review
# ---------------------------------------------------------------------------


def infer_human_class(
    proposed_class: str,
    review_verdict: str,
    notes: str,
) -> str:
    """Best-effort human label for the confusion matrix.

    ``tp`` means the proposal class was correct. For ``fp`` the note often
    names the real organ or a non-organ subject; fall back to ``not_target``.
    """

    if (review_verdict or "").lower() == "tp":
        return proposed_class
    text = (notes or "").lower()
    if "cluster" in text or "inflorescence" in text:
        return "inflorescence"
    if "not needles" in text or "not a needle" in text:
        return "leaf"
    if "flower" in text and "flower not leaf" in text:
        return "flower"
    if "needle" in text and "no needle" not in text and "not needle" not in text:
        return "needle"
    if "flower" in text:
        return "flower"
    if "leaf" in text and "litter" not in text:
        return "leaf"
    return "not_target"


def _safe_div(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson score interval — the honest error bar for these small classes."""

    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)


def _binary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row["accept"] and row["human_accept"])
    fp = sum(1 for row in rows if row["accept"] and not row["human_accept"])
    tn = sum(1 for row in rows if not row["accept"] and not row["human_accept"])
    fn = sum(1 for row in rows if not row["accept"] and row["human_accept"])
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    precision_low, precision_high = wilson_interval(tp, tp + fp)
    recall_low, recall_high = wilson_interval(tp, tp + fn)
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "precision_ci95": [precision_low, precision_high],
        "recall_ci95": [recall_low, recall_high],
        "f1": _safe_div(2 * precision * recall, (precision or 0) + (recall or 0))
        if precision is not None and recall is not None
        else None,
        "accuracy": _safe_div(tp + tn, len(rows)),
        "human_accept_rate": _safe_div(tp + fn, len(rows)),
        "verifier_accept_rate": _safe_div(tp + fp, len(rows)),
    }


MIN_POLICY_SAMPLES = 20


def _trust_level(metrics: Mapping[str, Any]) -> str:
    """Per-class recommendation, gated on sample size and the precision error bar.

    A class needs >= 20 verified proposals before any policy call, and
    auto-accept additionally needs the *lower* bound of the 95% precision
    interval >= 0.90 — a single lucky false accept in 20 must not buy a class
    an unattended gate.
    """

    n = int(metrics.get("n") or 0)
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    precision_low = (metrics.get("precision_ci95") or [None, None])[0]
    if n < MIN_POLICY_SAMPLES or precision is None or recall is None:
        return "insufficient_evidence"
    if precision >= 0.95 and recall >= 0.8 and (precision_low or 0.0) >= 0.90:
        return "auto_accept"
    if precision >= 0.85 and recall >= 0.6:
        return "review_only"
    if precision >= 0.7:
        return "review_only_low_volume"
    return "unusable"


def build_report(
    *,
    review_rows: Sequence[Mapping[str, str]],
    verdict_rows: Sequence[Mapping[str, Any]],
    settings: OpenRouterSettings,
    review_path: str,
    proposals_path: str,
    crops_dir: str,
    calls_used: int,
    skipped: Sequence[Mapping[str, Any]],
    min_confidence: float,
    batch_size: int,
    accepted_tightness: Sequence[str] = ("keep",),
) -> dict[str, Any]:
    review_by_id = {row["proposal_id"]: row for row in review_rows}

    # Choose the effective verdict per proposal: pass 2 wins for needle when present.
    primary: dict[str, dict[str, Any]] = {}
    secondary: dict[str, dict[str, Any]] = {}
    for row in verdict_rows:
        if row["pass"] == 1:
            primary[row["proposal_id"]] = row
        else:
            secondary[row["proposal_id"]] = row
    effective: dict[str, dict[str, Any]] = {}
    for proposal_id, row in primary.items():
        effective[proposal_id] = row
    for proposal_id, row in secondary.items():
        # A second pass is a re-check: it replaces pass 1 for that proposal.
        effective[proposal_id] = row

    rows: list[dict[str, Any]] = []
    for proposal_id, row in effective.items():
        review = review_by_id.get(proposal_id, {})
        human_accept = (review.get("verdict") or "").lower() == "tp"
        proposed = row["proposed_class"]
        # Recompute the decision so a report can sweep policies without re-spend.
        decision = accept_decision(
            row["verdict"],
            proposed,
            min_confidence=min_confidence,
            accepted_tightness=accepted_tightness,
        )
        rows.append(
            {
                "proposal_id": proposal_id,
                "class": proposed,
                "human_accept": human_accept,
                "human_verdict": (review.get("verdict") or "").lower(),
                "human_class": infer_human_class(
                    proposed, review.get("verdict") or "", review.get("notes") or ""
                ),
                "human_notes": review.get("notes") or "",
                "accept": bool(decision["accept"]),
                "reasons": decision["reasons"],
                "verdict_class": row["verdict"]["verdict_class"],
                "confidence": row["verdict"]["confidence"],
                "resolved_enough": row["verdict"]["resolved_enough"],
                "bbox_tightness": row["verdict"]["bbox_tightness"],
                "reason": row["verdict"]["reason"],
                "pass": row["pass"],
                "proposal_box_score": row["proposal_box_score"],
            }
        )

    per_class: dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        subset = [row for row in rows if row["class"] == class_name]
        metrics = _binary_metrics(subset)
        metrics["trust"] = _trust_level(metrics)
        metrics["confusion"] = _confusion(subset, review_by_id)
        per_class[class_name] = metrics
    overall = _binary_metrics(rows)
    overall["trust"] = _trust_level(overall)

    # Confusion over the whole reviewed set (verifier verdict class x human class).
    confusion = _confusion(rows, review_by_id)

    disagreements = [
        {
            "proposal_id": row["proposal_id"],
            "class": row["class"],
            "human_accept": row["human_accept"],
            "human_verdict": row["human_verdict"],
            "human_class": row["human_class"],
            "human_notes": row["human_notes"],
            "verifier_accept": row["accept"],
            "verifier_class": row["verdict_class"],
            "verifier_reason": row["reason"],
            "verifier_confidence": row["confidence"],
            "verifier_tightness": row["bbox_tightness"],
            "reasons": row["reasons"],
            "pass": row["pass"],
        }
        for row in rows
        if row["accept"] != row["human_accept"]
        or (row["accept"] and row["human_accept"] and row["verdict_class"] != row["class"])
    ]

    confidence_buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = f"{min(0.9, int(row['confidence'] * 10) / 10):.1f}"
        entry = confidence_buckets.setdefault(bucket, {"n": 0, "agree": 0})
        entry["n"] += 1
        entry["agree"] += int(row["accept"] == row["human_accept"])

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    latencies: list[float] = []
    for verdict in verdict_rows:
        usage = verdict.get("usage") or {}
        for key in total_usage:
            total_usage[key] += int(usage.get(key) or 0)
        latencies.append(float(verdict.get("latency_ms") or 0.0))

    sweep = policy_sweep(
        review_rows=review_rows,
        verdict_rows=verdict_rows,
        review_by_id=review_by_id,
    )
    policy = recommend_policy(per_class)
    return {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "model": settings.public(),
        "reviewed_proposals": len(review_rows),
        "verified_proposals": len(rows),
        "primary_verdicts": len(primary),
        "second_pass_verdicts": len(secondary),
        "calls_used": calls_used,
        "skipped": list(skipped),
        "artifacts": {
            "review": review_path,
            "proposals": proposals_path,
            "crops_dir": crops_dir,
        },
        "parameters": {
            "min_confidence": min_confidence,
            "batch_size": batch_size,
            "accepted_tightness": list(accepted_tightness),
        },
        "overall": overall,
        "per_class": per_class,
        "confusion": confusion,
        "confidence_buckets": confidence_buckets,
        "disagreements": disagreements,
        "usage": total_usage,
        "latency_ms_mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "recommended_policy": policy,
        "profile_sweep": profile_sweep(verdict_rows=verdict_rows, review_by_id=review_by_id),
        "policy_sweep": sweep,
        "policy_sweep_best": _best_sweep(sweep),
    }


def _effective_rows(verdict_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    effective: dict[str, Mapping[str, Any]] = {}
    for row in verdict_rows:
        effective[row["proposal_id"]] = row
    return effective


def policy_sweep(
    *,
    review_rows: Sequence[Mapping[str, str]],
    verdict_rows: Sequence[Mapping[str, Any]],
    review_by_id: Mapping[str, Mapping[str, str]] | None = None,
    confidences: Sequence[float] = (0.0, 0.5, 0.7, 0.8, 0.9),
    tightness_sets: Sequence[Sequence[str]] = (("keep",), ("keep", "too_tight")),
) -> list[dict[str, Any]]:
    """Recompute accept metrics across thresholds — pure, zero spend.

    Lets the report recommend a threshold from the data instead of asserting one.
    """

    review_by_id = review_by_id or {
        row["proposal_id"]: row for row in review_rows
    }
    effective = _effective_rows(verdict_rows)
    results: list[dict[str, Any]] = []
    for tightness in tightness_sets:
        for confidence in confidences:
            per_class_rows: dict[str, list[dict[str, Any]]] = {}
            for proposal_id, row in effective.items():
                review = review_by_id.get(proposal_id) or {}
                decision = accept_decision(
                    row["verdict"],
                    row["proposed_class"],
                    min_confidence=confidence,
                    accepted_tightness=tightness,
                )
                per_class_rows.setdefault(row["proposed_class"], []).append(
                    {
                        "accept": bool(decision["accept"]),
                        "human_accept": (review.get("verdict") or "").lower() == "tp",
                    }
                )
            all_rows = [item for rows in per_class_rows.values() for item in rows]
            metrics = _binary_metrics(all_rows)
            results.append(
                {
                    "min_confidence": confidence,
                    "accepted_tightness": list(tightness),
                    "n": metrics["n"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "per_class": {
                        name: {
                            key: _binary_metrics(rows)[key]
                            for key in ("n", "tp", "fp", "fn", "precision", "recall")
                        }
                        for name, rows in per_class_rows.items()
                    },
                }
            )
    return results


def profile_sweep(
    *,
    verdict_rows: Sequence[Mapping[str, Any]],
    review_by_id: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Evaluate the named accept profiles (pure; no network, no spend)."""

    effective = _effective_rows(verdict_rows)
    results: dict[str, Any] = {}
    for name in POLICY_PROFILES:
        per_class: dict[str, list[dict[str, Any]]] = {}
        flagged: list[dict[str, Any]] = []
        for proposal_id, row in effective.items():
            review = review_by_id.get(proposal_id) or {}
            decision = accept_decision(row["verdict"], row["proposed_class"], profile=name)
            per_class.setdefault(row["proposed_class"], []).append(
                {
                    "accept": bool(decision["accept"]),
                    "human_accept": (review.get("verdict") or "").lower() == "tp",
                }
            )
            if decision["accept"] and decision.get("flags"):
                flagged.append({"proposal_id": proposal_id, "flags": decision["flags"]})
        all_rows = [item for rows in per_class.values() for item in rows]
        results[name] = {
            "description": POLICY_PROFILES[name]["description"],
            "overall": _binary_metrics(all_rows),
            "per_class": {cls: _binary_metrics(rows) for cls, rows in per_class.items()},
            "flagged_accepts": flagged,
        }
    return results


def _best_sweep(sweep: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Highest-F1 policy, preferring higher precision on ties."""

    candidates = [row for row in sweep if row.get("f1") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["f1"], row["precision"] or 0.0, row["recall"] or 0.0))


def _confusion(
    rows: Sequence[Mapping[str, Any]],
    review_by_id: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        human = row.get("human_class") or "not_target"
        predicted = row.get("verdict_class") or "ambiguous"
        matrix.setdefault(human, {})
        matrix[human][predicted] = matrix[human].get(predicted, 0) + 1
    return matrix


def recommend_policy(per_class: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    auto_accept: list[str] = []
    review_only: list[str] = []
    insufficient: list[str] = []
    unusable: list[str] = []
    not_evaluated: list[str] = []
    for class_name, metrics in per_class.items():
        trust = metrics.get("trust")
        if not int(metrics.get("n") or 0):
            not_evaluated.append(class_name)
        elif trust == "auto_accept":
            auto_accept.append(class_name)
        elif trust in {"review_only", "review_only_low_volume"}:
            review_only.append(class_name)
        elif trust == "insufficient_evidence":
            insufficient.append(class_name)
        else:
            unusable.append(class_name)
    return {
        "auto_accept_classes": auto_accept,
        "review_only_classes": review_only,
        "insufficient_evidence_classes": insufficient,
        "unusable_classes": unusable,
        "not_evaluated_classes": not_evaluated,
        "rule": (
            "auto_accept needs n>=20, accept precision>=0.95 with 95% Wilson "
            "lower bound >=0.90, and recall>=0.80; review_only: precision>=0.85, "
            "recall>=0.60; insufficient_evidence below n=20 or when a ratio is "
            "undefined; unusable: precision<0.70"
        ),
    }


def reparse_raw_calls(
    *,
    raw_calls_path: str | Path,
    verdicts_path: str | Path | None = None,
    known_names: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Re-derive verdict rows from stored raw responses (no network, no spend).

    Prompt or guard changes must never require paying for the same crops twice.
    Prefers the per-item metadata stored inside each raw call, so a crashed run
    can be fully reconstructed from ``raw_calls.jsonl`` alone.
    """

    stored: list[dict[str, Any]] = []
    if verdicts_path is not None and Path(verdicts_path).exists():
        stored = [
            json.loads(line)
            for line in Path(verdicts_path).read_text().splitlines()
            if line.strip()
        ]
    by_id: dict[tuple[str, int], dict[str, Any]] = {
        (row["proposal_id"], int(row.get("pass") or 1)): row for row in stored
    }
    known_names = known_names or {}

    revalidated: dict[str, dict[str, Any]] = {}
    for line in Path(raw_calls_path).read_text().splitlines():
        if not line.strip():
            continue
        call = json.loads(line)
        content = call.get("content")
        if not call.get("ok") or not content:
            continue
        proposal_ids = list(call.get("proposal_ids") or [])
        pass_number = int(call.get("pass") or 1)
        templates: list[dict[str, Any]] = []
        for pid in proposal_ids:
            stored_row = by_id.get((pid, pass_number))
            if stored_row is not None:
                templates.append(stored_row)
                continue
            item = next(
                (entry for entry in call.get("items") or [] if entry.get("proposal_id") == pid),
                None,
            )
            if item is None:
                break
            templates.append(
                {
                    "proposal_id": pid,
                    "photo_id": item.get("photo_id"),
                    "pass": pass_number,
                    "proposed_class": item.get("proposed_class"),
                    "proposal_box_score": item.get("proposal_box_score"),
                    "review_verdict": item.get("review_verdict") or "",
                    "review_notes": item.get("review_notes") or "",
                    "model": call.get("model"),
                    "prompt_hash": call.get("prompt_hash"),
                    "crop": item.get("crop") or {},
                    "latency_ms": call.get("latency_ms"),
                    "usage": call.get("usage") or {},
                }
            )
        if len(templates) != len(proposal_ids):
            continue
        try:
            verdicts = parse_verdicts(
                content,
                expected=len(templates),
                proposed_classes=[row["proposed_class"] for row in templates],
                known_names=[
                    name
                    for row in templates
                    for name in known_names.get(str(row.get("photo_id")), [])
                ],
            )
        except VerdictParseError:
            continue
        payload = extract_json_payload(content)
        raw_items = payload.get("verdicts") if isinstance(payload, Mapping) else payload
        for index, (template, verdict) in enumerate(zip(templates, verdicts)):
            updated = dict(template)
            updated["verdict"] = verdict
            updated["raw_verdict"] = raw_items[index]
            decision = accept_decision(
                verdict,
                str(updated.get("proposed_class")),
            )
            updated["decision"] = decision
            revalidated[template["proposal_id"] + f":{template['pass']}"] = updated

    ordered: list[dict[str, Any]] = []
    for row in stored:
        ordered.append(revalidated.get(f"{row['proposal_id']}:{row['pass']}", row))
    return ordered


def render_report_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Anatomy4 crop-verifier evaluation")
    lines.append("")
    lines.append(f"- schema: `{report['schema']}`")
    lines.append(f"- model: `{report['model']['model']}` (temperature {report['model']['temperature']})")
    lines.append(f"- prompt: `{report['prompt_version']}` `{report['prompt_hash']}`")
    lines.append(
        f"- reviewed proposals: {report['reviewed_proposals']}; verified: "
        f"{report['verified_proposals']}; calls: {report['calls_used']}; "
        f"second pass: {report['second_pass_verdicts']}"
    )
    lines.append(
        f"- tokens: {report['usage']['total_tokens']} total "
        f"(prompt {report['usage']['prompt_tokens']}, completion {report['usage']['completion_tokens']}); "
        f"mean latency {report['latency_ms_mean']} ms"
    )
    lines.append("")
    lines.append("## Accept-decision agreement with the human review")
    lines.append("")
    lines.append(
        "| class | n | TP | FP | TN | FN | precision | precision 95% CI | recall | recall 95% CI | F1 | trust |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |"
    )
    for class_name, metrics in report["per_class"].items():
        lines.append(
            f"| {class_name} | {metrics['n']} | {metrics['tp']} | {metrics['fp']} | "
            f"{metrics['tn']} | {metrics['fn']} | {_fmt(metrics['precision'])} | "
            f"{_ci(metrics.get('precision_ci95'))} | {_fmt(metrics['recall'])} | "
            f"{_ci(metrics.get('recall_ci95'))} | {_fmt(metrics['f1'])} | {metrics['trust']} |"
        )
    overall = report["overall"]
    lines.append(
        f"| **all** | {overall['n']} | {overall['tp']} | {overall['fp']} | {overall['tn']} | "
        f"{overall['fn']} | {_fmt(overall['precision'])} | {_ci(overall.get('precision_ci95'))} | "
        f"{_fmt(overall['recall'])} | {_ci(overall.get('recall_ci95'))} | "
        f"{_fmt(overall['f1'])} | {overall['trust']} |"
    )
    lines.append("")
    lines.append("## Verdict-class confusion (rows = human label, columns = verifier)")
    lines.append("")
    columns = [*CLASS_NAMES, "not_target", "ambiguous"]
    lines.append("| human \\ verifier | " + " | ".join(columns) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in columns) + " |")
    for human in [*CLASS_NAMES, "not_target", "ambiguous"]:
        row = report["confusion"].get(human, {})
        total = sum(row.values())
        if not total:
            continue
        lines.append(
            f"| {human} | " + " | ".join(str(row.get(column, 0)) for column in columns) + " |"
        )
    lines.append("")
    lines.append("## Disagreements")
    lines.append("")
    if not report["disagreements"]:
        lines.append("None.")
    else:
        lines.append(
            "| proposal | class | human | verifier | conf | verifier reason | human note |"
        )
        lines.append("| --- | --- | --- | --- | ---: | --- | --- |")
        for item in report["disagreements"]:
            human = "accept" if item["human_accept"] else "reject"
            verifier = "accept" if item["verifier_accept"] else "reject"
            lines.append(
                f"| {item['proposal_id']} | {item['class']} | {human} | "
                f"{verifier} ({item['verifier_class']}) | {item['verifier_confidence']:.2f} | "
                f"{_escape(item['verifier_reason'])} | {_escape(item['human_notes'])} |"
            )
    lines.append("")
    lines.append("## Recommended policy")
    lines.append("")
    policy = report["recommended_policy"]
    lines.append(f"- auto-accept: {', '.join(policy['auto_accept_classes']) or 'none'}")
    lines.append(f"- review-only: {', '.join(policy['review_only_classes']) or 'none'}")
    lines.append(
        f"- insufficient evidence (n<{MIN_POLICY_SAMPLES}): "
        f"{', '.join(policy.get('insufficient_evidence_classes') or []) or 'none'}"
    )
    lines.append(f"- unusable: {', '.join(policy['unusable_classes']) or 'none'}")
    lines.append(
        f"- not evaluated: {', '.join(policy.get('not_evaluated_classes') or []) or 'none'}"
    )
    lines.append(f"- rule: {policy['rule']}")
    lines.append("")
    profiles = report.get("profile_sweep") or {}
    if profiles:
        lines.append("## Accept-profile comparison")
        lines.append("")
        lines.append("| profile | TP | FP | FN | precision | recall | F1 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name, entry in profiles.items():
            overall_profile = entry["overall"]
            lines.append(
                f"| {name} | {overall_profile['tp']} | {overall_profile['fp']} | "
                f"{overall_profile['fn']} | {_fmt(overall_profile['precision'])} | "
                f"{_fmt(overall_profile['recall'])} | {_fmt(overall_profile['f1'])} |"
            )
        lines.append("")
        for name, entry in profiles.items():
            classes = ", ".join(
                f"{cls} P={_fmt(metrics['precision'])} R={_fmt(metrics['recall'])}"
                for cls, metrics in entry["per_class"].items()
            )
            lines.append(f"- `{name}`: {entry['description']} — {classes}")
        flagged = {
            name: entry["flagged_accepts"]
            for name, entry in profiles.items()
            if entry["flagged_accepts"]
        }
        if flagged:
            lines.append("")
            for name, items in flagged.items():
                rendered = ", ".join(
                    f"{item['proposal_id']} ({'+'.join(item['flags'])})" for item in items
                )
                lines.append(f"- `{name}` accepts carrying a tightness flag: {rendered}")
        lines.append("")
    lines.append("## Threshold / tightness sweep (thresholds are data-driven, no re-spend)")
    lines.append("")
    lines.append(
        "| min confidence | accepted tightness | TP | FP | FN | precision | recall | F1 |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report.get("policy_sweep") or []:
        lines.append(
            f"| {row['min_confidence']:.2f} | {'+'.join(row['accepted_tightness'])} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | {_fmt(row['precision'])} | "
            f"{_fmt(row['recall'])} | {_fmt(row['f1'])} |"
        )
    best = report.get("policy_sweep_best")
    if best:
        lines.append("")
        lines.append(
            f"Best F1 in the sweep: min_confidence={best['min_confidence']:.2f}, "
            f"tightness={'+'.join(best['accepted_tightness'])} "
            f"(P={_fmt(best['precision'])}, R={_fmt(best['recall'])}, F1={_fmt(best['f1'])})."
        )
    lines.append("")
    lines.append("## Confidence calibration (accept-decision agreement)")
    lines.append("")
    lines.append("| confidence bucket | n | agreement |")
    lines.append("| --- | ---: | ---: |")
    for bucket, entry in sorted(report["confidence_buckets"].items()):
        lines.append(
            f"| {bucket} | {entry['n']} | {entry['agree']}/{entry['n']} |"
        )
    lines.append("")
    if report.get("skipped"):
        lines.append("## Skipped")
        lines.append("")
        for item in report["skipped"]:
            lines.append(f"- {item['proposal_id']}: {item['reason']}")
        lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _ci(value: Any) -> str:
    if not value or len(value) != 2 or value[0] is None or value[1] is None:
        return "—"
    return f"{float(value[0]):.2f}–{float(value[1]):.2f}"


def _escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
