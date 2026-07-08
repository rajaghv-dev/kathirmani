"""Versioned prompt packs for the VLM clip-reasoning task (spec/11 — prompt is a
separate axis from model/runtime/schema, so prompts version independently).

A pack is a `render(event) -> str`. Phrasing is **neutral retail-security**: we
ask the model to describe what it observes and judge against a stated hypothesis,
never to accuse a person — the output is loss-prevention triage evidence, not a
verdict on an individual. The pack name is referenced by `prompt_pack:` in the
active profile's `vlm_clip_reasoning` task (configs/models.yaml: `retail_loss_v1`).

The prompt always asks for a single JSON object matching the VLMVerification
schema (§8.4), which `parser.py` then extracts/repairs.
"""
from __future__ import annotations

from typing import Any, Callable

# The exact field contract we want the model to emit — kept here (not in the
# free-text instructions) so the prompt and the parser stay in lock-step.
_JSON_CONTRACT = (
    '{"verdict": "suspicious|benign|unclear", '
    '"confidence": 0.0, '
    '"observed_actions": ["..."], '
    '"missing_evidence": ["..."], '
    '"recommended_next_step": "...", '
    '"structured_event_type": "...", '
    '"explanation": "..."}'
)


def _event_summary(event: dict[str, Any]) -> str:
    """Compact, human-readable line describing the rule_engine hypothesis under
    review (tolerant of both the raw event dict and its nested metadata_json)."""
    meta = event.get("metadata_json") or {}

    def g(key: str, default: Any = None) -> Any:
        return event.get(key, meta.get(key, default))

    etype = event.get("event_type", "unspecified_concern")
    camera = event.get("camera_id", "unknown")
    zones = g("zones", []) or []
    objects = g("objects", []) or []
    why = g("explanation", "") or ""
    parts = [f"hypothesis='{etype}'", f"camera='{camera}'"]
    if zones:
        parts.append(f"zones={list(zones)}")
    if objects:
        parts.append(f"objects={list(objects)}")
    if why:
        parts.append(f"rule_reason=\"{why}\"")
    return ", ".join(parts)


def render_retail_loss_v1(event: dict[str, Any]) -> str:
    """retail_loss_v1 — neutral retail loss-prevention verification prompt."""
    summary = _event_summary(event)
    return (
        "You are a retail store-operations analyst reviewing a short surveillance "
        "video clip. An automated rule flagged this clip for a second look. Your job "
        "is to describe what is visible and judge whether the clip supports, "
        "contradicts, or is inconclusive about the stated concern. Be factual and "
        "neutral: describe observable actions only, do not accuse any individual, and "
        "do not infer intent beyond what the footage shows.\n\n"
        f"Flagged concern: {summary}\n\n"
        "Decide a verdict:\n"
        "  - \"suspicious\": the footage clearly supports the concern.\n"
        "  - \"benign\": the footage shows ordinary, expected store activity.\n"
        "  - \"unclear\": the footage is inconclusive or too obstructed to judge.\n\n"
        "Respond with ONLY a single JSON object (no prose, no markdown fences) of "
        "exactly this shape:\n"
        f"{_JSON_CONTRACT}\n\n"
        "Where: confidence is 0.0-1.0; observed_actions lists the concrete actions "
        "you saw; missing_evidence lists what you would need to be more certain; "
        "recommended_next_step is a brief operational suggestion (e.g. 'review billing "
        "footage', 'no action'); structured_event_type is a short snake_case label; "
        "explanation is one or two sentences justifying the verdict."
    )


# Registry: prompt_pack name -> render fn. `host`/`plugin` look the pack up here.
PROMPT_PACKS: dict[str, Callable[[dict[str, Any]], str]] = {
    "retail_loss_v1": render_retail_loss_v1,
}

DEFAULT_PACK = "retail_loss_v1"


def render(event: dict[str, Any], pack: str = DEFAULT_PACK) -> str:
    """Render `event` with the named prompt pack (falls back to the default)."""
    fn = PROMPT_PACKS.get(pack, PROMPT_PACKS[DEFAULT_PACK])
    return fn(event)
