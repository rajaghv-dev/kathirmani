"""Rule loading + pure condition evaluators (master plan §7.3, Phase 5).

The rule engine is **deterministic** — safety logic stays in code, never inside
a learned policy (spec/10 anti-goals). This module:

  * loads `configs/event_rules.yaml` into `Rule` objects (id + conditions + action), and
  * provides one *pure function* per condition keyword. Each evaluator takes a
    small `TrackContext` (the recent events for one person/track) and a threshold,
    and returns a `Match` (fired bool + human-readable `why`). No I/O, no DB, no ML —
    so they're trivially unit-testable and explainable.

The CV worker (Phase 4) emits low-level `CommonEvent`s (§8.3) onto the queue; here
each event is folded into the matching track's context, then the rules are evaluated
against that context window. See engine.py for the orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# event_rules.yaml lives at repo root /configs; this file is services/rule-engine/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = _REPO_ROOT / "configs" / "event_rules.yaml"


# --------------------------------------------------------------------------
# Track / event context — the small in-memory window a rule reasons over.
# --------------------------------------------------------------------------
@dataclass
class TrackEvent:
    """One low-level observation about a track at a point in time.

    Built from a CommonEvent-ish dict (see engine.event_to_track_event). The
    fields are the minimum the deterministic rules need: when, where (zones +
    their types), what was seen (objects), and a few booleans the CV layer may
    pre-compute (object_near_hand, has_item, camera_health_score)."""
    ts: float                                   # event time, seconds (monotonic ok)
    camera_id: str = ""
    zones: list[str] = field(default_factory=list)       # zone ids
    zone_types: list[str] = field(default_factory=list)  # e.g. ["shelf"]
    objects: list[str] = field(default_factory=list)     # detected labels
    object_near_hand: bool = False
    has_item: bool = False
    camera_health_score: float | None = None
    event_type: str = ""
    raw: dict = field(default_factory=dict)              # original payload (provenance)


@dataclass
class TrackContext:
    """A bounded sequence of events for one (camera, track) subject, ordered by ts.

    The engine maintains one of these per track and feeds it to the evaluators.
    `add` keeps events time-sorted; nothing here mutates global state."""
    track_id: str
    store_id: str = "kathirmani_01"
    camera_id: str = ""
    events: list[TrackEvent] = field(default_factory=list)

    def add(self, ev: TrackEvent) -> None:
        self.events.append(ev)
        self.events.sort(key=lambda e: e.ts)
        if ev.camera_id and not self.camera_id:
            self.camera_id = ev.camera_id

    @property
    def latest(self) -> TrackEvent | None:
        return self.events[-1] if self.events else None

    def zone_types_seen(self) -> set[str]:
        return {zt for e in self.events for zt in e.zone_types}

    def all_objects(self) -> set[str]:
        return {o for e in self.events for o in e.objects}


# --------------------------------------------------------------------------
# Match — the explainable result of one evaluator.
# --------------------------------------------------------------------------
@dataclass
class Match:
    fired: bool
    why: str = ""

    def __bool__(self) -> bool:                 # so `if person_in_zone(...):` works
        return self.fired


# --------------------------------------------------------------------------
# Pure condition evaluators. Signature: (ctx, value) -> Match.
# `value` is the YAML threshold/argument for that condition.
# --------------------------------------------------------------------------
def person_in_zone(ctx: TrackContext, zone_type: str) -> Match:
    """Did the track appear in any zone of `zone_type` (e.g. 'shelf')?"""
    hit = any(zone_type in e.zone_types for e in ctx.events)
    return Match(hit, f"track {ctx.track_id} {'in' if hit else 'never in'} {zone_type} zone")


def dwell_time_sec_gte(ctx: TrackContext, seconds: float) -> Match:
    """Has the track been observed for >= `seconds` (last ts - first ts)?

    Dwell is the span of the context window — the time between the first and
    last observation of this subject. With <2 events dwell is 0."""
    if len(ctx.events) < 2:
        return Match(False, f"dwell 0.0s < {seconds}s (only {len(ctx.events)} obs)")
    dwell = ctx.events[-1].ts - ctx.events[0].ts
    ok = dwell >= float(seconds)
    return Match(ok, f"dwell {dwell:.1f}s {'>=' if ok else '<'} {seconds}s")


def object_near_person_hand(ctx: TrackContext, expected: bool = True) -> Match:
    """Was an object detected near the person's hand in any observation?

    The CV layer sets `object_near_hand` (proximity heuristic between a hand/
    person bbox and an item bbox). We only assert presence, not classification."""
    near = any(e.object_near_hand for e in ctx.events)
    ok = near == bool(expected)
    return Match(ok, f"object_near_hand={near} (expected {bool(expected)})")


def person_has_item(ctx: TrackContext, expected: bool = True) -> Match:
    """Is the track carrying an item (has_item flag, or a non-person object seen)?"""
    carried = any(e.has_item for e in ctx.events) or bool(
        ctx.all_objects() - {"person"})
    ok = carried == bool(expected)
    return Match(ok, f"has_item={carried} (expected {bool(expected)})")


def visited_billing_zone(ctx: TrackContext, expected: bool) -> Match:
    """Did the track ever enter a billing_counter zone?

    Note YAML asks `visited_billing_zone: false` for the bypass rule — so this
    fires when the *observed* visited-state matches the expected boolean."""
    visited = "billing_counter" in ctx.zone_types_seen()
    ok = visited == bool(expected)
    return Match(ok, f"visited_billing={visited} (expected {bool(expected)})")


def moved_to_zone(ctx: TrackContext, zone_type: str) -> Match:
    """Is the track's *most recent* zone of type `zone_type` (e.g. 'exit')?

    'Moved to' = the latest observation lands in that zone type — a transition
    toward it, distinct from merely having visited it earlier."""
    last = ctx.latest
    ok = bool(last and zone_type in last.zone_types)
    return Match(ok, f"latest zone_types={last.zone_types if last else []} "
                     f"{'includes' if ok else 'excludes'} {zone_type}")


def camera_health_score_lte(ctx: TrackContext, threshold: float) -> Match:
    """Did any observation report a camera_health_score <= threshold?

    Used for the camera-health rule (blocked/black/frozen frame). The lowest
    observed score is the worst case, so we test that against the threshold."""
    scores = [e.camera_health_score for e in ctx.events
              if e.camera_health_score is not None]
    if not scores:
        return Match(False, "no camera_health_score observed")
    worst = min(scores)
    ok = worst <= float(threshold)
    return Match(ok, f"camera_health_score {worst:.2f} {'<=' if ok else '>'} {threshold}")


# Registry: YAML condition keyword -> evaluator. The engine dispatches through this.
EVALUATORS: dict[str, Callable[[TrackContext, Any], Match]] = {
    "person_in_zone": person_in_zone,
    "dwell_time_sec_gte": dwell_time_sec_gte,
    "object_near_person_hand": object_near_person_hand,
    "person_has_item": person_has_item,
    "visited_billing_zone": visited_billing_zone,
    "moved_to_zone": moved_to_zone,
    "camera_health_score_lte": camera_health_score_lte,
}


# --------------------------------------------------------------------------
# Rule model + loader.
# --------------------------------------------------------------------------
@dataclass
class Action:
    """The `action:` block of a rule — exactly one of create_event/create_incident."""
    create_event: str | None = None
    create_incident: str | None = None
    severity: str = "info"
    needs_vlm_verification: bool = False


@dataclass
class Rule:
    id: str
    description: str
    conditions: list[tuple[str, Any]]           # [(keyword, value), ...] — all must hold
    action: Action

    def unknown_conditions(self) -> list[str]:
        """Condition keywords with no registered evaluator (config drift guard)."""
        return [k for k, _ in self.conditions if k not in EVALUATORS]


def _parse_conditions(raw: list[dict]) -> list[tuple[str, Any]]:
    """YAML conditions are a list of single-key dicts: [{person_in_zone: shelf}, ...]."""
    out: list[tuple[str, Any]] = []
    for item in raw or []:
        if isinstance(item, dict):
            for k, v in item.items():
                out.append((k, v))
    return out


def _parse_action(raw: dict) -> Action:
    raw = raw or {}
    return Action(
        create_event=raw.get("create_event"),
        create_incident=raw.get("create_incident"),
        severity=raw.get("severity", "info"),
        needs_vlm_verification=bool(raw.get("needs_vlm_verification", False)),
    )


def load_rules(path: str | Path | None = None) -> list[Rule]:
    """Load + parse `event_rules.yaml`. Best-effort: a missing file / no PyYAML
    yields an empty rule set so the worker still runs (just fires nothing)."""
    p = Path(path) if path else DEFAULT_RULES_PATH
    try:
        import yaml
        with open(p) as fh:
            doc = yaml.safe_load(fh) or {}
    except Exception:
        return []
    rules: list[Rule] = []
    for r in doc.get("rules", []):
        rules.append(Rule(
            id=r.get("id", ""),
            description=r.get("description", ""),
            conditions=_parse_conditions(r.get("conditions", [])),
            action=_parse_action(r.get("action", {})),
        ))
    return rules
