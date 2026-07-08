"""RuleEngine — deterministic event intelligence (master plan §3.3, Phase 5).

Folds an incoming stream of low-level events (the §8.3 CommonEvents the CV worker
publishes onto the queue) into per-track context windows, then evaluates the
`configs/event_rules.yaml` rules against each window. When a rule's conditions ALL
hold it yields an explainable `Action`:

  * `create_event(event_type, severity, needs_vlm_verification)`  — a higher-level
    hypothesis (e.g. suspicious_item_interaction) for the Phase-6 VLM to verify, or
  * `create_incident(incident_type, severity)`                    — e.g. possible_loss.

No VLM, no ML — the rules are pure functions (rules.py) and every emitted action
carries `rule_id` + `why` (the per-condition explanations) so a human can audit
exactly why it fired. Each (track, rule) fires at most once (de-dup) within an engine
instance, so re-seeing the same evidence doesn't spam duplicates.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from .rules import (EVALUATORS, Match, Rule, TrackContext, TrackEvent,
                    load_rules)


# --------------------------------------------------------------------------
# EngineAction — the explainable output unit.
# --------------------------------------------------------------------------
@dataclass
class EngineAction:
    kind: str                                   # "create_event" | "create_incident"
    type: str                                   # event_type or incident_type
    severity: str = "info"
    needs_vlm_verification: bool = False
    rule_id: str = ""
    store_id: str = "kathirmani_01"
    camera_id: str = ""
    track_id: str = ""
    zones: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)   # contributing low-level events
    why: list[str] = field(default_factory=list)                # per-condition explanations

    def explanation(self) -> str:
        return f"rule '{self.rule_id}' fired: " + "; ".join(self.why)


# --------------------------------------------------------------------------
# Event normalization: a raw queue payload -> a TrackEvent.
# --------------------------------------------------------------------------
def event_to_track_event(ev: dict) -> TrackEvent:
    """Map a CommonEvent-ish dict (queue payload) to a TrackEvent.

    Tolerant of the fields the CV worker actually sets plus the few hints the
    rules want (object_near_hand / has_item / camera_health_score / zone_types);
    when zone_types isn't pre-resolved we fall back to the raw `zones` list."""
    meta = ev.get("metadata_json") or {}        # DB-shaped rows nest extras here
    def g(key, default=None):
        return ev.get(key, meta.get(key, default))

    zones = g("zones", []) or []
    zone_types = g("zone_types", []) or []
    return TrackEvent(
        ts=float(g("ts", g("event_time_sec", 0.0)) or 0.0),
        camera_id=ev.get("camera_id", "") or "",
        zones=list(zones),
        zone_types=list(zone_types),
        objects=list(g("objects", []) or []),
        object_near_hand=bool(g("object_near_hand", False)),
        has_item=bool(g("has_item", False)),
        camera_health_score=_opt_float(g("camera_health_score")),
        event_type=ev.get("event_type", "") or "",
        raw=dict(ev),
    )


def _opt_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _track_key(ev: dict) -> str:
    """Subject identity for context bucketing. Prefer an explicit track id; else
    use the first track_id; else fall back to camera (so camera-health events,
    which have no person track, still bucket sensibly)."""
    if ev.get("track_id"):
        return str(ev["track_id"])
    tids = ev.get("track_ids") or (ev.get("metadata_json") or {}).get("track_ids") or []
    if tids:
        return str(tids[0])
    return f"cam:{ev.get('camera_id', 'unknown')}"


# --------------------------------------------------------------------------
# RuleEngine.
# --------------------------------------------------------------------------
class RuleEngine:
    """Maintains per-track context windows and applies the YAML rules to a stream.

    Deterministic + stateful only in the bounded context windows + a fired-set for
    de-dup. Construct once, then call `feed(event)` per incoming event (returns the
    actions that newly fired), or `feed_many(events)` for a batch/sequence."""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules: list[Rule] = rules if rules is not None else load_rules()
        self._contexts: dict[str, TrackContext] = {}
        self._fired: set[tuple[str, str]] = set()   # (track_id, rule_id) already emitted

    # ---- context management ------------------------------------------------
    def _context_for(self, ev: dict) -> TrackContext:
        key = _track_key(ev)
        ctx = self._contexts.get(key)
        if ctx is None:
            ctx = TrackContext(track_id=key,
                               store_id=ev.get("store_id", "kathirmani_01") or "kathirmani_01",
                               camera_id=ev.get("camera_id", "") or "")
            self._contexts[key] = ctx
        return ctx

    # ---- evaluation --------------------------------------------------------
    def _evaluate(self, rule: Rule, ctx: TrackContext) -> tuple[bool, list[str]]:
        """All conditions must hold. Returns (fired, [why,...]). An unknown
        condition keyword (config drift) blocks the rule rather than firing it."""
        whys: list[str] = []
        for keyword, value in rule.conditions:
            fn = EVALUATORS.get(keyword)
            if fn is None:                       # unknown keyword -> cannot satisfy
                return False, [f"unknown condition '{keyword}'"]
            m: Match = fn(ctx, value)
            whys.append(m.why)
            if not m.fired:
                return False, whys
        return True, whys

    def feed(self, ev: dict) -> list[EngineAction]:
        """Ingest one event; return any actions that newly fire for its track."""
        ctx = self._context_for(ev)
        ctx.add(event_to_track_event(ev))
        actions: list[EngineAction] = []
        for rule in self.rules:
            key = (ctx.track_id, rule.id)
            if key in self._fired:
                continue
            fired, whys = self._evaluate(rule, ctx)
            if not fired:
                continue
            self._fired.add(key)
            actions.append(self._build_action(rule, ctx, whys))
        return actions

    def feed_many(self, events: Iterable[dict]) -> list[EngineAction]:
        out: list[EngineAction] = []
        for ev in events:
            out.extend(self.feed(ev))
        return out

    def _build_action(self, rule: Rule, ctx: TrackContext,
                      whys: list[str]) -> EngineAction:
        a = rule.action
        kind = "create_incident" if a.create_incident else "create_event"
        out_type = a.create_incident or a.create_event or rule.id
        src_ids = [e.raw.get("event_id") for e in ctx.events if e.raw.get("event_id")]
        zones = sorted({z for e in ctx.events for z in e.zones})
        return EngineAction(
            kind=kind,
            type=out_type,
            severity=a.severity,
            needs_vlm_verification=a.needs_vlm_verification,
            rule_id=rule.id,
            store_id=ctx.store_id,
            camera_id=ctx.camera_id,
            track_id=ctx.track_id,
            zones=zones,
            source_event_ids=[s for s in src_ids if s],
            why=whys,
        )


# --------------------------------------------------------------------------
# Action -> §8.3 CommonEvent dict (what we persist/republish for events).
# --------------------------------------------------------------------------
def action_to_event_dict(action: EngineAction) -> dict:
    """Render a create_event action as a queue/DB-shaped CommonEvent dict.

    event_id is a deterministic uuid5 of (rule_id, track_id, type) so re-runs are
    idempotent and dedupe naturally at the DB layer."""
    seed = f"{action.rule_id}|{action.track_id}|{action.type}"
    return {
        "type": "event.created",
        "event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, seed)),
        "store_id": action.store_id,
        "camera_id": action.camera_id,
        "event_type": action.type,
        "severity": action.severity,
        "confidence": 0.0,                       # deterministic rule — no model score
        "source_engine": "rule_engine",
        "track_ids": [action.track_id],
        "zones": action.zones,
        "needs_vlm_verification": action.needs_vlm_verification,
        "rule_id": action.rule_id,
        "explanation": action.explanation(),
        "source_event_ids": action.source_event_ids,
    }
