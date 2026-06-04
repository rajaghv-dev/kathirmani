"""rule-engine — deterministic event intelligence (master plan Phase 5).

Consumes the low-level §8.3 CommonEvents the CV worker publishes (`event.created`),
folds them into per-track context windows, and applies `configs/event_rules.yaml`
rules (zone/dwell/interaction/billing-bypass/camera-health) to raise higher-level
events + incidents — WITHOUT a VLM (Phase 6 verifies later). Pure-python + pyyaml;
every emitted action is explainable (which rule fired + why). Safety logic stays
deterministic, never inside a learned policy (spec/10 anti-goals).
"""
from __future__ import annotations

# Imports tolerate both package use and the repo's path-based test setup (the
# dir name `rule-engine` isn't a valid module name, so relative-import context
# isn't always present).
try:
    from .engine import (EngineAction, RuleEngine, action_to_event_dict,
                         event_to_track_event)
    from .rules import (EVALUATORS, Action, Match, Rule, TrackContext,
                        TrackEvent, load_rules)
except ImportError:                                   # path-based / direct import
    from engine import (EngineAction, RuleEngine, action_to_event_dict,
                        event_to_track_event)
    from rules import (EVALUATORS, Action, Match, Rule, TrackContext,
                       TrackEvent, load_rules)

__all__ = [
    "RuleEngine",
    "EngineAction",
    "action_to_event_dict",
    "event_to_track_event",
    "Rule",
    "Action",
    "Match",
    "TrackContext",
    "TrackEvent",
    "EVALUATORS",
    "load_rules",
]
