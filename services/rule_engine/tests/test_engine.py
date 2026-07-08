"""Integration tests for the RuleEngine over synthetic + golden event sequences.

No DB / services / GPU — feed dict events to the engine and assert the
explainable actions it yields, plus a regression test against the golden
fixtures (master plan §17 shape).
"""
from __future__ import annotations

import json
from pathlib import Path

from services.rule_engine.engine import RuleEngine, action_to_event_dict

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "testdata" / "golden"


def _kinds(actions, kind):
    return [a for a in actions if a.kind == kind]


# --------------------------------------------------------------------------
# Synthetic sequences (the deliverable's named scenarios).
# --------------------------------------------------------------------------
def test_shelf_dwell_then_object_near_hand_fires_suspicious():
    """Person enters shelf zone -> dwells 4s -> object near hand =>
    suspicious_item_interaction event (needs VLM verification)."""
    eng = RuleEngine()
    seq = [
        {"event_id": "e1", "store_id": "kathirmani_01", "camera_id": "left_aisle",
         "track_id": "p1", "ts": 0.0, "zone_types": ["shelf"], "zones": ["left_shelf_zone"],
         "objects": ["person"]},
        {"event_id": "e2", "camera_id": "left_aisle", "track_id": "p1", "ts": 4.0,
         "zone_types": ["shelf"], "zones": ["left_shelf_zone"],
         "objects": ["person", "item"], "object_near_hand": True},
    ]
    actions = eng.feed_many(seq)
    evs = _kinds(actions, "create_event")
    assert any(a.type == "suspicious_item_interaction" for a in evs)
    sus = next(a for a in evs if a.type == "suspicious_item_interaction")
    assert sus.needs_vlm_verification is True
    assert sus.severity == "medium"
    assert sus.rule_id == "suspicious_item_interaction"
    assert sus.why                                 # explanation is populated


def test_no_dwell_no_suspicious():
    """A single shelf observation (no dwell) must NOT fire the suspicious rule."""
    eng = RuleEngine()
    actions = eng.feed({"event_id": "e1", "camera_id": "left_aisle", "track_id": "p9",
                        "ts": 0.0, "zone_types": ["shelf"], "objects": ["person", "item"],
                        "object_near_hand": True})
    assert not any(a.type == "suspicious_item_interaction"
                   for a in _kinds(actions, "create_event"))


def test_has_item_skip_billing_to_exit_creates_possible_loss_incident():
    """Person has item -> never visits billing -> moves to exit => possible_loss incident."""
    eng = RuleEngine()
    seq = [
        {"event_id": "e1", "store_id": "kathirmani_01", "camera_id": "left_aisle",
         "track_id": "p2", "ts": 0.0, "zone_types": ["shelf"], "has_item": True,
         "objects": ["person", "item"]},
        {"event_id": "e2", "camera_id": "entry_exit", "track_id": "p2", "ts": 9.0,
         "zone_types": ["exit"], "zones": ["exit_zone"], "has_item": True,
         "objects": ["person"]},
    ]
    actions = eng.feed_many(seq)
    incidents = _kinds(actions, "create_incident")
    assert any(a.type == "possible_loss" for a in incidents)
    inc = next(a for a in incidents if a.type == "possible_loss")
    assert inc.severity == "high"
    assert inc.rule_id == "possible_billing_bypass"


def test_visiting_billing_suppresses_loss_incident():
    """Same path but through billing first => no possible_loss incident."""
    eng = RuleEngine()
    seq = [
        {"event_id": "e1", "camera_id": "left_aisle", "track_id": "p3", "ts": 0.0,
         "zone_types": ["shelf"], "has_item": True, "objects": ["person", "item"]},
        {"event_id": "e2", "camera_id": "bill_counter", "track_id": "p3", "ts": 5.0,
         "zone_types": ["billing_counter"], "has_item": True, "objects": ["person", "item"]},
        {"event_id": "e3", "camera_id": "entry_exit", "track_id": "p3", "ts": 10.0,
         "zone_types": ["exit"], "has_item": True, "objects": ["person"]},
    ]
    actions = eng.feed_many(seq)
    assert not any(a.type == "possible_loss" for a in _kinds(actions, "create_incident"))


def test_low_camera_health_fires_camera_health_issue():
    eng = RuleEngine()
    actions = eng.feed({"event_id": "e1", "camera_id": "right_aisle",
                        "event_type": "camera_health", "ts": 0.0,
                        "camera_health_score": 0.2})
    evs = _kinds(actions, "create_event")
    assert any(a.type == "camera_health_issue" for a in evs)
    hi = next(a for a in evs if a.type == "camera_health_issue")
    assert hi.severity == "warning"
    assert hi.needs_vlm_verification is False


def test_healthy_camera_does_not_fire():
    eng = RuleEngine()
    actions = eng.feed({"event_id": "e1", "camera_id": "right_aisle",
                        "ts": 0.0, "camera_health_score": 0.95})
    assert not any(a.type == "camera_health_issue" for a in _kinds(actions, "create_event"))


def test_rule_fires_at_most_once_per_track():
    """De-dup: re-seeing the same evidence must not re-emit the action."""
    eng = RuleEngine()
    seq = [
        {"event_id": "e1", "camera_id": "left_aisle", "track_id": "p4", "ts": 0.0,
         "zone_types": ["shelf"], "objects": ["person"]},
        {"event_id": "e2", "camera_id": "left_aisle", "track_id": "p4", "ts": 4.0,
         "zone_types": ["shelf"], "objects": ["person", "item"], "object_near_hand": True},
    ]
    first = eng.feed_many(seq)
    assert any(a.type == "suspicious_item_interaction" for a in first)
    # feed another matching event for the same track -> no new suspicious action
    again = eng.feed({"event_id": "e3", "camera_id": "left_aisle", "track_id": "p4",
                      "ts": 6.0, "zone_types": ["shelf"], "objects": ["person", "item"],
                      "object_near_hand": True})
    assert not any(a.type == "suspicious_item_interaction" for a in again)


def test_action_to_event_dict_is_deterministic_and_shaped():
    eng = RuleEngine()
    actions = eng.feed_many([
        {"event_id": "e1", "camera_id": "left_aisle", "track_id": "p5", "ts": 0.0,
         "zone_types": ["shelf"], "objects": ["person"]},
        {"event_id": "e2", "camera_id": "left_aisle", "track_id": "p5", "ts": 4.0,
         "zone_types": ["shelf"], "objects": ["person", "item"], "object_near_hand": True},
    ])
    a = next(x for x in actions if x.type == "suspicious_item_interaction")
    d1, d2 = action_to_event_dict(a), action_to_event_dict(a)
    assert d1["event_id"] == d2["event_id"]        # deterministic uuid5
    assert d1["source_engine"] == "rule_engine"
    assert d1["type"] == "event.created"
    assert d1["needs_vlm_verification"] is True
    assert "rule" in d1["explanation"]


# --------------------------------------------------------------------------
# track_id provenance: prefer the real track_id; fall back to cam:<id> only when
# the CV worker emits no identity at all (the known Phase-4 follow-up).
# --------------------------------------------------------------------------
def test_real_track_id_buckets_two_subjects_separately():
    """Two distinct track_ids on the same camera must NOT share a context, so
    one subject's evidence can't satisfy a rule on the other's behalf."""
    from services.rule_engine.engine import _track_key
    eng = RuleEngine()
    # subject p_a dwells in shelf; subject p_b appears once with an item near hand.
    eng.feed_many([
        {"event_id": "a1", "camera_id": "left_aisle", "track_id": "p_a", "ts": 0.0,
         "zone_types": ["shelf"], "objects": ["person"]},
        {"event_id": "a2", "camera_id": "left_aisle", "track_id": "p_a", "ts": 4.0,
         "zone_types": ["shelf"], "objects": ["person"]},
    ])
    b_actions = eng.feed({"event_id": "b1", "camera_id": "left_aisle",
                          "track_id": "p_b", "ts": 5.0, "zone_types": ["shelf"],
                          "objects": ["person", "item"], "object_near_hand": True})
    # p_b alone has no dwell -> must not fire; the contexts are keyed by real id.
    assert _track_key({"track_id": "p_a", "camera_id": "left_aisle"}) == "p_a"
    assert not any(a.type == "suspicious_item_interaction" for a in b_actions)


def test_track_ids_list_from_cv_worker_is_used_over_camera_fallback():
    """The CV worker emits `track_ids` (list); the engine must bucket by that
    real id, not the cam:<id> fallback."""
    from services.rule_engine.engine import _track_key
    ev = {"camera_id": "left_aisle", "track_ids": ["42"], "zone_types": ["shelf"]}
    assert _track_key(ev) == "42"
    # only when no identity is present do we fall back to the camera bucket
    assert _track_key({"camera_id": "left_aisle"}) == "cam:left_aisle"


# --------------------------------------------------------------------------
# Golden regression — replay fixtures, assert expected_events / expected_incident.
# --------------------------------------------------------------------------
def _run_golden(path: Path):
    doc = json.loads(path.read_text())
    eng = RuleEngine()
    actions = eng.feed_many(doc["events"])
    events = [a for a in actions if a.kind == "create_event"]
    incidents = [a for a in actions if a.kind == "create_incident"]
    return doc, events, incidents


def test_golden_fixtures_present():
    files = sorted(GOLDEN_DIR.glob("*.json"))
    assert len(files) >= 3, [f.name for f in files]


def test_golden_regression():
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        doc, events, incidents = _run_golden(path)

        got_event_types = {a.type for a in events}
        for exp in doc.get("expected_events", []):
            assert exp["event_type"] in got_event_types, (path.name, exp, got_event_types)
            match = next(a for a in events if a.type == exp["event_type"])
            assert match.rule_id == exp["rule_id"], (path.name, match.rule_id)
            assert match.needs_vlm_verification == exp["needs_vlm_verification"], path.name

        exp_inc = doc.get("expected_incident")
        if exp_inc is None:
            assert incidents == [], (path.name, [a.type for a in incidents])
        else:
            assert any(a.type == exp_inc["incident_type"] for a in incidents), \
                (path.name, [a.type for a in incidents])
            inc = next(a for a in incidents if a.type == exp_inc["incident_type"])
            assert inc.rule_id == exp_inc["rule_id"], path.name
            assert inc.severity == exp_inc["severity"], path.name
