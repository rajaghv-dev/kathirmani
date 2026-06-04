"""Unit tests for the pure condition evaluators (rules.py).

Each evaluator operates on a small TrackContext and returns an explainable
Match. No DB / services / GPU — pure functions only.
"""
from __future__ import annotations

from rules import (TrackContext, TrackEvent, camera_health_score_lte,
                   dwell_time_sec_gte, load_rules, moved_to_zone,
                   object_near_person_hand, person_has_item, person_in_zone,
                   visited_billing_zone)


def _ctx(*events: TrackEvent, track_id: str = "p1") -> TrackContext:
    c = TrackContext(track_id=track_id)
    for e in events:
        c.add(e)
    return c


def test_person_in_zone():
    c = _ctx(TrackEvent(ts=0.0, zone_types=["shelf"]))
    assert person_in_zone(c, "shelf").fired
    assert not person_in_zone(c, "billing_counter").fired
    # explanation is human-readable
    assert "shelf" in person_in_zone(c, "shelf").why


def test_dwell_time_gte():
    c = _ctx(TrackEvent(ts=0.0, zone_types=["shelf"]),
             TrackEvent(ts=4.0, zone_types=["shelf"]))
    assert dwell_time_sec_gte(c, 3).fired          # 4s >= 3s
    assert not dwell_time_sec_gte(c, 5).fired      # 4s < 5s


def test_dwell_time_single_obs_is_zero():
    c = _ctx(TrackEvent(ts=2.0, zone_types=["shelf"]))
    assert not dwell_time_sec_gte(c, 1).fired      # only one observation -> 0s dwell


def test_object_near_person_hand():
    near = _ctx(TrackEvent(ts=0.0, object_near_hand=True))
    far = _ctx(TrackEvent(ts=0.0, object_near_hand=False))
    assert object_near_person_hand(near, True).fired
    assert not object_near_person_hand(far, True).fired


def test_person_has_item_flag_or_inferred():
    flagged = _ctx(TrackEvent(ts=0.0, has_item=True))
    inferred = _ctx(TrackEvent(ts=0.0, objects=["person", "bottle"]))
    none = _ctx(TrackEvent(ts=0.0, objects=["person"]))
    assert person_has_item(flagged, True).fired
    assert person_has_item(inferred, True).fired   # non-person object => carrying
    assert not person_has_item(none, True).fired


def test_visited_billing_zone_matches_expected_bool():
    visited = _ctx(TrackEvent(ts=0.0, zone_types=["billing_counter"]))
    not_visited = _ctx(TrackEvent(ts=0.0, zone_types=["shelf"]))
    # rule asks `visited_billing_zone: false` -> fires only when NOT visited
    assert visited_billing_zone(not_visited, False).fired
    assert not visited_billing_zone(visited, False).fired
    assert visited_billing_zone(visited, True).fired


def test_moved_to_zone_uses_latest():
    c = _ctx(TrackEvent(ts=0.0, zone_types=["shelf"]),
             TrackEvent(ts=5.0, zone_types=["exit"]))
    assert moved_to_zone(c, "exit").fired          # latest obs is exit
    assert not moved_to_zone(c, "shelf").fired     # was shelf, but moved on


def test_camera_health_score_lte():
    low = _ctx(TrackEvent(ts=0.0, camera_health_score=0.2))
    ok = _ctx(TrackEvent(ts=0.0, camera_health_score=0.9))
    none = _ctx(TrackEvent(ts=0.0))
    assert camera_health_score_lte(low, 0.5).fired
    assert not camera_health_score_lte(ok, 0.5).fired
    assert not camera_health_score_lte(none, 0.5).fired  # no score reported


def test_load_rules_from_config():
    rules = load_rules()                           # default configs/event_rules.yaml
    ids = {r.id for r in rules}
    assert {"suspicious_item_interaction", "possible_billing_bypass",
            "camera_health_issue"} <= ids
    # no rule references an unknown condition keyword
    for r in rules:
        assert r.unknown_conditions() == [], (r.id, r.unknown_conditions())
