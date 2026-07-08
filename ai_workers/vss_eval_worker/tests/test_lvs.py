"""lvs.py — the §A4.2 staged fold over a synthetic event list (no DB/GPU).

Verifies: output shape, clip->5min->hour nesting, filtering, timeline, open
questions, and recommended review items.
"""
from __future__ import annotations

from ai_workers.vss_eval_worker import lvs
from ai_workers.vss_eval_worker.lvs import filter_events, is_suspicious, summarize


def _req():
    return {
        "store_id": "kathirmani_01", "summary_mode": "loss_prevention",
        "time_range": {"start": "2026-06-01T10:00:00+05:30",
                       "end": "2026-06-01T11:00:00+05:30"},
        "cameras": ["left_aisle", "billing"], "include_low_confidence": False,
    }


def _events():
    """Events spread across multiple 10s clips, 5-min windows, and an hour so the
    nesting has >1 bucket at each level."""
    return [
        # 10:00:05 — clip 0, 5-min 0, hour 0  (suspicious, high)
        {"event_id": "e1", "event_time": "2026-06-01T10:00:05+05:30",
         "camera_id": "left_aisle", "event_type": "suspicious_item_interaction",
         "severity": "high", "confidence": 0.9, "needs_vlm_verification": True,
         "evidence_segment_id": "seg-1", "incident_id": "inc-1",
         "metadata_json": {"zones": ["shelf_zone_A"], "objects": ["person", "item"]},
         "vlm": {"verdict": "suspicious", "explanation": "person concealed an item",
                 "missing_evidence": ["a clear view of the bag"]}},
        # 10:00:25 — clip 2, same 5-min/hour (low conf, benign -> filtered out)
        {"event_id": "e2", "event_time": "2026-06-01T10:00:25+05:30",
         "camera_id": "left_aisle", "event_type": "person_present",
         "severity": "info", "confidence": 0.2},
        # 10:07:10 — clip in 5-min bucket 1, hour 0  (suspicious on billing)
        {"event_id": "e3", "event_time": "2026-06-01T10:07:10+05:30",
         "camera_id": "billing", "event_type": "possible_billing_bypass",
         "severity": "critical", "confidence": 0.7, "incident_id": "inc-2",
         "metadata_json": {"zones": ["billing_counter"]}},
    ]


def test_output_has_a4_2_shape():
    rep = summarize(_req(), events=_events())
    for key in ("summary", "timeline", "open_questions", "recommended_review_items"):
        assert key in rep
    assert isinstance(rep["summary"], str) and rep["summary"]
    assert isinstance(rep["timeline"], list)
    assert isinstance(rep["open_questions"], list)
    assert isinstance(rep["recommended_review_items"], list)


def test_nesting_clip_5min_hour():
    rep = summarize(_req(), events=_events())
    nests = rep["nests"]
    assert nests["clip_captions"], "expected per-clip captions"
    assert nests["five_minute_summaries"], "expected per-5-min summaries"
    assert nests["hour_summaries"], "expected per-hour summary"
    # two suspicious events land in two different 5-min buckets within one hour
    assert len(nests["five_minute_summaries"]) >= 2
    assert len(nests["hour_summaries"]) == 1
    # each 5-min summary references the clips it folded
    for w in nests["five_minute_summaries"]:
        assert "clip_indices" in w


def test_low_confidence_benign_filtered_but_suspicious_kept():
    kept = filter_events(_events(), _req())
    ids = {e["event_id"] for e in kept}
    assert "e2" not in ids                            # benign low-conf dropped
    assert {"e1", "e3"} <= ids                        # suspicious kept


def test_include_low_confidence_keeps_everything():
    req = _req()
    req["include_low_confidence"] = True
    kept = filter_events(_events(), req)
    assert {e["event_id"] for e in kept} == {"e1", "e2", "e3"}


def test_timeline_only_suspicious_with_evidence():
    rep = summarize(_req(), events=_events())
    times = [t["time"] for t in rep["timeline"]]
    assert "10:00:05" in times and "10:07:10" in times
    assert "10:00:25" not in times                    # benign not in timeline
    e1 = next(t for t in rep["timeline"] if t["time"] == "10:00:05")
    assert e1["camera"] == "left_aisle"
    assert e1["evidence_segment_id"] == "seg-1"


def test_review_items_and_open_questions():
    rep = summarize(_req(), events=_events())
    # explicit incident ids surface as review items
    assert "inc-1" in rep["recommended_review_items"]
    assert "inc-2" in rep["recommended_review_items"]
    # the VLM missing-evidence becomes an open question
    assert any("bag" in q for q in rep["open_questions"])
    # cross-camera follow-up is suggested (left_aisle + billing both suspicious)
    assert any("billing" in q or "left_aisle" in q for q in rep["open_questions"])


def test_empty_feed_produces_valid_empty_report():
    rep = summarize(_req(), events=[])
    assert rep["timeline"] == []
    assert rep["recommended_review_items"] == []
    assert rep["summary"]                             # still a (no-activity) summary
    assert rep["stats"]["events_kept"] == 0


def test_camera_filter_applies():
    req = _req()
    req["cameras"] = ["billing"]
    kept = filter_events(_events(), req)
    assert {e["event_id"] for e in kept} == {"e3"}


def test_is_suspicious_helper():
    assert is_suspicious({"severity": "high"})
    assert is_suspicious({"needs_vlm_verification": True})
    assert not is_suspicious({"severity": "info"})


def test_caption_line_format():
    line = lvs.event_caption_line(_events()[0])
    assert "10:00:05" in line and "left_aisle" in line
    assert "concealed an item" in line                # VLM explanation surfaced
