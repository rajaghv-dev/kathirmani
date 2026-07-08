"""Worker loop tests — hermetic (no DB, no real queue service).

Force the file backend and a temp queue dir; assert run() consumes events,
the engine fires, and create_event actions get republished to the queue.
"""
from __future__ import annotations

import json

from services.rule_engine import worker
from services.rule_engine.engine import RuleEngine


def _seed_file_queue(tmp_path, events):
    f = tmp_path / "event_created.jsonl"          # STREAM 'event.created' -> file name
    f.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return f


def test_run_once_file_backend_fires_and_republishes(tmp_path, monkeypatch):
    # point the worker's file queue at a temp dir
    monkeypatch.setattr(worker, "EVENTS_DIR", tmp_path)
    monkeypatch.setenv("INGEST_QUEUE", "file")
    _seed_file_queue(tmp_path, [
        {"event_id": "e1", "store_id": "kathirmani_01", "camera_id": "left_aisle",
         "track_id": "p1", "ts": 0.0, "zone_types": ["shelf"], "objects": ["person"]},
        {"event_id": "e2", "camera_id": "left_aisle", "track_id": "p1", "ts": 4.0,
         "zone_types": ["shelf"], "objects": ["person", "item"], "object_near_hand": True},
    ])
    summaries = worker.run(once=True, limit=10, backend="file")
    total_events = sum(s["events_created"] for s in summaries)
    assert total_events >= 1
    fired = [r for s in summaries for r in s["rules_fired"]]
    assert "suspicious_item_interaction" in fired

    # the create_event action was republished onto the file queue
    out = tmp_path / "event_created.jsonl"
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert any(l.get("source_engine") == "rule_engine"
               and l.get("event_type") == "suspicious_item_interaction" for l in lines)


def test_process_event_no_db_no_crash(tmp_path, monkeypatch):
    """process_event must no-op cleanly when conn is None (Postgres down)."""
    monkeypatch.setattr(worker, "EVENTS_DIR", tmp_path)
    eng = RuleEngine()
    queue = worker.make_queue("file", tmp_path)
    try:
        s1 = worker.process_event(
            {"event_id": "e1", "camera_id": "left_aisle", "track_id": "p2",
             "ts": 0.0, "zone_types": ["shelf"], "objects": ["person"]},
            eng, queue, conn=None)
        s2 = worker.process_event(
            {"event_id": "e2", "camera_id": "left_aisle", "track_id": "p2", "ts": 4.0,
             "zone_types": ["shelf"], "objects": ["person", "item"],
             "object_near_hand": True}, eng, queue, conn=None)
    finally:
        queue.close()
    assert s2["events_created"] >= 1              # fired without a DB


def test_camera_health_incident_path(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "EVENTS_DIR", tmp_path)
    _seed_file_queue(tmp_path, [
        {"event_id": "h1", "camera_id": "right_aisle", "event_type": "camera_health",
         "ts": 0.0, "camera_health_score": 0.2},
    ])
    summaries = worker.run(once=True, limit=10, backend="file")
    fired = [r for s in summaries for r in s["rules_fired"]]
    assert "camera_health_issue" in fired
