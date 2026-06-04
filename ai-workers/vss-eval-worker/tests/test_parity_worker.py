"""parity.py + worker.py — the parity report has all 5 capabilities with a score
or an honest `pending`; the worker CLI entrypoints run hermetically (no DB/GPU).
"""
from __future__ import annotations

import json

import parity
import worker
from parity import CAPABILITIES, build_report


def _has_db() -> bool:
    try:
        from db import connect
        with connect() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def test_report_has_all_five_capabilities():
    rep = build_report(conn=None)                     # no DB -> pending/fake
    caps = {d["capability"] for d in rep["capabilities"]}
    assert caps == set(CAPABILITIES)
    assert len(rep["capabilities"]) == 5


def test_each_dimension_has_score_or_pending():
    rep = build_report(conn=None)
    for d in rep["capabilities"]:
        assert d["status"] in {"measured", "fake", "pending"}
        if d["status"] == "pending":
            assert d["parity_score"] is None
            assert d["note"]                          # honest reason given
        else:
            assert isinstance(d["parity_score"], float)
            assert 0.0 <= d["parity_score"] <= 1.0


def test_lvs_dimension_is_self_tested_fake():
    """lvs is always measurable — it self-tests the §A4.2 wiring on synthetic
    events — so it scores 1.0 and is flagged `fake` (wiring proven, no weights),
    regardless of whether a DB is present."""
    rep = build_report(conn=None)
    lvs_dim = next(d for d in rep["capabilities"] if d["capability"] == "lvs")
    assert lvs_dim["status"] == "fake"
    assert lvs_dim["parity_score"] == 1.0
    assert lvs_dim["value"] is True


def test_db_dimensions_pending_without_db():
    """Force the no-DB path with a connection that raises on use; the DB-backed
    dimensions must then be `pending` with an honest reason (never fabricated)."""
    class _DeadConn:
        def cursor(self):
            raise RuntimeError("no DB in this test")

        def rollback(self):
            pass

    rep = build_report(conn=_DeadConn())
    for cap in ("rt_cv", "rt_embedding", "vios"):
        d = next(x for x in rep["capabilities"] if x["capability"] == cap)
        assert d["status"] == "pending"
        assert d["parity_score"] is None


def test_report_invariants_and_no_vss_dependency():
    rep = build_report(conn=None)
    assert rep["vss_runtime_dependency"] is False     # spec/10/11 invariant
    assert rep["schema"] == "vss_parity_report.v1"
    assert rep["n_total"] == 5
    # overall is the mean of *measurable* dims only (or None if none measurable)
    if rep["n_measurable"]:
        assert 0.0 <= rep["overall_parity_score"] <= 1.0


def test_write_report_persists_json(tmp_path):
    out = tmp_path / "parity_report.json"
    rep = parity.write_report(path=str(out), conn=None)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["schema"] == rep["schema"]
    assert len(loaded["capabilities"]) == 5


def test_worker_summarize_run_with_injected_events():
    req = {"store_id": "s1", "summary_mode": "loss_prevention",
           "time_range": {"start": "2026-06-01T10:00:00+05:30",
                          "end": "2026-06-01T11:00:00+05:30"},
           "cameras": ["left_aisle"]}
    events = [{"event_id": "e1", "event_time": "2026-06-01T10:05:00+05:30",
               "camera_id": "left_aisle", "event_type": "suspicious_item_interaction",
               "severity": "high", "confidence": 0.8, "incident_id": "inc-9"}]
    rep = worker.summarize_run(req, events=events)
    assert rep["summary"]
    assert rep["recommended_review_items"] == ["inc-9"]
    assert rep["nests"]["clip_captions"]


def test_worker_parity_run_hermetic(tmp_path):
    out = tmp_path / "parity_report.json"
    rep = worker.parity_run(path=str(out))
    assert out.exists()
    assert len(rep["capabilities"]) == 5
