"""review_ui tests — TestClient over a FAKE psycopg connection (no Postgres).

The fake mimics enough of psycopg's `conn.cursor(row_factory=dict_row)` surface
(execute/fetchall/fetchone/rowcount, commit) to drive the read + review flow,
and lets us assert the verdict mutation + immutable audit append. /health and
the 503-when-no-DB paths are also covered.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from services.review_ui import app as review_app

client = TestClient(review_app.app)


# --------------------------------------------------------------------------
# A tiny in-memory fake of the bits of psycopg the app uses.
# --------------------------------------------------------------------------
class FakeCursor:
    def __init__(self, store):
        self.store = store
        self._result: list[dict] = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        if s.startswith("SELECT 1"):
            self._result = [{"?column?": 1}]
        elif "FROM incidents WHERE status='open'" in s and "title" in s and "evidence_path" in s:
            self._result = [i for i in self.store["incidents"].values() if i["status"] == "open"]
        elif "FROM incidents WHERE status='open'" in s:                 # HTML index query
            self._result = [i for i in self.store["incidents"].values() if i["status"] == "open"]
        elif "FROM incidents WHERE id=" in s:
            inc = self.store["incidents"].get(params[0])
            self._result = [dict(inc)] if inc else []
        elif "FROM incident_events WHERE incident_id=" in s:
            links = self.store["incident_events"].get(params[0], [])
            self._result = [{"event_id": e} for e in links]
        elif "FROM events WHERE id = ANY(" in s:
            ids = set(params[0])
            self._result = [e for e in self.store["events"] if e["id"] in ids]
        elif "FROM vlm_observations WHERE event_id = ANY(" in s:
            ids = set(params[0])
            self._result = [o for o in self.store["observations"] if o["event_id"] in ids]
        elif s.startswith("UPDATE incidents SET status="):
            status, summary, iid = params
            inc = self.store["incidents"].get(iid)
            if inc:
                inc["status"] = status
                inc["summary"] = summary
                self.rowcount = 1
            self._result = []
        else:
            self._result = []

    def fetchall(self):
        return list(self._result)

    def fetchone(self):
        return self._result[0] if self._result else None


class FakeConn:
    def __init__(self, store):
        self.store = store
        self.committed = False
        self.closed = False

    def cursor(self, row_factory=None):
        return FakeCursor(self.store)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


@pytest.fixture
def store():
    return {
        "incidents": {
            "inc-1": {
                "id": "inc-1", "store_id": "kathirmani_01",
                "title": "possible_loss", "incident_type": "possible_loss",
                "severity": "warn", "status": "open",
                "start_time": "2026-06-03T10:00:00+00:00", "end_time": None,
                "evidence_path": "/data/evidence/inc-1.mp4",
                "summary": json.dumps({
                    "incident_id": "inc-1",
                    "timeline": [
                        {"kind": "event", "ts": "2026-06-03T10:00:05+00:00",
                         "event_type": "suspicious_item_interaction"},
                        {"kind": "vlm_observation", "ts": "2026-06-03T10:00:08+00:00",
                         "model_name": "nemotron-vl", "verdict": "suspicious",
                         "confidence": 0.83},
                    ],
                    "headline_verdict": {"kind": "vlm_observation",
                                         "model_name": "nemotron-vl",
                                         "verdict": "suspicious", "confidence": 0.83},
                    "chain_of_custody": {"evidence_sha256": "deadbeef"},
                }),
            }
        },
        "incident_events": {"inc-1": ["e1"]},
        "events": [{"id": "e1", "camera_id": "shelf_a",
                    "event_type": "suspicious_item_interaction", "severity": "warn",
                    "confidence": 0.6, "event_time": "2026-06-03T10:00:05+00:00"}],
        "observations": [{"id": "o1", "event_id": "e1", "model_name": "nemotron-vl",
                          "prompt_version": "retail_loss_v1",
                          "parsed_json": {"verdict": "suspicious", "confidence": 0.83},
                          "parse_success": True,
                          "created_at": "2026-06-03T10:00:08+00:00"}],
    }


@pytest.fixture
def with_db(monkeypatch, store):
    conn = FakeConn(store)
    monkeypatch.setattr(review_app, "get_conn", lambda: conn)
    return conn


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------
def test_health_with_fake_db(with_db):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "db": True}


def test_health_no_db(monkeypatch):
    monkeypatch.setattr(review_app, "get_conn", lambda: None)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["db"] is False


def test_endpoints_503_when_no_db(monkeypatch):
    monkeypatch.setattr(review_app, "get_conn", lambda: None)
    assert client.get("/incidents").status_code == 503
    assert client.get("/incidents/inc-1").status_code == 503
    assert client.post("/incidents/inc-1/review",
                       json={"decision": "approve", "reviewer": "amy"}).status_code == 503


# --------------------------------------------------------------------------
# read endpoints
# --------------------------------------------------------------------------
def test_list_open_incidents(with_db):
    r = client.get("/incidents")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1 and body[0]["id"] == "inc-1"


def test_get_incident_returns_timeline_and_verdict(with_db):
    r = client.get("/incidents/inc-1")
    assert r.status_code == 200
    body = r.json()
    assert body["evidence_path"] == "/data/evidence/inc-1.mp4"
    assert len(body["timeline"]) == 2
    assert body["headline_verdict"]["verdict"] == "suspicious"
    assert body["chain_of_custody"]["evidence_sha256"] == "deadbeef"
    assert len(body["events"]) == 1 and len(body["observations"]) == 1
    assert body["audit_log"] == []                       # not reviewed yet


def test_get_unknown_incident_404(with_db):
    assert client.get("/incidents/nope").status_code == 404


# --------------------------------------------------------------------------
# review flow + audit
# --------------------------------------------------------------------------
def test_review_approve_sets_confirmed_and_audits(with_db, store):
    r = client.post("/incidents/inc-1/review",
                    json={"decision": "approve", "reviewer": "amy", "notes": "clear theft"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "confirmed"
    audit = body["audit"]
    # chain-of-custody fields present
    assert audit["reviewer"] == "amy"
    assert audit["decision"] == "approve"
    assert audit["model_version"] == "nemotron-vl"      # pulled from the package
    assert audit["notes"] == "clear theft"
    assert "decided_at" in audit
    # persisted: incident status mutated + audit stored in summary JSON
    inc = store["incidents"]["inc-1"]
    assert inc["status"] == "confirmed"
    pkg = json.loads(inc["summary"])
    assert pkg["audit_log"][0]["reviewer"] == "amy"


def test_review_reject_sets_dismissed(with_db, store):
    r = client.post("/incidents/inc-1/review",
                    json={"decision": "reject", "reviewer": "bob"})
    assert r.status_code == 200 and r.json()["status"] == "dismissed"
    assert store["incidents"]["inc-1"]["status"] == "dismissed"


def test_audit_log_is_append_only(with_db, store):
    client.post("/incidents/inc-1/review", json={"decision": "approve", "reviewer": "amy"})
    r2 = client.post("/incidents/inc-1/review", json={"decision": "reject", "reviewer": "carol"})
    assert r2.json()["audit_count"] == 2                 # history is kept, not overwritten
    pkg = json.loads(store["incidents"]["inc-1"]["summary"])
    assert [a["reviewer"] for a in pkg["audit_log"]] == ["amy", "carol"]


def test_review_validates_decision_and_reviewer(with_db):
    assert client.post("/incidents/inc-1/review",
                       json={"decision": "maybe", "reviewer": "amy"}).status_code == 422
    assert client.post("/incidents/inc-1/review",
                       json={"decision": "approve", "reviewer": "  "}).status_code == 422


def test_review_unknown_incident_404(with_db):
    assert client.post("/incidents/nope/review",
                       json={"decision": "approve", "reviewer": "amy"}).status_code == 404


# --------------------------------------------------------------------------
# HTML index (a plus)
# --------------------------------------------------------------------------
def test_html_index_lists_open_incidents(with_db):
    r = client.get("/")
    assert r.status_code == 200
    assert "Open incidents" in r.text and "inc-1" in r.text
    assert "CVAT" in r.text                               # review-only disclaimer
