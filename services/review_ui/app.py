"""review_ui (master plan Phase 7) — human approve/reject of loss hypotheses.

A small, **separate** FastAPI app (port 8010) that lets a reviewer look at an
incident's evidence package (events + timeline + VLM verdict + evidence clip
path) and record a verdict. It is **review/verdict only** — explicitly NOT an
annotation tool (bounding-boxes/labels live in CVAT + Label Studio, per the
Kathirmani viz-vs-annotation split). The goal: a human reviews a loss hypothesis
in <60s.

Chain-of-custody (master plan Security, Phase 7): every verdict appends an
immutable **audit record** — `{reviewer, decided_at, decision, model_version,
notes}` — embedded in the incident's stored evidence package
(`incidents.summary`, which the evidence_builder writes as JSON). We do NOT edit
`db/`, so we co-locate the audit log inside that JSON column rather than a new
table; a dedicated `audit_log` table is the recommended follow-up migration (see
integration notes / NEEDS_AUDIT_LOG_TABLE below).

Endpoints (JSON is the requirement; an HTML index is a plus):
  GET  /health                      -> {status, db}
  GET  /                            -> minimal HTML incident index (open ones)
  GET  /incidents                   -> open incidents (JSON)
  GET  /incidents/{id}              -> events + timeline + vlm verdict + evidence_path
  POST /incidents/{id}/review       -> {decision: approve|reject, reviewer, notes}
                                       updates status (confirmed|dismissed) + audits

DB access goes through a single injectable `get_conn()` so the suite can run the
review flow against a fake DB (TestClient) with no Postgres. With Postgres down
the read endpoints return 503 and the app still boots (same guarded posture as
the workers).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator


# Recommended follow-up migration (we must not edit db/): a first-class audit
# table. Until it lands the audit record is embedded in incidents.summary JSON.
NEEDS_AUDIT_LOG_TABLE = (
    "CREATE TABLE audit_log (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
    "incident_id UUID REFERENCES incidents(id), reviewer TEXT NOT NULL, "
    "decision TEXT NOT NULL, model_version TEXT, notes TEXT, "
    "created_at TIMESTAMPTZ DEFAULT now());"
)

# decision -> resulting incident status (master plan: confirmed | dismissed).
_DECISION_STATUS = {"approve": "confirmed", "reject": "dismissed"}

app = FastAPI(title="Kathirmani Review UI", version="0.1.0")


# --------------------------------------------------------------------------
# DB indirection — one seam so tests can inject a fake connection.
# --------------------------------------------------------------------------
def get_conn():
    """Return a live psycopg connection, or None if Postgres is unavailable.

    Overridden in tests via `app.dependency_overrides`-style monkeypatch (we
    patch this module-level function) to return a fake connection.
    """
    try:
        from db import connect
        return connect()
    except Exception:
        return None


def _rows(conn, sql: str, params: tuple = ()) -> list[dict]:
    from psycopg.rows import dict_row
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Review request body.
# --------------------------------------------------------------------------
class ReviewIn(BaseModel):
    decision: str                       # approve | reject
    reviewer: str
    notes: str = ""

    @field_validator("decision")
    @classmethod
    def _valid_decision(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in _DECISION_STATUS:
            raise ValueError("decision must be 'approve' or 'reject'")
        return v

    @field_validator("reviewer")
    @classmethod
    def _nonempty_reviewer(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError("reviewer is required (chain-of-custody)")
        return v.strip()


# --------------------------------------------------------------------------
# Endpoints.
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    conn = get_conn()
    db_ok = conn is not None
    if conn is not None:
        try:
            _rows(conn, "SELECT 1")
        except Exception:
            db_ok = False
        finally:
            _close(conn)
    return {"status": "ok", "db": db_ok}


@app.get("/incidents")
def list_incidents(limit: int = 50):
    """Open incidents awaiting review (status='open')."""
    conn = _require_conn()
    try:
        return _rows(conn,
                     "SELECT id, store_id, title, incident_type, severity, status, "
                     "start_time, evidence_path FROM incidents WHERE status='open' "
                     "ORDER BY created_at DESC LIMIT %s", (limit,))
    finally:
        _close(conn)


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str):
    """Full review view: incident + events + timeline + VLM verdict + evidence path.

    The timeline + headline verdict come from the evidence package the
    evidence_builder stored in `incidents.summary` (JSON). We still fetch the
    raw `events` + `vlm_observations` so the reviewer sees source rows even if no
    package was built yet.
    """
    conn = _require_conn()
    try:
        rows = _rows(conn,
                     "SELECT id, store_id, title, incident_type, severity, status, "
                     "start_time, end_time, summary, evidence_path "
                     "FROM incidents WHERE id=%s", (incident_id,))
        if not rows:
            raise HTTPException(404, f"unknown incident {incident_id}")
        incident = rows[0]

        package, audit = _parse_summary(incident.get("summary"))

        event_ids = [r["event_id"] for r in _rows(
            conn, "SELECT event_id FROM incident_events WHERE incident_id=%s",
            (incident_id,))]
        events, observations = [], []
        if event_ids:
            events = _rows(conn,
                           "SELECT id, camera_id, event_type, severity, confidence, "
                           "event_time FROM events WHERE id = ANY(%s) "
                           "ORDER BY event_time", (event_ids,))
            observations = _rows(conn,
                                 "SELECT id, event_id, model_name, prompt_version, "
                                 "parsed_json, parse_success, created_at "
                                 "FROM vlm_observations WHERE event_id = ANY(%s) "
                                 "ORDER BY created_at", (event_ids,))
        return {
            "incident": {k: incident[k] for k in incident if k != "summary"},
            "evidence_path": incident.get("evidence_path"),
            "events": events,
            "observations": observations,
            "timeline": (package or {}).get("timeline", []),
            "headline_verdict": (package or {}).get("headline_verdict"),
            "chain_of_custody": (package or {}).get("chain_of_custody"),
            "audit_log": audit,
        }
    finally:
        _close(conn)


@app.post("/incidents/{incident_id}/review")
def review(incident_id: str, body: ReviewIn):
    """Record a human verdict: update status + append an immutable audit record.

    Audit record (chain-of-custody): reviewer, decided_at (ts), decision,
    model_version (the VLM model that produced the hypothesis, pulled from the
    evidence package / latest observation), notes.
    """
    conn = _require_conn()
    try:
        rows = _rows(conn, "SELECT id, status, summary FROM incidents WHERE id=%s",
                     (incident_id,))
        if not rows:
            raise HTTPException(404, f"unknown incident {incident_id}")
        incident = rows[0]
        new_status = _DECISION_STATUS[body.decision]

        package, audit = _parse_summary(incident.get("summary"))
        model_version = _model_version(package)

        record = {
            "reviewer": body.reviewer,
            "decision": body.decision,
            "resulting_status": new_status,
            "model_version": model_version,
            "notes": body.notes,
            "decided_at": _now(),
        }
        audit.append(record)                            # append-only (immutable history)

        # Persist the audit log back into the package JSON in `incidents.summary`.
        package = package or {}
        package["audit_log"] = audit
        summary_json = json.dumps(package)

        with conn.cursor() as cur:
            cur.execute("UPDATE incidents SET status=%s, summary=%s, updated_at=now() "
                        "WHERE id=%s", (new_status, summary_json, incident_id))
        conn.commit()
        return {"incident_id": incident_id, "status": new_status, "audit": record,
                "audit_count": len(audit)}
    finally:
        _close(conn)


# --------------------------------------------------------------------------
# Minimal HTML index (a plus; JSON endpoints are the requirement).
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    conn = get_conn()
    if conn is None:
        return HTMLResponse("<h1>Review UI</h1><p>Database unavailable.</p>", status_code=503)
    try:
        incidents = _rows(conn,
                          "SELECT id, title, incident_type, severity, start_time "
                          "FROM incidents WHERE status='open' ORDER BY created_at DESC "
                          "LIMIT 100")
    except Exception:
        return HTMLResponse("<h1>Review UI</h1><p>Database error.</p>", status_code=503)
    finally:
        _close(conn)
    rows = "".join(
        f"<tr><td><a href='/incidents/{i['id']}'>{i['id']}</a></td>"
        f"<td>{_esc(i.get('title'))}</td><td>{_esc(i.get('incident_type'))}</td>"
        f"<td>{_esc(i.get('severity'))}</td><td>{_esc(i.get('start_time'))}</td></tr>"
        for i in incidents)
    return HTMLResponse(
        "<html><head><title>Kathirmani Review</title></head><body>"
        "<h1>Open incidents for review</h1>"
        "<p>Review/verdict only — annotation lives in CVAT + Label Studio.</p>"
        "<table border='1' cellpadding='4'><tr><th>ID</th><th>Title</th>"
        "<th>Type</th><th>Severity</th><th>Start</th></tr>"
        f"{rows}</table></body></html>")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _require_conn():
    conn = get_conn()
    if conn is None:
        raise HTTPException(503, "database unavailable")
    return conn


def _close(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def _parse_summary(summary: Any) -> tuple[dict | None, list[dict]]:
    """The evidence package lives in `incidents.summary` as JSON. Parse it and
    pull any existing audit_log. Tolerates a plain-text summary (returns no
    package, empty audit) so a pre-Phase-7 incident still reviews fine."""
    if not summary:
        return None, []
    if isinstance(summary, dict):
        package = summary
    else:
        try:
            package = json.loads(summary)
        except (ValueError, TypeError):
            return None, []
    if not isinstance(package, dict):
        return None, []
    audit = package.get("audit_log") or []
    if not isinstance(audit, list):
        audit = []
    return package, list(audit)


def _model_version(package: dict | None) -> str | None:
    """The VLM model_version behind the hypothesis (chain-of-custody)."""
    if not package:
        return None
    hv = package.get("headline_verdict") or {}
    if hv.get("model_name"):
        return hv["model_name"]
    for t in reversed(package.get("timeline") or []):
        if t.get("kind") == "vlm_observation" and t.get("model_name"):
            return t["model_name"]
    return None


def _esc(v: Any) -> str:
    import html
    return html.escape("" if v is None else str(v))


if __name__ == "__main__":                              # python -m app (dev)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
