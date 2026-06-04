"""Platform API (FastAPI) — the stable surface workers + the review UI depend on.

Read-mostly over Postgres (the single structured store). Model-registry endpoints
(master plan A2.5) let a profile be activated by API as well as by config. Run:
`make api` (uvicorn services.api.app:app --port 8000).
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from db import connect  # noqa: E402

app = FastAPI(title="Kathirmani Video-AI Platform API", version="0.2.0")


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    from psycopg.rows import dict_row
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@app.get("/health")
def health():
    db_ok = True
    try:
        _rows("SELECT 1")
    except Exception:
        db_ok = False
    return {"status": "ok", "db": db_ok}


@app.get("/cameras")
def cameras():
    return _rows("SELECT id, store_id, name, role, enabled FROM cameras ORDER BY id")


@app.get("/segments")
def segments(camera_id: str | None = None, limit: int = 50):
    if camera_id:
        return _rows("SELECT id, camera_id, start_time, end_time, duration_sec, storage_path, "
                     "checksum, health_json FROM video_segments WHERE camera_id=%s "
                     "ORDER BY start_time DESC LIMIT %s", (camera_id, limit))
    return _rows("SELECT id, camera_id, start_time, end_time, duration_sec, storage_path, "
                 "checksum FROM video_segments ORDER BY start_time DESC LIMIT %s", (limit,))


@app.get("/events")
def events(limit: int = 50):
    return _rows("SELECT id, store_id, camera_id, event_type, severity, confidence, "
                 "event_time FROM events ORDER BY event_time DESC LIMIT %s", (limit,))


@app.get("/incidents")
def incidents(limit: int = 50):
    return _rows("SELECT id, store_id, title, incident_type, severity, status, start_time "
                 "FROM incidents ORDER BY created_at DESC LIMIT %s", (limit,))


# ---- Model registry (master plan A2.5) -------------------------------------
@app.get("/api/model-registry")
def model_registry():
    return _rows("SELECT model_id, model_family, vendor, supported_tasks, supported_runtimes, "
                 "revision, source_url FROM model_registry ORDER BY model_id")


@app.get("/api/model-registry/active")
def active_profile():
    rows = _rows("SELECT profile_name, description, config_json FROM model_profiles WHERE active")
    if not rows:
        raise HTTPException(404, "no active profile")
    return rows[0]


@app.post("/api/model-registry/activate/{profile_name}")
def activate(profile_name: str):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM model_profiles WHERE profile_name=%s", (profile_name,))
        if not cur.fetchone():
            raise HTTPException(404, f"unknown profile {profile_name}")
        cur.execute("UPDATE model_profiles SET active = (profile_name=%s)", (profile_name,))
        conn.commit()
    return {"active_profile": profile_name}


@app.get("/api/model-runs")
def model_runs(limit: int = 50):
    return _rows("SELECT model_profile_name, model_id, task, runtime, latency_ms, ttft_ms, "
                 "parse_success, created_at FROM model_runs ORDER BY created_at DESC LIMIT %s", (limit,))
