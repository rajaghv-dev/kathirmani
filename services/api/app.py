"""Platform API (FastAPI) — the stable surface workers + the review UI depend on.

Read-mostly over Postgres (the single structured store). Model-registry endpoints
(master plan A2.5) let a profile be activated by API as well as by config. Run:
`make api` (uvicorn services.api.app:app --port 8000).
"""
from __future__ import annotations

import sys
from pathlib import Path

import logging

from fastapi import Depends, FastAPI, Header, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from db import connect  # noqa: E402
from services.security.auth import _configured_key, require_auth  # noqa: E402
from services.security.rbac import require_role  # noqa: E402

if _configured_key() is None:  # secure-by-default primitive; the API opens it in dev
    logging.getLogger("uvicorn.error").warning(
        "PLATFORM_API_KEY unset — API is UNAUTHENTICATED (dev mode). Set it to enforce auth/RBAC.")


def _optional_auth(authorization: str | None = Header(default=None),
                   x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    """Dev-permissive wrapper over the strict require_auth: OPEN when no key is
    configured (local/dev/tests), ENFORCED the moment PLATFORM_API_KEY is set."""
    if _configured_key() is None:
        return "anonymous-dev"
    return require_auth(authorization, x_api_key)


# Global authn (dev-permissive without a key); per-route RBAC below.
app = FastAPI(title="Kathirmani Video-AI Platform API", version="0.2.0",
              dependencies=[Depends(_optional_auth)])


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


@app.post("/api/model-registry/activate/{profile_name}",
          dependencies=[Depends(require_role("admin"))])
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


# ---- Natural-language search (Phase 8) — the same A4.3 pipeline `make search`
# runs, exposed over HTTP so the console / clients can query results uniformly.
@app.get("/search")
def search(q: str, k: int = 10):
    """Run the natural-language video search and return ranked hits. Degrades to an
    empty result set (with an `error` note) when the embedding worker or DB is
    absent, so the endpoint never 500s the console."""
    if not q or not q.strip():
        return {"query": q, "results": [], "error": "empty query"}
    try:
        from ai_workers.embedding_worker.worker import query_run
        return {"query": q, "results": query_run(q, k=k)}
    except Exception as e:
        return {"query": q, "results": [], "error": f"{type(e).__name__}: {e}"}
