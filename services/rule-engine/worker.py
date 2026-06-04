"""rule-engine worker loop (master plan Phase 5).

Chains *after* the CV worker: the CV worker publishes low-level §8.3 CommonEvents
onto the queue as `event.created`; this worker consumes that stream, feeds each
event to a deterministic `RuleEngine`, and persists what fires:

  * `create_event` actions  -> a higher-level row in `events` (+ republished as
    `event.created` so Phase 6's VLM worker can pick suspicious ones up), and
  * `create_incident` actions -> a row in `incidents` + links its contributing
    low-level events in `incident_events`.

Everything that touches a service is guarded: with Postgres down, `make_queue`
falls back to `FileQueue` and all DB writes no-op, so `run()` degrades to a
hermetic "consume from file / write nothing" loop instead of crashing — same
posture as the CV worker.

Entrypoint: `run(once=False, limit=N)`.  `python -m` friendly via __main__.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ingestion.bus import make_queue                 # noqa: E402

try:                                                 # package vs direct/script import
    from .engine import EngineAction, RuleEngine, action_to_event_dict
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine import EngineAction, RuleEngine, action_to_event_dict

# We consume the events the CV worker (and we ourselves) publish.
STREAM = "event.created"
EVENTS_DIR = _REPO_ROOT / "data" / "queue"           # FileQueue out_dir (hermetic default)


# --------------------------------------------------------------------------
# DB helpers — all best-effort; no-op cleanly when Postgres is unavailable.
# --------------------------------------------------------------------------
def _db_conn():
    try:
        from db import connect
        return connect()
    except Exception:
        return None


def _safe_commit(conn) -> None:
    """Commit; on any error roll back so the connection stays usable. All DB
    writes are best-effort — a rejected row must never take down the loop."""
    try:
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def _write_event(conn, ev: dict) -> str | None:
    """Insert a rule-engine CommonEvent into `events`; returns event_id or None."""
    if conn is None:
        return None
    meta = {k: ev[k] for k in ("track_ids", "zones", "needs_vlm_verification",
                               "rule_id", "explanation", "source_event_ids")
            if k in ev}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (id, store_id, camera_id, event_type, severity, "
                "confidence, source_engine, metadata_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                (ev["event_id"], ev["store_id"], ev.get("camera_id"),
                 ev["event_type"], ev.get("severity", "info"),
                 ev.get("confidence", 0.0), ev.get("source_engine", "rule_engine"),
                 json.dumps(meta)))
        conn.commit()
        return ev["event_id"]
    except Exception:
        _safe_commit(conn)
        return None


def _write_incident(conn, action: EngineAction) -> str | None:
    """Insert an incident + link its contributing events; returns incident_id."""
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO incidents (store_id, title, incident_type, severity, "
                "summary) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (action.store_id, f"{action.type} ({action.rule_id})",
                 action.type, action.severity, action.explanation()))
            incident_id = cur.fetchone()[0]
            # link the low-level events that contributed (best-effort; events the
            # CV worker wrote share these ids).
            for eid in action.source_event_ids:
                try:
                    cur.execute(
                        "INSERT INTO incident_events (incident_id, event_id) "
                        "VALUES (%s, %s) ON CONFLICT DO NOTHING", (incident_id, eid))
                except Exception:
                    pass
        conn.commit()
        return str(incident_id)
    except Exception:
        _safe_commit(conn)
        return None


# --------------------------------------------------------------------------
# File fallback consumer — read events from the FileQueue jsonl.
# --------------------------------------------------------------------------
def _file_jobs(out_dir: Path, limit: int) -> list[dict]:
    f = out_dir / f"{STREAM.replace('.', '_')}.jsonl"
    if not f.exists():
        return []
    jobs: list[dict] = []
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(jobs) >= limit:
                break
    return jobs


# --------------------------------------------------------------------------
# Core: feed one event to the engine + persist whatever fires.
# --------------------------------------------------------------------------
def process_event(ev: dict, engine: RuleEngine, queue, conn) -> dict:
    """Feed one low-level event; persist + republish any actions it triggers.
    Returns a small summary dict for logging/tests."""
    actions = engine.feed(ev)
    n_event, n_incident = 0, 0
    for a in actions:
        if a.kind == "create_incident":
            _write_incident(conn, a)             # no-op without DB
            n_incident += 1
        else:
            out = action_to_event_dict(a)
            _write_event(conn, out)              # no-op without DB
            try:
                queue.publish(out)               # republish for the VLM worker
            except Exception:
                pass
            n_event += 1
    return {"in_event_type": ev.get("event_type"), "actions": len(actions),
            "events_created": n_event, "incidents_created": n_incident,
            "rules_fired": [a.rule_id for a in actions]}


def run(once: bool = False, limit: int = 8, backend: str | None = None,
        sleep_sec: float = 2.0) -> list[dict]:
    """Worker entrypoint.

    once=True  -> drain up to `limit` events then return.
    once=False -> loop forever, polling `limit` at a time.
    Returns the list of per-event summaries (useful in tests)."""
    queue = make_queue(backend, EVENTS_DIR)
    conn = _db_conn()
    engine = RuleEngine()
    is_pg = getattr(queue, "name", "") == "pg"
    summaries: list[dict] = []
    try:
        while True:
            if is_pg:
                claimed = queue.consume(STREAM, limit=limit)    # [(job_id, payload)]
                jobs = [(jid, p) for jid, p in claimed]
            else:                                # file fallback: no ack/claim
                jobs = [(None, p) for p in _file_jobs(EVENTS_DIR, limit)]
            for job_id, ev in jobs:
                try:
                    s = process_event(ev, engine, queue, conn)
                    summaries.append(s)
                    if is_pg and job_id is not None:
                        queue.ack(job_id)
                except Exception as e:           # one bad event must not kill the loop
                    print(f"[rule-engine] event failed: {e}")
                    if is_pg and job_id is not None:
                        queue.fail(job_id)
            if once or not jobs:
                if once:
                    break
                time.sleep(sleep_sec)            # idle backoff
        return summaries
    finally:
        try:
            queue.close()
        except Exception:
            pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":                           # python -m worker / direct
    import argparse
    ap = argparse.ArgumentParser(description="rule-engine worker")
    ap.add_argument("--once", action="store_true", help="drain then exit")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--backend", default=None, help="pg|file|redis (default: env)")
    args = ap.parse_args()
    res = run(once=args.once, limit=args.limit, backend=args.backend)
    fired = sum(s["actions"] for s in res)
    print(f"[rule-engine] processed {len(res)} event(s), fired {fired} action(s): {res}")
