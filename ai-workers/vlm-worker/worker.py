"""vlm-worker loop (master plan Phase 6 / spec/11 "don't use the VLM for everything").

Chains *after* the rule-engine: that worker republishes higher-level §8.3
CommonEvents onto `event.created` with `needs_vlm_verification=True`. This worker
consumes that stream, **skips** events not flagged for verification (the main
compute saving — only suspicious windows hit the VLM), and for the flagged ones
loads the active profile's `vlm_clip_reasoning` plugin (host.py) and runs it.

Per verified event it:
  * writes a `vlm_observations` row (raw_output, parsed_json, parse_success,
    ttft_ms, output_tokens, tokens_per_sec, latency_ms),
  * updates the source event's `confidence` from the VLM verdict confidence,
  * writes a `model_runs` row,
  * increments the `model_*` metrics (ttft/tokens/json_parse/verdict — inside the
    plugin), and publishes a `vlm.verified` event for downstream consumers.

Everything that touches a service is guarded: with Postgres down, `make_queue`
falls back to `FileQueue` and DB writes no-op, so `run()` degrades to a hermetic
"consume from file / write nothing" loop — same posture as the cv/rule workers.

Entrypoint: `run(once=False, limit=N)`.  `python -m` friendly via __main__.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ingestion.bus import make_queue                          # noqa: E402

try:                                                          # package vs direct import
    from .host import load_plugin
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from host import load_plugin

STREAM = "event.created"                                      # the rule-engine output stream
EVENTS_DIR = _REPO_ROOT / "data" / "queue"                    # FileQueue out_dir (hermetic)


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
    """Commit; on error roll back so the connection stays usable (best-effort)."""
    try:
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def _write_vlm_observation(conn, out: dict, event_id: str | None) -> str | None:
    """Insert a `vlm_observations` row; returns observation id or None."""
    if conn is None:
        return None
    mr = out["model_run"]
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO vlm_observations (event_id, model_name, prompt_version, "
                "input_clip_path, raw_output, parsed_json, parse_success, ttft_ms, "
                "output_tokens, tokens_per_sec, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s) RETURNING id",
                (event_id, mr.get("model_id", ""), out.get("prompt_version"),
                 out.get("clip_path"), out.get("raw_output"),
                 json.dumps(out["verification"]), out["parse_success"],
                 mr.get("ttft_ms"), mr.get("output_tokens"),
                 mr.get("tokens_per_sec"), mr.get("latency_ms")))
            obs_id = cur.fetchone()[0]
        conn.commit()
        return str(obs_id)
    except Exception:
        _safe_commit(conn)
        return None


def _update_event_confidence(conn, event_id: str | None, confidence: float) -> bool:
    """Update the source event's confidence from the VLM verdict (best-effort).
    `events` is partitioned by event_time; update by id only (matches all parts)."""
    if conn is None or not event_id:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE events SET confidence=%s WHERE id=%s",
                        (confidence, event_id))
            updated = cur.rowcount
        conn.commit()
        return bool(updated)
    except Exception:
        _safe_commit(conn)
        return False


def _write_model_run(conn, run: dict, obs_id: str | None) -> None:
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_runs (model_profile_name, model_id, task, runtime, "
                "output_entity_type, latency_ms, ttft_ms, output_tokens, tokens_per_sec, "
                "parse_success, error) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (run.get("model_profile_name", "unknown"), run.get("model_id", ""),
                 run.get("task", "vlm_clip_reasoning"), run.get("runtime", ""),
                 "vlm_observation", run.get("latency_ms"), run.get("ttft_ms"),
                 run.get("output_tokens"), run.get("tokens_per_sec"),
                 run.get("parse_success"), run.get("error")))
        conn.commit()
    except Exception:
        _safe_commit(conn)


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


def needs_verification(ev: dict) -> bool:
    """True iff the event asks for VLM verification (top-level or nested meta)."""
    meta = ev.get("metadata_json") or {}
    return bool(ev.get("needs_vlm_verification", meta.get("needs_vlm_verification", False)))


# --------------------------------------------------------------------------
# Core: verify one flagged event with the plugin + persist its outputs.
# --------------------------------------------------------------------------
def process_event(ev: dict, plugin, queue, conn) -> dict:
    """Run the VLM plugin on one flagged `event.created` and persist outputs.
    Non-flagged events are skipped cheaply. Returns a summary dict for tests."""
    if not needs_verification(ev):
        return {"event_id": ev.get("event_id"), "skipped": True}

    request = {
        "clip_path": ev.get("evidence_path") or ev.get("clip_path"),
        "event": ev,
        "event_id": ev.get("event_id"),
    }
    out = plugin.infer(request)
    event_id = ev.get("event_id")
    obs_id = _write_vlm_observation(conn, out, event_id)
    confidence = float(out["verification"].get("confidence", 0.0))
    _update_event_confidence(conn, event_id, confidence)
    _write_model_run(conn, out["model_run"], obs_id)

    try:
        queue.publish({
            "type": "vlm.verified",
            "event_id": event_id,
            "verdict": out["verification"]["verdict"],
            "confidence": confidence,
            "structured_event_type": out["verification"].get("structured_event_type", ""),
            "parse_success": out["parse_success"],
            "source_engine": "vlm_worker",
        })
    except Exception:
        pass

    return {"event_id": event_id, "skipped": False,
            "verdict": out["verification"]["verdict"], "confidence": confidence,
            "parse_success": out["parse_success"],
            "faked": out["model_run"].get("faked", False),
            "observation_written": obs_id is not None}


def run(once: bool = False, limit: int = 8, backend: str | None = None,
        sleep_sec: float = 2.0) -> list[dict]:
    """Worker entrypoint.

    once=True  -> drain up to `limit` events then return.
    once=False -> loop forever, polling `limit` at a time.
    Returns the list of per-event summaries (useful in tests)."""
    queue = make_queue(backend, EVENTS_DIR)
    conn = _db_conn()
    plugin = load_plugin()                                    # active-profile plugin
    plugin.load()                                             # lazy/non-fatal
    is_pg = getattr(queue, "name", "") == "pg"
    summaries: list[dict] = []
    try:
        while True:
            if is_pg:
                claimed = queue.consume(STREAM, limit=limit)  # [(job_id, payload)]
                jobs = [(jid, p) for jid, p in claimed]
            else:                                             # file fallback: no ack/claim
                jobs = [(None, p) for p in _file_jobs(EVENTS_DIR, limit)]
            for job_id, ev in jobs:
                try:
                    s = process_event(ev, plugin, queue, conn)
                    summaries.append(s)
                    if is_pg and job_id is not None:
                        queue.ack(job_id)
                except Exception as e:                        # one bad event must not kill the loop
                    print(f"[vlm-worker] event failed: {e}")
                    if is_pg and job_id is not None:
                        queue.fail(job_id)
            if once or not jobs:
                if once:
                    break
                time.sleep(sleep_sec)                         # idle backoff
        return summaries
    finally:
        try:
            plugin.unload()
        except Exception:
            pass
        try:
            queue.close()
        except Exception:
            pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":                                   # python -m worker / direct
    import argparse
    ap = argparse.ArgumentParser(description="vlm-worker")
    ap.add_argument("--once", action="store_true", help="drain then exit")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--backend", default=None, help="pg|file|redis (default: env)")
    args = ap.parse_args()
    res = run(once=args.once, limit=args.limit, backend=args.backend)
    verified = sum(1 for s in res if not s.get("skipped"))
    print(f"[vlm-worker] processed {len(res)} event(s), verified {verified}: {res}")
