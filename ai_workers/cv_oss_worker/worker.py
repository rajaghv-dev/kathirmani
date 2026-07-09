"""cv_oss_worker loop (master plan Phase 4).

Consumes the `ai_window.ready` stream, runs `CvOssDetector` per window, then:
  * writes `detections` rows,
  * writes a §8.3 `CommonEvent` into `events` (+ publishes it to the queue),
  * writes a `model_runs` row,
  * increments the `model_*` metrics (done inside the plugin).

Everything that touches a service is guarded: with Postgres down, `make_queue`
falls back to `FileQueue` and all DB writes no-op, so `run()` degrades to a
hermetic "consume from file / write nothing" loop instead of crashing.

Entrypoint: `run(once=False, limit=N)`.  `python -m` friendly via __main__.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

from ingestion.bus import make_queue                 # noqa: E402

from .plugin import CvOssDetector

STREAM = "ai_window.ready"
EVENTS_DIR = _REPO_ROOT / "data" / "queue"           # FileQueue out_dir (hermetic default)


# --------------------------------------------------------------------------
# DB helpers — all best-effort; return False/None when Postgres is unavailable.
# --------------------------------------------------------------------------
def _db_conn():
    try:
        from db import connect
        return connect()
    except Exception:
        return None


def _safe_commit(conn) -> None:
    """Commit; on any error roll back so the connection stays usable. All DB
    writes are best-effort — a rejected row (e.g. missing FK row when running
    against a not-yet-seeded DB) must never take down the worker loop."""
    try:
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def _write_detections(conn, window_id: str, detections: list[dict]) -> int:
    """Insert detection rows; returns count written (0 if no DB / no window /
    rejected). Each row is committed independently so one bad row (or a missing
    ai_windows FK) doesn't poison the rest."""
    if conn is None or not window_id:
        return 0
    n = 0
    for d in detections:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO detections (ai_window_id, model_name, label, confidence, "
                    "bbox_json, frame_time_sec) VALUES (%s, %s, %s, %s, %s::jsonb, %s)",
                    (window_id, d.get("model_name", ""), d["label"], d["confidence"],
                     json.dumps(d["bbox"]), d.get("frame_time_sec", 0.0)))
            conn.commit()
            n += 1
        except Exception:
            _safe_commit(conn)                          # rollback, keep going
    return n


def _write_event(conn, ev: dict) -> str | None:
    """Insert a CommonEvent into `events` (best-effort); returns event_id or None."""
    if conn is None:
        return None
    meta = {k: ev[k] for k in ("objects", "track_ids", "zones",
                               "needs_vlm_verification", "evidence_path") if k in ev}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (id, store_id, camera_id, segment_id, ai_window_id, "
                "event_type, severity, confidence, source_engine, metadata_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                (ev["event_id"], ev["store_id"], ev.get("camera_id"),
                 ev.get("segment_id"), ev.get("ai_window_id"), ev["event_type"],
                 ev.get("severity", "info"), ev.get("confidence", 0.0),
                 ev.get("source_engine", ""), json.dumps(meta)))
        conn.commit()
        return ev["event_id"]
    except Exception:
        _safe_commit(conn)
        return None


def _write_tracks(conn, camera_id: str, tracks: list) -> int:
    """Upsert the tracker's live tracks into the `tracks` table (best-effort).

    Keyed on (camera_id, track_external_id): an existing row has its last_seen +
    metadata refreshed, otherwise a new row is inserted. Returns count touched.
    Each track is committed independently so one bad/rejected row (e.g. a missing
    cameras FK against an unseeded DB) never poisons the rest or the loop."""
    if conn is None or not tracks:
        return 0
    n = 0
    for t in tracks:
        ext_id = str(getattr(t, "track_id", ""))
        if not ext_id:
            continue
        meta = json.dumps({"first_seen_sec": getattr(t, "first_seen", 0.0),
                            "last_seen_sec": getattr(t, "last_seen", 0.0),
                            "hits": getattr(t, "hits", 0)})
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE tracks SET last_seen = now(), label = %s, "
                    "metadata_json = %s::jsonb "
                    "WHERE camera_id = %s AND track_external_id = %s",
                    (getattr(t, "label", ""), meta, camera_id, ext_id))
                if cur.rowcount == 0:
                    cur.execute(
                        "INSERT INTO tracks (camera_id, track_external_id, label, "
                        "first_seen, last_seen, metadata_json) "
                        "VALUES (%s, %s, %s, now(), now(), %s::jsonb)",
                        (camera_id, ext_id, getattr(t, "label", ""), meta))
            conn.commit()
            n += 1
        except Exception:
            _safe_commit(conn)                          # rollback, keep going
    return n


def _write_model_run(conn, run: dict, n_det: int) -> None:
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_runs (model_profile_name, model_id, task, runtime, "
                "latency_ms, error) VALUES (%s, %s, %s, %s, %s, %s)",
                (run.get("model_profile_name", "unknown"), run.get("model_id", ""),
                 run.get("task", "detection"), run.get("runtime", ""),
                 run.get("latency_ms"), run.get("error")))
        conn.commit()
    except Exception:
        _safe_commit(conn)


# --------------------------------------------------------------------------
# File fallback consumer — read unprocessed lines from the FileQueue jsonl.
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
# Core: process one window with the plugin + persist its outputs.
# --------------------------------------------------------------------------
def process_window(window: dict, plugin: CvOssDetector, queue, conn) -> dict:
    """Run the plugin on one ai_window.ready message and persist outputs.
    Returns a small summary dict (counts) for logging/tests."""
    _t0 = time.time()
    out = plugin.infer(window)
    try:  # telemetry hook: per-model runtime resources + run annotation (never fatal)
        from model_plugins.base.run_hooks import record_model_run
        record_model_run(getattr(plugin, "config", None), _t0, time.time(), out.get("model_run"))
    except Exception:
        pass
    n_det = _write_detections(conn, window.get("window_id"), out["detections"])
    camera_id = window.get("camera_id", "")
    n_trk = _write_tracks(conn, camera_id, plugin.live_tracks(camera_id))
    n_ev = 0
    for ev in out["events"]:
        _write_event(conn, ev)                       # no-op without DB
        try:
            queue.publish({"type": "event.created", **ev})
        except Exception:
            pass
        n_ev += 1
    _write_model_run(conn, out["model_run"], n_det)
    return {"window_id": window.get("window_id"), "detections": len(out["detections"]),
            "detections_written": n_det, "tracks_written": n_trk, "events": n_ev,
            "faked": out["model_run"].get("faked", False)}


def make_detection_plugin() -> CvOssDetector:
    """Detection plugin from the active profile in configs/models.yaml
    (`tasks.detection.plugin`; MODEL_PROFILE env overrides the profile).
    Unknown/missing config → the default CvOssDetector, same as before."""
    import os
    name = "cv_oss_detector"
    try:
        import yaml
        doc = yaml.safe_load((_REPO_ROOT / "configs" / "models.yaml").read_text())
        prof = os.environ.get("MODEL_PROFILE") or doc.get("active_profile", "")
        name = doc["profiles"][prof]["tasks"]["detection"].get("plugin", name)
    except Exception:
        pass
    if name == "nvidia_locate_anything":
        from .locate_plugin import NvidiaLocateAnythingDetector
        return NvidiaLocateAnythingDetector()
    return CvOssDetector()


def run(once: bool = False, limit: int = 8, backend: str | None = None,
        sleep_sec: float = 2.0) -> list[dict]:
    """Worker entrypoint.

    once=True  -> drain up to `limit` windows then return.
    once=False -> loop forever, polling `limit` at a time.
    Returns the list of per-window summaries processed (useful in tests).
    """
    queue = make_queue(backend, EVENTS_DIR)
    conn = _db_conn()
    plugin = make_detection_plugin()
    plugin.load()                                    # lazy/non-fatal
    is_pg = getattr(queue, "name", "") == "pg"
    summaries: list[dict] = []
    try:
        while True:
            if is_pg:
                claimed = queue.consume(STREAM, limit=limit)    # [(job_id, payload)]
                jobs = [(jid, p) for jid, p in claimed]
            else:                                    # file fallback: no ack/claim
                jobs = [(None, p) for p in _file_jobs(EVENTS_DIR, limit)]
            for job_id, window in jobs:
                try:
                    s = process_window(window, plugin, queue, conn)
                    summaries.append(s)
                    if is_pg and job_id is not None:
                        queue.ack(job_id)
                except Exception as e:               # one bad window must not kill the loop
                    print(f"[cv-worker] window failed: {e}")
                    if is_pg and job_id is not None:
                        queue.fail(job_id)
            if once or not jobs:
                if once:
                    break
                time.sleep(sleep_sec)                # idle backoff
        return summaries
    finally:
        plugin.unload()
        try:
            queue.close()
        except Exception:
            pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":                           # python -m ai_workers... / direct
    import argparse
    ap = argparse.ArgumentParser(description="cv_oss_worker")
    ap.add_argument("--once", action="store_true", help="drain then exit")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--backend", default=None, help="pg|file|redis (default: env)")
    args = ap.parse_args()
    res = run(once=args.once, limit=args.limit, backend=args.backend)
    print(f"[cv-worker] processed {len(res)} window(s): {res}")
