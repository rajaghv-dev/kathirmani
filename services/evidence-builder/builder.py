"""Evidence builder (master plan Phase 7 / spec/10 §3.2 evidence clip).

Turns an incident into a reviewable, auditable evidence package:

  * gather the incident's events (via `incident_events`), each event's
    `video_segments` and `vlm_observations`,
  * stitch a ~20-sec **pre/during/post** evidence clip from the stored segment
    clips around the event time using **PyAV** (concat + trim); if the source
    clips are missing/unreadable we degrade gracefully to a *manifest-only*
    package (no bytes, just the JSON evidence record),
  * compute a **sha256** of the clip bytes (chain-of-custody integrity, mirrors
    `video_segments.checksum`),
  * build a `timeline` of events + observations ordered by time,
  * write `evidence_path` + a `summary` (the JSON package) back to the incident.

Everything that touches Postgres is best-effort: with the DB unavailable the
builder still assembles the package from whatever it is handed (so it is unit-
testable with synthetic rows and degrades like the other Phase-4/5/6 workers).

Public API:
  * `build_evidence(incident_id, *, conn=None, out_dir=None) -> dict`
  * `assemble_timeline(events, observations) -> list[dict]`   (pure, no I/O)
  * `sha256_file(path) -> str` / `sha256_bytes(b) -> str`
  * `stitch_clip(segments, event_time, out_path, *, pre=20, post=20) -> dict`
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Default evidence drop — local NVMe, mirrors the "clip blobs live on the
# filesystem; rows store paths" decision (spec/10 Data stack).
EVIDENCE_DIR = _REPO_ROOT / "data" / "evidence"

# Evidence window around the event time (master plan §3.2: 20s pre/during/post).
PRE_SEC = 20.0
POST_SEC = 20.0


# --------------------------------------------------------------------------
# Time helpers — rows may carry datetimes (psycopg) or ISO strings (synthetic).
# --------------------------------------------------------------------------
def _to_dt(v: Any) -> datetime | None:
    """Coerce a timestamp (datetime | ISO8601 str | None) to an aware datetime."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _iso(v: Any) -> str | None:
    dt = _to_dt(v)
    return dt.isoformat() if dt else None


# --------------------------------------------------------------------------
# sha256 — chain-of-custody integrity for the evidence bytes.
# --------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Streaming sha256 of a file (so a multi-MB clip never lands fully in RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Timeline — pure assembly of events + observations, ordered by time.
# --------------------------------------------------------------------------
def assemble_timeline(events: list[dict], observations: list[dict]) -> list[dict]:
    """Merge events + VLM observations into one chronologically ordered timeline.

    Pure: no DB, no I/O. Each entry: {kind, ts, ...}. Entries with no usable
    timestamp sort last (stable) so an evidence package is never dropped just
    because one row lacks a time.
    """
    items: list[dict] = []
    for e in events or []:
        ts = e.get("event_time")
        items.append({
            "kind": "event",
            "ts": _iso(ts),
            "event_id": str(e.get("id") or e.get("event_id") or ""),
            "event_type": e.get("event_type"),
            "severity": e.get("severity"),
            "confidence": _num(e.get("confidence")),
            "camera_id": e.get("camera_id"),
            "source_engine": e.get("source_engine"),
        })
    for o in observations or []:
        parsed = o.get("parsed_json") or {}
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except (ValueError, TypeError):
                parsed = {}
        items.append({
            "kind": "vlm_observation",
            "ts": _iso(o.get("created_at")),
            "observation_id": str(o.get("id") or ""),
            "event_id": str(o.get("event_id") or ""),
            "model_name": o.get("model_name"),
            "verdict": parsed.get("verdict"),
            "confidence": _num(parsed.get("confidence")),
            "parse_success": o.get("parse_success"),
        })
    # Sort by ISO ts; None last (ISO strings sort lexicographically == chronologically).
    items.sort(key=lambda i: (i["ts"] is None, i["ts"] or ""))
    return items


def _num(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------
# Clip stitching — PyAV concat + trim of the segment clips around the event.
# --------------------------------------------------------------------------
def _select_window(segments: list[dict], event_time: datetime | None,
                   pre: float, post: float) -> tuple[datetime | None, datetime | None]:
    """Compute the [window_start, window_end] we want to cover, from the event
    time (or the segments' own span if no event time is known)."""
    if event_time is not None:
        return (event_time.timestamp() - pre, event_time.timestamp() + post)
    # No event time: fall back to the union of segment spans (best-effort).
    starts = [_to_dt(s.get("start_time")) for s in segments]
    ends = [_to_dt(s.get("end_time")) for s in segments]
    starts = [s.timestamp() for s in starts if s]
    ends = [e.timestamp() for e in ends if e]
    if not starts or not ends:
        return (None, None)
    return (min(starts), max(ends))


def stitch_clip(segments: list[dict], event_time: Any, out_path: str | Path,
                *, pre: float = PRE_SEC, post: float = POST_SEC) -> dict:
    """Stitch the segment clips covering [event-pre, event+post] into one MP4.

    `segments` rows must carry `storage_path` + `start_time` (+ optional
    `end_time`). We open each source clip that overlaps the window with PyAV,
    decode/re-encode its frames into a single output container, trimming to the
    window. Audio is dropped (CCTV evidence is video-only here).

    Returns a result dict:
      {ok, reason, out_path, sha256, segments_used, window_start, window_end}.
    On any failure (no PyAV, missing files, decode error) it returns ok=False
    with a `reason` so the caller can fall back to a *manifest-only* package —
    the evidence record is always produced, the bytes are best-effort.
    """
    out_path = Path(out_path)
    et = _to_dt(event_time)
    win_start, win_end = _select_window(segments, et, pre, post)
    result: dict[str, Any] = {
        "ok": False, "reason": None, "out_path": str(out_path), "sha256": None,
        "segments_used": [], "window_start": None, "window_end": None,
    }
    if win_start is not None:
        result["window_start"] = datetime.fromtimestamp(win_start, timezone.utc).isoformat()
        result["window_end"] = datetime.fromtimestamp(win_end, timezone.utc).isoformat()

    # Order candidate sources by start time; keep only those whose file exists
    # and whose [start,end] overlaps the requested window.
    candidates = []
    for s in segments:
        sp = s.get("storage_path")
        if not sp or not Path(sp).exists():
            continue
        s_start = _to_dt(s.get("start_time"))
        s_end = _to_dt(s.get("end_time"))
        st = s_start.timestamp() if s_start else None
        en = s_end.timestamp() if s_end else (st if st is not None else None)
        if win_start is not None and st is not None and en is not None:
            if en < win_start or st > win_end:        # no overlap
                continue
        candidates.append((st, sp, st, en))
    candidates.sort(key=lambda c: (c[0] is None, c[0] or 0))

    if not candidates:
        result["reason"] = "no source clips on disk (manifest-only package)"
        return result

    try:
        import av                                       # PyAV (lazy — degrade if absent)
    except Exception as exc:                            # pragma: no cover - env dependent
        result["reason"] = f"PyAV unavailable: {exc}"
        return result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    used: list[str] = []
    try:
        out = av.open(str(out_path), mode="w")
        ostream = None
        try:
            for _key, sp, seg_start, seg_end in candidates:
                try:
                    inp = av.open(sp)
                except Exception:
                    continue                            # skip a single unreadable clip
                try:
                    ivs = inp.streams.video[0] if inp.streams.video else None
                    if ivs is None:
                        continue
                    if ostream is None:
                        # Match the first usable source's geometry; H.264 is the
                        # admissible/portable default for the review UI.
                        ostream = out.add_stream("libx264", rate=ivs.average_rate or 25)
                        ostream.width = ivs.codec_context.width
                        ostream.height = ivs.codec_context.height
                        ostream.pix_fmt = "yuv420p"
                    for frame in inp.decode(ivs):
                        # Trim to the window using the frame's wall-clock time
                        # (segment start + frame PTS), when we know the window.
                        if win_start is not None and seg_start is not None and frame.time is not None:
                            ft = seg_start + float(frame.time)
                            if ft < win_start or ft > win_end:
                                continue
                        frame.pts = None                # let the encoder assign PTS
                        for pkt in ostream.encode(frame):
                            out.mux(pkt)
                    used.append(sp)
                finally:
                    inp.close()
            if ostream is not None:
                for pkt in ostream.encode():            # flush
                    out.mux(pkt)
        finally:
            out.close()
    except Exception as exc:
        result["reason"] = f"stitch failed: {exc}"
        # leave a partial/empty file out of the way
        try:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except OSError:
            pass
        return result

    if not used or not out_path.exists() or out_path.stat().st_size == 0:
        result["reason"] = "no frames in evidence window (manifest-only package)"
        try:
            if out_path.exists():
                out_path.unlink()
        except OSError:
            pass
        return result

    result["ok"] = True
    result["segments_used"] = used
    result["sha256"] = sha256_file(out_path)
    return result


# --------------------------------------------------------------------------
# DB gather — best-effort reads (no-op / None when Postgres is unavailable).
# --------------------------------------------------------------------------
def _db_conn():
    try:
        from db import connect
        return connect()
    except Exception:
        return None


def _gather_from_db(conn, incident_id: str) -> dict | None:
    """Pull the incident + its events + segments + vlm_observations.

    Returns None if the incident is unknown / DB unusable, so the caller can
    decide whether to error or fall back. `events` is partitioned by event_time;
    we read by id (matches every partition).
    """
    from psycopg.rows import dict_row
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, store_id, title, incident_type, severity, status, "
                "start_time, end_time FROM incidents WHERE id=%s", (incident_id,))
            incident = cur.fetchone()
            if not incident:
                return None

            cur.execute("SELECT event_id FROM incident_events WHERE incident_id=%s",
                        (incident_id,))
            event_ids = [r["event_id"] for r in cur.fetchall()]

            events: list[dict] = []
            observations: list[dict] = []
            segments: list[dict] = []
            if event_ids:
                cur.execute(
                    "SELECT id, store_id, camera_id, segment_id, event_type, severity, "
                    "confidence, source_engine, event_time FROM events "
                    "WHERE id = ANY(%s)", (event_ids,))
                events = cur.fetchall()

                cur.execute(
                    "SELECT id, event_id, model_name, prompt_version, parsed_json, "
                    "parse_success, created_at FROM vlm_observations "
                    "WHERE event_id = ANY(%s) ORDER BY created_at", (event_ids,))
                observations = cur.fetchall()

                seg_ids = [e["segment_id"] for e in events if e.get("segment_id")]
                if seg_ids:
                    cur.execute(
                        "SELECT id, camera_id, start_time, end_time, duration_sec, "
                        "storage_path, checksum FROM video_segments WHERE id = ANY(%s)",
                        (seg_ids,))
                    segments = cur.fetchall()
        return {"incident": incident, "events": events,
                "observations": observations, "segments": segments}
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return None


def _write_back(conn, incident_id: str, evidence_path: str | None, summary: str) -> bool:
    """Write `evidence_path` + `summary` (the JSON package) back to the incident."""
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE incidents SET evidence_path=%s, summary=%s, updated_at=now() "
                "WHERE id=%s", (evidence_path, summary, incident_id))
            updated = cur.rowcount
        conn.commit()
        return bool(updated)
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False


# --------------------------------------------------------------------------
# Core: build_evidence.
# --------------------------------------------------------------------------
def build_package(incident: dict, events: list[dict], observations: list[dict],
                  segments: list[dict], *, out_dir: str | Path | None = None,
                  pre: float = PRE_SEC, post: float = POST_SEC) -> dict:
    """Assemble an evidence package dict from already-gathered rows (pure of DB).

    Stitches the clip (best-effort), builds the timeline, computes the sha256,
    and returns the package — the JSON we persist into `incidents.summary`.
    Exposed separately so it is unit-testable with synthetic rows.
    """
    out_dir = Path(out_dir) if out_dir else EVIDENCE_DIR
    incident_id = str(incident.get("id") or incident.get("incident_id") or "unknown")

    # Event "anchor" time = earliest event time (the window centers on the event).
    ev_times = [_to_dt(e.get("event_time")) for e in events]
    ev_times = [t for t in ev_times if t]
    anchor = min(ev_times) if ev_times else _to_dt(incident.get("start_time"))

    out_clip = out_dir / f"{incident_id}.mp4"
    stitch = stitch_clip(segments, anchor, out_clip, pre=pre, post=post)
    timeline = assemble_timeline(events, observations)

    # The latest VLM verdict (if any) is the headline hypothesis for the reviewer.
    verdicts = [t for t in timeline if t["kind"] == "vlm_observation" and t.get("verdict")]
    headline = verdicts[-1] if verdicts else None

    package = {
        "incident_id": incident_id,
        "title": incident.get("title"),
        "incident_type": incident.get("incident_type"),
        "severity": incident.get("severity"),
        "store_id": incident.get("store_id"),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "window": {"pre_sec": pre, "post_sec": post,
                   "anchor": anchor.isoformat() if anchor else None,
                   "start": stitch["window_start"], "end": stitch["window_end"]},
        "evidence_path": stitch["out_path"] if stitch["ok"] else None,
        "manifest_only": not stitch["ok"],
        "clip_reason": stitch["reason"],
        "chain_of_custody": {
            # sha256 of the produced clip (or null when manifest-only). The
            # source segment checksums are recorded so tampering with an input
            # is detectable too (master plan Security: tamper-evident).
            "evidence_sha256": stitch["sha256"],
            "segments_used": stitch["segments_used"],
            "source_segments": [
                {"id": str(s.get("id")), "storage_path": s.get("storage_path"),
                 "checksum": s.get("checksum")} for s in segments
            ],
        },
        "counts": {"events": len(events), "observations": len(observations),
                   "segments": len(segments)},
        "headline_verdict": headline,
        "timeline": timeline,
    }
    return package


def build_evidence(incident_id: str, *, conn: Any = None,
                   out_dir: str | Path | None = None,
                   events: list[dict] | None = None,
                   observations: list[dict] | None = None,
                   segments: list[dict] | None = None,
                   incident: dict | None = None,
                   pre: float = PRE_SEC, post: float = POST_SEC) -> dict:
    """Build (and, with a DB, persist) an incident's evidence package.

    Two modes:
      * **DB mode** (default): opens Postgres (or uses the passed `conn`), reads
        the incident's events/segments/observations, builds the package, and
        writes `evidence_path`+`summary` back to the incident row.
      * **Injected mode** (tests / no-DB): pass `incident`/`events`/`observations`/
        `segments` directly and it builds the package without any DB access.

    Always returns the package dict. `persisted` is True only when a DB row was
    updated. Degrades gracefully: missing clips -> manifest-only; DB down ->
    package still returned, just not persisted.
    """
    injected = incident is not None or events is not None
    owns_conn = False
    if not injected:
        if conn is None:
            conn = _db_conn()
            owns_conn = conn is not None
        gathered = _gather_from_db(conn, incident_id) if conn is not None else None
        if gathered is None:
            # Unknown incident or DB unusable — return an empty manifest package.
            if owns_conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return {"incident_id": incident_id, "persisted": False,
                    "manifest_only": True,
                    "clip_reason": "incident not found or DB unavailable",
                    "timeline": [], "counts": {"events": 0, "observations": 0,
                                               "segments": 0}}
        incident = gathered["incident"]
        events = gathered["events"]
        observations = gathered["observations"]
        segments = gathered["segments"]

    incident = incident or {"id": incident_id}
    package = build_package(incident, events or [], observations or [],
                            segments or [], out_dir=out_dir, pre=pre, post=post)

    persisted = False
    if conn is not None:
        persisted = _write_back(conn, incident_id, package["evidence_path"],
                                json.dumps(package))
    package["persisted"] = persisted

    if owns_conn:
        try:
            conn.close()
        except Exception:
            pass
    return package


if __name__ == "__main__":                              # python -m builder <incident_id>
    import argparse
    ap = argparse.ArgumentParser(description="evidence-builder")
    ap.add_argument("incident_id", help="incident UUID to build evidence for")
    ap.add_argument("--out-dir", default=None, help="evidence output dir")
    args = ap.parse_args()
    pkg = build_evidence(args.incident_id, out_dir=args.out_dir)
    print(json.dumps({k: v for k, v in pkg.items() if k != "timeline"}, indent=2))
    print(f"timeline: {len(pkg.get('timeline', []))} entries; "
          f"persisted={pkg.get('persisted')}; manifest_only={pkg.get('manifest_only')}")
