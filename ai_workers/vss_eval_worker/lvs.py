"""lvs.py — OSS long-video summarization, the §A4.2 staged pipeline.

Implements the master-plan §A4.2 hierarchical fold **without any VSS dependency**
(spec/10/11: VSS is reference-only, `allow_vss_runtime_dependency: false`):

    10-sec clips
      -> 5-sec AI windows
      -> detector/tracker events            (the input: events + vlm_observations)
      -> suspicious / important moments     (severity / needs_vlm filtering)
      -> per-clip captions                  (summarizer, per 10s bucket)
      -> per-5-minute summaries             (fold captions)
      -> per-hour summary                   (fold 5-min summaries)
      -> incident-aware report              (the §A4.2 output shape)

Input (§A4.2 summarization-worker input):
    {
      "store_id": "kathirmani_01",
      "time_range": {"start": ISO8601, "end": ISO8601},
      "cameras": ["left_aisle", "billing", "entry_exit"],
      "summary_mode": "loss_prevention",
      "include_events": true,
      "include_low_confidence": false
    }

Output (§A4.2 summarization output):
    {
      "summary": str,
      "timeline": [{"time","camera","event","evidence_segment_id"}, ...],
      "open_questions": [str, ...],
      "recommended_review_items": [incident_id, ...]
    }

Events come from the DB (events + vlm_observations) OR from an injected list
(`events=[...]`) so the whole fold is unit-testable with no DB/GPU/weights. The
summarizer is `NemotronSummaryPlugin` (fake by default).

An event item is a plain dict (the §8.3 CommonEvent shape, loosely):
    {
      "event_id"|"id": str,
      "event_time": ISO8601 str,            # or "time"
      "camera_id": str,
      "event_type": str,
      "severity": "info"|"low"|"medium"|"high"|"critical",
      "confidence": float,
      "needs_vlm_verification": bool,
      "evidence_segment_id"|"segment_id": str | None,
      "incident_id": str | None,
      "metadata_json": {"zones": [...], "objects": [...]} | None,
      "vlm": {"verdict","explanation","observed_actions",...} | None,
    }
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


from .summarizer import NemotronSummaryPlugin

# Fold bucket sizes (§A4.2). Clips are 10s; windows roll up to 5-min then 1-hour.
CLIP_SECONDS = 10
FIVE_MIN_SECONDS = 5 * 60
HOUR_SECONDS = 60 * 60

# Severities that count as "suspicious / important moments" worth surfacing.
_SUSPICIOUS_SEVERITIES = {"medium", "high", "critical"}


# ---- request normalization --------------------------------------------------
def _parse_time(val: Any) -> datetime | None:
    """Parse an ISO8601 (or 'HH:MM:SS') timestamp; None on failure. Tolerant so
    injected test fixtures need only a parseable string."""
    if isinstance(val, datetime):
        return val
    if not val:
        return None
    s = str(val)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        for fmt in ("%H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
    return None


def _ev_get(ev: dict, *keys: str, default=None):
    for k in keys:
        if ev.get(k) is not None:
            return ev[k]
    return default


def _ev_time(ev: dict) -> datetime | None:
    return _parse_time(_ev_get(ev, "event_time", "time", "start_time"))


def _bucket_index(t: datetime, origin: datetime, span: int) -> int:
    return int((t - origin).total_seconds() // span)


# ---- event -> text ----------------------------------------------------------
def event_caption_line(ev: dict) -> str:
    """One human-readable line for a single event (used inside a clip caption).
    Prefers a VLM explanation when present, else builds from type/zone/objects."""
    t = _ev_time(ev)
    tstr = t.strftime("%H:%M:%S") if t else "?"
    cam = _ev_get(ev, "camera_id", "camera", default="?")
    etype = _ev_get(ev, "event_type", default="event")
    meta = ev.get("metadata_json") or {}
    if isinstance(meta, str):
        meta = {}
    zones = meta.get("zones") or ev.get("zones") or []
    objs = meta.get("objects") or ev.get("objects") or []
    vlm = ev.get("vlm") or {}

    desc = etype.replace("_", " ")
    if zones:
        desc += f" near {', '.join(map(str, zones))}"
    if objs:
        desc += f" (objects: {', '.join(map(str, objs))})"
    if vlm.get("explanation"):
        desc += f" — VLM: {vlm['explanation']}"
    elif vlm.get("verdict"):
        desc += f" — VLM verdict: {vlm['verdict']}"
    return f"{tstr} {cam}: {desc}"


# ---- filtering (suspicious / important moments) -----------------------------
def filter_events(events: list[dict], req: dict[str, Any]) -> list[dict]:
    """Apply the §A4.2 request flags: camera filter, time_range, and
    include_low_confidence. Always keeps suspicious-severity / needs-VLM events
    even when low-confidence filtering is on (they're the point of the report)."""
    cams = set(req.get("cameras") or [])
    include_low = bool(req.get("include_low_confidence", False))
    tr = req.get("time_range") or {}
    t0, t1 = _parse_time(tr.get("start")), _parse_time(tr.get("end"))

    out = []
    for ev in events:
        cam = _ev_get(ev, "camera_id", "camera")
        if cams and cam not in cams:
            continue
        t = _ev_time(ev)
        if t0 and t and t < t0:
            continue
        if t1 and t and t > t1:
            continue
        if not include_low:
            sev = str(_ev_get(ev, "severity", default="info"))
            conf = float(_ev_get(ev, "confidence", default=0.0) or 0.0)
            suspicious = sev in _SUSPICIOUS_SEVERITIES or bool(
                ev.get("needs_vlm_verification"))
            if not suspicious and conf < 0.5:
                continue
        out.append(ev)
    return out


def is_suspicious(ev: dict) -> bool:
    sev = str(_ev_get(ev, "severity", default="info"))
    return sev in _SUSPICIOUS_SEVERITIES or bool(ev.get("needs_vlm_verification"))


# ---- the staged fold --------------------------------------------------------
def _origin(events: list[dict], req: dict[str, Any]) -> datetime:
    """Bucket origin = time_range.start, else the earliest event, else epoch."""
    tr = req.get("time_range") or {}
    t0 = _parse_time(tr.get("start"))
    if t0:
        return t0
    times = [t for t in (_ev_time(e) for e in events) if t]
    return min(times) if times else datetime(1970, 1, 1, tzinfo=timezone.utc)


def _group(events: list[dict], origin: datetime, span: int) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for ev in events:
        t = _ev_time(ev)
        idx = _bucket_index(t, origin, span) if t else 0
        groups.setdefault(idx, []).append(ev)
    return dict(sorted(groups.items()))


def summarize(req: dict[str, Any], events: list[dict] | None = None,
              conn=None, plugin: NemotronSummaryPlugin | None = None) -> dict[str, Any]:
    """Run the full §A4.2 staged pipeline and return the §A4.2 output shape, with
    a `nests` block exposing the clip->5min->hour hierarchy (for tests/audit).

    events: injected list (tests). If None, pulled from the DB via `conn` (or a
            fresh connection); with no DB the event list is empty (empty report).
    plugin: summarizer; defaults to NemotronSummaryPlugin (fake_infer fallback).
    """
    plugin = plugin or NemotronSummaryPlugin()
    plugin.load()                                     # lazy/non-fatal
    mode = req.get("summary_mode", "loss_prevention")

    raw_events = events if events is not None else _load_events(conn, req)
    kept = filter_events(raw_events, req)
    kept.sort(key=lambda e: (_ev_time(e) or datetime.min.replace(tzinfo=timezone.utc)))

    origin = _origin(kept, req)
    base_ctx = {"store_id": req.get("store_id"), "summary_mode": mode,
                "time_range": req.get("time_range")}

    # --- level 1: per-clip captions (10-sec buckets) ----------------------
    clip_groups = _group(kept, origin, CLIP_SECONDS)
    clip_caps: dict[int, dict] = {}
    for idx, evs in clip_groups.items():
        lines = [event_caption_line(e) for e in evs]
        cam = _ev_get(evs[0], "camera_id", "camera", default="?")
        out = plugin.infer({"level": "clip", "items": lines,
                            "context": {**base_ctx, "camera": cam}})
        clip_caps[idx] = {"clip_index": idx, "caption": out["summary"],
                          "n_events": len(evs), "events": evs}

    # --- level 2: per-5-minute summaries ----------------------------------
    five_groups = _group(kept, origin, FIVE_MIN_SECONDS)
    five_sums: dict[int, dict] = {}
    for idx, evs in five_groups.items():
        # the captions whose clip bucket falls in this 5-min bucket
        sub = [clip_caps[ci]["caption"] for ci in clip_caps
               if (ci * CLIP_SECONDS) // FIVE_MIN_SECONDS == idx]
        out = plugin.infer({"level": "5min", "items": sub or
                            [event_caption_line(e) for e in evs], "context": base_ctx})
        five_sums[idx] = {"window_index": idx, "summary": out["summary"],
                          "n_clips": len(sub), "clip_indices":
                          [ci for ci in clip_caps
                           if (ci * CLIP_SECONDS) // FIVE_MIN_SECONDS == idx]}

    # --- level 3: per-hour summaries --------------------------------------
    hour_sums: dict[int, dict] = {}
    hour_groups: dict[int, list[int]] = {}
    for idx in five_sums:
        h = (idx * FIVE_MIN_SECONDS) // HOUR_SECONDS
        hour_groups.setdefault(h, []).append(idx)
    for h, five_idxs in sorted(hour_groups.items()):
        sub = [five_sums[i]["summary"] for i in five_idxs]
        out = plugin.infer({"level": "hour", "items": sub, "context": base_ctx})
        hour_sums[h] = {"hour_index": h, "summary": out["summary"],
                        "window_indices": five_idxs}

    # --- level 4: incident-aware report (the §A4.2 output) ----------------
    report_items = [hs["summary"] for hs in hour_sums.values()] or \
        ["No activity in the requested time range."]
    report = plugin.infer({"level": "report", "items": report_items,
                           "context": base_ctx})

    timeline = _build_timeline(kept)
    return {
        "store_id": req.get("store_id"),
        "time_range": req.get("time_range"),
        "summary_mode": mode,
        "summary": report["summary"],
        "timeline": timeline,
        "open_questions": _open_questions(kept),
        "recommended_review_items": _review_items(kept),
        # audit/test view of the hierarchy (extra to the §A4.2 shape)
        "nests": {
            "clip_captions": list(clip_caps.values()),
            "five_minute_summaries": list(five_sums.values()),
            "hour_summaries": list(hour_sums.values()),
        },
        "stats": {"events_in": len(raw_events), "events_kept": len(kept),
                  "n_clips": len(clip_caps), "n_5min": len(five_sums),
                  "n_hours": len(hour_sums), "faked": bool(report.get("faked"))},
    }


# ---- §A4.2 output assembly --------------------------------------------------
def _build_timeline(events: list[dict]) -> list[dict[str, Any]]:
    """One timeline entry per suspicious/important moment (§A4.2 timeline[])."""
    out = []
    for ev in events:
        if not is_suspicious(ev):
            continue
        t = _ev_time(ev)
        out.append({
            "time": t.strftime("%H:%M:%S") if t else "?",
            "camera": _ev_get(ev, "camera_id", "camera", default="?"),
            "event": _describe_moment(ev),
            "evidence_segment_id": _ev_get(ev, "evidence_segment_id",
                                           "segment_id", "ai_window_id"),
        })
    return out


def _describe_moment(ev: dict) -> str:
    vlm = ev.get("vlm") or {}
    if vlm.get("explanation"):
        return str(vlm["explanation"])
    etype = _ev_get(ev, "event_type", default="event").replace("_", " ")
    meta = ev.get("metadata_json") or {}
    zones = (meta.get("zones") if isinstance(meta, dict) else None) or ev.get("zones") or []
    return etype + (f" near {', '.join(map(str, zones))}" if zones else "")


def _open_questions(events: list[dict]) -> list[str]:
    """Cross-camera / missing-evidence follow-ups (§A4.2 open_questions[]).
    Driven by VLM missing-evidence + multi-camera suspicious co-occurrence."""
    qs: list[str] = []
    cams = sorted({_ev_get(e, "camera_id", "camera") for e in events
                   if is_suspicious(e) and _ev_get(e, "camera_id", "camera")})
    for ev in events:
        if not is_suspicious(ev):
            continue
        t = _ev_time(ev)
        tstr = t.strftime("%H:%M") if t else "?"
        cam = _ev_get(ev, "camera_id", "camera", default="?")
        vlm = ev.get("vlm") or {}
        for miss in (vlm.get("missing_evidence") or []):
            qs.append(f"Around {tstr} on {cam}: obtain {miss}.")
        # if billing exists in scope, suggest checking it for the same person
        others = [c for c in cams if c != cam]
        if others:
            qs.append(f"Check {', '.join(others)} for the same person around {tstr} "
                      f"(suspicious activity on {cam}).")
    # de-dup, keep order
    seen, uniq = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def _review_items(events: list[dict]) -> list[str]:
    """Incident ids a reviewer should open (§A4.2 recommended_review_items[]).
    Prefers explicit incident_id; falls back to the event id of high-severity
    moments so the report always points somewhere reviewable."""
    items: list[str] = []
    for ev in events:
        iid = ev.get("incident_id")
        if iid:
            items.append(str(iid))
        elif str(_ev_get(ev, "severity", default="info")) in {"high", "critical"}:
            eid = _ev_get(ev, "event_id", "id")
            if eid:
                items.append(str(eid))
    seen, uniq = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


# ---- DB load (best-effort; empty with no DB) --------------------------------
def _load_events(conn, req: dict[str, Any]) -> list[dict]:
    """Pull events (+ their latest vlm_observation) for the store/time-range/
    cameras from Postgres. Best-effort: returns [] with no DB / on any error, so
    `summarize` runs hermetically. The DB path is intentionally simple — the fold
    logic is identical whether events are injected or queried."""
    if conn is None:
        conn = _db_conn()
    if conn is None:
        return []
    tr = req.get("time_range") or {}
    store = req.get("store_id")
    cams = req.get("cameras") or []
    sql = ("SELECT e.id::text, e.camera_id, e.event_type, e.severity, "
           "e.confidence, e.event_time, e.segment_id::text, e.ai_window_id::text, "
           "e.metadata_json, v.parsed_json "
           "FROM events e "
           "LEFT JOIN LATERAL (SELECT parsed_json FROM vlm_observations vo "
           "  WHERE vo.event_id = e.id ORDER BY vo.created_at DESC LIMIT 1) v ON true "
           "WHERE e.store_id = %s")
    params: list[Any] = [store]
    if tr.get("start"):
        sql += " AND e.event_time >= %s"
        params.append(tr["start"])
    if tr.get("end"):
        sql += " AND e.event_time <= %s"
        params.append(tr["end"])
    if cams:
        sql += " AND e.camera_id = ANY(%s)"
        params.append(list(cams))
    sql += " ORDER BY e.event_time ASC LIMIT 5000"
    cols = ("id", "camera_id", "event_type", "severity", "confidence",
            "event_time", "segment_id", "ai_window_id", "metadata_json", "parsed_json")
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return []
    out = []
    for r in rows:
        ev = dict(r)
        ev["event_id"] = r["id"]
        ev["event_time"] = (r["event_time"].isoformat()
                            if hasattr(r["event_time"], "isoformat") else r["event_time"])
        ev["evidence_segment_id"] = r.get("segment_id") or r.get("ai_window_id")
        if r.get("parsed_json"):
            ev["vlm"] = r["parsed_json"]
        out.append(ev)
    return out


def _db_conn():
    try:
        from db import connect
        return connect()
    except Exception:
        return None
