"""Retention enforcement (spec/10 Security: PII/retention, Phase 1+12).

CCTV footage is PII/biometric, so it is time-boxed per `configs/retention.yaml`.
This selects rows/files older than the per-type limit and deletes them — BUT clips
tied to an OPEN incident are LOCKED (evidence chain-of-custody) and skipped until
the incident closes.

Safety: **dry-run by default** — it prints what *would* be deleted; deletion only
happens with `--apply`. Guarded for no-DB so the selection logic is unit-testable
on synthetic data without Postgres.

Maps retention keys → (table, time column):
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

# retention.yaml key → (table, timestamp column). Clip blobs live on the FS; the
# row carries the path, so deleting a video_segments row also unlinks its file.
TABLE_FOR = {
    "raw_clips_days": ("video_segments", "created_at"),
    "ai_windows_days": ("ai_windows", "created_at"),
    "detections_days": ("detections", "created_at"),
    "events_days": ("events", "event_time"),
    "vlm_observations_days": ("vlm_observations", "created_at"),
    "incidents_days": ("incidents", "created_at"),
    "embeddings_days": ("embeddings", "created_at"),
}

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "retention.yaml"


def load_policy(path: str | Path | None = None) -> dict:
    """Load configs/retention.yaml → dict. Read-only; we never write configs/."""
    import yaml
    p = Path(path) if path else _DEFAULT_CONFIG
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cutoff(days: int, now: datetime | None = None) -> datetime:
    """Rows/files created at/before this instant are past their retention limit."""
    now = now or datetime.now(timezone.utc)
    return now - timedelta(days=days)


def is_evidence_locked(row: dict, open_incident_segment_ids: set) -> bool:
    """A video_segments row is locked if its clip is part of an OPEN incident.

    `open_incident_segment_ids` is the set of segment ids referenced (via
    incident_events → events → segment_id) by incidents whose status='open'.
    """
    return row.get("id") in open_incident_segment_ids


def select_expired(
    rows: Iterable[dict],
    days: int,
    *,
    time_field: str = "created_at",
    now: datetime | None = None,
    locked_ids: set | None = None,
) -> list[dict]:
    """Pure selection: rows older than `days`, minus evidence-locked ones.

    Works on synthetic dicts (unit-tested) and on DB rows alike. `locked_ids`
    only applies evidence-lock when provided (raw clips path).
    """
    edge = cutoff(days, now)
    locked = locked_ids or set()
    out = []
    for r in rows:
        ts = r.get(time_field)
        if ts is None or ts > edge:
            continue
        if r.get("id") in locked:
            continue  # evidence-lock: open incident → skip
        out.append(r)
    return out


# ---- DB-backed path (guarded; no-op without Postgres) ----------------------
def _open_incident_segment_ids(cur) -> set:
    """segment_ids referenced by events that belong to OPEN incidents."""
    cur.execute(
        "SELECT DISTINCT e.segment_id "
        "FROM incident_events ie "
        "JOIN incidents i ON i.id = ie.incident_id AND i.status = 'open' "
        "JOIN events e ON e.id = ie.event_id "
        "WHERE e.segment_id IS NOT NULL"
    )
    return {row[0] for row in cur.fetchall()}


def plan(policy: dict, now: datetime | None = None) -> list[dict]:
    """Build a deletion plan from the live DB: list of {key,table,count,locked_skipped}.

    Returns an empty plan if no DB is reachable (guarded), so dry-run never crashes.
    """
    try:
        from db import connect
    except Exception:
        return []
    retention = policy.get("retention", {})
    out: list[dict] = []
    try:
        with connect() as conn, conn.cursor() as cur:
            locked = _open_incident_segment_ids(cur)
            for key, days in retention.items():
                if key not in TABLE_FOR:
                    continue
                table, tcol = TABLE_FOR[key]
                edge = cutoff(int(days), now)
                cur.execute(f"SELECT count(*) FROM {table} WHERE {tcol} <= %s", (edge,))
                total = cur.fetchone()[0]
                locked_skipped = 0
                if table == "video_segments" and locked:
                    cur.execute(
                        f"SELECT count(*) FROM {table} WHERE {tcol} <= %s AND id = ANY(%s)",
                        (edge, list(locked)),
                    )
                    locked_skipped = cur.fetchone()[0]
                out.append({
                    "key": key, "table": table, "days": int(days),
                    "would_delete": total - locked_skipped,
                    "locked_skipped": locked_skipped,
                })
    except Exception as exc:  # pragma: no cover - depends on live DB
        print(f"[retention] DB unavailable, nothing planned: {exc}", file=sys.stderr)
        return []
    return out


def apply(policy: dict, now: datetime | None = None) -> list[dict]:
    """Execute the plan: DELETE expired, non-locked rows. Returns per-table counts.

    Audited. Clip-file unlinking for video_segments is left to the FS sweep step
    (rows carry storage_path); here we delete the metadata rows transactionally.
    """
    try:
        from db import connect
    except Exception:
        return []
    from .audit import audit
    retention = policy.get("retention", {})
    out: list[dict] = []
    with connect() as conn, conn.cursor() as cur:
        locked = _open_incident_segment_ids(cur)
        for key, days in retention.items():
            if key not in TABLE_FOR:
                continue
            table, tcol = TABLE_FOR[key]
            edge = cutoff(int(days), now)
            if table == "video_segments" and locked:
                cur.execute(
                    f"DELETE FROM {table} WHERE {tcol} <= %s AND NOT (id = ANY(%s))",
                    (edge, list(locked)),
                )
            else:
                cur.execute(f"DELETE FROM {table} WHERE {tcol} <= %s", (edge,))
            out.append({"key": key, "table": table, "deleted": cur.rowcount})
        conn.commit()
    audit("retention.apply", actor="retention-cli", target=None,
          deleted=out, policy_keys=list(retention))
    return out


def _print_plan(rows: list[dict], applied: bool) -> None:
    verb = "DELETED" if applied else "would delete"
    if not rows:
        print("[retention] no DB reachable or nothing expired — nothing to do.")
        return
    for r in rows:
        if applied:
            print(f"  {r['table']:<18} {verb} {r['deleted']}")
        else:
            extra = f" (locked-skipped {r['locked_skipped']})" if r.get("locked_skipped") else ""
            print(f"  {r['table']:<18} {verb} {r['would_delete']} (>{r['days']}d){extra}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Enforce configs/retention.yaml (dry-run by default).")
    ap.add_argument("--config", help="path to retention.yaml (default: configs/retention.yaml)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry-run — print only)")
    args = ap.parse_args(argv)

    policy = load_policy(args.config)
    if args.apply:
        print("[retention] APPLY mode — deleting expired, non-evidence-locked data.")
        _print_plan(apply(policy), applied=True)
    else:
        print("[retention] DRY-RUN (use --apply to delete). Evidence-locked clips skipped.")
        _print_plan(plan(policy), applied=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
