#!/usr/bin/env python3
"""Validate the DB component (psycopg + connectivity + migrations + seed).

Import = hard fail; a down/unmigrated DB = warn (dev boxes may not run Postgres).
"""
from __future__ import annotations

from _common import C, cli, try_import

CORE_TABLES = ("stores", "cameras", "zones", "video_segments", "ai_windows",
               "events", "model_runs", "job_queue")


def checks():
    out = [try_import("psycopg")]
    try:
        from db.db import connect
        with connect() as conn, conn.cursor() as cur:
            out.append(C("postgres reachable", True))
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            have = {r[0] for r in cur.fetchall()}
            missing = [t for t in CORE_TABLES if t not in have]
            out.append(C("core tables present", not missing,
                         f"missing {missing}; run 'make migrate'" if missing else "", warn=bool(missing)))
            cur.execute("SELECT count(*) FROM stores") if "stores" in have else None
            n = cur.fetchone()[0] if "stores" in have else 0
            out.append(C("seeded (stores>0)", n > 0, "run 'make seed'" if not n else "", warn=n == 0))
    except Exception as e:
        out.append(C("postgres reachable", False, str(e)[:70], warn=True))
    return out


if __name__ == "__main__":
    raise SystemExit(cli("db — Postgres (metadata + queue)", checks()))
