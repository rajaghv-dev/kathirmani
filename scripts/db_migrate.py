#!/usr/bin/env python3
"""Forward/back migration runner (psycopg). Applies db/migrations/*.up.sql in order,
tracks applied versions in schema_migrations. `make migrate` / `make migrate-down`.

Usage:
  python scripts/db_migrate.py up          # apply all pending
  python scripts/db_migrate.py down         # roll back the latest
  python scripts/db_migrate.py status
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db.db import connect  # noqa: E402

MIG = Path(__file__).resolve().parents[1] / "db" / "migrations"


def _versions() -> list[str]:
    return sorted({p.name.split(".")[0] for p in MIG.glob("*.up.sql")})


def _ensure_table(cur):
    cur.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())")


def _applied(cur) -> set[str]:
    cur.execute("SELECT version FROM schema_migrations")
    return {r[0] for r in cur.fetchall()}


def up():
    with connect() as conn, conn.cursor() as cur:
        _ensure_table(cur); done = _applied(cur)
        for v in _versions():
            if v in done:
                continue
            sql = (MIG / f"{v}.up.sql").read_text()
            print(f"applying {v} ...")
            cur.execute(sql)
            cur.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (v,))
        conn.commit()
    print("migrate up: OK")


def down():
    with connect() as conn, conn.cursor() as cur:
        _ensure_table(cur); done = sorted(_applied(cur))
        if not done:
            print("nothing to roll back"); return
        v = done[-1]
        print(f"rolling back {v} ...")
        cur.execute((MIG / f"{v}.down.sql").read_text())
        cur.execute("DELETE FROM schema_migrations WHERE version=%s", (v,))
        conn.commit()
    print("migrate down: OK")


def status():
    with connect() as conn, conn.cursor() as cur:
        _ensure_table(cur); done = _applied(cur)
        for v in _versions():
            print(f"  [{'x' if v in done else ' '}] {v}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "up"
    {"up": up, "down": down, "status": status}.get(cmd, up)()
