"""Append-only audit log (spec/10 Security: auditability + chain-of-custody, Phase 12).

Every consequential action (approve/reject a hypothesis, activate a profile, run
retention/backup) is recorded with actor + target + timestamp + (model) context, so
every decision is traceable — trust through legibility.

Persistence is best-effort and degrades gracefully:
  1. if an `audit_log` table exists  → INSERT a row (transactional, append-only);
  2. else                            → append a line to data/audit/audit.jsonl.
Records are redacted before writing, so secrets never enter the audit trail.

`audit_log` is NOT created here (we don't edit db/); see the migration DDL in the
module docstring footer / integration notes.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .redaction import redact

# repo-root-relative fallback path; overridable for tests via AUDIT_LOG_PATH.
_DEFAULT_JSONL = Path(__file__).resolve().parents[2] / "data" / "audit" / "audit.jsonl"


def _jsonl_path() -> Path:
    return Path(os.environ.get("AUDIT_LOG_PATH", str(_DEFAULT_JSONL)))


def _record(action: str, actor: str, target: str | None, fields: dict[str, Any]) -> dict:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "actor": actor,
        "target": target,
        "fields": fields,
    }
    return redact(rec)  # belt-and-suspenders: never persist a secret


def _try_db_insert(rec: dict) -> bool:
    """Insert into audit_log if the table exists. Returns True on success."""
    try:
        from db import connect
    except Exception:
        return False
    try:
        with connect() as conn, conn.cursor() as cur:
            # to_regclass returns NULL if the table is absent → fall back to JSONL.
            cur.execute("SELECT to_regclass('public.audit_log')")
            if cur.fetchone()[0] is None:
                return False
            cur.execute(
                "INSERT INTO audit_log (ts, action, actor, target, fields) "
                "VALUES (%s, %s, %s, %s, %s)",
                (rec["ts"], rec["action"], rec["actor"], rec["target"],
                 json.dumps(rec["fields"])),
            )
            conn.commit()
        return True
    except Exception:
        return False  # never let auditing break the caller; JSONL still captures it


def _jsonl_append(rec: dict) -> Path:
    path = _jsonl_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def audit(action: str, actor: str, target: str | None = None, **fields: Any) -> dict:
    """Record an audit event. Writes to audit_log if present, else JSONL fallback.

    Returns the (redacted) record that was written, so callers/tests can inspect it.
    Secrets in `fields` are masked before persistence.
    """
    rec = _record(action, actor, target, dict(fields))
    if not _try_db_insert(rec):
        _jsonl_append(rec)
    return rec
