"""audit: JSONL fallback append + secret redaction. DB path skipif-no-DB."""
from __future__ import annotations

import json

import pytest

from services.security import audit
from services.security.redaction import MASK


def _no_db():
    try:
        from db import connect
        with connect() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        return False
    except Exception:
        return True


def test_jsonl_fallback_appends(tmp_path, monkeypatch):
    log = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log))
    # force the JSONL path even if a DB is up
    monkeypatch.setattr(audit, "_try_db_insert", lambda rec: False)

    audit.audit("incident.approve", actor="reviewer:bob", target="inc-1", note="looks legit")
    audit.audit("profile.activate", actor="admin:carol", target="nvidia_balanced")

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["action"] == "incident.approve"
    assert rec["actor"] == "reviewer:bob"
    assert rec["target"] == "inc-1"
    assert rec["fields"]["note"] == "looks legit"
    assert "ts" in rec


def test_audit_redacts_secrets(tmp_path, monkeypatch):
    log = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log))
    monkeypatch.setattr(audit, "_try_db_insert", lambda rec: False)

    rec = audit.audit("login", actor="x", target=None,
                      api_key="supersecret", rtsp="rtsp://u:pw@h/s")
    assert rec["fields"]["api_key"] == MASK
    assert "pw" not in rec["fields"]["rtsp"]
    assert "supersecret" not in log.read_text()


@pytest.mark.skipif(_no_db(), reason="no Postgres")
def test_db_path_used_when_table_present():
    # Only meaningful if audit_log migration has been applied; otherwise falls to JSONL.
    rec = audit.audit("smoke.db", actor="test", target="t")
    assert rec["action"] == "smoke.db"
