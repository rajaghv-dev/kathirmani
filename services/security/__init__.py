"""Phase 12 security/hardening building blocks (master plan §12 + spec/10 Security).

Standalone, importable modules — the orchestrator wires them into the API later:
  - auth.py      Bearer/API-key authentication (FastAPI dependency)
  - rbac.py      viewer|reviewer|admin roles + least-privilege matrix
  - audit.py     append-only audit log (DB table or JSONL fallback), secret-safe
  - redaction.py mask secrets/creds for safe logging
  - retention.py enforce configs/retention.yaml (dry-run by default, evidence-lock aware)
  - backup.py    pg_dump + clips-manifest backup helper

Nothing here imports the API app or mutates DB schema; persistence is best-effort
and degrades gracefully with no DB (so unit logic needs no services).
"""
