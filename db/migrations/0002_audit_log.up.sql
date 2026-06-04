-- Phase 12 — append-only audit log (security & trust). Used by services/security/audit.py
-- (falls back to JSONL until this is applied). Append-only posture: app role gets
-- INSERT/SELECT only (no UPDATE/DELETE) — enforce via GRANTs in prod.
CREATE TABLE audit_log (
  id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
  action  TEXT NOT NULL,
  actor   TEXT NOT NULL,
  target  TEXT,
  fields  JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX idx_audit_action_ts ON audit_log (action, ts DESC);
