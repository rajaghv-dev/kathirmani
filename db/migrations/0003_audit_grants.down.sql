-- Roll back the audit_log append-only GRANTs. We drop the privileges and the group
-- role; we do NOT restore UPDATE/DELETE to PUBLIC (default for a fresh table is no
-- PUBLIC table privileges anyway, so the REVOKE is a no-op to undo).
REVOKE ALL ON audit_log FROM marlin_app;
-- Drop the role only if no login role still depends on (is a member of) it.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'marlin_app') THEN
    DROP ROLE marlin_app;
  END IF;
END
$$;
