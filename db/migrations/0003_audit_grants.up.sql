-- Phase 12 — enforce the append-only posture of audit_log at the DB privilege layer
-- (spec/10 Security: tamper-evident chain-of-custody). 0002 created the table and
-- noted "enforce via GRANTs in prod"; this is that enforcement.
--
-- We introduce a NOLOGIN *group* role `kathirmani_app` that the application's login
-- role(s) join (GRANT kathirmani_app TO <login_role>). The group holds exactly
-- INSERT + SELECT on audit_log and is explicitly denied UPDATE/DELETE/TRUNCATE, so
-- a compromised app credential can append and read history but can never rewrite or
-- erase it. Roles are cluster-global, so creation is guarded to stay idempotent.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kathirmani_app') THEN
    CREATE ROLE kathirmani_app NOLOGIN;
  END IF;
END
$$;

-- Append-only: grant only INSERT + SELECT; never UPDATE/DELETE.
-- (audit_log.id is GENERATED ALWAYS AS IDENTITY → its implicit sequence is covered
-- by the table INSERT privilege; no separate sequence grant needed.)
REVOKE ALL ON audit_log FROM kathirmani_app;
GRANT INSERT, SELECT ON audit_log TO kathirmani_app;

-- Belt-and-suspenders: nobody mutates history via the PUBLIC pseudo-role either.
REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM PUBLIC;

-- To wire an app login role in prod (run once, outside migrations as it depends on
-- your deployment's login role name):
--   GRANT kathirmani_app TO <app_login_role>;
-- and connect the audit writer (services/security/audit.py) as that login role.
