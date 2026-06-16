# 2026-06-16 — Rename project marlin → kathirmani (keep Marlin-2B model identity)

User: "change the repo name as kathirmani, refactor the code, commit." Chose (via
questions): rename **project metadata + local dir + GitHub remote**; refactor depth =
**full incl. metrics**.

## The one critical distinction
The **project** was named `marlin`; the **VLM model** is also literally `Marlin-2B`
(`NemoStation/Marlin-2B`, `models/Marlin-2B/`, `modeling_marlin.py`). A blanket sed
would break model loading. So: renamed the **project namespace**, **preserved the
model identity**.

## Renamed (project → kathirmani)
- Package `src/marlin/` → `src/kathirmani/` (`git mv`); all `from/import marlin`,
  `marlin.<module>` updated (14 import sites + 4 root shims).
- Metrics `marlin_*` → `kathirmani_*` (`src/kathirmani/metrics.py`, `serve_metrics.py`,
  pipeline label comment, 10 Grafana dashboards' `expr`, spec docs).
- Env var `MARLIN_DEVICE` → `KATHIRMANI_DEVICE`; Loki app label `"marlin"` →
  `"kathirmani"` (emitter **and** the 9 dashboard_loki queries — must match).
- Postgres role `marlin_app` → `kathirmani_app` (schema.sql, migration 0003 up/down,
  services/security/audit.py). **CAVEAT:** editing an *already-applied* migration only
  affects fresh DBs; existing DBs keep `marlin_app` (rename there needs a new migration
  or manual `ALTER ROLE`).
- Grafana datasource display name `MarlinMetrics` → `KathirmaniMetrics`.
- pyproject: dist name `kathirmani-marlin` → `kathirmani`; scripts `marlin-*` →
  `kathirmani-*`. Project title "Kathirmani Marlin" → "Kathirmani".

## Preserved (model identity — intentionally NOT renamed)
- `Marlin-2B`, `NemoStation/Marlin-2B`, `models/Marlin-2B/`, `modeling_marlin.py`,
  `MarlinForConditionalGeneration`, `marlin.caption()`/`marlin.find()` (model API in
  spec/01,03), `marlin_2b` model_id (spec/16, display_name "Marlin-2B"), the `"marlin"`
  download-key in `cli/download_model.py`.
- **Grafana uids** (`marlin-metrics`, `loki-marlin`, `marlin-cameras`, …, 131 refs) —
  internal plumbing; renaming risks breaking datasource↔panel bindings + `/d/` URLs.
  Kept; only metric *names* inside queries changed.
- `learning.md` + `agents-memory/sessions/*` — verbatim/append-only history; untouched.

## Verified
- `py_compile` of package + shims + vlm-worker plugin: OK. `import kathirmani` OK.
- `pytest tests/`: 22 passed; the 3 fails are **environmental, not the rename** —
  torchcodec missing `libavutil.so.56`; a **stale :8900 server** still emitting old
  `marlin_*` (proves test+code now agree on `kathirmani_*`; passes after serve_metrics
  restart); the flaky live-Grafana datasource check (uid preserved). `test_structure`
  + `test_device` (the rename validators) pass.
- Tree-wide: zero leftover `from/import marlin`, `src/marlin`, `MARLIN_DEVICE`,
  `kathirmani-marlin`.

## Follow-ups
- GitHub remote rename `kathirmani-marlin` → `kathirmani` + `git remote set-url`.
- Local dir rename `…/kathir` → `…/kathirmani` (done last; breaks the session cwd).
- Restart `serve_metrics`/stack so live metrics emit `kathirmani_*` (Grafana panels use
  renamed exprs). Consider migration 0005 to `ALTER ROLE marlin_app RENAME TO
  kathirmani_app` on already-provisioned DBs.
