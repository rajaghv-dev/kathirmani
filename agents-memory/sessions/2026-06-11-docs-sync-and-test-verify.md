# 2026-06-11 — Spec/docs sync + full-suite re-verify

User: "update the spec in detail and other md files in sync with this repo and results."
No code changed — doc-only sync pass against the actual tree + a real `make test` run.

## What I verified (the "results" half)
- Ran every `TESTDIRS` component (per-dir, like `make test`). All green:
  **299 passed, 1 deselected** (the live-Grafana datasource check). Per-dir:
  tests 24 · ingestion 24 · api 4 · digital-twin 5 · rule-engine 24 · evidence-builder 12 ·
  review-ui 12 · console 5 · security 38 · cv-oss 23 · vlm 23 (slow: ~7.5min, heavy import) ·
  embedding 29 · vss-eval 24 · benchmarks 28 · bakeoff 20 · observability 4.
- **Lesson: trust pytest counts, not `grep -c "def test_"`** — parametrize expands them
  (e.g. benchmarks grepped 23 but pytest reports 28). The per-phase spec numbers were
  written from real runs, so several were already right; only the drifted ones were fixed.
- `benchmarks/results/*.json` are **gitignored** (only `.gitkeep` tracked) — the spec's
  "synthetic/fake-mode" framing is correct; did NOT bake fake p50/TCO numbers into docs.
  `results/*.json` (legacy cascade: 5 videos, 9 events, economy ₹0.0091/event) ARE committed.

## Drift fixed (concrete)
- Dashboard count **7→8** (the model-perf dashboard exists): `spec/README.md`,
  `spec/08-dashboards.md` ("## The N dashboards"), `spec/10` gap-table "today" column.
- Per-phase test counts in `spec/10` table → actual pytest: Phase 4 **14→23**, Phase 5
  **22→24**, Phase 6 **22→23**, Phase 8 **26→29**. (Phases 7/9/10/11/12/13 already matched.)
- Added the verified full-suite total (**299 passed, 1 deselected**, 2026-06-11) to
  `spec/10` What's-to-be-done intro + `spec/09` Tests section.
- `agents-memory/memory.md` phases line: `make test` 264 → **299 passed, 1 deselected**.

## Audited as already in-sync (no change needed)
- spec 01–08 (legacy cascade — still accurate as the comparison baseline), 09, 11, 12,
  13, 14; root `README.md` + `PLATFORM.md` (Console :8080 / ports / `make platform` all
  current). spec/03 coupling warning (FIND_QUERIES ↔ Prometheus labels/Loki/qwen dashboard
  regex) confirmed still load-bearing — leave intact.
- Blocked-on-infra/user items unchanged (sudo nvidia-docker, NGC key, real-GPU numbers,
  `make migrate` per-deploy) — see [2026-06-10](2026-06-10-pending-followups-and-console.md).
