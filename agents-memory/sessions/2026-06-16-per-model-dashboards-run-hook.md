# 2026-06-16 — Per-model dashboards + model-run hook (Netdata runtime attribution)

User: "have a dashboard for every model used, and link its runtime metrics from netdata,
thru hooks developed somehow whenever it gets kicked off kind of."

## What I built
- **`model-plugins/base/run_hooks.py`** (new) — `record_model_run(config, t0, t1, model_run)`
  + `model_run(config)` context manager. Fires on each model kickoff/finish:
  - range-queries **Netdata** `/api/v1/data?after&before` over [t0,t1] for system.cpu /
    system.ram / system.io; samples nvidia-smi for GPU power.
  - sets `model_resource_{cpu_percent,ram_used_mb,io_read_kbps,io_write_kbps,
    gpu_power_watts,run_seconds}` + `model_runs_total`, labelled by the _COMMON set
    (model_id/task/runtime/profile).
  - pushes a **Grafana region annotation** (time+timeEnd) tagged `["kathir","model-run",
    model_id]`.
  - **best-effort, never raises**; opt-out via `KATHIRMANI_RUN_HOOK=0`. No-ops cleanly
    when Netdata/Grafana/nvidia-smi absent (verified on this CPU box).
- **`model-plugins/base/metrics.py`** — added the `model_resource_*` gauges + `model_runs_total`.
- **Wired into the two infer choke-points** (additive, try/except-guarded so a hook
  failure can't break inference): `ai-workers/cv-oss-worker/worker.py:process_window`,
  `ai-workers/vlm-worker/worker.py:process_event`.
- **`observability/grafana/make_dashboards.py`** — `model_inventory()` (reads catalog +
  non_nvidia_exceptions + active profiles, deduped, static fallback) + `model_dashboard()`;
  `build()` gained an optional `annotations=` arg. `main()` now also writes one dashboard
  per model to `dashboards/models/<slug>.json`. Per-model panels: model_* traffic/latency/
  quality filtered `{model_id="…"}` + the Netdata `model_resource_*` runtime panels + the
  model-run tag annotation on the timeline.
- **`observability/README.md`** import loop now also globs `dashboards/models/*.json`.
- **spec/08** — documented the per-model dashboards + run hook. New test
  `test_per_model_dashboards` in `observability/tests/test_dashboards.py`.

## Verified
- Generator: 18 platform + **9 per-model** dashboards, all valid JSON; per-model carry
  the model-run annotation + a `model_resource_*` panel.
- `observability/tests` 5 passed; `cv-oss-worker` 23 passed (exercises hooked
  process_window). Hook smoke test: `record_model_run` sets run_seconds + runs_total,
  no exception with no stack.

## Design notes / follow-ups
- Attribution is **time-window correlation** (model kickoff→finish window → Netdata
  range), not per-process cgroup accounting — Netdata is host-level. Good enough for the
  "what did this model cost the box while running" view; per-process would need cgroup/
  pid metrics.
- GPU power is a point sample (nvidia-smi has no cheap range query); fine since runs are
  short. For long runs, consider a sampler thread like the legacy `poll_dgx_metrics`.
- embedding-worker not wired (no model_runs path); add if its runs should appear.
- `model_resource_*` are last-value gauges (overwritten each run); the annotation
  timeline is what gives per-run history. A histogram/series per run would need a push
  with timestamp or a DB column.
