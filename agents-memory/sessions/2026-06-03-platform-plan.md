# 2026-06-03 — Platform roadmap: master plan v2 → spec/10 + spec/11

## Goal

Turn the master plan `oss_ingestion_nvidia_model_plugin_master_plan_v2.md` (a
3.5k-line vision for an OSS-ingestion-first, NVIDIA-model-first, plugin-driven
video intelligence platform) into an **executable, repo-grounded plan** an agent
can follow — without throwing away the current `src/marlin` cascade. Planning/docs
only; **no platform code written**.

## Context

A git divergence surfaced mid-task: a local line had refactored into
`inference/core/stages/obs/` + drafted the platform plan against that layout, while
`origin/main` had independently refactored into the **`src/marlin` package**
(+ device auto-detection, compressed `agents-memory/`, numbered `spec/`). Confirmed
the remote `src/marlin/pipeline.py` already contains the locate/decode-once/fusion
work — the divergence was **organizational, not feature loss**. Per the user's
call, reset local `main` to `origin/main` (canonical) and re-authored the platform
plan onto the remote's structure (numbered spec docs, `src/marlin` references).

## What was done

- **`spec/10-platform-roadmap.md`** — current-vs-target gap table, the 14-phase
  build (0–13) adapted to this repo (`marlin.locate` → cv-oss-worker;
  `qwen_vl.py` → `research_qwen_baseline` VLM plugin; `metrics.py` → `model_*`),
  critical path, anti-goals, whole-platform acceptance.
- **`spec/11-model-plugin-policy.md`** — the NVIDIA-only model policy + `ModelPlugin`
  layer, task contracts, registry/profiles/catalog, `model_runs` audit, additive
  `model_*` metrics, the 10-point plugin test, and the Qwen-as-comparison-only
  reconciliation.
- **`spec/README.md`** — Contents table extended with 10 + 11 (flagged
  forward-looking).
- **Master plan source** restored to repo root as the full-detail reference the
  two new docs point at.
- **`agents-memory/memory.md`** — added a terse Roadmap pointer + the NVIDIA-only
  invariant (links to spec/10, spec/11).

## Not done / notes

- **No platform code** — Makefile, compose, `configs/`, DB, plugins, services all
  still to build (Phase 0+). Docs `01`–`09` remain the current-system truth.
- **Key locked decision:** master plan mandates NVIDIA-only defaults but the whole
  current cascade (Marlin Qwen3-VL / Qwen2.5-VL / YOLOE) is disallowed-as-default →
  kept as a runnable comparison-only `research_qwen_baseline` profile behind the
  plugin layer; NVIDIA models (possibly stubbed) are the production default. The
  plugin indirection is what makes the later swap a config change, not a rewrite.
- The local `inference/core/stages/obs` refactor was **dropped** (superseded by the
  canonical `src/marlin` package); only the net-new platform-plan docs were kept.

## Status

**Docs complete, pushed.** Next: Phase 0 when greenlit — Makefile, 3 compose files
(base/gpu/observability wrapping the existing stack), `configs/`, the `ModelPlugin`
interface, NVIDIA model policy + `validate-model-config`.
