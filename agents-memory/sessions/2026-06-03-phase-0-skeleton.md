# 2026-06-03 — Phase 0: skeleton, contracts, NVIDIA model policy

## Goal

Execute Phase 0 of the platform roadmap (`spec/10`): repo skeleton, the model-plugin
contract, the NVIDIA-only model policy + validator, and the compose/Makefile stack —
plus fetch the free NVIDIA model shortlist. Branch `phase-0-skeleton`.

## Context

Box preflight (2026-06-03): GB10 aarch64, CUDA 13, ~1.1 TB free; **HF token works**
(free NVIDIA models pull now); **no NGC key** (`nvcr.io` 401 → DeepStream/TAO
deferred); Docker + nvidia-ctk installed but the **nvidia runtime isn't registered**.
Decision: production-default profile uses real free NVIDIA models; YOLOE is the free
interim CV path; generative runtime = Ollama/transformers (vLLM later).

## What was done

- **configs/** — `models.yaml` (NVIDIA-only policy + `nvidia_gb10_retail_balanced`
  default with real IDs + `research_qwen_baseline` non_default), `model_catalog/
  nvidia_models.yaml` (real free IDs), `cameras.yaml` (real 5-camera kathirmani),
  `stores/kathirmani.yaml`, `zones/kathirmani_zones.yaml` (placeholder polygons),
  `event_rules.yaml`, `retention.yaml` (PII/retention/evidence-lock).
- **model-plugins/base/** — `ModelPlugin` ABC (`load/infer/health/metrics/unload` +
  `fake_infer` for GPU-free tests), `schemas.py` (task contracts + §8 wire schemas),
  `metrics.py` (the `model_*` Prometheus namespace, A8).
- **scripts/** — `validate_model_config.py` (enforces A5.2; passes NVIDIA, **fails**
  non-NVIDIA default — both verified), `fetch_models.py` (trust-aware: pins revision
  sha, writes `models/PROVENANCE.json`), `setup_nvidia_docker.sh` (registers the
  Docker nvidia runtime, sudo, idempotent).
- **Stack** — `Makefile` (§18 targets; stubs marked per phase), `docker-compose.yml`
  (postgres pgvector / redis / minio), `.observability.yml` (wraps prometheus/grafana/
  loki/netdata + dcgm), `.gpu.yml` (ollama + worker slots), `.env.example`.
- **Models fetched + pinned:** `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` (17 GB,
  @437f4e28) + `nvidia/C-RADIOv4-H` (4 GB, @0057b339). Provenance in
  `models/PROVENANCE.json` (force-tracked; `models/` weights gitignored).
- **Security:** added `.env` to `.gitignore` (secrets never in repo).

## Update (later same day) — merged to main + operationalized

- **Merged `phase-0-skeleton` → main, pushed** (ff). All 3 models now fetched + pinned
  in `models/PROVENANCE.json`: Nemotron-Nano-VL-8B `@437f4e28`, C-RADIOv4-H `@0057b339`,
  **Cosmos-Reason2-2B `@9ce19a19`** (user accepted the HF license → 403 resolved).
- **Base services up:** `docker compose -f docker-compose.yml up -d` (postgres/redis/
  minio) — clean, no port clash with the running start_stack.sh observability stack.
- **New Grafana dashboard:** `grafana/dashboard_model_performance.json`
  (uid `marlin-model-perf`, "Model Performance & Usefulness") — 21 panels over the
  `marlin_*` metrics from the tested 5-camera scenarios; imported live into Grafana
  12.4.1. Fixed `grafana/setup_grafana.sh` (it imported a non-existent `dashboard.json`
  under `set -e`) to loop all `dashboard_*.json` via `/api/dashboards/db`. Documented
  in `spec/08-dashboards.md`.

## Not done / notes

- **`make setup-nvidia-docker`** still pending — needs **sudo** and this box has **no
  passwordless sudo**, so the user must run it (before GPU containers ollama/dcgm).
- **NGC-gated** DeepStream/NIM/TAO still blocked (no key). YOLOE/RT-DETR is the CV path.
- Cosmos downloaded but `digital_twin_simulation` stays `enabled: false` in the profile
  (Phase-10 wiring; model is ready).
- Verified: `make lint` / `config-check` / `docker-config` pass; validator negative test
  passes. No worker/DB code yet — Phase 1/2.

## Status

**Phase 0 complete + on main.** Next: user runs `make setup-nvidia-docker`; then
Phase 1 (ingestion) — wrap the 5 `.mkv` into 10-sec clips + 5-sec windows → MinIO +
Postgres + Redis.
