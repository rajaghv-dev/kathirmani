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

## Not done / notes

- **`nvidia/Cosmos-Reason2-2B` → 403 gated.** `gated=auto` did not auto-grant; user
  `rajaghv-dev` must click "Agree and access" on the HF page, then
  `python scripts/fetch_models.py nvidia/Cosmos-Reason2-2B`. Not on the critical path
  (digital-twin = Phase 10, `enabled: false` in the active profile).
- **NGC-gated** DeepStream/NIM/TAO still blocked (no key). YOLOE/RT-DETR is the CV path.
- **`make setup-nvidia-docker`** not yet run (needs sudo) — required before GPU
  containers (ollama/dcgm).
- Verified: `make lint`, `make config-check`, `make docker-config` all pass; validator
  negative test passes. No worker/DB code yet — those are Phase 1/2.

## Status

**Phase 0 skeleton complete on branch `phase-0-skeleton`** (not pushed). Next:
run `make setup-nvidia-docker`, `make up` to bring the base+observability stack green,
then Phase 1 (ingestion).
