# 12 — Frameworks & per-component setup/validation

How the platform's dependencies are organized and how each component is **set up**
and **validated**. Frameworks are declared per component in `requirements/`, set up
by `scripts/setup/<component>.sh`, and verified by `scripts/validate/<component>.py`
(aggregated by `make validate` / `doctor`). This keeps "what each layer needs" and
"is it correctly installed/running" explicit and checkable.

## The map

| Component | Frameworks (why) | requirements | setup | validate |
|-----------|------------------|--------------|-------|----------|
| **base / env** | Python ≥3.12; `prometheus_client`, `requests`, `pyyaml`, `numpy`, `pillow` — shared everywhere | `base.txt` | `setup/env.sh` | `validate/env.py` |
| **inference** | `torch`, `transformers`, `accelerate`, `qwen-vl-utils`, `torchcodec`, **PyAV** — the `src/marlin` cascade (heavy, GPU box) | `inference.txt` | `setup-inference` (`setup.sh`, machine-aware torch) | (marlin `tests/`) |
| **ingestion** | **PyAV** (libav, bundled FFmpeg) for file segmentation; **GStreamer** for live RTSP (`splitmuxsink`) + frame-gating | `ingestion.txt` | `setup/ingestion.sh` | `validate/ingestion.py` |
| **db** | **Postgres 16 + pgvector** (the single structured store: metadata, events, vectors, model_runs, queue); **psycopg v3** | `db.txt` | `setup/db.sh` (migrate + seed) | `validate/db.py` |
| **queue** | **Postgres `job_queue`** (`SKIP LOCKED`); **Redis** optional (high-fanout) | `db.txt` / `optional.txt` | (db) | (db) |
| **storage** | **local NVMe** (clip blobs); **MinIO** optional (S3/multi-node) | — / `optional.txt` | — | (ingestion) |
| **api** | **FastAPI** + **uvicorn** — platform + model-registry endpoints | `api.txt` | `setup/api.sh` | `validate/api.py` |
| **models** | **huggingface_hub** (fetch + pin); runtimes **Ollama/llama.cpp/transformers** now, **vLLM/Triton/NIM/TensorRT-LLM** later | `models.txt` | `setup/models.sh` (`FETCH=1` to download) | `validate/models.py` |
| **observability** | **Prometheus · Grafana · Loki · Netdata · DCGM · grafana-image-renderer** (Docker services) | `observability.txt` (Docker, not pip) | `setup/observability.sh` | `validate/observability.py` |
| **gpu / runtime** | NVIDIA driver/CUDA, **Docker**, **NVIDIA Container Toolkit** | — (system) | `setup/gpu.sh` (sudo) | `validate/gpu.py` |
| **dev / docs** | `pytest`, `matplotlib` (infographic figures) | `dev.txt` | (env) | — |
| **infra** | **Docker Compose** (3 files), **Makefile** | — | — | `make docker-config` |

`requirements.txt` at the root simply `-r`'s the per-component files, so
`pip install -r requirements.txt` still works; install one layer with
`pip install -r requirements/<component>.txt`.

## Setup

- **All Python deps:** `make setup` → `scripts/setup/all.sh` (env + ingestion +
  models + api). Heavy inference (`torch`) is separate: `make setup-inference`.
- **Per component:** `make setup-<component>` (env / ingestion / db / models / api /
  observability / gpu). Service/sudo steps (`db`, `observability`, `gpu`,
  `models --fetch`) are never auto-run by `all.sh` — they're called out explicitly.
- **`.env`** is created from `.env.example` by `setup/env.sh`; it is gitignored
  (secrets never in the repo — spec/10 Security).

## Validation — `make validate` (a.k.a. `doctor`)

`scripts/validate/doctor.py` runs every component validator and prints a status
matrix. Convention: a **missing framework (import fails) is a hard FAIL**; a
**service that isn't running / not yet built is a WARN** (so a dev box without
Postgres/GPU still passes). Run one component with `make validate-<component>` or
`python scripts/validate/<component>.py`.

```
OK     env            7 ok, 0 warn, 0 fail
OK     ingestion      4 ok, 0 warn, 0 fail
OK     db             4 ok, 0 warn, 0 fail
OK     models         6 ok, 0 warn, 0 fail
OK     observability  6 ok, 0 warn, 0 fail
WARN   api            (FastAPI installed; app/endpoint Phase 2)
WARN   gpu            (Docker NVIDIA runtime not registered — make setup-gpu)
```

## Deliberate choices

- **PyAV, not system FFmpeg** — bundled FFmpeg libs, no apt install; GStreamer is
  added only for the live/HW path (`spec/10`, `ingestion/sources.py`).
- **Postgres + filesystem, not Redis/MinIO** by default (`spec/10` Data stack).
- **Ollama/transformers before vLLM** on aarch64/Blackwell (`spec/11`).
- **Off-the-shelf observability** (Prometheus/Grafana/Loki) so the operator surface
  is standard (`spec/04`, `spec/08`).

## Related

[09-repo-structure.md](09-repo-structure.md) · [10-platform-roadmap.md](10-platform-roadmap.md) · [13-models.md](13-models.md) · [04-observability-stack.md](04-observability-stack.md)
