# 13 — Models (catalog, provenance, runtimes)

The consolidated model map: what models the platform uses, where they come from,
how they're pinned, and which runtime serves each. Complements
[03-models-and-query-modes.md](03-models-and-query-modes.md) (the current cascade's
query design) and [11-model-plugin-policy.md](11-model-plugin-policy.md) (the
NVIDIA-only policy + the `ModelPlugin` layer). Catalog file:
`configs/model_catalog/nvidia_models.yaml`; pinned weights: `models/PROVENANCE.json`;
DB registry: `model_registry` (seeded by `make seed`).

## Two tiers

**Current cascade (research baseline).** The runnable `src/marlin` pipeline —
Marlin-2B (WHEN) → Locate/YOLOE (WHERE) → Qwen2.5-VL (WHAT). All Qwen/YOLOE-based,
so under the NVIDIA-only policy they live as the comparison-only
`research_qwen_baseline` profile (never the production default). Detail: `spec/03`.

**Production / benchmark default — free, open NVIDIA models (on HuggingFace).**
Verified + fetched + pinned 2026-06-03/04:

| Task | Model | Revision | Runtime (this box) |
|------|-------|----------|--------------------|
| VLM clip-reasoning | `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` | `437f4e28` | transformers / Ollama |
| Summary · search-critic | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (or `-4B`) | — | Ollama (GGUF) |
| Visual embedding / search | `nvidia/C-RADIOv4-H` | `0057b339` | transformers (Triton later) |
| Digital-twin reasoning | `nvidia/Cosmos-Reason2-2B` | `9ce19a19` | transformers |
| Detection (free, no NGC) | YOLOE / `PekingU/rtdetr_v2_r50vd` | — | ultralytics / transformers |

Bigger/alternate variants and the NGC-gated TAO detectors are catalogued in
`configs/model_catalog/nvidia_models.yaml`.

## Provenance & trust

Models are fetched by `scripts/fetch_models.py`, which **pins each to its exact
commit sha** and records `repo_id`, `revision`, `url`, `license` in
`models/PROVENANCE.json` (part of the supply-chain trust posture — `spec/10`
Security). `make seed` loads the catalog + pinned revisions into the `model_registry`
table, so every model on disk is auditable from the DB. Weights live under `models/`
(gitignored); only the provenance manifest is tracked.

## Runtimes — config-driven, swappable

Each task in `configs/models.yaml` binds a model + plugin + **runtime**; switch the
whole set with `MODEL_PROFILE=<name> make up` (no code change — the `ModelPlugin`
contract, `spec/11`). On the GB10 (aarch64, Blackwell, CUDA 13) the default
generative runtime is **Ollama/llama.cpp/transformers**; **vLLM / TensorRT-LLM / NIM**
are the throughput targets to attempt in the Phase-13 bake-off (prebuilt aarch64
Blackwell support is unproven). NGC-hosted runtimes (Triton images, NIM, DeepStream)
need an **NGC API key** the box doesn't have yet.

## Acquisition & validation

- Fetch: `make fetch-models` (or `bash scripts/setup/models.sh --fetch`).
- Validate: `make validate-models` checks the NVIDIA-only policy (hard), provenance
  pinning, and that the weights are on disk (warn if absent). The policy gate
  (`scripts/validate_model_config.py`) **fails** a non-NVIDIA default.

## Per-inference tracking

Every model call is (Phase 6+) written to `model_runs` (latency, ttft, tokens,
parse_success, quality) and every benchmark to `model_benchmark_runs` — what the
model dashboards (11–18, `spec/11`/master plan A7) query for across-run comparison.

## Related

[03-models-and-query-modes.md](03-models-and-query-modes.md) · [11-model-plugin-policy.md](11-model-plugin-policy.md) · [12-frameworks.md](12-frameworks.md) · [10-platform-roadmap.md](10-platform-roadmap.md)
