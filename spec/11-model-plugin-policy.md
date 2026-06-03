# 11 — Model Plugin Policy (NVIDIA-only) + the plugin layer

> Companion to [10-platform-roadmap.md](10-platform-roadmap.md). Source: master plan
> [Addendum A](../oss_ingestion_nvidia_model_plugin_master_plan_v2.md) (A1–A13). This
> is the *durable model map* for the target platform; the roadmap is the build order.

## Why a plugin layer at all

The master plan's core thesis is **stable contracts ⇒ no refactor later**. Models
change often (new NVIDIA releases, new runtimes); business logic must not. So every
model sits behind a `ModelPlugin` ABC and a **task contract**, and a config-only
profile selects which model+runtime serves each task. Swapping a model is editing
YAML, never code (`MODEL_PROFILE=… make up`).

```
model family ≠ runtime ≠ business rule ≠ model prompt ≠ output schema   (A1.3)
e.g. model: nvidia/nemotron-nano-v2-vl · runtime: vllm/nim/triton/tensorrt_llm
     task: suspicious_event_verification · prompt: retail_loss_verifier_v3
     output schema: SuspiciousEventVerification
```

## The NVIDIA-only policy (A1) — and where Qwen goes

`configs/models.yaml` declares `model_policy: {vendor_scope: nvidia_only,
allow_non_nvidia_models: false, allow_vss_runtime_dependency: false,
vss_allowed_as_reference: true}`.

- **Allowed defaults (A1.1):** NVIDIA **Nemotron** (language/VL/Omni/retrieval/
  safety), **Cosmos** (video/world/physical-AI for the digital twin),
  **CV/Metropolis** (person/object detection, ReID, action, TAO/TensorRT), NVIDIA
  **embedding** (RADIO-CLIP/SigLIP-style, retrieval), and **safety/guard** models.
- **Disallowed as default (A1.2):** Qwen, InternVL, LLaVA, Gemma, Phi, DeepSeek,
  OpenAI, Gemini, Claude.

**The repo's current models are all disallowed-as-default** — Marlin (Qwen3-VL
based), Qwen2.5-VL ([03-models-and-query-modes.md](03-models-and-query-modes.md)),
YOLOE. So they become the `research_qwen_baseline` profile: **runnable,
comparison-only, `non_default: true`**. It is the fallback that keeps the platform
end-to-end testable *today* while NVIDIA weights are unavailable. The production +
benchmark default declares NVIDIA models, which may be **config-declared and
plugin-stubbed until weights land**. `make validate-model-config` enforces A5.2:
vendor scope nvidia_only; no non-NVIDIA default `model_id`; every task → a known
plugin; every plugin → declared runtimes + endpoint; output schema declared; metrics
enabled. This is the single highest-stakes decision in the whole platform — building
production directly on Qwen would require a rewrite when NVIDIA weights arrive; the
plugin indirection is the insurance.

## Plugin interface & task contracts (A2)

Every plugin implements `ModelPlugin` — `load(config)`, `infer(request)`,
`health()`, `metrics()`, `unload()` — and exactly one **task contract**:

| Contract | Input → Output |
|----------|----------------|
| `DetectionPlugin` | frames/clip → detections |
| `TrackingPlugin` | detections + frames → tracks |
| `EmbeddingPlugin` | clip/text/image/event → vector + metadata |
| `VLMClipReasoningPlugin` | clip + prompt + hypothesis → structured JSON |
| `SummarizationPlugin` | clips/events/time-range → timestamped summary |
| `SearchCriticPlugin` | query + candidates → reranked/critiqued results |

Folder layout (A10): `model-plugins/{base,deepstream_detector,deepstream_tracker,
nvidia_openai_compatible_vlm,nvidia_embedding,nvidia_summary,nvidia_search_critic,
nvidia_cosmos}/`. The `metrics.py` gauge-registry pattern already in `src/marlin`
seeds `model-plugins/base/metrics.py`.

## Registry, profiles, catalog (A2.3, A3, A5)

- **Profiles** (`configs/models.yaml`) bind a model+plugin+runtime per task. Two
  ship: `nvidia_a6000_retail_balanced` (vLLM/Triton, batch 8) and
  `nvidia_dgx_retail_high_throughput` (TensorRT-LLM/NIM, batch 32, concurrency 16/32),
  plus `research_qwen_baseline`.
- **Catalog** (`configs/model_catalog/nvidia_models.yaml`, A3) — versioned; each
  model declares modality/tasks/precision/runtimes/default_metrics. Seed entries:
  `nemotron-nano-v2-vl` (clip reasoning/verification), `nemotron-omni` (multimodal),
  `cosmos-3` (digital-twin simulation), `radio-clip-or-compatible` (embedding/search),
  `deepstream-detector-tao-compatible` (detection), `deepstream-reid-compatible` (ReID).

## The model-run audit trail (A6) — DB + dashboards

Beyond live metrics, **every** inference is persisted to `model_runs` and every
benchmark to `model_benchmark_runs` (master plan §6/A6 tables, indexed in Postgres —
see [10-platform-roadmap.md](10-platform-roadmap.md) Phase 2). Live metrics show
*now*; these tables show *across runs/profiles/GPUs*, making the Phase-13 NVIDIA
bake-off reproducible.

## Metrics & dashboards (A7/A8) — additive migration

The plugin layer adds the `model_*` Prometheus namespace (`model_requests_total`,
`model_latency_ms_bucket`, `model_ttft_ms_bucket`, `model_tokens_per_second`,
`model_clips_per_min_gpu`, `model_event_precision`, …) **alongside** the existing
`marlin_*` ([04-observability-stack.md](04-observability-stack.md)). Deprecate old
panels only after the new ones validate — a rename orphans Prometheus/Loki series and
Grafana regexes ([03-models-and-query-modes.md](03-models-and-query-modes.md) coupling
warning). New model-benchmark dashboards 11–18 (registry, throughput, latency,
quality, TCO, VSS-parity, failure/drift, runtime-comparison) live under
`observability/grafana/dashboards/model-benchmarks/`, each with an "About this
dashboard" intro panel.

## Plugin test gate (A11) — runnable without the real model

Each plugin passes 10 checks via `make test-plugin PLUGIN=<name>`: config-schema,
health, **fake inference**, real-sample inference, metrics emission, timeout, GPU-OOM,
bad-output, golden regression, benchmark-harness compat. The fake-inference + timeout
+ OOM + bad-output checks mean a plugin is testable **without** a GPU or real
weights — critical while NVIDIA Nemotron/Cosmos weights are absent. The
`research_qwen_baseline` plugin (wrapping `src/marlin/qwen_vl.py`) is the first to
pass all 10.

## Don't use the VLM for everything (master plan §9.2)

Detector + rules run on every window; **only suspicious windows** hit the VLM (empty
route ⇒ skip — the main compute saving). This is the cascade-gating thesis the
current pipeline already embodies
([05-performance-and-optimizations.md](05-performance-and-optimizations.md)), carried
into the plugin layer. VSS-style capabilities are implemented in OSS + NVIDIA models;
**VSS is a reference, never a runtime dependency** (`allow_vss_runtime_dependency:
false`).

## Related

[10-platform-roadmap.md](10-platform-roadmap.md) · [03-models-and-query-modes.md](03-models-and-query-modes.md) · [04-observability-stack.md](04-observability-stack.md)
