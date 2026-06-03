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

The policy is also a **trust/supply-chain** decision: a single, vetted vendor scope
narrows model provenance, licensing, and safety review (pin
`model_registry.source_url` + `license_notes`; pair with NVIDIA safety/guard models
for output validation). Model trust is part of the platform's cross-cutting security
posture — see [10-platform-roadmap.md](10-platform-roadmap.md) "Security & trust".

**The repo's current models are all disallowed-as-default** — Marlin (Qwen3-VL
based), Qwen2.5-VL ([03-models-and-query-modes.md](03-models-and-query-modes.md)),
YOLOE. So they become the `research_qwen_baseline` profile: **runnable,
comparison-only, `non_default: true`**. It is the fallback that keeps the platform
end-to-end testable *today* and as the comparison arm of the bake-off. The
production + benchmark default declares **real, free NVIDIA models** that are
**acquirable now** from HuggingFace (see the selected-models table below); only
NGC-hosted assets (DeepStream/NIM/TAO) remain stubbed until an NGC key exists. `make validate-model-config` enforces A5.2:
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
  model declares modality/tasks/precision/runtimes/default_metrics + provenance.

### Selected free/OSS NVIDIA models (real IDs, verified on HF 2026-06-03)

The master plan's `…-or-compatible` placeholders are resolved to **actual,
free, open** NVIDIA models on HuggingFace — all readable with the repo's HF token
(`gated=False`, Cosmos `gated=auto` auto-grants). Per the directive "use free/OSS
NVIDIA, check HF equivalents":

| Task | NVIDIA model (free, HF) | Runtime on this box | HF equivalent / quant |
|------|-------------------------|---------------------|------------------------|
| VLM clip-reasoning | `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` | Ollama/llama.cpp (GGUF) or transformers | `forkjoin-ai/…-vl-8b-v1-gguf`; bigger: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-{BF16,FP8,NVFP4-QAD}` |
| Summary / search-critic (LLM) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (MoE, ~3B active) or `…-Nano-4B` | Ollama/llama.cpp (GGUF) | `unsloth/Nemotron-3-Nano-30B-A3B-GGUF`, `lmstudio-community/…-4B-GGUF` |
| Visual embedding / search | `nvidia/C-RADIOv4-H` | transformers/PyTorch (Triton later) | `nvidia/RADIO-L`, `nvidia/C-RADIOv3-H` |
| Digital-twin reasoning | `nvidia/Cosmos-Reason2-2B` (gated=auto) | transformers/PyTorch | `nvidia/Cosmos-Reason2-8B`, `nvidia/Cosmos-Embed1-448p-anomaly-detection` |
| CV detection (free, **no NGC**) | YOLOE (already local) or `PekingU/rtdetr_v2_r50vd` (Apache-2.0) | existing locate stage | TAO PeopleNet/ReID later — **needs NGC key** |

### Acquisition & runtime on this box (GB10 aarch64, Blackwell, CUDA 13)

- **HF token works** → the VLM/LLM/embedding/Cosmos models above download **now**;
  they are **not stubbed**. Only NGC-hosted assets (DeepStream SDK, NIM, TAO CV
  models, NGC Triton images) need an **NGC API key** (`nvcr.io` returns 401 without
  one) — those stay deferred; YOLOE/RT-DETR is the free CV path meanwhile.
- **Default runtime = Ollama / llama.cpp (GGUF)** for the generative plugins, not
  vLLM: prebuilt vLLM on aarch64 + Blackwell + CUDA 13 is unproven and may need a
  source build, while Ollama/llama.cpp run GGUF portably here. vLLM/TensorRT-LLM
  stay the throughput target to attempt in the Phase-13 bake-off. RADIO + Cosmos run
  directly under transformers/PyTorch (the cascade's existing CUDA path).
- **Docker GPU containers** (NIM/Triton/DeepStream) need the NVIDIA runtime
  registered first: `nvidia-ctk runtime configure --runtime=docker` + daemon restart
  (installed but not yet wired — a Phase-0 host step).
- **Supply-chain trust** (the cross-cutting security posture,
  [10-platform-roadmap.md](10-platform-roadmap.md)): pin each model to an exact
  **revision (commit sha)**, record `source_url` + `license` + checksum in
  `model_registry`, and accept licenses deliberately (esp. Cosmos `gated=auto`).

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
