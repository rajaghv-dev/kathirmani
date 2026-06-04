# Kathirmani Marlin — Spec Documentation

This spec/ directory is the reorganized, structured documentation for the Kathirmani Marlin video-intelligence pipeline. It is derived from learning.md (kept at repo root as the original working notes).

> **Note:** [06-hardware-portability.md](06-hardware-portability.md) and [09-repo-structure.md](09-repo-structure.md) cover the runtime device-detection design and the `src/` package layout introduced in the refactor.

## Contents

| File | Description |
|---|---|
| [01-overview.md](01-overview.md) | What the system is — Marlin-2B across 5 store cameras, caption + find modes, what it extracts, where results land. |
| [02-architecture.md](02-architecture.md) | The three perception axes (when/where/what), the ThreadPoolExecutor + single-GPU lock reality, and the "databases" (Prometheus, Loki, results JSON). |
| [03-models-and-query-modes.md](03-models-and-query-modes.md) | Marlin caption vs find, temporal grounding, good-query rules, the redesigned atomic query set, the coupling warning, and Qwen2.5-VL's role. |
| [04-observability-stack.md](04-observability-stack.md) | Netdata runtime metrics, nvidia-smi GPU power, NVML compute util, and the model-caching gotcha. |
| [05-performance-and-optimizations.md](05-performance-and-optimizations.md) | Why it's slow despite big memory, the seven bottlenecks, the decode-once refactor, and the Tier 1/2/3 optimization roadmap. |
| [06-hardware-portability.md](06-hardware-portability.md) | Runtime accelerator detection — one checkout across NVIDIA DGX (CUDA), Apple Mac Studio (MPS), and CPU; how the loaders, metrics, and scripts adapt. |
| [07-runbook.md](07-runbook.md) | How to run and view — setup, run_inference.py flags, outputs, and the Streamlit viewer. |
| [08-dashboards.md](08-dashboards.md) | What each of the 7 Grafana dashboards means, the shared vocabulary, and the telemetry-preflight troubleshooting note. |
| [09-repo-structure.md](09-repo-structure.md) | The `src/marlin` package layout, the root entry-point shims, path anchoring, and how every command stays unchanged. |
| [10-platform-roadmap.md](10-platform-roadmap.md) | **Forward-looking.** The OSS-ingestion + NVIDIA-model platform the repo is heading toward (master plan v2): current-vs-target gap, the 14-phase build, critical path, anti-goals, acceptance. |
| [11-model-plugin-policy.md](11-model-plugin-policy.md) | **Forward-looking.** The NVIDIA-only model policy + the `ModelPlugin` layer, registry/profiles/catalog, `model_runs` audit, additive `model_*` metrics, and why Qwen is comparison-only. |
| [12-frameworks.md](12-frameworks.md) | Frameworks per component (the `requirements/` split) + how each is set up (`scripts/setup/`) and validated (`scripts/validate/` + `make validate`/`doctor`). |
| [13-models.md](13-models.md) | Consolidated model map: current cascade + the free NVIDIA lineup (real IDs), provenance pinning, the config-driven runtimes, and per-inference tracking. |
