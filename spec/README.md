# Kathirmani Marlin — Spec Documentation

This spec/ directory is the reorganized, structured documentation for the Kathirmani Marlin video-intelligence pipeline. It is derived from learning.md (kept at repo root as the original working notes).

> **Note:** docs `01`–`08` describe the legacy Marlin research cascade; `09`–`14`
> describe the **platform** the repo became; `15`–`17` are the **v3 capability-hook
> layer** (forward-looking — plug-and-play models + image-quality gate + POS). Start at
> [09-repo-structure.md](09-repo-structure.md) (repo map),
> [14-console-and-deployment.md](14-console-and-deployment.md) (the front door + one-command launch),
> and [16-capability-hooks-profiles-router.md](16-capability-hooks-profiles-router.md)
> (where the system is heading).
>
> **Design statement (master prompt v3).** Frontier models are not the product — the
> product is a disciplined evidence pipeline. *Video AI detects visible candidate events ·
> image-quality gates improve model reliability · POS checks transaction context · rules
> prioritize review · humans decide · observability proves cost, speed, and quality.*
> Better frames, atomic prompts, cheap-models-first, selective VLM routing, POS
> correlation, rules, human review, and observability make **lower-cost models useful
> enough for real retail operations**.

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
| [08-dashboards.md](08-dashboards.md) | What each of the 8 Grafana dashboards means, the shared vocabulary, and the telemetry-preflight troubleshooting note. |
| [09-repo-structure.md](09-repo-structure.md) | The `src/marlin` package layout, the root entry-point shims, path anchoring, and how every command stays unchanged. |
| [10-platform-roadmap.md](10-platform-roadmap.md) | **Forward-looking.** The OSS-ingestion + NVIDIA-model platform the repo is heading toward (master plan v2): current-vs-target gap, the 14-phase build, critical path, anti-goals, acceptance. |
| [11-model-plugin-policy.md](11-model-plugin-policy.md) | **Forward-looking.** The NVIDIA-only model policy + the `ModelPlugin` layer, registry/profiles/catalog, `model_runs` audit, additive `model_*` metrics, and why Qwen is comparison-only. |
| [12-frameworks.md](12-frameworks.md) | Frameworks per component (the `requirements/` split) + how each is set up (`scripts/setup/`) and validated (`scripts/validate/` + `make validate`/`doctor`). |
| [13-models.md](13-models.md) | Consolidated model map: current cascade + the free NVIDIA lineup (real IDs), provenance pinning, the config-driven runtimes, and per-inference tracking. |
| [14-console-and-deployment.md](14-console-and-deployment.md) | **The front door.** The Console BFF/gateway (`services/console`, :8080) + one-command launch (`Dockerfile.app`, `docker-compose.platform.yml`, `make platform`); ports, env, auth, verification. |
| [15-quality-gate.md](15-quality-gate.md) | **Forward-looking (v3).** The QUALITY hook — ingestion image/video quality scoring that gates the models; scores, schema, gate decisions, routing, metrics, file layout, tests. |
| [16-capability-hooks-profiles-router.md](16-capability-hooks-profiles-router.md) | **Forward-looking (v3).** Plug-and-play WHEN/WHERE/WHAT/QUALITY capability hooks, the model registry/plugin schema, economy/balanced/benchmark profiles, the router, small-model parity + `model_performance` report, and the NVIDIA-only-policy relaxation. |
| [17-pos-and-time-alignment.md](17-pos-and-time-alignment.md) | **Forward-looking (v3).** POS-aware reasoning (Store-POS read-only adapter, modes, config), wall-clock time alignment, POS context in evidence, the full QUALITY→WHEN→WHERE→WHAT→POS→RULES→HUMAN hook map, review UI/outcomes. |
