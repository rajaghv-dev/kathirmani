# Agent memory

Terse edit/run context. *What/why* lives in `spec/` — follow the links.

**What:** Marlin-2B + Qwen2.5-VL video intelligence over 5 store cameras →
Prometheus/Grafana/Loki + Streamlit viewer. → `spec/01`, `spec/02`.

## Repo map → `spec/09`
- Code in `src/marlin/`; root `*.py` are thin shims (don't add logic there).
- Entry points: `run_inference.py`, `serve_metrics.py`, `download_model.py`,
  `viz_app.py`, `start_stack.sh`. Commands unchanged post-refactor.
- Paths anchor on `marlin.PROJECT_ROOT/MODELS_DIR/RESULTS_DIR` (not cwd).

## Invariants — don't break
- Device is runtime-detected; load via `marlin.device.detect_device()`, place
  tensors with `.to(model.device)`. Never hardcode `"cuda"`/dtype. → `spec/06`.
- Keep root shims + path anchors working; keep `learning.md` as-is.
- Guard NVIDIA/Linux-only calls (nvidia-smi, NVML, `/sys`,`/proc`, `hostname -I`).

## Run / test (dev box = CPU, no model)
- `bash setup.sh` (picks torch per machine), `bash run.sh`. Override:
  `MARLIN_DEVICE=cuda|mps|cpu`.
- `pytest -q tests/` → expect ~18 passed, ~7 skipped (torch/model/server absent).
  `test_device.py` + `test_structure.py` are torch-free.

## Pitfalls (detail in spec)
- "Re-downloading" = `transformers_modules` + compile cache regen, not weights
  (offline). → `spec/04`.
- Marlin is Qwen3-VL: pre-decoded frames need `video_metadata` or find spans
  break. → `spec/05`.
- One GPU; `_GPU_LOCK` serializes forward passes (only decode/IO overlap). →
  `spec/02`, `spec/05`.
- `inference/locate.py` (`--locate`) absent — optional lazy stub (`marlin.locate`).

## Roadmap (future platform — not built yet) → `spec/10`, `spec/11`
- Target: OSS-ingestion-first, NVIDIA-model-first, plugin platform (master plan
  `oss_ingestion_nvidia_model_plugin_master_plan_v2.md` at root). Current
  `src/marlin` cascade = the AI-worker seed. 14 phases; start at Phase 0.
- **Invariant — NVIDIA-only default models.** Qwen/Marlin/YOLOE are disallowed as
  defaults → kept as comparison-only `research_qwen_baseline` behind the plugin
  layer; NVIDIA models (maybe stubbed) are production default. Swap = config, not
  code. → `spec/11`.

## Spec index
`01` overview · `02` architecture · `03` models/queries · `04` observability ·
`05` performance · `06` hardware-portability · `07` runbook · `08` dashboards ·
`09` repo-structure · `10` platform-roadmap · `11` model-plugin-policy.
