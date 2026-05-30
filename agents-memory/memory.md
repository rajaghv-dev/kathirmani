# Project memory

Durable, cross-session facts about this project. Read before starting work.

## Project

Marlin-2B (a video VLM) plus Qwen2.5-VL run video intelligence over 5
Kathirmani store security cameras (Bill Counter, Center, Left Side, Near Entry
Door, Right Side). The pipeline extracts dense captions, second-precise event
timestamps, and grounded "find" event spans, then emits metrics and logs to a
Prometheus / Grafana / Loki / Netdata observability stack. A Streamlit viewer
provides ad-hoc visualization of the on-disk results.

## Target machines

This repo is meant to run on **two** production accelerators:

- **NVIDIA DGX box** — CUDA, GB10 (Blackwell) with unified LPDDR5X memory (no
  discrete VRAM).
- **Apple Mac Studio** — Apple Silicon, Metal / MPS.

It must **auto-detect the accelerator at runtime**, preferring
**CUDA → MPS → CPU** in that order, and pick an appropriate dtype per device.
Development happens on other boxes where **only CPU may be available and the
model weights are not present**, so the code must degrade gracefully (no hard
dependency on a GPU or on the model being downloaded).

## Key entry points

- **`run_inference.py`** — the main pipeline (Marlin caption + find across the 5
  cameras; opt-in `--locate` YOLOE gate and Qwen cascade).
- **`serve_metrics.py`** — Prometheus exporter; reads `results/*.json` and
  exposes them between runs.
- **`download_model.py`** — fetches the model weights.
- **`viz_app.py`** — Streamlit viewer for the results.
- **`start_stack.sh`** — brings up the observability Docker stack
  (Prometheus, Loki, Grafana, Netdata).

## Datastores

There is **no SQL/NoSQL database**. The datastores are:

- **Prometheus** (TSDB) — metric gauges/histograms set inline per camera and
  scraped from `serve_metrics.py` and Netdata.
- **Loki** — log lines pushed directly by the pipeline.
- **`results/*.json`** — the on-disk **source of truth**, intentionally
  git-tracked.

## Gotchas worth remembering

- **The model is not "re-downloading."** Weights load with
  `local_files_only=True` / `HF_HUB_OFFLINE=1` and are never re-fetched. What
  regenerates each run is the `trust_remote_code` module cache
  (`transformers_modules`) and the `torch.compile` artifacts. These are pinned
  into the repo via `HF_HOME` and `TORCHINDUCTOR_CACHE_DIR` (gitignored cache
  dirs) so they persist across runs.
- **Marlin is Qwen3-VL based, not Qwen2.5-VL.** The processor is a Qwen3-VL
  processor that builds prompts from **frame timestamps**, so cached/pre-decoded
  frames must be passed with `video_metadata` — otherwise it defaults to
  fps=24 and the find spans come out wrong (captions still look fine).
- **Single GPU serializes all forward passes via `_GPU_LOCK`.** Cameras run in a
  `ThreadPoolExecutor`, but only decode/parse/IO/metric inserts overlap; the GPU
  math is serialized. Expect a modest wall-time win, not 5x.
- **Decode-once optimization is already implemented.** The video is decoded once
  per camera and the frames are reused across all 21 prompts (1 caption + 20
  finds), eliminating ~95% of redundant video decode. Decode runs outside
  `_GPU_LOCK`.

## See also

`learning.md` (repo root) and `spec/` are the human-facing documentation —
consult them for the detailed reasoning, dataflow, and roadmap.
