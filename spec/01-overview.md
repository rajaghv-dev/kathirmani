# 01 — Overview

## What the system is

The Kathirmani pipeline runs [NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B), a 2B-parameter video VLM, across **5 store security camera feeds** to extract dense captions, second-precise timestamps, and event detections. Metrics are exposed to Grafana via Prometheus.

Marlin-2B is a fine-tune of Qwen3.5-2B, trained on *atomic, temporally-grounded events* with explicit `<start-end>` boundaries (HC-STVG, VidSTG, TimeLens). That training shape dictates how the model should be queried (see [03-models-and-query-modes.md](03-models-and-query-modes.md)).

## The cameras

Footage lives in **`store-videos/`** at the repo root (gitignored; the filenames
below match `configs/cameras.yaml` → `source_file` exactly, typos included):

| File (store-videos/) | Location |
|---|---|
| Kathirmani Store Near Bill Counter.mkv | Cash register area |
| Kathirmani Center Camera.mkv | Center of store *(file not yet present — 4 of 5)* |
| Kathirmani Left Side Camera.mkv | Left aisle |
| Kathirmni Entry Cam.mkv | Entrance |
| Kathirmani Right Side Cam.mkv | Right aisle |

The 5 cameras cover the same store area from different angles.

## What the model extracts — the two modes

Marlin answers two questions about a video: **what** happens (caption) and **when** does a named event happen (find).

### Caption mode (per video)
`marlin.caption(video)` → `{caption, scene, events:[{start,end,description}]}`.
- Full scene description
- Dense event list with second-precise timestamps — returns the **full list** of time-ranged events automatically.

Use caption for "what happened overall" — it is the right tool for *multiple / recurring* events.

### Find mode (per video, per query)
`marlin.find(video, event=Q)` returns exactly one time span `[start, end]` in seconds for a named event. It grounds what is **seen**; there is no reasoning layer. Full query-design rules are in [03-models-and-query-modes.md](03-models-and-query-modes.md).

## Where results are saved

Results land on disk under `results/`:

- `results/<camera>.json` — caption, events, `find_results` (plus `locate_analysis` / `qwen_analysis` when those stages ran)
- `results/fused.json` — best-camera find spans + deduped store-wide events
- `results/summary.json`
- `results/economy.json`

`results/*.json` is the on-disk source of truth for the whole stack.

## Stack at a glance

- **Model:** Marlin-2B (Qwen3.5-2B base, fine-tuned for video VLM)
- **Runtime:** PyTorch 2.12 + CUDA 13.0 on NVIDIA GB10
- **Metrics:** `prometheus_client` → Grafana 12.4.1
- **System monitoring:** Netdata v2.9.0 (Prometheus endpoint)

## Related

[02-architecture.md](02-architecture.md) · [03-models-and-query-modes.md](03-models-and-query-modes.md) · [07-runbook.md](07-runbook.md)
