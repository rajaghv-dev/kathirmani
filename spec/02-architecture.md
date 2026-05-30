# 02 — Architecture

## The three perception axes

The pipeline combines three models along three orthogonal axes, cascade-gated:

| Axis | Model | Question it answers |
|---|---|---|
| **When** (temporal) | Marlin | When does a named event happen? (caption events + find spans) |
| **Where** (spatial) | Locate (YOLOE) | Where are the objects? Also the cheap front-gate emitting `route_to_vlm`. |
| **What** (semantic) | Qwen2.5-VL | What is happening, in detail — runs **only** on routed keyframes. |

Full dataflow lives in `docs/ARCHITECTURE.md`.

## The parallelism reality

`run_all()` processes the 5 cameras through a `ThreadPoolExecutor` (`--max-workers`, default one worker per video).

**But there is one GPU and one shared model**, so `_GPU_LOCK` serializes every forward pass. The parallelism overlaps:

- video decode
- parsing
- JSON IO
- the Loki / Prometheus / annotation inserts

…but **not** the GPU math. Expect a modest wall-time win, not 5×. True parallel GPU throughput would need cross-camera frame batching or node sharding (the `serve_stream.py` two-tier design, not yet built).

## The "databases" in this stack

There is **no SQL/NoSQL DB**. The datastores are:

- **Prometheus** (TSDB) — fed by metric gauges set inline per camera + scraped from `serve_metrics.py` (which reads `results/*.json`) and Netdata.
- **Loki** (logs) — fed directly by `inference/loki.py`.
- **`results/*.json`** — the on-disk source of truth.

Each parallel worker inserts to Loki + sets Prometheus gauges + pushes Grafana annotations as it finishes — so "DB inserts" happen concurrently across cameras. (The old sequential `run_all` had silently dropped all Loki logging; this was restored.)
