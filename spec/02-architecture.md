# 02 — Architecture

## The three perception axes

The pipeline combines three models along three orthogonal axes, cascade-gated:

| Axis | Model | Question it answers |
|---|---|---|
| **When** (temporal) | Marlin | When does a named event happen? (caption events + find spans) |
| **Where** (spatial) | Locate (YOLOE) | Where are the objects? Also the cheap front-gate emitting `route_to_vlm`. |
| **What** (semantic) | Qwen2.5-VL | What is happening, in detail — runs **only** on routed keyframes. |

Full dataflow lives in `docs/ARCHITECTURE.md`.

## The axes are capability hooks, not fixed models

The three axes above are the **current defaults**, but the architecture treats each as
a **plug-and-play capability hook** — the pipeline calls a *capability*
(`temporal_find`, `spatial_grounding`, `semantic_vlm`), and a config profile binds the
concrete model. So Marlin/YOLOE/Qwen can be swapped or A/B-compared without refactoring.
A fourth hook, **QUALITY**, runs *before* all three to decide whether the input is even
usable, and a **POS** hook runs *after* to settle the transaction question. Full hook
map and selection rules:

```text
QUALITY → Is this visual input good enough?          (spec/15)
WHEN    → When did a visible event happen?           (Marlin, spec/03)
WHERE   → Where is the relevant person/object/zone?  (Locate/YOLOE + route_to_vlm)
WHAT    → What does the keyframe/clip appear to show? (Qwen, routed-only)
POS     → Was there a matching transaction?          (spec/17)
RULES   → Should this become a review item?          (rule_engine, spec/10 Phase 5)
HUMAN   → What is the final decision?                (review_ui, spec/10 Phase 7)
```

Hooks + cost-tiered model profiles + the router:
[16-capability-hooks-profiles-router.md](16-capability-hooks-profiles-router.md). The
QUALITY gate: [15-quality-gate.md](15-quality-gate.md). POS + wall-clock alignment:
[17-pos-and-time-alignment.md](17-pos-and-time-alignment.md).

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

## Related

[01-overview.md](01-overview.md) · [04-observability-stack.md](04-observability-stack.md) · [05-performance-and-optimizations.md](05-performance-and-optimizations.md) · [06-hardware-portability.md](06-hardware-portability.md) · [15-quality-gate.md](15-quality-gate.md) · [16-capability-hooks-profiles-router.md](16-capability-hooks-profiles-router.md) · [17-pos-and-time-alignment.md](17-pos-and-time-alignment.md)
