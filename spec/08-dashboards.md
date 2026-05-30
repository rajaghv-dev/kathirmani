# 08 — Grafana Dashboards

There are 7 dashboards. They all read from **Prometheus** (the time-series DB that scrapes `serve_metrics:8900` and `netdata:19999`) except the Loki one, which reads log lines from **Loki**. If a panel says **"No data"** the usual cause is not the panel — it's that Prometheus or its scrape targets aren't running (see the troubleshooting note at the bottom).

## Shared vocabulary

- **Event** — one thing Marlin's *caption* mode noticed in a camera's video (e.g. "person picks up item"). "Events detected" = how many per camera.
- **Find / span** — Marlin's *find* mode answers one query with a time window `[start, end]` in seconds. "Span duration" = `end − start`. A query either **hits** (returns a span) or **misses** (no span).
- **Parse success** — did the model's raw text come back in the expected format so we could extract a span? It measures output *formatting*, not whether the event truly happened. ~100% is the goal.
- **Fused / dedup** — the same real event is seen by several cameras. Fusion collapses those into one "unique event". `raw events` = before dedup, `unique events` = after.
- **Security signal** (Qwen dashboard) — a Qwen2.5-VL answer to a loss-prevention question that came back non-empty/affirmative (e.g. "is someone concealing an item?"). It's a *flag to review*, not a confirmed theft.

## The 7 dashboards

### Model & Runtime (`/d/marlin-model-runtime`) — "is the box healthy?"
The hardware/infra view. DGX power (W), compute utilization %, temperature, unified-memory and CPU/RAM timelines (CPU/RAM come from Netdata). Use it to confirm the GB10 is actually working during a run and isn't thermal/memory constrained. The Videos/Events stats are just a sanity check that a run landed.

### Application / Pipeline (`/d/marlin-pipeline`) — "did the model do its job?"
The per-run quality view, focused on the Bill Counter camera as the example. Total events, find queries resolved, **parse success rate**, and a table + bar gauge of each find query's span. This is where you check the *model output* is well-formed and queries are resolving — independent of hardware.

### Camera Views (`/d/marlin-cameras`) — "what happened on each camera?"
One row per camera (all 5). Each row: events detected, find-span duration per query, parse success. Use it to compare cameras — e.g. the entry door sees more events than a side aisle, or one camera's parse rate is low.

### Store View / Fused (`/d/marlin-fused`) — "what happened in the store?"
The cross-camera rollup. Unique events after dedup vs. raw events before, "best camera" span per query (the tightest/most confident window across all angles), an un-fused comparison table, and an events-over-time series. The closest thing to a single store-wide truth, since the 5 cameras cover the same area from different angles.

### Visual Analysis / Qwen (`/d/marlin-qwen`) — "loss-prevention read"
Qwen2.5-VL's answers layered on top of Marlin. Queries OK per camera, **security signals** per camera, and loss-prevention event durations from Marlin's temporal finds. This is the surveillance/theft-review lens; the bars tell you which camera to look at first.

### Inference Logs / Loki (`/d/marlin-loki`) — "the play-by-play"
Not metrics — actual log lines. Live feed of all inference logs, plus per-camera **captions** (the model's own description of what it saw), find hits, queries not found (coverage gaps), errors, and run start/complete markers. Go here when a number on another dashboard looks wrong and you want the raw story behind it.

### Compute Economy (`/d/marlin-economy`) — "what did this run cost / save?"
The money + efficiency view. Energy (Wh), estimated cost (₹), cost per event, events per Wh, avg power, wall time. Plus the **"Compute saved — decode-once"** row: decode time saved, redundant decodes avoided, and decode-reduction %. Caveat: the decode-savings gauges are emitted **only by a live `run_inference.py` process**; `serve_metrics.py` (which feeds Grafana between runs) doesn't republish them, so that row is populated during a run and blank afterwards.

## Troubleshooting — if everything says "No data"

The dashboards are only as alive as the stack behind them. Most "No data" is one of two things:

1. **Prometheus or `serve_metrics` container is down.** Check `docker ps -a`; if `prometheus`/`serve_metrics` show `Exited`, run `bash start_stack.sh`. Verify with `curl localhost:9090/api/v1/targets` (targets should be `up`).
2. **No run has happened yet** — there are no `results/*.json` for `serve_metrics.py` to expose.

`run_inference.py` now runs a **telemetry preflight** (`_check_telemetry()`) at startup that pings Prometheus/Grafana/Loki/Netdata and prints a warning + `bash start_stack.sh` hint if any are down, so a run no longer silently finishes into empty dashboards.

## Related

[04-observability-stack.md](04-observability-stack.md) · [07-runbook.md](07-runbook.md)
