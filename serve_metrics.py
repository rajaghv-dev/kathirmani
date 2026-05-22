#!/usr/bin/env python3
"""
Persistent Prometheus metrics server for Marlin inference results.

Reads results/*.json and exposes all inference metrics on :8900/metrics.
Reloads automatically when results change — no need to keep run_inference.py alive.

Usage:
    source .venv/bin/activate
    python serve_metrics.py            # serves on :8900
    python serve_metrics.py --port 8901
"""
import argparse
import json
import time
import threading
from pathlib import Path

from prometheus_client import (
    start_http_server, Gauge, Counter, Histogram, Info, REGISTRY
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from prometheus_client.registry import Collector

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

# --- DGX Spark (Grace Blackwell) hardware metrics — live during inference ---
# GB10 is integrated GPU with 119GB unified memory; no discrete VRAM
DGX_COMPUTE = Gauge("marlin_dgx_compute_utilization_percent", "DGX compute utilization (%)")
DGX_TEMP    = Gauge("marlin_dgx_temperature_celsius",         "DGX system temperature (°C)")
DGX_POWER   = Gauge("marlin_dgx_power_watts",                 "DGX power draw (W)")
DGX_CPU_U   = Gauge("marlin_dgx_cpu_utilization_user_percent","ARM CPU user utilization (%)")
DGX_CPU_S   = Gauge("marlin_dgx_cpu_utilization_system_percent","ARM CPU system utilization (%)")
DGX_MEM_U   = Gauge("marlin_dgx_unified_mem_used_gb",         "Unified memory used (GB)")
DGX_MEM_T   = Gauge("marlin_dgx_unified_mem_total_gb",        "Unified memory total (GB)")

# --- static metrics loaded from JSON results ---
EVENTS    = Gauge("marlin_events_detected_total",   "Events detected per camera", ["video"])
PROCESSED = Gauge("marlin_videos_processed_total",  "Videos with results on disk")
TOTAL_EV  = Gauge("marlin_total_events",            "Sum of events across all cameras")

FIND_START  = Gauge("marlin_find_span_start_seconds", "Find result span start (s)", ["video", "query"])
FIND_END    = Gauge("marlin_find_span_end_seconds",   "Find result span end (s)",   ["video", "query"])
FIND_OK     = Gauge("marlin_find_parse_success",       "1 if find result parsed OK", ["video", "query"])
FIND_DUR    = Gauge("marlin_find_span_duration_seconds","Find span duration (s)",    ["video", "query"])

CAP_TIME    = Gauge("marlin_caption_duration_seconds", "Time to caption a video (s)", ["video"])
FIND_TIME   = Gauge("marlin_find_duration_seconds",    "Avg find query time (s)",     ["video"])

MODEL_INFO  = Info("marlin_model", "Model metadata")

# --- fused / store-wide metrics (from results/fused.json) ---
FUSED_EVENTS    = Gauge("marlin_fused_unique_events",         "Unique events after multi-view dedup")
FUSED_RAW       = Gauge("marlin_fused_raw_events_total",      "Raw events before dedup across cameras")
FUSED_FIND_START= Gauge("marlin_fused_find_span_start_seconds","Fused find span start (s)", ["query"])
FUSED_FIND_END  = Gauge("marlin_fused_find_span_end_seconds",  "Fused find span end (s)",   ["query"])
FUSED_FIND_DUR  = Gauge("marlin_fused_find_span_duration_seconds","Fused span duration (s)", ["query"])

# --- token economy / cost metrics (from results/economy.json) ---
ECONOMY_ENERGY   = Gauge("marlin_energy_consumed_wh",            "Estimated GPU energy used this run (Wh)")
ECONOMY_COST     = Gauge("marlin_cost_inr",                      "Estimated inference cost in INR (₹)")
ECONOMY_CPE      = Gauge("marlin_cost_per_event_inr",            "INR per event detected")
ECONOMY_EPW      = Gauge("marlin_events_per_wh",                 "Events detected per watt-hour (efficiency)")
ECONOMY_WALLTIME = Gauge("marlin_total_inference_seconds",       "Total wall time of last run (s)")
ECONOMY_AVGPOWER = Gauge("marlin_avg_dgx_power_during_run_watts","Average GPU power during last inference run (W)")

_loaded: set[str] = set()


def load_results():
    """Read all result JSONs and push values into Prometheus gauges."""
    if not RESULTS_DIR.exists():
        return

    videos_done = 0
    total_events = 0

    for jf in RESULTS_DIR.glob("*.json"):
        if jf.name in ("summary.json", "fused.json"):
            continue
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue

        label = data.get("label", jf.stem)
        events = data.get("events", [])
        EVENTS.labels(video=label).set(len(events))
        total_events += len(events)
        videos_done += 1

        for query, res in data.get("find_results", {}).items():
            span = res.get("span")
            ok   = 1 if res.get("format_ok") else 0
            FIND_OK.labels(video=label, query=query).set(ok)
            if span and len(span) == 2:
                FIND_START.labels(video=label, query=query).set(span[0])
                FIND_END.labels(video=label, query=query).set(span[1])
                FIND_DUR.labels(video=label, query=query).set(max(0, span[1] - span[0]))

        if jf.name not in _loaded:
            _loaded.add(jf.name)
            print(f"[serve] Loaded {label}: {len(events)} events, "
                  f"{len(data.get('find_results', {}))} find results")

    PROCESSED.set(videos_done)
    TOTAL_EV.set(total_events)
    _load_fused()
    _load_economy()


def _load_economy():
    """Load results/economy.json and update token economy gauges."""
    economy_path = RESULTS_DIR / "economy.json"
    if not economy_path.exists():
        return
    try:
        data = json.loads(economy_path.read_text())
    except Exception:
        return

    if "energy_wh" in data:
        ECONOMY_ENERGY.set(data["energy_wh"])
    if "cost_inr" in data:
        ECONOMY_COST.set(data["cost_inr"])
    if "cost_per_event_inr" in data:
        ECONOMY_CPE.set(data["cost_per_event_inr"])
    if "events_per_wh" in data:
        ECONOMY_EPW.set(data["events_per_wh"])
    if "wall_time_sec" in data:
        ECONOMY_WALLTIME.set(data["wall_time_sec"])
    if "avg_power_w" in data:
        ECONOMY_AVGPOWER.set(data["avg_power_w"])


def _load_fused():
    """Load results/fused.json and update store-wide metrics."""
    fused_path = RESULTS_DIR / "fused.json"
    if not fused_path.exists():
        return
    try:
        data = json.loads(fused_path.read_text())
    except Exception:
        return

    FUSED_EVENTS.set(data.get("unique_event_count", 0))
    FUSED_RAW.set(data.get("total_raw_events", 0))

    for query, res in data.get("find_results", {}).items():
        span = res.get("span")
        if span and len(span) == 2:
            FUSED_FIND_START.labels(query=query).set(span[0])
            FUSED_FIND_END.labels(query=query).set(span[1])
            FUSED_FIND_DUR.labels(query=query).set(max(0, span[1] - span[0]))


def poll_loop(interval: int):
    """Reload results from disk every `interval` seconds."""
    while True:
        try:
            load_results()
        except Exception as e:
            print(f"[serve] reload error: {e}")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--reload-interval", type=int, default=30,
                        help="Re-read results every N seconds")
    args = parser.parse_args()

    # Set static model metadata
    MODEL_INFO.info({
        "name": "Marlin-2B",
        "base": "Qwen3.5-2B",
        "source": "NemoStation/Marlin-2B",
    })

    # Initial load
    load_results()

    # Start background reload
    t = threading.Thread(target=poll_loop, args=(args.reload_interval,), daemon=True)
    t.start()

    # Start HTTP server
    start_http_server(args.port)
    print(f"[serve] Metrics at http://localhost:{args.port}/metrics  (reloads every {args.reload_interval}s)")
    print("[serve] Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[serve] Stopped.")


if __name__ == "__main__":
    main()
