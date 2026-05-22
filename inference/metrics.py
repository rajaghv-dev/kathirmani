"""Prometheus metrics server for Marlin inference monitoring."""
import socket
import time
import threading
import subprocess
from prometheus_client import (
    start_http_server, Gauge, Counter, Histogram, Summary, Info
)

# Inference timing
INFERENCE_DURATION = Histogram(
    "marlin_inference_duration_seconds",
    "Time to run Marlin inference on a video",
    ["video", "mode"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
)

# Counts
EVENTS_DETECTED = Gauge(
    "marlin_events_detected_total",
    "Number of events detected per video",
    ["video"]
)

VIDEOS_PROCESSED = Counter(
    "marlin_videos_processed_total",
    "Total videos processed",
    ["status"]
)

FIND_SPAN_START = Gauge(
    "marlin_find_span_start_seconds",
    "Start of found event span in video",
    ["video", "query"]
)

FIND_SPAN_END = Gauge(
    "marlin_find_span_end_seconds",
    "End of found event span in video",
    ["video", "query"]
)

FIND_PARSE_OK = Gauge(
    "marlin_find_parse_success",
    "Whether find result was parsed cleanly",
    ["video", "query"]
)

# GPU
GPU_MEMORY_USED_MB = Gauge(
    "marlin_gpu_memory_used_mb",
    "GPU memory used during inference (MB) — N/A on unified-memory GPUs"
)

GPU_UTILIZATION = Gauge(
    "marlin_gpu_utilization_percent",
    "GPU utilization during inference (%)"
)

GPU_TEMPERATURE = Gauge(
    "marlin_gpu_temperature_celsius",
    "GPU temperature (°C)"
)

GPU_POWER_WATTS = Gauge(
    "marlin_gpu_power_watts",
    "GPU power draw (W)"
)

# Token economy / cost metrics
ENERGY_WH = Gauge("marlin_energy_consumed_wh", "Estimated GPU energy used this run (Wh)")
COST_INR  = Gauge("marlin_cost_inr", "Estimated inference cost in INR (₹)")
COST_PER_EVENT = Gauge("marlin_cost_per_event_inr", "INR per event detected")
EVENTS_PER_WH  = Gauge("marlin_events_per_wh", "Events detected per watt-hour (efficiency)")
INFERENCE_TOTAL_SEC = Gauge("marlin_total_inference_seconds", "Total wall time of last run (s)")
AVG_POWER_DURING_RUN = Gauge("marlin_avg_gpu_power_during_run_watts", "Average GPU power during last inference run (W)")

# Power samples accumulated during an inference run
_power_samples: list[float] = []

# Model info
MODEL_INFO = Info("marlin_model", "Model metadata")

# Fused / multi-view metrics
FUSED_EVENTS = Gauge(
    "marlin_fused_unique_events",
    "Unique events after multi-view dedup"
)

FUSED_TOTAL_RAW = Gauge(
    "marlin_fused_raw_events_total",
    "Total raw events before dedup"
)

FUSED_FIND_START = Gauge(
    "marlin_fused_find_span_start_seconds",
    "Fused find span start",
    ["query"]
)

FUSED_FIND_END = Gauge(
    "marlin_fused_find_span_end_seconds",
    "Fused find span end",
    ["query"]
)

FUSED_FIND_CAM = Gauge(
    "marlin_fused_find_best_camera",
    "Which camera had the best detection (index 0-4)",
    ["query"]
)


def _parse_smi_value(s: str) -> float | None:
    s = s.strip().replace("[N/A]", "").replace("N/A", "").strip()
    return float(s) if s else None


def poll_gpu_metrics(stop_event: threading.Event, interval: float = 1.0):
    """Background thread to poll GPU metrics via nvidia-smi.
    Handles GB10 unified-memory GPUs where memory.used reports [N/A]."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used",
                 "--format=csv,noheader,nounits"],
                text=True
            ).strip()
            fields = [x.strip() for x in out.split(",")]
            util_gpu  = _parse_smi_value(fields[0])
            util_mem  = _parse_smi_value(fields[1])
            temp      = _parse_smi_value(fields[2])
            power     = _parse_smi_value(fields[3])
            mem_used  = _parse_smi_value(fields[4])

            if util_gpu  is not None: GPU_UTILIZATION.set(util_gpu)
            if temp      is not None: GPU_TEMPERATURE.set(temp)
            if power     is not None:
                GPU_POWER_WATTS.set(power)
                _power_samples.append(power)
            if mem_used  is not None: GPU_MEMORY_USED_MB.set(mem_used)
        except Exception:
            pass
        time.sleep(interval)


def compute_economy(total_events: int, wall_time_sec: float, electricity_rate_inr_per_kwh: float = 10.0) -> dict:
    """Compute energy and cost from collected power samples."""
    if not _power_samples:
        return {}
    avg_power_w = sum(_power_samples) / len(_power_samples)
    energy_wh = avg_power_w * (wall_time_sec / 3600)
    cost_inr = energy_wh / 1000 * electricity_rate_inr_per_kwh
    events_per_wh = total_events / max(energy_wh, 0.001)
    cost_per_event = cost_inr / max(total_events, 1)

    ENERGY_WH.set(round(energy_wh, 4))
    COST_INR.set(round(cost_inr, 6))
    EVENTS_PER_WH.set(round(events_per_wh, 2))
    COST_PER_EVENT.set(round(cost_per_event, 6))
    INFERENCE_TOTAL_SEC.set(round(wall_time_sec, 1))
    AVG_POWER_DURING_RUN.set(round(avg_power_w, 2))

    _power_samples.clear()
    return {
        "avg_power_w": round(avg_power_w, 2),
        "energy_wh": round(energy_wh, 4),
        "cost_inr": round(cost_inr, 6),
        "events_per_wh": round(events_per_wh, 2),
        "cost_per_event_inr": round(cost_per_event, 6),
        "wall_time_sec": round(wall_time_sec, 1),
        "total_events": total_events,
    }


def start_metrics_server(port: int = 8900):
    """Start Prometheus metrics HTTP server."""
    while True:
        try:
            start_http_server(port)
            break
        except OSError:
            port += 1
    MODEL_INFO.info({
        "name": "Marlin-2B",
        "base": "Qwen3.5-2B",
        "source": "NemoStation/Marlin-2B",
    })
    print(f"[metrics] Prometheus endpoint: http://localhost:{port}/metrics")
