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
            if power     is not None: GPU_POWER_WATTS.set(power)
            if mem_used  is not None: GPU_MEMORY_USED_MB.set(mem_used)
        except Exception:
            pass
        time.sleep(interval)


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
