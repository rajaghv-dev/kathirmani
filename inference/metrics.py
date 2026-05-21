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
    "GPU memory used during inference (MB)"
)

GPU_UTILIZATION = Gauge(
    "marlin_gpu_utilization_percent",
    "GPU utilization during inference (%)"
)

GPU_TEMPERATURE = Gauge(
    "marlin_gpu_temperature_celsius",
    "GPU temperature (°C)"
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


def poll_gpu_metrics(stop_event: threading.Event, interval: float = 1.0):
    """Background thread to poll GPU metrics via nvidia-smi."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                text=True
            ).strip()
            mem, util, temp = [x.strip() for x in out.split(",")]
            GPU_MEMORY_USED_MB.set(float(mem))
            GPU_UTILIZATION.set(float(util))
            GPU_TEMPERATURE.set(float(temp))
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
