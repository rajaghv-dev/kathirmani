"""The `model_*` Prometheus namespace (master plan A8).

Additive to the existing `marlin_*` gauges (src/marlin/metrics.py) — the plugin
layer emits these; legacy panels keep working until the new dashboards (11–18)
validate, then `marlin_*` is deprecated (spec/11 'additive migration').

Mirrors the repo's prometheus_client idiom. Import the registry, set/observe in a
plugin's `infer()`, and the worker's `/metrics` exposes them.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

_COMMON = ["model_id", "task", "runtime", "profile"]

# ---- Common (A8.1) ----------------------------------------------------------
model_requests_total = Counter(
    "model_requests_total", "Model inference requests", _COMMON + ["status"])
model_latency_ms = Histogram(
    "model_latency_ms", "End-to-end model latency (ms)", _COMMON,
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000))
model_errors_total = Counter(
    "model_errors_total", "Model errors", _COMMON + ["error_type"])
model_gpu_memory_used_bytes = Gauge(
    "model_gpu_memory_used_bytes", "GPU memory used by model", _COMMON + ["gpu"])
model_gpu_utilization_ratio = Gauge(
    "model_gpu_utilization_ratio", "GPU utilization 0..1", _COMMON + ["gpu"])

# ---- VLM (A8.2) -------------------------------------------------------------
model_ttft_ms = Histogram(
    "model_ttft_ms", "Time to first token (ms)", _COMMON,
    buckets=(50, 100, 250, 500, 1000, 2000, 5000, 10000))
model_output_tokens_total = Counter(
    "model_output_tokens_total", "Output tokens", _COMMON)
model_tokens_per_second = Gauge(
    "model_tokens_per_second", "Decode throughput (tok/s)", _COMMON)
model_json_parse_total = Counter(
    "model_json_parse_total", "VLM JSON parse outcomes", _COMMON + ["result"])
model_verdict_total = Counter(
    "model_verdict_total", "VLM verdict distribution", _COMMON + ["verdict"])

# ---- Video throughput (A8.3) ------------------------------------------------
model_clips_processed_total = Counter(
    "model_clips_processed_total", "Clips processed", _COMMON)
model_video_seconds_processed_total = Counter(
    "model_video_seconds_processed_total", "Video seconds processed", _COMMON)
model_frames_processed_total = Counter(
    "model_frames_processed_total", "Frames processed", _COMMON)

# ---- Quality (A8.4) ---------------------------------------------------------
model_event_precision = Gauge(
    "model_event_precision", "Event precision on a dataset", _COMMON + ["dataset"])
model_event_recall = Gauge(
    "model_event_recall", "Event recall on a dataset", _COMMON + ["dataset"])
model_golden_regression_score = Gauge(
    "model_golden_regression_score", "Golden-set regression score", _COMMON + ["dataset"])


def labels(config) -> dict[str, str]:
    """Build the common label set from a PluginConfig."""
    return {
        "model_id": config.model_id,
        "task": config.task,
        "runtime": config.runtime,
        "profile": config.profile,
    }
