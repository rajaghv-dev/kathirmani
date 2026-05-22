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

# DGX / compute metrics (GB10 Grace Blackwell unified-memory system)
DGX_COMPUTE_UTIL = Gauge(
    "marlin_dgx_compute_utilization_percent",
    "Compute utilization during inference (%)"
)

DGX_TEMPERATURE = Gauge(
    "marlin_dgx_temperature_celsius",
    "System temperature (°C)"
)

DGX_POWER_WATTS = Gauge(
    "marlin_dgx_power_watts",
    "Power draw (W)"
)

DGX_CPU_UTIL_USER = Gauge(
    "marlin_dgx_cpu_utilization_user_percent",
    "ARM CPU user utilization %"
)

DGX_CPU_UTIL_SYS = Gauge(
    "marlin_dgx_cpu_utilization_system_percent",
    "ARM CPU system utilization %"
)

DGX_UNIFIED_MEM_USED_GB = Gauge(
    "marlin_dgx_unified_mem_used_gb",
    "Unified memory used (GB)"
)

DGX_UNIFIED_MEM_TOTAL_GB = Gauge(
    "marlin_dgx_unified_mem_total_gb",
    "Unified memory total (GB)"
)

DGX_CPU_TEMP_CELSIUS = Gauge(
    "marlin_dgx_cpu_temperature_celsius",
    "ARM CPU temperature from ACPI (°C)"
)

# Token economy / cost metrics
ENERGY_WH = Gauge("marlin_energy_consumed_wh", "Estimated energy used this run (Wh)")
COST_INR  = Gauge("marlin_cost_inr", "Estimated inference cost in INR (₹)")
COST_PER_EVENT = Gauge("marlin_cost_per_event_inr", "INR per event detected")
EVENTS_PER_WH  = Gauge("marlin_events_per_wh", "Events detected per watt-hour (efficiency)")
INFERENCE_TOTAL_SEC = Gauge("marlin_total_inference_seconds", "Total wall time of last run (s)")
AVG_POWER_DURING_RUN = Gauge("marlin_avg_dgx_power_during_run_watts", "Average power during last inference run (W)")

# Power samples accumulated during an inference run
_dgx_power_samples: list[float] = []

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


def _read_proc_stat_cpu() -> tuple[float, float]:
    """Parse /proc/stat and return (user_pct, sys_pct) since last call."""
    # We keep state between calls using a mutable container on the function itself.
    try:
        with open("/proc/stat") as f:
            line = f.readline()  # first line: cpu aggregate
        fields = line.split()
        # fields: cpu user nice system idle iowait irq softirq steal guest guest_nice
        user  = int(fields[1]) + int(fields[2])   # user + nice
        system = int(fields[3])
        idle   = int(fields[4])
        total  = sum(int(x) for x in fields[1:])

        prev = _read_proc_stat_cpu._prev
        if prev is None:
            _read_proc_stat_cpu._prev = (user, system, total)
            return 0.0, 0.0

        d_user   = user   - prev[0]
        d_system = system - prev[1]
        d_total  = total  - prev[2]

        _read_proc_stat_cpu._prev = (user, system, total)

        if d_total <= 0:
            return 0.0, 0.0

        user_pct = 100.0 * d_user   / d_total
        sys_pct  = 100.0 * d_system / d_total
        return round(user_pct, 2), round(sys_pct, 2)
    except Exception:
        return 0.0, 0.0


_read_proc_stat_cpu._prev = None  # type: ignore[attr-defined]


def _read_unified_mem_gb() -> tuple[float, float]:
    """Return (used_gb, total_gb) from /proc/meminfo."""
    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", 0)
        total_gb = total_kb / (1024 ** 2)
        used_gb  = (total_kb - avail_kb) / (1024 ** 2)
        return round(used_gb, 3), round(total_gb, 3)
    except Exception:
        return 0.0, 0.0


def _read_thermal_zone_temp(zone: int = 0) -> float | None:
    """Read temperature from /sys/class/thermal/thermal_zone<zone>/temp (divide by 1000)."""
    try:
        with open(f"/sys/class/thermal/thermal_zone{zone}/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def poll_dgx_metrics(stop_event: threading.Event, interval: float = 1.0):
    """Background thread to poll DGX/GB10 hardware metrics.

    Sources:
    - Temperature  : /sys/class/thermal/thermal_zone0/temp
    - Power        : nvidia-smi --query-gpu=power.draw
    - Compute util : pynvml.nvmlDeviceGetUtilizationRates (falls back to 0)
    - CPU util     : /proc/stat
    - Unified mem  : /proc/meminfo
    """
    # Attempt to initialise NVML once; errors are silently swallowed.
    nvml_handle = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        pynvml = None  # type: ignore[assignment]

    while not stop_event.is_set():
        # --- Temperature from sysfs ACPI zone 0 ---
        temp = _read_thermal_zone_temp(zone=0)
        if temp is not None:
            DGX_TEMPERATURE.set(temp)
            DGX_CPU_TEMP_CELSIUS.set(temp)

        # --- Power from nvidia-smi ---
        try:
            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            power = _parse_smi_value(smi_out)
            if power is not None:
                DGX_POWER_WATTS.set(power)
                _dgx_power_samples.append(power)
        except Exception:
            pass

        # --- Compute utilisation from NVML ---
        if nvml_handle is not None:
            try:
                import pynvml as _pynvml
                util = _pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                DGX_COMPUTE_UTIL.set(util.gpu)
            except Exception:
                DGX_COMPUTE_UTIL.set(0)
        else:
            DGX_COMPUTE_UTIL.set(0)

        # --- CPU utilisation from /proc/stat ---
        user_pct, sys_pct = _read_proc_stat_cpu()
        DGX_CPU_UTIL_USER.set(user_pct)
        DGX_CPU_UTIL_SYS.set(sys_pct)

        # --- Unified memory from /proc/meminfo ---
        used_gb, total_gb = _read_unified_mem_gb()
        DGX_UNIFIED_MEM_USED_GB.set(used_gb)
        DGX_UNIFIED_MEM_TOTAL_GB.set(total_gb)

        time.sleep(interval)


def compute_economy(total_events: int, wall_time_sec: float, electricity_rate_inr_per_kwh: float = 10.0) -> dict:
    """Compute energy and cost from collected power samples."""
    if not _dgx_power_samples:
        return {}
    avg_power_w = sum(_dgx_power_samples) / len(_dgx_power_samples)
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

    _dgx_power_samples.clear()
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
