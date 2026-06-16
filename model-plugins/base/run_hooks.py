"""Model-run lifecycle hook — fires whenever a model is kicked off.

The pipeline (workers) wraps each `plugin.infer()` call with this hook so that, for
*every* model run, we:

  1. push a Grafana **region annotation** (start→end) tagged with the model_id, so the
     per-model dashboard timeline shows exactly when that model ran; and
  2. range-query **Netdata** over the run window [start, end] for host CPU / RAM / disk
     I/O, sample **GPU power** from nvidia-smi, and publish them as `model_resource_*`
     gauges labelled by (model_id, task, runtime, profile).

That is the link from "a model got kicked off" to "its runtime resource cost on the
box", per spec/04 (Netdata is the runtime-metric source) and spec/16 (per-model
dashboards). It is **best-effort and never raises**: if Netdata/Grafana/nvidia-smi are
absent (e.g. a CPU-only dev box), the gauges that can't be filled are simply skipped and
inference is unaffected.

Usage (worker side, around the single infer choke-point):

    from base.run_hooks import record_model_run
    t0 = time.time()
    out = plugin.infer(window)
    record_model_run(plugin.config, t0, time.time(), out.get("model_run"))

or as a context manager:

    with model_run(plugin.config):
        out = plugin.infer(window)
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import time
from typing import Any

try:  # in-package (workers put model-plugins/ on sys.path → `base` is importable)
    from . import metrics as M
except ImportError:  # pragma: no cover - direct-script fallback
    import metrics as M  # type: ignore

NETDATA_URL = os.environ.get("NETDATA_URL", "http://localhost:19999")
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = os.environ.get("GRAFANA_USER", "admin")
GRAFANA_PASS = os.environ.get("GRAFANA_PASS", "admin")
GRAFANA_API_KEY = os.environ.get("GRAFANA_API_KEY", "")
# Annotations/Netdata calls are opt-out: set KATHIRMANI_RUN_HOOK=0 to disable entirely.
_ENABLED = os.environ.get("KATHIRMANI_RUN_HOOK", "1") not in ("0", "false", "False")
_HAVE_SMI = shutil.which("nvidia-smi") is not None


def _labels(config) -> dict[str, str]:
    """Common label set; tolerant of a missing/partial config."""
    g = lambda k, d="unknown": getattr(config, k, d) or d  # noqa: E731
    return {"model_id": g("model_id"), "task": g("task"),
            "runtime": g("runtime"), "profile": g("profile")}


def _netdata_window(chart: str, after_ts: float, before_ts: float,
                    timeout: float = 1.5) -> dict[str, float]:
    """Average per-dimension values for `chart` over [after_ts, before_ts] (unix sec).

    Uses Netdata's `after`/`before` range with a single averaged point. Returns {} if
    Netdata is unreachable (the dev-box / no-stack case)."""
    try:
        import requests  # local import: keep base import-light
        r = requests.get(
            f"{NETDATA_URL}/api/v1/data",
            params={"chart": chart, "after": int(after_ts), "before": int(before_ts),
                    "points": 1, "group": "average", "format": "json"},
            timeout=timeout,
        )
        j = r.json()
        labels, row = j["labels"], j["data"][0]
        return {labels[i]: row[i] for i in range(1, len(labels))
                if isinstance(row[i], (int, float))}
    except Exception:
        return {}


def _gpu_power_watts() -> float | None:
    """Instantaneous GPU power from nvidia-smi, or None on a no-GPU box."""
    if not _HAVE_SMI:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=2.0).strip().splitlines()
        vals = [float(x) for x in out if x.strip()]
        return round(sum(vals), 2) if vals else None
    except Exception:
        return None


def _push_annotation(text: str, tags: list[str], start_ms: int, end_ms: int) -> None:
    """Best-effort Grafana region annotation (start→end). No-op if unreachable."""
    try:
        import requests
        headers, auth = {}, None
        if GRAFANA_API_KEY:
            headers["Authorization"] = f"Bearer {GRAFANA_API_KEY}"
        else:
            auth = (GRAFANA_USER, GRAFANA_PASS)
        requests.post(
            f"{GRAFANA_URL}/api/annotations",
            json={"text": text, "tags": tags, "time": start_ms, "timeEnd": end_ms},
            headers=headers, auth=auth, timeout=1.5,
        )
    except Exception:
        pass


def record_model_run(config, started_at: float, ended_at: float,
                     model_run: dict[str, Any] | None = None) -> None:
    """Record one model run's runtime-resource footprint + a Grafana run annotation.

    `config` is the plugin's PluginConfig (model_id/task/runtime/profile). `started_at`
    / `ended_at` are wall-clock unix seconds bracketing the inference. Never raises."""
    if not _ENABLED:
        return
    try:
        lab = _labels(config)
        dur = max(0.0, ended_at - started_at)
        M.model_resource_run_seconds.labels(**lab).set(round(dur, 3))
        M.model_runs_total.labels(**lab).inc()

        # Host CPU / RAM / IO over the run window, from Netdata.
        cpu = _netdata_window("system.cpu", started_at, ended_at)
        if cpu:
            M.model_resource_cpu_percent.labels(**lab).set(
                round(sum(v for v in cpu.values() if isinstance(v, (int, float))), 2))
        ram = _netdata_window("system.ram", started_at, ended_at)
        if "used" in ram:
            M.model_resource_ram_used_mb.labels(**lab).set(round(ram["used"], 1))
        io = _netdata_window("system.io", started_at, ended_at)
        if io:
            # Netdata reports reads as +, writes as - (KiB/s); take magnitudes.
            reads = next((io[k] for k in io if k.lower().startswith("read") or k == "in"), None)
            writes = next((io[k] for k in io if k.lower().startswith("writ") or k == "out"), None)
            if reads is not None:
                M.model_resource_io_read_kbps.labels(**lab).set(round(abs(reads), 1))
            if writes is not None:
                M.model_resource_io_write_kbps.labels(**lab).set(round(abs(writes), 1))

        # GPU power (point sample; nvidia-smi has no cheap range query).
        gpu_w = _gpu_power_watts()
        if gpu_w is not None:
            M.model_resource_gpu_power_watts.labels(**lab).set(gpu_w)

        # Run-window annotation for the per-model dashboard timeline.
        mid = lab["model_id"]
        faked = (model_run or {}).get("faked")
        lat = (model_run or {}).get("latency_ms")
        suffix = f" · {lat:.0f}ms" if isinstance(lat, (int, float)) else ""
        suffix += " · faked" if faked else ""
        _push_annotation(
            text=f"▶ {mid} [{lab['task']}]{suffix}",
            tags=["kathir", "model-run", mid],
            start_ms=int(started_at * 1000), end_ms=int(ended_at * 1000),
        )
    except Exception:
        # A telemetry hook must never break inference.
        pass


@contextlib.contextmanager
def model_run(config, model_run: dict[str, Any] | None = None):
    """Context manager form: stamps start/end around the wrapped infer call."""
    t0 = time.time()
    try:
        yield
    finally:
        record_model_run(config, t0, time.time(), model_run)
