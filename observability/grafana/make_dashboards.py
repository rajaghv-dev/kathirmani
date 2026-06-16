#!/usr/bin/env python3
"""Headless generator for the Kathirmani platform Grafana dashboard set (01-18).

Emits valid Grafana dashboard JSON (schemaVersion 39) under
``observability/grafana/dashboards/*.json``. No matplotlib / no Grafana needed at
generation time — this just writes JSON, mirroring the structure of the existing
``grafana/dashboard_pipeline.json`` (same datasource uid ``marlin-metrics``, same
"About this dashboard" text panel convention from spec/08).

The platform set (01-10) references the documented ``ingest_*`` family
(``ingestion/metrics.py``) and ``kathirmani_*`` where relevant; the model-benchmark set
(11-18) references the ``model_*`` namespace (``model-plugins/base/metrics.py``,
spec/11 + master plan A8). Every dashboard gets a unique ``uid`` (``kathir-NN-...``)
and an About panel — that intro panel is a durable user preference (explain meaning).

Usage:
    .venv/bin/python observability/grafana/make_dashboards.py [--out DIR]

Import into Grafana with the existing ``grafana/setup_grafana.sh`` pattern
(POST each file to ``/api/dashboards/db``) — see ``observability/README.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = 39
DS = {"type": "prometheus", "uid": "marlin-metrics"}

# ---------------------------------------------------------------------------
# Small panel builders. gridPos.y is auto-stacked per dashboard by `build()`.
# ---------------------------------------------------------------------------


def _about(content: str) -> dict:
    return {
        "type": "text",
        "title": "About this dashboard",
        "gridPos": {"x": 0, "y": 0, "w": 24, "h": 5},
        "options": {"mode": "markdown", "content": content},
    }


def stat(title: str, expr: str, *, w: int = 6, h: int = 4, unit: str = "short",
         color: str = "blue", legend: str = "") -> dict:
    return {
        "type": "stat",
        "title": title,
        "gridPos": {"x": 0, "y": 0, "w": w, "h": h},
        "datasource": DS,
        "options": {
            "colorMode": "background",
            "reduceOptions": {"calcs": ["lastNotNull"]},
        },
        "fieldConfig": {"defaults": {
            "unit": unit,
            "color": {"mode": "thresholds"},
            "thresholds": {"mode": "absolute", "steps": [{"value": None, "color": color}]},
        }},
        "targets": [{"expr": expr, "legendFormat": legend or title}],
    }


def timeseries(title: str, exprs, *, w: int = 24, h: int = 8, unit: str = "short",
               style: str = "line") -> dict:
    if isinstance(exprs, str):
        exprs = [(exprs, "")]
    return {
        "type": "timeseries",
        "title": title,
        "gridPos": {"x": 0, "y": 0, "w": w, "h": h},
        "datasource": DS,
        "fieldConfig": {"defaults": {
            "unit": unit,
            "custom": {"lineWidth": 1, "fillOpacity": 15, "drawStyle": style},
        }},
        "targets": [{"expr": e, "legendFormat": lf} for e, lf in exprs],
    }


def gauge(title: str, expr: str, *, w: int = 6, h: int = 6, unit: str = "short",
          mn: float = 0, mx: float = 1) -> dict:
    return {
        "type": "gauge",
        "title": title,
        "gridPos": {"x": 0, "y": 0, "w": w, "h": h},
        "datasource": DS,
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "showThresholdMarkers": True},
        "fieldConfig": {"defaults": {
            "unit": unit, "min": mn, "max": mx,
            "color": {"mode": "thresholds"},
            "thresholds": {"mode": "absolute", "steps": [
                {"value": None, "color": "red"},
                {"value": mn + (mx - mn) * 0.6, "color": "yellow"},
                {"value": mn + (mx - mn) * 0.85, "color": "green"},
            ]},
        }},
        "targets": [{"expr": expr, "legendFormat": title}],
    }


def table(title: str, exprs, *, w: int = 24, h: int = 10) -> dict:
    if isinstance(exprs, str):
        exprs = [exprs]
    return {
        "type": "table",
        "title": title,
        "gridPos": {"x": 0, "y": 0, "w": w, "h": h},
        "datasource": DS,
        "options": {"showHeader": True},
        "targets": [{"expr": e, "legendFormat": "", "instant": True,
                     "refId": chr(65 + i)} for i, e in enumerate(exprs)],
    }


def bargauge(title: str, expr: str, *, w: int = 24, h: int = 8, unit: str = "short",
             legend: str = "") -> dict:
    return {
        "type": "bargauge",
        "title": title,
        "gridPos": {"x": 0, "y": 0, "w": w, "h": h},
        "datasource": DS,
        "options": {"orientation": "horizontal", "displayMode": "gradient",
                    "reduceOptions": {"calcs": ["lastNotNull"]}},
        "fieldConfig": {"defaults": {"unit": unit, "min": 0,
                                     "color": {"mode": "continuous-BlYlRd"}}},
        "targets": [{"expr": expr, "legendFormat": legend or title}],
    }


# ---------------------------------------------------------------------------
# Dashboard assembly. Each spec is (uid, title, tags, about, [panels]).
# ---------------------------------------------------------------------------

def build(uid: str, title: str, tags, about: str, panels: list[dict],
          annotations: list[dict] | None = None) -> dict:
    about_panel = _about(about)
    laid_out = [about_panel]
    # Auto-stack: lay panels left-to-right within a 24-col grid, wrapping rows.
    x = 0
    y = about_panel["gridPos"]["h"]
    row_h = 0
    for i, p in enumerate(panels):
        p = dict(p)
        gp = dict(p["gridPos"])
        w = gp["w"]
        if x + w > 24:
            x = 0
            y += row_h
            row_h = 0
        gp["x"] = x
        gp["y"] = y
        p["gridPos"] = gp
        p["id"] = i + 2  # id 1 reserved-ish; about gets 1
        x += w
        row_h = max(row_h, gp["h"])
        laid_out.append(p)
    about_panel["id"] = 1
    dash = {
        "title": title,
        "uid": uid,
        "schemaVersion": SCHEMA_VERSION,
        "version": 1,
        "refresh": "15s",
        "time": {"from": "now-6h", "to": "now"},
        "tags": list(tags),
        "panels": laid_out,
    }
    if annotations:
        dash["annotations"] = {"list": annotations}
    return dash


def dashboards() -> list[dict]:
    out: list[dict] = []
    P = ["kathir", "platform"]
    M = ["kathir", "model-benchmarks"]

    # ---- 01 Ingestion health -------------------------------------------------
    out.append(build(
        "kathir-01-ingestion", "01 — Ingestion Health", P,
        "### 01 — Ingestion Health\n"
        "Is footage flowing? **Segments** = durable 10-sec clips written per camera; "
        "**windows** = 5-sec AI windows derived from them; **frames decoded** = decode "
        "throughput. Rising `ingest_errors_total` or a flat segment rate means ingestion "
        "is stalling. Metrics: `ingest_*` (ingestion/metrics.py, scraped on :8901).",
        [
            stat("Segments written (total)", "sum(ingest_segments_written_total)", color="green"),
            stat("AI windows created", "sum(ingest_windows_created_total)", color="blue"),
            stat("Frames decoded", "sum(ingest_frames_decoded_total)", color="purple"),
            stat("Ingestion errors", "sum(ingest_errors_total)", color="red"),
            timeseries("Segments written /s (per camera)",
                       [("rate(ingest_segments_written_total[5m])", "{{store_id}}/{{camera_id}}")],
                       unit="ops"),
            timeseries("Windows created /s (per camera)",
                       [("rate(ingest_windows_created_total[5m])", "{{store_id}}/{{camera_id}}")],
                       unit="ops"),
            timeseries("Segment write latency p95 (s)",
                       [("histogram_quantile(0.95, sum(rate(ingest_segment_write_latency_seconds_bucket[5m])) by (le, camera_id))",
                         "{{camera_id}}")], unit="s"),
            timeseries("Ingestion errors /s by stage",
                       [("sum(rate(ingest_errors_total[5m])) by (stage)", "{{stage}}")], unit="ops"),
        ]))

    # ---- 02 Camera health ----------------------------------------------------
    out.append(build(
        "kathir-02-camera", "02 — Camera Health", P,
        "### 02 — Camera Health\n"
        "Per-camera signal quality. `ingest_rtsp_connected` (1=open) shows the feed is "
        "live; `ingest_camera_health_score` is the 0..1 composite. Black-frame / blur / "
        "freeze scores (1=bad) and observed fps (`ingest_camera_fps_actual`) localise *why* "
        "a camera is degraded. Metrics: `ingest_*`.",
        [
            bargauge("Camera health score (0..1)", "ingest_camera_health_score",
                     unit="percentunit", legend="{{camera_id}}"),
            timeseries("RTSP connected (1=open)",
                       [("ingest_rtsp_connected", "{{camera_id}}")], style="bars"),
            timeseries("Observed fps", [("ingest_camera_fps_actual", "{{camera_id}}")]),
            timeseries("Black-frame score (1=dark)",
                       [("ingest_camera_black_frame_score", "{{camera_id}}")], unit="percentunit"),
            timeseries("Blur score (1=blurry)",
                       [("ingest_camera_blur_score", "{{camera_id}}")], unit="percentunit"),
            timeseries("Freeze score (1=frozen)",
                       [("ingest_camera_freeze_score", "{{camera_id}}")], unit="percentunit"),
        ]))

    # ---- 03 Queue / backlog --------------------------------------------------
    out.append(build(
        "kathir-03-queue", "03 — Queue / Backlog", P,
        "### 03 — Queue / Backlog\n"
        "The segment→window→worker pipeline depth (Postgres `job_queue` via SKIP LOCKED, "
        "spec/10). Backlog = windows created but not yet processed by a worker; derived as "
        "`ingest_windows_created_total − model_clips_processed_total`. A persistently growing "
        "gap means workers can't keep up with ingestion. Metrics: `ingest_*`, `model_*`.",
        [
            stat("Windows created", "sum(ingest_windows_created_total)", color="blue"),
            stat("Clips processed (workers)", "sum(model_clips_processed_total)", color="green"),
            stat("Approx backlog",
                 "sum(ingest_windows_created_total) - sum(model_clips_processed_total)",
                 color="orange"),
            timeseries("Enqueue vs process rate /s",
                       [("rate(ingest_windows_created_total[5m])", "enqueued"),
                        ("rate(model_clips_processed_total[5m])", "processed")], unit="ops"),
            timeseries("Approx backlog over time",
                       [("sum(ingest_windows_created_total) - sum(model_clips_processed_total)",
                         "backlog")]),
        ]))

    # ---- 04 CV worker --------------------------------------------------------
    out.append(build(
        "kathir-04-cv-worker", "04 — CV Worker", P,
        "### 04 — CV Worker\n"
        "Detection/tracking workers (DetectionPlugin / TrackingPlugin, spec/11). Filtered to "
        "`task=~\"detection|tracking\"`. Throughput = clips/frames processed; latency p95 from "
        "`model_latency_ms`; request success vs error split. A failed event traces back to a "
        "worker here (Phase-3 done-when). Metrics: `model_*`.",
        [
            stat("Clips processed", "sum(model_clips_processed_total{task=~\"detection|tracking\"})", color="green"),
            stat("Frames processed", "sum(model_frames_processed_total{task=~\"detection|tracking\"})", color="blue"),
            stat("Requests (ok)", "sum(model_requests_total{task=~\"detection|tracking\",status=\"ok\"})", color="green"),
            stat("Errors", "sum(model_errors_total{task=~\"detection|tracking\"})", color="red"),
            timeseries("Detection latency p95 (ms) by model",
                       [("histogram_quantile(0.95, sum(rate(model_latency_ms_bucket{task=~\"detection|tracking\"}[5m])) by (le, model_id))",
                         "{{model_id}}")], unit="ms"),
            timeseries("Frames processed /s by model",
                       [("sum(rate(model_frames_processed_total{task=~\"detection|tracking\"}[5m])) by (model_id)",
                         "{{model_id}}")], unit="ops"),
            timeseries("Errors /s by type",
                       [("sum(rate(model_errors_total{task=~\"detection|tracking\"}[5m])) by (error_type)",
                         "{{error_type}}")], unit="ops"),
        ]))

    # ---- 05 VLM worker -------------------------------------------------------
    out.append(build(
        "kathir-05-vlm-worker", "05 — VLM Worker", P,
        "### 05 — VLM Worker\n"
        "The clip-reasoning VLM plugin host (VLMClipReasoningPlugin). TTFT (`model_ttft_ms`) "
        "and decode tok/s (`model_tokens_per_second`) gate latency; `model_json_parse_total` "
        "tracks structured-output health (aim ~100% ok); `model_verdict_total` shows the "
        "verdict distribution. The VLM only runs on suspicious windows (cascade gating, "
        "spec/11). Metrics: `model_*` filtered `task=\"suspicious_event_verification\"`.",
        [
            stat("VLM requests", "sum(model_requests_total{task=\"suspicious_event_verification\"})", color="purple"),
            stat("Output tokens", "sum(model_output_tokens_total{task=\"suspicious_event_verification\"})", color="blue"),
            stat("JSON parse OK", "sum(model_json_parse_total{result=\"ok\"})", color="green"),
            stat("JSON parse fail", "sum(model_json_parse_total{result=\"fail\"})", color="red"),
            timeseries("TTFT p95 (ms) by model",
                       [("histogram_quantile(0.95, sum(rate(model_ttft_ms_bucket{task=\"suspicious_event_verification\"}[5m])) by (le, model_id))",
                         "{{model_id}}")], unit="ms"),
            timeseries("Decode tok/s by model",
                       [("model_tokens_per_second{task=\"suspicious_event_verification\"}", "{{model_id}}")]),
            timeseries("End-to-end latency p95 (ms)",
                       [("histogram_quantile(0.95, sum(rate(model_latency_ms_bucket{task=\"suspicious_event_verification\"}[5m])) by (le, model_id))",
                         "{{model_id}}")], unit="ms"),
            bargauge("Verdict distribution",
                     "sum(model_verdict_total) by (verdict)", legend="{{verdict}}"),
        ]))

    # ---- 06 GPU --------------------------------------------------------------
    out.append(build(
        "kathir-06-gpu", "06 — GPU", P,
        "### 06 — GPU\n"
        "GPU pressure as seen by the model plugins (`model_gpu_*`, labelled by `gpu`) plus "
        "DCGM exporter where available. Utilization 0..1 and per-model memory bytes tell you "
        "whether a model is GPU-bound or memory-bound on the GB10. (DCGM panels populate once "
        "the `gpu-metrics` compose profile is up.) Metrics: `model_gpu_*`, optional `DCGM_*`.",
        [
            gauge("Max GPU utilization", "max(model_gpu_utilization_ratio)", unit="percentunit"),
            stat("GPU memory used (max)", "max(model_gpu_memory_used_bytes)", w=6, unit="bytes", color="orange"),
            timeseries("GPU utilization by model/gpu",
                       [("model_gpu_utilization_ratio", "{{model_id}}@{{gpu}}")], unit="percentunit"),
            timeseries("GPU memory used by model/gpu",
                       [("model_gpu_memory_used_bytes", "{{model_id}}@{{gpu}}")], unit="bytes"),
            timeseries("DCGM GPU utilization (if exporter up)",
                       [("DCGM_FI_DEV_GPU_UTIL", "{{gpu}}")], unit="percent"),
        ]))

    # ---- 07 Event quality ----------------------------------------------------
    out.append(build(
        "kathir-07-event-quality", "07 — Event Quality", P,
        "### 07 — Event Quality\n"
        "Are the model's events worth acting on? Precision/recall on the golden dataset "
        "(`model_event_precision` / `model_event_recall`, labelled by `dataset`) and the "
        "golden-regression score guard against drift. These are gauges set by the benchmark "
        "harness, not live per-request. Metrics: `model_event_*`, `model_golden_regression_score`.",
        [
            gauge("Event precision (latest)", "max(model_event_precision)", unit="percentunit"),
            gauge("Event recall (latest)", "max(model_event_recall)", unit="percentunit"),
            gauge("Golden regression score", "max(model_golden_regression_score)", unit="percentunit"),
            timeseries("Precision by model/dataset",
                       [("model_event_precision", "{{model_id}}/{{dataset}}")], unit="percentunit"),
            timeseries("Recall by model/dataset",
                       [("model_event_recall", "{{model_id}}/{{dataset}}")], unit="percentunit"),
            table("Quality snapshot (precision / recall / golden)",
                  ["model_event_precision", "model_event_recall", "model_golden_regression_score"]),
        ]))

    # ---- 08 Incidents --------------------------------------------------------
    out.append(build(
        "kathir-08-incidents", "08 — Incidents", P,
        "### 08 — Incidents\n"
        "Loss-prevention incidents = suspicious events the rule engine raised and the VLM "
        "confirmed. Approximated here from VLM verdicts (`model_verdict_total{verdict=...}`) "
        "until the dedicated incident metric lands in the rule engine (Phase 5). An incident "
        "is a *flag to review*, never a confirmed theft (spec/08). Metrics: `model_verdict_total`.",
        [
            stat("Suspicious verdicts", "sum(model_verdict_total{verdict=~\"suspicious|confirmed\"})", color="red"),
            stat("Cleared verdicts", "sum(model_verdict_total{verdict=~\"clear|benign\"})", color="green"),
            stat("Total verdicts", "sum(model_verdict_total)", color="blue"),
            timeseries("Verdicts /s by class",
                       [("sum(rate(model_verdict_total[15m])) by (verdict)", "{{verdict}}")], unit="ops"),
            bargauge("Verdict distribution", "sum(model_verdict_total) by (verdict)", legend="{{verdict}}"),
        ]))

    # ---- 09 Digital twin -----------------------------------------------------
    out.append(build(
        "kathir-09-digital-twin", "09 — Digital Twin", P,
        "### 09 — Digital Twin\n"
        "Store/camera/zone rollup (YAML twin, Phase 10). Until twin-specific metrics exist, "
        "this rolls up the existing per-camera `ingest_*` and event signals by `store_id` so a "
        "second store onboards by YAML with this view already wired. **Template** — panels are "
        "valid and reference real metrics; richer zone overlays land with Phase 10.",
        [
            stat("Cameras reporting", "count(ingest_rtsp_connected)", color="blue"),
            stat("Stores reporting", "count(count(ingest_camera_health_score) by (store_id))", color="purple"),
            timeseries("Segments /s by store",
                       [("sum(rate(ingest_segments_written_total[5m])) by (store_id)", "{{store_id}}")],
                       unit="ops"),
            bargauge("Mean camera health by store",
                     "avg(ingest_camera_health_score) by (store_id)",
                     unit="percentunit", legend="{{store_id}}"),
            table("Per-camera health by store",
                  ["ingest_camera_health_score", "ingest_camera_fps_actual"]),
        ]))

    # ---- 10 TCO --------------------------------------------------------------
    out.append(build(
        "kathir-10-tco", "10 — TCO", P,
        "### 10 — TCO\n"
        "Cost / efficiency of the running platform. Anti-goal #4 (spec/10): measure "
        "video-sec/sec/GPU and clips/min/GPU, not tokens/sec alone. Here: video seconds "
        "processed, clips/min/GPU throughput, and request volume as a cost proxy. Money "
        "deltas across GPU profiles live in dashboard 15. Metrics: `model_*`, `kathirmani_*`.",
        [
            stat("Video seconds processed", "sum(model_video_seconds_processed_total)", unit="s", color="green"),
            stat("Clips processed", "sum(model_clips_processed_total)", color="blue"),
            stat("Total requests", "sum(model_requests_total)", color="purple"),
            timeseries("Video-sec processed /s (throughput)",
                       [("sum(rate(model_video_seconds_processed_total[5m])) by (profile)", "{{profile}}")],
                       unit="ops"),
            timeseries("Clips/min by profile",
                       [("60 * sum(rate(model_clips_processed_total[5m])) by (profile)", "{{profile}}")]),
        ]))

    # ===================== MODEL-BENCHMARK SET 11-18 =========================

    # ---- 11 Model registry ---------------------------------------------------
    out.append(build(
        "kathir-11-model-registry", "11 — Model Registry", M,
        "### 11 — Model Registry\n"
        "Which models/runtimes/profiles are live and serving traffic (spec/11 plugin layer). "
        "Each series is a `(model_id, task, runtime, profile)` tuple from `model_requests_total`. "
        "The NVIDIA-only policy means production defaults are Nemotron/Cosmos; "
        "`research_qwen_baseline` shows as a comparison-only profile. Metrics: `model_*`.",
        [
            stat("Active model series", "count(count(model_requests_total) by (model_id))", color="blue"),
            stat("Active runtimes", "count(count(model_requests_total) by (runtime))", color="purple"),
            stat("Active profiles", "count(count(model_requests_total) by (profile))", color="green"),
            table("Registered models (model / task / runtime / profile)",
                  ["sum(model_requests_total) by (model_id, task, runtime, profile)"]),
            bargauge("Requests by model", "sum(model_requests_total) by (model_id)", legend="{{model_id}}"),
        ]))

    # ---- 12 Model throughput -------------------------------------------------
    out.append(build(
        "kathir-12-model-throughput", "12 — Model Throughput", M,
        "### 12 — Model Throughput\n"
        "How much video each model/profile chews through. clips/min/GPU and "
        "video-sec/sec are the platform's true throughput metrics (anti-goal #4). tok/s is "
        "shown for the generative VLM/LLM only. Compare profiles side by side here ahead of "
        "the Phase-13 bake-off. Metrics: `model_clips_processed_total`, "
        "`model_video_seconds_processed_total`, `model_tokens_per_second`.",
        [
            stat("Clips processed", "sum(model_clips_processed_total)", color="green"),
            stat("Video seconds processed", "sum(model_video_seconds_processed_total)", unit="s", color="blue"),
            stat("Output tokens", "sum(model_output_tokens_total)", color="purple"),
            timeseries("Clips/min by profile",
                       [("60 * sum(rate(model_clips_processed_total[5m])) by (profile)", "{{profile}}")]),
            timeseries("Video-sec/sec by model",
                       [("sum(rate(model_video_seconds_processed_total[5m])) by (model_id)", "{{model_id}}")],
                       unit="ops"),
            timeseries("Decode tok/s by model",
                       [("model_tokens_per_second", "{{model_id}}")]),
        ]))

    # ---- 13 Model latency ----------------------------------------------------
    out.append(build(
        "kathir-13-model-latency", "13 — Model Latency", M,
        "### 13 — Model Latency\n"
        "Latency distribution per model/runtime. p50/p95/p99 of `model_latency_ms` (end-to-end) "
        "and `model_ttft_ms` (time-to-first-token for generative models). p95 is the SLO lens; "
        "watch for runtime-dependent tails (vLLM vs Ollama vs TensorRT-LLM). Metrics: "
        "`model_latency_ms`, `model_ttft_ms`.",
        [
            timeseries("Latency p50 / p95 / p99 (ms)",
                       [("histogram_quantile(0.50, sum(rate(model_latency_ms_bucket[5m])) by (le))", "p50"),
                        ("histogram_quantile(0.95, sum(rate(model_latency_ms_bucket[5m])) by (le))", "p95"),
                        ("histogram_quantile(0.99, sum(rate(model_latency_ms_bucket[5m])) by (le))", "p99")],
                       unit="ms"),
            timeseries("Latency p95 by model/runtime",
                       [("histogram_quantile(0.95, sum(rate(model_latency_ms_bucket[5m])) by (le, model_id, runtime))",
                         "{{model_id}}/{{runtime}}")], unit="ms"),
            timeseries("TTFT p95 by model (generative)",
                       [("histogram_quantile(0.95, sum(rate(model_ttft_ms_bucket[5m])) by (le, model_id))",
                         "{{model_id}}")], unit="ms"),
        ]))

    # ---- 14 Model quality ----------------------------------------------------
    out.append(build(
        "kathir-14-model-quality", "14 — Model Quality", M,
        "### 14 — Model Quality\n"
        "Precision/recall and golden-regression per model/dataset — the accuracy arm of the "
        "bake-off. Pair with throughput (12) and latency (13) to judge whether a faster model "
        "sacrifices quality. JSON-parse success is the structured-output reliability proxy. "
        "Metrics: `model_event_precision`, `model_event_recall`, `model_golden_regression_score`, "
        "`model_json_parse_total`.",
        [
            gauge("Best precision", "max(model_event_precision)", unit="percentunit"),
            gauge("Best recall", "max(model_event_recall)", unit="percentunit"),
            gauge("Best golden score", "max(model_golden_regression_score)", unit="percentunit"),
            bargauge("Precision by model", "max(model_event_precision) by (model_id)",
                     unit="percentunit", legend="{{model_id}}"),
            bargauge("Recall by model", "max(model_event_recall) by (model_id)",
                     unit="percentunit", legend="{{model_id}}"),
            timeseries("JSON parse success ratio by model",
                       [("sum(rate(model_json_parse_total{result=\"ok\"}[15m])) by (model_id) / sum(rate(model_json_parse_total[15m])) by (model_id)",
                         "{{model_id}}")], unit="percentunit"),
        ]))

    # ---- 15 Model TCO --------------------------------------------------------
    out.append(build(
        "kathir-15-model-tco", "15 — Model TCO", M,
        "### 15 — Model TCO\n"
        "Cost efficiency per model/profile: clips/min/GPU and video-sec throughput as the "
        "per-GPU work rate, with GPU memory footprint as the capacity cost. Lets the Phase-13 "
        "bake-off rank GPU/server profiles by cost/store-month. **Template** for currency math "
        "(needs a price input); throughput + footprint are real. Metrics: `model_*`.",
        [
            stat("Video seconds processed", "sum(model_video_seconds_processed_total)", unit="s", color="green"),
            stat("Peak GPU memory", "max(model_gpu_memory_used_bytes)", unit="bytes", color="orange"),
            timeseries("Clips/min/GPU by profile",
                       [("60 * sum(rate(model_clips_processed_total[5m])) by (profile)", "{{profile}}")]),
            bargauge("Peak GPU memory by model", "max(model_gpu_memory_used_bytes) by (model_id)",
                     unit="bytes", legend="{{model_id}}"),
        ]))

    # ---- 16 VSS parity -------------------------------------------------------
    out.append(build(
        "kathir-16-vss-parity", "16 — VSS Parity", M,
        "### 16 — VSS Parity\n"
        "Tracks which VSS-style capabilities the OSS+NVIDIA stack reaches parity on (RT-CV / "
        "embedding / LVS / search / VIOS — Phase 9). **VSS is a reference, never a runtime "
        "dependency** (spec/11). **Template:** proxies parity from quality (precision/recall) + "
        "throughput; a dedicated parity gauge lands in Phase 9. Metrics: `model_event_*`, "
        "`model_clips_processed_total`.",
        [
            bargauge("Capability quality proxy (precision by task)",
                     "max(model_event_precision) by (task)", unit="percentunit", legend="{{task}}"),
            timeseries("Throughput by task (clips/min)",
                       [("60 * sum(rate(model_clips_processed_total[5m])) by (task)", "{{task}}")]),
            table("Parity snapshot (precision / recall by task)",
                  ["max(model_event_precision) by (task)", "max(model_event_recall) by (task)"]),
        ]))

    # ---- 17 Model failure / drift -------------------------------------------
    out.append(build(
        "kathir-17-model-failure-drift", "17 — Model Failure / Drift", M,
        "### 17 — Model Failure / Drift\n"
        "Reliability + drift early-warning. Error rate by type (`model_errors_total`), the "
        "non-ok request ratio, and JSON-parse failures flag operational failure; a falling "
        "`model_golden_regression_score` over time flags quality drift that should trigger a "
        "rollback (Phase 12). Metrics: `model_errors_total`, `model_requests_total`, "
        "`model_json_parse_total`, `model_golden_regression_score`.",
        [
            stat("Total errors", "sum(model_errors_total)", color="red"),
            stat("Parse failures", "sum(model_json_parse_total{result=\"fail\"})", color="orange"),
            gauge("Min golden score (drift floor)", "min(model_golden_regression_score)", unit="percentunit"),
            timeseries("Error ratio by model",
                       [("sum(rate(model_requests_total{status!=\"ok\"}[5m])) by (model_id) / sum(rate(model_requests_total[5m])) by (model_id)",
                         "{{model_id}}")], unit="percentunit"),
            timeseries("Errors /s by type",
                       [("sum(rate(model_errors_total[5m])) by (error_type)", "{{error_type}}")], unit="ops"),
            timeseries("Golden regression score over time (drift)",
                       [("model_golden_regression_score", "{{model_id}}/{{dataset}}")], unit="percentunit"),
        ]))

    # ---- 18 Runtime comparison ----------------------------------------------
    out.append(build(
        "kathir-18-runtime-comparison", "18 — Runtime Comparison", M,
        "### 18 — Runtime Comparison\n"
        "The Phase-13 NVIDIA bake-off lens: same model, different `runtime` "
        "(Ollama / llama.cpp / vLLM / Triton / NIM / TensorRT-LLM). Compares latency p95, "
        "throughput (clips/min, tok/s) and error rate **by runtime** so a config-only profile "
        "swap shows its Grafana deltas (spec/10 Phase 13 done-when). Metrics: `model_*` grouped "
        "by `runtime`.",
        [
            bargauge("Requests by runtime", "sum(model_requests_total) by (runtime)", legend="{{runtime}}"),
            timeseries("Latency p95 by runtime (ms)",
                       [("histogram_quantile(0.95, sum(rate(model_latency_ms_bucket[5m])) by (le, runtime))",
                         "{{runtime}}")], unit="ms"),
            timeseries("Clips/min by runtime",
                       [("60 * sum(rate(model_clips_processed_total[5m])) by (runtime)", "{{runtime}}")]),
            timeseries("Decode tok/s by runtime",
                       [("avg(model_tokens_per_second) by (runtime)", "{{runtime}}")]),
            timeseries("Error ratio by runtime",
                       [("sum(rate(model_requests_total{status!=\"ok\"}[5m])) by (runtime) / sum(rate(model_requests_total[5m])) by (runtime)",
                         "{{runtime}}")], unit="percentunit"),
        ]))

    return out


# ---------------------------------------------------------------------------
# Per-model dashboards (one per model in the catalog + active profiles).
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]


def model_inventory() -> list[dict]:
    """Every model the platform can run → [{id, family, tasks}], deduped.

    Reads the catalog + the non-NVIDIA exceptions + whatever the profiles in
    configs/models.yaml actually bind. Falls back to a static list if PyYAML or the
    files are unavailable, so the generator never hard-fails."""
    seen: dict[str, dict] = {}

    def add(mid: str, family: str = "", tasks=None):
        if not mid:
            return
        e = seen.setdefault(mid, {"id": mid, "family": family, "tasks": []})
        if family and not e["family"]:
            e["family"] = family
        for t in (tasks or []):
            if t not in e["tasks"]:
                e["tasks"].append(t)

    try:
        import yaml
        cat = yaml.safe_load((_REPO / "configs/model_catalog/nvidia_models.yaml").read_text())
        for m in (cat.get("models") or []):
            add(m.get("id"), m.get("family", ""), m.get("tasks"))
        for m in (cat.get("non_nvidia_exceptions") or []):
            add(m.get("id"), "non_nvidia_exception")
        prof = yaml.safe_load((_REPO / "configs/models.yaml").read_text())
        for pname, pcfg in (prof.get("profiles") or {}).items():
            for task, tcfg in (pcfg.get("tasks") or {}).items():
                if isinstance(tcfg, dict) and tcfg.get("model_id"):
                    add(tcfg["model_id"], "", [task])
    except Exception:
        for mid in ["nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
                    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                    "nvidia/C-RADIOv4-H", "nvidia/Cosmos-Reason2-2B",
                    "jameslahm/yoloe-11s-seg"]:
            add(mid)
    return sorted(seen.values(), key=lambda e: e["id"])


def _slug(model_id: str) -> str:
    return model_id.replace("/", "-").replace(".", "-").replace("_", "-").lower()


def model_dashboard(model: dict) -> dict:
    """One dashboard for a single model: its `model_*` traffic/latency/quality + the
    Netdata-sourced `model_resource_*` runtime footprint (set by base/run_hooks when the
    model is kicked off), with run-window annotations on every timeline."""
    mid = model["id"]
    sel = '{model_id="' + mid + '"}'           # PromQL selector for this model
    selb = '{model_id="' + mid + '"}'           # for *_bucket series too (le added below)
    fam = model.get("family") or "model"
    tasks = ", ".join(model.get("tasks") or []) or "—"

    # Grafana run-window annotations pushed by base/run_hooks (tags: model-run + model_id)
    annos = [
        {"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
         "enable": True, "name": "Annotations & Alerts", "type": "dashboard"},
        {"name": "Model runs", "datasource": {"type": "grafana", "uid": "-- Grafana --"},
         "enable": True, "iconColor": "purple", "type": "tags",
         "tags": ["model-run", mid]},
    ]

    panels = [
        # --- traffic / reliability ---
        stat("Runs (total)", f"sum(model_runs_total{sel})", color="blue"),
        stat("Requests (ok)", f'sum(model_requests_total{{model_id="{mid}",status="ok"}})', color="green"),
        stat("Errors", f"sum(model_errors_total{sel})", color="red"),
        stat("Last run (s)", f"max(model_resource_run_seconds{sel})", unit="s", color="purple"),
        # --- latency ---
        timeseries("Latency p50 / p95 / p99 (ms)",
                   [(f"histogram_quantile(0.50, sum(rate(model_latency_ms_bucket{selb}[5m])) by (le))", "p50"),
                    (f"histogram_quantile(0.95, sum(rate(model_latency_ms_bucket{selb}[5m])) by (le))", "p95"),
                    (f"histogram_quantile(0.99, sum(rate(model_latency_ms_bucket{selb}[5m])) by (le))", "p99")],
                   unit="ms"),
        # --- RUNTIME RESOURCES (Netdata via run_hooks) ---
        timeseries("Host CPU during this model's runs (%)",
                   [(f"model_resource_cpu_percent{sel}", "{{task}}/{{runtime}}")], unit="percent"),
        timeseries("Host RAM used during runs (MB)",
                   [(f"model_resource_ram_used_mb{sel}", "{{task}}/{{runtime}}")], unit="decmbytes"),
        timeseries("Disk I/O during runs (KiB/s)",
                   [(f"model_resource_io_read_kbps{sel}", "read"),
                    (f"model_resource_io_write_kbps{sel}", "write")], unit="KiBs"),
        timeseries("GPU power during runs (W)",
                   [(f"model_resource_gpu_power_watts{sel}", "{{runtime}}")], unit="watt"),
        timeseries("GPU memory / utilization (per gpu)",
                   [(f"model_gpu_memory_used_bytes{sel}", "mem {{gpu}}"),
                    (f"model_gpu_utilization_ratio{sel}", "util {{gpu}}")]),
        # --- throughput ---
        timeseries("Throughput — clips/min & video-sec/sec",
                   [(f"60 * sum(rate(model_clips_processed_total{sel}[5m]))", "clips/min"),
                    (f"sum(rate(model_video_seconds_processed_total{sel}[5m]))", "video-sec/sec")]),
        # --- generative-only (empty for non-generative models, harmless) ---
        timeseries("TTFT p95 (ms) & decode tok/s (generative)",
                   [(f"histogram_quantile(0.95, sum(rate(model_ttft_ms_bucket{selb}[5m])) by (le))", "ttft p95 (ms)"),
                    (f"avg(model_tokens_per_second{sel})", "tok/s")]),
        # --- quality ---
        timeseries("Event precision / recall (by dataset)",
                   [(f"model_event_precision{sel}", "precision {{dataset}}"),
                    (f"model_event_recall{sel}", "recall {{dataset}}")], unit="percentunit"),
        table("Run log (runtime / profile / latency)",
              [f"sum(model_runs_total{sel}) by (runtime, profile)"]),
    ]
    return build(
        f"kathir-model-{_slug(mid)}", f"Model — {mid}", ["kathir", "model", "per-model"],
        f"### Model — `{mid}`\n"
        f"Family **{fam}** · tasks: {tasks}. Per-model view: traffic, latency, quality "
        f"(`model_*` filtered to this `model_id`) **plus its runtime footprint on the box** "
        f"— host CPU/RAM/disk-I/O range-queried from **Netdata** and GPU power from "
        f"nvidia-smi over each run window, attributed to this model by `base/run_hooks` "
        f"whenever it is kicked off (spec/04 + spec/16). Purple markers on the timelines "
        f"are this model's run windows.",
        panels, annotations=annos)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate platform Grafana dashboards 01-18.")
    ap.add_argument("--out", default=str(Path(__file__).parent / "dashboards"),
                    help="output directory for *.json")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename prefix derives from the uid: kathir-01-ingestion -> 01-ingestion.json
    written = []
    for dash in dashboards():
        name = dash["uid"].removeprefix("kathir-") + ".json"
        path = out_dir / name
        path.write_text(json.dumps(dash, indent=2, ensure_ascii=False) + "\n")
        written.append(path.name)

    # One dashboard per model, under dashboards/models/.
    models = model_inventory()
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_written = []
    for m in models:
        dash = model_dashboard(m)
        path = model_dir / (dash["uid"].removeprefix("kathir-model-") + ".json")
        path.write_text(json.dumps(dash, indent=2, ensure_ascii=False) + "\n")
        model_written.append(f"models/{path.name}")

    print(f"Wrote {len(written)} platform dashboards + {len(model_written)} per-model "
          f"dashboards to {out_dir}:")
    for n in written + model_written:
        print(f"  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
