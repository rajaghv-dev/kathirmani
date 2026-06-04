"""Phase 11 — Benchmarking + TCO.

A no-GPU-by-default benchmark harness for the Kathirmani video-AI platform. It
times a pluggable "runner" over N synthetic items, computes throughput/latency
(p50/p95, clips/min, video-sec/sec, tokens/sec) and a TCO cost model
(cost/1000-clips, cost/camera-month, cost/store-month), and records a row to the
`model_benchmark_runs` table (master plan §A6.4 / Phase 11).

The default runner is a *fake timing model* — no GPU, no DB, no weights. Numbers
produced in fake mode are SYNTHETIC placeholders until a real worker/plugin
runner is wired in and run on the GPU.
"""
from __future__ import annotations
