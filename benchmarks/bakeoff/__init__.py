"""Phase 13 — NVIDIA model/runtime bake-off + production profile selection.

The platform's FINAL phase. Compares the SAME NVIDIA model across runtimes
(Ollama / llama.cpp / vLLM / TensorRT-LLM / NIM / transformers), ranks them on a
documented balanced score (throughput / latency / quality / cost), and selects a
**production** + **fallback** profile. Builds ON the Phase-11 harness
(`benchmarks.harness`) + TCO (`benchmarks.tco`); runs in **fake mode** because most
runtimes are not installed/proven on this aarch64/Blackwell box.

>>> ALL throughput/latency/cost numbers here are SYNTHETIC (fake timing model)
    until the candidate runtimes actually run on the GPU. The structure, scoring,
    selection, and rollback are real.

Modules:
  runtimes.py — candidate matrix + real `available()` probes + synthetic profiles
  bakeoff.py  — run via the harness, rank, `select_profiles()` (production+fallback)
  rollback.py — pure profile-history + thin DB/file effect (same path as the API)
  report.py   — bakeoff_report.json + guarded model_benchmark_runs inserts
  run.py      — CLI: `python -m benchmarks.bakeoff.run --task vlm_clip_reasoning`

Repo root onto sys.path so `benchmarks.*` imports work without an editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
