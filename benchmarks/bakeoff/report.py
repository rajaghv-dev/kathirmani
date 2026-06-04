"""Bake-off reporting — write bakeoff_report.json + insert model_benchmark_runs.

Takes the ranked `RuntimeResult`s + the `Selection` and produces:
  - a `bakeoff_report.json` artifact (per-runtime metrics + production/fallback +
    rationale + the synthetic warning), and
  - one row per runtime in `model_benchmark_runs` (guarded: a no-DB box just gets a
    'skipped (...)' status; never raises).

The DB write reuses the same table + column shape as `benchmarks/run.py` so the
bake-off rows live alongside the Phase-11 benchmark rows and feed dashboard 18.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from benchmarks.bakeoff.bakeoff import RuntimeResult, Selection
except ImportError:  # path fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.bakeoff.bakeoff import RuntimeResult, Selection

_REPORT_DIR = Path(__file__).resolve().parent / "results"
SYNTHETIC_WARNING = (
    "SYNTHETIC — fake timing model, not measured on a GPU. The bake-off structure, "
    "scoring, and rollback are real; throughput/latency/cost are placeholders until "
    "the candidate runtimes (vLLM/Triton/NIM/TensorRT-LLM) run on the GPU.")


def build_report(
    task: str,
    results: list[RuntimeResult],
    selection: Selection,
    *,
    profile_name: str = "nvidia_gb10_retail_balanced",
    gpu_name: str = "DGX_SPARK_GB10",
    dataset_name: str = "golden_retail_loss_v1",
) -> dict[str, Any]:
    """Assemble the bake-off report dict (pure — no I/O)."""
    return {
        "task": task,
        "model_profile_name": profile_name,
        "gpu_name": gpu_name,
        "dataset_name": dataset_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
        "synthetic_warning": SYNTHETIC_WARNING,
        "weights": selection.weights,
        "production": selection.production,
        "fallback": selection.fallback,
        "rationale": selection.rationale,
        # ranked, best-first
        "runtimes": [r.as_dict() for r in results],
    }


def write_report(report: dict[str, Any], *, out_dir: Path | None = None) -> Path:
    """Write bakeoff_report.json (+ a per-task copy). Returns the canonical path.

    `out_dir` defaults to the module-level `_REPORT_DIR` resolved at call time so a
    test can monkeypatch it.
    """
    out_dir = out_dir if out_dir is not None else _REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = out_dir / "bakeoff_report.json"
    canonical.write_text(json.dumps(report, indent=2))
    # Also keep a per-task copy so successive tasks don't clobber each other.
    (out_dir / f"bakeoff_{report['task']}.json").write_text(json.dumps(report, indent=2))
    return canonical


# ---- DB write (guarded — mirrors benchmarks/run.write_db_row) ---------------
def write_db_rows(report: dict[str, Any]) -> str:
    """Insert one model_benchmark_runs row per runtime. Returns a status string.

    Guarded: import errors / no Postgres / no psycopg -> 'skipped (...)'.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from db import connect  # type: ignore
    except Exception as e:  # pragma: no cover - import-time guard
        return f"skipped (db module unavailable: {e})"
    try:
        written = 0
        with connect() as conn, conn.cursor() as cur:
            for r in report["runtimes"]:
                meta = {
                    "bakeoff_task": report["task"],
                    "runtime": r["runtime"],
                    "available": r["available"],
                    "balanced_score": r["balanced_score"],
                    "norm": r["norm"],
                    "is_production": r["runtime"] == report["production"],
                    "is_fallback": r["runtime"] == report["fallback"],
                    "ttft_ms_p95": r["ttft_ms_p95"],
                    "tokens_per_sec": r["tokens_per_sec"],
                    "quality_score": r["quality_score"],
                    "avg_power_w": r["avg_power_w"],
                    "tco": r["tco"],
                    "synthetic": True,
                }
                cur.execute(
                    "INSERT INTO model_benchmark_runs ("
                    " benchmark_name, model_profile_name, gpu_name, dataset_name,"
                    " total_clips, clips_per_min_gpu, p95_latency_ms,"
                    " parse_success_rate, estimated_cost_per_1000_clips, metadata_json)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        f"bakeoff_{report['task']}_{r['runtime']}",
                        report["model_profile_name"],
                        report["gpu_name"],
                        report["dataset_name"],
                        None,  # total_clips: n/a per-runtime headline
                        r["clips_per_min"],
                        r["p95_latency_ms"],
                        r["quality_score"],  # quality proxy in the parse-rate column
                        r["cost_per_1000_clips_inr"],
                        json.dumps(meta),
                    ),
                )
                written += 1
            conn.commit()
        return f"wrote {written} model_benchmark_runs rows"
    except Exception as e:  # pragma: no cover - runtime DB guard
        return f"skipped (db unavailable: {e})"
