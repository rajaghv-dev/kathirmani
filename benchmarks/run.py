"""Phase 11 benchmark CLI.

Run a named benchmark (fake mode by default — no GPU), print a report, write a
row to `model_benchmark_runs` (guarded: skipped if no Postgres), and dump a JSON
to benchmarks/results/.

Usage:
  .venv/bin/python -m benchmarks.run --config benchmarks/configs/05_vlm_verify_100.yaml
  .venv/bin/python -m benchmarks.run --all                 # every config
  .venv/bin/python -m benchmarks.run --config ... --rate 8  # custom ₹/kWh
  .venv/bin/python -m benchmarks.run --config ... --no-db   # never touch DB

Real (non-fake) mode: pass --runner module:factory pointing at a callable that
returns a runner(item)->RunnerResult driving an actual worker/plugin on the GPU
(see README in this module's docstring + integration notes). Default is fake.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

# Tolerate both `python -m benchmarks.run` and direct `python benchmarks/run.py`.
try:
    from benchmarks.harness import FakeRunner, RunnerResult, run_benchmark
    from benchmarks import tco
except ImportError:  # direct-script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.harness import FakeRunner, RunnerResult, run_benchmark
    from benchmarks import tco

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---- Config + synthetic items ----------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_items(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Synthesize N items carrying their index + per-item video seconds."""
    n = int(cfg["n_items"])
    vps = float(cfg.get("video_sec_per_item", 0.0))
    return [{"index": i, "n": n, "video_sec": vps} for i in range(n)]


def build_runner(cfg: dict[str, Any], spec: str | None) -> Callable[[dict], RunnerResult]:
    """Default = FakeRunner from the config's `fake:` block.

    `spec` ("module:attr") loads a real runner factory: factory(cfg) -> runner.
    """
    if spec:
        mod_name, _, attr = spec.partition(":")
        import importlib
        factory = getattr(importlib.import_module(mod_name), attr or "make_runner")
        return factory(cfg)
    fk = cfg.get("fake", {}) or {}
    return FakeRunner(
        task=cfg.get("task", "detection"),
        base_latency_ms=float(fk.get("base_latency_ms", 40.0)),
        jitter_ms=float(fk.get("jitter_ms", 10.0)),
        power_w=float(fk.get("power_w", 60.0)),
        tokens_per_item=int(fk.get("tokens_per_item", 0)),
        ttft_ms=float(fk.get("ttft_ms", 0.0)),
        stages=tuple(fk.get("stages", ("preprocess", "infer", "postprocess"))),
        sleep=bool(fk.get("sleep", False)),
    )


# ---- DB write (guarded) -----------------------------------------------------
def write_db_row(payload: dict[str, Any]) -> str:
    """Insert a model_benchmark_runs row. Returns a status string (never raises).

    Guarded: import errors / no Postgres / no psycopg → 'skipped (...)'.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "db"))
        from db import connect  # type: ignore
    except Exception as e:  # pragma: no cover - import-time guard
        return f"skipped (db module unavailable: {e})"
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_benchmark_runs ("
                " benchmark_name, model_profile_name, gpu_name, dataset_name,"
                " total_clips, clips_per_min_gpu, p95_latency_ms,"
                " parse_success_rate, estimated_cost_per_1000_clips, metadata_json)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (
                    payload["benchmark_name"],
                    payload["model_profile_name"],
                    payload.get("gpu_name"),
                    payload.get("dataset_name"),
                    payload.get("total_clips"),
                    payload.get("clips_per_min_gpu"),
                    payload.get("p95_latency_ms"),
                    payload.get("parse_success_rate"),
                    payload.get("estimated_cost_per_1000_clips"),
                    json.dumps(payload.get("metadata_json", {})),
                ),
            )
            row_id = cur.fetchone()[0]
            conn.commit()
            return f"wrote model_benchmark_runs id={row_id}"
    except Exception as e:  # pragma: no cover - runtime DB guard
        return f"skipped (db unavailable: {e})"


# ---- Orchestration ----------------------------------------------------------
def run_one(
    cfg_path: str | Path,
    *,
    rate: float = tco.DEFAULT_RATE_INR_PER_KWH,
    runner_spec: str | None = None,
    use_db: bool = True,
) -> dict[str, Any]:
    """Run a single benchmark config → metrics + TCO + persisted artifacts."""
    cfg = load_config(cfg_path)
    items = make_items(cfg)
    runner = build_runner(cfg, runner_spec)
    res = run_benchmark(
        cfg["name"], cfg.get("task", "detection"), items, runner,
        gpu_name=cfg.get("gpu_name"))

    n_cameras = int(cfg.get("n_cameras", cfg.get("streams", 1)) or 1)
    cost = tco.tco_report(res.avg_power_w, res.clips_per_min, n_cameras, rate_inr_per_kwh=rate)

    is_fake = runner_spec is None
    report = {
        "benchmark_name": res.name,
        "task": res.task,
        "model_profile_name": cfg.get("model_profile_name", "unknown"),
        "gpu_name": res.gpu_name,
        "dataset_name": cfg.get("dataset_name"),
        "total_clips": res.n_items,
        "clips_per_min_gpu": res.clips_per_min,
        "video_sec_per_sec_gpu": res.video_sec_per_sec,
        "p50_latency_ms": res.p50_latency_ms,
        "p95_latency_ms": res.p95_latency_ms,
        "mean_latency_ms": res.mean_latency_ms,
        "ttft_ms_p95": res.ttft_ms_p95,
        "tokens_per_sec": res.tokens_per_sec,
        "parse_success_rate": res.parse_success_rate,
        "avg_power_w": res.avg_power_w,
        "estimated_cost_per_1000_clips": cost["cost_per_1000_clips_inr"],
        "tco": cost,
        "fake": is_fake,
        "synthetic_warning": (
            "SYNTHETIC — fake timing model, not measured on a GPU" if is_fake
            else "real runner"),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "metadata_json": {
            "fake": is_fake,
            "n_cameras": n_cameras,
            "streams": cfg.get("streams"),
            "video_sec_per_sec_gpu": res.video_sec_per_sec,
            "tokens_per_sec": res.tokens_per_sec,
            "ttft_ms_p95": res.ttft_ms_p95,
            "p50_latency_ms": res.p50_latency_ms,
            "stage_ms_mean": res.stage_ms_mean,
            "tco": cost,
            "synthetic": is_fake,
        },
    }

    # JSON artifact (always).
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{res.name}.json"
    out_path.write_text(json.dumps(report, indent=2))
    report["_json_path"] = str(out_path)

    # DB row (guarded).
    report["_db_status"] = write_db_row(report) if use_db else "skipped (--no-db)"
    return report


def print_report(r: dict[str, Any]) -> None:
    print(f"\n=== {r['benchmark_name']} ({r['task']}) ===")
    if r["fake"]:
        print("  ! SYNTHETIC numbers (fake timing model — not a GPU measurement)")
    print(f"  profile           : {r['model_profile_name']}  gpu={r['gpu_name']}")
    print(f"  clips             : {r['total_clips']}")
    print(f"  clips/min/GPU     : {r['clips_per_min_gpu']:.1f}")
    print(f"  video-sec/sec/GPU : {r['video_sec_per_sec_gpu']:.2f}")
    print(f"  latency p50/p95   : {r['p50_latency_ms']:.1f} / {r['p95_latency_ms']:.1f} ms")
    if r["tokens_per_sec"] is not None:
        print(f"  VLM ttft p95      : {r['ttft_ms_p95']:.1f} ms")
        print(f"  VLM tokens/sec    : {r['tokens_per_sec']:.1f}")
        print(f"  parse success     : {r['parse_success_rate']:.2%}")
    print(f"  avg power (W)      : {r['avg_power_w']:.1f}")
    t = r["tco"]
    print(f"  cost/1000-clips    : ₹{t['cost_per_1000_clips_inr']:.4f}")
    print(f"  cost/camera-month  : ₹{t['cost_per_camera_month_inr']:.2f}")
    print(f"  cost/store-month   : ₹{t['cost_per_store_month_inr']:.2f}")
    print(f"  json               : {r['_json_path']}")
    print(f"  db                 : {r['_db_status']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11 benchmark runner (fake by default).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="path to a benchmark YAML config")
    g.add_argument("--all", action="store_true", help="run every config in configs/")
    ap.add_argument("--rate", type=float, default=tco.DEFAULT_RATE_INR_PER_KWH,
                    help=f"electricity ₹/kWh (default {tco.DEFAULT_RATE_INR_PER_KWH})")
    ap.add_argument("--runner", default=None,
                    help="real runner factory 'module:attr' (default = fake)")
    ap.add_argument("--no-db", action="store_true", help="never write to Postgres")
    args = ap.parse_args(argv)

    cfgs = (sorted(_CONFIG_DIR.glob("*.yaml")) if args.all else [Path(args.config)])
    for cfg_path in cfgs:
        r = run_one(cfg_path, rate=args.rate, runner_spec=args.runner, use_db=not args.no_db)
        print_report(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
