"""Phase-13 bake-off CLI.

Run the NVIDIA model/runtime bake-off for a task (fake mode — no GPU/DB), print the
ranked comparison + the recommended production/fallback, and write the report.

Usage:
  .venv/bin/python -m benchmarks.bakeoff.run --task vlm_clip_reasoning
  .venv/bin/python -m benchmarks.bakeoff.run --task vlm_clip_reasoning --no-db
  .venv/bin/python -m benchmarks.bakeoff.run --task vlm_clip_reasoning --rate 8

Real runtimes that are actually reachable on this host (probed by runtimes.py) are
marked `available=True`; everything else is SYNTHETIC. Set
BAKEOFF_FORCE_AVAILABLE="vllm,nim" to force-mark runtimes available (e.g. once a
real vLLM is up) without code changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from benchmarks import tco
    from benchmarks.bakeoff import bakeoff as bk
    from benchmarks.bakeoff import report as rp
    from benchmarks.bakeoff import runtimes as rt
except ImportError:  # direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks import tco
    from benchmarks.bakeoff import bakeoff as bk
    from benchmarks.bakeoff import report as rp
    from benchmarks.bakeoff import runtimes as rt


def print_table(task: str, results: list[bk.RuntimeResult], sel: bk.Selection) -> None:
    print(f"\n=== NVIDIA bake-off: {task} ===")
    print("  ! SYNTHETIC numbers (fake timing model — not a GPU measurement)")
    print(f"  scoring weights: {sel.weights}")
    hdr = (f"  {'rank':<4} {'runtime':<14} {'avail':<6} {'clips/min':>9} "
           f"{'p95 ms':>8} {'tok/s':>7} {'qual':>5} {'₹/1k':>8} {'score':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, r in enumerate(results, 1):
        tps = f"{r.tokens_per_sec:.0f}" if r.tokens_per_sec is not None else "-"
        print(f"  {i:<4} {r.runtime:<14} {str(r.available):<6} "
              f"{r.clips_per_min:>9.1f} {r.p95_latency_ms:>8.0f} {tps:>7} "
              f"{r.quality_score:>5.2f} {r.cost_per_1000_clips:>8.4f} "
              f"{r.balanced_score:>6.3f}")
    print(f"\n  -> PRODUCTION : {sel.production}")
    print(f"  -> FALLBACK   : {sel.fallback}")
    print(f"     {sel.rationale['production']}")
    print(f"     {sel.rationale['fallback']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 13 NVIDIA runtime bake-off (fake).")
    ap.add_argument("--task", default="vlm_clip_reasoning",
                    help=f"bake-off task (known: {sorted(rt.RUNTIME_MATRIX)})")
    ap.add_argument("--rate", type=float, default=tco.DEFAULT_RATE_INR_PER_KWH,
                    help=f"electricity ₹/kWh (default {tco.DEFAULT_RATE_INR_PER_KWH})")
    ap.add_argument("--no-db", action="store_true", help="never write to Postgres")
    args = ap.parse_args(argv)

    results = bk.run_bakeoff(args.task, rate_inr_per_kwh=args.rate)
    sel = bk.select_profiles(results, task=args.task)
    print_table(args.task, results, sel)

    report = rp.build_report(args.task, results, sel)
    path = rp.write_report(report)
    db_status = "skipped (--no-db)" if args.no_db else rp.write_db_rows(report)
    print(f"\n  report: {path}")
    print(f"  db    : {db_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
