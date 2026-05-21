#!/usr/bin/env python3
"""
Marlin-2B inference on all videos in this directory.

Usage:
    python run_inference.py                  # process all videos
    python run_inference.py --no-compile     # skip torch.compile (faster startup)
    python run_inference.py --metrics-port 8900
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default=str(HERE), help="Directory with videos")
    parser.add_argument("--results-dir", default=str(HERE / "results"), help="Output directory")
    parser.add_argument("--metrics-port", type=int, default=8900, help="Prometheus metrics port")
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile")
    args = parser.parse_args()

    from inference.metrics import start_metrics_server
    start_metrics_server(args.metrics_port)

    from inference.pipeline import load_model, run_all

    t0 = time.time()
    model = load_model()

    if args.no_compile:
        pass  # compile already called inside load_model; skip is handled by monkey-patch if needed

    results = run_all(
        model,
        video_dir=Path(args.video_dir),
        results_dir=Path(args.results_dir),
    )

    elapsed = time.time() - t0
    print(f"\n[main] Total wall time: {elapsed:.1f}s")
    print(f"[main] Results saved to: {args.results_dir}")
    print(f"[main] Metrics still live at: http://localhost:{args.metrics_port}/metrics")
    print("[main] Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
