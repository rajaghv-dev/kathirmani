#!/usr/bin/env python3
"""
Marlin-2B inference on all videos in this directory.

Usage:
    python run_inference.py                        # all videos
    python run_inference.py --video "Bill Counter" # one video (substring match)
    python run_inference.py --no-compile           # skip torch.compile (faster startup)
    python run_inference.py --metrics-port 8900
"""
import os
import ctypes
import pathlib
import sys

# ---------------------------------------------------------------------------
# Pre-load PyAV's bundled FFmpeg libs with RTLD_GLOBAL so torchcodec can find
# them without system FFmpeg or a manually set LD_LIBRARY_PATH.
# Must happen before ANY import of torch / torchcodec / transformers.
# ---------------------------------------------------------------------------
def _preload_av_ffmpeg():
    candidates = [
        pathlib.Path(sys.prefix) / "lib/python3.12/site-packages/av.libs",
        pathlib.Path(__file__).parent / ".venv/lib/python3.12/site-packages/av.libs",
    ]
    avlibs = next((p for p in candidates if p.exists()), None)
    if avlibs is None:
        return
    # Load in dependency order with RTLD_GLOBAL so subsequent dlopen calls find them
    for name in [
        "libavutil.so.60", "libswresample.so.6", "libswscale.so.9",
        "libavcodec.so.62", "libavformat.so.62", "libavfilter.so.11",
    ]:
        so = avlibs / name
        if so.exists():
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_av_ffmpeg()

# Prevent transformers / huggingface_hub from phoning home during load.
# The model is already in models/Marlin-2B/ — no network access needed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# These env vars must be set before importing transformers / qwen-vl-utils.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("VIDEO_MAX_PIXELS", "200704")
os.environ.setdefault("FPS", "2.0")
os.environ.setdefault("FPS_MAX_FRAMES", "240")
os.environ.setdefault("FPS_MIN_FRAMES", "4")

import argparse
import time
from pathlib import Path

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default=str(HERE), help="Directory with videos")
    parser.add_argument("--results-dir", default=str(HERE / "results"), help="Output directory")
    parser.add_argument("--metrics-port", type=int, default=8900, help="Prometheus metrics port")
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile")
    parser.add_argument("--video", default=None, help="Run only one video (substring of filename)")
    parser.add_argument("--duration", type=float, default=None, help="Only analyse first N seconds of each video")
    parser.add_argument("--model-path", default=None, help="Path to local model dir (default: models/Marlin-2B)")
    parser.add_argument("--keep-alive", action="store_true", help="Keep metrics server running after inference (default: exit)")
    parser.add_argument("--skip-qwen", action="store_true", help="Skip Qwen2.5-VL spatial analysis")
    args = parser.parse_args()

    from inference.metrics import start_metrics_server
    from inference.loki import log_inference_start, log_run_complete
    start_metrics_server(args.metrics_port)

    from inference.pipeline import load_model, run_all, process_video, LOCAL_MODEL_PATH

    model_path = Path(args.model_path) if args.model_path else LOCAL_MODEL_PATH

    t0 = time.time()
    model = load_model(model_path=model_path, compile=not args.no_compile)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    if args.video:
        video_dir = Path(args.video_dir)
        matches = [
            v for ext in ("*.mkv", "*.mp4", "*.avi", "*.mov", "*.webm")
            for v in video_dir.glob(ext)
            if args.video.lower() in v.stem.lower()
        ]
        if not matches:
            print(f"[main] No video matching '{args.video}' found in {video_dir}")
            return
        log_inference_start([matches[0].stem])
        results = [process_video(model, matches[0], results_dir, duration=args.duration)]
    else:
        video_dir = Path(args.video_dir)
        all_videos = [v for ext in ("*.mkv","*.mp4","*.avi","*.mov","*.webm") for v in sorted(video_dir.glob(ext))]
        log_inference_start([v.stem for v in all_videos])
        results = run_all(model, video_dir=video_dir, results_dir=results_dir, duration=args.duration)

    elapsed = time.time() - t0
    log_run_complete(len(results), sum(len(r.get("events",[])) for r in results), elapsed)
    from inference.annotations import annotate_run_complete
    annotate_run_complete(len(results), sum(len(r.get("events",[])) for r in results), elapsed)

    from inference.metrics import compute_economy
    economy = compute_economy(
        total_events=sum(len(r.get("events", [])) for r in results),
        wall_time_sec=elapsed,
    )
    # Save economy data alongside results
    import json
    if economy:
        (Path(args.results_dir) / "economy.json").write_text(json.dumps(economy, indent=2))

    # --- Qwen2.5-VL spatial analysis (optional, runs after Marlin) ---
    qwen_path = HERE / "models" / "Qwen2.5-VL-7B-Instruct"
    if qwen_path.exists() and not args.skip_qwen:
        from inference.qwen_vl import load_qwen_model, analyze_video, QWEN_QUERIES
        from inference.pipeline import _GPU_LOCK
        print("[main] Loading Qwen2.5-VL-7B for spatial analysis ...")
        qwen_model, qwen_processor = load_qwen_model(qwen_path)

        results_dir = Path(args.results_dir)
        for result in results:
            label = result["label"]
            # Use the trimmed clip if duration was specified; otherwise locate original
            video_file = next(
                (v for ext in ("*.mkv", "*.mp4", "*.avi", "*.mov", "*.webm")
                 for v in Path(args.video_dir).glob(ext)
                 if v.stem == label), None
            )
            if video_file:
                qwen_result = analyze_video(qwen_model, qwen_processor, video_file, label, _GPU_LOCK)
                # Log each Qwen answer to Loki
                from inference.loki import log_qwen_answer
                for query, res in qwen_result.get("queries", {}).items():
                    if res.get("ok") and res.get("answer"):
                        log_qwen_answer(label, query, res["answer"])
                # Merge into existing result JSON
                existing = json.loads((results_dir / f"{label}.json").read_text())
                existing["qwen_analysis"] = qwen_result
                (results_dir / f"{label}.json").write_text(json.dumps(existing, indent=2))
                print(f"[main] ✓ Qwen2.5-VL done for {label}: {len(QWEN_QUERIES)} queries")
    else:
        if not qwen_path.exists():
            print("[main] Qwen2.5-VL model not found — run: python download_model.py --model qwen-vl")

    _print_summary(results, elapsed, args)


def _grafana_host() -> str:
    """Best-guess host IP for Grafana URLs (prefers non-loopback)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def _print_summary(results: list, elapsed: float, args) -> None:
    total_events = sum(len(r.get("events", [])) for r in results)
    host = _grafana_host()

    sep = "=" * 62
    print(f"\n{sep}")
    print("  INFERENCE COMPLETE")
    print(sep)
    print(f"  Videos processed : {len(results)}")
    print(f"  Events detected  : {total_events}")
    print(f"  Wall time        : {elapsed:.1f}s")
    print(f"  Results saved to : {args.results_dir}")
    print()
    print("  GRAFANA DASHBOARDS")
    print(f"  {'Model & Runtime':20s}  http://{host}:3000/d/marlin-model-runtime")
    print(f"  {'Application/Pipeline':20s}  http://{host}:3000/d/marlin-pipeline")
    print(f"  {'Camera Views':20s}  http://{host}:3000/d/marlin-cameras")
    print(f"  {'Store View (Fused)':20s}  http://{host}:3000/d/marlin-fused")
    print(f"  {'Visual Analysis (Qwen)':20s}  http://{host}:3000/d/marlin-qwen")
    print(f"  {'Inference Logs (Loki)':20s}  http://{host}:3000/d/marlin-loki")
    print(f"  {'Compute Economy':20s}  http://{host}:3000/d/marlin-economy")
    print(f"  {'Login':20s}  admin / admin")
    print()
    print("  Dashboards auto-refresh every 15s — data visible immediately.")
    print(sep)

    if args.keep_alive:
        print("\n[main] --keep-alive: holding metrics server open. Ctrl+C to exit.")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
