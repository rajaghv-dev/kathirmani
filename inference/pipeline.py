"""Core Marlin-2B inference pipeline for security camera videos."""
import json
import time
import threading
from pathlib import Path

import torch

from .metrics import (
    INFERENCE_DURATION, EVENTS_DETECTED, VIDEOS_PROCESSED,
    FIND_SPAN_START, FIND_SPAN_END, FIND_PARSE_OK, poll_gpu_metrics
)

# Queries tailored for retail store surveillance footage
FIND_QUERIES = [
    "customer enters the store",
    "customer leaves the store",
    "cash payment transaction at counter",
    "person picks up items from shelf",
    "employee interacts with customer",
    "person using mobile phone",
    "group of people gathered",
    "person walking through aisle",
    "suspicious or unusual behavior",
    "person standing near entry door",
    "crowded area with multiple customers",
    "person carrying a bag",
]


def load_model(model_id: str = "NemoStation/Marlin-2B") -> object:
    print(f"[pipeline] Loading {model_id} ...")
    model = torch.hub.load if False else None  # placeholder guard

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    print("[pipeline] Model loaded. Compiling ...")
    model.compile()
    print("[pipeline] Ready.")
    return model


def _run_caption(model, video_path: str, label: str) -> dict:
    with INFERENCE_DURATION.labels(video=label, mode="caption").time():
        result = model.caption(video_path)
    return result


def _run_find(model, video_path: str, label: str, query: str) -> dict:
    with INFERENCE_DURATION.labels(video=label, mode="find").time():
        result = model.find(video_path, event=query)
    return result


def process_video(model, video_path: Path, results_dir: Path) -> dict:
    label = video_path.stem
    print(f"\n{'='*60}")
    print(f"[pipeline] Processing: {label}")
    print(f"{'='*60}")

    output = {
        "video": video_path.name,
        "label": label,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caption": None,
        "events": [],
        "find_results": {},
    }

    # --- Caption mode ---
    try:
        print(f"[pipeline]   Running caption mode ...")
        cap_result = _run_caption(model, str(video_path), label)
        output["caption"] = cap_result.get("caption", "")
        output["scene"] = cap_result.get("scene", "")
        output["events"] = cap_result.get("events", [])
        EVENTS_DETECTED.labels(video=label).set(len(output["events"]))
        print(f"[pipeline]   Detected {len(output['events'])} events.")
        for ev in output["events"]:
            print(f"            [{ev.get('start',0):.1f}s – {ev.get('end',0):.1f}s] {ev.get('description','')}")
    except Exception as e:
        print(f"[pipeline]   Caption failed: {e}")
        output["caption_error"] = str(e)

    # --- Find mode ---
    print(f"[pipeline]   Running {len(FIND_QUERIES)} find queries ...")
    for query in FIND_QUERIES:
        try:
            find_result = _run_find(model, str(video_path), label, query)
            span = find_result.get("span")
            ok = find_result.get("format_ok", False)
            output["find_results"][query] = {
                "raw": find_result.get("raw", ""),
                "span": list(span) if span else None,
                "format_ok": ok,
            }
            if span:
                FIND_SPAN_START.labels(video=label, query=query).set(span[0])
                FIND_SPAN_END.labels(video=label, query=query).set(span[1])
            FIND_PARSE_OK.labels(video=label, query=query).set(1 if ok else 0)
            status = f"[{span[0]:.1f}s–{span[1]:.1f}s]" if span else "not found"
            print(f"            '{query}' → {status}")
        except Exception as e:
            output["find_results"][query] = {"error": str(e)}

    VIDEOS_PROCESSED.labels(status="success").inc()

    # Save JSON result
    out_file = results_dir / f"{label}.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"[pipeline]   Saved → {out_file}")
    return output


def run_all(model, video_dir: Path, results_dir: Path) -> list[dict]:
    videos = sorted(video_dir.glob("*.mkv")) + \
             sorted(video_dir.glob("*.mp4")) + \
             sorted(video_dir.glob("*.avi")) + \
             sorted(video_dir.glob("*.mov")) + \
             sorted(video_dir.glob("*.webm"))

    if not videos:
        print("[pipeline] No video files found in", video_dir)
        return []

    print(f"[pipeline] Found {len(videos)} videos: {[v.name for v in videos]}")
    results_dir.mkdir(exist_ok=True)

    # Start GPU polling
    stop_gpu = threading.Event()
    gpu_thread = threading.Thread(target=poll_gpu_metrics, args=(stop_gpu,), daemon=True)
    gpu_thread.start()

    all_results = []
    for video in videos:
        try:
            r = process_video(model, video, results_dir)
            all_results.append(r)
        except Exception as e:
            print(f"[pipeline] FAILED {video.name}: {e}")
            VIDEOS_PROCESSED.labels(status="error").inc()

    stop_gpu.set()

    # Write summary
    summary = {
        "total_videos": len(videos),
        "processed": len(all_results),
        "videos": [r["label"] for r in all_results],
        "total_events": sum(len(r.get("events", [])) for r in all_results),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[pipeline] Done. Summary: {summary}")
    return all_results
