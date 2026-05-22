"""Core Marlin-2B inference pipeline for security camera videos."""
import json
import time
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
import torch

from .metrics import (
    INFERENCE_DURATION, EVENTS_DETECTED, VIDEOS_PROCESSED,
    FIND_SPAN_START, FIND_SPAN_END, FIND_PARSE_OK, poll_dgx_metrics,
    FUSED_EVENTS, FUSED_TOTAL_RAW, FUSED_FIND_START, FUSED_FIND_END, FUSED_FIND_CAM,
)

_GPU_LOCK = threading.Lock()

# Kathirmani store-specific surveillance queries
# Each maps to a Prometheus metric label: marlin_find_span_*{query="..."}
FIND_QUERIES = [
    # Entry / Exit
    "customer enters Kathirmani store",
    "customer exits Kathirmani store",
    # Bill counter area
    "cash payment at bill counter",
    "billing staff at counter",
    "customer queue at billing",
    # Shopping floor
    "customer browsing shelves",
    "person picking item from rack",
    "customer carrying shopping basket",
    # Security
    "unattended bag or package",
    "person loitering near exit",
    "suspicious or unusual behavior",
    # Crowd & operations
    "crowded aisle or bottleneck",
    # Loss prevention — temporal signals (when these behaviors occur)
    "customer leaving store without approaching billing counter",
    "person handling merchandise near store exit or boundary",
    "group of customers gathered around a product display blocking view",
    "person placing merchandise into personal bag or clothing at shelf",
    "customer returning to the same shelf section multiple times",
    "billing staff engaged by customer while another person handles merchandise",
    "person carrying large bag or backpack near merchandise shelves",
    "customer switching or swapping product packaging at shelf",
]


MODEL_ID = "NemoStation/Marlin-2B"
LOCAL_MODEL_PATH = Path(__file__).parent.parent / "models" / "Marlin-2B"

_tmp_clips: list[Path] = []  # track temp files for cleanup


def trim_video(src: Path, duration_sec: float) -> Path:
    """Decode and re-encode the first `duration_sec` seconds of src to a temp MP4."""
    tmp = Path(tempfile.mktemp(suffix=".mp4"))
    _tmp_clips.append(tmp)
    with av.open(str(src)) as inp:
        in_vs = inp.streams.video[0]
        fps = float(in_vs.average_rate or 25)
        with av.open(str(tmp), "w", format="mp4") as out:
            from fractions import Fraction
            out_vs = out.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
            out_vs.width = in_vs.width
            out_vs.height = in_vs.height
            out_vs.pix_fmt = "yuv420p"
            out_vs.options = {"crf": "23", "preset": "ultrafast"}
            for frame in inp.decode(in_vs):
                t = float(frame.pts * in_vs.time_base) if frame.pts is not None else 0.0
                if t > duration_sec:
                    break
                frame = frame.reformat(format="yuv420p")
                for pkt in out_vs.encode(frame):
                    out.mux(pkt)
            for pkt in out_vs.encode():
                out.mux(pkt)
    return tmp


def cleanup_tmp():
    for p in _tmp_clips:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def load_model(model_path: Path | None = None, compile: bool = True) -> object:
    resolved = model_path or LOCAL_MODEL_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Model not found at {resolved}. "
            f"Run: python download_model.py"
        )
    source = str(resolved)
    print(f"[pipeline] Loading from: {source}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        source,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        local_files_only=True,   # never contact HuggingFace Hub
    )
    if compile:
        print("[pipeline] Compiling (first run ~3 min) ...")
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


def process_video(model, video_path: Path, results_dir: Path, duration: float | None = None) -> dict:
    label = video_path.stem
    print(f"\n{'='*60}")
    print(f"[pipeline] Processing: {label}")
    print(f"{'='*60}")

    # Trim to duration if requested (for testing or short-clip analysis)
    infer_path = video_path
    if duration is not None:
        print(f"[pipeline]   Trimming to {duration}s ...")
        infer_path = trim_video(video_path, duration)

    output = {
        "video": video_path.name,
        "label": label,
        "duration_tested_sec": duration,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caption": None,
        "events": [],
        "find_results": {},
    }

    from . import loki

    # --- Caption mode ---
    try:
        print(f"[pipeline]   Running caption mode ...")
        cap_result = _run_caption(model, str(infer_path), label)
        output["caption"] = cap_result.get("caption", "")
        output["scene"] = cap_result.get("scene", "")
        output["events"] = cap_result.get("events", [])
        EVENTS_DETECTED.labels(video=label).set(len(output["events"]))
        print(f"[pipeline]   Detected {len(output['events'])} events.")
        for ev in output["events"]:
            print(f"            [{ev.get('start',0):.1f}s – {ev.get('end',0):.1f}s] {ev.get('description','')}")
        loki.log_caption(label, output["scene"], output["events"])
        from . import annotations
        annotations.annotate_camera_done(label, len(output["events"]))
    except Exception as e:
        print(f"[pipeline]   Caption failed: {e}")
        output["caption_error"] = str(e)
        loki.log_error(label, "caption", str(e))

    # --- Find mode ---
    print(f"[pipeline]   Running {len(FIND_QUERIES)} find queries ...")
    for query in FIND_QUERIES:
        try:
            find_result = _run_find(model, str(infer_path), label, query)
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
                loki.log_find_hit(label, query, list(span), find_result.get("raw", ""))
                if query in annotations.SECURITY_QUERIES:
                    annotations.annotate_security_event(label, query, list(span))
            else:
                loki.log_find_miss(label, query)
            FIND_PARSE_OK.labels(video=label, query=query).set(1 if ok else 0)
            status = f"[{span[0]:.1f}s–{span[1]:.1f}s]" if span else "not found"
            print(f"            '{query}' → {status}")
        except Exception as e:
            output["find_results"][query] = {"error": str(e)}
            loki.log_error(label, f"find:{query}", str(e))

    VIDEOS_PROCESSED.labels(status="success").inc()

    out_file = results_dir / f"{label}.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"[pipeline]   Saved → {out_file}")
    return output


def fuse_results(all_results: list[dict]) -> dict:
    cameras = [r["label"] for r in all_results]

    fused_find: dict = {}
    for query in FIND_QUERIES:
        candidates = []
        for idx, r in enumerate(all_results):
            fr = r.get("find_results", {}).get(query, {})
            if fr.get("format_ok") and fr.get("span"):
                span = fr["span"]
                length = span[1] - span[0]
                candidates.append((length, idx, span, r["label"]))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_len, best_idx, best_span, best_cam = candidates[0]
            fused_find[query] = {
                "span": best_span,
                "camera": best_cam,
                "format_ok": True,
            }
            FUSED_FIND_START.labels(query=query).set(best_span[0])
            FUSED_FIND_END.labels(query=query).set(best_span[1])
            FUSED_FIND_CAM.labels(query=query).set(best_idx)
        else:
            fused_find[query] = {"span": None, "camera": None, "format_ok": False}

    raw_events: list[dict] = []
    for r in all_results:
        raw_events.extend(r.get("events", []))

    total_raw = len(raw_events)

    def _overlap_ratio(a: dict, b: dict) -> float:
        a0, a1 = float(a.get("start", 0)), float(a.get("end", 0))
        b0, b1 = float(b.get("start", 0)), float(b.get("end", 0))
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        union = max(a1, b1) - min(a0, b0)
        return inter / union if union > 0 else 0.0

    def _desc_similar(a: dict, b: dict) -> bool:
        da = (a.get("description") or "").lower().split()
        db = (b.get("description") or "").lower().split()
        if not da or not db:
            return False
        shared = set(da) & set(db)
        return len(shared) / max(len(set(da)), len(set(db))) >= 0.5

    deduplicated: list[dict] = []
    for ev in raw_events:
        merged = False
        for i, kept in enumerate(deduplicated):
            if _desc_similar(ev, kept) and _overlap_ratio(ev, kept) > 0.3:
                if len(ev.get("description", "")) > len(kept.get("description", "")):
                    deduplicated[i] = ev
                merged = True
                break
        if not merged:
            deduplicated.append(ev)

    unique_count = len(deduplicated)

    FUSED_EVENTS.set(unique_count)
    FUSED_TOTAL_RAW.set(total_raw)

    return {
        "source": "fused_5_cameras",
        "cameras": cameras,
        "find_results": fused_find,
        "events": deduplicated,
        "unique_event_count": unique_count,
        "total_raw_events": total_raw,
        "dedup_ratio": total_raw / max(unique_count, 1),
    }


def run_all(model, video_dir: Path, results_dir: Path, duration: float | None = None) -> list[dict]:
    videos = [
        v for ext in ("*.mkv", "*.mp4", "*.avi", "*.mov", "*.webm")
        for v in sorted(video_dir.glob(ext))
    ]

    if not videos:
        print("[pipeline] No video files found in", video_dir)
        return []

    print(f"[pipeline] Found {len(videos)} videos: {[v.name for v in videos]}")
    results_dir.mkdir(exist_ok=True)

    stop_gpu = threading.Event()
    gpu_thread = threading.Thread(target=poll_dgx_metrics, args=(stop_gpu,), daemon=True)
    gpu_thread.start()

    total = len(videos)
    trimmed: dict[Path, Path] = {}

    def _trim(v: Path) -> tuple[Path, Path]:
        if duration is not None:
            return v, trim_video(v, duration)
        return v, v

    with ThreadPoolExecutor(max_workers=min(total, 8)) as pool:
        futures = {pool.submit(_trim, v): v for v in videos}
        for fut in as_completed(futures):
            orig, clipped = fut.result()
            trimmed[orig] = clipped

    results_by_index: dict[int, dict] = {}
    completed = 0

    for idx, video in enumerate(videos):
        infer_path = trimmed[video]
        label = video.stem
        try:
            with _GPU_LOCK:
                output = {
                    "video": video.name,
                    "label": label,
                    "duration_tested_sec": duration,
                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "caption": None,
                    "events": [],
                    "find_results": {},
                }

                try:
                    cap_result = _run_caption(model, str(infer_path), label)
                    output["caption"] = cap_result.get("caption", "")
                    output["scene"] = cap_result.get("scene", "")
                    output["events"] = cap_result.get("events", [])
                    EVENTS_DETECTED.labels(video=label).set(len(output["events"]))
                except Exception as e:
                    output["caption_error"] = str(e)

                for query in FIND_QUERIES:
                    try:
                        find_result = _run_find(model, str(infer_path), label, query)
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
                    except Exception as e:
                        output["find_results"][query] = {"error": str(e)}

                VIDEOS_PROCESSED.labels(status="success").inc()

            out_file = results_dir / f"{label}.json"
            out_file.write_text(json.dumps(output, indent=2))
            results_by_index[idx] = output
            completed += 1
            print(f"[pipeline] ✓ {label} done ({completed}/{total})")
        except Exception as e:
            print(f"[pipeline] FAILED {video.name}: {e}")
            VIDEOS_PROCESSED.labels(status="error").inc()

    stop_gpu.set()
    cleanup_tmp()

    all_results = [results_by_index[i] for i in sorted(results_by_index)]

    fused = fuse_results(all_results)
    (results_dir / "fused.json").write_text(json.dumps(fused, indent=2))
    print(f"[pipeline] Fused result saved → {results_dir / 'fused.json'}")

    summary = {
        "total_videos": total,
        "processed": len(all_results),
        "videos": [r["label"] for r in all_results],
        "total_events": sum(len(r.get("events", [])) for r in all_results),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[pipeline] Done. Summary: {summary}")
    return all_results
