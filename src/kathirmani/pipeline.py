"""Core Marlin-2B inference pipeline for security camera videos."""
import importlib
import json
import time
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
import torch

from . import MODELS_DIR, VIDEO_EXTS
from .device import detect_device
from .metrics import (
    INFERENCE_DURATION, EVENTS_DETECTED, VIDEOS_PROCESSED,
    FIND_SPAN_START, FIND_SPAN_END, FIND_PARSE_OK, poll_dgx_metrics,
    FUSED_EVENTS, FUSED_TOTAL_RAW, FUSED_FIND_START, FUSED_FIND_END, FUSED_FIND_CAM,
    DECODE_SECONDS_ACTUAL, DECODE_SECONDS_SAVED, DECODES_DONE, DECODES_AVOIDED,
)

_GPU_LOCK = threading.Lock()


class _Progress:
    """Thread-safe GPU-call progress tracker.

    The single GPU serializes every forward pass, so the total work for a run is
    a fixed count: cameras × (1 caption + N find queries). Ticking once per
    completed GPU call gives a true percent-done + ETA across all parallel
    cameras — without it the parallel run goes silent between camera-done lines.
    """

    def __init__(self, total_calls: int):
        self.total = max(total_calls, 1)
        self.done = 0
        self.start = time.time()
        self._lock = threading.Lock()

    def _bump(self, n: int) -> int:
        with self._lock:
            self.done += n
            return self.done

    def tick(self) -> str:
        """Count one finished GPU call; return a 'GPU x/y pct · elapsed · ETA' string."""
        done = self._bump(1)
        elapsed = time.time() - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        pct = 100.0 * done / self.total
        return f"GPU {done}/{self.total} {pct:3.0f}% · {elapsed:.0f}s · ETA ~{eta:.0f}s"

    def skip(self, n: int) -> None:
        """Account for GPU calls that won't run (e.g. a camera whose decode failed)."""
        self._bump(n)


# Kathirmani store-specific surveillance queries
# Each maps to a Prometheus metric label: kathirmani_find_span_*{query="..."}
FIND_QUERIES = [
    # Entry / exit
    "a person walks in through the entrance door",
    "a person walks out through the exit door",
    # Billing
    "a customer places items on the billing counter",
    "a staff member scans or bills an item at the counter",
    "a customer hands cash or a card to the billing staff",
    "several customers wait in line at the billing counter",
    # Shopping floor
    "a customer browses products on a shelf",
    "a person takes an item off a shelf",
    "a person puts an item back on a shelf",
    "a customer carries a shopping basket",
    "a customer pushes a shopping cart",
    # Loss-prevention PRIMITIVES (visible atomic actions only — combine in rules/Qwen)
    "a person puts a product into a personal bag or backpack",
    "a person puts a product into their pocket or clothing",
    "a person conceals an item with their hand or body",
    "a person carries a large bag or backpack near the shelves",
    "a person walks toward the exit holding an item",
    "a person lingers near the exit",
    "a person opens or tears product packaging at a shelf",
    # Crowd / operations
    "a group of people crowds together in an aisle",
    "a bag or object is left unattended on the floor",
]


MODEL_ID = "NemoStation/Marlin-2B"
LOCAL_MODEL_PATH = MODELS_DIR / "Marlin-2B"

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
            nframes = 0
            for frame in inp.decode(in_vs):
                t = float(frame.pts * in_vs.time_base) if frame.pts is not None else 0.0
                if t > duration_sec:
                    break
                # Rebase PTS to a local frame counter: encoder output must not
                # inherit source timestamps (invalid DTS on real footage; same
                # fix as ingestion/segmenter.py).
                frame = frame.reformat(format="yuv420p")
                frame.pts = nframes
                frame.time_base = Fraction(1, int(round(fps))) if fps else None
                nframes += 1
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
    cfg = detect_device()
    print(f"[pipeline] Loading from: {source}")
    print(cfg.describe())

    load_kwargs = dict(
        trust_remote_code=True,
        dtype=cfg.dtype,
        local_files_only=True,   # never contact HuggingFace Hub
    )
    # device_map (accelerate placement) is CUDA-only here; on MPS/CPU we load
    # then move the whole model with .to(device).
    if cfg.device_map is not None:
        load_kwargs["device_map"] = cfg.device_map

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
    if cfg.device_map is None:
        model = model.to(cfg.torch_device)

    if compile and cfg.supports_compile:
        print("[pipeline] Compiling (first run ~3 min) ...")
        model.compile()
    elif compile:
        print(f"[pipeline] Skipping torch.compile on {cfg.kind}.")
    print("[pipeline] Ready.")
    return model


def _kathirmani_module(model):
    """The dynamically-loaded modeling_marlin module backing this model.

    Gives us CAPTION_PROMPT / GROUNDING_PROMPT_TEMPLATE / parse_caption /
    parse_span without re-importing through the Hub.
    """
    return importlib.import_module(type(model).__module__)


def _decode_video_once(model, video_path: str):
    """Decode a camera's frames ONCE so all 21 prompts can reuse them.

    Marlin.caption()/find() each re-decode the video from scratch via the
    processor (torchcodec), so one camera costs 1 + 20 = 21 full decodes of the
    same file. Here we run vision decoding a single time and hand the resulting
    tensors to every prompt — the biggest single compute-time win (see
    learning.md §6). Decode is CPU/torchcodec work and is intentionally run
    OUTSIDE _GPU_LOCK so it overlaps across cameras.

    Returns (image_inputs, frames, metas, video_kwargs) for processor().
    Marlin is Qwen3-VL based: temporal grounding (find spans) is only correct
    if frame timestamps are supplied, so we capture per-video `video_metadata`
    and pass it through — without it the processor defaults to fps=24 and the
    find spans come out wrong.
    """
    from qwen_vl_utils import process_vision_info
    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": str(video_path)}],
    }]
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )
    # With return_video_metadata each item is a (frames_tensor, metadata) tuple.
    frames = [v[0] for v in video_inputs]
    metas = [v[1] for v in video_inputs]
    return image_inputs, frames, metas, video_kwargs


def _generate_cached(model, decoded, prompt: str, max_tokens: int) -> str:
    """One greedy generate() over PRE-DECODED frames + a text prompt.

    Mirrors MarlinForConditionalGeneration._generate_video but feeds the cached
    vision tensors instead of a path, so the video is decoded once per camera
    rather than once per query. The chat template emits a single video
    placeholder (path-independent at tokenize=False); processor() expands it to
    the correct token count from the decoded `video_inputs` + `video_kwargs`.
    """
    image_inputs, frames, metas, video_kwargs = decoded
    proc = model.processor
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": ""},
            {"type": "text", "text": prompt},
        ],
    }]
    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = proc(
        text=[text],
        images=image_inputs,
        videos=frames,
        video_metadata=metas,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    out = out[:, prompt_len:]
    return proc.batch_decode(out, skip_special_tokens=True)[0]


def _run_caption(model, decoded, label: str) -> dict:
    mod = _kathirmani_module(model)
    with INFERENCE_DURATION.labels(video=label, mode="caption").time():
        raw = _generate_cached(model, decoded, mod.CAPTION_PROMPT, max_tokens=2048)
    cleaned, scene, events = mod.parse_caption(raw)
    return {"caption": cleaned, "scene": scene, "events": events, "raw": raw}


def _run_find(model, decoded, label: str, query: str) -> dict:
    mod = _kathirmani_module(model)
    prompt = mod.GROUNDING_PROMPT_TEMPLATE.format(event=query.strip())
    with INFERENCE_DURATION.labels(video=label, mode="find").time():
        raw = _generate_cached(model, decoded, prompt, max_tokens=64)
    cleaned, span = mod.parse_span(raw)
    return {"raw": cleaned, "span": span, "format_ok": span is not None}


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

    # Decode the video ONCE; reused by caption + all find queries below.
    try:
        decoded = _decode_video_once(model, str(infer_path))
    except Exception as e:
        print(f"[pipeline]   Decode failed: {e}")
        output["caption_error"] = f"decode failed: {e}"
        loki.log_error(label, "decode", str(e))
        VIDEOS_PROCESSED.labels(status="error").inc()
        out_file = results_dir / f"{label}.json"
        out_file.write_text(json.dumps(output, indent=2))
        return output

    # --- Caption mode ---
    try:
        print(f"[pipeline]   Running caption mode ...")
        cap_result = _run_caption(model, decoded, label)
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
            find_result = _run_find(model, decoded, label, query)
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


def _infer_one_video(model, video: Path, infer_path: Path, results_dir: Path,
                     duration: float | None, progress: "_Progress | None" = None) -> dict:
    """Run caption + all find queries for a single camera.

    GPU calls are individually guarded by _GPU_LOCK so multiple cameras can run
    concurrently — their video decode / parsing / JSON IO overlap while the
    actual model forward serializes on the single GPU. `progress` (if given) is
    ticked once per completed GPU call to report run-wide percent-done + ETA.
    """
    from . import loki, annotations

    label = video.stem
    n_q = len(FIND_QUERIES)
    output = {
        "video": video.name,
        "label": label,
        "duration_tested_sec": duration,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caption": None,
        "events": [],
        "find_results": {},
    }

    # Decode ONCE per camera, outside the lock, so decodes of the 5 cameras
    # overlap on CPU while the GPU forward passes serialize on _GPU_LOCK.
    t_dec = time.time()
    try:
        decoded = _decode_video_once(model, str(infer_path))
    except Exception as e:
        output["caption_error"] = f"decode failed: {e}"
        loki.log_error(label, "decode", str(e))
        VIDEOS_PROCESSED.labels(status="error").inc()
        if progress:  # these GPU calls will never run — keep the run total honest
            progress.skip(1 + n_q)
        (results_dir / f"{label}.json").write_text(json.dumps(output, indent=2))
        return output
    dec_s = time.time() - t_dec
    metas = decoded[2]
    nframes = len(metas[0].get("frames_indices", [])) if metas else 0
    # Decode-once savings: the naive path would decode once per prompt (1 + n_q);
    # we decode once and reuse, avoiding n_q redundant decodes for this camera.
    DECODE_SECONDS_ACTUAL.inc(dec_s)
    DECODE_SECONDS_SAVED.inc(dec_s * n_q)
    DECODES_DONE.inc(1)
    DECODES_AVOIDED.inc(n_q)
    print(f"[{label}] decoded {nframes} frames in {dec_s:.1f}s "
          f"(reused for {1 + n_q} prompts; saved ~{dec_s * n_q:.0f}s decode)", flush=True)

    try:
        t = time.time()
        with _GPU_LOCK:
            cap_result = _run_caption(model, decoded, label)
        output["caption"] = cap_result.get("caption", "")
        output["scene"] = cap_result.get("scene", "")
        output["events"] = cap_result.get("events", [])
        EVENTS_DETECTED.labels(video=label).set(len(output["events"]))
        # Insert to the log DB (Loki) + Grafana annotations as this camera finishes.
        loki.log_caption(label, output["scene"], output["events"])
        annotations.annotate_camera_done(label, len(output["events"]))
        prog = progress.tick() if progress else ""
        print(f"[{label}] caption ✓ {len(output['events'])} events · "
              f"{time.time() - t:.1f}s | {prog}", flush=True)
    except Exception as e:
        output["caption_error"] = str(e)
        loki.log_error(label, "caption", str(e))
        prog = progress.tick() if progress else ""
        print(f"[{label}] caption ERROR · {e} | {prog}", flush=True)

    for i, query in enumerate(FIND_QUERIES, 1):
        try:
            t = time.time()
            with _GPU_LOCK:
                find_result = _run_find(model, decoded, label, query)
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
            prog = progress.tick() if progress else ""
            tag = f"HIT [{span[0]:.0f}-{span[1]:.0f}s]" if span else "miss        "
            print(f"[{label}] find {i:2d}/{n_q} {tag} · {time.time() - t:.1f}s | {prog}",
                  flush=True)
        except Exception as e:
            output["find_results"][query] = {"error": str(e)}
            loki.log_error(label, f"find:{query}", str(e))
            prog = progress.tick() if progress else ""
            print(f"[{label}] find {i:2d}/{n_q} ERROR · {e} | {prog}", flush=True)

    VIDEOS_PROCESSED.labels(status="success").inc()
    out_file = results_dir / f"{label}.json"
    out_file.write_text(json.dumps(output, indent=2))
    return output


def run_all(model, video_dir: Path, results_dir: Path, duration: float | None = None,
            max_workers: int | None = None) -> list[dict]:
    videos = [
        v for ext in VIDEO_EXTS
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
    workers = max_workers if max_workers is not None else total
    workers = max(1, min(workers, total))
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

    prompts_per_cam = 1 + len(FIND_QUERIES)
    total_gpu_calls = total * prompts_per_cam
    progress = _Progress(total_gpu_calls)
    t_run = time.time()

    print(f"[pipeline] Processing {total} cameras with {workers} parallel workers "
          f"(GPU forward serialized on the single device).")
    print(f"[pipeline] Work: {total_gpu_calls} GPU calls "
          f"({total} cameras × {prompts_per_cam} prompts) · "
          f"{total} video decodes (decode-once; was {total_gpu_calls}).")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_infer_one_video, model, video, trimmed[video], results_dir,
                        duration, progress): idx
            for idx, video in enumerate(videos)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            video = videos[idx]
            try:
                results_by_index[idx] = fut.result()
                completed += 1
                print(f"[pipeline] ✓ {video.stem} done — {completed}/{total} cameras · "
                      f"{progress.done}/{total_gpu_calls} GPU calls · "
                      f"{time.time() - t_run:.0f}s elapsed", flush=True)
            except Exception as e:
                print(f"[pipeline] FAILED {video.name}: {e}")
                VIDEOS_PROCESSED.labels(status="error").inc()

    run_s = time.time() - t_run
    per_call = run_s / max(progress.done, 1)
    print(f"[pipeline] All cameras done in {run_s:.0f}s · "
          f"{progress.done} GPU calls · {per_call:.1f}s/call avg", flush=True)

    stop_gpu.set()
    cleanup_tmp()

    all_results = [results_by_index[i] for i in sorted(results_by_index)]

    fused = fuse_results(all_results)
    (results_dir / "fused.json").write_text(json.dumps(fused, indent=2))
    print(f"[pipeline] Fused result saved → {results_dir / 'fused.json'}")

    from . import loki
    best_cameras = {
        q: fr["camera"] for q, fr in fused.get("find_results", {}).items()
        if fr.get("camera")
    }
    loki.log_fusion(fused.get("unique_event_count", 0),
                    fused.get("total_raw_events", 0), best_cameras)

    summary = {
        "total_videos": total,
        "processed": len(all_results),
        "videos": [r["label"] for r in all_results],
        "total_events": sum(len(r.get("events", [])) for r in all_results),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[pipeline] Done. Summary: {summary}")
    return all_results
