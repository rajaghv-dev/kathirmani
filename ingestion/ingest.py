"""Ingestion orchestrator: Source → 30-sec clips (5-sec overlap) → windows → store + metadata + queue.

Per camera: segment the source, and for each clip store it (checksummed for evidence
integrity), thumbnail it, score camera health, write a metadata row, publish
`video_segment.ready`, then create 5-sec windows and publish `ai_window.ready`.
Backends (storage/metadata/queue) are pluggable; defaults are hermetic file-first.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config as C
from . import metrics as M
from .bus import make_queue
from .health import health_scores
from .metadata import make_sink
from .records import SegmentRecord
from .segmenter import segment_source
from .sources import source_for_camera
from .storage import make_storage
from .thumbnailer import save_thumbnail
from .windows import windows_for_segment


@dataclass
class IngestOptions:
    duration_limit: float | None = None     # seconds of source to ingest (smoke tests)
    storage: str | None = None              # local|minio
    metadata: str | None = None             # jsonl|postgres
    queue: str | None = None                # file|redis
    serve_metrics: bool = False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_camera(cam, cfg, opts, storage, sink, queue, base_time, seen: set) -> dict:
    seg_cfg = cfg.segmenting
    labels = dict(store_id=cfg.store_id, camera_id=cam.id)
    staging = C.DATA_DIR / "_staging" / cam.id
    src = source_for_camera(cam, C.PROJECT_ROOT)

    n_seg = n_win = 0
    try:
        src.open_target()
        M.rtsp_connected.labels(**labels).set(1)
    except Exception as e:
        M.rtsp_connected.labels(**labels).set(0)
        M.errors_total.labels(**labels, stage="open").inc()
        print(f"[{cam.id}] source error: {e}")
        return {"camera": cam.id, "segments": 0, "windows": 0, "error": str(e)}

    for seg in segment_source(src, staging, seg_cfg.storage_clip_duration_sec,
                              opts.duration_limit,
                              overlap_sec=seg_cfg.clip_overlap_sec):
        t0 = time.time()
        dedup_key = f"{cam.id}:{seg.index}"
        if dedup_key in seen:                                  # duplicate-segment protection
            continue
        seen.add(dedup_key)

        # Clips overlap: clip k starts every stride (= duration − overlap) secs.
        start = base_time + timedelta(seconds=seg.index * seg_cfg.clip_stride_sec)
        end = start + timedelta(seconds=seg.duration_sec)
        rec = SegmentRecord(
            camera_id=cam.id, store_id=cfg.store_id, index=seg.index,
            start_time=start.isoformat(), end_time=end.isoformat(),
            duration_sec=seg.duration_sec, storage_path="", codec=seg.codec,
            fps=seg.fps, width=seg.width, height=seg.height,
            checksum=_sha256(seg.path),
        )
        key = f"{cfg.store_id}/{cam.id}/{start:%Y/%m/%d}/seg_{seg.index:05d}.mp4"
        rec.storage_path = storage.put_clip(seg.path, key)

        if seg.first_frame is not None:
            tlocal = staging / f"thumb_{seg.index:05d}.jpg"
            save_thumbnail(seg.first_frame, tlocal)
            rec.thumbnail_path = storage.put_thumb(tlocal, key.replace(".mp4", ".jpg"))

        rec.health = health_scores(seg.samples, seg.fps)
        sink.write_segment(rec.to_row())
        queue.publish(rec.to_message())

        wins = windows_for_segment(
            rec, seg_cfg.ai_window_duration_sec, seg_cfg.ai_window_stride_sec,
            # Overlapping clips: drop windows fully inside the shared prefix —
            # the previous clip's tail windows already cover those spans.
            skip_before_sec=(seg_cfg.clip_overlap_sec if seg.index > 0 else 0),
        )
        for w in wins:
            sink.write_window(w.to_row())
            queue.publish(w.to_message())

        # metrics
        M.segments_written_total.labels(**labels).inc()
        M.windows_created_total.labels(**labels).inc(len(wins))
        M.segment_write_latency.labels(**labels).observe(time.time() - t0)
        h = rec.health
        M.camera_health_score.labels(**labels).set(h["health_score"])
        M.camera_black_frame_score.labels(**labels).set(h["black_frame_score"])
        M.camera_blur_score.labels(**labels).set(h["blur_score"])
        M.camera_freeze_score.labels(**labels).set(h["freeze_score"])
        M.camera_fps_actual.labels(**labels).set(h["fps_actual"])
        n_seg += 1; n_win += len(wins)

    return {"camera": cam.id, "segments": n_seg, "windows": n_win}


def run_ingest(opts: IngestOptions, cameras_yaml: Path | None = None,
               only: list[str] | None = None) -> dict:
    cfg = C.load_cameras(cameras_yaml)
    storage = make_storage(opts.storage, C.CLIPS_DIR)
    sink = make_sink(opts.metadata, C.META_DIR)
    queue = make_queue(opts.queue, C.QUEUE_DIR)
    if opts.serve_metrics:
        M.serve(C.METRICS_PORT)

    base_time = datetime.now(timezone.utc)
    seen: set = set()
    results = []
    print(f"[ingest] store={cfg.store_id} storage={storage.name} "
          f"metadata={sink.name} queue={queue.name}")
    try:
        for cam in cfg.cameras:
            if not cam.enabled or (only and cam.id not in only and cam.name not in only):
                continue
            print(f"[ingest] camera {cam.id} ({cam.role}) ...")
            results.append(ingest_camera(cam, cfg, opts, storage, sink, queue, base_time, seen))
    finally:
        sink.close(); queue.close()

    tot_s = sum(r["segments"] for r in results)
    tot_w = sum(r["windows"] for r in results)
    print(f"[ingest] done: {tot_s} segments, {tot_w} windows across {len(results)} cameras")
    return {"store_id": cfg.store_id, "cameras": results, "segments": tot_s, "windows": tot_w}
