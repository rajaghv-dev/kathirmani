"""Exercises the real PyAV segmentation path on a tiny synthetic clip (no big .mkv,
no external services) + the file-first backends + health scoring."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest


def _make_clip(path: Path, seconds: int = 7, fps: int = 10, w: int = 160, h: int = 90):
    """Write a short solid-color (moving brightness) mp4 so health/segment have input."""
    with av.open(str(path), "w") as out:
        vs = out.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        for i in range(seconds * fps):
            arr = np.full((h, w, 3), (i * 3) % 200 + 30, dtype="uint8")  # changing brightness
            arr[:, :, 1] = (i * 7) % 255                                  # some texture/motion
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
            for pkt in vs.encode(frame):
                out.mux(pkt)
        for pkt in vs.encode():
            out.mux(pkt)


def test_segment_source_produces_clips(tmp_path):
    from ingestion.segmenter import segment_source
    from ingestion.sources import FileSource

    src_file = tmp_path / "src.mp4"
    _make_clip(src_file, seconds=7, fps=10)

    segs = list(segment_source(FileSource(src_file), tmp_path / "out", clip_sec=3))
    assert len(segs) >= 2                          # 7s @ 3s clips -> 3 segments (0,1,2)
    for s in segs:
        assert s.path.exists() and s.path.stat().st_size > 0
        assert s.first_frame is not None and s.samples
        assert s.width == 160 and s.height == 90


def test_health_scores_shape(tmp_path):
    from ingestion.health import health_scores
    frames = [np.full((90, 160, 3), 120, dtype="uint8") for _ in range(3)]
    h = health_scores(frames, fps_actual=10)
    assert set(h) == {"black_frame_score", "blur_score", "freeze_score", "fps_actual", "health_score"}
    assert 0.0 <= h["health_score"] <= 1.0


def test_black_frames_low_health():
    from ingestion.health import health_scores
    black = [np.zeros((90, 160, 3), dtype="uint8") for _ in range(3)]
    assert health_scores(black)["black_frame_score"] == 1.0
    assert health_scores(black)["health_score"] < 0.5     # rule engine would flag it


def test_file_backends_roundtrip(tmp_path):
    from ingestion.bus import FileQueue
    from ingestion.metadata import JsonlSink
    from ingestion.storage import LocalFSStorage

    clip = tmp_path / "c.mp4"; clip.write_bytes(b"x" * 100)
    store = LocalFSStorage(tmp_path / "store")
    uri = store.put_clip(clip, "store/cam/seg_00000.mp4")
    assert Path(uri).exists() and not clip.exists()       # moved into store

    sink = JsonlSink(tmp_path / "meta")
    sink.write_segment({"segment_id": "s1"}); sink.write_window({"window_id": "w1"})
    sink.close()
    assert (tmp_path / "meta" / "segments.jsonl").read_text().strip()

    q = FileQueue(tmp_path / "q")
    q.publish({"type": "video_segment.ready", "segment_id": "s1"})
    q.publish({"type": "ai_window.ready", "window_id": "w1"})
    q.close()
    assert (tmp_path / "q" / "video_segment_ready.jsonl").exists()
    assert (tmp_path / "q" / "ai_window_ready.jsonl").exists()


def test_segment_source_overlap(tmp_path):
    """30s/5s-overlap geometry, scaled down: clip_sec=3, overlap=1 on a 7s
    source -> clips start every 2s: [0,3) [2,5) [4,7) [6,7]. Frames in the
    shared second land in BOTH adjacent clips."""
    from ingestion.segmenter import segment_source
    from ingestion.sources import FileSource

    src_file = tmp_path / "src.mp4"
    _make_clip(src_file, seconds=7, fps=10)
    segs = list(segment_source(FileSource(src_file), tmp_path / "out",
                               clip_sec=3, overlap_sec=1))
    assert [s.index for s in segs] == [0, 1, 2, 3]
    # full clips are clip_sec long; starts advance by stride=2
    assert segs[0].duration_sec == 3.0 and segs[1].duration_sec == 3.0
    assert segs[-1].duration_sec <= 3.0                 # tail clip is shorter
    # overlap means the total encoded duration EXCEEDS the source duration
    assert sum(s.duration_sec for s in segs) > 7.0
    for s in segs:
        assert s.path.exists() and s.path.stat().st_size > 0


def test_windows_skip_overlap_prefix():
    """Windows fully inside the overlap prefix are dropped (covered by the
    previous clip's tail); boundary-crossing windows survive."""
    from ingestion.windows import windows_for_segment

    class Seg:
        segment_id = "s1"; camera_id = "cam"; storage_path = "p.mp4"
        duration_sec = 30.0

    all_w = windows_for_segment(Seg(), 5, 2)
    kept = windows_for_segment(Seg(), 5, 2, skip_before_sec=5)
    dropped = [w for w in all_w if w.window_start_sec not in
               {k.window_start_sec for k in kept}]
    assert all(w.window_end_sec <= 5 for w in dropped), "only fully-inside-overlap dropped"
    assert any(w.window_start_sec < 5 < w.window_end_sec for w in kept), \
        "boundary-crossing windows kept"
