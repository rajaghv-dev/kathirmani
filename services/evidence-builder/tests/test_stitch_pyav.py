"""Real-PyAV stitch test — exercises the success path with a tiny synthetic
clip (no GPU, no network, no DB). Skips cleanly if PyAV/numpy aren't present."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import builder

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

T0 = datetime(2026, 6, 3, 10, 0, 0, tzinfo=timezone.utc)


def _make_clip(path, n_frames=20, fps=10, w=64, h=48):
    out = av.open(str(path), "w")
    st = out.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    for i in range(n_frames):
        arr = np.full((h, w, 3), (i * 12) % 255, dtype="uint8")
        for pkt in st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()


def test_stitch_produces_clip_and_sha256(tmp_path):
    src = tmp_path / "seg.mp4"
    _make_clip(src, n_frames=20, fps=10)              # ~2s of video
    # Segment spans 10:00:00..10:00:02; event at +1s, wide window covers it all.
    segs = [{"id": "s1", "storage_path": str(src),
             "start_time": T0.isoformat(),
             "end_time": T0.replace(second=2).isoformat()}]
    out = tmp_path / "evidence.mp4"
    res = builder.stitch_clip(segs, T0.replace(second=1).isoformat(), out,
                              pre=20, post=20)
    assert res["ok"] is True, res["reason"]
    assert out.exists() and out.stat().st_size > 0
    assert res["sha256"] and len(res["sha256"]) == 64
    assert res["sha256"] == builder.sha256_file(out)  # determinism vs file
    assert res["segments_used"] == [str(src)]


def test_stitch_trims_to_window(tmp_path):
    src = tmp_path / "seg.mp4"
    _make_clip(src, n_frames=40, fps=10)              # 4s of video
    segs = [{"id": "s1", "storage_path": str(src),
             "start_time": T0.isoformat(),
             "end_time": T0.replace(second=4).isoformat()}]
    # Narrow window 10:00:01..10:00:02 (event at +1.5s, pre/post 0.5s).
    out = tmp_path / "trimmed.mp4"
    res = builder.stitch_clip(segs, T0.replace(microsecond=500000, second=1).isoformat(),
                              out, pre=0.5, post=0.5)
    assert res["ok"] is True, res["reason"]
    # trimmed output should have fewer frames than the 4s source
    inp = av.open(str(out))
    n = sum(1 for _ in inp.decode(inp.streams.video[0]))
    inp.close()
    assert 0 < n < 40
