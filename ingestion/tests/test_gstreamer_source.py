"""GStreamerSource (Phase 1.5): the live-RTSP splitmuxsink 10-sec clip writer.

Driven via the gst-launch CLI (no Python `gi` on the box). These tests are hermetic:
they assert the argv/pipeline construction, the absent-binary graceful path, and a
fully-mocked recording loop — no real RTSP server, no gstreamer install required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.sources import (
    DEFAULT_SEGMENT_NS,
    GStreamerSource,
    GStreamerUnavailable,
    RtspSource,
    catalog_live_clips,
    source_for_camera,
)


URL = "rtsp://cam.local:554/stream"


# ── command / pipeline construction ──────────────────────────────────────────
def test_build_command_is_streamcopy_splitmuxsink(tmp_path):
    src = GStreamerSource(URL, out_dir=tmp_path)
    argv = src.build_command()

    assert argv[0] == "gst-launch-1.0"
    assert "-e" in argv                                  # EOS on shutdown → clean final clip
    assert f"location={URL}" in argv                     # rtspsrc location
    # stream-copy depay/parse, no decode/encode element anywhere
    assert "rtph264depay" in argv and "h264parse" in argv
    joined = " ".join(argv)
    assert "decodebin" not in joined and "x264enc" not in joined and "nvv4l2" not in joined
    # splitmuxsink with 10s (ns) keyframe-aligned segments writing seg_%05d.mp4
    assert "splitmuxsink" in argv
    assert f"max-size-time={DEFAULT_SEGMENT_NS}" in argv
    assert f"location={tmp_path / 'seg_%05d.mp4'}" in argv


def test_default_segment_is_ten_seconds_in_ns():
    assert DEFAULT_SEGMENT_NS == 10_000_000_000


def test_pipeline_string_matches_docstring_spec(tmp_path):
    src = GStreamerSource(URL, out_dir=tmp_path, segment_ns=10_000_000_000)
    expected = (
        f"rtspsrc location={URL} ! rtph264depay ! h264parse "
        f"! splitmuxsink max-size-time=10000000000 "
        f"location={tmp_path / 'seg_%05d.mp4'}"
    )
    assert src.pipeline() == expected


def test_custom_segment_duration_and_pattern(tmp_path):
    src = GStreamerSource(URL, out_dir=tmp_path, segment_ns=5_000_000_000,
                          clip_pattern="clip_%03d.mp4")
    argv = src.build_command()
    assert "max-size-time=5000000000" in argv
    assert f"location={tmp_path / 'clip_%03d.mp4'}" in argv


def test_empty_url_rejected():
    with pytest.raises(ValueError):
        GStreamerSource("")


def test_open_target_returns_url_does_not_crash():
    # open_target must NOT raise (no import/wiring crash); availability is separate.
    assert GStreamerSource(URL).open_target() == URL


# ── graceful when gst-launch is absent ───────────────────────────────────────
def test_available_reflects_which(monkeypatch):
    monkeypatch.setattr("ingestion.sources.shutil.which", lambda _b: None)
    assert GStreamerSource.available() is False
    monkeypatch.setattr("ingestion.sources.shutil.which", lambda _b: "/usr/bin/gst-launch-1.0")
    assert GStreamerSource.available() is True


def test_start_raises_catchable_error_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.sources.shutil.which", lambda _b: None)
    src = GStreamerSource(URL, out_dir=tmp_path)
    with pytest.raises(GStreamerUnavailable):
        src.start()
    # GStreamerUnavailable is a RuntimeError → catchable like the FileSource pattern
    assert issubclass(GStreamerUnavailable, RuntimeError)


# ── mocked recording loop (no real gst, no RTSP) ─────────────────────────────
class _FakeProc:
    """Minimal subprocess.Popen stand-in that 'writes' clips when started."""
    def __init__(self, argv, out_dir, n_clips, **_kw):
        self.argv = argv
        self.returncode = None
        self._signaled = False
        # simulate splitmuxsink emitting N finished clips into out_dir
        for i in range(n_clips):
            (out_dir / f"seg_{i:05d}.mp4").write_bytes(b"FAKEMP4" * (i + 1))

    def poll(self):
        return self.returncode

    def send_signal(self, _sig):
        self._signaled = True

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


def test_recording_loop_start_stop_and_enumerate(tmp_path, monkeypatch):
    out = tmp_path / "live"
    captured = {}

    def fake_popen(argv, **kw):
        out.mkdir(parents=True, exist_ok=True)
        captured["argv"] = argv
        return _FakeProc(argv, out, n_clips=3, **kw)

    monkeypatch.setattr("ingestion.sources.shutil.which", lambda _b: "/usr/bin/gst-launch-1.0")
    monkeypatch.setattr("ingestion.sources.subprocess.Popen", fake_popen)

    src = GStreamerSource(URL, out_dir=out)
    proc = src.start()
    assert captured["argv"][0] == "gst-launch-1.0"

    rc = src.stop()
    assert rc == 0 and proc._signaled is True            # clean EOS via SIGINT

    clips = src.clips()
    assert [c.name for c in clips] == ["seg_00000.mp4", "seg_00001.mp4", "seg_00002.mp4"]
    assert all(c.stat().st_size > 0 for c in clips)


def test_stop_with_no_process_is_noop(tmp_path):
    assert GStreamerSource(URL, out_dir=tmp_path).stop() is None


def test_clips_empty_when_dir_missing(tmp_path):
    assert GStreamerSource(URL, out_dir=tmp_path / "nope").clips() == []


# ── produced clips reach the existing catalog path (SegmentRecord shape) ──────
def test_catalog_live_clips_yields_segmentrecords(tmp_path):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    out = tmp_path / "live"
    out.mkdir()
    for i in range(2):
        (out / f"seg_{i:05d}.mp4").write_bytes(b"data" * (i + 1))

    cam = SimpleNamespace(id="cam_1", rtsp_url_env=None, source_file=None)
    cfg = SimpleNamespace(
        store_id="store_42",
        segmenting=SimpleNamespace(storage_clip_duration_sec=10),
    )
    src = GStreamerSource(URL, out_dir=out)
    base = datetime.now(timezone.utc)

    rows = list(catalog_live_clips(src, cam, cfg, base, out))
    assert len(rows) == 2
    for idx, (clip, rec) in enumerate(rows):
        assert clip.exists()
        assert rec.camera_id == "cam_1" and rec.store_id == "store_42"
        assert rec.index == idx and rec.codec == "h264"
        assert len(rec.checksum) == 64                   # sha256 hex — evidence integrity
        msg = rec.to_message()
        assert msg["type"] == "video_segment.ready"


# ── selector: opt-in without changing default PyAV behaviour ──────────────────
def test_source_for_camera_default_stays_pyav(monkeypatch, tmp_path):
    monkeypatch.delenv("INGEST_SOURCE_BACKEND", raising=False)
    monkeypatch.setenv("CAM_URL", URL)
    from types import SimpleNamespace
    cam = SimpleNamespace(id="c", rtsp_url_env="CAM_URL", source_file=None)
    assert isinstance(source_for_camera(cam, tmp_path), RtspSource)


def test_source_for_camera_opts_into_gstreamer(monkeypatch, tmp_path):
    monkeypatch.setenv("CAM_URL", URL)
    monkeypatch.setenv("INGEST_SOURCE_BACKEND", "gstreamer")
    monkeypatch.setattr("ingestion.sources.shutil.which", lambda _b: "/usr/bin/gst-launch-1.0")
    from types import SimpleNamespace
    cam = SimpleNamespace(id="c", rtsp_url_env="CAM_URL", source_file=None)
    assert isinstance(source_for_camera(cam, tmp_path), GStreamerSource)


def test_gstreamer_opt_in_falls_back_when_cli_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("CAM_URL", URL)
    monkeypatch.setenv("INGEST_SOURCE_BACKEND", "gstreamer")
    monkeypatch.setattr("ingestion.sources.shutil.which", lambda _b: None)
    from types import SimpleNamespace
    cam = SimpleNamespace(id="c", rtsp_url_env="CAM_URL", source_file=None)
    # no gst on PATH → transparently falls back to PyAV RtspSource
    assert isinstance(source_for_camera(cam, tmp_path), RtspSource)
