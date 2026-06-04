"""Video sources — the pluggable input to the segmenter.

**Decision (see spec/10 / the GStreamer note):** OSS file + RTSP ingestion uses
**PyAV** (FFmpeg's libav bindings) — already a vetted repo dep, works on this
aarch64 box via the bundled-libs preload, and is in-process + testable. PyAV opens
both local files and `rtsp://` URLs, so `FileSource` and `RtspSource` share a path.

**GStreamer is deliberately deferred.** It earns its place for live-RTSP-*at-scale*
(jitter buffer, reconnection, NVIDIA HW decode `nvv4l2decoder`) and is intrinsic to
DeepStream (Phase 4 = a GStreamer pipeline). `GStreamerSource` is the slot it drops
into — implementing it later does not touch the segmenter/queue/storage.
"""
from __future__ import annotations

import abc
import os
from pathlib import Path


class Source(abc.ABC):
    """Resolves to something PyAV (or a future backend) can open."""
    kind: str = "abstract"

    @abc.abstractmethod
    def open_target(self) -> str:
        """The string passed to av.open() (a file path or an rtsp:// URL)."""

    @property
    @abc.abstractmethod
    def label(self) -> str:
        ...


class FileSource(Source):
    kind = "file"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def open_target(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"source file missing: {self.path}")
        return str(self.path)

    @property
    def label(self) -> str:
        return self.path.name


class RtspSource(Source):
    """Live RTSP via PyAV (fine for moderate stream counts). For many concurrent
    streams or HW decode, prefer GStreamerSource (future)."""
    kind = "rtsp"

    def __init__(self, url: str):
        self.url = url

    def open_target(self) -> str:
        if not self.url:
            raise ValueError("empty RTSP URL")
        return self.url

    @property
    def label(self) -> str:
        return self.url


class GStreamerSource(Source):
    """The live + HW-accelerated path. gst 1.24 is on the box (CLI); Python `gi`
    bindings are not installed yet, so this is driven via gst-launch/subprocess
    until PyGObject lands. GStreamer is chosen for two concrete jobs:

    1. **RTSP → 10-sec clip recording, stream-copy (no re-encode):**
         rtspsrc location=<url> ! rtph264depay ! h264parse !
         splitmuxsink max-size-time=10000000000 location=seg_%05d.mp4
       Keyframe-aligned, near-zero CPU — the live equivalent of the PyAV segmenter.

    2. **Frame reduction *mechanism* for inference ("ignore certain frames"):**
       videorate (fps cap) · keyframe-only · DeepStream `nvinfer interval=N`
       (skip inference on N of N+1 frames) · motioncells (drop static frames).

    The *policy* of which frames matter (motion/zone activity, cascade route_to_vlm,
    redundant-frame skip) stays OUR logic — deterministic where safety-relevant.
    Phase 1 uses PyAV File/RtspSource; this slots in for live-at-scale (Phase 1.5)
    and is intrinsic to DeepStream (Phase 4)."""
    kind = "gstreamer"

    def __init__(self, pipeline: str):
        self.pipeline = pipeline

    def open_target(self) -> str:
        raise NotImplementedError(
            "GStreamerSource is the live recording (splitmuxsink) + frame-gating / "
            "DeepStream path (Phase 1.5 / Phase 4). Phase 1 uses PyAV File/RtspSource."
        )

    @property
    def label(self) -> str:
        return "gstreamer-pipeline"


def source_for_camera(cam, project_root: Path) -> Source:
    """Pick a source for a camera: prefer the RTSP env var if set, else the file."""
    if cam.rtsp_url_env and os.environ.get(cam.rtsp_url_env):
        return RtspSource(os.environ[cam.rtsp_url_env])
    if cam.source_file:
        return FileSource(project_root / cam.source_file)
    raise ValueError(f"camera {cam.id}: no rtsp_url_env set and no source_file")
