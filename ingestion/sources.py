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
import shutil
import signal
import subprocess
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


# The CLI we drive (no Python `gi`/PyGObject on the box; subprocess instead).
GST_LAUNCH_BIN = "gst-launch-1.0"
# splitmuxsink wants nanoseconds; 10 s = 10_000_000_000 ns.
DEFAULT_SEGMENT_NS = 10_000_000_000
# splitmuxsink fills %0Nd in `location` with the running clip index.
CLIP_NAME_PATTERN = "seg_%05d.mp4"


class GStreamerUnavailable(RuntimeError):
    """Raised when the gst-launch CLI is absent (this box may be CPU-only / no gst).

    Catchable like FileSource's FileNotFoundError — callers detect the optional
    live path is unavailable and fall back to PyAV, instead of crashing."""


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
    and is intrinsic to DeepStream (Phase 4).

    Recording is finite-friendly: pass `max_clips` (and/or call `stop()`) so a test
    or smoke run terminates instead of recording forever.
    """
    kind = "gstreamer"

    def __init__(
        self,
        url: str,
        out_dir: str | Path | None = None,
        segment_ns: int = DEFAULT_SEGMENT_NS,
        clip_pattern: str = CLIP_NAME_PATTERN,
        max_clips: int | None = None,
    ):
        if not url:
            raise ValueError("empty RTSP URL")
        self.url = url
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.segment_ns = int(segment_ns)
        self.clip_pattern = clip_pattern
        self.max_clips = max_clips
        self._proc: subprocess.Popen | None = None

    # ── capability (graceful when gst-launch is absent) ──────────────────────
    @staticmethod
    def available() -> bool:
        """True iff the gst-launch CLI is on PATH. Lets callers feature-detect the
        optional live path without triggering an import-time or run-time crash."""
        return shutil.which(GST_LAUNCH_BIN) is not None

    def open_target(self) -> str:
        """Source ABC contract. The recorder uses `build_command()`; this returns the
        RTSP URL so the source still resolves like RtspSource for callers that only
        need the target string. Raising here is avoided so imports/wiring never crash;
        absence of the CLI is surfaced by `available()` / `start()`."""
        return self.url

    # ── pipeline / command construction ──────────────────────────────────────
    def location(self, out_dir: str | Path | None = None) -> str:
        """Absolute splitmuxsink `location=` (out_dir + clip pattern)."""
        d = Path(out_dir) if out_dir is not None else self.out_dir
        if d is None:
            raise ValueError("GStreamerSource: no out_dir given for clip location")
        return str(Path(d) / self.clip_pattern)

    def pipeline(self, out_dir: str | Path | None = None) -> str:
        """The gst-launch pipeline string (stream-copy, keyframe-aligned 10 s clips)."""
        return (
            f"rtspsrc location={self.url} "
            f"! rtph264depay ! h264parse "
            f"! splitmuxsink max-size-time={self.segment_ns} "
            f"location={self.location(out_dir)}"
        )

    def build_command(self, out_dir: str | Path | None = None) -> list[str]:
        """The argv handed to subprocess. `-e` sends EOS on shutdown so the final
        (partial) clip is finalised cleanly rather than truncated."""
        return [
            GST_LAUNCH_BIN, "-e",
            "rtspsrc", f"location={self.url}",
            "!", "rtph264depay", "!", "h264parse",
            "!", "splitmuxsink",
            f"max-size-time={self.segment_ns}",
            f"location={self.location(out_dir)}",
        ]

    # ── start / stop the recorder ────────────────────────────────────────────
    def start(self, out_dir: str | Path | None = None) -> subprocess.Popen:
        """Launch the recorder subprocess. Raises GStreamerUnavailable (catchable) if
        the CLI is missing, so the caller can fall back to the PyAV path."""
        if not self.available():
            raise GStreamerUnavailable(
                f"{GST_LAUNCH_BIN} not found on PATH — install gstreamer1.0-tools "
                "(and good/bad plugins) or use the PyAV RtspSource live path instead."
            )
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        d = Path(out_dir) if out_dir is not None else self.out_dir
        if d is None:
            raise ValueError("GStreamerSource: no out_dir given to start()")
        d.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            self.build_command(d),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return self._proc

    def stop(self, timeout: float = 10.0) -> int | None:
        """Signal EOS (SIGINT, like Ctrl-C on gst-launch -e) and wait. Returns the
        process return code, or None if nothing was running."""
        proc, self._proc = self._proc, None
        if proc is None:
            return None
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)   # gst-launch -e: clean EOS, finalise clip
                proc.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                proc.kill()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass
        return proc.returncode

    def clips(self, out_dir: str | Path | None = None) -> list[Path]:
        """Enumerate the .mp4 clips produced so far, sorted by name (== record order).
        Handed to the existing cataloging path (sha256 + SegmentRecord) just like a
        PyAV SegmentData.path."""
        d = Path(out_dir) if out_dir is not None else self.out_dir
        if d is None or not Path(d).exists():
            return []
        stem = self.clip_pattern.split("%", 1)[0]   # "seg_"
        return sorted(p for p in Path(d).glob(f"{stem}*.mp4") if p.is_file())

    @property
    def label(self) -> str:
        return self.url


def catalog_live_clips(
    src: GStreamerSource, cam, cfg, base_time, out_dir: str | Path | None = None,
):
    """Turn the clips a GStreamerSource produced into the SAME record shape PyAV
    segments get (paths + sha256 checksum → SegmentRecord), so live clips travel the
    existing cataloging path. Reuses ingest._sha256 (late import avoids a cycle).

    Yields (clip_path, SegmentRecord) so the caller can store + publish exactly as
    ingest_camera does for PyAV SegmentData."""
    from datetime import timedelta

    from .ingest import _sha256                         # reuse, don't duplicate
    from .records import SegmentRecord

    clip_sec = cfg.segmenting.storage_clip_duration_sec
    for idx, clip in enumerate(src.clips(out_dir)):
        start = base_time + timedelta(seconds=idx * clip_sec)
        end = start + timedelta(seconds=clip_sec)
        rec = SegmentRecord(
            camera_id=cam.id, store_id=cfg.store_id, index=idx,
            start_time=start.isoformat(), end_time=end.isoformat(),
            duration_sec=float(clip_sec), storage_path="", codec="h264",
            checksum=_sha256(clip),
        )
        yield clip, rec


def source_for_camera(cam, project_root: Path) -> Source:
    """Pick a source for a camera: prefer the RTSP env var if set, else the file.

    Default behaviour is unchanged (PyAV File/RtspSource). A camera opts into the
    GStreamer live recorder only when `INGEST_SOURCE_BACKEND=gstreamer` (or a
    per-camera `source_backend: gstreamer`) AND an RTSP URL is available — and only
    if the gst-launch CLI is actually present; otherwise we transparently fall back
    to the PyAV RtspSource so existing callers/tests keep working everywhere."""
    rtsp_url = os.environ.get(cam.rtsp_url_env) if cam.rtsp_url_env else None
    backend = (getattr(cam, "source_backend", None)
               or os.environ.get("INGEST_SOURCE_BACKEND") or "pyav").lower()

    if backend == "gstreamer" and rtsp_url and GStreamerSource.available():
        return GStreamerSource(rtsp_url)
    if rtsp_url:
        return RtspSource(rtsp_url)
    if cam.source_file:
        return FileSource(project_root / cam.source_file)
    raise ValueError(f"camera {cam.id}: no rtsp_url_env set and no source_file")
