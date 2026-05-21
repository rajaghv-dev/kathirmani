"""
conftest.py — shared pytest fixtures and session-wide setup.

Sets environment variables required by qwen-vl-utils / torchcodec before any
import of transformers, and pre-loads PyAV's bundled FFmpeg shared libraries
with RTLD_GLOBAL so torchcodec can resolve them without system FFmpeg.
"""
import ctypes
import os
import pathlib
import sys

# ---------------------------------------------------------------------------
# 1. Environment variables — must be set before importing transformers.
# ---------------------------------------------------------------------------
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("VIDEO_MAX_PIXELS", "200704")
os.environ.setdefault("FPS", "2.0")
os.environ.setdefault("FPS_MAX_FRAMES", "240")
os.environ.setdefault("FPS_MIN_FRAMES", "4")


# ---------------------------------------------------------------------------
# 2. Pre-load PyAV's bundled FFmpeg libs (mirrors _preload_av_ffmpeg in
#    run_inference.py) so torchcodec can find them via RTLD_GLOBAL.
# ---------------------------------------------------------------------------
def _preload_av_ffmpeg():
    candidates = [
        pathlib.Path(sys.prefix) / "lib/python3.12/site-packages/av.libs",
        pathlib.Path(__file__).parent.parent / ".venv/lib/python3.12/site-packages/av.libs",
    ]
    avlibs = next((p for p in candidates if p.exists()), None)
    if avlibs is None:
        return
    for name in [
        "libavutil.so.60",
        "libswresample.so.6",
        "libswscale.so.9",
        "libavcodec.so.62",
        "libavformat.so.62",
        "libavfilter.so.11",
    ]:
        so = avlibs / name
        if so.exists():
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_av_ffmpeg()
