"""
conftest.py — shared pytest fixtures and session-wide setup.

Puts src/ on sys.path so `import kathirmani` works without an editable install,
sets the env vars qwen-vl-utils / torchcodec expect, and pre-loads PyAV's
bundled FFmpeg libs (Linux) via the package's portable preloader.
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Environment variables — must be set before importing transformers.
# ---------------------------------------------------------------------------
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")
os.environ.setdefault("VIDEO_MAX_PIXELS", "200704")
os.environ.setdefault("FPS", "2.0")
os.environ.setdefault("FPS_MAX_FRAMES", "240")
os.environ.setdefault("FPS_MIN_FRAMES", "4")

# ---------------------------------------------------------------------------
# Pre-load PyAV's bundled FFmpeg libs (no-op off Linux / if torch absent).
# ---------------------------------------------------------------------------
try:
    from kathirmani.ffmpeg_preload import preload_av_ffmpeg

    preload_av_ffmpeg(_ROOT)
except Exception:
    pass
