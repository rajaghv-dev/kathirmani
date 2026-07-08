"""Kathirmani video-intelligence pipeline.

This package runs unchanged on two very different boxes — an NVIDIA DGX (CUDA)
and an Apple Mac Studio (Metal/MPS) — by detecting the accelerator at runtime.
See ``kathirmani.device`` and ``spec/06-hardware-portability.md``.
"""
from pathlib import Path

__version__ = "1.0.0"

# Repo root resolved relative to this file: src/kathirmani/__init__.py -> parents[2].
# Everything (store-videos/, models/, results/) lives at the repo root regardless
# of where the CLI is launched from, so we anchor on the package, not the cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Source footage home (gitignored). Falls back to the repo root, where the .mkv
# files sat before store-videos/ existed, so old checkouts keep working.
VIDEOS_DIR = PROJECT_ROOT / "store-videos"
if not VIDEOS_DIR.is_dir():
    VIDEOS_DIR = PROJECT_ROOT

# Video container globs the pipeline and viewer scan for. Single source of truth
# so run_inference, pipeline, and the Streamlit viewer stay in agreement.
VIDEO_EXTS = ("*.mkv", "*.mp4", "*.avi", "*.mov", "*.webm")

__all__ = ["PROJECT_ROOT", "MODELS_DIR", "RESULTS_DIR", "VIDEOS_DIR", "VIDEO_EXTS",
           "__version__"]
