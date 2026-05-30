"""Kathirmani Marlin video-intelligence pipeline.

This package runs unchanged on two very different boxes — an NVIDIA DGX (CUDA)
and an Apple Mac Studio (Metal/MPS) — by detecting the accelerator at runtime.
See ``marlin.device`` and ``spec/06-hardware-portability.md``.
"""
from pathlib import Path

__version__ = "1.0.0"

# Repo root resolved relative to this file: src/marlin/__init__.py -> parents[2].
# Everything (videos, models/, results/) lives at the repo root regardless of
# where the CLI is launched from, so we anchor on the package, not the cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

__all__ = ["PROJECT_ROOT", "MODELS_DIR", "RESULTS_DIR", "__version__"]
