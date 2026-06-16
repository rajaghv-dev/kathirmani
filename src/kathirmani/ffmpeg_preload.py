"""Pre-load PyAV's bundled FFmpeg shared libraries with RTLD_GLOBAL.

torchcodec resolves FFmpeg symbols via the global symbol table, so on Linux we
dlopen PyAV's bundled ``av.libs/*.so`` before importing torch/torchcodec. This
avoids needing a system FFmpeg install or a hand-set LD_LIBRARY_PATH.

On macOS (Apple Mac Studio) PyAV/torchcodec ship their own dylibs and resolve
them without this dance, so the preload is a Linux-only best-effort no-op
elsewhere. Must run before ANY import of torch / torchcodec / transformers.
"""
from __future__ import annotations

import ctypes
import platform
import sys
from pathlib import Path

# Bundled lib basenames, in dependency order. The trailing soname version can
# drift between PyAV releases, so we also glob by stem as a fallback.
_AV_LIB_STEMS = [
    "libavutil",
    "libswresample",
    "libswscale",
    "libavcodec",
    "libavformat",
    "libavfilter",
]


def _candidate_dirs(project_root: Path | None) -> list[Path]:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    dirs = [Path(sys.prefix) / "lib" / pyver / "site-packages" / "av.libs"]
    if project_root is not None:
        dirs.append(project_root / ".venv" / "lib" / pyver / "site-packages" / "av.libs")
    # Last resort: scan whatever site-packages this interpreter actually uses.
    try:
        import av  # noqa: F401  (only to locate its install dir)

        dirs.append(Path(av.__file__).resolve().parent.parent / "av.libs")
    except Exception:
        pass
    return dirs


def preload_av_ffmpeg(project_root: Path | None = None) -> None:
    """Best-effort dlopen of PyAV's bundled FFmpeg libs. Never raises."""
    if platform.system() != "Linux":
        return  # macOS/other: torchcodec resolves its own ffmpeg

    avlibs = next((d for d in _candidate_dirs(project_root) if d.exists()), None)
    if avlibs is None:
        return

    for stem in _AV_LIB_STEMS:
        so = next(iter(sorted(avlibs.glob(f"{stem}.so*"))), None)
        if so is not None:
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
