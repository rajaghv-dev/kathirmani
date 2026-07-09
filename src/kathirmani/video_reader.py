"""PyAV video-reader backend for qwen-vl-utils (Marlin/Qwen video loading).

qwen-vl-utils ships three readers — torchcodec, torchvision, decord — and on
this box all three are unavailable: torchcodec can't dlopen FFmpeg (PyAV ≥ 17
bundles FFmpeg 8 with auditwheel-mangled sonames, which torchcodec's exact
soname linkage cannot resolve), torchvision ≥ 0.23 removed ``io.read_video``,
and decord has no aarch64 wheel. PyAV itself decodes fine — it is the repo's
vetted media dependency (spec/10 ingestion stack) — so we register it as a
first-class backend and prefer it whenever torchcodec is broken.

Usage (before the first video is processed, ideally before transformers import):

    from kathirmani.video_reader import ensure_reader
    ensure_reader()
"""
from __future__ import annotations

import os


def _read_video_pyav(ele: dict):
    """Read ``ele['video']`` with PyAV. Mirrors the torchvision backend's
    contract: returns (video TCHW uint8 tensor, video_metadata, sample_fps)."""
    import av
    import numpy as np
    import torch
    from qwen_vl_utils.vision_process import smart_nframes

    path = str(ele["video"])
    if path.startswith("file://"):
        path = path[7:]
    start = float(ele.get("video_start") or 0.0)
    end = ele.get("video_end")

    with av.open(path) as container:
        vs = container.streams.video[0]
        fps = float(vs.average_rate or 25.0)
        duration = 0.0
        if vs.duration is not None and vs.time_base is not None:
            duration = float(vs.duration * vs.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        clip_end = min(float(end), duration) if end is not None else duration
        total_frames = max(int(round((clip_end - start) * fps)), 1)

        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        wanted = set(idx.tolist())
        last_wanted = int(idx[-1].item()) if nframes else -1

        frames: list[np.ndarray] = []
        i = 0
        for frame in container.decode(vs):
            t = float(frame.pts * vs.time_base) if frame.pts is not None else start
            if t < start:
                continue
            if end is not None and t > clip_end:
                break
            if i in wanted:
                frames.append(frame.to_ndarray(format="rgb24"))
            i += 1
            if i > last_wanted:
                break

    if not frames:
        raise RuntimeError(f"pyav reader: no frames decoded from {path}")
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    sample_fps = len(frames) / max(total_frames, 1e-6) * fps
    video_metadata = dict(
        fps=fps,
        frames_indices=idx[: len(frames)],
        total_num_frames=total_frames,
        video_backend="pyav",
    )
    return video, video_metadata, sample_fps


def torchcodec_usable() -> bool:
    """True if torchcodec's native decoder actually loads on this box."""
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        return True
    except Exception:
        return False


def ensure_reader() -> str:
    """Register the ``pyav`` backend and pick a working default reader.

    Honors an explicit FORCE_QWENVL_VIDEO_READER; otherwise prefers torchcodec
    (GPU decode) when loadable and falls back to pyav. Returns the chosen name.
    """
    from qwen_vl_utils import vision_process as vp

    vp.VIDEO_READER_BACKENDS["pyav"] = _read_video_pyav
    chosen = os.environ.get("FORCE_QWENVL_VIDEO_READER")
    if not chosen:
        chosen = "torchcodec" if torchcodec_usable() else "pyav"
        os.environ["FORCE_QWENVL_VIDEO_READER"] = chosen
    # vision_process freezes the env var into a module constant at import time
    # and memoizes the backend choice — override both so the choice sticks
    # regardless of import order.
    vp.FORCE_QWENVL_VIDEO_READER = chosen
    vp.get_video_reader_backend.cache_clear()
    return chosen
