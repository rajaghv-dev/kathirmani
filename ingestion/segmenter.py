"""Split a Source into fixed-length clips with PyAV (one decode pass).

Re-encodes (libx264 ultrafast) into N-second clips and, in the same pass, collects
the first frame (thumbnail) + a few sampled frames (health) so we don't decode twice.
Mirrors `src/kathirmani/pipeline.py:trim_video`. GStreamer would replace this for the
HW-accelerated/at-scale path (see sources.py); the rest of ingestion is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np

try:                                   # use the repo's bundled-FFmpeg preload if present
    from kathirmani.ffmpeg_preload import preload_av_ffmpeg
except Exception:                      # pragma: no cover
    def preload_av_ffmpeg(*_a, **_k): ...


@dataclass
class SegmentData:
    index: int
    path: Path
    duration_sec: float
    fps: float
    codec: str
    width: int
    height: int
    first_frame: np.ndarray | None = None      # RGB, for thumbnail
    samples: list[np.ndarray] = field(default_factory=list)  # small RGB frames, for health


class _OpenSegment:
    """One in-flight clip encoder (clips can overlap, so ≥1 may be open)."""

    def __init__(self, idx: int, start_t: float, out_dir: Path, in_vs, fps: float):
        self.idx = idx
        self.start_t = start_t
        self.path = out_dir / f"seg_{idx:05d}.mp4"
        self.out = av.open(str(self.path), "w", format="mp4")
        vs = self.out.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        vs.width, vs.height, vs.pix_fmt = in_vs.width, in_vs.height, "yuv420p"
        vs.options = {"crf": "23", "preset": "ultrafast"}
        self.vs = vs
        self.first: np.ndarray | None = None
        self.samples: list[np.ndarray] = []
        self.fcount = 0


def segment_source(
    source, out_dir: Path, clip_sec: float,
    duration_limit: float | None = None, health_samples: int = 5,
    overlap_sec: float = 0.0,
) -> Iterator[SegmentData]:
    """Cut `source` into `clip_sec` clips; consecutive clips overlap by
    `overlap_sec` (stride = clip_sec - overlap_sec), so events crossing a clip
    boundary are fully contained in the next clip. overlap_sec=0 keeps the old
    back-to-back behaviour. During an overlap a frame is encoded into every
    clip that spans it (multiple encoders open at once)."""
    if not (0 <= overlap_sec < clip_sec):
        raise ValueError("overlap_sec must satisfy 0 <= overlap_sec < clip_sec")
    stride = clip_sec - overlap_sec
    preload_av_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = source.open_target()

    with av.open(target) as inp:
        in_vs = inp.streams.video[0]
        fps = float(in_vs.average_rate or 25)
        tb = in_vs.time_base
        sample_every = max(1, int(fps * clip_sec / max(1, health_samples)))
        open_segs: dict[int, _OpenSegment] = {}
        last_t = 0.0

        def _close_emit(seg: _OpenSegment, end_t: float) -> SegmentData:
            for pkt in seg.vs.encode():
                seg.out.mux(pkt)
            seg.out.close()
            return SegmentData(
                index=seg.idx, path=seg.path,
                duration_sec=round(end_t - seg.start_t, 3),
                fps=fps, codec="h264", width=in_vs.width, height=in_vs.height,
                first_frame=seg.first, samples=seg.samples,
            )

        for frame in inp.decode(in_vs):
            t = float(frame.pts * tb) if frame.pts is not None else last_t
            last_t = t
            if duration_limit is not None and t > duration_limit:
                break

            # close clips this frame has passed
            for k in sorted(open_segs):
                if t >= k * stride + clip_sec:
                    yield _close_emit(open_segs.pop(k), k * stride + clip_sec)

            # every clip index whose span [k*stride, k*stride+clip_sec) holds t
            k_lo = max(0, int((t - clip_sec) // stride) + 1)
            k_hi = int(t // stride)
            for k in range(k_lo, k_hi + 1):
                if k not in open_segs:
                    open_segs[k] = _OpenSegment(k, k * stride, out_dir, in_vs, fps)
                seg = open_segs[k]
                if seg.first is None:
                    seg.first = frame.to_ndarray(format="rgb24")
                if seg.fcount % sample_every == 0 and len(seg.samples) < health_samples:
                    seg.samples.append(frame.reformat(
                        width=160, height=90, format="rgb24").to_ndarray())
                # Re-encoded frames must NOT inherit the source PTS (invalid DTS
                # on real footage) — rebase to a per-clip frame counter.
                nf = frame.reformat(format="yuv420p")
                nf.pts = seg.fcount
                nf.time_base = Fraction(1, int(round(fps))) if fps else None
                seg.fcount += 1
                for pkt in seg.vs.encode(nf):
                    seg.out.mux(pkt)

        for k in sorted(open_segs):
            yield _close_emit(open_segs.pop(k), last_t + (1.0 / fps if fps else 0))
