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


def segment_source(
    source, out_dir: Path, clip_sec: float,
    duration_limit: float | None = None, health_samples: int = 5,
) -> Iterator[SegmentData]:
    preload_av_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = source.open_target()

    with av.open(target) as inp:
        in_vs = inp.streams.video[0]
        fps = float(in_vs.average_rate or 25)
        tb = in_vs.time_base
        sample_every = max(1, int(fps * clip_sec / max(1, health_samples)))

        cur_idx: int | None = None
        out = out_vs = None
        path: Path | None = None
        first = None
        samples: list[np.ndarray] = []
        seg_start = 0.0
        last_t = 0.0
        fcount = 0

        def _open(idx: int):
            p = out_dir / f"seg_{idx:05d}.mp4"
            o = av.open(str(p), "w", format="mp4")
            vs = o.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
            vs.width, vs.height, vs.pix_fmt = in_vs.width, in_vs.height, "yuv420p"
            vs.options = {"crf": "23", "preset": "ultrafast"}
            return p, o, vs

        def _close_emit(end_t: float) -> SegmentData:
            for pkt in out_vs.encode():
                out.mux(pkt)
            out.close()
            return SegmentData(
                index=cur_idx, path=path, duration_sec=round(end_t - seg_start, 3),
                fps=fps, codec="h264", width=in_vs.width, height=in_vs.height,
                first_frame=first, samples=samples,
            )

        for frame in inp.decode(in_vs):
            t = float(frame.pts * tb) if frame.pts is not None else last_t
            last_t = t
            if duration_limit is not None and t > duration_limit:
                break
            idx = int(t // clip_sec)
            if idx != cur_idx:
                if out is not None:
                    yield _close_emit(seg_start + clip_sec)
                cur_idx = idx
                seg_start = idx * clip_sec
                path, out, out_vs = _open(idx)
                first, samples, fcount = None, [], 0

            if first is None:
                first = frame.to_ndarray(format="rgb24")
            if fcount % sample_every == 0 and len(samples) < health_samples:
                small = frame.reformat(width=160, height=90, format="rgb24").to_ndarray()
                samples.append(small)
            fcount += 1

            # Re-encoded frames must NOT inherit the source PTS: each segment has
            # a fresh encoder, and source timestamps (huge, in the source
            # time_base) produce a non-monotonic/invalid DTS at a segment
            # boundary on real footage. Rebase to a per-segment frame counter
            # in 1/fps time_base (PyAV 17 requires explicit pts).
            nf = frame.reformat(format="yuv420p")
            nf.pts = fcount - 1                      # frames since segment start
            nf.time_base = Fraction(1, int(round(fps))) if fps else None
            for pkt in out_vs.encode(nf):
                out.mux(pkt)

        if out is not None:
            yield _close_emit(last_t + (1.0 / fps if fps else 0))
