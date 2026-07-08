"""evidence_builder — incident → reviewable, auditable evidence package (Phase 7).

Gathers an incident's events + their `video_segments` + `vlm_observations`,
stitches a ~20-sec pre/during/post evidence clip with PyAV (manifest-only if the
source clips are missing — degrades gracefully), sha256s the clip for chain-of-
custody, builds a time-ordered `timeline`, and writes `evidence_path`+`summary`
back to the incident. The review_ui (separate FastAPI app) presents this package
to a human for approve/reject. NOT an annotation tool (that's CVAT/Label Studio).
"""
from __future__ import annotations

from .builder import (assemble_timeline, build_evidence, build_package,
                      sha256_bytes, sha256_file, stitch_clip)

__all__ = [
    "build_evidence",
    "build_package",
    "assemble_timeline",
    "stitch_clip",
    "sha256_file",
    "sha256_bytes",
]
