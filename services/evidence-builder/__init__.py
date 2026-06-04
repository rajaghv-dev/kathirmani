"""evidence-builder — incident → reviewable, auditable evidence package (Phase 7).

Gathers an incident's events + their `video_segments` + `vlm_observations`,
stitches a ~20-sec pre/during/post evidence clip with PyAV (manifest-only if the
source clips are missing — degrades gracefully), sha256s the clip for chain-of-
custody, builds a time-ordered `timeline`, and writes `evidence_path`+`summary`
back to the incident. The review-ui (separate FastAPI app) presents this package
to a human for approve/reject. NOT an annotation tool (that's CVAT/Label Studio).

Imports tolerate both package use and the repo's path-based test setup (the dir
name `evidence-builder` isn't a valid module name).
"""
from __future__ import annotations

try:
    from .builder import (assemble_timeline, build_evidence, build_package,
                          sha256_bytes, sha256_file, stitch_clip)
except ImportError:                                     # path-based / direct import
    from builder import (assemble_timeline, build_evidence, build_package,
                         sha256_bytes, sha256_file, stitch_clip)

__all__ = [
    "build_evidence",
    "build_package",
    "assemble_timeline",
    "stitch_clip",
    "sha256_file",
    "sha256_bytes",
]
