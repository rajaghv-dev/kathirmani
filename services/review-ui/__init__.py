"""review-ui — human approve/reject of loss hypotheses (master plan Phase 7).

A separate FastAPI app (port 8010) presenting an incident's evidence package
(events + timeline + VLM verdict + evidence clip path) for a human verdict, and
appending an immutable chain-of-custody audit record (reviewer + timestamp +
model_version + decision). Review/verdict ONLY — annotation lives in CVAT +
Label Studio (Kathirmani viz-vs-annotation split).

Imports tolerate both package use and the repo's path-based test setup (the dir
name `review-ui` isn't a valid module name).
"""
from __future__ import annotations

try:
    from .app import app
except ImportError:                                     # path-based / direct import
    from app import app

__all__ = ["app"]
