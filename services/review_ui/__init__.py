"""review_ui — human approve/reject of loss hypotheses (master plan Phase 7).

A separate FastAPI app (port 8010) presenting an incident's evidence package
(events + timeline + VLM verdict + evidence clip path) for a human verdict, and
appending an immutable chain-of-custody audit record (reviewer + timestamp +
model_version + decision). Review/verdict ONLY — annotation lives in CVAT +
Label Studio (Kathirmani viz-vs-annotation split).

The FastAPI instance lives in `services.review_ui.app` (uvicorn target
`services.review_ui.app:app`); it is deliberately not re-exported here so the
`app` submodule stays importable (tests monkeypatch its module attributes).
"""
from __future__ import annotations
