"""vss-eval-worker — VSS-style OSS+NVIDIA parity (master plan Phase 9 / spec/10-11).

Implements VSS-style capabilities in OSS + NVIDIA models with **no VSS runtime
dependency** (`allow_vss_runtime_dependency: false`; VSS is reference-only) and a
parity harness that measures how close the stack gets. Centerpiece = the §A4.2
hierarchical long-video summarization (clip -> 5min -> hour -> incident report).

The summarizer (Nemotron-3-Nano) and all DB access are optional: a deterministic
`fake_infer` / no-DB path keeps the pipeline, the parity report, the A11 plugin
test, and CI runnable with no GPU / weights / DB.
"""
from __future__ import annotations

# Tolerate both package use and the repo's path-based test setup (the hyphenated
# dir name `vss-eval-worker` isn't a valid module name).
try:
    from .summarizer import (DEFAULT_MODEL_ID, NemotronSummaryPlugin, default_config)
    from .lvs import (CLIP_SECONDS, FIVE_MIN_SECONDS, HOUR_SECONDS,
                      event_caption_line, filter_events, is_suspicious, summarize)
    from .parity import (CAPABILITIES, build_report, write_report,
                        DEFAULT_REPORT_PATH)
except ImportError:                                   # path-based / direct import
    from summarizer import (DEFAULT_MODEL_ID, NemotronSummaryPlugin, default_config)
    from lvs import (CLIP_SECONDS, FIVE_MIN_SECONDS, HOUR_SECONDS,
                     event_caption_line, filter_events, is_suspicious, summarize)
    from parity import (CAPABILITIES, build_report, write_report,
                       DEFAULT_REPORT_PATH)

__all__ = [
    "NemotronSummaryPlugin", "default_config", "DEFAULT_MODEL_ID",
    "summarize", "filter_events", "is_suspicious", "event_caption_line",
    "CLIP_SECONDS", "FIVE_MIN_SECONDS", "HOUR_SECONDS",
    "build_report", "write_report", "CAPABILITIES", "DEFAULT_REPORT_PATH",
]
