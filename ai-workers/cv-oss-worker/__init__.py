"""cv-oss-worker — the OSS CV worker (master plan Phase 4).

Consumes `ai_window.ready` jobs, runs an open-vocab detector behind the
`DetectionPlugin` contract (model-plugins/base), writes detections + a
§8.3 CommonEvent, maps objects to zones, and emits a suspicious-item
hypothesis. GPU/weights/Postgres are all optional — everything degrades to
a deterministic `fake_infer` / file-queue path so it runs hermetically.
"""
from __future__ import annotations

# Imports tolerate both package use (`from cv_oss_worker import ...`) and the
# repo's path-based test setup (dir name `cv-oss-worker` isn't a valid module
# name, so relative-import context isn't always present).
try:
    from .plugin import CvOssDetector
    from .zones import (ZoneMap, load_zones, map_detection_to_zones,
                        point_in_polygon)
except ImportError:                                   # path-based / direct import
    from plugin import CvOssDetector
    from zones import (ZoneMap, load_zones, map_detection_to_zones,
                       point_in_polygon)

__all__ = [
    "CvOssDetector",
    "ZoneMap",
    "load_zones",
    "map_detection_to_zones",
    "point_in_polygon",
]
