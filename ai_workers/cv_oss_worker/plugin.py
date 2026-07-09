"""CvOssDetector — the OSS open-vocab detector behind the DetectionPlugin
contract (master plan Phase 4 / spec/11).

Wraps **ultralytics YOLOE** (weights `models/yoloe/yoloe-11s-seg.pt`) but the
real model load is *lazy and optional*: if ultralytics or the weights are
missing — or anything else goes wrong — `infer()` transparently falls back to
`fake_infer()`, which returns deterministic, schema-valid `Detection` objects
(schemas.py). That keeps the worker, the 10-point plugin test (A11), and CI
runnable on a box with no GPU / no weights.

Task: `detection`  (contract: frames|clip_path -> detections[]).
infer(request): {clip_path, window_id, camera_id, ...}
infer -> {detections: [Detection-dict, ...], events: [CommonEvent-dict, ...],
          model_run: {...}}  (plain dicts — they cross the queue/DB boundary).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

from model_plugins.base.plugin import Health, ModelPlugin, PluginConfig      # noqa: E402
from model_plugins.base.schemas import CommonEvent, Detection                # noqa: E402

from .tracking import Tracker

# Default weights location (configs/models.yaml: model_id jameslahm/yoloe-11s-seg,
# local: models/yoloe). Resolved lazily — absence is non-fatal.
DEFAULT_WEIGHTS = _REPO_ROOT / "models" / "yoloe" / "yoloe-11s-seg.pt"

# Open-vocab prompt set for retail (YOLOE set_classes); also the fake_infer labels.
DEFAULT_CLASSES = ["person", "handbag", "backpack", "bottle", "box", "item"]

# ---- Tier-0/1 frame gating (spec/16 §Frame gating) --------------------------
# Detect on ~1 frame/sec, and skip a sampled frame when the scene barely moved:
# a frame whose motion delta vs the LAST DETECTED frame is under the threshold
# never reaches YOLOE. Deltas are measured on 160×90 grayscale thumbnails, so
# the gate itself costs microseconds of numpy per frame.
DETECT_FPS = float(os.environ.get("KATHIRMANI_DETECT_FPS", "1.0"))
MOTION_MIN_CHANGED = float(os.environ.get("KATHIRMANI_MOTION_MIN_CHANGED", "0.15"))
MOTION_PIXEL_DELTA = int(os.environ.get("KATHIRMANI_MOTION_PIXEL_DELTA", "25"))


def motion_changed_fraction(prev, cur, pixel_delta: int = MOTION_PIXEL_DELTA) -> float:
    """Fraction of pixels whose absolute grayscale delta exceeds `pixel_delta`.

    Pure numpy on small grayscale arrays; comparing against the last *detected*
    frame (not the previous sample) lets slow drift accumulate until it trips
    the threshold instead of being skipped forever."""
    import numpy as np
    d = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    return float((d > pixel_delta).mean())


def default_config(profile: str = "nvidia_gb10_retail_balanced") -> PluginConfig:
    """A PluginConfig matching the `detection` task of the active profile,
    so the worker can construct the plugin without a config loader."""
    return PluginConfig(
        task="detection",
        plugin="cv_oss_detector",
        model_id="jameslahm/yoloe-11s-seg",
        runtime="ultralytics",
        endpoint="local",
        profile=profile,
        params={"weights": str(DEFAULT_WEIGHTS), "classes": list(DEFAULT_CLASSES)},
    )


class CvOssDetector(ModelPlugin):
    """Open-vocab detector (YOLOE) for the `detection` task; fake-infer fallback."""

    def __init__(self, config: PluginConfig | None = None) -> None:
        super().__init__(config or default_config())
        self._model = None                      # the lazily-loaded ultralytics model
        self._loaded = False
        self._load_error = ""
        self._classes = list(self.config.params.get("classes", DEFAULT_CLASSES))
        self._weights = Path(self.config.params.get("weights", DEFAULT_WEIGHTS))
        # Persistent multi-object tracker — pure-python, no GPU/weights. Shared
        # across infer() calls so track_ids persist across windows per camera.
        self._tracker = Tracker()
        self._last_gate: dict[str, Any] = {}   # frame-gate stats of the last clip
        # Segment-level result cache: windows of one clip share a single
        # decode+detect+track pass (spec/18 "detect once per segment").
        from collections import OrderedDict
        self._clip_cache: OrderedDict[str, tuple] = OrderedDict()
        self._clip_cache_size = 4
        self._n_cache_hits = 0
        # lightweight metric counters (also mirrored to prometheus in infer())
        self._n_requests = 0
        self._n_fake = 0
        self._last_latency_ms = 0.0

    # ---- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Try to load YOLOE. Idempotent and *non-fatal*: on any failure we
        record the reason and leave the plugin in fake-infer mode."""
        if self._loaded or self._model is not None:
            return
        try:
            if not self._weights.exists():
                raise FileNotFoundError(f"weights not found: {self._weights}")
            from ultralytics import YOLOE       # heavy import — kept lazy
            model = YOLOE(str(self._weights))
            # open-vocab text prompts (best-effort; not all builds expose this)
            try:
                model.set_classes(self._classes, model.get_text_pe(self._classes))
            except Exception:
                pass
            self._model = model
            self._loaded = True
            self._load_error = ""
        except Exception as e:                   # GPU/weights/ultralytics absent
            self._model = None
            self._loaded = False
            self._load_error = f"{type(e).__name__}: {e}"

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        try:                                     # free GPU memory if torch is present
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def health(self) -> Health:
        if self._loaded and self._model is not None:
            return Health(ok=True, detail=f"yoloe loaded ({self._weights.name})")
        # not loaded is still "ok" — fake_infer keeps the worker functional
        detail = "fake-infer mode" + (f" ({self._load_error})" if self._load_error else "")
        return Health(ok=True, detail=detail)

    def metrics(self) -> dict[str, float]:
        return {
            "requests_total": float(self._n_requests),
            "fake_infer_total": float(self._n_fake),
            "clip_cache_hits_total": float(self._n_cache_hits),
            "last_latency_ms": float(self._last_latency_ms),
            "model_loaded": 1.0 if self._loaded else 0.0,
        }

    # ---- tracking accessor (for persistence) -------------------------------
    def live_tracks(self, camera_id: str | None = None) -> list:
        """Snapshot of the tracker's live tracks (used by the worker to persist
        `tracks` rows). Best-effort: returns [] if the tracker is unset."""
        return self._tracker.live_tracks(camera_id) if self._tracker else []

    # ---- inference ---------------------------------------------------------
    def fake_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Deterministic, schema-valid stub (A11 #3). Emits a person standing
        in a shelf zone plus an item, so the suspicious-item hypothesis path is
        exercised end-to-end without a GPU. Deterministic per camera_id."""
        camera_id = request.get("camera_id", "")
        t = float(request.get("window_start_sec", 0.0)) + 0.5
        mn = self.config.model_id
        # bbox = [x, y, w, h] normalized; centers land inside the placeholder
        # shelf polygons ([0..1] wide) so zone-mapping yields a shelf.
        dets = [
            Detection(label="person", confidence=0.92,
                      bbox=[0.40, 0.45, 0.18, 0.40], frame_time_sec=t, model_name=mn),
            Detection(label="item", confidence=0.71,
                      bbox=[0.46, 0.55, 0.06, 0.06], frame_time_sec=t, model_name=mn),
        ]
        return self._package(dets, request, faked=True)

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run detection for one WINDOW of `request['clip_path']`.

        Segment-level result cache (spec/18): all ~14 windows of a clip share
        one decode + YOLOE + tracker pass — the first window computes it, the
        rest slice the cached per-frame results down to their own
        [window_start_sec, window_end_sec) span. Falls back to fake_infer if
        the model isn't loaded or inference raises."""
        self._n_requests += 1
        start = time.perf_counter()
        if not self._loaded:
            self.load()                          # lazy, non-fatal
        clip_path = request.get("clip_path", "")
        try:
            if not clip_path or not Path(clip_path).exists():
                raise RuntimeError("clip missing")
            cached = clip_path in self._clip_cache
            if cached:
                frames, gate = self._clip_cache[clip_path]
                self._clip_cache.move_to_end(clip_path)
                self._n_cache_hits += 1
            else:
                if self._model is None:
                    raise RuntimeError("model unavailable")
                frames = self._run_yoloe(clip_path)
                camera_id = request.get("camera_id", "")
                for fdets in frames:             # tracker: once per CLIP
                    if fdets:
                        self._tracker.update(fdets, camera_id,
                                             fdets[0].frame_time_sec)
                gate = dict(self._last_gate)
                self._clip_cache[clip_path] = (frames, gate)
                while len(self._clip_cache) > self._clip_cache_size:
                    self._clip_cache.popitem(last=False)
            ws = float(request.get("window_start_sec", 0.0))
            we = float(request.get("window_end_sec", 1e9))
            dets = [d for fr in frames for d in fr
                    if ws <= d.frame_time_sec < we]
            self._last_gate = {**gate, "clip_cache_hit": cached}
            out = self._package(dets, request, faked=False,
                                update_tracker=False)
        except Exception as e:
            out = self.fake_infer(request)
            out["model_run"]["error"] = f"fallback: {type(e).__name__}: {e}"
        self._last_latency_ms = (time.perf_counter() - start) * 1000.0
        out["model_run"]["latency_ms"] = self._last_latency_ms
        self._emit_prometheus(out, faked=out["model_run"].get("faked", False))
        return out

    def _run_yoloe(self, clip_path: str) -> list[list[Detection]]:
        """Real ultralytics path: per-FRAME detection lists over the clip.

        Frame gating (spec/16): decode with PyAV, sample ~DETECT_FPS frames per
        second, and run YOLOE only on sampled frames whose motion delta vs the
        last detected frame is ≥ MOTION_MIN_CHANGED (numpy on 160×90 gray
        thumbnails). Keeping the per-frame structure is what lets the tracker
        associate subjects across frames."""
        import av
        frames: list[list[Detection]] = []
        sampled = detected = skipped = 0
        prev_small = None
        with av.open(clip_path) as container:
            vs = container.streams.video[0]
            fps = float(vs.average_rate or 25.0)
            step = max(int(round(fps / DETECT_FPS)), 1)
            for fidx, frame in enumerate(container.decode(vs)):
                if fidx % step:
                    continue
                sampled += 1
                small = frame.reformat(width=160, height=90,
                                       format="gray8").to_ndarray()
                if (prev_small is not None
                        and motion_changed_fraction(prev_small, small)
                        < MOTION_MIN_CHANGED):
                    skipped += 1
                    continue
                prev_small = small
                detected += 1
                t = round(fidx / fps, 3)
                frames.append(self._detect_frame(
                    frame.to_ndarray(format="rgb24"), t))
        self._last_gate = {"frames_sampled": sampled, "frames_detected": detected,
                           "frames_skipped_static": skipped,
                           "detect_fps": DETECT_FPS,
                           "motion_min_changed": MOTION_MIN_CHANGED}
        return frames

    def _detect_frame(self, rgb, t: float) -> list[Detection]:
        """Run YOLOE on one RGB ndarray frame; parse boxes to Detections."""
        frame_dets: list[Detection] = []
        for r in self._model.predict(rgb, verbose=False):
            names = getattr(r, "names", {}) or {}
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            ow = float(getattr(r, "orig_shape", rgb.shape)[1]) or 1.0
            oh = float(getattr(r, "orig_shape", rgb.shape)[0]) or 1.0
            for b in boxes:
                cls = int(b.cls.item()) if hasattr(b, "cls") else 0
                conf = float(b.conf.item()) if hasattr(b, "conf") else 0.0
                x1, y1, x2, y2 = b.xyxy[0].tolist()  # pixel [x1, y1, x2, y2]
                bbox = [x1 / ow, y1 / oh, (x2 - x1) / ow, (y2 - y1) / oh]
                frame_dets.append(Detection(
                    label=str(names.get(cls, cls)), confidence=conf,
                    bbox=[round(v, 5) for v in bbox], frame_time_sec=t,
                    model_name=self.config.model_id))
        return frame_dets

    # ---- packaging: detections + the §8.3 event hypothesis -----------------
    def _package(self, dets: list[Detection] | list[list[Detection]],
                 request: dict[str, Any], faked: bool,
                 update_tracker: bool = True) -> dict[str, Any]:
        # Assign persistent track_ids (in place) before building events/rows, so
        # detections, the §8.3 event, and the tracks table all share one identity.
        # The real path tracks once per CLIP in infer() (update_tracker=False
        # here); per-frame lists are fed one frame at a time (that's what lets
        # the tracker associate the same subject across frames).
        camera_id = request.get("camera_id", "")
        if dets and isinstance(dets[0], list):
            flat: list[Detection] = []
            for frame_dets in dets:
                if not frame_dets:
                    continue
                if update_tracker:
                    self._tracker.update(frame_dets, camera_id,
                                         frame_dets[0].frame_time_sec)
                flat.extend(frame_dets)
            dets = flat
        elif update_tracker:
            frame_time = float(request.get("window_start_sec", 0.0))
            if dets:
                frame_time = dets[0].frame_time_sec or frame_time
            self._tracker.update(dets, camera_id, frame_time)
        events = self.detections_to_events(dets, request)
        return {
            "detections": [asdict(d) for d in dets],
            "events": [asdict(e) for e in events],
            "model_run": {
                "model_profile_name": self.config.profile,
                "model_id": self.config.model_id,
                "task": self.config.task,
                "runtime": self.config.runtime,
                "faked": faked,
                "error": None,
                "frame_gate": dict(self._last_gate),
            },
        }

    def detections_to_events(self, dets: list[Detection],
                             request: dict[str, Any]) -> list[CommonEvent]:
        """Basic Phase-4 hypothesis (full rules are Phase 5): a `person` whose
        bbox-center falls in a `shelf` zone, with another non-person object
        present, ⇒ one `suspicious_item_interaction` CommonEvent flagged for
        VLM verification (event_rules.yaml). Returns [] otherwise."""
        from .zones import load_zones, map_detection_to_zones

        camera_id = request.get("camera_id", "")
        store_id = request.get("store_id") or "kathirmani_01"
        zm = load_zones()
        persons = [d for d in dets if d.label == "person"]
        items = [d for d in dets if d.label != "person"]
        events: list[CommonEvent] = []
        # `dets` spans EVERY frame of the window, so the same subject appears
        # once per frame (~125×). Emit ONE event per subject (track_id) per
        # window, not one per frame appearance — real footage otherwise floods
        # ~600 duplicate events per busy window.
        seen_subjects: set[str] = set()
        for p in persons:
            subject = p.track_id or "untracked"
            if subject in seen_subjects:
                continue
            zone_ids = map_detection_to_zones(tuple(p.bbox), camera_id, zm)
            zone_types = zm.zone_types(zone_ids)
            if "shelf" in zone_types and items:
                seen_subjects.add(subject)
                # Carry real persistent track_ids for the subjects in this event;
                # the person's id leads so the rule engine buckets by that subject.
                track_ids = [p.track_id] if p.track_id is not None else []
                track_ids += sorted({i.track_id for i in items if i.track_id is not None})
                ev = CommonEvent(
                    event_id=self._event_id(request, "suspicious_item_interaction",
                                            subject),
                    store_id=store_id,
                    camera_id=camera_id,
                    event_type="suspicious_item_interaction",
                    severity="medium",
                    confidence=round(min(p.confidence, max(i.confidence for i in items)), 4),
                    source_engine="cv_oss_detector",
                    ai_window_id=request.get("window_id"),
                    segment_id=request.get("segment_id"),
                    objects=sorted({d.label for d in dets}),
                    track_ids=track_ids,
                    zones=zone_ids,
                    needs_vlm_verification=True,
                    evidence_path=request.get("clip_path"),
                )
                events.append(ev)
        return events

    @staticmethod
    def _event_id(request: dict[str, Any], kind: str, subject: str = "") -> str:
        import uuid
        seed = (f"{request.get('window_id','')}|{request.get('camera_id','')}"
                f"|{kind}|{subject}")
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    # ---- prometheus mirror (best-effort) -----------------------------------
    def _emit_prometheus(self, out: dict[str, Any], faked: bool) -> None:
        if faked:
            self._n_fake += 1
        try:
            from model_plugins.base import metrics as M
            lbl = M.labels(self.config)
            status = "fake" if faked else ("error" if out["model_run"].get("error") else "ok")
            M.model_requests_total.labels(**lbl, status=status).inc()
            M.model_latency_ms.labels(**lbl).observe(out["model_run"]["latency_ms"])
            M.model_clips_processed_total.labels(**lbl).inc()
            if out["model_run"].get("error") and not faked:
                M.model_errors_total.labels(**lbl, error_type="infer").inc()
        except Exception:
            pass                                  # metrics never break inference
