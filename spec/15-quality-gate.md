# 15 — Image/Video Quality Gate (the QUALITY hook)

> **Forward-looking.** Per the master prompt
> [`00-master-plan-v2.md`](00-master-plan-v2.md)
> v3 ("Plug-and-play models + image-quality gate + POS"). No `quality/` module exists
> yet — this doc is the **target contract** for it. The capability-hook framing it
> plugs into is [16-capability-hooks-profiles-router.md](16-capability-hooks-profiles-router.md);
> the value framing is [01-overview.md](01-overview.md).

## Why quality is a first-class hook

Before any model is asked to *understand* the video, the ingestion layer must decide
whether the frame/chunk is even **usable**. CCTV model accuracy is dominated by input
quality — blur, low light, glare, occlusion, bad angle, low resolution, compression
artifacts, duplicate frames, scene motion, camera obstruction, mis-exposure, missing
timestamp alignment, wrong camera→zone mapping. A small/cheap model performs close to
a frontier model on this narrow retail task **when it is handed better input**: a sharp
frame, the right camera, the right time window, a good ROI/crop, an atomic query, POS
context, rules that remove impossible interpretations, and a frontier model used only
as a teacher/benchmark.

So QUALITY runs **before** WHEN/WHERE/WHAT and gates them:

```text
video / chunk
→ QUALITY scoring
→ frame filtering / enhancement / ROI selection
→ cheap model first  (WHEN → WHERE)
→ escalate only when uncertain  (→ WHAT / stronger model)
→ POS / rules correlation
→ human review
```

This is the same cascade-gating thesis the cascade already embodies for the VLM
([05-performance-and-optimizations.md](05-performance-and-optimizations.md),
[11-model-plugin-policy.md](11-model-plugin-policy.md) §"Don't use the VLM for
everything") — extended one stage earlier, to the pixels themselves.

## Where it lives — extends the existing `ingestion/health.py`

**The camera-health half already exists — do not rebuild it.**
`ingestion/health.py:health_scores()` already computes `black_frame_score`,
`blur_score`, `freeze_score`, `fps_actual`, and a composite `health_score` ∈ [0,1]
(1 = healthy), **numpy-only with no OpenCV dependency** — a deliberate portability
choice ([06](06-hardware-portability.md)). Those are exported as the live
`ingest_camera_*` metrics, and the rule engine already raises `camera_health_issue`
when `health_score <= 0.5` (`configs/event_rules.yaml`, [10](10-platform-roadmap.md)
Phase 5). The QUALITY hook **extends this module; it does not replace it.**

The master prompt names a fresh `src/retail_video_ai/quality/` package; in this repo
the work lands **in/next to `ingestion/health.py`** (the ingestion package, spec/12),
adding the per-chunk *usability summary*, the zone-visibility proxies, and the routing
decision **on top of** the existing health scores:

```text
ingestion/
  health.py             # EXISTING — black/blur/freeze + composite health_score (numpy)
  quality.py     (new)  # usable_frame_score, zone-visibility proxies, per-chunk decision
  quality_schema.py (new)# QualityResult / QualityDecision dataclasses
  quality_router.py (new)# quality → use_for_{when,where,what} routing hints
```

It is a **capability hook** behind the `ImageQualityModel` contract
([16](16-capability-hooks-profiles-router.md)). **The default is the existing numpy
heuristics** (`health.py`); a no-reference IQA model, a small classifier, or an OpenCV
backend are *optional* swaps, never required — keeping the ingestion tier
dependency-light and CPU-portable ([06](06-hardware-portability.md)).

## What it scores

For every video/chunk/frame, compute (0–1, higher = better unless noted):

```text
brightness_score              motion_score
contrast_score                compression_artifact_score   (higher = worse)
blur_score        (worse)     occlusion_score              (higher = worse)
sharpness_score               person_visibility_score
frame_duplicate_score         shelf_visibility_score
usable_frame_score (summary)  billing_counter_visibility_score
camera_health_score           entry_exit_visibility_score
```

`black_frame_score`, `blur_score`, `freeze_score`, and `health_score`
(= `camera_health_score`) are **already produced** by `ingestion/health.py`; the rest are
the QUALITY-hook additions. `usable_frame_score` is the headline summary the router keys
off; the `*_visibility_score`s are **zone-aware** (computed against the camera's
role/zone from `configs/cameras.yaml` + the digital twin, [10](10-platform-roadmap.md)).

### Output schema (`QualityResult`)

```json
{
  "camera_id": "center",
  "chunk_id": "center_000012",
  "start_sec": 60.0,
  "end_sec": 65.0,
  "quality": {
    "brightness_score": 0.81,
    "blur_score": 0.12,
    "sharpness_score": 0.78,
    "motion_score": 0.44,
    "compression_artifact_score": 0.18,
    "person_visibility_score": 0.72,
    "shelf_visibility_score": 0.69,
    "usable_frame_score": 0.76
  },
  "decision": {
    "use_for_when": true,
    "use_for_where": true,
    "use_for_what": false,
    "reason": "usable for temporal and spatial pass; semantic VLM skipped unless routed later"
  }
}
```

`QualityResult` is persisted next to the chunk (the ingestion JSONL / `video_segments`
catalog row, spec/10) so a later replay can re-use or recompute it.

## Gate decisions

The gate maps quality + context to *how hard* to think, never to a verdict:

```text
High quality + simple visible action   → cheap model only
Medium quality + important zone        → cheap model + rule confirmation
Low quality + high-risk candidate      → escalate to stronger model OR mark needs-review
Very poor quality                      → do not infer strongly; raise a camera/frame quality issue
```

The decision is **advisory routing**, consumed by the model router
([16](16-capability-hooks-profiles-router.md) §router): quality thresholds appear in
the routing policy (`skip_if_quality_below`, `usable_frame_score < …`), so a poor frame
cannot produce an overconfident conclusion.

**Thresholds are per-camera/zone, not global.** A chronically dim or soft camera (e.g.
the entry door at night) must **not** be silently gated off by an absolute floor — that
would drop real events on that camera, the opposite of the loss-prevention goal.
Thresholds are **relative to each camera/zone's rolling baseline** (or explicitly
calibrated in `configs/cameras.yaml`); the absolute `usable_frame_score` is only a
last-resort floor.

**Gating trades recall for cost/precision — so the poor-quality path must never silently
drop a high-risk candidate.** "Very poor quality" sends a high-risk candidate to
**needs-review** (a human still looks, flagged "low quality"); it does not suppress it.
Only **low-risk** low-quality chunks are dropped. This is the ingestion-side mirror of
the "Poor video quality, cannot decide" review outcome ([17](17-pos-and-time-alignment.md)).

## Metrics

Additive under the existing **`ingest_*`** namespace ([04](04-observability-stack.md)) —
the camera-health gauges `ingest_camera_blur_score` / `_black_frame_score` /
`_freeze_score` / `_health_score` **already exist**. The QUALITY hook adds the usability
+ rejection counters under the same convention; the master prompt's `retail_video_ai_*`
names are **not** used (one namespace, no fragmentation):

```text
ingest_quality_usable_frame_score      (gauge, by camera/zone)
ingest_quality_usable_frames_total
ingest_quality_rejected_frames_total   (by reason: blur/low_light/duplicate/occlusion)
# chronically-poor cameras already surface via the rule engine's camera_health_issue
```

Stable labels only — `camera_id`, `zone`, `reason`, `run_id` (never raw paths or
captions; the [03](03-models-and-query-modes.md) coupling/cardinality warning applies).
Surfaced in the new **Image Quality / Ingestion Health** dashboard
([08](08-dashboards.md) extension): usable-frame score by camera, rejected frames by
reason, blur/brightness/motion trends, camera quality warnings, and **quality vs query
hit rate** (the proof that better frames → better cheap-model accuracy).

## OTel spans

`retail.quality.score` and `retail.quality.route` join the existing span tree
([13](13-models.md) tracking / master-plan OTel), with attributes `quality_score`,
`usable_frame_score`, `route_to_vlm`, `camera_zone` (no raw text).

## Test gate (target)

Forward-looking; mirrors the master prompt's quality tests:

```text
test_quality_model_can_be_swapped
test_quality_gate_rejects_blurry_frame
test_quality_gate_marks_low_light_frame
test_quality_score_controls_routing
test_gpu_metrics_safe_on_cpu_machine   (quality runs CPU-only; NVML absent → safe defaults)
```

The gate must run on a **CPU-only** box (OpenCV default) and emit safe defaults rather
than crash when GPU/NVML is unavailable ([06](06-hardware-portability.md)).

## Related

[16-capability-hooks-profiles-router.md](16-capability-hooks-profiles-router.md) ·
[17-pos-and-time-alignment.md](17-pos-and-time-alignment.md) ·
[05-performance-and-optimizations.md](05-performance-and-optimizations.md) ·
[08-dashboards.md](08-dashboards.md) · [10-platform-roadmap.md](10-platform-roadmap.md)
