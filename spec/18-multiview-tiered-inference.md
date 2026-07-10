# 18 — Multi-View & Tiered Inference (same room, 5 angles, one GPU)

> **Design (2026-07-09), grounded in measured runs on the GB10 against the real
> `store-videos/` footage.** Companion to [16](16-capability-hooks-profiles-router.md)
> (hooks/profiles/router — the frame gate in §Frame gating is this doc's Tier 0/1,
> already implemented) and [11](11-model-plugin-policy.md) (model policy + licensing).

## The structural insight

All 5 cameras observe the **same store area from different sides**
([01](01-overview.md)). The pipeline previously treated them as 5 independent
rooms: every camera independently detects, tracks, hypothesizes, and (potentially)
VLM-verifies the same physical moment, and de-duplication happened only post-hoc
(legacy `fused.json`) or never (platform). Measured on real footage:

- Legacy fusion collapsed **141 raw per-camera events → 91 unique** (~35% of all
  "events" were another camera seeing the same moment).
- Ungated detection cost ~**1.3× the footage's own runtime** (5.1 GPU-min for
  4 camera-minutes) — compute exceeded the video.
- Nemotron-VL verification costs **~44 s per clip**; verifying every raw CV
  hypothesis would run ~30× real time. The rule engine fired **0** actions on the
  same input that raised 176 `needs_vlm` hypotheses — raw CV hypotheses are noise
  relative to rules.
- Every window of a segment re-ran YOLOE over the same 10-s clip (4× duplicate
  GPU work, byte-identical outputs).

The fix is to move "same room" knowledge **upstream** — from post-hoc fusion to
pre-compute routing — and to make every model tier ≥10× cheaper than the tier
above it, with each tier keeping ≥80–90% of its input away from the next.

## The tier hierarchy

| Tier | What | Cost | Job |
|---|---|---|---|
| **0** | Motion gate (numpy on 160×90 thumbnails) | ~free, CPU | Idle footage never enters the queue. **Implemented** in the detection plugin (spec/16 §Frame gating: 1 frame/sec sampling + skip when <15% of pixels changed) |
| **0.5** | View election (digital twin zone↔camera map) | free | Only each zone's **primary camera** raises hypotheses for that zone; other views contribute evidence clips only. Center/overview cam = tracker anchor + camera health |
| **1** | Small detector (TAO detector / YOLOE-class), sampled + gated frames | ~ms/frame | Tracks, zone hits, object presence. Detect once per **segment**, never per overlapping window |
| **2** | Deterministic logic: tracker → rules → **cross-view fusion in SQL** | free, CPU | Fold per-camera events into ONE fused incident per physical moment via time-overlap × zone-adjacency — the `tstzrange` + GiST overlap indexes (migration 0004) exist for exactly this join |
| **3** | Small-VLM triage: **Cosmos-Reason2-2B**, strict-JSON `reject/confirm/escalate` @ 2 fps, 448 px | ~10–15 s/clip | Kill 70–90% of hypotheses that survive rules. (Research profile may use Marlin-2B here — comparison-only per spec/11) |
| **4** | Full VLM verdict: **Nemotron-VL-8B**, once per FUSED incident, token-bucket budget (e.g. 30/h ≈ 22 GPU-min/h) | ~44 s/clip | The expensive opinion, only on rule-fired or Tier-3-escalated incidents. Batch queued clips; quantized (GGUF/FP8) build ≈ 2× faster |
| **5** | C-RADIO embeddings (emitted incidents only, near-dup collapse across views) + human review | cheap | One multi-angle incident for the human, not five duplicates |

**Budget outcome (per hour of 5-camera live footage, one GB10):** naive ≈ 6–30×
real time → hierarchy ≈ Tier 1 at 15–20% GPU + Tiers 3/4 under ~27 min/h combined
= comfortably **< 1 GPU-hour per hour**, i.e. live-capable.

## Same-room optimizations (by measured ROI)

1. **Detect once per segment, not per window** (~75% of detection GPU — measured
   4× byte-identical YOLOE runs per segment).
2. **Frame sampling + motion gate** (implemented — spec/16): measured 250 frames
   → 10 sampled → 1 detected on a quiet aisle clip; 3.2 s → 1.13 s per clip.
3. **Don't persist per-frame detections** (84,576 rows from 4 camera-minutes ⇒
   ~30M rows/day): persist track summaries + ~1 box/sec samples; full detail only
   inside evidence windows.
4. **Primary-view election per zone** (kills N-view duplicate events at the source;
   evidence packages gain multi-angle clips for free).
5. **Verify once per fused incident, never per camera view** (44 s × N views for
   one question, otherwise).
6. **Gate the VLM on rules, not raw CV hypotheses** (176 hypotheses vs 0 rule
   firings on the same footage).
7. **Deterministic cross-camera track handoff first, re-id model later**: zone-time
   co-occurrence across views (±2 s in overlapping zones) before any ReID network —
   consistent with "safety logic stays deterministic" ([10](10-platform-roadmap.md)
   anti-goals). TAO ReIdentificationNet upgrades this when the NGC key lands.
8. **Near-dup collapse in the search index**: C-RADIO embeddings of the same moment
   from different views cluster; fold under one incident id at index time.
9. **Legacy cascade: room-level queries run once, not per camera** — most of the
   20 "find" queries are camera-agnostic; run them on the center/fused view only,
   side-specific queries on their primary cams (≈ 80 → 35 generates).
10. **Queue topology split** (prerequisite): `event.created` → rules/fusion →
    `event.needs_vlm` → VLM. Measured failure: rules and the VLM worker both
    consume `event.created`; rules drained all 176 messages and the VLM got 0.

## Measured: the full-footage run (2026-07-09, GB10)

The first end-to-end run over ALL footage — 4 cameras × 60 min = **4 hours of
2688×1520@25fps video** — with the implemented tiers (frame gate, segment
cache, stream split, verdict cache):

| Stage | Wall | vs footage runtime | Detail |
|---|---|---|---|
| Ingest (4 cams parallel) | 16.2 min | **14.8× faster** | 580 clips (30 s/5 s overlap) → 7,484 windows |
| Backfill → Postgres | 3 s | — | 580 segments + 7,484 windows |
| CV detection | 13.3 min | **18× faster** | 92% of windows (6,908/7,484) served from the segment cache; 17,305 frames sampled @1 fps; 15,916 motion-skipped; **1,389 of ~360,000 frames reached YOLOE (0.39%)**; 12,125 detections |
| Rule engine | <1 s | — | 18 hypotheses in → **8 unique verifications** forwarded on `event.needs_vlm` (subject/clip dedup) |
| VLM (Nemotron-8B) | 4.6 min | — | 8 verifications (incl. ≥1 verdict-cache hit on a same-clip duplicate). Verdicts: 8× `unclear` @ avg 0.40 confidence — the model's honest read of ambiguous footage. (Its JSON systematically closes arrays with `)` instead of `]`; the parser's mismatched-closer repair — added 2026-07-10 — now recovers 8/8, where all 8 previously failed parsing.) |
| Embedding index (C-RADIO) | 6 s | — | 48 events + 18 observations indexed |

**Against the measured naive baselines:**
- Events: **18 from the full hour** vs 17,632 duplicates from 60 *seconds* of
  the same footage pre-fix (~10⁵× noise reduction at equal footage).
- Detection GPU: pre-gate detection measured 1.27× real time (≈5 GPU-hours for
  this footage; ~14× worse again if each 30-s clip were re-detected per
  window) → actual 13.3 min = **~20–90× less GPU**.
- VLM: raw per-window hypothesizing would have queued hundreds of ~44-s
  verifications; dedup + gating reduced it to **8**.

Net: the full pipeline processes an hour of 5-camera-class footage in well
under real time on the single GB10 — the spec's "< 1 GPU-hour per hour"
budget is met with a wide margin.

## Operating modes → profiles (spec/16)

- **Hot path (live)** = Tiers 0–4 with budget caps: bounded latency, real-time
  guaranteed. Overflow defers to cold, never stalls.
- **Cold path (batch/overnight)** = same recorded hour replayed with caps lifted:
  deeper Tier-4 coverage, benchmark + research (Marlin/Qwen) arms. Clips are
  durable and cataloged, so replay is a `job_queue` re-enqueue by span, not a
  re-ingest.
- Profile mapping: **economy** = Tiers 0–3 (Tier 4 off, flag-for-human) ·
  **balanced** = full hierarchy · **benchmark** = cold path, caps lifted.

## Status

| Piece | State |
|---|---|
| Tier 0/1 frame gate (1 fps + <15% motion skip) | ✅ implemented + measured (spec/16) |
| Per-segment (not per-window) detection | ✅ 2026-07-09 — clip-level LRU result cache in the detection plugin; windows slice the cached per-frame results to their span; tracker runs once per clip; `model_run.frame_gate.clip_cache_hit` + worker AGGREGATE line report the savings |
| Queue stream split (`event.needs_vlm`) | ✅ 2026-07-09 — rule engine forwards flagged hypotheses to `event.needs_vlm` (deduped per camera+subject+type+clip); VLM worker consumes that stream, ending the rules-vs-VLM race on `event.created` |
| Verification result cache ("pipeline KV-cache") | ✅ 2026-07-09 — VLM worker reuses a clip+event-type verdict instead of re-watching (~44 s GPU per hit). True prompt-prefix KV reuse is blocked by Nemotron's `model.chat` API — the cache layer is the pipeline-level equivalent |
| Strict rules-fired-only VLM gating | pending (currently: needs_vlm + subject dedup) |
| View election / cross-view fusion (SQL overlap) | pending (indexes exist, unused) |
| Tier-3 Cosmos triage plugin | pending |
| Detection-row thinning | partial (per-window slicing ended whole-clip row duplication; sampled-frame rows remain) |

## Related

[16-capability-hooks-profiles-router.md](16-capability-hooks-profiles-router.md) ·
[11-model-plugin-policy.md](11-model-plugin-policy.md) ·
[10-platform-roadmap.md](10-platform-roadmap.md) ·
[17-pos-and-time-alignment.md](17-pos-and-time-alignment.md)
