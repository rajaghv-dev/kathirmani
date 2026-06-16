# 17 — POS Correlation, Time Alignment & the Full Hook Map

> **Forward-looking.** Per the master prompt
> [`oss_ingestion_nvidia_model_plugin_master_plan_v2.md`](../oss_ingestion_nvidia_model_plugin_master_plan_v2.md)
> v3. No POS adapter exists yet — this is the **target contract**. Completes the
> capability-hook picture: QUALITY ([15](15-quality-gate.md)) and the WHEN/WHERE/WHAT
> hooks + router ([16](16-capability-hooks-profiles-router.md)) feed into POS + rules +
> human review here.

## Why POS, not vision, for the transaction question

The product goal is **not** "AI catches shoplifters." It is: *video AI detects visible
candidate events · image-quality gates improve model reliability · POS checks
transaction context · rules prioritize review · humans decide · observability proves
cost/speed/quality.* (See the design statement in [README.md](README.md).)

CCTV SKU recognition is unreliable and a false-positive factory. So **never ask vision
to identify the product or whether it was paid for.** Use POS as ground truth for:
paid-transaction-exists-or-not, time of billing, cashier/till, total amount, product
lines, transaction status. Vision says *what visibly happened and when*; POS says
*whether a matching transaction exists*; rules combine them; humans decide.

```text
video action timeline  +  POS transaction timeline  +  camera-zone metadata  +  rules
= candidate review item
```

A candidate event with **no matching paid transaction** in its window is what gets
prioritized for review — and one that **is** POS-matched is cleared. This is how a
cheap vision stack stays useful: POS removes the interpretations vision can't settle.

## Store-POS adapter

Integrate the open-source **Store-POS** as a temporary, **read-only** reference POS
(the production POS integration is future — [01](01-overview.md) scope; no Zoho /
ManageEngine / cloud dependency in this version). It is a **capability hook** like the
models: multiple POS adapters behind one contract, selectable by config.

```yaml
pos:
  enabled: true
  adapter: store_pos
  mode: read_only_http        # read_only_http | readonly_db_export | mock_json
  base_url: http://localhost:3000
  timezone: Asia/Kolkata
  transaction_window_before_sec: 180
  transaction_window_after_sec: 240
```

- **`mock_json`** is the always-available fallback (a JSON fixture of transactions) so
  the whole pipeline is end-to-end testable with **no live POS** — the same
  fake-infer-fallback discipline the plugins use ([11](11-model-plugin-policy.md) §test
  gate). `make`-level runs default to mock unless a real Store-POS is reachable.
- **`read_only_http`** hits a running Store-POS; **`readonly_db_export`** reads a DB
  export. Both are strictly read-only — the platform never writes to POS.

Suggested location: `services/pos-adapter/` (a service alongside `rule-engine` /
`evidence-builder`, [09](09-repo-structure.md)), or an `ai-workers` consumer if it runs
off the queue.

## POS context on every candidate

Every candidate event carries its POS lookup result (included in the evidence bundle,
spec/10 Phase 7):

```json
{
  "pos_context": {
    "adapter": "store_pos",
    "lookup_window": { "start": "2026-06-15T10:31:12+05:30", "end": "2026-06-15T10:38:34+05:30" },
    "paid_transaction_found": false,
    "transactions_checked": [],
    "pos_match_status": "no_match"
  }
}
```

`pos_match_status` ∈ `{match, no_match, ambiguous, pos_unavailable}`. A `no_match`
**raises** review priority; a `match` can **clear** the event (a review outcome —
§review below). `pos_unavailable` must degrade safely (event still reviewable, flagged
"POS not checked"), never block the pipeline.

## Time alignment — the glue between video and POS

POS correlation only works if every video span maps to **wall-clock time**. Config
gives each camera's recording start; spans convert via offset + drift:

```yaml
time_alignment:
  store_timezone: Asia/Kolkata
  video_start_wall_clock:
    bill_counter: "2026-06-15T10:00:00+05:30"
    center:       "2026-06-15T10:00:00+05:30"
    left_aisle:   "2026-06-15T10:00:00+05:30"
    entry:        "2026-06-15T10:00:00+05:30"
    right_aisle:  "2026-06-15T10:00:00+05:30"
  default_clock_drift_sec: 0
```

Every event output gets wall-clock bounds:

```json
{ "camera_id": "center", "start_sec": 132.5, "end_sec": 136.0,
  "wall_clock_start": "2026-06-15T10:02:12.500+05:30",
  "wall_clock_end":   "2026-06-15T10:02:16.000+05:30" }
```

This complements the platform's **pg-cataloged time spans** — `video_segments` /
`ai_windows` / `events` carry absolute `tstzrange` spans with GiST overlap indexes
([10](10-platform-roadmap.md) §temporal correlation), so "the POS transaction at
10:34" and "the events overlapping 10:31–10:38" are an index-served join. Wall-clock
alignment here is what makes the cross-clip / cross-camera / video-vs-POS correlation
possible. Config likely belongs in `configs/cameras.yaml` (per-camera) +
`configs/stores/` (store timezone).

## The full hook map

QUALITY and POS bookend the three perception axes; rules and humans close it:

```text
QUALITY → Is this visual input good enough?        (spec/15)
WHEN    → When did a visible event happen?         (Marlin, spec/03)
WHERE   → Where is the relevant person/object/zone? (Locate/YOLOE, route_to_vlm)
WHAT    → What does the keyframe/clip appear to show? (Qwen, routed-only)
POS     → Was there a matching transaction?         (this doc)
RULES   → Should this become a review item?         (rule-engine, spec/10 Phase 5)
HUMAN   → What is the final decision?               (review-ui, spec/10 Phase 7)
```

Rules **never confirm theft** — they raise *candidate review items* by combining atomic
find spans + Locate routes + POS status ([03](03-models-and-query-modes.md): "Compound
LP logic → the deterministic rule engine"). Vision and POS emit primitives; rules
prioritize; humans decide.

## POS in observability

Metrics (additive; master prompt `retail_video_ai_pos_*`):

```text
retail_video_ai_pos_transactions_checked_total
retail_video_ai_pos_matches_total · _pos_misses_total
```

OTel spans `retail.pos.healthcheck`, `retail.pos.transaction_lookup` join the tree
([13](13-models.md)); attributes `pos_adapter`, `pos_match_status` (never raw item
names/amounts — [03](03-models-and-query-modes.md) cardinality/PII warning). New
**POS Correlation** dashboard ([08](08-dashboards.md) extension): matches, misses,
events cleared by POS, events escalated due to no match.

## Review UI & outcomes

The review surface ([10](10-platform-roadmap.md) Phase 7, [14](14-console-and-deployment.md))
must show, per candidate: clip, best frame, camera, zone, **quality score + why the
frame was accepted**, the WHEN/temporal result, the WHERE result, the `route_to_vlm`
decision, the Qwen answer **or** its skip reason, the selected model profile + cost
summary, the **POS lookup window + match/no-match**, and the rule that fired.

Review buttons (saved to `results/reviews/review_decisions.jsonl`):

```text
Confirmed issue · False positive · POS matched, clear event ·
Needs more review · Ignore · Poor video quality, cannot decide
```

The last button closes the loop with the QUALITY hook — a frame too poor to judge is a
first-class outcome, not a forced verdict.

## Test gate (target)

```text
test_pos_match_reduces_risk          test_pos_miss_increases_risk
test_rules_never_confirm_theft
test_review_ui_saves_poor_quality_decision
```

## Related

[15-quality-gate.md](15-quality-gate.md) ·
[16-capability-hooks-profiles-router.md](16-capability-hooks-profiles-router.md) ·
[03-models-and-query-modes.md](03-models-and-query-modes.md) ·
[10-platform-roadmap.md](10-platform-roadmap.md) ·
[14-console-and-deployment.md](14-console-and-deployment.md)
