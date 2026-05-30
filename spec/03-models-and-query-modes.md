# 03 — Models and Query Modes

Marlin answers two questions about a video: **what** happens (caption) and **when** does a named event happen (find). It is a 2B fine-tune of Qwen3.5-2B, trained on *atomic, temporally-grounded events* with explicit `<start-end>` boundaries (HC-STVG, VidSTG, TimeLens). That training shape dictates everything about how you should query it.

## Caption mode

`marlin.caption(video)` → `{caption, scene, events:[{start,end,description}]}`.

Returns the **full list** of time-ranged events automatically. Use it for "what happened overall" — it is the right tool for *multiple / recurring* events.

## Find mode — the query template matters

`marlin.find(video, event=Q)` wraps `Q` in a fixed prompt (`modeling_marlin.py`):

```
Identify the timestamps during which "{Q}" takes place.
Output the time range as "From <start> to <end>." (numbers in seconds).
```

Consequences that drive query design:

- `Q` is slotted into *"during which {Q} takes place"* → it must read as a **visible event that happens over a span**, present tense, not a question/label.
- **find returns exactly ONE span** (`parse_span` extracts a single `From X to Y`). It is *not* multi-instance and *not* a counter. Recurring events lose all but one occurrence.
- It grounds what is **seen** — there is no reasoning layer: no negation, no conjunction, no intent inference.

## The 5 rules for a good find query

1. **Atomic action** — one subject, one verb, one object. No compound clauses.
2. **Present-tense & visible** — fits "during which X takes place"; not a static state.
3. **No negation / conjunction** — "without", "while", "and then" are logic, not vision.
4. **No abstract judgement** — "suspicious" is not a visual primitive; ground the concrete actions that *make* it suspicious.
5. **No un-seeable text / proper nouns** — the model can't read that it's "Kathirmani".

## What does NOT belong in find (and where it goes instead)

- **Counting / "how many"** → the Locate stage (`inference/locate.py`).
- **Recurrence / "happened N times"** → caption's event list, or Locate.
- **Attributes / "is there… / what colour"** → Qwen VQA on routed keyframes.
- **Compound LP logic** ("left without paying", "accomplice while staff engaged") → the deterministic **rule engine**: combine atomic find spans + Locate routes downstream. Marlin emits the primitives; rules/Qwen do the reasoning.

## The redesigned query set (in `inference/pipeline.py:FIND_QUERIES`)

The original 20 mixed in negation ("leaving without approaching billing"), conjunction ("staff engaged while another handles merchandise"), recurrence ("returning to the same shelf multiple times"), abstraction ("suspicious behavior") and static states ("billing staff at counter") — all of which Marlin grounds poorly.

The new set is 20 **atomic, visible, present-tense** clauses grouped: entry/exit, billing, shopping floor, loss-prevention primitives, crowd/ops.

> **COUPLING WARNING:** query strings are **Prometheus metric label values** (`marlin_find_span_*{query="…"}`), and appear in Loki (`log_find_hit`), the `SECURITY_QUERIES` set (`inference/annotations.py`), and a regex in `grafana/dashboard_qwen.json`. Renaming a query orphans its old time series and blanks any panel that hardcodes it. When you change `FIND_QUERIES`, also update those three places. (Most dashboard panels aggregate by `{video="…"}` and are query-agnostic — only the Qwen "Loss Prevention" panel hardcodes query names.)

## Qwen2.5-VL — the "what" axis

Qwen2.5-VL provides the semantic ("what") reasoning layer. It runs **only** over keyframes the Locate stage flagged into `route_to_vlm` — an empty route means Qwen is skipped entirely for that camera (the main Qwen compute saving).

Its job is the loss-prevention / VQA read: it answers attribute and intent questions ("is someone concealing an item?", "what colour bag?") that Marlin's pure visual grounding cannot. A non-empty/affirmative Qwen answer to a loss-prevention question is a **security signal** — a flag to review, not a confirmed theft. Stock Qwen also runs in vLLM today, which makes it the first candidate for engine-level acceleration (see [05-performance-and-optimizations.md](05-performance-and-optimizations.md)).
