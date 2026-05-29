# Learnings — Kathirmani inference pipeline

Running notes on how this stack actually behaves, why it was built this way, and
the non-obvious gotchas. Companion to `docs/ARCHITECTURE.md` (the dataflow) and
`docs/PLAN-locate-stage.md` (the roadmap).

---

## 1. Marlin-2B: how the two modes work, and how to query them

Marlin answers two questions about a video: **what** happens (caption) and
**when** does a named event happen (find). It is a 2B fine-tune of Qwen3.5-2B,
trained on *atomic, temporally-grounded events* with explicit `<start-end>`
boundaries (HC-STVG, VidSTG, TimeLens). That training shape dictates everything
about how you should query it.

### caption mode
`marlin.caption(video)` → `{caption, scene, events:[{start,end,description}]}`.
Returns the **full list** of time-ranged events automatically. Use it for "what
happened overall" — it is the right tool for *multiple / recurring* events.

### find mode — the query template matters
`marlin.find(video, event=Q)` wraps `Q` in a fixed prompt (`modeling_marlin.py`):

```
Identify the timestamps during which "{Q}" takes place.
Output the time range as "From <start> to <end>." (numbers in seconds).
```

Consequences that drive query design:
- `Q` is slotted into *"during which {Q} takes place"* → it must read as a
  **visible event that happens over a span**, present tense, not a question/label.
- **find returns exactly ONE span** (`parse_span` extracts a single `From X to Y`).
  It is *not* multi-instance and *not* a counter. Recurring events lose all but
  one occurrence.
- It grounds what is **seen** — there is no reasoning layer: no negation, no
  conjunction, no intent inference.

### The 5 rules for a good find query
1. **Atomic action** — one subject, one verb, one object. No compound clauses.
2. **Present-tense & visible** — fits "during which X takes place"; not a static state.
3. **No negation / conjunction** — "without", "while", "and then" are logic, not vision.
4. **No abstract judgement** — "suspicious" is not a visual primitive; ground the
   concrete actions that *make* it suspicious.
5. **No un-seeable text / proper nouns** — the model can't read that it's "Kathirmani".

### What does NOT belong in find (and where it goes instead)
- **Counting / "how many"** → the Locate stage (`inference/locate.py`).
- **Recurrence / "happened N times"** → caption's event list, or Locate.
- **Attributes / "is there… / what colour"** → Qwen VQA on routed keyframes.
- **Compound LP logic** ("left without paying", "accomplice while staff engaged")
  → the deterministic **rule engine**: combine atomic find spans + Locate routes
  downstream. Marlin emits the primitives; rules/Qwen do the reasoning.

### The redesigned query set (in `inference/pipeline.py:FIND_QUERIES`)
The original 20 mixed in negation ("leaving without approaching billing"),
conjunction ("staff engaged while another handles merchandise"), recurrence
("returning to the same shelf multiple times"), abstraction ("suspicious
behavior") and static states ("billing staff at counter") — all of which Marlin
grounds poorly. The new set is 20 **atomic, visible, present-tense** clauses
grouped: entry/exit, billing, shopping floor, loss-prevention primitives, crowd/ops.

> **COUPLING WARNING:** query strings are **Prometheus metric label values**
> (`marlin_find_span_*{query="…"}`), and appear in Loki (`log_find_hit`), the
> `SECURITY_QUERIES` set (`inference/annotations.py`), and a regex in
> `grafana/dashboard_qwen.json`. Renaming a query orphans its old time series and
> blanks any panel that hardcodes it. When you change `FIND_QUERIES`, also update
> those three places. (Most dashboard panels aggregate by `{video="…"}` and are
> query-agnostic — only the Qwen "Loss Prevention" panel hardcodes query names.)

---

## 2. Architecture & the parallelism reality

Three perception axes, cascade-gated: **Marlin = when**, **Locate = where** (+ cheap
front-gate emitting `route_to_vlm`), **Qwen = what** (only on routed keyframes).
Full dataflow in `docs/ARCHITECTURE.md`.

`run_all()` processes the 5 cameras through a `ThreadPoolExecutor` (`--max-workers`,
default one per video). **But there is one GPU and one shared model**, so
`_GPU_LOCK` serializes every forward pass. The parallelism overlaps **decode,
parsing, JSON IO, and the Loki/Prometheus/annotation inserts** — *not* the GPU
math. Expect a modest wall-time win, not 5×. True parallel GPU throughput needs
cross-camera frame batching or node sharding (the `serve_stream.py` two-tier
design, not yet built).

---

## 3. The "databases" in this stack

There is **no SQL/NoSQL DB**. The datastores are:
- **Prometheus** (TSDB) — fed by metric gauges set inline per camera + scraped
  from `serve_metrics.py` (reads `results/*.json`) and Netdata.
- **Loki** (logs) — fed directly by `inference/loki.py`.
- `results/*.json` is the on-disk source of truth.

Each parallel worker inserts to Loki + sets Prometheus gauges + pushes Grafana
annotations as it finishes — so "DB inserts" happen concurrently across cameras.
(The old sequential `run_all` had silently dropped all Loki logging; restored.)

---

## 4. Runtime metrics come from Netdata

CPU / memory / disk are read from the **Netdata REST API**
(`_netdata_dims()` in `inference/metrics.py`, `NETDATA_URL` default
`http://localhost:19999`, override to `http://netdata:19999` inside Docker).
Charts used: `system.cpu` (user/system %), `system.ram` (MiB → GB),
`disk_space./` (GiB used/avail), `system.io` (reads/writes KiB/s). Falls back to
`/proc` if Netdata is unreachable. GPU power still comes from `nvidia-smi`,
compute util from NVML.

---

## 5. Why the model "downloads repeatedly" (it doesn't) + caching

All `from_pretrained` calls use `local_files_only=True` and `HF_HUB_OFFLINE=1`,
and `models/Marlin-2B/` is complete — so **weights are never re-downloaded**.
What *did* regenerate every run:
- the `trust_remote_code` module cache (`transformers_modules`), and
- `torch.compile` artifacts (recompiled from scratch each process, ~3 min).

On an ephemeral `$HOME` these vanish between runs and look like "downloading
again". Fixed by pinning both into the repo (`run_inference.py`):
`HF_HOME=models/.cache/hf`, `TORCHINDUCTOR_CACHE_DIR=models/.cache/inductor`,
`TORCHINDUCTOR_FX_GRAPH_CACHE=1` (gitignored). First run still compiles once;
every run after reuses the cache.

---

## 6. Why it's slow even with 119 GB and 20 cores available

The bottleneck is **how the work is structured**, not capacity. Big memory and
idle cores can't help a workload that funnels through one serialized GPU op at a
time and re-decodes the same video over and over.

1. **Single-stream serial GPU.** One model instance + `_GPU_LOCK` → throughput is
   bound by one forward pass at a time. Free RAM/cores don't add GPU parallelism.
2. **Redundant decode (biggest waste).** Every `caption()` and `find()` decodes
   the video from scratch via torchcodec. Per camera = 1 + 20 = **21 full decodes
   of the same file**; 5 cameras ≈ 105 decodes. Identical frames every time.
3. **No CPU↔GPU pipelining.** Decode (CPU) and inference (GPU) run sequentially
   inside the same locked call, so the GPU idles during decode and vice-versa.
4. **Tiny batches on a big chip.** batch_size=1 (one video, one query) leaves the
   Blackwell ALUs mostly idle — you're not feeding the hardware.
5. **Unified memory ≠ fast.** GB10 has no discrete VRAM; 119 GB is shared LPDDR5X
   (lower bandwidth than HBM). Capacity lets you *load* big models; it does not
   speed up compute/bandwidth-bound inference. "119 GB free" is irrelevant to latency.
6. **torch.compile cold start** (~3 min) dominated short runs — now cached (§5).
7. **20 sequential generate() calls per camera**, each re-encoding the video frames.

**Fix hierarchy (what actually moves the needle):**
- **Decode once, reuse frames** across caption + all finds — eliminates the ~20×
  redundant decode. Biggest single win. **DONE** (see §7).
- **Batch** queries/cameras into the GPU to fill the ALUs.
- **Cascade** so the expensive model runs rarely (already wired for Qwen).
- **Quantize** FP8/INT8 to cut per-call latency + memory-bandwidth pressure.
- Persistent compile cache (done).

Headline: **latency-bound on a serialized single-stream pipeline with massive
redundant decode — not capacity-bound.**

---

## 7. Decode-once refactor (implemented) + the gotcha that bit us

`_run_caption`/`_run_find` no longer call `model.caption(path)` / `model.find(path)`
— those each re-decode the video. Instead, per camera we decode **once** and
reuse the frames for all 21 prompts (`inference/pipeline.py`):

- `_decode_video_once(model, path)` → `process_vision_info(msgs,
  return_video_kwargs=True, return_video_metadata=True)`. Returns
  `(image_inputs, frames, metas, video_kwargs)`. Run **outside `_GPU_LOCK`** so
  the 5 cameras' decodes overlap on CPU while GPU forwards serialize.
- `_generate_cached(model, decoded, prompt, max_tokens)` rebuilds only the text
  turn (`apply_chat_template(tokenize=False)`) and calls `processor(text=...,
  videos=frames, video_metadata=metas, **video_kwargs)` → `model.generate`.
  Mirrors `modeling_marlin._generate_video` but feeds cached frames.
- Module helpers (`CAPTION_PROMPT`, `GROUNDING_PROMPT_TEMPLATE`, `parse_caption`,
  `parse_span`) are reached via `importlib.import_module(type(model).__module__)`.

> **THE GOTCHA — Marlin is Qwen3-VL based, not Qwen2.5-VL.** The processor is
> `Qwen3VLProcessor`, which builds the prompt from **frame timestamps**. If you
> pass pre-decoded frames *without* `video_metadata`, it warns "fps could not be
> inferred … Defaulting to fps=24" and the **find spans come out wrong** (caption
> still looks fine — it has no absolute timestamps to corrupt). Fix: use
> `return_video_metadata=True` (frames arrive as `(tensor, meta)` tuples; split
> them) and pass `video_metadata=metas`. Verified on a 12 s Bill-Counter clip:
> decode-once spans == path-based baseline `(10.0, 12.0)`; mismatch before the fix.

Expected win: 21 decodes/camera → 1. The GPU math is unchanged (still 21
forwards), so this attacks the CPU-decode half of §6, not the serialized-GPU half.

**Progress + savings visibility.** Because the single GPU serializes every
forward, total work per run is a fixed count (`cameras × (1 + N)` GPU calls), so
`_Progress` ticks once per completed call and every line reports
`GPU x/y pct · elapsed · ETA` plus per-call seconds (no more silent parallel
runs). Decode savings are measured live and exported to Prometheus:
`marlin_decode_seconds_saved` / `marlin_decode_seconds_actual`,
`marlin_video_decodes_avoided` / `marlin_video_decodes_done`. Saved decode-seconds
per camera = (its decode time) × N (the reused find prompts). Surfaced on the
**Compute Economy** Grafana dashboard ("Compute saved — decode-once" row:
seconds saved, decodes avoided, reduction %).

---

## 8. The optimization roadmap (what's next, ranked for *this* setup)

5 fixed cameras of the same scene, 21 Marlin calls/camera + Qwen cascade, single
GB10, GPU serialized by `_GPU_LOCK`. Two cost centers, different fixes: redundant
**decode** (CPU) vs serialized **generate()** (GPU). Fixed camera angle is the
biggest untapped lever — it lets you do *less* work, which beats faster work.

### Tier 1 — do less work (highest ROI; exploits fixed CCTV)
1. **Decode-once** — DONE (§7).
2. **Shared vision prefix / prefix-KV reuse — the big GPU win.** All 21 prompts =
   `[identical video tokens] + [short text]`. The vision encode + the LLM prefill
   over the video tokens can be computed **once** and reused as the KV prefix for
   all 21 queries; only the short text tail + decode differ. vLLM's automatic
   prefix caching does this for free across requests; PureKV reports ~3.16×
   prefill speedup on Qwen2.5-VL video. This is the GPU-side analogue of
   decode-once and is close to ideal for our 21-share-one-video pattern.
3. **Motion gating front-gate (only valid because cameras are fixed).** OpenCV
   MOG2/KNN background subtraction is cheap; skip time windows with no foreground
   motion before they reach Marlin/Qwen, and feed Marlin only changing spans
   instead of uniform 2 fps. Pushes the same idea as the Locate→Qwen cascade
   *upstream* of Marlin (which currently has no gate).
4. **ROI / static-region masking.** Fixed angle ⇒ known door/counter/shelf
   regions. Crop/mask to ROIs → fewer vision tokens → cheaper prefill.

### Tier 2 — same work, faster (engine-level)
5. **Serve through vLLM** instead of raw transformers + `_GPU_LOCK`: continuous
   batching (fills the ALUs, kills batch_size=1), PagedAttention + automatic
   prefix caching, `--gpu-memory-utilization=0.95`. Caveat: Marlin is a
   `trust_remote_code` Qwen3-VL fine-tune → may need a vLLM adapter; **stock Qwen
   (the "what" model) runs in vLLM today**, so move it first.
6. **FP8/INT8 quantization.** GB10 has no HBM — shared LPDDR5X ⇒ we're
   **bandwidth-bound**, not capacity-bound. Quant's win here is halving bytes/token
   moved, exactly the bottleneck. FP8 has Blackwell HW support. VidKV pushes video
   KV cache to ~1.58-bit with near-zero quality loss if going aggressive.
7. **CPU↔GPU pipelining.** Decode camera N+1 while N runs on GPU (decode-once
   already moved decode off the lock; this overlaps it fully). vLLM's scheduler
   does this for you.

### Tier 3 — smarter architecture
8. **Persistent scene priors** — the static scene caption barely changes between
   runs; cache it, spend budget on the varying find queries.
9. **Cross-camera frame batching** — same-timestamp frames from the 5 cameras in
   one forward pass (the `serve_stream.py` two-tier design; true GPU parallelism,
   not just overlapped IO).
10. **Per-camera query gating** — with motion/Locate signals, skip finds that
    can't apply on a given camera (e.g. "lingers near exit" on the bill counter).
    Fewer generate() calls beats faster ones.

Recommended sequence: decode-once (done) → motion-gate front → vLLM spike (Qwen
first, then Marlin) → FP8 → ROI + query gating. On bandwidth-bound GB10, "do less
work" (Tier 1/3) wins more than "use the hardware harder" (Tier 2).

Refs: vLLM prefix caching (docs.vllm.ai/en/stable/design/prefix_caching),
Qwen2.5-VL vLLM recipe, PureKV (arxiv 2510.25600), VidKV (arxiv 2503.16257).

---

## 9. "LocateAnything" — it's a stage, not a model (and it's opt-in)

> **Naming trap:** "LocateAnything" is **our localization stage**
> (`inference/locate.py`), not an off-the-shelf model. The model actually doing
> the detection is **YOLOE** (the default backend).

- **Role** — the spatial ("where") axis Marlin (when) and Qwen (what) lack.
  Samples frames (every `SAMPLE_INTERVAL_SEC=2s`), runs an open-vocab detector
  over `LOCATE_VOCAB` (person, shopping basket, handbag, backpack, product on
  shelf, shopping cart, mobile phone), and writes `locate_analysis` into each
  `results/<camera>.json`: per-class `objects` counts, `max_people`, per-frame
  `frames`, and `route_to_vlm` timestamps.
- **It is the cascade front-gate.** Frames containing an `INTERACTION_CLASSES`
  item (handbag/backpack/shopping basket) are flagged into `route_to_vlm`; Qwen
  then reasons **only** over those frames. Empty route ⇒ Qwen skipped entirely
  for that camera (the main Qwen compute saving) — see `run_inference.py` Qwen block.
- **Opt-in.** A plain `python run_inference.py` runs **Marlin only** — no Locate,
  no Qwen cascade gating. Locate runs only with `--locate` (after Marlin, looping
  over results in `main()`), backend chosen by `--locate-backend` (default `yoloe`).
- **Backends — only YOLOE is runnable today.**
  - `yoloe` (default): weights present — `models/yoloe/yoloe-11s-seg.pt` +
    `mobileclip_blt.ts` (the text-prompt embedder). Open-vocab embeds `LOCATE_VOCAB`
    once and reuses across all frames/cameras, so adding prompts is near-free.
  - `grounding-dino`/`dino`: weights **absent** (`models/grounding-dino-base` missing).
  - `rf-detr`: experimental, auto-downloads; closed-vocab COCO.
  - `sam3`: stub — weights not published yet.
- **Decode note:** Locate decodes the video separately from the Marlin/Qwen
  decode (its own sampled-frame pass). Unifying that with the §7 decode-once is a
  listed follow-up (`docs/PLAN-locate-stage.md`).

---

## 10. Running & viewing (current runbook)

Reflects decode-once (§7), parallel cameras (`--max-workers`), the opt-in
`--locate` YOLOE gate (§9), and the Streamlit Optimizations tab (§8).

### Setup (per shell)
```bash
cd <repo>
source .venv/bin/activate
docker compose up -d      # Prometheus, Loki, Grafana(:3000), Netdata(:19999)
```

### Run inference
```bash
python run_inference.py                                   # all 5 cameras, Marlin only, parallel
python run_inference.py --locate                          # + YOLOE Locate gate → Qwen only on routed frames
python run_inference.py --video "Bill Counter" --duration 12 --no-compile --locate   # fast smoke test
python run_inference.py --max-workers 3                   # cap parallel camera workers (default: one/video)
```
Flags: `--locate` (opt-in, default backend `yoloe`), `--locate-backend`,
`--skip-qwen`, `--duration N`, `--no-compile` (skip ~3-min compile),
`--max-workers N`, `--keep-alive`. Single `--video` uses `process_video`
(sequential); the multi-camera path uses `run_all` (parallel). Locate + Qwen run
**after** Marlin, looping over results in `main()`.

### Outputs (`results/`)
- `<camera>.json` — caption, events, `find_results` (+ `locate_analysis` /
  `qwen_analysis` when those stages ran)
- `fused.json` — best-camera find spans + deduped store events
- `summary.json`, `economy.json`

### View — Grafana (`http://<host>:3000`, admin/admin; URLs printed at run end)
Model & Runtime · Pipeline · Camera Views · Store (Fused) · Visual Analysis
(Qwen) · Inference Logs (Loki) · Compute Economy.

### View — Streamlit (visualization ONLY; annotation is CVAT/Label Studio)
```bash
streamlit run viz_app.py        # http://localhost:8501
```
Tabs: Overview · Detections · Timeline · Qwen Q&A · Store (fused) · Backend
compare · **Optimizations** (decode-once win + Tier 1/2/3 roadmap; renders even
with no results on disk).
