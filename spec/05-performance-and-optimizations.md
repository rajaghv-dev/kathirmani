# 05 — Performance and Optimizations

## Why it's slow even with 119 GB and 20 cores available

The bottleneck is **how the work is structured**, not capacity. Big memory and idle cores can't help a workload that funnels through one serialized GPU op at a time and re-decodes the same video over and over.

### The seven bottlenecks

1. **Single-stream serial GPU.** One model instance + `_GPU_LOCK` → throughput is bound by one forward pass at a time. Free RAM/cores don't add GPU parallelism.
2. **Redundant decode (biggest waste).** Every `caption()` and `find()` decodes the video from scratch via torchcodec. Per camera = 1 + 20 = **21 full decodes of the same file**; 5 cameras ≈ 105 decodes. Identical frames every time.
3. **No CPU↔GPU pipelining.** Decode (CPU) and inference (GPU) run sequentially inside the same locked call, so the GPU idles during decode and vice-versa.
4. **Tiny batches on a big chip.** batch_size=1 (one video, one query) leaves the Blackwell ALUs mostly idle — you're not feeding the hardware.
5. **Unified memory ≠ fast.** GB10 has no discrete VRAM; 119 GB is shared LPDDR5X (lower bandwidth than HBM). Capacity lets you *load* big models; it does not speed up compute/bandwidth-bound inference. "119 GB free" is irrelevant to latency.
6. **torch.compile cold start** (~3 min) dominated short runs — now cached (see [04-observability-stack.md](04-observability-stack.md)).
7. **20 sequential generate() calls per camera**, each re-encoding the video frames.

### Fix hierarchy (what actually moves the needle)

- **Decode once, reuse frames** across caption + all finds — eliminates the ~20× redundant decode. Biggest single win. **DONE** (see below).
- **Batch** queries/cameras into the GPU to fill the ALUs.
- **Cascade** so the expensive model runs rarely (already wired for Qwen).
- **Quantize** FP8/INT8 to cut per-call latency + memory-bandwidth pressure.
- Persistent compile cache (done).

**Headline:** latency-bound on a serialized single-stream pipeline with massive redundant decode — not capacity-bound.

---

## Decode-once refactor (implemented) + the gotcha that bit us

### Before → after

This was a real code change to `inference/pipeline.py`, not a config tweak.

**Before.** Each query re-read the whole video. `_run_caption` called `model.caption(path)` and each `_run_find` called `model.find(path)`. Both go through `modeling_marlin._generate_video`, which decodes the video *inside* `apply_chat_template(tokenize=True)`. So for one camera we asked the model 21 questions (1 caption + 20 finds) and **decoded the same MKV 21 times** — the exact same pixels, re-read and re-sampled from disk on every question.

```
Before, per camera:  21 questions → 21 full video decodes
Across 5 cameras:     105 questions → 105 video decodes
```

**After.** We split "decode the video" from "ask a question". Per camera we decode **once** (`_decode_video_once`) and feed those cached frames to all 21 questions (`_generate_cached`). The model still *answers* 21 times (GPU work is unchanged), but the video is only read off disk and sampled into frames once.

```
After, per camera:   21 questions → 1 video decode  (20 decodes avoided)
Across 5 cameras:     105 questions → 5 video decodes (100 decodes avoided)
```

**The saving, in numbers.** If decoding one camera's clip takes `D` seconds, the old way spent `21·D` on decode per camera; the new way spends `1·D`. So we save `20·D` per camera (= `N · D`, where N=20 find queries). That's a **~95% cut in video-decode work** (1 of 21 decodes remains, 20/21 ≈ 95% removed). On a 12 s Bill-Counter clip the live log prints e.g. `decoded 96 frames in 1.4s (reused for 21 prompts; saved ~28s decode)`. Decoding also overlaps across the 5 cameras because it runs **outside `_GPU_LOCK`** (CPU work) while only the GPU forward passes serialize.

**What this does *not* speed up.** The GPU still runs 21 forward passes per camera, serialized by the single-GPU `_GPU_LOCK`. Decode-once attacks the CPU-decode half of the cost, not the GPU half. Cutting GPU work needs the Tier-1/2/3 ideas below.

**Where to see it.** Live in the run log (per-camera decode line) and in Grafana → Compute Economy → "Compute saved — decode-once" row (`kathirmani_decode_seconds_saved`, `kathirmani_video_decodes_avoided`, reduction %).

### Mechanics

`_run_caption`/`_run_find` no longer call `model.caption(path)` / `model.find(path)` — those each re-decode the video. Instead, per camera we decode **once** and reuse the frames for all 21 prompts (`inference/pipeline.py`):

- `_decode_video_once(model, path)` → `process_vision_info(msgs, return_video_kwargs=True, return_video_metadata=True)`. Returns `(image_inputs, frames, metas, video_kwargs)`. Run **outside `_GPU_LOCK`** so the 5 cameras' decodes overlap on CPU while GPU forwards serialize.
- `_generate_cached(model, decoded, prompt, max_tokens)` rebuilds only the text turn (`apply_chat_template(tokenize=False)`) and calls `processor(text=..., videos=frames, video_metadata=metas, **video_kwargs)` → `model.generate`. Mirrors `modeling_marlin._generate_video` but feeds cached frames.
- Module helpers (`CAPTION_PROMPT`, `GROUNDING_PROMPT_TEMPLATE`, `parse_caption`, `parse_span`) are reached via `importlib.import_module(type(model).__module__)`.

> **THE GOTCHA — Marlin is Qwen3-VL based, not Qwen2.5-VL.** The processor is `Qwen3VLProcessor`, which builds the prompt from **frame timestamps**. If you pass pre-decoded frames *without* `video_metadata`, it warns "fps could not be inferred … Defaulting to fps=24" and the **find spans come out wrong** (caption still looks fine — it has no absolute timestamps to corrupt). Fix: use `return_video_metadata=True` (frames arrive as `(tensor, meta)` tuples; split them) and pass `video_metadata=metas`. Verified on a 12 s Bill-Counter clip: decode-once spans == path-based baseline `(10.0, 12.0)`; mismatch before the fix.

Expected win: 21 decodes/camera → 1. The GPU math is unchanged (still 21 forwards), so this attacks the CPU-decode half, not the serialized-GPU half.

### Progress + savings visibility

Because the single GPU serializes every forward, total work per run is a fixed count (`cameras × (1 + N)` GPU calls), so `_Progress` ticks once per completed call and every line reports `GPU x/y pct · elapsed · ETA` plus per-call seconds. Decode savings are measured live and exported to Prometheus: `kathirmani_decode_seconds_saved` / `kathirmani_decode_seconds_actual`, `kathirmani_video_decodes_avoided` / `kathirmani_video_decodes_done`. Saved decode-seconds per camera = (its decode time) × N (the reused find prompts). Surfaced on the **Compute Economy** Grafana dashboard.

---

## The optimization roadmap (ranked for *this* setup)

5 fixed cameras of the same scene, 21 Marlin calls/camera + Qwen cascade, single GB10, GPU serialized by `_GPU_LOCK`. Two cost centers, different fixes: redundant **decode** (CPU) vs serialized **generate()** (GPU). Fixed camera angle is the biggest untapped lever — it lets you do *less* work, which beats faster work.

### Tier 1 — do less work (highest ROI; exploits fixed CCTV)

1. **Decode-once** — DONE (above).
2. **Shared vision prefix / prefix-KV reuse — the big GPU win.** All 21 prompts = `[identical video tokens] + [short text]`. The vision encode + the LLM prefill over the video tokens can be computed **once** and reused as the KV prefix for all 21 queries; only the short text tail + decode differ. vLLM's automatic prefix caching does this for free across requests; PureKV reports ~3.16× prefill speedup on Qwen2.5-VL video. The GPU-side analogue of decode-once, close to ideal for our 21-share-one-video pattern.
3. **Motion gating front-gate (only valid because cameras are fixed).** OpenCV MOG2/KNN background subtraction is cheap; skip time windows with no foreground motion before they reach Marlin/Qwen, and feed Marlin only changing spans instead of uniform 2 fps. Pushes the Locate→Qwen cascade idea *upstream* of Marlin (which currently has no gate).
4. **ROI / static-region masking.** Fixed angle ⇒ known door/counter/shelf regions. Crop/mask to ROIs → fewer vision tokens → cheaper prefill.

### Tier 2 — same work, faster (engine-level)

5. **Serve through vLLM** instead of raw transformers + `_GPU_LOCK`: continuous batching (fills the ALUs, kills batch_size=1), PagedAttention + automatic prefix caching, `--gpu-memory-utilization=0.95`. Caveat: Marlin is a `trust_remote_code` Qwen3-VL fine-tune → may need a vLLM adapter; **stock Qwen (the "what" model) runs in vLLM today**, so move it first.
6. **FP8/INT8 quantization.** GB10 has no HBM — shared LPDDR5X ⇒ we're **bandwidth-bound**, not capacity-bound. Quant's win here is halving bytes/token moved, exactly the bottleneck. FP8 has Blackwell HW support. VidKV pushes video KV cache to ~1.58-bit with near-zero quality loss if going aggressive.
7. **CPU↔GPU pipelining.** Decode camera N+1 while N runs on GPU (decode-once already moved decode off the lock; this overlaps it fully). vLLM's scheduler does this for you.

### Tier 3 — smarter architecture

8. **Persistent scene priors** — the static scene caption barely changes between runs; cache it, spend budget on the varying find queries.
9. **Cross-camera frame batching** — same-timestamp frames from the 5 cameras in one forward pass (the `serve_stream.py` two-tier design; true GPU parallelism, not just overlapped IO).
10. **Per-camera query gating** — with motion/Locate signals, skip finds that can't apply on a given camera (e.g. "lingers near exit" on the bill counter). Fewer generate() calls beats faster ones.

**Recommended sequence:** decode-once (done) → motion-gate front → vLLM spike (Qwen first, then Marlin) → FP8 → ROI + query gating. On bandwidth-bound GB10, "do less work" (Tier 1/3) wins more than "use the hardware harder" (Tier 2).

**Refs:** vLLM prefix caching (docs.vllm.ai/en/stable/design/prefix_caching), Qwen2.5-VL vLLM recipe, PureKV (arxiv 2510.25600), VidKV (arxiv 2503.16257).

## Related

[02-architecture.md](02-architecture.md) · [04-observability-stack.md](04-observability-stack.md) · [06-hardware-portability.md](06-hardware-portability.md)
