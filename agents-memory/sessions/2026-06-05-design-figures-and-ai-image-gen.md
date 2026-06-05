# 2026-06-05 — ingestion-reasoning figures + AI image-gen integration

Context-builder session (no pipeline code changed). All on `main`.

## Figures (design/make_figures.py → design/figures/)
- Added `fig_ingestion` → `data_ingestion_layer.png` — RTSP→model-feed stages
  (capture/segment/validate/catalog/queue/feed).
- Added 3 "why pg-cataloged ingestion" figures, grounded in `db/schema.sql`:
  - `ingest_multiscale_time.png` — variable-extent subtle actions (palming 3s /
    tag-swap 12s / sweethearting 35s) over 10s segments + 5s windows + cross-camera
    track link; "ingest once, reason at any scale".
  - `ingest_coverage_quadrant.png` — duration × connectedness; naive fixed-window box
    vs the valuable cases outside it.
  - `ingest_reasoning.png` — 5-band Reality→fail→required→pg-delivers→payoff.
- Style: reuse NV green `#76B900` + FancyBboxPatch; all wired into the render loop.

## AI image generation (design-time only — NOT inference)
- `design/ai_images.py`: provider abstraction for Google **Nano Banana**
  (`gemini-2.5-flash-image`, SDK `google-genai`) + OpenAI **`gpt-image-1`** (SDK
  `openai`). Text→image and ref-image editing/fusion (`--ref`). CLI + `make gen-image`.
- `requirements/design.txt` (google-genai/openai/pillow); `make setup-design`/`figures`.
- Keys via env ONLY (`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `OPENAI_API_KEY`) → `.env.example`;
  `design/figures/generated/` gitignored (keep `.gitkeep`). Verified: py_compile + clean
  key-missing error. Policy boundary noted in spec/12 (NVIDIA-only governs inference,
  not design assets).

## Real follow-up surfaced (spec/10 Temporal correlation)
- `ai_windows` hold *relative* sec-offsets; indexes are plain btree. For true
  overlap joins on long/cross-clip actions → add an absolute **`tstzrange` column +
  `btree_gist`** overlap index on segments/events/incidents. Not yet built.

## Specs touched
- `spec/10` += "Temporal correlation — why pg-cataloged spans" + the tstzrange gap.
- `spec/12` += "Design-asset image generation (design-time only)".
