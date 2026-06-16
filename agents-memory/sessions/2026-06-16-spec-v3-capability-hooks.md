# 2026-06-16 — Integrate master-prompt v3 into spec/ (capability hooks + quality gate + POS)

User: "update the spec dir in detail by integrating it all" (master prompt v3 pasted —
plug-and-play models, image-quality gate, POS, observability). **Doc-only**; no code.

## Decisions (asked the user)
1. **Model policy:** v3 **relaxes** the hard NVIDIA-only default (spec/11). Selection =
   measured cost/quality parity on retail KPIs; NVIDIA-only → one optional/recommended
   profile, not mandated. `validate-model-config` should relax from "fail non-NVIDIA
   default" → "require cost_profile+quality_profile+parity (or `parity: pending`)".
2. **Structure:** new numbered docs **15/16/17** (not folded into existing).
3. **Detail:** full forward-looking contracts (file layouts, metric names, schemas,
   routing-policy YAML, test matrix) — tagged not-yet-built, like spec/10/11.

## Done
- **spec/15-quality-gate.md** — QUALITY hook (ImageQualityModel). Lives in
  `ingestion/quality/` (repo maps v3's `src/retail_video_ai/quality/`). Scores, schema,
  gate decisions, router hints, `retail_video_ai_quality_*`/`ingest_*` metrics, OTel
  `retail.quality.*`, CPU-only test gate.
- **spec/16-capability-hooks-profiles-router.md** — WHEN/WHERE/WHAT/QUALITY as hooks
  (vs spec/11 task contracts — mapping table), registry/plugin manifest + interfaces,
  economy/balanced/benchmark profiles, router + routing-policy YAML, low-cost→frontier
  parity strategy, `model_performance.csv` schema, the **policy-reconciliation** section
  (the governing rule for the relaxation).
- **spec/17-pos-and-time-alignment.md** — Store-POS read-only adapter (modes incl.
  mock_json fallback), pos_context schema, wall-clock time_alignment (ties to spec/10
  tstzrange/GiST), full QUALITY→WHEN→WHERE→WHAT→POS→RULES→HUMAN hook map, review-UI
  fields + outcomes (incl. "poor quality, cannot decide"), POS metrics/dashboards.
- **spec/02** — added "axes are capability hooks" section + 7-stage hook map + links.
- **spec/11** — added top `⚠️ Superseded default (v3)` banner pointing to spec/16
  §Policy reconciliation; kept the NVIDIA-only doc as the trust argument.
- **spec/README.md** — index rows 15/16/17 + v3 note + the design statement.
- **agents-memory/memory.md** — roadmap header now cites spec/15–17 + a v3 bullet.

## Reconciled v3-vs-repo naming
- v3 `src/retail_video_ai/` → repo `ingestion/` (quality) + `model-plugins/` +
  `ai-workers/*/plugin.py` (models). Kept v3's `retail_video_ai_*` metric names as the
  stated contract but noted repo follows `ingest_*`/`model_*` additive convention.
- v3 "no RTSP / local-video-first" = scope emphasis; RTSP (GStreamer Phase 1.5) already
  exists — did NOT regress that in spec/10. Left as current reality.

## Follow-ups (not done — forward-looking)
- Build `ingestion/quality/`, `services/pos-adapter/`, the registry/router/evaluator,
  economy/balanced/benchmark profiles in `configs/models.yaml`, and relax
  `scripts/validate_model_config.py` per the new contract. New dashboards
  (Image Quality / Model Routing / Cost-vs-Quality / POS Correlation).
- Status: **spec integrated; implementation pending.** No tests run (doc-only).
