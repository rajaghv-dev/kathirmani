# 16 — Capability Hooks, Model Profiles, Router & Small-Model Parity

> **Forward-looking.** Per the master prompt
> [`oss_ingestion_nvidia_model_plugin_master_plan_v2.md`](../oss_ingestion_nvidia_model_plugin_master_plan_v2.md)
> v3. Generalizes the existing `ModelPlugin` task contracts
> ([11-model-plugin-policy.md](11-model-plugin-policy.md)) into **plug-and-play
> capability hooks** and adds cost-tiered profiles, a model router, and a small-model
> parity discipline. Pairs with the QUALITY hook
> ([15-quality-gate.md](15-quality-gate.md)) and POS correlation
> ([17-pos-and-time-alignment.md](17-pos-and-time-alignment.md)).

## The architecture shift — capabilities, not model names

Do **not** hardcode the pipeline as `WHEN → Marlin`, `WHERE → YOLOE`, `WHAT → Qwen`.
Implement each axis as a **capability hook** the pipeline calls by capability; the
concrete model is config:

```text
QUALITY → image_quality_model    plugin   (default: opencv_quality_v1)   ← spec/15
WHEN    → temporal_event_model   plugin   (default: marlin_2b)
WHERE   → spatial_grounding_model plugin  (default: locate / yoloe)
WHAT    → semantic_vlm_model     plugin   (default: qwen2_5_vl, routed-only)
```

Pipeline code resolves a capability, never a name:

```python
# Bad — model name welded into the pipeline
result = marlin.find(video, query)

# Good — capability resolved from the active profile
temporal_model = registry.get_capability("temporal_find", profile="default")
result = temporal_model.find(video, query)
```

This is the same "stable contracts ⇒ no refactor" thesis already in spec/11 (`model
family ≠ runtime ≠ business rule ≠ prompt ≠ output schema`), restated as a **four-hook
capability map** so a model can be replaced or A/B-compared without pipeline changes.
The system must support: multiple WHEN/WHERE/WHAT/QUALITY models, multiple POS adapters
([17](17-pos-and-time-alignment.md)), multiple rule profiles & routing policies, model
shadow mode, A/B evaluation, fallback, and cost/quality reporting.

### How this relates to the existing plugin contracts (spec/11)

spec/11 defines task contracts (`DetectionPlugin`, `VLMClipReasoningPlugin`, …). The
capability hooks here are the **pipeline-facing view** of those same plugins:

| Capability hook (this doc) | Backing task contract (spec/11) |
|----------------------------|---------------------------------|
| `temporal_event_model` (WHEN) | (new) temporal-grounding contract; Marlin baseline |
| `spatial_grounding_model` (WHERE) | `DetectionPlugin` (+ `TrackingPlugin`) + `route_to_vlm` |
| `semantic_vlm_model` (WHAT) | `VLMClipReasoningPlugin` |
| `image_quality_model` (QUALITY) | (new) IQA contract, spec/15 |

One plugin host, one registry — the hooks are how the **pipeline** asks for work; the
task contracts are how a **plugin** declares what it can do.

## Plugin registry & layout

```text
src/retail_video_ai/models/      # repo mapping: model-plugins/ + ai-workers/*/plugin.py
  base.py registry.py profiles.py router.py evaluator.py cost_model.py
  plugins/{marlin_plugin,yoloe_plugin,qwen_plugin,mock_plugin}.py
```

Repo reality: the plugin ABC + schemas already live in `model-plugins/base/`, and each
worker carries its `plugin.py` ([09](09-repo-structure.md)). The registry/router/
evaluator/cost-model are the **new** pieces this doc specifies.

Every plugin declares a manifest:

```yaml
model_id: "marlin_2b"
display_name: "Marlin-2B"
provider: "local"
capabilities: [temporal_caption, temporal_find]
runtime: { engine: "pytorch", devices: [cuda, mps, cpu] }
cost_profile: { relative_cost: "low", expected_vram_gb: 6, expected_latency: "medium" }
quality_profile:
  good_for: [atomic temporal events, visible action spans]
  weak_for: [SKU recognition, intent inference, negation, counting]
```

Common + capability interfaces:

```python
class ModelPlugin:                       # load/healthcheck/run/unload
    model_id: str; capabilities: list[str]

class TemporalEventModel(ModelPlugin):   # WHEN
    def caption(self, video_or_chunk) -> CaptionResult: ...
    def find(self, video_or_chunk, query: AtomicQuery) -> FindResult: ...

class SpatialGroundingModel(ModelPlugin):# WHERE
    def detect(self, frame_or_chunk) -> SpatialResult: ...
    def route_to_vlm(self, spatial, quality, policy) -> RouteDecision: ...

class SemanticVLMModel(ModelPlugin):     # WHAT (routed-only)
    def analyze(self, frame_or_clip, prompt, context) -> SemanticResult: ...

class ImageQualityModel(ModelPlugin):    # QUALITY — spec/15
    def score(self, frame_or_chunk) -> QualityResult: ...
```

## Model profiles (`configs/models.yaml`)

Three cost-tiered profiles ship alongside the existing NVIDIA profiles:

```yaml
profiles:
  economy:      # lowest cost, routine runs — no VLM by default
    quality_model: opencv_quality_v1
    temporal_event_model: marlin_2b
    spatial_grounding_model: yoloe_small
    semantic_vlm_model: null
    escalation_model: qwen2_5_vl
    routing_policy: quality_and_uncertainty
  balanced:     # default — selective VLM routing
    quality_model: opencv_quality_v1
    temporal_event_model: marlin_2b
    spatial_grounding_model: yoloe_base
    semantic_vlm_model: qwen2_5_vl
    routing_policy: route_to_vlm_only
  benchmark:    # compare low-cost vs stronger/reference model
    quality_model: opencv_quality_v1
    temporal_event_model: [marlin_2b, candidate_temporal_model]
    spatial_grounding_model: [yoloe_small, candidate_detector]
    semantic_vlm_model: [qwen2_5_vl, frontier_reference_optional]
    routing_policy: shadow_eval
```

Every run records: `model_profile`, `model_id`, `capability`, `device`, `precision`,
`runtime_engine`, `cost_profile`, `quality_profile` — written to `model_runs`
([11](11-model-plugin-policy.md) §audit / [13](13-models.md) §per-inference tracking).

### Policy reconciliation — v3 relaxes NVIDIA-only-default

spec/11 made **NVIDIA-only the hard production default** (Qwen/YOLOE/Marlin =
comparison-only `research_qwen_baseline`). **v3 supersedes that hard rule:** the
selection criterion becomes **measured cost/quality parity on the retail KPIs**
(§"Small-model parity" below), not vendor. Consequences:

- **NVIDIA-only is demoted to one optional profile** (e.g. `nvidia_gb10_retail_balanced`
  stays available + valid), not the mandated default. The supply-chain/trust *benefits*
  of a narrow vendor scope (provenance pinning, license/safety review —
  [10](10-platform-roadmap.md) §Security) remain a real reason to *prefer* it, now
  weighed against cost/quality rather than enforced.
- `economy`/`balanced` may run **non-NVIDIA** models (Marlin/YOLOE/Qwen) as first-class
  defaults — the `vendor_exception` machinery in `configs/models.yaml` is no longer the
  only sanctioned path off NVIDIA.
- `make validate-model-config` should be relaxed from "fail any non-NVIDIA default" to
  "require a declared `cost_profile` + `quality_profile` + a parity result (or explicit
  `parity: pending`) for every default model". Vendor scope becomes an *advisory* check,
  not a hard gate. (This changes the spec/11 A5.2 contract — update spec/11 when the
  validator changes.) **`parity: pending` is only temporary:** the validator **warns** on
  a `pending` older than a set window (N runs / days), and a model stuck on `pending`
  must be marked `non_default`/`experimental` — it cannot silently serve as a production
  default forever (otherwise the parity gate is toothless).
- Frontier / closed models stay **teacher/auditor-only** (§teacher below) — relaxing
  the *vendor* rule does not make a paid frontier model a production default.

> This is a deliberate product decision recorded here (chosen 2026-06-16). spec/11
> keeps the full NVIDIA-only rationale as the trust argument; this section is the
> governing rule when the two differ.

## The model router (`router.py`)

Decides which model runs at each stage. Inputs: `camera_id`, `zone`,
`chunk_quality_score`, `motion_score`, `person_visibility_score`, `query_category`,
`risk_context`, `pos_context_available`, `previous_model_confidence`, `route_to_vlm`,
`budget_profile`. Emits an auditable decision:

```json
{ "stage": "semantic_vlm", "decision": "skip", "selected_model": null,
  "reason": "route_to_vlm=false and quality sufficient for temporal-only rule",
  "estimated_cost_saved": 0.83 }
```

Escalation policy (YAML, per profile):

```yaml
routing_policies:
  quality_and_uncertainty:
    run_quality_gate_first: true
    temporal:  { default_model: marlin_2b, fallback_model: candidate_temporal_model, skip_if_quality_below: 0.25 }
    spatial:   { default_model: yoloe_small, escalate_if: ["person_visibility_score < 0.45", "object_confidence < 0.35"] }
    semantic:
      default_model: qwen2_5_vl
      run_only_if: ["route_to_vlm == true", "candidate_risk != low"]
      skip_if:     ["route_to_vlm == false", "usable_frame_score < 0.30"]
    frontier_reference:
      enabled: false
      run_only_for: [benchmark, disagreement, active_learning_sample]
```

The router is where QUALITY ([15](15-quality-gate.md)) and `route_to_vlm`
([02](02-architecture.md), [03](03-models-and-query-modes.md)) converge into a single
"how hard to think" decision — preserving the cascade-gating saving (empty route ⇒ Qwen
skipped) and adding a quality floor below which nothing infers strongly.

## How a low-cost model approaches frontier usefulness

Don't expect a small model to beat a frontier model in general — make it win on *this
narrow retail task* through system design:

1. **Narrow the problem** — atomic visible actions only ([03](03-models-and-query-modes.md)
   query rules + the atomic query set), never "is this shoplifting?".
2. **Improve input** — skip duplicates, reject blur, pick sharper frames, crop ROI,
   choose the best camera angle, use zone metadata, decode-once
   ([05](05-performance-and-optimizations.md)), quality-aware routing
   ([15](15-quality-gate.md)).
3. **Cascade routing** — cheap quality → cheap temporal → cheap detector → rules/POS →
   stronger VLM only when needed.
4. **POS as ground truth** — never ask vision for the SKU; use POS for
   paid/not-paid, time, till, amount, lines, status ([17](17-pos-and-time-alignment.md)).
5. **Frontier as teacher/auditor only** — benchmark-set creation, disagreement review,
   hard cases, prompt calibration, pseudo-labels, failure analysis. Never the runtime
   default.
6. **Measure task-level parity** (next section).

> **⚠️ Frontier-as-teacher on real footage is a PII-egress event.** A frontier/closed
> reference model is typically a **cloud API**, and store CCTV frames are PII/biometric
> data ([10](10-platform-roadmap.md) §Security). Sending real footage to it violates the
> "no cloud dependency" scope ([01](01-overview.md)) and the trust posture. **Rule:** the
> frontier reference runs **only on synthetic or explicitly consented/redacted data**,
> never on raw store footage, and every such call is logged; it stays `enabled: false`
> by default (routing policy `frontier_reference`). Relaxing the *vendor* policy (below)
> does **not** relax this.

## Small-model parity report

A low-cost model is "good enough" only when it **matches the reference model on
business KPIs**, not on generic benchmarks:

```text
candidate recall · false-positive rate · review time saved · POS-cleared event rate ·
evidence-bundle usefulness · cost per useful review item · latency per camera ·
energy per event
```

Outputs (extending the Phase-11 benchmark harness, [10](10-platform-roadmap.md)):

```text
results/model_runs/model_performance.csv
results/model_runs/model_performance.json
results/model_runs/model_quality_report.json
```

`model_performance.csv` columns (target schema):

```text
run_id, camera_id, zone, chunk_id, quality_score, usable_frame_score,
capability, model_id, model_profile, device, precision, query_id, route_to_vlm,
qwen_ran, qwen_skipped_reason, video_seconds, decode_seconds, quality_seconds,
inference_seconds, postprocess_seconds, effective_fps, cpu_avg_percent, ram_peak_mb,
gpu_mem_peak_mb, gpu_avg_util_percent, gpu_avg_power_watts, disk_read_mb, disk_write_mb,
events_detected, query_hit, parse_success, candidate_events_created,
pos_transactions_checked, pos_matches, pos_misses, human_review_decision,
false_positive, useful_review_item, cost_per_useful_review_item
```

The report must answer: did image quality affect accuracy? which camera gives the best
evidence? cheapest model per capability? which low-cost model matches the reference on
this task? when does Qwen actually help vs. skip safely? which queries cause false
positives? which rules create useful review items? cost per useful evidence bundle?

## Metrics

Additive under the existing **`model_*`** convention ([11](11-model-plugin-policy.md)
§metrics) — the master prompt's `retail_video_ai_*` names are **not** used:

```text
model_selected_total / model_skipped_total / model_escalations_total
model_inference_seconds / model_cost_units_total
model_route_to_vlm_total · model_qwen_runs_total · model_qwen_skips_total
```

Stable labels: `capability, model_id, model_profile, device, precision, query_id,
status` (no high-cardinality labels — [03](03-models-and-query-modes.md) warning).
Surfaced in the **Model Routing** and **Model Cost vs Quality** dashboards
([08](08-dashboards.md) extensions; precursor: Model Performance & Usefulness).

## Test gate (target)

```text
test_model_registry_loads_plugins
test_{temporal,spatial,semantic,quality}_model_can_be_swapped
test_route_to_vlm_false_skips_semantic_vlm   test_route_to_vlm_true_runs_semantic_vlm
test_low_cost_model_runs_before_expensive_model
test_uncertain_result_escalates_to_vlm
test_model_performance_report_contains_quality_fields
test_no_high_cardinality_metric_labels
```

## Related

[11-model-plugin-policy.md](11-model-plugin-policy.md) ·
[15-quality-gate.md](15-quality-gate.md) ·
[17-pos-and-time-alignment.md](17-pos-and-time-alignment.md) ·
[03-models-and-query-modes.md](03-models-and-query-modes.md) ·
[13-models.md](13-models.md) · [10-platform-roadmap.md](10-platform-roadmap.md)
