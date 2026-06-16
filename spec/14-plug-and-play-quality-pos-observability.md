# 14 — Plug-and-Play Models, Image Quality Gate, POS Correlation & Observability

## Purpose

This is the implementation master prompt for evolving the Kathirmani retail video AI pipeline from a fixed model cascade into a configurable, measurable, lower-cost-first evidence platform.

The key architectural change is that the traditional `WHEN / WHERE / WHAT` stack must become **plug-and-play capability hooks**, with an additional first-class `QUALITY` hook at ingestion time.

The goal is not:

```text
AI catches shoplifters.
```

The correct goal is:

```text
Video AI detects visible candidate events.
Image-quality gates improve model reliability.
POS checks transaction context.
Rules prioritize review.
Humans decide.
Observability proves cost, speed, and quality.
```

---

## 1. Scope

Current version includes:

- Local CCTV videos already available in a directory.
- Local-video-first pipeline.
- No live RTSP dependency.
- No Zoho integration.
- No ManageEngine integration.
- No cloud workflow dependency.
- OpenTelemetry.
- Prometheus.
- Grafana.
- Loki.
- Netdata.
- Store-POS as temporary open-source POS reference.
- Local evidence bundles.
- Local review UI.
- Plug-and-play model registry.
- Model quality/cost benchmarking.
- Ingestion-layer image/video quality scoring.

Future only:

- RTSP ingestion.
- 5-second live chunk writer.
- NVR-safe ingestion.
- Switch/SPAN/TAP ingestion.
- Enterprise workflow integration.
- Production POS integration.
- SKU-level product recognition.

---

## 2. Correct architecture shift

Do not hardcode the pipeline as:

```text
WHEN  -> Marlin
WHERE -> YOLOE
WHAT  -> Qwen
```

Instead, implement these as **capability hooks**:

```text
WHEN  -> temporal_event_model plugin
WHERE -> spatial_grounding_model plugin
WHAT  -> semantic_vlm_model plugin
```

The current default plugins may be:

```text
WHEN  -> Marlin-2B caption/find
WHERE -> Locate / YOLOE / route_to_vlm
WHAT  -> Qwen2.5-VL only on routed frames
```

But the architecture must allow replacing or comparing them with lower-cost or better models without refactoring the pipeline.

The system must support:

- Multiple WHEN models.
- Multiple WHERE models.
- Multiple WHAT models.
- Multiple image-quality models.
- Multiple POS adapters.
- Multiple rule profiles.
- Multiple routing policies.
- Model shadow mode.
- Model A/B evaluation.
- Model fallback.
- Model cost/quality reporting.

---

## 3. Why image quality must be first-class

Before asking any model to understand the video, the ingestion layer must measure whether the frame/chunk is usable.

CCTV model accuracy depends heavily on:

- Blur.
- Low light.
- Glare.
- Occlusion.
- Bad angle.
- Low resolution.
- Compression artifacts.
- Duplicate frames.
- Scene motion.
- Camera obstruction.
- Underexposure / overexposure.
- Missing timestamp alignment.
- Wrong camera-zone mapping.

A small model can perform surprisingly well when:

- The frame is sharp.
- The right camera is selected.
- The right time window is selected.
- The crop/ROI is good.
- The query is atomic.
- POS context is used.
- Rules remove impossible interpretations.
- Frontier models are used only as teacher/benchmark.

Therefore, implement:

```text
Video / chunk
-> image-quality scoring
-> frame filtering / enhancement / ROI selection
-> cheap model first
-> escalate only when uncertain
-> POS/rules correlation
-> human review
```

---

## 4. Ingestion-layer quality gate

Create an ingestion quality module.

Suggested files:

```text
src/retail_video_ai/quality/
  __init__.py
  image_quality.py
  video_quality.py
  roi_quality.py
  quality_schema.py
  quality_router.py
```

Use the current package layout if the repo still uses `src/marlin` rather than `src/retail_video_ai`.

For every video/chunk/frame, compute:

```text
brightness_score
contrast_score
blur_score
sharpness_score
motion_score
compression_artifact_score
occlusion_score
person_visibility_score
shelf_visibility_score
billing_counter_visibility_score
entry_exit_visibility_score
frame_duplicate_score
usable_frame_score
camera_health_score
```

Output:

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

Quality gate decisions:

```text
High quality + simple visible action
-> use low-cost model

Medium quality + important zone
-> use low-cost model + rule confirmation

Low quality + high-risk candidate
-> escalate to stronger model or mark needs review

Very poor quality
-> do not infer strongly; mark camera/frame quality issue
```

---

## 5. Model plugin registry

Create a model plugin system.

Suggested files:

```text
src/retail_video_ai/models/
  __init__.py
  base.py
  registry.py
  profiles.py
  router.py
  evaluator.py
  cost_model.py
  plugins/
    marlin_plugin.py
    yoloe_plugin.py
    qwen_plugin.py
    mock_plugin.py
```

Each plugin must declare:

```yaml
model_id: "marlin_2b"
display_name: "Marlin-2B"
provider: "local"
capabilities:
  - temporal_caption
  - temporal_find
runtime:
  engine: "pytorch"
  devices:
    - cuda
    - mps
    - cpu
cost_profile:
  relative_cost: "low"
  expected_vram_gb: 6
  expected_latency: "medium"
quality_profile:
  good_for:
    - atomic temporal events
    - visible action spans
  weak_for:
    - SKU recognition
    - intent inference
    - negation
    - counting
```

Common model interface:

```python
class ModelPlugin:
    model_id: str
    capabilities: list[str]

    def load(self, device, precision, config):
        ...

    def healthcheck(self):
        ...

    def run(self, request):
        ...

    def unload(self):
        ...
```

Capability-specific interfaces:

```python
class TemporalEventModel:
    def caption(self, video_or_chunk) -> CaptionResult:
        ...

    def find(self, video_or_chunk, query: AtomicQuery) -> FindResult:
        ...

class SpatialGroundingModel:
    def detect(self, frame_or_chunk) -> SpatialResult:
        ...

    def route_to_vlm(self, spatial_result, quality_result, policy) -> RouteDecision:
        ...

class SemanticVLMModel:
    def analyze(self, frame_or_clip, prompt, context) -> SemanticResult:
        ...

class ImageQualityModel:
    def score(self, frame_or_chunk) -> QualityResult:
        ...
```

The pipeline should call capabilities, not model names.

Bad:

```python
result = marlin.find(video, query)
```

Better:

```python
temporal_model = registry.get_capability("temporal_find", profile="default")
result = temporal_model.find(video, query)
```

---

## 6. Model profiles

Create config:

```text
configs/models.yaml
```

Example:

```yaml
profiles:
  economy:
    description: "Lowest cost path for routine runs"
    quality_model: "opencv_quality_v1"
    temporal_event_model: "marlin_2b"
    spatial_grounding_model: "yoloe_small"
    semantic_vlm_model: null
    escalation_model: "qwen2_5_vl"
    routing_policy: "quality_and_uncertainty"

  balanced:
    description: "Default run with selective VLM routing"
    quality_model: "opencv_quality_v1"
    temporal_event_model: "marlin_2b"
    spatial_grounding_model: "yoloe_base"
    semantic_vlm_model: "qwen2_5_vl"
    routing_policy: "route_to_vlm_only"

  benchmark:
    description: "Compare lower-cost models against stronger model"
    quality_model: "opencv_quality_v1"
    temporal_event_model:
      - "marlin_2b"
      - "candidate_temporal_model"
    spatial_grounding_model:
      - "yoloe_small"
      - "candidate_detector"
    semantic_vlm_model:
      - "qwen2_5_vl"
      - "frontier_reference_optional"
    routing_policy: "shadow_eval"
```

Every run must record:

```text
model_profile
model_id
capability
device
precision
runtime_engine
cost_profile
quality_profile
```

---

## 7. How lower-cost models can approach frontier-model usefulness

Do not expect a small model to beat a frontier model generally.

Instead, make it win on this narrow retail task using system design.

### 7.1 Narrow the problem

Use atomic visible actions:

```text
person picks item from shelf
person moves item to bag
person exits store
item appears on billing counter
person interacts with cashier
```

Avoid broad prompts:

```text
is this shoplifting?
what suspicious thing happened?
did the customer steal?
```

### 7.2 Improve input quality

Before inference:

- Skip duplicate frames.
- Reject very blurry frames.
- Choose sharper frames.
- Crop useful regions.
- Select best camera angle.
- Use billing/entry/shelf zone metadata.
- Use decode-once.
- Use quality-aware routing.

### 7.3 Use cascade routing

```text
cheap quality check
-> cheap temporal model
-> cheap detector
-> rules/POS
-> stronger VLM only when needed
```

### 7.4 Use POS as ground truth

Do not ask vision to identify SKU from CCTV.

Use POS for:

- Paid transaction exists or not.
- Time of billing.
- Cashier/till.
- Total amount.
- Product lines.
- Transaction status.

### 7.5 Use frontier model only as teacher / auditor

Use stronger models only for:

- Benchmark dataset creation.
- Disagreement review.
- Hard cases.
- Prompt calibration.
- Pseudo-label generation.
- Failure analysis.

### 7.6 Measure task-level parity

A lower-cost model is “good enough” only if it matches frontier/reference performance on business KPIs:

```text
candidate recall
false-positive rate
review time saved
POS-cleared event rate
evidence bundle usefulness
cost per useful review item
latency per camera
energy per event
```

---

## 8. Model router

Create a model router.

Suggested file:

```text
src/retail_video_ai/models/router.py
```

Routing inputs:

```text
camera_id
zone
chunk_quality_score
motion_score
person_visibility_score
query_category
risk_context
pos_context_available
previous_model_confidence
route_to_vlm
budget_profile
```

Routing decisions:

```json
{
  "stage": "semantic_vlm",
  "decision": "skip",
  "selected_model": null,
  "reason": "route_to_vlm=false and quality sufficient for temporal-only rule",
  "estimated_cost_saved": 0.83
}
```

Escalation policy:

```yaml
routing_policies:
  quality_and_uncertainty:
    run_quality_gate_first: true

    temporal:
      default_model: "marlin_2b"
      fallback_model: "candidate_temporal_model"
      skip_if_quality_below: 0.25

    spatial:
      default_model: "yoloe_small"
      escalate_if:
        - person_visibility_score < 0.45
        - object_confidence < 0.35

    semantic:
      default_model: "qwen2_5_vl"
      run_only_if:
        - route_to_vlm == true
        - candidate_risk != "low"
      skip_if:
        - route_to_vlm == false
        - usable_frame_score < 0.30

    frontier_reference:
      enabled: false
      run_only_for:
        - benchmark
        - disagreement
        - active_learning_sample
```

---

## 9. Preserve and generalize the three-axis design

Rewrite the existing design like this:

```text
WHEN hook
- capability: temporal_caption, temporal_find
- default: Marlin-2B
- alternatives: any video temporal grounding model
- output: event spans, dense captions, query hits/misses

WHERE hook
- capability: spatial_grounding, object/person/region detection, route_to_vlm
- default: Locate / YOLOE
- alternatives: YOLO variants, RT-DETR, GroundingDINO-style detector, small custom detector
- output: boxes, regions, person/item/shelf/counter visibility, VLM routing decision

WHAT hook
- capability: semantic_vlm_analysis
- default: Qwen2.5-VL
- alternatives: any local VLM or frontier reference
- output: semantic review signal, not proof
- runs only when routed
```

Add a fourth hook before all three:

```text
QUALITY hook
- capability: image/video quality scoring
- default: OpenCV + lightweight heuristics
- alternatives: no-reference IQA model, small classifier, learned camera-health model
- output: usable_frame_score, routing hints, camera-quality warnings
```

Final hook map:

```text
QUALITY -> Is this visual input good enough?
WHEN    -> When did a visible event happen?
WHERE   -> Where is the relevant person/object/zone?
WHAT    -> What does the selected keyframe/clip appear to show?
POS     -> Was there a matching transaction?
RULES   -> Should this become a review item?
HUMAN   -> What is the final decision?
```

---

## 10. Atomic query set

Keep Marlin/find queries atomic and visible.

Example:

```yaml
queries:
  - id: shelf_pick
    text: "person picks an item from a shelf"
    category: "item_interaction"

  - id: hand_to_bag
    text: "person moves an item from hand into a bag"
    category: "concealment_candidate"

  - id: hand_to_pocket
    text: "person moves an item from hand into a pocket"
    category: "concealment_candidate"

  - id: item_on_billing_counter
    text: "item is placed on the billing counter"
    category: "billing"

  - id: billing_interaction
    text: "person interacts with cashier at billing counter"
    category: "billing"

  - id: exit_after_item_pick
    text: "person exits the store after holding an item"
    category: "exit_correlation"

  - id: item_return_to_shelf
    text: "person returns an item to the shelf"
    category: "normalizing_event"

  - id: repeated_hand_movement
    text: "person repeatedly moves hand between shelf and bag"
    category: "subtle_action_candidate"

  - id: group_distraction
    text: "multiple people gather closely near a shelf or staff member"
    category: "attention_gap"
```

Do not use:

```text
person steals an item
customer shoplifts
person leaves without paying
suspicious behavior
billing fraud
```

These belong in rules, not vision prompts.

---

## 11. POS-aware reasoning

Integrate open-source Store-POS as a temporary read-only POS adapter.

Use POS to avoid weak CCTV SKU detection.

Pipeline:

```text
video action timeline
+ POS transaction timeline
+ camera zone metadata
+ rules
= candidate review item
```

Adapter modes:

```yaml
pos:
  enabled: true
  adapter: "store_pos"
  mode: "read_only_http"   # read_only_http | readonly_db_export | mock_json
  base_url: "http://localhost:3000"
  timezone: "Asia/Kolkata"
  transaction_window_before_sec: 180
  transaction_window_after_sec: 240
```

Every candidate event must include POS context:

```json
{
  "pos_context": {
    "adapter": "store_pos",
    "lookup_window": {
      "start": "2026-06-15T10:31:12+05:30",
      "end": "2026-06-15T10:38:34+05:30"
    },
    "paid_transaction_found": false,
    "transactions_checked": [],
    "pos_match_status": "no_match"
  }
}
```

---

## 12. Time alignment

Every video span must map to wall-clock time.

Config:

```yaml
time_alignment:
  store_timezone: "Asia/Kolkata"
  video_start_wall_clock:
    bill_counter: "2026-06-15T10:00:00+05:30"
    center: "2026-06-15T10:00:00+05:30"
    left_aisle: "2026-06-15T10:00:00+05:30"
    entry: "2026-06-15T10:00:00+05:30"
    right_aisle: "2026-06-15T10:00:00+05:30"
  default_clock_drift_sec: 0
```

Every event output:

```json
{
  "camera_id": "center",
  "start_sec": 132.5,
  "end_sec": 136.0,
  "wall_clock_start": "2026-06-15T10:02:12.500+05:30",
  "wall_clock_end": "2026-06-15T10:02:16.000+05:30"
}
```

---

## 13. OpenTelemetry

Instrument every major stage.

Required spans:

```text
retail.run
retail.config.load
retail.video.discover
retail.quality.score
retail.quality.route
retail.camera.process
retail.video.decode
retail.model.select
retail.model.load
retail.model.run
retail.model.caption
retail.model.find
retail.locate.run
retail.route_to_vlm.evaluate
retail.qwen.run
retail.qwen.skip
retail.pos.healthcheck
retail.pos.transaction_lookup
retail.fusion.timeline
retail.rules.evaluate
retail.evidence.clip_extract
retail.evidence.frame_extract
retail.review.queue_create
retail.metrics.export
```

Important span attributes:

```text
run_id
store_id
camera_id
camera_zone
capability
model_id
model_profile
device_type
precision
quality_score
usable_frame_score
route_to_vlm
qwen_skipped_reason
query_id
rule_id
candidate_event_id
pos_adapter
pos_match_status
status
error_type
```

Do not put raw captions, raw customer data, full paths, or large text in span attributes.

---

## 14. Netdata and resource monitoring

Use Netdata plus app metrics to track cost.

For every model stage, record:

```text
cpu_percent
rss_memory_mb
vms_memory_mb
disk_read_mb
disk_write_mb
gpu_memory_mb
gpu_util_percent
gpu_power_watts
gpu_temperature
decode_seconds
inference_seconds
postprocess_seconds
effective_fps
```

This must work even on CPU-only machines.

If GPU/NVML is unavailable, metrics should emit safe defaults and not crash.

---

## 15. Prometheus metrics

Add metrics for quality, model routing, and cost.

```text
retail_video_ai_quality_score
retail_video_ai_usable_frames_total
retail_video_ai_rejected_frames_total
retail_video_ai_camera_quality_warnings_total

retail_video_ai_model_selected_total
retail_video_ai_model_skipped_total
retail_video_ai_model_escalations_total
retail_video_ai_model_inference_seconds
retail_video_ai_model_cost_units_total

retail_video_ai_route_to_vlm_total
retail_video_ai_qwen_runs_total
retail_video_ai_qwen_skips_total

retail_video_ai_pos_transactions_checked_total
retail_video_ai_pos_matches_total
retail_video_ai_pos_misses_total

retail_video_ai_candidate_events_total
retail_video_ai_evidence_bundles_total
retail_video_ai_review_decisions_total

retail_video_ai_process_cpu_percent
retail_video_ai_process_memory_rss_bytes
retail_video_ai_process_read_bytes_total
retail_video_ai_process_write_bytes_total
retail_video_ai_gpu_memory_used_bytes
retail_video_ai_gpu_utilization_percent
retail_video_ai_gpu_power_watts
```

Use stable labels:

```text
run_id
store_id
camera_id
zone
capability
model_id
model_profile
device
precision
query_id
rule_id
pos_adapter
status
```

Avoid high-cardinality labels:

```text
full file path
raw caption
raw error message
raw POS item name
full query text
```

---

## 16. Model evaluation and small-model parity report

Create:

```text
results/model_runs/model_performance.csv
results/model_runs/model_performance.json
results/model_runs/model_quality_report.json
```

Each row should capture:

```text
run_id
camera_id
zone
chunk_id
quality_score
usable_frame_score
capability
model_id
model_profile
device
precision
query_id
route_to_vlm
qwen_ran
qwen_skipped_reason
video_seconds
decode_seconds
quality_seconds
inference_seconds
postprocess_seconds
effective_fps
cpu_avg_percent
ram_peak_mb
gpu_mem_peak_mb
gpu_avg_util_percent
gpu_avg_power_watts
disk_read_mb
disk_write_mb
events_detected
query_hit
parse_success
candidate_events_created
pos_transactions_checked
pos_matches
pos_misses
human_review_decision
false_positive
useful_review_item
cost_per_useful_review_item
```

The report must answer:

- Did image quality affect accuracy?
- Which camera gives the best evidence?
- Which model is cheapest for each capability?
- Which low-cost model matches the reference model on this task?
- When does Qwen actually help?
- When is Qwen skipped safely?
- Which queries cause false positives?
- Which rules create useful review items?
- What is the cost per useful evidence bundle?

---

## 17. Dashboards

Preserve existing 7 dashboards.

Extend them with these views.

### Image Quality / Ingestion Health

- Usable frame score by camera.
- Rejected frames by reason.
- Blur/brightness/motion trends.
- Camera quality warnings.
- Quality vs query hit rate.

### Model Routing

- Model selected by capability.
- Cheap model vs escalated model.
- `route_to_vlm` true/false.
- Qwen runs vs skips.
- Escalation reason.

### Model Cost vs Quality

- Latency by model.
- Memory by model.
- GPU power by model.
- Cost per useful review item.
- False positives by model/query/rule.
- Low-cost vs reference comparison.

### POS Correlation

- POS matches.
- POS misses.
- Candidate events cleared by POS.
- Events escalated due to no POS match.

### Review Outcomes

- Confirmed issue.
- False positive.
- POS matched, cleared.
- Needs more review.
- Ignored.

---

## 18. Local review UI

The review UI must show:

- Video clip.
- Best frame.
- Camera.
- Zone.
- Quality score.
- Why frame/chunk was accepted.
- Marlin/temporal event result.
- Locate/WHERE result.
- `route_to_vlm` decision.
- Qwen answer if Qwen ran.
- Qwen skipped reason if skipped.
- Selected model profile.
- Model cost summary.
- POS lookup window.
- POS match/no-match.
- Rule that fired.
- Review buttons.

Review buttons:

```text
Confirmed issue
False positive
POS matched, clear event
Needs more review
Ignore
Poor video quality, cannot decide
```

Save to:

```text
results/reviews/review_decisions.jsonl
```

---

## 19. Tests

Add tests:

```text
test_model_registry_loads_plugins
test_temporal_model_can_be_swapped
test_spatial_model_can_be_swapped
test_semantic_model_can_be_swapped
test_quality_model_can_be_swapped
test_quality_gate_rejects_blurry_frame
test_quality_gate_marks_low_light_frame
test_quality_score_controls_routing
test_route_to_vlm_false_skips_semantic_vlm
test_route_to_vlm_true_runs_semantic_vlm
test_low_cost_model_runs_before_expensive_model
test_uncertain_result_escalates_to_vlm
test_pos_match_reduces_risk
test_pos_miss_increases_risk
test_rules_never_confirm_theft
test_model_performance_report_contains_quality_fields
test_no_high_cardinality_metric_labels
test_gpu_metrics_safe_on_cpu_machine
test_review_ui_saves_poor_quality_decision
```

---

## 20. Acceptance criteria

The work is complete only when:

- WHEN / WHERE / WHAT are implemented as plugin hooks.
- QUALITY is implemented as a first-class ingestion hook.
- Marlin remains the default temporal model.
- Locate/YOLOE remains the default spatial route model.
- Qwen remains optional and runs only when routed.
- Models can be swapped from config.
- Low-cost profile runs without Qwen by default.
- Balanced profile uses Qwen selectively.
- Benchmark profile compares models.
- Image quality is recorded for every camera/chunk.
- Poor quality frames do not create overconfident conclusions.
- POS adapter works in mock mode.
- Store-POS adapter exists or is stubbed with clear instructions.
- POS context is included in evidence bundles.
- OpenTelemetry spans include model selection and quality routing.
- Netdata + Prometheus show CPU/RAM/GPU/I/O behavior.
- Grafana shows quality, cost, routing, and review outcomes.
- Review UI shows model, quality, POS, and evidence context.
- `make validate` passes.
- `make test` passes.
- Docs explain how low-cost models can be made useful through quality gating, routing, POS correlation, and human review.

---

## 21. Final design statement

Frontier models are not the product.

The product is a disciplined evidence pipeline:

```text
Better frames,
atomic prompts,
cheap models first,
selective VLM routing,
POS correlation,
rules,
human review,
and observability
can make lower-cost models useful enough for real retail operations.
```
