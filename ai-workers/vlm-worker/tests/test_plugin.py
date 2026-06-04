"""VLM plugins — must work with no GPU, no weights, no DB.

Covers the hermetic 10-point essentials: fake_infer shape/schema, lifecycle +
health/metrics, the no-clip fallback path, and prompt rendering through the plugin.
"""
from __future__ import annotations

from dataclasses import fields

from base.schemas import VLMVerification
from plugin import (NvidiaVlmPlugin, QwenBaselinePlugin,
                    nvidia_default_config, qwen_default_config)


def _event() -> dict:
    return {
        "type": "event.created",
        "event_id": "11111111-1111-1111-1111-111111111111",
        "store_id": "kathirmani_01",
        "camera_id": "left_aisle",
        "event_type": "suspicious_item_interaction",
        "severity": "medium",
        "zones": ["left_aisle_shelf_1"],
        "objects": ["person", "item"],
        "needs_vlm_verification": True,
        "evidence_path": "/nonexistent/clip.mp4",          # forces fake path
    }


def _assert_valid_verification(v: dict) -> None:
    keys = {f.name for f in fields(VLMVerification)}
    obj = VLMVerification(**{k: v[k] for k in keys})       # schema-valid round-trip
    assert obj.verdict in {"suspicious", "benign", "unclear"}
    assert 0.0 <= obj.confidence <= 1.0
    assert isinstance(obj.observed_actions, list)
    assert isinstance(obj.missing_evidence, list)


def test_nvidia_fake_infer_valid_verification():
    p = NvidiaVlmPlugin()
    out = p.fake_infer({"event": _event(), "clip_path": "/nope.mp4"})
    assert set(out) >= {"verification", "raw_output", "parse_success", "model_run"}
    _assert_valid_verification(out["verification"])
    assert out["model_run"]["faked"] is True
    assert out["verification"]["structured_event_type"] == "suspicious_item_interaction"


def test_qwen_fake_infer_valid_verification():
    p = QwenBaselinePlugin()
    out = p.fake_infer({"event": _event()})
    _assert_valid_verification(out["verification"])
    assert out["model_run"]["faked"] is True
    assert out["model_run"]["model_id"] == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_infer_falls_back_to_fake_without_model():
    p = NvidiaVlmPlugin()
    request = {"event": _event(), "clip_path": "/nonexistent/clip.mp4"}
    out = p.infer(request)                                  # no weights/clip -> fake path
    _assert_valid_verification(out["verification"])
    assert out["model_run"]["faked"] is True
    assert "latency_ms" in out["model_run"]


def test_plugin_builds_prompt_from_event():
    p = NvidiaVlmPlugin()
    prompt = p.build_prompt({"event": _event()})
    assert "suspicious_item_interaction" in prompt
    assert "JSON" in prompt
    # caller-supplied prompt wins
    assert p.build_prompt({"prompt": "custom"}) == "custom"


def test_lifecycle_and_health_metrics():
    p = NvidiaVlmPlugin()
    p.load()                                               # non-fatal even without weights
    h = p.health()
    assert h.ok is True                                    # fake-infer mode is still healthy
    m = p.metrics()
    assert set(m) >= {"requests_total", "fake_infer_total", "parse_ok_total",
                      "model_loaded"}
    p.infer({"event": _event(), "clip_path": "/nope.mp4"})
    assert p.metrics()["requests_total"] >= 1.0
    p.unload()
    assert p.metrics()["model_loaded"] == 0.0


def test_default_configs_match_task():
    cfg = nvidia_default_config()
    assert cfg.task == "vlm_clip_reasoning"
    assert cfg.plugin == "nvidia_openai_compatible_vlm"
    assert cfg.model_id == "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
    qcfg = qwen_default_config()
    assert qcfg.plugin == "vlm_qwen_baseline"
    assert qcfg.profile == "research_qwen_baseline"
