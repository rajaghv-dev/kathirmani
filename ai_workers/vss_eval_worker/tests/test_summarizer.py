"""NemotronSummaryPlugin — fake_infer determinism + lifecycle (no GPU/weights)."""
from __future__ import annotations

from ai_workers.vss_eval_worker.summarizer import NemotronSummaryPlugin, default_config


def test_default_config_is_summarization_nvidia():
    c = default_config()
    assert c.task == "summarization"
    assert c.plugin == "nvidia_summary"
    assert c.model_id.startswith("nvidia/")


def test_fake_infer_is_deterministic_and_nonempty():
    p = NemotronSummaryPlugin()
    req = {"level": "clip", "items": ["10:32:04 left_aisle: person handled item"],
           "context": {"camera": "left_aisle", "summary_mode": "loss_prevention"}}
    a = p.fake_infer(req)
    b = p.fake_infer(req)
    assert a["summary"] == b["summary"]               # deterministic
    assert a["summary"]                               # non-empty
    assert a["faked"] is True
    assert a["model_name"].startswith("nvidia/")


def test_fake_infer_distinguishes_inputs():
    p = NemotronSummaryPlugin()
    a = p.fake_infer({"level": "clip", "items": ["one event"]})
    b = p.fake_infer({"level": "clip", "items": ["a different event"]})
    assert a["summary"] != b["summary"]


def test_fake_infer_empty_items():
    p = NemotronSummaryPlugin()
    out = p.fake_infer({"level": "5min", "items": []})
    assert "No activity" in out["summary"]


def test_infer_falls_back_to_fake_without_weights():
    """No Ollama/transformers/weights in CI -> infer() must transparently fake."""
    p = NemotronSummaryPlugin()
    out = p.infer({"level": "report", "items": ["hourly summary 1"]})
    assert out["faked"] is True
    assert out["summary"]
    assert "latency_ms" in out


def test_health_and_metrics_runnable_without_model():
    p = NemotronSummaryPlugin()
    p.load()                                          # non-fatal even with nothing
    assert p.health().ok is True                      # fake mode is healthy
    p.infer({"level": "clip", "items": ["x"]})
    m = p.metrics()
    assert m["requests_total"] >= 1
    assert m["fake_infer_total"] >= 1
    p.unload()
