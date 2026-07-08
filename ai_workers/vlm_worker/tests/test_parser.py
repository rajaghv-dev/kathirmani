"""JSON parse + repair/retry (A11 #8 bad-output): clean JSON, fenced blocks,
trailing prose, and pure garbage -> never crash, always schema-valid."""
from __future__ import annotations

from dataclasses import fields

from model_plugins.base.schemas import VLMVerification
from ai_workers.vlm_worker.parser import parse_verification, coerce

_KEYS = {f.name for f in fields(VLMVerification)}


def _valid(v: dict) -> None:
    VLMVerification(**{k: v[k] for k in _KEYS})            # round-trips


def test_truncated_json_repaired():
    # small-VLM output cut off mid-list (hit token cap) — repaired to valid JSON
    trunc = ('{"verdict": "unclear", "confidence": 0.1, "observed_actions": '
             '["Person at the counter.", "Shopping basket on the counter')
    v, _, ok = parse_verification(trunc)
    assert ok is True
    _valid(v)
    assert v["verdict"] == "unclear"
    assert any("Shopping basket" in a for a in v["observed_actions"])


def test_clean_json():
    raw = ('{"verdict": "suspicious", "confidence": 0.8, '
           '"observed_actions": ["picked up item"], "missing_evidence": [], '
           '"recommended_next_step": "review billing", '
           '"structured_event_type": "concealment", "explanation": "clear"}')
    v, _, ok = parse_verification(raw)
    assert ok is True
    _valid(v)
    assert v["verdict"] == "suspicious"
    assert v["confidence"] == 0.8


def test_fenced_json_block():
    raw = ("Here is the analysis:\n```json\n"
           '{"verdict": "benign", "confidence": 0.3, "explanation": "normal"}\n'
           "```\nHope that helps.")
    v, _, ok = parse_verification(raw)
    assert ok is True
    assert v["verdict"] == "benign"
    _valid(v)


def test_trailing_prose_after_object():
    raw = '{"verdict": "unclear", "confidence": 50} and that is my assessment.'
    v, _, ok = parse_verification(raw)
    assert ok is True
    assert v["verdict"] == "unclear"
    assert v["confidence"] == 0.5                          # 50 -> percentage -> 0.5


def test_garbage_is_safe():
    v, raw, ok = parse_verification("the camera was blurry, no JSON here at all")
    assert ok is False
    _valid(v)                                              # still schema-valid
    assert v["verdict"] == "unclear"
    assert v["confidence"] == 0.0
    assert raw                                             # raw text preserved


def test_empty_and_none_are_safe():
    for raw in ("", None):
        v, _, ok = parse_verification(raw)                 # type: ignore[arg-type]
        assert ok is False
        _valid(v)


def test_unknown_verdict_collapses_to_unclear():
    v = coerce({"verdict": "definitely_a_thief", "confidence": 150})
    assert v.verdict == "unclear"
    assert v.confidence == 1.0                             # >100 clamps to 1.0


def test_string_lists_coerced():
    v = coerce({"verdict": "benign", "observed_actions": "walked in; looked around"})
    assert v.observed_actions == ["walked in", "looked around"]
