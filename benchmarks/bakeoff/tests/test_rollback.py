"""Rollback: pure profile-history returns to the previous profile + file effect."""
import pytest

from benchmarks.bakeoff import rollback as rb


def test_history_tracks_current_and_previous():
    h = rb.ProfileHistory()
    assert h.current is None and h.previous is None
    h.activate("research_qwen_baseline")
    h.activate("nvidia_gb10_retail_balanced")
    assert h.current == "nvidia_gb10_retail_balanced"
    assert h.previous == "research_qwen_baseline"


def test_reactivating_same_profile_is_noop():
    h = rb.ProfileHistory()
    h.activate("a")
    h.activate("a")            # no-op: must not grow the stack
    assert h.stack == ["a"]
    assert h.previous is None


def test_rollback_returns_to_previous_profile():
    h = rb.ProfileHistory()
    h.activate("research_qwen_baseline")
    h.activate("nvidia_gb10_retail_balanced")
    target = h.rollback()
    assert target == "research_qwen_baseline"
    assert h.current == "research_qwen_baseline"   # switch-back recorded


def test_rollback_then_rollback_toggles_forward():
    h = rb.ProfileHistory()
    h.activate("a")
    h.activate("b")
    assert h.rollback() == "a"   # b -> a
    assert h.rollback() == "b"   # a -> b (deterministic toggle, not infinite unwind)


def test_rollback_with_no_history_raises():
    h = rb.ProfileHistory()
    with pytest.raises(rb.RollbackError):
        h.rollback()
    h.activate("only")
    with pytest.raises(rb.RollbackError):   # only one entry -> no previous
        h.rollback()


def test_apply_activation_falls_back_to_file_marker(tmp_path, monkeypatch):
    # Force the DB path to report "down" so the file marker is written.
    monkeypatch.setattr(rb, "_activate_in_db", lambda name: None)
    marker = tmp_path / "active_profile"
    status = rb.apply_activation("nvidia_gb10_retail_balanced", marker=marker)
    assert "file:" in status
    assert marker.read_text().strip() == "nvidia_gb10_retail_balanced"


def test_rollback_active_wires_decision_and_effect(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "_activate_in_db", lambda name: None)  # DB down
    marker = tmp_path / "active_profile"
    h = rb.ProfileHistory()
    h.activate("research_qwen_baseline")
    h.activate("nvidia_gb10_retail_balanced")
    target, status = rb.rollback_active(h, marker=marker)
    assert target == "research_qwen_baseline"
    assert marker.read_text().strip() == "research_qwen_baseline"
    assert "file:" in status
