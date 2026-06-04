"""End-to-end: the bake-off runs in fake mode and emits a report with a
production + fallback set. No GPU/DB."""
import json
from pathlib import Path

from benchmarks.bakeoff import bakeoff as bk
from benchmarks.bakeoff import report as rp
from benchmarks.bakeoff import run as cli
from benchmarks.bakeoff import runtimes as rt


def test_runtime_matrix_has_expected_vlm_candidates():
    names = {c.name for c in rt.candidates_for("vlm_clip_reasoning")}
    assert {"ollama", "llama_cpp", "vllm", "tensorrt_llm", "nim"} <= names


def test_unknown_task_raises():
    import pytest
    with pytest.raises(KeyError):
        rt.candidates_for("does_not_exist")


def test_force_available_override(monkeypatch):
    monkeypatch.setenv("BAKEOFF_FORCE_AVAILABLE", "vllm")
    cand = next(c for c in rt.candidates_for("vlm_clip_reasoning") if c.name == "vllm")
    assert rt.is_available(cand) is True


def test_bakeoff_runs_fake_and_ranks():
    results = bk.run_bakeoff("vlm_clip_reasoning")
    assert len(results) == 6
    # ranked best-first by balanced_score
    scores = [r.balanced_score for r in results]
    assert scores == sorted(scores, reverse=True)
    # every result is flagged synthetic + carries metrics
    for r in results:
        assert r.synthetic is True
        assert r.clips_per_min > 0
        assert r.p95_latency_ms > 0
        assert r.tokens_per_sec is not None
        assert r.cost_per_1000_clips >= 0


def test_select_production_and_fallback_set():
    results = bk.run_bakeoff("vlm_clip_reasoning")
    sel = bk.select_profiles(results, task="vlm_clip_reasoning")
    names = {r.runtime for r in results}
    assert sel.production in names
    assert sel.fallback in names
    assert "synthetic" in sel.rationale["note"].lower()


def test_report_written_with_production_and_fallback(tmp_path):
    results = bk.run_bakeoff("vlm_clip_reasoning")
    sel = bk.select_profiles(results, task="vlm_clip_reasoning")
    report = rp.build_report("vlm_clip_reasoning", results, sel)
    assert report["production"] and report["fallback"]
    assert report["synthetic"] is True
    assert len(report["runtimes"]) == 6
    path = rp.write_report(report, out_dir=tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["production"] == sel.production
    assert loaded["fallback"] == sel.fallback


def test_db_write_guarded_not_raising(monkeypatch):
    # Force the db import to fail -> write_db_rows returns a 'skipped' status.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "db":
            raise ImportError("no db in test")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    status = rp.write_db_rows({"task": "t", "runtimes": [], "production": "x",
                               "fallback": "y", "model_profile_name": "p",
                               "gpu_name": "g", "dataset_name": "d"})
    assert status.startswith("skipped")


def test_cli_main_runs_end_to_end(tmp_path, monkeypatch):
    # Point the report dir at tmp; run the CLI with --no-db.
    monkeypatch.setattr(rp, "_REPORT_DIR", tmp_path)
    # build_report default out_dir is captured at call time via write_report default,
    # so also patch the module-level default used by write_report.
    rc = cli.main(["--task", "vlm_clip_reasoning", "--no-db"])
    assert rc == 0
