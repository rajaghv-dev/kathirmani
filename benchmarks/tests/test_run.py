"""End-to-end: a config loads + runs in fake mode, writes JSON, guards the DB."""
from pathlib import Path

import pytest

from benchmarks import run as runner

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_ALL_CONFIGS = sorted(_CONFIG_DIR.glob("*.yaml"))


def test_six_benchmark_configs_exist():
    # Phase 11 master plan: benchmarks 1-6.
    assert len(_ALL_CONFIGS) == 6


@pytest.mark.parametrize("cfg_path", _ALL_CONFIGS, ids=lambda p: p.stem)
def test_each_config_loads_and_runs_fake_mode(cfg_path, tmp_path, monkeypatch):
    # Point results dir at tmp so the test doesn't pollute benchmarks/results.
    monkeypatch.setattr(runner, "_RESULTS_DIR", tmp_path)
    cfg = runner.load_config(cfg_path)
    assert {"name", "task", "n_items"} <= set(cfg)

    r = runner.run_one(cfg_path, use_db=False)   # no DB in tests
    assert r["fake"] is True
    assert "SYNTHETIC" in r["synthetic_warning"]
    assert r["total_clips"] == cfg["n_items"]
    assert r["clips_per_min_gpu"] > 0
    assert r["p95_latency_ms"] >= r["p50_latency_ms"]
    assert r["avg_power_w"] > 0
    # TCO headline metrics present + positive.
    t = r["tco"]
    assert t["cost_per_1000_clips_inr"] >= 0
    assert t["cost_per_store_month_inr"] > 0
    assert t["cost_per_camera_month_inr"] > 0
    # JSON artifact written.
    assert Path(r["_json_path"]).exists()
    # DB explicitly skipped.
    assert r["_db_status"] == "skipped (--no-db)"


def test_vlm_config_reports_token_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_RESULTS_DIR", tmp_path)
    r = runner.run_one(_CONFIG_DIR / "05_vlm_verify_100.yaml", use_db=False)
    assert r["tokens_per_sec"] is not None
    assert r["ttft_ms_p95"] is not None
    assert r["parse_success_rate"] == 1.0   # fake runner always parses


def test_detection_config_has_no_token_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_RESULTS_DIR", tmp_path)
    r = runner.run_one(_CONFIG_DIR / "01_streams_5.yaml", use_db=False)
    assert r["tokens_per_sec"] is None


def test_db_write_is_guarded_not_raising(monkeypatch):
    # Force the db import to fail -> write_db_row returns a 'skipped' status.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "db":
            raise ImportError("no db in test")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    status = runner.write_db_row({"benchmark_name": "x", "model_profile_name": "y"})
    assert status.startswith("skipped")
