"""Phase-3 dashboard generator tests.

Runs the headless generator and asserts the platform set (01-18) is emitted as
valid Grafana JSON: 18 files, each parseable, each with a unique uid, >=1 panel,
and an "About this dashboard" panel (the durable explain-meaning preference).

Run:  .venv/bin/pytest observability/tests/ -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

OBS = Path(__file__).resolve().parents[1]
GEN = OBS / "grafana" / "make_dashboards.py"

EXPECTED_STEMS = [
    "01-ingestion", "02-camera", "03-queue", "04-cv-worker", "05-vlm-worker",
    "06-gpu", "07-event-quality", "08-incidents", "09-digital-twin", "10-tco",
    "11-model-registry", "12-model-throughput", "13-model-latency",
    "14-model-quality", "15-model-tco", "16-vss-parity",
    "17-model-failure-drift", "18-runtime-comparison",
]


def _generate(out_dir: Path) -> None:
    res = subprocess.run(
        [sys.executable, str(GEN), "--out", str(out_dir)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"generator failed:\n{res.stdout}\n{res.stderr}"


def _load_all(tmp_path) -> dict[str, dict]:
    out_dir = tmp_path / "dashboards"
    _generate(out_dir)
    dashes: dict[str, dict] = {}
    for stem in EXPECTED_STEMS:
        f = out_dir / f"{stem}.json"
        assert f.exists(), f"missing dashboard file: {f.name}"
        dashes[stem] = json.loads(f.read_text())  # valid JSON or raises
    return dashes


def test_all_18_files_exist_and_parse(tmp_path):
    dashes = _load_all(tmp_path)
    assert len(dashes) == 18


def test_unique_uids(tmp_path):
    dashes = _load_all(tmp_path)
    uids = [d["uid"] for d in dashes.values()]
    assert len(set(uids)) == len(uids) == 18, f"duplicate/missing uids: {uids}"
    assert all(u.startswith("kathir-") for u in uids)


def test_each_has_panels_and_about(tmp_path):
    dashes = _load_all(tmp_path)
    for stem, d in dashes.items():
        assert d.get("schemaVersion") == 39, f"{stem}: wrong schemaVersion"
        panels = d.get("panels", [])
        assert len(panels) >= 1, f"{stem}: no panels"
        about = [p for p in panels if p.get("title") == "About this dashboard"]
        assert len(about) == 1, f"{stem}: expected exactly one About panel"
        assert about[0]["type"] == "text"
        # at least one real data panel beyond the About text panel
        assert any(p.get("targets") for p in panels), f"{stem}: no metric targets"


def test_datasource_and_metric_names(tmp_path):
    """Targets reference the documented metric families on the marlin-metrics DS."""
    dashes = _load_all(tmp_path)
    all_exprs = []
    for d in dashes.values():
        for p in d["panels"]:
            for t in p.get("targets", []):
                all_exprs.append(t["expr"])
            ds = p.get("datasource")
            if ds is not None:
                assert ds == {"type": "prometheus", "uid": "marlin-metrics"}
    blob = "\n".join(all_exprs)
    assert "ingest_" in blob, "no ingest_* metric referenced"
    assert "model_" in blob, "no model_* metric referenced"
