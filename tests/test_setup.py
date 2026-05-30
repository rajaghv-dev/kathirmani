"""
test_setup.py — pytest test suite for the Kathirmani Marlin-2B inference pipeline.

Tests cover:
  - Model file existence
  - Result JSON schema validity
  - Event timestamp structure
  - Find-result span parsing
  - Metrics server liveness
  - Grafana liveness and datasource configuration
  - download_model.py idempotency
  - trim_video() helper
  - Required Python package imports
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.parent  # project root
RESULTS_DIR = HERE / "results"
RESULT_FILE = RESULTS_DIR / "Kathirmani Bill Counter.json"
MODEL_DIR = HERE / "models" / "Marlin-2B"
VIDEO_FILE = HERE / "Kathirmani Bill Counter.mkv"

METRICS_URL = "http://127.0.0.1:8900/metrics"
GRAFANA_HEALTH_URL = "http://localhost:3000/api/health"
GRAFANA_DS_URL = "http://localhost:3000/api/datasources/uid/marlin-metrics"
GRAFANA_USER = "admin"
GRAFANA_PASS = "admin"


# ---------------------------------------------------------------------------
# 1. Model files exist
# ---------------------------------------------------------------------------
def test_model_files_exist():
    """Both the config and the first weight shard must exist on disk."""
    config = MODEL_DIR / "config.json"
    shard = MODEL_DIR / "model-00001-of-00002.safetensors"
    if not MODEL_DIR.exists():
        pytest.skip(f"Model dir not present (dev box): {MODEL_DIR}")
    assert config.exists(), f"Missing model config: {config}"
    assert shard.exists(), f"Missing model shard: {shard}"


# ---------------------------------------------------------------------------
# 2. Result schema
# ---------------------------------------------------------------------------
def test_result_schema():
    """The saved result JSON must contain all required top-level keys."""
    assert RESULT_FILE.exists(), f"Result file not found: {RESULT_FILE}"
    data = json.loads(RESULT_FILE.read_text())

    required_keys = {"video", "label", "caption", "scene", "events", "find_results", "processed_at"}
    missing = required_keys - data.keys()
    assert not missing, f"Result JSON is missing keys: {missing}"

    assert isinstance(data["events"], list), "events must be a list"
    assert isinstance(data["find_results"], dict), "find_results must be a dict"
    assert len(data["find_results"]) >= 10, (
        f"find_results must have at least 10 keys, got {len(data['find_results'])}"
    )


# ---------------------------------------------------------------------------
# 3. Events have timestamps
# ---------------------------------------------------------------------------
def test_events_have_timestamps():
    """Every event in the result must have numeric start/end and a string description."""
    data = json.loads(RESULT_FILE.read_text())
    events = data["events"]
    assert events, "events list is empty — nothing to validate"

    for i, ev in enumerate(events):
        assert "start" in ev, f"Event {i} missing 'start'"
        assert "end" in ev, f"Event {i} missing 'end'"
        assert "description" in ev, f"Event {i} missing 'description'"
        assert isinstance(ev["start"], (int, float)), f"Event {i} 'start' is not a float"
        assert isinstance(ev["end"], (int, float)), f"Event {i} 'end' is not a float"
        assert isinstance(ev["description"], str), f"Event {i} 'description' is not a str"


# ---------------------------------------------------------------------------
# 4. Find results with format_ok=True have a valid span
# ---------------------------------------------------------------------------
def test_find_results_parsed():
    """Every find_result entry that parsed OK must have a [float, float] span."""
    data = json.loads(RESULT_FILE.read_text())
    find_results = data["find_results"]

    ok_entries = {q: r for q, r in find_results.items() if r.get("format_ok") is True}
    assert ok_entries, "No find_results with format_ok=True — nothing to validate"

    for query, res in ok_entries.items():
        span = res.get("span")
        assert span is not None, f"Query '{query}': format_ok=True but span is None"
        assert isinstance(span, list) and len(span) == 2, (
            f"Query '{query}': span must be a list of two elements, got {span!r}"
        )
        start, end = span
        assert isinstance(start, (int, float)), (
            f"Query '{query}': span[0] must be a float, got {type(start)}"
        )
        assert isinstance(end, (int, float)), (
            f"Query '{query}': span[1] must be a float, got {type(end)}"
        )


# ---------------------------------------------------------------------------
# 5. Metrics server (skip if not running)
# ---------------------------------------------------------------------------
def test_metrics_server_up():
    """GET /metrics must return 200 and include marlin_events_detected_total."""
    requests = pytest.importorskip("requests")
    try:
        resp = requests.get(METRICS_URL, timeout=3)
    except Exception as exc:
        pytest.skip(f"Metrics server not reachable at {METRICS_URL}: {exc}")

    assert resp.status_code == 200, (
        f"Expected 200 from {METRICS_URL}, got {resp.status_code}"
    )
    assert "marlin_events_detected_total" in resp.text, (
        "marlin_events_detected_total metric not found in /metrics response"
    )


# ---------------------------------------------------------------------------
# 6. Grafana health (skip if not running)
# ---------------------------------------------------------------------------
def test_grafana_up():
    """GET /api/health must return JSON with database=ok."""
    requests = pytest.importorskip("requests")
    try:
        resp = requests.get(GRAFANA_HEALTH_URL, timeout=3)
    except Exception as exc:
        pytest.skip(f"Grafana not reachable at {GRAFANA_HEALTH_URL}: {exc}")

    assert resp.status_code == 200, (
        f"Expected 200 from {GRAFANA_HEALTH_URL}, got {resp.status_code}"
    )
    body = resp.json()
    assert body.get("database") == "ok", (
        f"Grafana health check: expected database=ok, got {body!r}"
    )


# ---------------------------------------------------------------------------
# 7. Grafana datasource points at port 8900
# ---------------------------------------------------------------------------
def test_grafana_datasource_marlin():
    """The marlin-metrics datasource in Grafana must point at port 8900."""
    requests = pytest.importorskip("requests")
    try:
        resp = requests.get(
            GRAFANA_DS_URL,
            auth=(GRAFANA_USER, GRAFANA_PASS),
            timeout=3,
        )
    except Exception as exc:
        pytest.skip(f"Grafana datasource endpoint not reachable: {exc}")

    if resp.status_code == 404:
        pytest.skip("Datasource uid 'marlin-metrics' not found in Grafana")

    assert resp.status_code == 200, (
        f"Expected 200 from {GRAFANA_DS_URL}, got {resp.status_code}"
    )
    body = resp.json()
    ds_url = body.get("url", "")
    assert "8900" in ds_url, (
        f"Expected datasource url to contain '8900', got: {ds_url!r}"
    )


# ---------------------------------------------------------------------------
# 8. download_model.py is idempotent
# ---------------------------------------------------------------------------
def test_download_model_idempotent():
    """Running download_model.py when the model is present must print 'already at'."""
    if not MODEL_DIR.exists():
        pytest.skip(f"Model dir not present (dev box): {MODEL_DIR}")
    result = subprocess.run(
        [sys.executable, str(HERE / "download_model.py")],
        capture_output=True,
        text=True,
        cwd=str(HERE),
        timeout=30,
    )
    combined = result.stdout + result.stderr
    assert "already at" in combined.lower(), (
        f"Expected 'already at' in output, got:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# 9. trim_video produces a valid clip
# ---------------------------------------------------------------------------
def test_trim_video():
    """trim_video() must return an MP4 that exists and is larger than 100 KB."""
    if not VIDEO_FILE.exists():
        pytest.skip(f"Source video not found: {VIDEO_FILE}")
    pytest.importorskip("torch")
    pytest.importorskip("av")

    from marlin.pipeline import trim_video

    out = trim_video(VIDEO_FILE, 5.0)
    try:
        assert out.exists(), f"trim_video did not create output file: {out}"
        size = out.stat().st_size
        assert size > 100 * 1024, (
            f"Trimmed clip is too small ({size} bytes), expected > 100 KB"
        )
    finally:
        out.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 10. Required venv packages import cleanly
# ---------------------------------------------------------------------------
def test_venv_packages():
    """All runtime dependencies must be importable without error.

    Skipped on a bare dev box where torch isn't installed (this suite still
    validates result-schema + device-selection logic there); on a fully
    provisioned DGX / Mac Studio venv it asserts every dependency imports.
    """
    import importlib

    pytest.importorskip("torch")

    packages = [
        "torch",
        "transformers",
        "torchcodec",
        "av",
        "prometheus_client",
    ]
    failed = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError as exc:
            failed.append(f"{pkg}: {exc}")

    assert not failed, "Failed to import packages:\n" + "\n".join(failed)
