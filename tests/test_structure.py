"""Structural tests for the src/ package refactor.

Verify the package layout, the root entry-point shims, and that no hardcoded
CUDA device strings crept back into the model loaders. All run without torch.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "marlin"


def test_package_modules_present():
    for rel in [
        "__init__.py", "device.py", "ffmpeg_preload.py", "pipeline.py",
        "qwen_vl.py", "metrics.py", "loki.py", "annotations.py", "serve_metrics.py",
        "cli/run_inference.py", "cli/download_model.py", "viz/app.py",
    ]:
        assert (SRC / rel).exists(), f"missing package module: src/marlin/{rel}"


def test_root_shims_point_at_package():
    for shim in ["run_inference.py", "download_model.py", "serve_metrics.py", "viz_app.py"]:
        text = (ROOT / shim).read_text()
        assert "from marlin" in text, f"{shim} should import from the marlin package"


def test_no_hardcoded_cuda_in_loaders():
    """Model loaders must go through marlin.device, not hardcode 'cuda'."""
    for mod in ["pipeline.py", "qwen_vl.py"]:
        text = (SRC / mod).read_text()
        assert '"": "cuda"' not in text and ".to(\"cuda\")" not in text, (
            f"{mod} still hardcodes a CUDA device string"
        )
        assert "detect_device" in text, f"{mod} should use marlin.device.detect_device"


def test_old_inference_package_removed():
    assert not (ROOT / "inference").exists(), "old inference/ package should be gone"


def test_sources_parse():
    """Every package .py file must be syntactically valid."""
    for py in SRC.rglob("*.py"):
        ast.parse(py.read_text(), filename=str(py))


def test_spec_and_memory_dirs_exist():
    assert (ROOT / "spec").is_dir(), "spec/ documentation dir missing"
    assert (ROOT / "agents-memory").is_dir(), "agents-memory/ dir missing"
