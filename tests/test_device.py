"""Unit tests for the accelerator-selection logic (kathirmani.device).

These exercise the pure `select_kind` decision function, so they run on any box
(no torch / GPU required) — which is the whole point: this is the code that
makes the repo portable across NVIDIA DGX (CUDA) and Apple Mac Studio (MPS).
"""
from kathirmani.device import select_kind, detect_device, DeviceConfig, VALID_KINDS


def _kind(prefer, has_cuda, has_mps, system="Linux", arch="x86_64", cuda_name=None):
    return select_kind(prefer, has_cuda, has_mps, system, arch, cuda_name)[0]


def test_auto_prefers_cuda_when_available():
    assert _kind(None, True, False) == "cuda"
    # CUDA wins even if MPS were somehow also present.
    assert _kind(None, True, True) == "cuda"


def test_auto_picks_mps_on_apple_silicon():
    assert _kind(None, False, True, system="Darwin", arch="arm64") == "mps"


def test_auto_falls_back_to_cpu():
    assert _kind(None, False, False) == "cpu"


def test_machine_label_describes_apple():
    _, label, _ = select_kind(None, False, True, "Darwin", "arm64")
    assert "Apple" in label and "MPS" in label


def test_machine_label_describes_cuda_name():
    _, label, _ = select_kind(None, True, False, "Linux", "x86_64", cuda_name="GB10")
    assert "GB10" in label and "CUDA" in label


def test_explicit_preference_respected():
    assert _kind("cpu", True, True) == "cpu"
    assert _kind("mps", False, True, system="Darwin", arch="arm64") == "mps"
    assert _kind("cuda", True, False) == "cuda"


def test_unavailable_preference_falls_back_with_warning():
    kind, _, warnings = select_kind("cuda", False, False, "Linux", "x86_64")
    assert kind == "cpu"
    assert any("cuda" in w for w in warnings)


def test_unknown_preference_warns_and_autodetects():
    kind, _, warnings = select_kind("metal", False, True, "Darwin", "arm64")
    assert kind == "mps"
    assert any("unknown" in w for w in warnings)


def test_detect_device_returns_valid_config():
    """detect_device() must always return a usable config, even without torch."""
    cfg = detect_device(quiet=True)
    assert isinstance(cfg, DeviceConfig)
    assert cfg.kind in VALID_KINDS
    assert cfg.torch_device == cfg.kind
    assert isinstance(cfg.machine, str) and cfg.machine
    # device_map is CUDA-only; MPS/CPU must load-then-.to(device).
    if cfg.kind != "cuda":
        assert cfg.device_map is None
