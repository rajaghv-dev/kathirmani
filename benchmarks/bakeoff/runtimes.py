"""Candidate runtime matrix for the Phase-13 NVIDIA bake-off.

The bake-off compares the *same* NVIDIA model served by *different* runtimes
(Ollama / llama.cpp / vLLM / TensorRT-LLM / NIM / Triton / transformers). On this
box (GB10 aarch64, Blackwell, CUDA 13) most of those runtimes are NOT installed —
prebuilt vLLM/Triton/NIM/TensorRT-LLM are unproven on this arch (spec/11). So each
candidate carries a **clearly-synthetic** timing/throughput/power profile that
seeds the Phase-11 `FakeRunner`, plus an `available()` probe that returns False
unless the runtime is *really* reachable on this host.

>>> EVERY number in `FAKE_PROFILE` below is a SYNTHETIC placeholder. The relative
    ordering encodes expected behaviour (server runtimes faster + higher power,
    portable runtimes slower + cheaper) so the selection logic is exercisable, but
    the absolute values are NOT measurements. They are overwritten the moment a
    runtime is actually run on the GPU.

A `RuntimeCandidate` is a pure data record: its fake profile feeds
`benchmarks.harness.FakeRunner`, its `available()` decides whether the bake-off
would (in real mode) try to drive it, and its `quality_score` is a synthetic
output-quality proxy (parse-success / accuracy stand-in) for the same model under
that runtime (e.g. quantized GGUF trades a little quality for portability).
"""
from __future__ import annotations

import os
import shutil
import socket
from dataclasses import dataclass, field
from typing import Any, Callable


# ---- Fake timing/throughput/cost profile (SYNTHETIC) ------------------------
@dataclass(frozen=True)
class FakeProfile:
    """Synthetic per-item timing inputs for `harness.FakeRunner`.

    base_latency_ms / jitter_ms / power_w / tokens_per_item / ttft_ms map 1:1 to
    FakeRunner kwargs. `quality_score` is a 0..1 output-quality proxy for the same
    model under this runtime (synthetic). ALL fields are placeholders.
    """
    base_latency_ms: float
    jitter_ms: float
    power_w: float
    tokens_per_item: int
    ttft_ms: float
    quality_score: float  # 0..1 synthetic quality proxy (parse/accuracy stand-in)
    synthetic: bool = True  # always True here — never a real measurement


@dataclass(frozen=True)
class RuntimeCandidate:
    """One runtime serving the task's model. `probe` decides real availability."""
    name: str                      # runtime id (ollama, vllm, ...)
    kind: str                      # "local" | "server" | "container"
    fake: FakeProfile
    # availability probe — returns True only if the runtime is really reachable.
    # Defaults are conservative: unless the binary/endpoint is found, -> False.
    probe: Callable[[], bool] = field(default=lambda: False)
    notes: str = ""

    def available(self) -> bool:
        """True iff this runtime is actually reachable on this host (never fakes)."""
        try:
            return bool(self.probe())
        except Exception:
            return False


# ---- Availability probes (real, conservative) -------------------------------
def _bin_on_path(*names: str) -> Callable[[], bool]:
    """Probe: a runtime is available iff one of its CLI binaries is on PATH."""
    return lambda: any(shutil.which(n) is not None for n in names)


def _tcp_reachable(host: str, port: int, timeout: float = 0.25) -> Callable[[], bool]:
    """Probe: a server runtime is available iff its host:port accepts a socket."""
    def _probe() -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False
    return _probe


def _python_import(module: str) -> Callable[[], bool]:
    """Probe: a runtime is available iff its python package imports."""
    def _probe() -> bool:
        import importlib.util
        return importlib.util.find_spec(module) is not None
    return _probe


# ---- The candidate matrix, per task -----------------------------------------
# vlm_clip_reasoning: the gated cascade VLM step (Nemotron-Nano-VL). The model is
# fixed; the runtimes differ. Synthetic ordering rationale (NOT measurements):
#   - transformers : reference path, slow + power-hungry, full quality.
#   - ollama       : portable GGUF, slower, lower power, slight quality loss (quant).
#   - llama_cpp    : like ollama, marginally faster, same quant quality.
#   - vllm         : server, fast batched decode, high power, full quality (target).
#   - tensorrt_llm : fastest (compiled engine), highest power, full quality (target).
#   - nim          : container, near-TRT speed, full quality, ops-managed (target).
_VLM_CLIP_REASONING: list[RuntimeCandidate] = [
    RuntimeCandidate(
        name="transformers", kind="local",
        fake=FakeProfile(base_latency_ms=2600, jitter_ms=700, power_w=200,
                         tokens_per_item=180, ttft_ms=620, quality_score=0.96),
        probe=_python_import("transformers"),
        notes="reference eager path (CUDA); installed today but slowest.",
    ),
    RuntimeCandidate(
        name="ollama", kind="server",
        fake=FakeProfile(base_latency_ms=2200, jitter_ms=600, power_w=150,
                         tokens_per_item=180, ttft_ms=500, quality_score=0.93),
        probe=_bin_on_path("ollama"),
        notes="portable GGUF; default generative runtime on this box.",
    ),
    RuntimeCandidate(
        name="llama_cpp", kind="local",
        fake=FakeProfile(base_latency_ms=2050, jitter_ms=550, power_w=145,
                         tokens_per_item=180, ttft_ms=470, quality_score=0.93),
        probe=_bin_on_path("llama-cli", "llama-server", "main"),
        notes="GGUF via llama.cpp; portable, slight quant quality loss.",
    ),
    RuntimeCandidate(
        name="vllm", kind="server",
        fake=FakeProfile(base_latency_ms=1300, jitter_ms=350, power_w=240,
                         tokens_per_item=180, ttft_ms=300, quality_score=0.96),
        probe=_tcp_reachable("localhost", 8001),
        notes="batched server; throughput target — UNPROVEN on aarch64/Blackwell.",
    ),
    RuntimeCandidate(
        name="tensorrt_llm", kind="server",
        fake=FakeProfile(base_latency_ms=950, jitter_ms=250, power_w=260,
                         tokens_per_item=180, ttft_ms=210, quality_score=0.96),
        probe=_tcp_reachable("localhost", 8002),
        notes="compiled engine; fastest target — UNPROVEN/uninstalled here.",
    ),
    RuntimeCandidate(
        name="nim", kind="container",
        fake=FakeProfile(base_latency_ms=1050, jitter_ms=280, power_w=250,
                         tokens_per_item=180, ttft_ms=240, quality_score=0.96),
        probe=_tcp_reachable("localhost", 8003),
        notes="NGC NIM container; needs NGC key + nvidia docker runtime — deferred.",
    ),
]

# Registry of task -> candidate runtimes. Add tasks as the bake-off expands.
RUNTIME_MATRIX: dict[str, list[RuntimeCandidate]] = {
    "vlm_clip_reasoning": _VLM_CLIP_REASONING,
}


def candidates_for(task: str) -> list[RuntimeCandidate]:
    """Return the candidate runtimes for `task` (raises on unknown task)."""
    try:
        return RUNTIME_MATRIX[task]
    except KeyError:
        raise KeyError(
            f"unknown bake-off task {task!r}; known: {sorted(RUNTIME_MATRIX)}")


def availability(task: str) -> dict[str, bool]:
    """Map runtime-name -> real availability for `task` (probes the host)."""
    return {c.name: c.available() for c in candidates_for(task)}


# Allow tests / ops to force-mark a runtime available via env (e.g. a real vLLM):
#   BAKEOFF_FORCE_AVAILABLE="vllm,nim"  -> those probes return True.
def _forced() -> set[str]:
    raw = os.environ.get("BAKEOFF_FORCE_AVAILABLE", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def is_available(candidate: RuntimeCandidate) -> bool:
    """Availability honoring the BAKEOFF_FORCE_AVAILABLE override (ops/testing)."""
    return candidate.name in _forced() or candidate.available()
