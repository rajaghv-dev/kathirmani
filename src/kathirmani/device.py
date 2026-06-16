"""Runtime accelerator detection.

This repo runs on two very different machines, plus dev/CI boxes:

  - NVIDIA DGX (GB10)  -> CUDA   + bfloat16 + device_map + torch.compile
  - Apple Mac Studio   -> MPS    + float16  (no device_map, no compile)
  - anything else      -> CPU    + float32  (no device_map, no compile)

The decision of *which* accelerator to use is a pure function (``select_kind``)
so it can be unit-tested without torch installed. ``detect_device`` wraps it,
queries torch for what is actually available, and attaches the matching dtype
and loading policy.

Override the auto-detection with the ``KATHIRMANI_DEVICE`` env var
(``cuda`` / ``mps`` / ``cpu`` / ``auto``) or the ``prefer`` argument.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass

VALID_KINDS = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class DeviceConfig:
    """Everything the loaders need to place a model on the local accelerator."""

    kind: str               # "cuda" | "mps" | "cpu"
    torch_device: str       # value passed to tensor/model .to(...)
    machine: str            # human-readable label for logs/metrics
    supports_compile: bool  # whether torch.compile is worth using here
    # The following two are populated by detect_device() (need torch):
    dtype: object = None        # torch dtype, or None if torch unavailable
    device_map: object = None   # dict for from_pretrained on CUDA, else None

    def describe(self) -> str:
        bits = [f"device={self.kind}", f"machine={self.machine}"]
        if self.dtype is not None:
            bits.append(f"dtype={getattr(self.dtype, '__name__', self.dtype)}")
        bits.append(f"compile={'on' if self.supports_compile else 'off'}")
        return "[device] " + ", ".join(bits)


def _machine_label(kind: str, system: str, arch: str, cuda_name: str | None) -> str:
    """A human-readable description of the box we landed on."""
    if kind == "cuda":
        return f"NVIDIA GPU / CUDA ({cuda_name})" if cuda_name else "NVIDIA GPU / CUDA"
    if kind == "mps":
        if system == "Darwin" and arch == "arm64":
            return "Apple Silicon / Metal (MPS) — e.g. Mac Studio"
        return "Metal (MPS)"
    return f"CPU ({system}/{arch})"


def select_kind(
    prefer: str | None,
    has_cuda: bool,
    has_mps: bool,
    system: str,
    arch: str,
    cuda_name: str | None = None,
) -> tuple[str, str, list[str]]:
    """Pure accelerator-selection logic. Returns (kind, machine_label, warnings).

    ``prefer`` (or KATHIRMANI_DEVICE) forces a kind when available; if the forced
    kind is unavailable we warn and fall through to auto-detection. Auto order
    is CUDA -> MPS -> CPU.
    """
    warnings: list[str] = []
    pref = (prefer or "").strip().lower()
    if pref in ("", "auto", "none"):
        pref = None

    if pref is not None and pref not in VALID_KINDS:
        warnings.append(f"unknown KATHIRMANI_DEVICE/prefer={pref!r}; auto-detecting")
        pref = None

    if pref == "cuda":
        if has_cuda:
            kind = "cuda"
        else:
            warnings.append("requested cuda but no CUDA device available; auto-detecting")
            pref = None
    if pref == "mps":
        if has_mps:
            kind = "mps"
        else:
            warnings.append("requested mps but no MPS backend available; auto-detecting")
            pref = None
    if pref == "cpu":
        kind = "cpu"

    if pref is None:
        if has_cuda:
            kind = "cuda"
        elif has_mps:
            kind = "mps"
        else:
            kind = "cpu"

    return kind, _machine_label(kind, system, arch, cuda_name), warnings


def detect_device(prefer: str | None = None, *, quiet: bool = False) -> DeviceConfig:
    """Detect the best local accelerator and the loading policy for it.

    Imports torch lazily so the rest of the package (and tests of
    ``select_kind``) work even where torch is not installed.
    """
    prefer = prefer if prefer is not None else os.environ.get("KATHIRMANI_DEVICE")
    system = platform.system()
    arch = platform.machine()

    has_cuda = False
    has_mps = False
    cuda_name = None
    torch = None
    try:
        import torch as _torch

        torch = _torch
        has_cuda = bool(torch.cuda.is_available())
        has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if has_cuda:
            try:
                cuda_name = torch.cuda.get_device_name(0)
            except Exception:
                cuda_name = None
    except Exception:
        # torch missing (e.g. a docs/CI box): fall back to CPU labelling.
        pass

    kind, machine, warnings = select_kind(prefer, has_cuda, has_mps, system, arch, cuda_name)

    if not quiet:
        for w in warnings:
            print(f"[device] warning: {w}")

    dtype = None
    device_map = None
    supports_compile = False

    if torch is not None:
        if kind == "cuda":
            # bfloat16 where the GPU supports it (Ampere+/GB10), else float16.
            bf16 = False
            try:
                bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16 = False
            dtype = torch.bfloat16 if bf16 else torch.float16
            device_map = {"": "cuda"}
            supports_compile = True
        elif kind == "mps":
            dtype = torch.float16
            device_map = None       # accelerate device_map is unreliable on MPS
            supports_compile = False  # torch.compile is flaky on the MPS backend
        else:  # cpu
            dtype = torch.float32
            device_map = None
            supports_compile = False

    return DeviceConfig(
        kind=kind,
        torch_device=kind,
        machine=machine,
        supports_compile=supports_compile,
        dtype=dtype,
        device_map=device_map,
    )
