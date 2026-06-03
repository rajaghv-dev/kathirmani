"""The ModelPlugin contract (master plan A2.1).

Every model — NVIDIA or the Qwen baseline, detector or VLM — is loaded behind this
one interface, so the active `configs/models.yaml` profile decides *which* model
serves *which* task without any caller change. This is the platform's anti-refactor
mechanism (spec/11-model-plugin-policy.md).

A concrete plugin lives in `model-plugins/<name>/plugin.py`, implements `ModelPlugin`
for exactly one task contract (see schemas.py), and ships the 10-point test
(spec/tests / master plan A11) — including a `fake_infer` path so it is testable
without a GPU or real weights.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginConfig:
    """Resolved from one task block of an active profile in configs/models.yaml."""
    task: str
    plugin: str
    model_id: str
    runtime: str
    endpoint: str = "local"
    profile: str = "unknown"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Health:
    ok: bool
    detail: str = ""


class ModelPlugin(abc.ABC):
    """Load once, infer many, expose health + metrics, unload cleanly."""

    def __init__(self, config: PluginConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def load(self) -> None:
        """Acquire the model/runtime. Idempotent; cheap if already loaded."""

    @abc.abstractmethod
    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run one inference. Input/output shapes follow the task contract
        (schemas.py). Must record a `model_runs` row + `model_*` metrics."""

    def fake_infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Deterministic stub output for tests without a GPU/weights (A11 #3).
        Override to return a schema-valid example for the plugin's task."""
        raise NotImplementedError

    @abc.abstractmethod
    def health(self) -> Health:
        """Liveness/readiness — model loaded, endpoint reachable."""

    @abc.abstractmethod
    def metrics(self) -> dict[str, float]:
        """Point-in-time metric snapshot (also exported via metrics.py gauges)."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Release GPU/host memory."""
