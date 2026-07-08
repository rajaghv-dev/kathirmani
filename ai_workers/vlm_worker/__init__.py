"""vlm_worker — the plugin-host VLM verification worker (master plan Phase 6).

Consumes `event.created` events flagged `needs_vlm_verification`, loads the active
profile's `vlm_clip_reasoning` plugin (host.load_plugin), runs it to produce a
§8.4 VLMVerification, then writes a `vlm_observations` row, updates the source
event's confidence, writes a `model_runs` row, and emits the `model_*` metrics.
GPU/weights/Postgres are all optional — everything degrades to a deterministic
`fake_infer` / file-queue path so it runs hermetically (spec/11 plugin test gate).
"""
from __future__ import annotations

from .plugin import (NvidiaVlmPlugin, QwenBaselinePlugin,
                     nvidia_default_config, qwen_default_config)
from .host import load_plugin
from .parser import parse_verification, coerce
from .prompts import render, PROMPT_PACKS

__all__ = [
    "NvidiaVlmPlugin",
    "QwenBaselinePlugin",
    "nvidia_default_config",
    "qwen_default_config",
    "load_plugin",
    "parse_verification",
    "coerce",
    "render",
    "PROMPT_PACKS",
]
