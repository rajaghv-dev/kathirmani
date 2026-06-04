"""Plugin host: pick the `vlm_clip_reasoning` plugin from the active profile.

This is the platform's anti-refactor seam (spec/11): the worker never names a
model — it asks the host to load whatever plugin the active `configs/models.yaml`
profile binds to the `vlm_clip_reasoning` task. Swapping Nemotron <-> the Qwen
baseline is a YAML edit (`active_profile` / the task's `plugin:`), no code change.

``load_plugin(profile_config)`` accepts either:
  * a resolved task dict (the profile's `vlm_clip_reasoning` block), or
  * a full models.yaml-shaped dict (we resolve the active profile's task), or
  * None -> read configs/models.yaml from disk (best-effort).

Unknown / missing plugin name -> the fake-capable NvidiaVlmPlugin (default), so the
worker always loads something runnable (every plugin has a fake_infer path).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "model-plugins")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base.plugin import ModelPlugin, PluginConfig             # noqa: E402

try:                                                          # package vs path import
    from .plugin import (NvidiaVlmPlugin, QwenBaselinePlugin,
                         nvidia_default_config, qwen_default_config)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plugin import (NvidiaVlmPlugin, QwenBaselinePlugin,
                        nvidia_default_config, qwen_default_config)

CONFIG_PATH = _REPO_ROOT / "configs" / "models.yaml"
DEFAULT_PLUGIN = "nvidia_openai_compatible_vlm"

# plugin name -> (class, default_config factory). The fake fallback is the Nvidia one.
_REGISTRY: dict[str, tuple[type[ModelPlugin], Any]] = {
    "nvidia_openai_compatible_vlm": (NvidiaVlmPlugin, nvidia_default_config),
    "vlm_qwen_baseline": (QwenBaselinePlugin, qwen_default_config),
}


def _load_yaml_config() -> dict[str, Any]:
    """Best-effort read of configs/models.yaml; {} if unavailable (no yaml/file)."""
    try:
        import yaml
        with open(CONFIG_PATH) as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _resolve_task_block(profile_config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the various accepted shapes down to the vlm_clip_reasoning task
    block (+ a synthesized `profile` name when we can determine it)."""
    cfg = profile_config if profile_config is not None else _load_yaml_config()
    if not isinstance(cfg, dict):
        return {}

    # Already a task block (carries a `plugin:` key)?
    if "plugin" in cfg and "tasks" not in cfg and "profiles" not in cfg:
        return dict(cfg)

    # Full models.yaml? -> active profile's task.
    if "profiles" in cfg:
        active = cfg.get("active_profile")
        profiles = cfg.get("profiles", {})
        prof = profiles.get(active) or {}
        task = dict((prof.get("tasks", {}) or {}).get("vlm_clip_reasoning", {}) or {})
        task.setdefault("profile", active or "unknown")
        return task

    # A single profile dict (has `tasks:`)?
    if "tasks" in cfg:
        return dict((cfg.get("tasks", {}) or {}).get("vlm_clip_reasoning", {}) or {})

    return {}


def _build_config(task: dict[str, Any], plugin_name: str,
                  factory) -> PluginConfig:
    """Merge the YAML task block over the plugin's default_config. The task block's
    scalar knobs (model_id, runtime, prompt_pack, max_output_tokens, …) win; unknown
    keys land in params so plugins can read them."""
    profile = task.get("profile", "unknown")
    base = factory(profile) if profile != "unknown" else factory()
    params = dict(base.params)
    known = {"task", "plugin", "model_id", "runtime", "endpoint", "profile",
             "enabled", "metrics", "description"}
    for k, v in task.items():
        if k not in known:
            params[k] = v
    return PluginConfig(
        task="vlm_clip_reasoning",
        plugin=plugin_name,
        model_id=task.get("model_id", base.model_id),
        runtime=task.get("runtime", base.runtime),
        endpoint=task.get("endpoint", base.endpoint),
        profile=profile if profile != "unknown" else base.profile,
        params=params,
    )


def load_plugin(profile_config: dict[str, Any] | None = None) -> ModelPlugin:
    """Factory: build the `vlm_clip_reasoning` plugin for the active profile.

    Does NOT call ``.load()`` — the worker does that (lazy/non-fatal). Always
    returns a runnable plugin (fake fallback) even with no config/yaml/weights."""
    task = _resolve_task_block(profile_config)
    plugin_name = task.get("plugin", DEFAULT_PLUGIN)
    cls, factory = _REGISTRY.get(plugin_name, _REGISTRY[DEFAULT_PLUGIN])
    config = _build_config(task, plugin_name if plugin_name in _REGISTRY else DEFAULT_PLUGIN,
                           factory)
    return cls(config)
