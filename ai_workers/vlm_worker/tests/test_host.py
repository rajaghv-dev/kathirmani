"""host.load_plugin picks the plugin by the active profile's
vlm_clip_reasoning.plugin name (with a fake-capable fallback)."""
from __future__ import annotations

from ai_workers.vlm_worker.host import load_plugin
from ai_workers.vlm_worker.plugin import NvidiaVlmPlugin, QwenBaselinePlugin


def test_picks_nvidia_from_task_block():
    task = {"plugin": "nvidia_openai_compatible_vlm",
            "model_id": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
            "runtime": "transformers", "profile": "nvidia_gb10_retail_balanced",
            "prompt_pack": "retail_loss_v1", "max_output_tokens": 512}
    p = load_plugin(task)
    assert isinstance(p, NvidiaVlmPlugin)
    assert p.config.plugin == "nvidia_openai_compatible_vlm"
    assert p.config.params.get("prompt_pack") == "retail_loss_v1"


def test_picks_qwen_baseline_from_task_block():
    task = {"plugin": "vlm_qwen_baseline",
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "runtime": "transformers", "profile": "research_qwen_baseline"}
    p = load_plugin(task)
    assert isinstance(p, QwenBaselinePlugin)
    assert p.config.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_full_models_yaml_resolves_active_profile():
    cfg = {
        "active_profile": "research_qwen_baseline",
        "profiles": {
            "research_qwen_baseline": {
                "tasks": {"vlm_clip_reasoning": {"plugin": "vlm_qwen_baseline",
                                                 "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                                                 "runtime": "transformers"}}},
        },
    }
    p = load_plugin(cfg)
    assert isinstance(p, QwenBaselinePlugin)
    assert p.config.profile == "research_qwen_baseline"


def test_unknown_plugin_falls_back_to_default():
    p = load_plugin({"plugin": "some_future_plugin"})
    assert isinstance(p, NvidiaVlmPlugin)                  # fake-capable default


def test_no_config_loads_active_profile_from_disk():
    # default profile in configs/models.yaml binds nvidia_openai_compatible_vlm.
    p = load_plugin(None)
    assert isinstance(p, NvidiaVlmPlugin)
    assert p.config.task == "vlm_clip_reasoning"
