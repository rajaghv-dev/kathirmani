#!/usr/bin/env python3
"""Enforce the NVIDIA-only model policy (master plan A5.2 / spec/11).

`make validate-model-config` → this. Exits non-zero on any violation, so a
non-NVIDIA model can never silently become the production default. Explicit
escape hatches (logged, never silent): a task with `vendor_exception: true` +
`exception_reason`, or a whole profile flagged `non_default: true`.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("FAIL: PyYAML not installed (pip install pyyaml)")
    raise SystemExit(2)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"

# Source of truth: model-plugins/base/schemas.py KNOWN_TASKS + the shipped plugins.
KNOWN_TASKS = {
    "detection", "tracking", "embedding", "vlm_clip_reasoning",
    "summarization", "search_critic", "digital_twin_simulation",
}
KNOWN_PLUGINS = {
    "cv_oss_detector", "deepstream_detector", "deepstream_tracker",
    "nvidia_openai_compatible_vlm", "vlm_qwen_baseline", "nvidia_embedding",
    "nvidia_summary", "nvidia_search_critic", "nvidia_cosmos",
}


def validate(doc: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []

    policy = doc.get("model_policy", {})
    if policy.get("vendor_scope") != "nvidia_only":
        errors.append("model_policy.vendor_scope must be 'nvidia_only'")
    if policy.get("allow_non_nvidia_models", False) is not False:
        errors.append("model_policy.allow_non_nvidia_models must be false")
    if policy.get("allow_vss_runtime_dependency", False) is not False:
        errors.append("model_policy.allow_vss_runtime_dependency must be false")

    profiles = doc.get("profiles", {})
    active = doc.get("active_profile")
    if active not in profiles:
        errors.append(f"active_profile '{active}' not found in profiles")
        return errors, warns

    for pname, prof in profiles.items():
        non_default = bool(prof.get("non_default", False))
        scope = "active" if pname == active else "profile"
        if pname == active and non_default:
            errors.append(f"active profile '{pname}' is flagged non_default — "
                          "a comparison-only profile must never be the default")
        for task, spec in (prof.get("tasks") or {}).items():
            if not spec.get("enabled", False):
                continue
            where = f"[{pname}.{task}]"
            if task not in KNOWN_TASKS:
                errors.append(f"{where} unknown task")
            for req in ("plugin", "runtime", "model_id", "endpoint"):
                if not spec.get(req):
                    errors.append(f"{where} missing required field '{req}'")
            if spec.get("plugin") not in KNOWN_PLUGINS:
                errors.append(f"{where} plugin '{spec.get('plugin')}' is not a known plugin")
            if not (spec.get("metrics") or {}).get("enabled", False):
                errors.append(f"{where} metrics.enabled must be true (A5.2)")

            model_id = str(spec.get("model_id", ""))
            is_nvidia = model_id.startswith("nvidia/")
            if not is_nvidia:
                if non_default:
                    warns.append(f"{where} non-NVIDIA model '{model_id}' allowed "
                                 f"(profile non_default; {scope})")
                elif spec.get("vendor_exception") and spec.get("exception_reason"):
                    warns.append(f"{where} non-NVIDIA '{model_id}' via vendor_exception: "
                                 f"{spec['exception_reason'].strip()[:80]}")
                else:
                    errors.append(f"{where} non-NVIDIA model_id '{model_id}' in a default "
                                  "profile without vendor_exception+reason")
    return errors, warns


def main() -> int:
    if not CONFIG.exists():
        print(f"FAIL: {CONFIG} not found")
        return 2
    doc = yaml.safe_load(CONFIG.read_text())
    errors, warns = validate(doc)
    for w in warns:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    if errors:
        print(f"\nvalidate-model-config: FAIL ({len(errors)} error(s), {len(warns)} warn(s))")
        return 1
    print(f"\nvalidate-model-config: OK  (active='{doc.get('active_profile')}', "
          f"{len(warns)} warn(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
