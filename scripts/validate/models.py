#!/usr/bin/env python3
"""Validate the models component (policy + provenance + weights on disk)."""
from __future__ import annotations

import json

from _common import C, ROOT, cli, try_import


def checks():
    out = [try_import("huggingface_hub")]
    # NVIDIA-only policy must pass (hard)
    try:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import yaml
        from validate_model_config import validate
        errs, _ = validate(yaml.safe_load((ROOT / "configs/models.yaml").read_text()))
        out.append(C("model policy valid (NVIDIA-only)", not errs,
                     f"{len(errs)} error(s)" if errs else ""))
    except Exception as e:
        out.append(C("model policy valid", False, str(e)[:60]))
    # provenance + weights
    prov = ROOT / "models" / "PROVENANCE.json"
    if prov.exists():
        d = json.loads(prov.read_text())
        out.append(C("provenance pinned", all(v.get("revision") for v in d.values()),
                     f"{len(d)} models"))
        for repo, meta in d.items():
            p = ROOT / meta.get("local_dir", "")
            out.append(C(f"weights: {repo.split('/')[-1]}", p.exists(),
                         "not on disk; run scripts/fetch_models.py" if not p.exists() else "", warn=not p.exists()))
    else:
        out.append(C("models fetched", False, "run scripts/fetch_models.py", warn=True))
    return out


if __name__ == "__main__":
    raise SystemExit(cli("models — NVIDIA-only policy + provenance", checks()))
