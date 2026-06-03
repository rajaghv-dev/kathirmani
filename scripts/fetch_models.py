#!/usr/bin/env python3
"""Trust-aware model fetcher for the Kathirmani platform.

Downloads NVIDIA open models from HuggingFace **pinned to a resolved commit sha**
and records provenance (repo, revision, url, license, sha, time) into
``models/PROVENANCE.json`` so every weight on disk is auditable — part of the
platform's supply-chain trust posture (see ``spec/10`` Security, ``spec/11``).

Usage:
    python scripts/fetch_models.py                 # default shortlist
    python scripts/fetch_models.py nvidia/RADIO-L  # specific repo(s)

Free/OSS NVIDIA models, verified on HF (see spec/11 table). Uses the locally
stored HF token automatically; gated=auto repos (Cosmos) need a one-time license
acceptance on the model page if a 403 occurs.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST = MODELS_DIR / "PROVENANCE.json"

# Default shortlist — the minimal runnable NVIDIA path (VLM + embedding + twin).
DEFAULT_SHORTLIST = [
    "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",  # VLM clip-reasoning
    "nvidia/C-RADIOv4-H",                        # visual embedding / search
    "nvidia/Cosmos-Reason2-2B",                  # digital-twin reasoning
]


def _license_of(info) -> str:
    tags = getattr(info, "tags", None) or []
    for t in tags:
        if isinstance(t, str) and t.startswith("license:"):
            return t.split(":", 1)[1]
    card = getattr(info, "card_data", None)
    if card and getattr(card, "license", None):
        return str(card.license)
    return "unknown"


def fetch(repo_id: str, api: HfApi) -> dict:
    local_dir = MODELS_DIR / repo_id.split("/", 1)[-1]
    info = api.model_info(repo_id, expand=["gated", "cardData", "tags", "sha"])
    sha = info.sha  # exact commit the 'main' ref currently points at
    print(f"[{repo_id}] revision={sha} gated={info.gated} -> {local_dir}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        revision=sha,                 # pin: never a moving ref
        local_dir=str(local_dir),
        # token is read from the local HF store automatically
    )
    return {
        "repo_id": repo_id,
        "revision": sha,
        "url": f"https://huggingface.co/{repo_id}/tree/{sha}",
        "license": _license_of(info),
        "gated": str(info.gated),
        "local_dir": str(local_dir.relative_to(REPO_ROOT)),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str]) -> int:
    repos = argv[1:] or DEFAULT_SHORTLIST
    MODELS_DIR.mkdir(exist_ok=True)
    api = HfApi()
    manifest: dict[str, dict] = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())

    failures = []
    for repo in repos:
        try:
            manifest[repo] = fetch(repo, api)
            print(f"[{repo}] OK", flush=True)
        except Exception as e:  # one bad model must not block the rest
            print(f"[{repo}] FAILED: {e!r}", flush=True)
            failures.append((repo, repr(e)))

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nProvenance written -> {MANIFEST.relative_to(REPO_ROOT)}", flush=True)
    if failures:
        print(f"\n{len(failures)} failed:", flush=True)
        for repo, err in failures:
            print(f"  - {repo}: {err}", flush=True)
        return 1
    print("All models fetched + pinned.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
