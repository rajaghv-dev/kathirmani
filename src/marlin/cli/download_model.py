#!/usr/bin/env python3
"""
Download inference models into models/.
Skips the download if the model is already present.

Usage:
    python download_model.py                        # download marlin (default)
    python download_model.py --model qwen-vl        # download Qwen2.5-VL-7B
    python download_model.py --model all            # download both models
    python download_model.py --token hf_...         # if not logged in via huggingface-cli
    python download_model.py --force                # re-download even if present
"""
import argparse
import os
import sys
from pathlib import Path

# Make the package importable regardless of launch style (root shim / -m / direct).
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from marlin import PROJECT_ROOT

MODELS = {
    "marlin": {
        "id": "NemoStation/Marlin-2B",
        "local": "models/Marlin-2B",
        "sentinel": "config.json",
    },
    "qwen-vl": {
        "id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "local": "models/Qwen2.5-VL-7B-Instruct",
        "sentinel": "config.json",
    },
}

BASE_DIR = PROJECT_ROOT  # models/ lives at the repo root, not next to this file


def is_downloaded(path: Path, sentinel: str) -> bool:
    return (path / sentinel).exists()


def download_one(model_key: str, token: str | None, force: bool) -> None:
    spec = MODELS[model_key]
    model_id = spec["id"]
    dest = BASE_DIR / spec["local"]
    sentinel = spec["sentinel"]

    if not force and is_downloaded(dest, sentinel):
        print(f"[download] {model_key}: already at {dest} — skipping.")
        print(f"[download] Use --force to re-download.")
        return

    print(f"[download] Downloading {model_id} → {dest}")
    if not token:
        print("[download] No token provided. If the repo is gated, set HF_TOKEN or pass --token.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[download] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,  # plain files, no symlinks to HF cache
        token=token,
    )
    # Verify sentinel exists after download
    if not (dest / sentinel).exists():
        print(f"[download] WARNING: download may be incomplete — {sentinel} not found in {dest}")
        sys.exit(1)
    size_gb = sum(f.stat().st_size for f in dest.rglob("*.safetensors")) / 1e9
    print(f"[download] Done: {dest}  ({size_gb:.1f} GB of weights)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="marlin",
        choices=list(MODELS.keys()) + ["all"],
        help='Which model to download: "marlin", "qwen-vl", or "all" (default: marlin)',
    )
    parser.add_argument("--token", default=None, help="HuggingFace access token")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    if args.model == "all":
        for key in MODELS:
            download_one(key, token, args.force)
    else:
        download_one(args.model, token, args.force)


if __name__ == "__main__":
    main()
