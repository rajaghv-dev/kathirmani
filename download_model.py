#!/usr/bin/env python3
"""
Download NemoStation/Marlin-2B into models/Marlin-2B/.
Skips the download if the model is already present.

Usage:
    python download_model.py
    python download_model.py --token hf_...   # if not logged in via huggingface-cli
    python download_model.py --force           # re-download even if present
"""
import argparse
import os
import sys
from pathlib import Path

MODEL_ID = "NemoStation/Marlin-2B"
DEFAULT_LOCAL_PATH = Path(__file__).parent / "models" / "Marlin-2B"
# Sentinel file that only exists after a complete download
SENTINEL = "config.json"


def is_downloaded(path: Path) -> bool:
    return (path / SENTINEL).exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEFAULT_LOCAL_PATH), help="Download destination")
    parser.add_argument("--token", default=None, help="HuggingFace access token")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()

    dest = Path(args.dest)

    if not args.force and is_downloaded(dest):
        print(f"[download] Model already at {dest} — skipping.")
        print(f"[download] Use --force to re-download.")
        return

    token = args.token or os.environ.get("HF_TOKEN")

    print(f"[download] Downloading {MODEL_ID} → {dest}")
    if not token:
        print("[download] No token provided. If the repo is gated, set HF_TOKEN or pass --token.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[download] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        MODEL_ID,
        local_dir=str(dest),
        local_dir_use_symlinks=False,  # plain files, no symlinks to HF cache
        token=token,
    )
    # Verify sentinel exists after download
    if not (dest / SENTINEL).exists():
        print(f"[download] WARNING: download may be incomplete — {SENTINEL} not found in {dest}")
        sys.exit(1)
    size_gb = sum(f.stat().st_size for f in dest.rglob("*.safetensors")) / 1e9
    print(f"[download] Done: {dest}  ({size_gb:.1f} GB of weights)")


if __name__ == "__main__":
    main()
