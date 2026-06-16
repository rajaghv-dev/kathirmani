#!/usr/bin/env python3
"""Thin shim → kathirmani.cli.download_model:main. See spec/09-repo-structure.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from kathirmani.cli.download_model import main

if __name__ == "__main__":
    main()
