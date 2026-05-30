#!/usr/bin/env python3
"""Thin shim → marlin.cli.run_inference:main.

The implementation moved into the src/marlin package during the refactor; this
keeps `python run_inference.py` working from the repo root. See
spec/09-repo-structure.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from marlin.cli.run_inference import main

if __name__ == "__main__":
    main()
