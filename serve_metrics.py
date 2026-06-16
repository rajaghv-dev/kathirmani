#!/usr/bin/env python3
"""Thin shim → kathirmani.serve_metrics:main. See spec/09-repo-structure.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from kathirmani.serve_metrics import main

if __name__ == "__main__":
    main()
