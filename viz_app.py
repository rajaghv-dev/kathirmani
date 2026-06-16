#!/usr/bin/env python3
"""Thin shim for `streamlit run viz_app.py`.

Streamlit re-executes this file on every interaction, so it calls main()
directly (rather than relying on app.py's __main__ guard). The real UI lives in
src/kathirmani/viz/app.py. See spec/09-repo-structure.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from kathirmani.viz.app import main

main()
