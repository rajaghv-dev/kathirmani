#!/usr/bin/env python3
"""Render platform-status infographics (phase board + test coverage) for status
updates / decks. Headless (Agg) → design/figures/status_*.png. Numbers reflect the
build as of the last full `make test`; update COUNTS if the suite grows."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NV, INK, SLATE, MUTED, CARD, BG = "#76B900", "#1a1a1a", "#3b4252", "#8a8f98", "#f5f7f2", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG, "savefig.facecolor": BG})

PHASES = [
    ("0", "Skeleton"), ("1", "Ingestion"), ("2", "DB + API"), ("3", "Observability"),
    ("4", "CV worker"), ("5", "Rule engine"), ("6", "VLM worker"), ("7", "Evidence+Review"),
    ("8", "Search"), ("9", "VSS-parity"), ("10", "Digital twin"), ("11", "Benchmark"),
    ("12", "Hardening"), ("13", "Bake-off"),
]
# tests per component (from `make test`, isolated per-dir)
COUNTS = [
    ("marlin baseline", 24), ("ingestion", 9), ("api", 4), ("digital-twin", 5),
    ("rule-engine", 22), ("evidence-builder", 12), ("review-ui", 12), ("security", 38),
    ("cv-oss-worker", 14), ("vlm-worker", 23), ("embedding-worker", 26),
    ("vss-eval-worker", 24), ("benchmarks", 28), ("bakeoff", 20), ("observability", 4),
]
TOTAL = sum(c for _, c in COUNTS)


def fig_phase_board():
    fig = plt.figure(figsize=(13, 6)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.4)
    ax.add_patch(plt.Rectangle((0, 5.9), 14, 0.5, color=NV))
    ax.text(7, 6.15, "Kathirmani Video-AI Platform — 14 / 14 phases complete",
            ha="center", va="center", fontsize=20, fontweight="bold", color="white")
    # 2 rows x 7 cols of phase chips
    cols, w, h, gx, gy = 7, 1.75, 1.35, 0.12, 0.45
    x0, y_top = 0.35, 4.2
    for i, (num, name) in enumerate(PHASES):
        r, c = divmod(i, cols)
        x = x0 + c * (w + gx); y = y_top - r * (h + gy)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=CARD, ec=NV, lw=2))
        ax.text(x + w / 2, y + h * 0.62, num, ha="center", va="center",
                fontsize=22, fontweight="bold", color=NV)
        ax.text(x + w / 2, y + h * 0.26, name, ha="center", va="center", fontsize=9.5, color=INK)
        ax.text(x + w - 0.22, y + h - 0.22, "✓", ha="center", va="center",
                fontsize=13, fontweight="bold", color=NV)
    # footer status strip
    facts = [f"{TOTAL} tests green", "real Nemotron-VL wired", "NVIDIA-only models",
             "Postgres + filesystem", "built by 5 waves of parallel agents"]
    ax.text(7, 0.62, "   ·   ".join(facts), ha="center", va="center",
            fontsize=11.5, color=SLATE)
    ax.text(7, 0.25, "ingest → clips/windows → queue → CV → rules → VLM verify → evidence → review · search · digital twin · benchmarks",
            ha="center", va="center", fontsize=9, style="italic", color=MUTED)
    fig.savefig(OUT / "status_phases.png", dpi=160); plt.close(fig)


def fig_test_coverage():
    labels = [n for n, _ in COUNTS][::-1]; vals = [c for _, c in COUNTS][::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, vals, color=NV, edgecolor="white", height=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, str(v),
                va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_title(f"Test coverage — {TOTAL} tests across {len(COUNTS)} components",
                 fontsize=17, fontweight="bold", color=INK, pad=14)
    ax.set_xlim(0, max(vals) + 5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=SLATE, labelsize=10.5)
    fig.text(0.5, 0.01, "each component runs isolated (`make test`); fakes keep them GPU/service-free",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "status_test_coverage.png", dpi=160)
    plt.close(fig)


for f in (fig_phase_board, fig_test_coverage):
    f(); print("wrote", f.__name__)
print(f"status figures ({TOTAL} tests) -> {OUT.relative_to(Path(__file__).resolve().parents[1])}")
