#!/usr/bin/env python3
"""Render infographic-ready figures from the repo's real tested-scenario results.

Headless (Agg). Outputs PNGs to design/figures/ for use as-is or as inputs to AI
design tools (Claude/Canva/Figma). Numbers come from results/*.json (the 5-camera
Marlin/Qwen run), so the infographics are grounded in real data, not placeholders.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "design" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NV = "#76B900"        # NVIDIA green
INK = "#1a1a1a"
SLATE = "#3b4252"
MUTED = "#8a8f98"
BG = "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG})


def _load(p):
    try: return json.load(open(p))
    except Exception: return {}


summary = _load(ROOT / "results/summary.json")
econ = _load(ROOT / "results/economy.json")
fused = _load(ROOT / "results/fused.json")

CAM_LABELS = {"Bill Counter": "Bill Counter", "Center Camera": "Center",
              "Left Side": "Left Aisle", "Near Entry": "Entry/Exit", "Right Side": "Right Aisle"}
cams = []
for p in sorted(glob.glob(str(ROOT / "results/Kathirmani*.json"))):
    d = _load(p); name = os.path.basename(p)
    label = next((v for k, v in CAM_LABELS.items() if k in name), name[:14])
    cams.append((label, len(d.get("events", []))))


# ---- Figure 1: per-camera events (bar) --------------------------------------
def fig_events():
    labels = [c[0] for c in cams]; vals = [c[1] for c in cams]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=NV, edgecolor="white", width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v),
                ha="center", va="bottom", fontsize=16, fontweight="bold", color=INK)
    ax.set_title("Events detected per camera", fontsize=20, fontweight="bold", color=INK, pad=16)
    ax.set_ylabel("events", fontsize=12, color=SLATE)
    ax.set_ylim(0, max(vals) + 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=SLATE, labelsize=12)
    fig.text(0.5, 0.01, "Kathirmani store · 5 cameras · Marlin-2B caption mode",
             ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "events_per_camera.png", dpi=160); plt.close(fig)


# ---- Figure 2: efficiency stat cards ----------------------------------------
def fig_efficiency():
    cards = [
        (f"{econ.get('energy_wh', 0):.1f}", "watt-hours", "energy for the full run"),
        (f"₹{econ.get('cost_inr', 0):.3f}", "cost", "to analyze 5 cameras"),
        (f"{econ.get('events_per_wh', 0):.1f}", "events / Wh", "useful signal per energy"),
        (f"₹{econ.get('cost_per_event_inr', 0):.4f}", "cost / event", "marginal cost"),
        (f"{econ.get('avg_power_w', 0):.0f} W", "avg GPU power", "during inference"),
        (f"{econ.get('wall_time_sec', 0)/60:.1f} min", "wall time", "end to end"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    for ax, (big, label, sub) in zip(axes.flat, cards):
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.04, 0.08), 0.92, 0.84, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc="#f5f7f2", ec=NV, lw=1.5, transform=ax.transAxes))
        ax.text(0.5, 0.62, big, ha="center", va="center", fontsize=30, fontweight="bold", color=INK)
        ax.text(0.5, 0.36, label, ha="center", va="center", fontsize=13, fontweight="bold", color=NV)
        ax.text(0.5, 0.20, sub, ha="center", va="center", fontsize=9.5, color=MUTED)
    fig.suptitle("Cost & efficiency — one 5-camera analysis run", fontsize=19, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "efficiency_cards.png", dpi=160); plt.close(fig)


# ---- Figure 3: cross-camera dedup -------------------------------------------
def fig_dedup():
    raw = fused.get("total_raw_events", 0); uniq = fused.get("unique_event_count", 0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Raw events\n(all cameras)", "Unique events\n(after fusion)"], [raw, uniq],
                  color=[MUTED, NV], width=0.5, edgecolor="white")
    for b, v in zip(bars, [raw, uniq]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, str(v), ha="center", fontsize=18, fontweight="bold", color=INK)
    ax.set_title("Cross-camera dedup\n5 angles → one store-wide truth", fontsize=17, fontweight="bold", color=INK)
    ax.set_ylim(0, max(raw, uniq) + 1.5); ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=SLATE, labelsize=11)
    fig.tight_layout(); fig.savefig(OUT / "cross_camera_dedup.png", dpi=160); plt.close(fig)


# ---- Figure 4: the 3-stage cascade ------------------------------------------
def fig_cascade():
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6)
    stages = [("WHEN", "Marlin-2B", "time-grounded\nevents", 1.2),
              ("WHERE", "Locate / YOLOE", "objects + the\ncheap gate", 4.6),
              ("WHAT", "Qwen / Nemotron-VL", "loss-prevention\nverification", 8.0)]
    for title, model, desc, x in stages:
        ax.add_patch(FancyBboxPatch((x, 0.9), 2.4, 1.8, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc="#f5f7f2", ec=NV, lw=2))
        ax.text(x + 1.2, 2.35, title, ha="center", fontsize=15, fontweight="bold", color=NV)
        ax.text(x + 1.2, 1.8, model, ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x + 1.2, 1.25, desc, ha="center", fontsize=9.5, color=SLATE)
    for x in (3.6, 7.0):
        ax.add_patch(FancyArrowPatch((x, 1.8), (x + 1.0, 1.8), arrowstyle="-|>", mutation_scale=22, color=SLATE, lw=2))
    ax.text(5.5, 0.45, "cheapest gate first — only suspicious windows reach the expensive VLM",
            ha="center", fontsize=10.5, style="italic", color=MUTED)
    ax.text(5.5, 3.25, "The 3-stage perception cascade", ha="center", fontsize=18, fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(OUT / "cascade_flow.png", dpi=160); plt.close(fig)


for f in (fig_events, fig_efficiency, fig_dedup, fig_cascade):
    f(); print("wrote", f.__name__)
print(f"figures -> {OUT.relative_to(ROOT)}")
