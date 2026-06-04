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


# ---- Figure 5: platform architecture (landscape flow) -----------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 4.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2)
    boxes = [
        ("Cameras", "5 CCTV feeds\n/ RTSP", 0.3),
        ("Ingestion", "10-sec clips\n5-sec windows", 2.45),
        ("Store", "Postgres + pgvector\n+ local NVMe", 4.6),
        ("AI workers", "YOLOE / DeepStream\nNemotron-VL", 6.75),
        ("Rules +\nevidence", "zones · dwell ·\nbilling-bypass", 8.9),
        ("Review +\nGrafana", "approve/reject\ndashboards", 11.05),
    ]
    for title, desc, x in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.5), 1.75, 1.5, boxstyle="round,pad=0.04,rounding_size=0.1",
                                    fc="#f5f7f2", ec=NV, lw=1.8))
        ax.text(x + 0.875, 2.62, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(x + 0.875, 1.95, desc, ha="center", va="center", fontsize=8.3, color=SLATE)
    for x in (2.05, 4.2, 6.35, 8.5, 10.65):
        ax.add_patch(FancyArrowPatch((x, 2.25), (x + 0.4, 2.25), arrowstyle="-|>", mutation_scale=16, color=SLATE, lw=1.8))
    ax.text(6.5, 3.6, "Kathirmani Video-AI Platform — OSS ingestion, NVIDIA models, evidence-first",
            ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(6.5, 0.75, "free/open NVIDIA models · swappable by config · explainable in Grafana",
            ha="center", fontsize=9.5, style="italic", color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "platform_architecture.png", dpi=160); plt.close(fig)


# ---- Figure 6: operator one-pager (portrait composite) ----------------------
def fig_one_pager():
    total_ev = summary.get("total_events", 0)
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # header band
    ax.add_patch(plt.Rectangle((0, 0.9), 1, 0.1, color=NV))
    ax.text(0.5, 0.95, "AI eyes on your store", ha="center", va="center", fontsize=26, fontweight="bold", color="white")
    ax.text(0.5, 0.875, "Turn the CCTV you already have into alerts you can trust",
            ha="center", fontsize=12, color=SLATE)
    # step row
    steps = [("5 cameras", "watch the store"), ("AI cascade", "finds what matters"),
             ("Alerts + clips", "with human review")]
    for i, (t, d) in enumerate(steps):
        x = 0.18 + i * 0.32
        ax.add_patch(plt.Circle((x, 0.76), 0.035, color=NV))
        ax.text(x, 0.76, str(i + 1), ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        ax.text(x, 0.70, t, ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x, 0.665, d, ha="center", fontsize=9, color=MUTED)
    # what it finds
    ax.text(0.08, 0.60, "What it spots", fontsize=15, fontweight="bold", color=NV)
    finds = ["Suspicious item handling near shelves", "Leaving without billing (loss prevention)",
             "Time-stamped events across all 5 cameras", "Camera health (blocked / frozen / dark)",
             "Natural-language search of footage"]
    for i, f in enumerate(finds):
        ax.text(0.10, 0.555 - i * 0.038, "•", fontsize=14, color=NV)
        ax.text(0.13, 0.555 - i * 0.038, f, fontsize=11, color=INK, va="center")
    # stat cards
    cards = [(f"₹{econ.get('cost_inr', 0):.3f}", "to analyze 5 cameras"),
             (f"{econ.get('energy_wh', 0):.1f} Wh", "of electricity"),
             (f"{total_ev}", "events flagged"),
             (f"~{econ.get('wall_time_sec', 0)/60:.0f} min", "end to end")]
    for i, (big, sub) in enumerate(cards):
        x = 0.08 + (i % 2) * 0.46; y = 0.28 - (i // 2) * 0.13
        ax.add_patch(FancyBboxPatch((x, y), 0.40, 0.10, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc="#f5f7f2", ec=NV, lw=1.3, transform=ax.transAxes))
        ax.text(x + 0.20, y + 0.066, big, ha="center", fontsize=20, fontweight="bold", color=INK)
        ax.text(x + 0.20, y + 0.025, sub, ha="center", fontsize=9.5, color=MUTED)
    # footer
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.05, color=NV))
    ax.text(0.5, 0.025, "Evidence, not just alerts — every flag links to a clip + a human decision",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    fig.savefig(OUT / "operator_one_pager.png", dpi=150); plt.close(fig)


for f in (fig_events, fig_efficiency, fig_dedup, fig_cascade, fig_architecture, fig_one_pager):
    f(); print("wrote", f.__name__)
print(f"figures -> {OUT.relative_to(ROOT)}")
