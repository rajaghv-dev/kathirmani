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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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


# ---- Figure 6b: data-ingestion layer (RTSP -> model feed) -------------------
def fig_ingestion():
    fig, ax = plt.subplots(figsize=(15.4, 5.4)); ax.axis("off")
    ax.set_xlim(0, 15.4); ax.set_ylim(0, 5.4)
    stages = [
        ("Camera sources", "5 CCTV feeds\nRTSP H.264 / .mkv", 0.25),
        ("Capture", "GStreamer splitmuxsink\nstream-copy (no re-encode)\nPyAV for files", 2.45),
        ("Segment", "10-sec clips -> NVMe\n5-sec AI windows\ndecode-once frames", 4.65),
        ("Validate + tag", "health: frozen/dark/blocked\nmetadata: fps/codec/ts\nsha256 checksum", 6.85),
        ("Catalog", "Postgres rows\nvideo_segments + ai_windows\npaths, not blobs", 9.05),
        ("Queue", "PgQueue · SKIP LOCKED\npublish ai_window.ready", 11.25),
        ("Model feed", "frame-gate (motion/zone/fps)\n-> CV (YOLOE) + VLM\n(Nemotron-VL)", 13.45),
    ]
    for title, desc, x in stages:
        ax.add_patch(FancyBboxPatch((x, 2.0), 1.75, 1.95, boxstyle="round,pad=0.04,rounding_size=0.1",
                                    fc="#f5f7f2", ec=NV, lw=1.8))
        ax.text(x + 0.875, 3.62, title, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(x + 0.875, 2.72, desc, ha="center", va="center", fontsize=7.6, color=SLATE)
    for x in (2.05, 4.25, 6.45, 8.65, 10.85, 13.05):
        ax.add_patch(FancyArrowPatch((x, 2.95), (x + 0.38, 2.95), arrowstyle="-|>",
                                     mutation_scale=15, color=SLATE, lw=1.8))
    ax.text(7.5, 4.85, "Data-ingestion layer — from RTSP to model feed",
            ha="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(7.5, 1.45, "mechanism (GStreamer fps-cap / keyframe-only) vs policy (our deterministic gating) "
            "· clip blobs on NVMe, Postgres holds paths + checksums",
            ha="center", fontsize=9, style="italic", color=MUTED)
    ax.text(7.5, 0.95, "back-pressure & at-least-once: SELECT … FOR UPDATE SKIP LOCKED · ack/fail/retry · "
            "evidence-first (every clip checksummed + path-addressed)",
            ha="center", fontsize=9, style="italic", color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "data_ingestion_layer.png", dpi=160); plt.close(fig)


# ---- Figure 7: multi-scale temporal window (ingest once, reason at any scale)-
def fig_multiscale_time():
    AMBER = "#d08700"; RED = "#c0392b"
    fig, ax = plt.subplots(figsize=(14, 6.6)); ax.axis("off")
    ax.set_xlim(-1, 51); ax.set_ylim(0, 7.2)
    ax.text(25, 6.85, "Ingest once at 10s — reason at any scale",
            ha="center", fontsize=18, fontweight="bold", color=INK)
    ax.text(25, 6.45, "subtle actions have variable extent and cross clip + camera boundaries; spans are reconstructed from Postgres, not from the model's window",
            ha="center", fontsize=9.5, style="italic", color=MUTED)

    # --- subtle-action bars (variable extent) ---
    actions = [
        ("palming · 3s", 4, 3, NV, 5.75, "fits one clip"),
        ("tag-swap · 12s", 15, 12, AMBER, 5.15, "crosses a clip boundary"),
        ("sweethearting · 35s", 8, 35, RED, 4.55, "spans 4 clips + 2 cameras"),
    ]
    for label, x0, dur, col, y, note in actions:
        ax.add_patch(FancyBboxPatch((x0, y), dur, 0.42, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc=col, ec="white", lw=1.2, alpha=0.92))
        if dur < 5:  # too narrow for an inside label — caption to the right
            ax.text(x0 + dur + 0.4, y + 0.21, label + " — fits one clip", ha="left", va="center",
                    fontsize=9, fontweight="bold", color=col)
            continue
        else:
            ax.text(x0 + dur / 2, y + 0.21, label, ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white")
        ax.text(x0 + dur + 0.4, y + 0.21, note, ha="left", va="center", fontsize=8, color=MUTED)

    # --- Camera A clip grid (10s segments) ---
    yA = 3.55
    ax.text(-0.9, yA + 0.78, "Camera A — 10s clips · video_segments (abs start/end + sha256)",
            fontsize=9, fontweight="bold", color=SLATE)
    for i in range(5):
        ax.add_patch(Rectangle((i * 10, yA), 10, 0.62, fc="#eef2e8", ec=NV, lw=1.4))
        ax.text(i * 10 + 5, yA + 0.31, f"seg_{i:02d}", ha="center", va="center", fontsize=8, color=SLATE)

    # --- Camera B lane + cross-camera track link ---
    yB = 2.55
    ax.text(-0.7, yB + 0.42, "Camera B (aisle)", fontsize=8.5, fontweight="bold", color=SLATE)
    ax.add_patch(Rectangle((30, yB), 12, 0.5, fc="#fdeee0", ec=AMBER, lw=1.3))
    ax.text(36, yB + 0.25, "same person · track_id", ha="center", va="center", fontsize=7.5, color=SLATE)
    ax.add_patch(FancyArrowPatch((38, 4.55), (36, yB + 0.5), arrowstyle="-|>", mutation_scale=13,
                                 color=RED, lw=1.4, linestyle="--", connectionstyle="arc3,rad=0.2"))
    ax.text(41.5, 3.6, "cross-camera link\n(tracks.first/last_seen)", ha="left", va="center",
            fontsize=7.5, color=RED, style="italic")

    # --- AI window grid (5s) ---
    yW = 1.35
    ax.text(-0.7, yW + 0.42, "ai_windows — 5s", fontsize=8.5, fontweight="bold", color=SLATE)
    for i in range(10):
        ax.add_patch(Rectangle((i * 5, yW), 5, 0.42, fc=BG, ec=MUTED, lw=0.8))
    ax.text(25, 0.78, "a single fixed 5s window can't contain the 12s / 35s actions — "
            "the span is rebuilt as segment.start_time + offset (→ a tstzrange to add)",
            ha="center", fontsize=8.5, style="italic", color=MUTED)

    # time axis ticks
    for t in range(0, 51, 10):
        ax.text(t, yW - 0.18, f"{t}s", ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "ingest_multiscale_time.png", dpi=160); plt.close(fig)


# ---- Figure 8: coverage quadrant (where naive ingestion goes blind) ---------
def fig_coverage_quadrant():
    import numpy as np
    fig, ax = plt.subplots(figsize=(11, 7))
    pts = [  # (label, duration_s, connectedness 1..4)
        ("palming", 3, 1.2), ("tag-swap", 10, 2.0), ("skip-scan / pass-around", 16, 2.3),
        ("basket-to-bag bypass", 25, 3.0), ("sweethearting", 33, 2.8),
        ("grazing / consumption", 130, 3.4), ("return fraud", 220, 3.9),
    ]
    # shaded "requires pg-backed ingestion" region (everything outside the naive box)
    ax.axhspan(0.5, 4.4, facecolor="#f5f7f2", alpha=0.6, zorder=0)
    # naive box: one fixed 5s window, single-window connectedness
    ax.add_patch(Rectangle((1, 0.6), 4, 0.9, fc="white", ec=MUTED, lw=1.4, ls="--", zorder=1))
    ax.text(2.2, 0.78, "fire-and-forget\n+ one fixed 5s window", fontsize=8, color=MUTED, ha="center", va="bottom")
    for label, d, c in pts:
        ax.scatter(d, c, s=240, color=NV, edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(label, (d, c), xytext=(8, 8), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 600); ax.set_ylim(0.5, 4.4)
    ax.set_xticks([1, 3, 10, 30, 60, 180, 600])
    ax.set_xticklabels(["1s", "3s", "10s", "30s", "1m", "3m", "10m"])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["single\nwindow", "cross-\nclip", "cross-\ncamera", "cross-\nsession"])
    ax.set_xlabel("action duration", fontsize=12, color=SLATE)
    ax.set_ylabel("how connected", fontsize=12, color=SLATE)
    ax.set_title("The valuable cases live where naive ingestion is blind",
                 fontsize=17, fontweight="bold", color=INK, pad=14)
    ax.tick_params(colors=SLATE); ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.015, "everything outside the dashed box needs durable, re-queryable, pg-cataloged spans",
             ha="center", fontsize=9.5, style="italic", color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "ingest_coverage_quadrant.png", dpi=160); plt.close(fig)


# ---- Figure 9: the reasoning graph (why pg-integrated ingestion) -------------
def fig_reasoning():
    fig, ax = plt.subplots(figsize=(16, 8.6)); ax.axis("off")
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.6)
    ax.text(8, 8.25, "Why the data-ingestion layer must be Postgres-integrated",
            ha="center", fontsize=18, fontweight="bold", color=INK)
    bands = [
        ("Reality", "#3b4252", [
            "actions vary: 3s … tens of s … minutes",
            "each micro-action looks innocent",
            "one action crosses clips + cameras"]),
        ("Why fire-and-forget fails", "#c0392b", [
            "fixed window cuts a 12s action in half",
            "decode→infer→discard = no second look",
            "no shared timeline to link views",
            "state lives only in RAM"]),
        ("What's required", "#d08700", [
            "multi-scale re-windowable spans",
            "replayable, exact bytes",
            "temporal + spatial correlation",
            "persistent cross-window state",
            "tamper-evident provenance"]),
        ("How pg delivers", NV, [
            "segments→windows→events→incidents",
            "incident_events (M:N) + tracks",
            "path+checksum + job_queue replay",
            "pgvector(768) similar moments",
            "sha256 + append-only audit_log"]),
        ("Payoff", "#2d6a00", [
            "catch long & subtle actions",
            "cascade re-reads only flagged spans",
            "one store-wide truth (fusion)",
            "evidence-first incidents"]),
    ]
    n = len(bands); colw = 16 / n
    for bi, (title, col, nodes) in enumerate(bands):
        cx = bi * colw + colw / 2
        ax.add_patch(FancyBboxPatch((bi * colw + 0.25, 7.1), colw - 0.5, 0.6,
                                    boxstyle="round,pad=0.02,rounding_size=0.06", fc=col, ec="none"))
        ax.text(cx, 7.4, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color="white")
        y = 6.4
        for node in nodes:
            h = 0.78
            ax.add_patch(FancyBboxPatch((bi * colw + 0.25, y - h + 0.1), colw - 0.5, h - 0.18,
                                        boxstyle="round,pad=0.02,rounding_size=0.05",
                                        fc="#f5f7f2", ec=col, lw=1.3))
            ax.text(cx, y - h / 2 + 0.02, node, ha="center", va="center", fontsize=8.3,
                    color=INK, wrap=True)
            y -= h + 0.12
    for bi in range(n - 1):
        x = (bi + 1) * colw
        ax.add_patch(FancyArrowPatch((x - 0.22, 3.6), (x + 0.22, 3.6), arrowstyle="-|>",
                                     mutation_scale=16, color=SLATE, lw=2))
    fig.tight_layout(); fig.savefig(OUT / "ingest_reasoning.png", dpi=150); plt.close(fig)


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


for f in (fig_events, fig_efficiency, fig_dedup, fig_cascade, fig_architecture, fig_ingestion,
          fig_multiscale_time, fig_coverage_quadrant, fig_reasoning, fig_one_pager):
    f(); print("wrote", f.__name__)
print(f"figures -> {OUT.relative_to(ROOT)}")
