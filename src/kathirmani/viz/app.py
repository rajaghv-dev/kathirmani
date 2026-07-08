#!/usr/bin/env python3
"""Kathirmani inference viewer — Streamlit.

Reads results/*.json (+ the source videos) and surfaces the whole pipeline:
KPIs, annotated detection frames, occupancy heatmap, Marlin temporal timeline,
Qwen Q&A, and a live backend-comparison view for validating detectors.

Run:
    source .venv/bin/activate
    streamlit run viz_app.py
    # then open the URL it prints (default http://localhost:8501)
"""
import colorsys
import json
import sys
from pathlib import Path

import streamlit as st

# Make the package importable when streamlit runs this file directly.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kathirmani import PROJECT_ROOT, VIDEO_EXTS, VIDEOS_DIR

HERE = VIDEOS_DIR               # store-videos/ (repo-root fallback for old checkouts)
RESULTS = PROJECT_ROOT / "results"
SKIP_JSON = {"summary.json", "fused.json", "economy.json"}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cameras() -> dict:
    cams = {}
    if RESULTS.exists():
        for jf in sorted(RESULTS.glob("*.json")):
            if jf.name in SKIP_JSON:
                continue
            try:
                cams[jf.stem] = json.loads(jf.read_text())
            except Exception:
                pass
    return cams


@st.cache_data(show_spinner=False)
def load_aux() -> dict:
    out = {}
    for name in ("fused", "economy", "summary"):
        p = RESULTS / f"{name}.json"
        if p.exists():
            try:
                out[name] = json.loads(p.read_text())
            except Exception:
                pass
    return out


def find_video(label: str) -> Path | None:
    for ext in VIDEO_EXTS:
        for v in HERE.glob(ext):
            if v.stem == label:
                return v
    return None


@st.cache_data(show_spinner=False)
def extract_frames(video_path: str, times: tuple, max_w: int = 960):
    """Decode the source video once, return {t: (PIL_image_downscaled, scale)}."""
    import av
    from PIL import Image

    targets = sorted(set(round(t, 2) for t in times))
    if not targets:
        return {}
    frames, ti = {}, 0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            while ti < len(targets) and t + 0.5 >= targets[ti]:
                img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                scale = 1.0
                if img.width > max_w:
                    scale = max_w / img.width
                    img = img.resize((max_w, int(img.height * scale)))
                frames[targets[ti]] = (img, scale)
                ti += 1
            if ti >= len(targets):
                break
    return frames


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def color_for(label: str) -> tuple:
    h = (hash(label) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_boxes(img, detections, scale):
    from PIL import ImageDraw
    img = img.copy()
    d = ImageDraw.Draw(img)
    for det in detections:
        x0, y0, x1, y1 = (v * scale for v in det["box"])
        c = color_for(det["cls"])
        d.rectangle([x0, y0, x1, y1], outline=c, width=3)
        tag = f'{det["cls"]} {det.get("score", 0):.2f}'
        d.rectangle([x0, max(0, y0 - 14), x0 + 7 * len(tag), y0], fill=c)
        d.text((x0 + 2, max(0, y0 - 13)), tag, fill=(0, 0, 0))
    return img


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
def tab_overview(label, data, aux):
    st.subheader(f"Overview — {label}")
    loc = data.get("locate_analysis") or {}
    qwen = data.get("qwen_analysis") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Marlin events", len(data.get("events", [])))
    c2.metric("Peak people", loc.get("max_people", "—"))
    c3.metric("Frames routed → VLM", len(loc.get("route_to_vlm", [])) if loc else "—")
    find_hits = sum(1 for r in data.get("find_results", {}).values() if r.get("span"))
    c4.metric("Marlin find hits", find_hits)

    if loc.get("objects"):
        st.caption("Detected object instances (summed over sampled frames)")
        st.bar_chart(loc["objects"])

    if data.get("scene"):
        st.caption("Marlin scene caption")
        st.info(data["scene"])

    econ = aux.get("economy")
    if econ:
        st.caption("Compute economy (last full run, store-wide)")
        e1, e2, e3 = st.columns(3)
        e1.metric("Energy (Wh)", econ.get("energy_wh", "—"))
        e2.metric("Cost (₹)", econ.get("cost_inr", "—"))
        e3.metric("Avg power (W)", econ.get("avg_power_w", "—"))


def tab_detections(label, data):
    st.subheader(f"Detections — {label}")
    loc = data.get("locate_analysis") or {}
    frames_meta = loc.get("frames") or []
    if not frames_meta:
        st.warning("No locate_analysis for this camera. Run: "
                   "`python run_inference.py --video \"<name>\" --locate`")
        return

    video = find_video(label)
    if not video:
        st.error(f"Source video for '{label}' not found in project root.")
        return

    routed = set(round(t, 2) for t in loc.get("route_to_vlm", []))
    only_routed = st.checkbox("Show only frames routed to the VLM", value=False)
    shown = [f for f in frames_meta if (not only_routed or round(f["t_sec"], 2) in routed)]
    shown = [f for f in shown if f.get("detections")] or shown

    times = tuple(f["t_sec"] for f in shown)
    with st.spinner("Decoding frames…"):
        imgs = extract_frames(str(video), times)

    cols = st.columns(3)
    for i, fmeta in enumerate(shown):
        t = round(fmeta["t_sec"], 2)
        entry = imgs.get(t)
        if not entry:
            continue
        img, scale = entry
        annotated = draw_boxes(img, fmeta.get("detections", []), scale)
        badge = " ⚡VLM" if t in routed else ""
        cols[i % 3].image(
            annotated,
            caption=f"t={t}s · {fmeta.get('people',0)} people · "
                    f"{len(fmeta.get('detections',[]))} dets{badge}",
            use_container_width=True,
        )

    _occupancy_heatmap(frames_meta, imgs, shown)


def _occupancy_heatmap(frames_meta, imgs, shown):
    import numpy as np
    centers = [
        ((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2)
        for f in frames_meta for d in f.get("detections", []) if d["cls"] == "person"
    ]
    if len(centers) < 3:
        return
    st.caption("Person-occupancy heatmap (where people appeared across sampled frames)")
    base = next((img for img, _ in imgs.values()), None)
    w, h = (base.size if base else (960, 540))
    xs, ys = zip(*centers)
    # Detection boxes are in ORIGINAL pixels; rescale to the (downscaled) base.
    scale = next((s for _, s in imgs.values()), 1.0)
    heat, _, _ = np.histogram2d(
        [y * scale for y in ys], [x * scale for x in xs],
        bins=[max(8, h // 40), max(8, w // 40)], range=[[0, h], [0, w]],
    )
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6 * h / max(w, 1)))
    if base is not None:
        ax.imshow(base, alpha=0.5)
        ax.imshow(heat, cmap="inferno", alpha=0.5, extent=[0, w, h, 0], aspect="auto")
    else:
        ax.imshow(heat, cmap="inferno")
    ax.axis("off")
    st.pyplot(fig)


def tab_timeline(label, data):
    st.subheader(f"Marlin temporal timeline — {label}")
    spans = [(q, r["span"]) for q, r in data.get("find_results", {}).items()
             if r.get("span")]
    if not spans:
        st.warning("No Marlin find spans for this camera.")
        return
    import matplotlib.pyplot as plt
    spans.sort(key=lambda x: x[1][0])
    fig, ax = plt.subplots(figsize=(9, max(2, 0.4 * len(spans))))
    for i, (q, (s, e)) in enumerate(spans):
        ax.barh(i, max(e - s, 0.2), left=s, color=[c / 255 for c in color_for(q)])
    ax.set_yticks(range(len(spans)))
    ax.set_yticklabels([q[:48] for q, _ in spans], fontsize=7)
    ax.set_xlabel("seconds")
    ax.invert_yaxis()
    st.pyplot(fig)


def tab_qwen(label, data):
    st.subheader(f"Qwen2.5-VL Q&A — {label}")
    qwen = data.get("qwen_analysis") or {}
    if qwen.get("skipped"):
        st.info(f"VLM skipped by cascade: {qwen['skipped']} (no interesting frames).")
        return
    queries = qwen.get("queries") or {}
    if not queries:
        st.warning("No Qwen analysis for this camera. Run without `--skip-qwen`.")
        return
    if qwen.get("cascade_keyframes") is not None:
        st.caption(f"Reasoned over {qwen['cascade_keyframes']} cascade keyframe(s).")
    for q, r in queries.items():
        with st.expander(q[:80] + ("…" if len(q) > 80 else "")):
            if r.get("ok"):
                st.write(r.get("answer", ""))
            else:
                st.error(r.get("error", "no answer"))


def tab_store(aux):
    st.subheader("Store-wide (fused)")
    fused = aux.get("fused")
    if not fused:
        st.warning("No results/fused.json yet (produced by a full multi-camera run).")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Unique events", fused.get("unique_event_count", "—"))
    c2.metric("Raw events", fused.get("total_raw_events", "—"))
    c3.metric("Dedup ratio", f'{fused.get("dedup_ratio", 0):.2f}×')
    st.caption("Cameras fused")
    st.write(fused.get("cameras", []))
    rows = [
        {"query": q, "camera": r.get("camera"), "span": r.get("span")}
        for q, r in fused.get("find_results", {}).items() if r.get("span")
    ]
    if rows:
        st.caption("Best-camera find spans")
        st.dataframe(rows, use_container_width=True)


def tab_compare(label):
    st.subheader(f"Backend comparison — {label}")
    st.caption("Run detectors live on one frame to validate which to keep. "
               "Heavy models load on first use.")
    video = find_video(label)
    if not video:
        st.error("Source video not found.")
        return
    t = st.slider("Timestamp (s)", 0.0, 60.0, 6.0, 0.5)
    backends = st.multiselect("Backends", ["yoloe", "grounding-dino", "rf-detr"],
                              default=["yoloe"])
    if not st.button("Run detectors"):
        return
    frames = extract_frames(str(video), (t,), max_w=1280)
    entry = frames.get(round(t, 2))
    if not entry:
        st.error("Could not extract that frame.")
        return
    img, scale = entry
    from kathirmani.locate import LOCATE_VOCAB
    cols = st.columns(len(backends))
    for col, backend in zip(cols, backends):
        with col:
            st.markdown(f"**{backend}**")
            try:
                det = _get_detector(backend)
                import numpy as np
                arr = np.array(img)
                results = det.detect([(t, arr)], LOCATE_VOCAB)[0]
                dets = [{"cls": d.cls, "score": d.score,
                         "box": [v / scale for v in d.box]} for d in results]
                st.image(draw_boxes(img, dets, scale), use_container_width=True)
                st.write({d["cls"]: round(d["score"], 2) for d in dets} or "no detections")
            except Exception as e:
                st.error(f"{backend}: {e}")


@st.cache_resource(show_spinner=True)
def _get_detector(backend: str):
    from kathirmani.locate import load_detector
    return load_detector(backend)


# --------------------------------------------------------------------------
# Optimization roadmap (visualization of learning.md §6–§8)
# --------------------------------------------------------------------------
# Each item: (tier, name, status, impact, effort, fixed_cam, note)
#   status: "done" | "planned"
#   fixed_cam: exploits the fixed CCTV angle (do-less-work lever)
OPTIMIZATIONS = [
    ("1 · do less work", "Decode video once / reuse frames", "done", "high", "med", True,
     "1 decode/camera instead of 1 + 20. Frames + video_metadata reused across "
     "all 21 prompts. Attacks the CPU-decode cost center."),
    ("1 · do less work", "Shared vision prefix / prefix-KV reuse", "planned", "high", "high", True,
     "All 21 prompts share identical video tokens — encode + prefill once, reuse "
     "the KV cache as prefix. vLLM auto prefix caching does this; PureKV ~3.16× prefill."),
    ("1 · do less work", "Motion-gating front-gate", "planned", "high", "med", True,
     "Fixed camera ⇒ cheap background subtraction (MOG2/KNN). Skip windows with no "
     "foreground motion before they reach Marlin/Qwen; feed only changing spans."),
    ("1 · do less work", "ROI / static-region masking", "planned", "med", "low", True,
     "Known door/counter/shelf regions — crop/mask to ROIs ⇒ fewer vision tokens ⇒ "
     "cheaper prefill."),
    ("2 · faster work", "Serve through vLLM", "planned", "high", "high", False,
     "Continuous batching fills the ALUs (kills batch_size=1) + PagedAttention + "
     "prefix caching. Stock Qwen runs in vLLM today; Marlin (Qwen3-VL fine-tune) "
     "may need an adapter — move Qwen first."),
    ("2 · faster work", "FP8 / INT8 quantization", "planned", "high", "med", False,
     "GB10 has no HBM — shared LPDDR5X ⇒ bandwidth-bound. Quant halves bytes/token "
     "moved, exactly the bottleneck. FP8 has Blackwell HW support."),
    ("2 · faster work", "CPU↔GPU pipelining", "planned", "med", "med", False,
     "Decode camera N+1 while N runs on GPU. Decode-once already moved decode off "
     "the GPU lock; this overlaps it fully."),
    ("3 · architecture", "Persistent scene priors", "planned", "med", "low", True,
     "Static scene caption barely changes between runs — cache it, spend budget on "
     "the varying find queries."),
    ("3 · architecture", "Cross-camera frame batching", "planned", "high", "high", True,
     "Same-timestamp frames from all 5 cameras in one forward pass — true GPU "
     "parallelism (serve_stream.py two-tier design), not just overlapped IO."),
    ("3 · architecture", "Per-camera query gating", "planned", "med", "med", True,
     "Use motion/Locate signals to skip finds that can't apply on a camera "
     "(e.g. 'lingers near exit' on the bill counter). Fewer generate() calls."),
]


def tab_optimizations(cams):
    st.subheader("Optimization roadmap")
    st.caption("Visualization of learning.md §6–§8. The pipeline is latency-bound "
               "on a serialized single-stream GPU with massive redundant decode — "
               "not capacity-bound. Two cost centers, different fixes.")

    # Real numbers from the running config.
    try:
        from kathirmani.pipeline import FIND_QUERIES
        prompts_per_cam = 1 + len(FIND_QUERIES)  # 1 caption + N finds
    except Exception:
        prompts_per_cam = 21
    n_cams = len(cams) or 5  # 5-camera store when no results loaded yet

    # --- The decode-once win (the one shipped optimization) ---
    st.markdown("#### Shipped: decode-once")
    before = prompts_per_cam * n_cams
    after = n_cams
    c1, c2, c3 = st.columns(3)
    c1.metric("Decodes / run — before", before)
    c2.metric("Decodes / run — after", after, delta=f"-{before - after}",
              delta_color="inverse")
    c3.metric("Redundant decode removed", f"{100 * (before - after) / before:.0f}%")
    st.bar_chart(
        {"decodes per run": {"before (1+20 per cam)": before, "after (1 per cam)": after}},
        horizontal=True,
    )
    st.caption(f"{n_cams} cameras × {prompts_per_cam} prompts each. GPU math is "
               "unchanged (still that many forward passes) — decode-once attacks the "
               "CPU half; the GPU half needs prefix-KV reuse / vLLM / quant below.")

    # --- The two cost centers ---
    st.markdown("#### The two cost centers")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.info("**Decode (CPU)** — torchcodec re-decodes per call.\n\n"
                "Fix: decode-once (done) + CPU↔GPU pipelining.")
    with cc2:
        st.warning("**generate() (GPU)** — one model + _GPU_LOCK serializes every "
                   "forward pass; batch_size=1.\n\n"
                   "Fix: prefix-KV reuse, vLLM continuous batching, FP8, cross-camera batching.")

    # --- The roadmap table ---
    st.markdown("#### Full roadmap")
    show_fixed_only = st.checkbox("Show only fixed-camera 'do-less-work' levers", value=False)
    rows = [
        {
            "Tier": tier,
            "Optimization": name,
            "Status": "✓ done" if status == "done" else "planned",
            "Impact": impact,
            "Effort": effort,
            "Fixed-cam lever": "yes" if fixed_cam else "—",
            "What / why": note,
        }
        for (tier, name, status, impact, effort, fixed_cam, note) in OPTIMIZATIONS
        if (not show_fixed_only or fixed_cam)
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    done = sum(1 for o in OPTIMIZATIONS if o[2] == "done")
    st.caption(f"{done}/{len(OPTIMIZATIONS)} shipped. Recommended sequence: "
               "decode-once (done) → motion-gate front → vLLM spike (Qwen first, "
               "then Marlin) → FP8 → ROI + query gating. On bandwidth-bound GB10, "
               "'do less work' (Tier 1/3) beats 'use the hardware harder' (Tier 2).")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Kathirmani Inference Viewer", layout="wide")
    st.title("Kathirmani — Inference Viewer")

    cams = load_cameras()
    aux = load_aux()
    if not cams:
        st.warning("No results found in results/. Run `python run_inference.py` first. "
                   "The Optimizations roadmap below does not need results.")
        tab_optimizations(cams)
        return

    with st.sidebar:
        st.header("Camera")
        label = st.radio("Select", list(cams.keys()))
        st.divider()
        st.caption(f"{len(cams)} cameras · "
                   f"{'fused ✓' if aux.get('fused') else 'no fused'} · "
                   f"{'economy ✓' if aux.get('economy') else 'no economy'}")

    data = cams[label]
    t_over, t_det, t_time, t_qwen, t_store, t_cmp, t_opt = st.tabs(
        ["Overview", "Detections", "Timeline", "Qwen Q&A", "Store (fused)",
         "Backend compare", "Optimizations"]
    )
    with t_over:
        tab_overview(label, data, aux)
    with t_det:
        tab_detections(label, data)
    with t_time:
        tab_timeline(label, data)
    with t_qwen:
        tab_qwen(label, data)
    with t_store:
        tab_store(aux)
    with t_cmp:
        tab_compare(label)
    with t_opt:
        tab_optimizations(cams)


if __name__ == "__main__":
    main()
