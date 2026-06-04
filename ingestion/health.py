"""Camera-health scores from sampled frames (numpy only, no OpenCV dep).

Emits the signals master plan §13.1 lists (black/blur/freeze) + a composite
health_score in [0,1] (1 = healthy). The rule engine (Phase 5) flags a camera when
health_score <= 0.5 (configs/event_rules.yaml camera_health_issue).
"""
from __future__ import annotations

import numpy as np


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _sharpness(luma: np.ndarray) -> float:
    # variance of a discrete Laplacian — low = blurry/defocused
    lap = (luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
           - 4 * luma[1:-1, 1:-1])
    return float(lap.var())


def health_scores(samples: list[np.ndarray], fps_actual: float = 0.0) -> dict[str, float]:
    if not samples:
        return {"black_frame_score": 1.0, "blur_score": 1.0, "freeze_score": 1.0,
                "fps_actual": round(fps_actual, 2), "health_score": 0.0}

    lumas = [_luma(s) for s in samples]
    brightness = float(np.mean([l.mean() for l in lumas]))           # 0..255
    black = float(np.mean([1.0 if l.mean() < 15 else 0.0 for l in lumas]))

    sharp = float(np.mean([_sharpness(l) for l in lumas]))
    blur = 1.0 - min(1.0, sharp / 100.0)                              # >=100 var ⇒ sharp

    if len(samples) > 1:
        diffs = [float(np.mean(np.abs(lumas[i] - lumas[i - 1]))) for i in range(1, len(lumas))]
        motion = float(np.mean(diffs))
        freeze = 1.0 if motion < 0.5 else max(0.0, 1.0 - motion / 5.0)
    else:
        freeze = 0.5

    bright_ok = min(1.0, brightness / 40.0)        # too-dark pulls health down
    health = max(0.0, min(1.0, bright_ok * (1.0 - blur) * (1.0 - freeze) * (1.0 - black)))
    return {
        "black_frame_score": round(black, 3),
        "blur_score": round(blur, 3),
        "freeze_score": round(freeze, 3),
        "fps_actual": round(fps_actual, 2),
        "health_score": round(health, 3),
    }
