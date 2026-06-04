"""Save a clip thumbnail from an already-decoded RGB frame (no second decode)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_thumbnail(frame_rgb: np.ndarray | None, dest: Path, max_w: int = 480) -> str | None:
    if frame_rgb is None:
        return None
    from PIL import Image
    img = Image.fromarray(frame_rgb.astype("uint8"), "RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=80)
    return str(dest)
