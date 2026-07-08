"""load_twin(store_yaml) — build a StoreTwin from the store YAML + its refs.

The store YAML carries relative `cameras_config` / `zones_config` paths (resolved
against the store YAML's own directory). This is the only entry point that touches
the filesystem; everything else operates on the in-memory graph. Onboarding a new
store = drop in its 3 YAMLs and call `load_twin` — no code change (spec/10 §Phase 10).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .twin import Camera, StoreTwin, Zone


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_twin(store_yaml_path: str | Path) -> StoreTwin:
    store_path = Path(store_yaml_path).resolve()
    doc = _load_yaml(store_path)
    store = doc["store"]

    base = store_path.parent
    cams_doc = _load_yaml((base / doc["cameras_config"]).resolve())
    zones_doc = _load_yaml((base / doc["zones_config"]).resolve())

    twin = StoreTwin(
        store_id=store["id"],
        name=store["name"],
        location=store.get("location"),
        timezone=store.get("timezone", "Asia/Kolkata"),
    )

    for c in cams_doc.get("cameras", []):
        twin.cameras[c["id"]] = Camera(
            id=c["id"],
            name=c["name"],
            role=c.get("role"),
            enabled=c.get("enabled", True),
        )

    for z in zones_doc.get("zones", []):
        twin.zones[z["id"]] = Zone(
            id=z["id"],
            camera_id=z["camera_id"],
            name=z["name"],
            zone_type=z["zone_type"],
            polygon=z["polygon"],
        )

    return twin
