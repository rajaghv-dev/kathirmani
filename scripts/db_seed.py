#!/usr/bin/env python3
"""Seed the platform DB from configs (idempotent). `make seed`.

- stores / cameras / zones  ← configs/stores, configs/cameras.yaml, configs/zones
- model_profiles            ← configs/models.yaml (each profile)
- model_registry            ← configs/model_catalog/nvidia_models.yaml + pinned
                              revisions from models/PROVENANCE.json (provenance)
Onboarding a new store = drop in its YAML + re-seed (master plan §10).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from db.db import connect  # noqa: E402

CFG = ROOT / "configs"


def _y(p): return yaml.safe_load(Path(p).read_text())


def seed(cur):
    cams = _y(CFG / "cameras.yaml")
    store = _y(CFG / "stores" / "kathirmani.yaml")["store"]
    zones = _y(CFG / "zones" / "kathirmani_zones.yaml")["zones"]

    cur.execute(
        "INSERT INTO stores(id,name,location,timezone) VALUES(%s,%s,%s,%s) "
        "ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name, location=EXCLUDED.location",
        (store["id"], store["name"], store.get("location"), store.get("timezone", "Asia/Kolkata")),
    )
    for c in cams["cameras"]:
        cur.execute(
            "INSERT INTO cameras(id,store_id,name,role,rtsp_url_env,enabled,fps_target) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE SET "
            "name=EXCLUDED.name, role=EXCLUDED.role, enabled=EXCLUDED.enabled",
            (c["id"], cams["store_id"], c["name"], c.get("role"), c.get("rtsp_url_env"),
             c.get("enabled", True), c.get("fps_target")),
        )
    for z in zones:
        cur.execute(
            "INSERT INTO zones(id,store_id,camera_id,name,zone_type,polygon_json) "
            "VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE SET "
            "name=EXCLUDED.name, zone_type=EXCLUDED.zone_type, polygon_json=EXCLUDED.polygon_json",
            (z["id"], zones[0].get("store_id", cams["store_id"]) if False else _y(CFG/"zones"/"kathirmani_zones.yaml")["store_id"],
             z["camera_id"], z["name"], z["zone_type"], json.dumps(z["polygon"])),
        )

    # model profiles
    models = _y(CFG / "models.yaml")
    active = models.get("active_profile")
    for name, prof in (models.get("profiles") or {}).items():
        cur.execute(
            "INSERT INTO model_profiles(profile_name,description,config_json,active) "
            "VALUES(%s,%s,%s,%s) ON CONFLICT (profile_name) DO UPDATE SET "
            "config_json=EXCLUDED.config_json, active=EXCLUDED.active",
            (name, prof.get("description", ""), json.dumps(prof), name == active),
        )

    # model registry (catalog + pinned provenance)
    prov = {}
    pf = ROOT / "models" / "PROVENANCE.json"
    if pf.exists():
        prov = json.loads(pf.read_text())
    catalog = _y(CFG / "model_catalog" / "nvidia_models.yaml")
    for m in catalog.get("models", []):
        rev = prov.get(m["id"], {}).get("revision")
        cur.execute(
            "INSERT INTO model_registry(model_id,model_family,vendor,modality,supported_tasks,"
            "supported_runtimes,license_notes,source_url,revision) "
            "VALUES(%s,%s,'nvidia',%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (m["id"], m.get("family", "unknown"), m.get("modality", []), m.get("tasks", []),
             m.get("runtimes", []), m.get("license"), m.get("source_url"), rev),
        )


def main():
    with connect() as conn, conn.cursor() as cur:
        seed(cur)
        conn.commit()
        for t in ("stores", "cameras", "zones", "model_profiles", "model_registry"):
            cur.execute(f"SELECT count(*) FROM {t}")
            print(f"  {t}: {cur.fetchone()[0]} rows")
    print("seed: OK")


if __name__ == "__main__":
    main()
