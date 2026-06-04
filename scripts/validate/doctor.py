#!/usr/bin/env python3
"""Run every component validator and print a status matrix. `make validate` / `make doctor`.

Exit 0 if no hard failures (warnings allowed — e.g. a service that isn't running on a
dev box). Run a single component with `python scripts/validate/<component>.py`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import GREEN, RED, YEL, RST  # noqa: E402

COMPONENTS = ["env", "ingestion", "db", "models", "observability", "api", "gpu"]


def _load(name):
    # load by path under a unique module name so validator files don't collide
    # with the real packages they test (e.g. ingestion/, db/).
    spec = importlib.util.spec_from_file_location(f"validate_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("Kathirmani platform — doctor\n" + "=" * 40)
    hard_fail = 0
    rows = []
    for name in COMPONENTS:
        try:
            mod = _load(name)
            checks = mod.checks()
        except Exception as e:
            rows.append((name, 0, 0, 1, f"validator error: {e}"))
            hard_fail += 1
            continue
        ok = sum(1 for c in checks if c.ok)
        warn = sum(1 for c in checks if not c.ok and c.warn)
        fail = sum(1 for c in checks if not c.ok and not c.warn)
        hard_fail += fail
        rows.append((name, ok, warn, fail, ""))

    for name, ok, warn, fail, err in rows:
        if err:
            status = f"{RED}ERROR{RST}"
        elif fail:
            status = f"{RED}FAIL{RST} "
        elif warn:
            status = f"{YEL}WARN{RST} "
        else:
            status = f"{GREEN}OK{RST}   "
        detail = err or f"{ok} ok, {warn} warn, {fail} fail"
        print(f"  {status}  {name:<14} {detail}")

    print("=" * 40)
    if hard_fail:
        print(f"{RED}{hard_fail} hard failure(s){RST} — run the failing component's validator for detail.")
        return 1
    print(f"{GREEN}All components OK{RST} (warnings = optional/services not running).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
