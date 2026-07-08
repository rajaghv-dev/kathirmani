"""vss_eval_worker CLI (master plan Phase 9 — VSS-style OSS parity).

Two subcommands, both runnable with NO GPU / weights / DB (they degrade to the
fake summarizer / pending parity instead of crashing):

  summarize    — run the §A4.2 staged long-video summarization for a
                 {store_id, time_range, cameras, summary_mode} request and print
                 the incident-aware report (summary / timeline / open_questions /
                 recommended_review_items). Pulls events from the DB if present.
  parity       — build the VSS-parity report and write parity_report.json.

Entrypoints: `summarize_run(...)`, `parity_run(...)`. `python -m` friendly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from .lvs import summarize
from .parity import DEFAULT_REPORT_PATH, write_report
from .summarizer import NemotronSummaryPlugin


def _db_conn():
    try:
        from db import connect
        return connect()
    except Exception:
        return None


def summarize_run(req: dict[str, Any], events: list[dict] | None = None) -> dict[str, Any]:
    """Run the staged summarization. `events` may be injected (tests); otherwise
    pulled from the DB. Returns the §A4.2 report dict."""
    conn = None if events is not None else _db_conn()
    plugin = NemotronSummaryPlugin()
    plugin.load()
    try:
        return summarize(req, events=events, conn=conn, plugin=plugin)
    finally:
        plugin.unload()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def parity_run(path: str = DEFAULT_REPORT_PATH) -> dict[str, Any]:
    """Build + write the parity report. Returns the report dict."""
    conn = _db_conn()
    try:
        return write_report(path=path, conn=conn)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _default_request() -> dict[str, Any]:
    """The §A4.2 example request (used when no --request file is given)."""
    return {
        "store_id": "kathirmani_01",
        "time_range": {"start": "2026-06-01T10:00:00+05:30",
                       "end": "2026-06-01T11:00:00+05:30"},
        "cameras": ["left_aisle", "billing", "entry_exit"],
        "summary_mode": "loss_prevention",
        "include_events": True,
        "include_low_confidence": False,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="vss_eval_worker (Phase 9)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("summarize", help="§A4.2 staged long-video summary")
    ps.add_argument("--request", help="path to a JSON request file (§A4.2 input)")
    pp = sub.add_parser("parity", help="build + write the VSS-parity report")
    pp.add_argument("--out", default=DEFAULT_REPORT_PATH)
    args = ap.parse_args()

    if args.cmd == "summarize":
        req = (json.loads(Path(args.request).read_text())
               if args.request else _default_request())
        report = summarize_run(req)
        print(json.dumps(report, indent=2, default=str))
    elif args.cmd == "parity":
        report = parity_run(path=args.out)
        print(f"[vss_eval_worker] parity report -> {args.out}")
        for d in report["capabilities"]:
            score = d["parity_score"]
            score_s = f"{score:.2f}" if score is not None else "  - "
            print(f"  {d['capability']:<13} score={score_s} [{d['status']}] "
                  f"{d['metric']}={d['value']}")
        ov = report["overall_parity_score"]
        print(f"  overall (measurable dims) = "
              f"{ov:.3f}" if ov is not None else "  overall = pending")
