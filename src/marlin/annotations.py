"""Grafana annotation pusher for Marlin inference pipeline.

Sends annotations to Grafana's API at key moments during inference
so operators can correlate events on dashboards with timeline markers.
"""
import requests
import time

GRAFANA_URL = "http://localhost:3000"
GRAFANA_AUTH = ("admin", "admin")

SECURITY_QUERIES = {
    "a person lingers near the exit",
    "a bag or object is left unattended on the floor",
    "a person conceals an item with their hand or body",
    "a person puts a product into their pocket or clothing",
    "a person walks toward the exit holding an item",
}


def push_annotation(text: str, tags: list[str], dashboard_uid: str = "") -> None:
    """Push a single annotation to Grafana.

    Failures are silently swallowed so annotation pushes never block inference.
    """
    payload = {
        "text": text,
        "tags": tags,
        "time": int(time.time() * 1000),
    }
    if dashboard_uid:
        payload["dashboardUID"] = dashboard_uid
    try:
        requests.post(
            f"{GRAFANA_URL}/api/annotations",
            json=payload,
            auth=GRAFANA_AUTH,
            timeout=3,
        )
    except Exception:
        pass


def annotate_camera_done(camera: str, events: int) -> None:
    """Mark completion of caption+find processing for one camera."""
    push_annotation(
        text=f"✓ {camera}: {events} events detected",
        tags=["marlin", "camera-complete", camera.lower().replace(" ", "-")],
    )


def annotate_security_event(camera: str, query: str, span: list) -> None:
    """Mark a security-relevant find hit on the Loki dashboard."""
    push_annotation(
        text=f"⚠ {camera}: {query} [{span[0]:.1f}s–{span[1]:.1f}s]",
        tags=["marlin", "security", "alert"],
        dashboard_uid="marlin-loki",
    )


def annotate_run_complete(cameras: int, events: int, wall_time: float) -> None:
    """Mark the end of a full inference run."""
    push_annotation(
        text=f"Inference complete: {cameras} cameras, {events} events, {wall_time:.0f}s",
        tags=["marlin", "run-complete"],
    )
