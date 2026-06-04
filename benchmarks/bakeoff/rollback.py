"""Model-profile rollback — return to the previously-active profile.

The API's `/api/model-registry/activate/{profile}` (services/api/app.py) is THE
runtime profile switch: it sets `model_profiles.active`. A config-only profile swap
plus this rollback is what satisfies the A12 done-criterion ("a config-only change
switches NVIDIA model profiles and Grafana shows … differences", with a safe undo).

This module is split:
  - PURE LOGIC: a `ProfileHistory` (append-only stack of activated profiles) whose
    `rollback()` computes the previous profile with no I/O — unit-testable, no GPU,
    no DB. This is the decision.
  - THIN EFFECT: `apply_activation()` performs the chosen switch by the SAME path
    the API uses (UPDATE model_profiles SET active=... ) when Postgres is up, else
    writes a local `configs/active_profile` marker so a degraded box still records
    intent. `rollback_active()` wires the two together.

Rollback never invents a profile: it only returns to one already in the history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The local marker the degraded path writes (read by ops / a future `make up`).
# Lives under configs/ but is a runtime artifact, not the committed config.
_REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PROFILE_MARKER = _REPO_ROOT / "configs" / "active_profile"


# ---- Pure logic: the activation history + rollback decision -----------------
@dataclass
class ProfileHistory:
    """Append-only stack of activated profile names. Pure — no I/O.

    `activate(name)` records a switch (ignoring a no-op re-activation of the same
    profile). `current` is the tip; `previous` is the one before it. `rollback()`
    returns the previous profile name and records the switch back (so a second
    rollback toggles forward again — a deterministic undo, not infinite unwind).
    """
    stack: list[str] = field(default_factory=list)

    @property
    def current(self) -> Optional[str]:
        return self.stack[-1] if self.stack else None

    @property
    def previous(self) -> Optional[str]:
        return self.stack[-2] if len(self.stack) >= 2 else None

    def activate(self, name: str) -> str:
        """Record activation of `name`; collapse a no-op re-activate. Returns name."""
        if not name:
            raise ValueError("profile name required")
        if self.current == name:
            return name  # no-op: already active, don't grow the stack
        self.stack.append(name)
        return name

    def rollback(self) -> str:
        """Return to the previous profile (pure). Raises if there's nothing to undo.

        Records the switch-back as a new activation so history stays a faithful log
        of what was active when.
        """
        prev = self.previous
        if prev is None:
            raise RollbackError(
                "no previous profile to roll back to "
                f"(history={self.stack!r})")
        self.stack.append(prev)
        return prev


class RollbackError(RuntimeError):
    """Raised when a rollback is requested but there is no prior profile."""


# ---- Thin effect: apply the switch (DB if up, else local marker) ------------
def _activate_in_db(profile_name: str) -> Optional[str]:
    """Run the same UPDATE the API's activate() does. Returns status, or None if
    the DB is unreachable (so the caller can fall back to the file marker)."""
    try:
        import sys
        sys.path.insert(0, str(_REPO_ROOT))
        from db import connect  # type: ignore
    except Exception:
        return None
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM model_profiles WHERE profile_name=%s",
                        (profile_name,))
            if not cur.fetchone():
                return f"db: unknown profile {profile_name}"
            # Identical to services/api/app.py activate(): single active row.
            cur.execute("UPDATE model_profiles SET active = (profile_name=%s)",
                        (profile_name,))
            conn.commit()
        return f"db: activated {profile_name}"
    except Exception as e:  # pragma: no cover - runtime DB guard
        return None  # treat any DB failure as "DB down" -> file fallback


def _write_marker(profile_name: str, marker: Path = ACTIVE_PROFILE_MARKER) -> str:
    """Degraded path: record the intended active profile in a local marker file."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(profile_name + "\n")
    return f"file: wrote {marker} = {profile_name}"


def apply_activation(profile_name: str, *, marker: Path = ACTIVE_PROFILE_MARKER) -> str:
    """Activate `profile_name` via the API's DB path; fall back to a file marker.

    Returns a human-readable status string (never raises on DB-down — that's the
    whole point of the degraded path).
    """
    status = _activate_in_db(profile_name)
    if status is not None and status.startswith("db: activated"):
        return status
    # DB unreachable OR unknown-profile (still record intent locally).
    return _write_marker(profile_name, marker)


def rollback_active(history: ProfileHistory, *,
                    marker: Path = ACTIVE_PROFILE_MARKER) -> tuple[str, str]:
    """Decide the previous profile (pure) then apply the switch (effect).

    Returns (rolled_back_to, apply_status). Raises RollbackError if there's no
    previous profile to return to.
    """
    target = history.rollback()              # pure decision (may raise)
    status = apply_activation(target, marker=marker)  # thin effect
    return target, status
