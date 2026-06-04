"""Role-based access control — least-privilege (spec/10 Security: AuthZ, Phase 2/12).

Three roles, additive privileges:
  viewer   — read only (cameras/segments/events/incidents/model-runs)
  reviewer — viewer + approve/reject loss hypotheses (Phase 7 chain-of-custody)
  admin    — reviewer + activate-profile + run retention/backup

The role is carried by the `X-Role` header (the orchestrator can swap this for a
claim from a real token later); `require_role` composes with `require_auth` so a
caller must be both authenticated AND sufficiently privileged.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from .auth import require_auth

ROLES = ("viewer", "reviewer", "admin")

# Action → minimum role rank. Higher rank inherits all lower-rank actions.
_RANK = {"viewer": 0, "reviewer": 1, "admin": 2}

# Least-privilege matrix: action → minimum role required.
_ACTION_MIN_ROLE = {
    "read": "viewer",
    "approve": "reviewer",
    "reject": "reviewer",
    "activate-profile": "admin",
    "retention": "admin",
    "backup": "admin",
}


def allowed(role: str, action: str) -> bool:
    """Pure predicate: may `role` perform `action`? Unknown role/action → False."""
    if role not in _RANK or action not in _ACTION_MIN_ROLE:
        return False
    return _RANK[role] >= _RANK[_ACTION_MIN_ROLE[action]]


def require_role(role: str):
    """Build a FastAPI dependency enforcing >= `role` (and authentication first).

    Usage:  @app.post(..., dependencies=[Depends(require_role("reviewer"))])
    """
    if role not in _RANK:
        raise ValueError(f"unknown role {role!r}")
    needed_rank = _RANK[role]

    def _dep(
        _principal: str = Depends(require_auth),
        x_role: str = Header(default="viewer", alias="X-Role"),
    ) -> str:
        if x_role not in _RANK or _RANK[x_role] < needed_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role '{role}' or higher required",
            )
        return x_role

    return _dep
