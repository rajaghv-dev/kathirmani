"""API authentication — Bearer token / API key (spec/10 Security: AuthN, Phase 2/12).

The platform API must never serve footage anonymously. This provides a FastAPI
dependency (`require_auth`) that checks a Bearer token / `X-API-Key` against the
`PLATFORM_API_KEY` env var. It is importable so the orchestrator can add it to
`services/api/app.py` without this module depending on the app.

Secrets come from the environment only (never hard-coded, never logged).
"""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status

ENV_KEY = "PLATFORM_API_KEY"


def _configured_key() -> str | None:
    """The expected API key from the environment (None if unset → API is closed)."""
    key = os.environ.get(ENV_KEY)
    return key or None


def issue_token() -> str:
    """Test/dev helper: return the configured key so tests can authenticate.

    Raises if no key is configured — tests should set PLATFORM_API_KEY first
    (e.g. via monkeypatch). This never invents a secret of its own.
    """
    key = _configured_key()
    if key is None:
        raise RuntimeError(f"{ENV_KEY} is not set; cannot issue a token")
    return key


def _extract(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull the presented credential from either header form."""
    if authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return authorization.strip()  # tolerate a bare token
    if x_api_key:
        return x_api_key.strip()
    return None


def verify(presented: str | None) -> bool:
    """Constant-time compare of a presented credential against the configured key."""
    expected = _configured_key()
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented, expected)


def require_auth(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """FastAPI dependency: 401 unless a valid Bearer token / API key is presented.

    Returns the (opaque) principal id on success. We do NOT echo the token in the
    error or anywhere else — only a generic message, so secrets never leak to logs.
    """
    presented = _extract(authorization, x_api_key)
    if not verify(presented):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return "api-key"  # principal label; real subject identity is out of scope here
