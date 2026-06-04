"""Secret redaction for safe logging (spec/10 Security: secrets never in logs/repo).

RTSP creds, HF tokens, DB passwords, API keys and bearer tokens must never reach
logs, audit records, or error payloads. `redact()` masks them in free text or
(recursively) in dicts/lists, by both key name and value pattern.
"""
from __future__ import annotations

import re
from typing import Any

MASK = "***REDACTED***"

# Dict keys whose values are always secret regardless of content.
_SECRET_KEYS = re.compile(
    r"(pass(word)?|secret|token|api[_-]?key|auth|credential|access[_-]?key|"
    r"private[_-]?key|dsn|database_url|bearer)",
    re.IGNORECASE,
)

# Value patterns to mask anywhere in free text.
_PATTERNS = [
    # URL userinfo creds, e.g. rtsp://user:pass@host or postgresql://u:p@host
    (re.compile(r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<user>[^:/?#\s]+):(?P<pw>[^@/?#\s]+)@"),
     lambda m: f"{m.group('scheme')}{m.group('user')}:{MASK}@"),
    # Bearer tokens in Authorization headers / text
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"), lambda m: f"Bearer {MASK}"),
    # HuggingFace tokens (hf_...) and common sk-/key- prefixes
    (re.compile(r"\bhf_[A-Za-z0-9]{10,}\b"), lambda m: MASK),
    (re.compile(r"\b(sk|pk|key|tok)-[A-Za-z0-9_\-]{8,}\b"), lambda m: MASK),
    # key=value / key: value pairs where the key looks secret. We don't re-mask a
    # value that's already a sentinel (so an earlier Bearer/HF match is preserved).
    (re.compile(r"(?i)\b([\w.-]*(?:password|passwd|secret|token|api[_-]?key|"
                r"apikey|access[_-]?key|auth)[\w.-]*)\s*[=:]\s*(\"[^\"]*\"|'[^']*'|\S+)"),
     lambda m: m.group(0)
     if (MASK in m.group(2) or m.group(2).lower() == "bearer")
     else f"{m.group(1)}={MASK}"),
]


def _redact_text(text: str) -> str:
    out = text
    for rx, repl in _PATTERNS:
        out = rx.sub(repl, out)
    return out


def redact(value: Any) -> Any:
    """Return a redacted copy of `value` (str, dict, list, or scalar).

    - dicts: values under secret-looking keys are fully masked; other string
      values are scanned for embedded secrets; nested structures recursed.
    - lists/tuples: each item redacted.
    - strings: secret patterns masked.
    - other scalars: returned unchanged.
    """
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            if isinstance(k, str) and _SECRET_KEYS.search(k):
                out[k] = MASK
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, (list, tuple)):
        red = [redact(v) for v in value]
        return type(value)(red) if isinstance(value, tuple) else red
    if isinstance(value, str):
        return _redact_text(value)
    return value
