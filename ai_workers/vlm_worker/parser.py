"""Robust VLM-output parsing -> a VLMVerification dict (§8.4).

VLMs rarely emit clean JSON: they wrap it in ```json fences, prepend "Here is the
analysis:", or trail a sentence after the closing brace. The bad-output check
(A11 #8) requires we never crash and always return a schema-valid object — so this
module is a small repair pipeline:

  1. try a straight ``json.loads``;
  2. else strip markdown fences and retry;
  3. else extract the first balanced ``{...}`` span and retry;
  4. else give up -> a schema-valid "unclear" object with parse_success=False.

Whatever happens, ``parse_verification`` returns ``(verification_dict, raw_text,
parse_success)`` and never raises. ``coerce`` then normalizes types (clamp
confidence, lists-from-strings, defaults) so the result always round-trips through
the VLMVerification dataclass — keeping the parse-success rate >95% on well-formed
model output while degrading safely on garbage.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any


from model_plugins.base.schemas import VLMVerification  # noqa: E402

_VALID_VERDICTS = {"suspicious", "benign", "unclear"}
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Return the contents of the first ```...``` block, or the text unchanged."""
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _first_json_object(text: str) -> str | None:
    """Extract the first balanced top-level ``{...}`` substring (brace-matched, so a
    trailing sentence after the JSON doesn't break parsing). Returns None if none."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair_truncated(text: str) -> str | None:
    """Salvage a JSON object truncated mid-output (small VLMs hit the token cap):
    close a dangling string, drop trailing commas, and append the missing closing
    brackets in stack order. Returns the repaired ``{...}`` string or None."""
    start = text.find("{")
    if start < 0:
        return None
    stack: list[str] = []
    in_str = esc = False
    out: list[str] = []
    for c in text[start:]:
        out.append(c)
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]" and stack and stack[-1] == c:
            stack.pop()
    res = "".join(out)
    if in_str:
        res += '"'                       # close the dangling string
    while stack:
        res = res.rstrip().rstrip(",")   # no trailing comma before a closer
        res += stack.pop()
    return res


def _try_load(text: str) -> dict[str, Any] | None:
    """json.loads that returns a dict or None (never raises)."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[str]:
    """Coerce a model field to list[str]: None->[], scalar->[scalar], split a
    semicolon/newline string, pass a list through (stringifying elements)."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    parts = re.split(r"[;\n]+", s)
    return [p.strip() for p in parts if p.strip()]


def _as_confidence(value: Any) -> float:
    """Clamp to [0, 1]; treat a 0-100 style number as a percentage; default 0.5."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5
    if f > 1.0:
        f = f / 100.0 if f <= 100.0 else 1.0
    return max(0.0, min(1.0, f))


def coerce(obj: dict[str, Any]) -> VLMVerification:
    """Normalize a parsed dict into a schema-valid VLMVerification (never raises).
    Unknown verdicts collapse to 'unclear' so downstream metrics stay bounded."""
    verdict = str(obj.get("verdict", "unclear")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        verdict = "unclear"
    return VLMVerification(
        verdict=verdict,
        confidence=_as_confidence(obj.get("confidence")),
        observed_actions=_as_list(obj.get("observed_actions")),
        missing_evidence=_as_list(obj.get("missing_evidence")),
        recommended_next_step=str(obj.get("recommended_next_step", "") or "").strip(),
        structured_event_type=str(obj.get("structured_event_type", "") or "").strip(),
        explanation=str(obj.get("explanation", "") or "").strip(),
    )


def _unparsed(raw_text: str) -> VLMVerification:
    """Schema-valid fallback when no JSON could be recovered from the output."""
    return VLMVerification(
        verdict="unclear",
        confidence=0.0,
        observed_actions=[],
        missing_evidence=["model output was not valid JSON"],
        recommended_next_step="manual_review",
        structured_event_type="parse_error",
        explanation=("Could not parse a JSON verification from the model output. "
                     f"Raw (truncated): {raw_text[:200]}"),
    )


def parse_verification(raw_text: str) -> tuple[dict[str, Any], str, bool]:
    """Parse/repair model text into a VLMVerification dict.

    Returns ``(verification_dict, raw_text, parse_success)``. parse_success is True
    iff a JSON object was recovered (after fence-strip / brace-extraction); the dict
    is always schema-valid (dataclasses.asdict of a coerced VLMVerification)."""
    from dataclasses import asdict

    raw_text = raw_text or ""
    # 1) clean parse, 2) de-fenced parse, 3) first balanced object.
    obj = _try_load(raw_text)
    if obj is None:
        obj = _try_load(_strip_fences(raw_text))
    if obj is None:
        span = _first_json_object(_strip_fences(raw_text))
        if span is not None:
            obj = _try_load(span)
    if obj is None:                                   # 4) repair a truncated object
        repaired = _repair_truncated(_strip_fences(raw_text))
        if repaired is not None:
            obj = _try_load(repaired)

    if obj is None:
        return asdict(_unparsed(raw_text)), raw_text, False
    return asdict(coerce(obj)), raw_text, True
