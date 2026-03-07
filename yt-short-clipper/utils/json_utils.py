"""Helpers for robust JSON extraction from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


def _extract_json_candidate(text: str) -> str:
    """Extract the most likely JSON substring from raw model output."""
    stripped = text.strip()
    if not stripped:
        return "{}"

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    starts = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx != -1]
    if not starts:
        return stripped
    start = min(starts)

    end_obj = stripped.rfind("}")
    end_arr = stripped.rfind("]")
    end = max(end_obj, end_arr)
    if end == -1 or end < start:
        return stripped[start:]
    return stripped[start : end + 1]


def safe_json_parse_with_validation(
    text: str, validator: Callable[[Any], bool] | None = None
) -> Any:
    """Parse JSON defensively from model output with optional result validation."""
    candidate = _extract_json_candidate(text)

    for value in (candidate, text):
        try:
            parsed = json.loads(value)
            if validator is None or validator(parsed):
                return parsed
        except Exception:  # noqa: BLE001
            continue

    # Last-ditch fallback: recover clip-like triplets if present.
    clip_pattern = re.compile(
        r'(?is)"start"\s*:\s*"?(?P<start>[^",}\n]+)"?\s*,\s*"end"\s*:\s*"?(?P<end>[^",}\n]+)"?\s*,\s*"hook"\s*:\s*"(?P<hook>[^"]+)"'
    )
    clips = [
        {"start": m.group("start").strip(), "end": m.group("end").strip(), "hook": m.group("hook").strip()}
        for m in clip_pattern.finditer(text)
    ]
    if clips and (validator is None or validator(clips)):
        return clips

    return {}


def safe_json_parse(text: str, validator: Callable[[Any], bool] | None = None) -> Any:
    """Parse JSON and return {} when parsed output fails validation."""
    return safe_json_parse_with_validation(text, validator)
