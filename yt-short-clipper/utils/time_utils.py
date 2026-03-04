"""Small reusable helpers for time calculations."""

from __future__ import annotations


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between min and max."""
    return max(min_value, min(max_value, value))
