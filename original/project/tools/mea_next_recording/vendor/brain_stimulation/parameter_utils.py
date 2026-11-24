"""Small, reusable helpers for normalizing experiment parameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_params(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy the nested ``params`` mapping from an experiment payload."""
    params = data.get("params") if isinstance(data, Mapping) else None
    return dict(params) if isinstance(params, Mapping) else {}


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int_list(value: Any) -> list[int]:
    """Normalize one or more integer IDs while preserving their first-seen order."""
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.replace(";", ",").replace("\n", ",").replace("\t", ",")
        items: list[int] = []
        for token in normalized.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token and not token.startswith("-"):
                left, right = token.split("-", 1)
                if left.strip().isdigit() and right.strip().isdigit():
                    start, end = int(left), int(right)
                    step = 1 if end >= start else -1
                    items.extend(range(start, end + step, step))
                    continue
            items.append(int(float(token)))
        return list(dict.fromkeys(items))
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(int(float(item)) for item in value if item is not None))
    return [int(float(value))]


def samples_from_us(value_us: float, sample_period_us: float, *, minimum: int = 1) -> int:
    return max(minimum, int(round(value_us / sample_period_us)))


def samples_from_seconds(value_s: float, sample_period_us: float, *, minimum: int = 0) -> int:
    return max(minimum, int(round(value_s * 1_000_000.0 / sample_period_us)))


def samples_to_seconds(samples: int, sample_period_us: float) -> float:
    return samples * sample_period_us / 1_000_000.0
