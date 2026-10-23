"""Shared validation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from numbers import Real
from pathlib import PurePosixPath


def text_value(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty' if empty else 'non-empty'} string")
    return value


def real_number(
    value: object, name: str, *, minimum: float | None = None, allow_string: bool = False
) -> float:
    if allow_string and isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a finite real number") from exc
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def time_bounds(start: object, end: object, name: str, *, allow_empty: bool = False) -> None:
    lo = real_number(start, f"{name} start", minimum=0)
    hi = real_number(end, f"{name} end", minimum=0)
    if hi < lo or (hi == lo and not allow_empty):
        relation = "end >= start" if allow_empty else "end > start"
        raise ValueError(f"{name} must have {relation}")


def probability(value: object, name: str) -> None:
    if real_number(value, name, minimum=0) > 1:
        raise ValueError(f"{name} must be in [0, 1]")


def count(value: object, name: str, *, minimum: int = 1, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def finite_sum(values: Iterable[float], name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    return real_number(result, name)


def audio_reference(value: object) -> str:
    value = text_value(value, "audio")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(p in {"", ".", ".."} for p in value.split("/"))
        or ":" in path.parts[0]
    ):
        raise ValueError("audio must be a relative POSIX path without traversal")
    return value


def json_object(raw: str, location: str) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON field {key!r}")
            out[key] = value
        return out

    try:
        obj = json.loads(raw, object_pairs_hook=pairs)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{location}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{location}: expected a JSON object")
    return obj
