"""Validate finite real values and time bounds."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real


def finite_real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(
            f"{name} must be finite and >= {minimum}"
            if minimum is not None
            else f"{name} must be finite"
        )
    return result


def time_bounds(start: object, end: object, name: str, *, allow_empty: bool = False) -> None:
    lo = finite_real(start, f"{name} start", minimum=0)
    hi = finite_real(end, f"{name} end", minimum=0)
    if hi < lo or (hi == lo and not allow_empty):
        relation = "end >= start" if allow_empty else "end > start"
        raise ValueError(f"{name} must have {relation}")


def probability(value: object, name: str) -> None:
    if finite_real(value, name, minimum=0) > 1:
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
    return finite_real(result, name)
