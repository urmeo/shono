"""Validation shared by metadata readers."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import PurePosixPath


def text_value(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty' if empty else 'non-empty'} string")
    return value


def real_number(value: object, name: str, *, allow_string: bool = False) -> float:
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
    return result


def audio_reference(value: object) -> str:
    value = text_value(value, "audio")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or ":" in path.parts[0]
        or any(p in {"", ".", ".."} for p in value.split("/"))
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
