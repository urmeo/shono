"""Validate generated destinations before replacing files or running inference."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path


def source_files(base: str | Path) -> tuple[Path, ...]:
    """Return project inputs outside the generated outputs and Git directories."""
    base = Path(base)
    files = []
    for directory, names, leaves in os.walk(base, followlinks=False):
        if Path(directory) == base:
            names[:] = [name for name in names if name not in {"outputs", ".git", ".venv"}]
        else:
            names[:] = [name for name in names if name != ".git"]
        files.extend(Path(directory) / leaf for leaf in leaves)
    return tuple(files)


def validate_output_destinations(
    destinations: Iterable[str | Path],
    *,
    protected_paths: Iterable[str | Path] = (),
    output_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Reject links, aliases and input collisions; ordinary output files may be replaced."""
    try:
        raw = tuple(Path(path) for path in destinations)
        protected = tuple(Path(path) for path in protected_paths)
        root = Path(output_root) if output_root is not None else None
        if root is not None and root.is_symlink():
            raise ValueError("output root must not be a symlink")
        resolved_root = root.resolve() if root is not None else None
        resolved = tuple(path.resolve() for path in raw)
        if len({str(path).casefold() for path in resolved}) != len(resolved):
            raise ValueError("output destinations alias each other")
        sources = tuple(path.resolve() for path in protected)
        for original, destination in zip(raw, resolved, strict=True):
            if original.is_symlink():
                raise ValueError(f"output destination must not be a symlink: {original}")
            if resolved_root is not None and (
                destination == resolved_root or resolved_root not in destination.parents
            ):
                raise ValueError(f"output destination escapes output root: {original}")
            if original.exists() and (not original.is_file() or original.stat().st_nlink > 1):
                raise ValueError(f"output destination must be a regular unlinked file: {original}")
            for source, resolved_source in zip(protected, sources, strict=True):
                aliases = str(destination).casefold() == str(resolved_source).casefold()
                if not aliases and original.exists() and source.exists():
                    aliases = os.path.samefile(original, source)
                if aliases:
                    raise ValueError(f"output would overwrite input {source}")
        for index, original in enumerate(raw):
            for other in raw[:index]:
                if original.exists() and other.exists() and os.path.samefile(original, other):
                    raise ValueError("output destinations alias each other")
        return resolved
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve output paths: {exc}") from exc


def write_text_atomic(path: str | Path, text: str) -> None:
    """Stage a complete text payload in the destination filesystem."""
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
