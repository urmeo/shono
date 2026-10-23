"""Validate generated destinations before replacing files or running inference."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path


def resolve_output_path(path: str | Path) -> Path:
    """Resolve a destination and reject paths through a .git directory."""
    original = Path(path)
    try:
        destination = original.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve output path: {exc}") from exc
    if any(part.casefold() == ".git" for part in (*original.parts, *destination.parts)):
        raise ValueError("outputs cannot overwrite Git metadata")
    return destination


def source_files(base: str | Path) -> tuple[Path, ...]:
    """Return input files and linked directories outside outputs and Git metadata."""
    base = Path(base)
    if not base.exists():
        return ()
    files = []

    def fail(error: OSError) -> None:
        raise error

    for directory, names, leaves in os.walk(base, followlinks=False, onerror=fail):
        if Path(directory) == base:
            names[:] = [name for name in names if name not in {"outputs", ".git", ".venv"}]
        else:
            names[:] = [name for name in names if name != ".git"]
        files.extend(
            Path(directory) / name for name in names if (Path(directory) / name).is_symlink()
        )
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
        resolved_root = resolve_output_path(root) if root is not None else None
        resolved = tuple(resolve_output_path(path) for path in raw)
        if len({str(path).casefold() for path in resolved}) != len(resolved):
            raise ValueError("output destinations alias each other")
        sources = tuple(path.resolve() for path in protected)
        for original, destination in zip(raw, resolved, strict=True):
            destination_key = Path(str(destination).casefold())
            if original.is_symlink():
                raise ValueError(f"output destination must not be a symlink: {original}")
            if resolved_root is not None and (
                destination == resolved_root or resolved_root not in destination.parents
            ):
                raise ValueError(f"output destination escapes output root: {original}")
            if original.exists() and (not original.is_file() or original.stat().st_nlink > 1):
                raise ValueError(f"output destination must be a regular unlinked file: {original}")
            for source, resolved_source in zip(protected, sources, strict=True):
                source_key = Path(str(resolved_source).casefold())
                aliases = destination_key == source_key or (
                    source.is_dir() and destination_key.is_relative_to(source_key)
                )
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
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_text_outputs(outputs: Mapping[Path, str]) -> None:
    """Stage related files and restore prior outputs if publication fails."""
    staged = []
    backups: dict[Path, Path | None] = {}
    published = []
    failed_restores = {}
    try:
        for destination, text in outputs.items():
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent, delete=False
            ) as handle:
                staged.append(Path(handle.name))
                handle.write(text)
        for destination in outputs:
            backups[destination] = None
            if destination.exists():
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                    backups[destination] = Path(handle.name)
                shutil.copy2(destination, backups[destination])
        for temporary, destination in zip(staged, outputs, strict=True):
            os.replace(temporary, destination)
            published.append(destination)
    except OSError as publication_error:
        for destination in reversed(published):
            backup = backups[destination]
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except OSError as restore_error:
                failed_restores[destination] = restore_error
        if failed_restores:
            recovery = "; ".join(
                f"{destination}: {restore_error} "
                f"(backup: {backups[destination] or 'no prior file'})"
                for destination, restore_error in sorted(failed_restores.items())
            )
            raise RuntimeError(
                f"output publication failed: {publication_error}; could not restore {recovery}"
            ) from publication_error
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup is not None and destination not in failed_restores:
                backup.unlink(missing_ok=True)
