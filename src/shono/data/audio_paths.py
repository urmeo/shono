"""Resolve local audio references and validate time windows before inference."""

from __future__ import annotations

import hashlib
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from shono.data.validation import audio_reference, real_number

if TYPE_CHECKING:
    from shono.data.manifest import Manifest

DurationFn = Callable[[Path], float]
WINDOW_TOLERANCE_S = 0.001


def resolved_path(path: str | Path) -> Path:
    try:
        return Path(path).resolve()
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"cannot resolve path {path}: {exc}") from exc


def validate_audio_paths(manifest: Manifest, audio_root: str | Path) -> tuple[Path, ...]:
    """Return existing files in segment order, confined to the explicit root."""
    root = resolved_path(audio_root)
    if not root.is_dir():
        raise ValueError(f"audio_root is not an existing directory: {root}")
    paths = []
    for seg in manifest.segments:
        path = resolved_path(root / audio_reference(seg.audio))
        if not path.is_relative_to(root):
            raise ValueError(f"segment {seg.id!r}: audio escapes audio_root")
        if not path.is_file():
            raise ValueError(f"segment {seg.id!r}: missing audio file {path}")
        paths.append(path)
    return tuple(paths)


def audio_duration(path: Path) -> float:
    """Read file duration without loading model weights."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as source:
                return source.getnframes() / source.getframerate()
        except (wave.Error, EOFError):
            pass
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ValueError("audio metadata requires soundfile; install shono[audio]") from exc
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"cannot read audio metadata: {path}") from exc
    return info.frames / info.samplerate


def validate_audio_window(
    path: str | Path,
    start_s: float | None = None,
    duration_s: float | None = None,
    *,
    duration_of: DurationFn | None = None,
    whole_file: bool | None = None,
    tolerance_s: float = WINDOW_TOLERANCE_S,
) -> float:
    """Return selected seconds. None start denotes a whole-file reference.

    Explicit zero denotes an aligned crop. None duration selects the remaining
    file. Metadata tolerance is 1 ms by default; it never permits empty crops.
    """
    path = resolved_path(path)
    if not path.is_file():
        raise ValueError(f"missing audio file: {path}")
    tolerance = real_number(tolerance_s, "tolerance_s")
    if tolerance < 0 or tolerance > 0.01:
        raise ValueError("tolerance_s must be between 0 and 0.01 seconds")
    if whole_file is not None and type(whole_file) is not bool:
        raise ValueError("whole_file must be a boolean or None")
    whole = start_s is None if whole_file is None else whole_file
    start = 0.0 if start_s is None else real_number(start_s, "start_s")
    if start < 0:
        raise ValueError("start_s must be nonnegative")
    duration = None if duration_s is None else real_number(duration_s, "duration_s")
    if duration is not None and duration <= 0:
        raise ValueError("duration_s must be positive")
    actual = real_number((duration_of or audio_duration)(path), "actual audio duration")
    if actual <= 0 or start >= actual:
        raise ValueError("audio window is empty or starts at/past EOF")
    if whole:
        if start != 0:
            raise ValueError("whole-file reference must start at zero")
        if duration is not None and abs(duration - actual) > tolerance:
            raise ValueError("whole-file duration metadata does not match actual audio duration")
        return actual
    selected = actual - start if duration is None else duration
    if selected > actual - start + tolerance:
        raise ValueError("audio window extends past EOF")
    return min(selected, actual - start)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest_audio(
    manifest: Manifest, audio_root: str | Path, *, duration_of: DurationFn | None = None
) -> tuple[Path, ...]:
    """Validate every path, window and declared complete-file checksum."""
    paths = validate_audio_paths(manifest, audio_root)
    durations: dict[Path, float] = {}
    hashes: dict[Path, str] = {}
    for seg, path in zip(manifest.segments, paths, strict=True):
        if path not in durations:
            durations[path] = (duration_of or audio_duration)(path)
        validate_audio_window(path, seg.start_s, seg.duration_s, duration_of=lambda p: durations[p])
        if seg.audio_sha256 is not None:
            if path not in hashes:
                hashes[path] = file_sha256(path)
            if hashes[path] != seg.audio_sha256:
                raise ValueError(f"segment {seg.id!r}: audio_sha256 does not match file bytes")
    return paths
