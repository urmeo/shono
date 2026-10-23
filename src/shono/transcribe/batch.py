"""Transcribe validated local audio manifests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from shono.data.audio_paths import validate_manifest_audio
from shono.data.manifest import Manifest
from shono.eval.report import Predictions
from shono.provenance import RunContext, safe_metadata


@runtime_checkable
class ManifestTranscriber(Protocol):
    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str: ...


def run_over_manifest(
    transcriber: ManifestTranscriber,
    manifest: Manifest,
    system: str,
    *,
    audio_root: str | Path = ".",
    run_context: RunContext | None = None,
    duration_of: Callable[[Path], float] | None = None,
) -> Predictions:
    """Validate every input before transcribing the first segment."""
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system label must be non-empty text")
    paths = validate_manifest_audio(manifest, audio_root=audio_root, duration_of=duration_of)
    hypotheses: dict[str, str] = {}
    for path, segment in zip(paths, manifest.segments, strict=True):
        text = transcriber.transcribe(str(path), segment.start_s, segment.duration_s)
        if not isinstance(text, str):
            raise ValueError(f"transcriber returned non-text for segment {segment.id!r}")
        hypotheses[segment.id] = text
    context = run_context.to_dict() if run_context else None
    observer = getattr(transcriber, "observed_runtime_context", None)
    if callable(observer):
        context = context or {}
        context["observed_runtime"] = safe_metadata(observer())
    return Predictions(
        system,
        manifest.name,
        hypotheses,
        run_context=context,
        input_paths=tuple(str(path) for path in paths),
    )
