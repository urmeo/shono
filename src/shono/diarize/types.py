"""Validated recording-local speaker intervals."""

from __future__ import annotations

from dataclasses import dataclass

from shono.transcribe._validation import time_bounds


@dataclass(frozen=True)
class SpeakerSegment:
    start_s: float
    end_s: float
    speaker: str

    def __post_init__(self) -> None:
        time_bounds(self.start_s, self.end_s, "speaker segment")
        if not isinstance(self.speaker, str) or not self.speaker.strip():
            raise ValueError("speaker label must be non-empty text")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s
