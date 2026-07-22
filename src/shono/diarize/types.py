"""Speaker-labelled time segments — the currency of diarization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerSegment:
    """A span of speech attributed to one speaker, in seconds from the recording start."""

    start_s: float
    end_s: float
    speaker: str

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(
                f"speaker segment must have end > start, got {self.start_s}..{self.end_s}"
            )
        if not self.speaker:
            raise ValueError("speaker label must be non-empty")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s
