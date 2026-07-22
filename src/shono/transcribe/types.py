"""The data that flows through the long-form pipeline — audio-free, timestamped.

A recording becomes: speech regions (from VAD) → chunk windows → per-chunk
transcriptions (from Whisper, in chunk-local time) → one merged, globally-timed
:class:`Transcript`. These types carry the timing that the chunker, merger, and
hallucination filter reason about; none of them touch audio, so the logic that
uses them is unit-testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeechSegment:
    """A region the VAD marked as speech, in seconds from the recording start."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(
                f"speech segment must have end > start, got {self.start_s}..{self.end_s}"
            )

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Word:
    """One word with its timing (chunk-local until the merger shifts it to global time)."""

    start_s: float
    end_s: float
    word: str
    probability: float = 1.0


@dataclass(frozen=True)
class ChunkTranscription:
    """Whisper's decoded output for one chunk.

    ``window_start_s``/``window_end_s`` are the chunk's place in the full
    recording; ``words`` carry chunk-local times (0 = chunk start) and are shifted
    to global time by the merger. The confidence signals drive hallucination
    filtering.
    """

    window_start_s: float
    window_end_s: float
    text: str
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class TranscriptSegment:
    """A segment of the final transcript in global recording time."""

    start_s: float
    end_s: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class Transcript:
    """The merged, globally-timed result of transcribing a recording."""

    segments: tuple[TranscriptSegment, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text)

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_s if self.segments else 0.0
