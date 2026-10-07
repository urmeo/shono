"""Validated chunk-local words and recording-local transcripts."""

from __future__ import annotations

from dataclasses import dataclass, field

from shono.transcribe._validation import finite_real, probability, time_bounds


@dataclass(frozen=True)
class SpeechSegment:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        time_bounds(self.start_s, self.end_s, "speech segment")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Word:
    start_s: float
    end_s: float
    word: str
    probability: float = 1.0

    def __post_init__(self) -> None:
        time_bounds(self.start_s, self.end_s, "word", allow_empty=True)
        probability(self.probability, "word probability")
        if not isinstance(self.word, str):
            raise ValueError("word text must be a string")


@dataclass(frozen=True)
class ChunkTranscription:
    window_start_s: float
    window_end_s: float
    text: str
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0
    words: tuple[Word, ...] = ()

    def __post_init__(self) -> None:
        time_bounds(self.window_start_s, self.window_end_s, "transcription window")
        if not isinstance(self.text, str):
            raise ValueError("transcription text must be a string")
        finite_real(self.avg_logprob, "avg_logprob")
        finite_real(self.compression_ratio, "compression_ratio", minimum=0)
        probability(self.no_speech_prob, "no_speech_prob")
        duration = self.window_end_s - self.window_start_s
        if any(not isinstance(w, Word) or w.end_s > duration + 1e-6 for w in self.words):
            raise ValueError("word times must lie within the transcription window")
        if any(a.start_s > b.start_s for a, b in zip(self.words, self.words[1:], strict=False)):
            raise ValueError("word times must be ordered")


@dataclass(frozen=True)
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str
    speaker: str | None = None
    timing_precision: str = "unknown"
    speaker_precision: str | None = None

    def __post_init__(self) -> None:
        time_bounds(self.start_s, self.end_s, "transcript segment", allow_empty=True)
        if not isinstance(self.text, str):
            raise ValueError("transcript text must be a string")
        if self.speaker is not None and (not isinstance(self.speaker, str) or not self.speaker):
            raise ValueError("speaker label must be non-empty text")
        if self.timing_precision not in {"unknown", "word", "chunk"}:
            raise ValueError("unknown timing precision")
        if self.speaker_precision not in {None, "dominant-segment"}:
            raise ValueError("unknown speaker precision")


@dataclass(frozen=True)
class Transcript:
    segments: tuple[TranscriptSegment, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(s, TranscriptSegment) for s in self.segments):
            raise ValueError("transcript must contain transcript segments")
        if any(not isinstance(note, str) for note in self.warnings):
            raise ValueError("transcript warnings must contain text")
        if any(
            a.start_s > b.start_s for a, b in zip(self.segments, self.segments[1:], strict=False)
        ):
            raise ValueError("transcript segments must be ordered")

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text)

    @property
    def duration_s(self) -> float:
        return max((s.end_s for s in self.segments), default=0.0)
