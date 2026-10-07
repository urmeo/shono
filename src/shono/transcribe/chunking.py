"""Plan bounded VAD windows, including padding in the maximum duration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from shono.transcribe._validation import count, finite_real, time_bounds
from shono.transcribe.types import SpeechSegment


@dataclass(frozen=True)
class Chunk:
    start_s: float
    end_s: float
    forced_split: bool = False

    def __post_init__(self) -> None:
        time_bounds(self.start_s, self.end_s, "chunk")
        if not isinstance(self.forced_split, bool):
            raise ValueError("forced_split must be boolean")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def union_speech(segments: list[SpeechSegment]) -> list[SpeechSegment]:
    """Return ordered, disjoint speech intervals without counting overlaps twice."""
    if any(not isinstance(s, SpeechSegment) for s in segments):
        raise ValueError("speech regions must contain SpeechSegment values")
    merged: list[SpeechSegment] = []
    for seg in sorted(segments, key=lambda s: (s.start_s, s.end_s)):
        if merged and seg.start_s <= merged[-1].end_s:
            merged[-1] = SpeechSegment(merged[-1].start_s, max(merged[-1].end_s, seg.end_s))
        else:
            merged.append(seg)
    return merged


def plan_chunks(
    segments: list[SpeechSegment],
    *,
    max_chunk_s: float = 28.0,
    pad_s: float = 0.0,
    audio_duration_s: float | None = None,
    max_chunks: int = 100_000,
) -> list[Chunk]:
    """Pack speech into windows; long regions require flagged grid splits."""
    maximum = finite_real(max_chunk_s, "max_chunk_s", minimum=0)
    padding = finite_real(pad_s, "pad_s", minimum=0)
    count(max_chunks, "max_chunks")
    core = maximum - 2 * padding
    if not math.isfinite(core) or core <= 0:
        raise ValueError("need 0 <= 2 * pad_s < max_chunk_s")
    duration = None
    if audio_duration_s is not None:
        duration = finite_real(audio_duration_s, "audio_duration_s", minimum=0)
        if duration <= 0:
            raise ValueError("audio_duration_s must be > 0")
    ordered = union_speech(segments)
    if duration is not None and any(s.end_s > duration + 1e-6 for s in ordered):
        raise ValueError("VAD speech regions extend beyond the recording")
    chunks: list[Chunk] = []
    current: tuple[float, float] | None = None

    def append(start: float, end: float, forced: bool = False) -> None:
        if len(chunks) >= max_chunks:
            raise ValueError("planned chunk count exceeds max_chunks")
        lo = max(0.0, start - padding)
        hi = end + padding
        if duration is not None:
            hi = min(hi, duration)
        chunk = Chunk(lo, hi, forced)
        if chunk.duration_s > maximum + 1e-6:
            raise ValueError("chunk geometry cannot represent the requested maximum")
        chunks.append(chunk)

    for seg in ordered:
        if seg.duration_s > core:
            if current is not None:
                append(*current)
                current = None
            ratio = seg.duration_s / core
            if not math.isfinite(ratio) or ratio > max_chunks - len(chunks):
                raise ValueError("planned chunk count exceeds max_chunks")
            n = math.ceil(ratio)
            step = seg.duration_s / n
            for i in range(n):
                lo = seg.start_s + i * step
                hi = seg.end_s if i == n - 1 else seg.start_s + (i + 1) * step
                append(lo, hi, True)
        elif current is None:
            current = (seg.start_s, seg.end_s)
        elif seg.end_s - current[0] <= core:
            current = (current[0], max(current[1], seg.end_s))
        else:
            append(*current)
            current = (seg.start_s, seg.end_s)
    if current is not None:
        append(*current)
    return chunks
