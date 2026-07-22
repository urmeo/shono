"""Plan chunk windows from VAD speech regions — cut at silence, never mid-word.

Whisper degrades and hallucinates on inputs longer than its ~30 s window and on
chunks that begin or end mid-word. The fix is to cut on the *silence* between
speech regions: greedily pack consecutive speech segments into a window until the
next one would exceed ``max_chunk_s``, then start a new window at that silence
gap. A single speech region longer than ``max_chunk_s`` (rare — someone talking
without pause) has no silence to cut on, so it is split on a forced grid, which
this module reports honestly rather than hiding.

Pure geometry over timestamps — no audio — so every edge is unit-tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shono.transcribe.types import SpeechSegment


@dataclass(frozen=True)
class Chunk:
    """A window to transcribe: ``[start_s, end_s]`` in the full recording."""

    start_s: float
    end_s: float
    forced_split: bool = False  # True when cut mid-region for lack of a silence gap

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _split_long_segment(seg: SpeechSegment, max_chunk_s: float) -> list[Chunk]:
    n = math.ceil(seg.duration_s / max_chunk_s)
    step = seg.duration_s / n
    return [
        Chunk(seg.start_s + i * step, seg.start_s + (i + 1) * step, forced_split=True)
        for i in range(n)
    ]


def plan_chunks(
    segments: list[SpeechSegment],
    *,
    max_chunk_s: float = 28.0,
    pad_s: float = 0.0,
) -> list[Chunk]:
    """Group speech ``segments`` into chunk windows no longer than ``max_chunk_s``.

    Windows are cut at the silence between speech regions. A region longer than
    ``max_chunk_s`` on its own is split on an even grid and each piece is flagged
    ``forced_split`` (a hallucination-risk boundary the caller may treat with
    care). ``pad_s`` extends each window slightly past the speech to avoid
    clipping onsets/offsets. ``segments`` need not be sorted.
    """
    if max_chunk_s <= 0:
        raise ValueError(f"max_chunk_s must be > 0, got {max_chunk_s}")
    ordered = sorted(segments, key=lambda s: s.start_s)
    chunks: list[Chunk] = []
    cur_start: float | None = None
    cur_end: float | None = None

    for seg in ordered:
        if seg.duration_s > max_chunk_s:
            if cur_start is not None:
                chunks.append(Chunk(cur_start, cur_end))
                cur_start = cur_end = None
            chunks.extend(_split_long_segment(seg, max_chunk_s))
            continue
        if cur_start is None:
            cur_start, cur_end = seg.start_s, seg.end_s
        elif seg.end_s - cur_start <= max_chunk_s:
            cur_end = max(cur_end, seg.end_s)  # never shrink on a nested segment
        else:
            chunks.append(Chunk(cur_start, cur_end))
            cur_start, cur_end = seg.start_s, seg.end_s

    if cur_start is not None:
        chunks.append(Chunk(cur_start, cur_end))

    if pad_s:
        chunks = [
            Chunk(max(0.0, c.start_s - pad_s), c.end_s + pad_s, c.forced_split) for c in chunks
        ]
    return chunks
