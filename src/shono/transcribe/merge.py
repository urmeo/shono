"""Merge timed words and retain ambiguous chunk text with explicit warnings."""

from __future__ import annotations

from shono.transcribe.types import ChunkTranscription, Transcript, TranscriptSegment, Word

_EPS = 1e-6


def _global_words(chunk: ChunkTranscription) -> list[Word]:
    offset = chunk.window_start_s
    return [Word(w.start_s + offset, w.end_s + offset, w.word, w.probability) for w in chunk.words]


def merge_transcriptions(chunks: list[ChunkTranscription]) -> Transcript:
    """Remove fully covered timed words; untimed partial overlaps remain approximate."""
    ordered = sorted(chunks, key=lambda c: c.window_start_s)
    segments: list[TranscriptSegment] = []
    warnings: list[str] = []
    covered_until = float("-inf")
    for chunk in ordered:
        if chunk.words:
            fresh = [
                w
                for w in _global_words(chunk)
                if w.word.strip()
                and (w.end_s > covered_until + _EPS or w.start_s >= covered_until - _EPS)
            ]
            if not fresh:
                continue
            if fresh[0].start_s < covered_until - _EPS:
                warnings.append("Overlapping word boundaries may retain a repeated word.")
            segments.append(
                TranscriptSegment(
                    fresh[0].start_s,
                    max(w.end_s for w in fresh),
                    " ".join(w.word.strip() for w in fresh),
                    timing_precision="word",
                )
            )
            covered_until = max(covered_until, max(w.end_s for w in fresh))
        else:
            text = chunk.text.strip()
            if not text:
                continue
            if chunk.window_end_s <= covered_until + _EPS:
                continue
            if chunk.window_start_s < covered_until - _EPS:
                warnings.append("Overlapping chunk text has no word timings; repeats may remain.")
            segments.append(
                TranscriptSegment(
                    chunk.window_start_s,
                    chunk.window_end_s,
                    text,
                    timing_precision="chunk",
                )
            )
            covered_until = max(covered_until, chunk.window_end_s)
    return Transcript(
        tuple(sorted(segments, key=lambda s: (s.start_s, s.end_s))),
        tuple(dict.fromkeys(warnings)),
    )
