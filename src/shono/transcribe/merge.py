"""Merge overlapping text without discarding distinct words or segments."""

from __future__ import annotations

from shono.transcribe.types import ChunkTranscription, Transcript, TranscriptSegment, Word

_EPS = 1e-6


def _global_words(chunk: ChunkTranscription) -> list[Word]:
    offset = chunk.window_start_s
    return [Word(w.start_s + offset, w.end_s + offset, w.word, w.probability) for w in chunk.words]


def merge_transcriptions(chunks: list[ChunkTranscription]) -> Transcript:
    """Remove matching, fully covered text and retain uncertain overlaps."""
    ordered = sorted(chunks, key=lambda c: c.window_start_s)
    entries: list[tuple[TranscriptSegment, int]] = []
    warnings: list[str] = []
    active_chunks: list[ChunkTranscription] = []
    active_words: list[Word] = []
    for index, chunk in enumerate(ordered):
        active_chunks = [
            prior for prior in active_chunks if prior.window_end_s > chunk.window_start_s + _EPS
        ]
        active_words = [
            prior for prior in active_words if prior.end_s > chunk.window_start_s + _EPS
        ]
        if chunk.words:
            fresh = [
                w
                for w in _global_words(chunk)
                if w.word.strip()
                and not any(
                    w.word.strip() == prior.word.strip()
                    and prior.start_s <= w.start_s + _EPS
                    and prior.end_s >= w.end_s - _EPS
                    and w.start_s < prior.end_s - _EPS
                    for prior in active_words
                )
            ]
            if not fresh:
                continue
            if active_chunks:
                warnings.append(
                    "Overlapping decoder segments may retain repeated or conflicting text."
                )
            entries.extend(
                (
                    TranscriptSegment(w.start_s, w.end_s, w.word.strip(), timing_precision="word"),
                    index,
                )
                for w in fresh
            )
            active_words.extend(fresh)
        else:
            text = chunk.text.strip()
            if not text:
                continue
            if any(
                " ".join(prior.text.split()) == " ".join(text.split())
                and prior.window_start_s <= chunk.window_start_s + _EPS
                and prior.window_end_s >= chunk.window_end_s - _EPS
                for prior in active_chunks
            ):
                continue
            if active_chunks:
                warnings.append("Overlapping chunk text has no word timings; repeats may remain.")
            entries.append(
                (
                    TranscriptSegment(
                        chunk.window_start_s,
                        chunk.window_end_s,
                        text,
                        timing_precision="chunk",
                    ),
                    index,
                )
            )
        active_chunks.append(chunk)

    segments: list[TranscriptSegment] = []
    previous_index: int | None = None
    for segment, index in sorted(entries, key=lambda entry: entry[0].start_s):
        if index == previous_index:
            previous = segments[-1]
            segments[-1] = TranscriptSegment(
                previous.start_s,
                max(previous.end_s, segment.end_s),
                f"{previous.text} {segment.text}",
                timing_precision="word",
            )
        else:
            segments.append(segment)
        previous_index = index
    return Transcript(tuple(segments), tuple(dict.fromkeys(warnings)))
