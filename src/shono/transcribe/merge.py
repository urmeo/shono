"""Merge per-chunk transcriptions into one globally-timed transcript.

Each chunk was decoded in isolation, so its word timings are chunk-local (0 = the
chunk's window start). Merging means two things: shift every word to global
recording time, and drop the duplicate words where padded/overlapping windows
cover the same speech twice. Word-level timing makes the dedup exact; when a
chunk carries no word timings, the merger falls back to its window times and
skips any chunk fully covered by an earlier one.
"""

from __future__ import annotations

from shono.transcribe.types import ChunkTranscription, Transcript, TranscriptSegment, Word

_EPS = 1e-6


def _global_words(chunk: ChunkTranscription) -> list[Word]:
    off = chunk.window_start_s
    return [Word(w.start_s + off, w.end_s + off, w.word, w.probability) for w in chunk.words]


def merge_transcriptions(chunks: list[ChunkTranscription]) -> Transcript:
    """Merge chunk transcriptions (in any order) into one :class:`Transcript`.

    Overlapping windows are de-duplicated: a word already covered by an earlier
    chunk (its global start falls before the running coverage boundary) is
    dropped, so padded boundaries do not double-print words.
    """
    ordered = sorted(chunks, key=lambda c: c.window_start_s)
    segments: list[TranscriptSegment] = []
    covered_until = float("-inf")

    for chunk in ordered:
        if chunk.words:
            fresh = [w for w in _global_words(chunk) if w.start_s >= covered_until - _EPS]
            if not fresh:
                continue
            text = " ".join(w.word.strip() for w in fresh if w.word.strip())
            if not text:
                continue
            segments.append(TranscriptSegment(fresh[0].start_s, fresh[-1].end_s, text))
            covered_until = max(covered_until, fresh[-1].end_s)
        else:
            # No word timings: dedup by window start, consistent with the word path.
            # A chunk that begins inside already-covered time is dropped rather than
            # re-emitted — its text cannot be trimmed without word timings, and
            # dropping is honest where duplicating would fabricate a repeat. The
            # transcriber supplies word timings in practice, so this is a fallback.
            if chunk.window_start_s < covered_until - _EPS:
                covered_until = max(covered_until, chunk.window_end_s)
                continue
            text = chunk.text.strip()
            covered_until = max(covered_until, chunk.window_end_s)
            if text:
                segments.append(TranscriptSegment(chunk.window_start_s, chunk.window_end_s, text))

    return Transcript(segments=tuple(segments))
