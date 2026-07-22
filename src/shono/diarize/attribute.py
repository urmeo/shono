"""Attach speakers to a transcript — who said each segment.

Given a transcript (from the long-form pipeline) and speaker turns (from the
diarizer), label each transcript segment with the speaker whose turns overlap it
most. Pure interval overlap, so the speaker-attributed transcript that decision
S4 asks for is testable without audio or a model.
"""

from __future__ import annotations

from shono.diarize.types import SpeakerSegment
from shono.transcribe.types import Transcript, TranscriptSegment


def _dominant_speaker(seg: TranscriptSegment, speakers: list[SpeakerSegment]) -> str | None:
    overlap: dict[str, float] = {}
    for turn in speakers:
        lo = max(seg.start_s, turn.start_s)
        hi = min(seg.end_s, turn.end_s)
        if hi > lo:
            overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + (hi - lo)
    if not overlap:
        return None
    # max overlap; ties broken by label for determinism
    return max(sorted(overlap), key=lambda spk: overlap[spk])


def attribute_speakers(transcript: Transcript, speakers: list[SpeakerSegment]) -> Transcript:
    """Return ``transcript`` with each segment's ``speaker`` set by max turn overlap."""
    labelled = tuple(
        TranscriptSegment(s.start_s, s.end_s, s.text, speaker=_dominant_speaker(s, speakers))
        for s in transcript.segments
    )
    return Transcript(segments=labelled)
