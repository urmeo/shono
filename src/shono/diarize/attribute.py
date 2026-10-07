"""Assign the dominant speaker to each transcript segment."""

from __future__ import annotations

from dataclasses import replace

from shono.diarize.diarizer import union_speakers
from shono.diarize.types import SpeakerSegment
from shono.transcribe.types import Transcript, TranscriptSegment


def _dominant_speaker(seg: TranscriptSegment, speakers: list[SpeakerSegment]) -> str | None:
    overlap: dict[str, float] = {}
    for turn in speakers:
        lo, hi = max(seg.start_s, turn.start_s), min(seg.end_s, turn.end_s)
        if hi > lo:
            overlap[turn.speaker] = overlap.get(turn.speaker, 0) + hi - lo
    return max(sorted(overlap), key=lambda speaker: overlap[speaker]) if overlap else None


def attribute_speakers(transcript: Transcript, speakers: list[SpeakerSegment]) -> Transcript:
    turns = union_speakers(speakers)
    labelled = []
    for segment in transcript.segments:
        label = _dominant_speaker(segment, turns)
        labelled.append(
            replace(segment, speaker=label, speaker_precision="dominant-segment" if label else None)
        )
    return Transcript(tuple(labelled), transcript.warnings)
