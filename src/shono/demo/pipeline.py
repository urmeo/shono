"""Format transcripts with explicit timing and dominant-speaker limits."""

from __future__ import annotations

from itertools import groupby

from shono.diarize.attribute import attribute_speakers
from shono.diarize.diarizer import Diarizer, vad_intersection
from shono.transcribe.pipeline import LongFormTranscriber
from shono.transcribe.types import Transcript


def transcribe_recording(
    audio_path: str,
    audio_duration_s: float,
    pipeline: LongFormTranscriber,
    diarizer: Diarizer | None = None,
) -> Transcript:
    """Transcribe ``audio_path`` and, if a diarizer is given, attribute speakers."""
    result = pipeline.transcribe(audio_path, audio_duration_s)
    if diarizer is None:
        return result.transcript
    speakers = vad_intersection(diarizer.diarize(audio_path), list(result.speech))
    return attribute_speakers(result.transcript, speakers)


def _fmt_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def format_transcript(transcript: Transcript) -> str:
    """Merge consecutive segments from the same speaker and include timing limits."""
    lines: list[str] = []
    for speaker, group in groupby(transcript.segments, key=lambda segment: segment.speaker):
        segments = list(group)
        start = segments[0].start_s
        end = max(segment.end_s for segment in segments)
        stamp = f"[{_fmt_time(start)}–{_fmt_time(end)}]"
        text = " ".join(segment.text for segment in segments)
        lines.append(f"**{speaker or 'Speaker'}** {stamp}: {text}")
    notes = list(transcript.warnings)
    if any(s.timing_precision == "chunk" for s in transcript.segments):
        notes.append("Some timings describe complete chunks; word timings were unavailable.")
    if any(s.speaker_precision == "dominant-segment" for s in transcript.segments):
        notes.append("Speaker labels show the dominant speaker per transcript segment.")
    return "\n\n".join([*(f"Note: {note}" for note in dict.fromkeys(notes)), *lines])
