"""Demo orchestration: transcribe + diarize + attribute → a readable transcript.

This is what the public demo runs when a user uploads audio: the long-form
transcriber and the diarizer (both injected behind their interfaces) produce a
speaker-attributed transcript, formatted for a human to read. Injecting the
pipeline and diarizer keeps this orchestration testable with fakes — no audio,
no GPU — while the app wires the real Silero/faster-whisper/pyannote stack.
"""

from __future__ import annotations

from shono.diarize.attribute import attribute_speakers
from shono.diarize.diarizer import Diarizer
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
    speakers = diarizer.diarize(audio_path)
    return attribute_speakers(result.transcript, speakers)


def _fmt_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def format_transcript(transcript: Transcript) -> str:
    """Render a transcript as readable lines, merging consecutive same-speaker segments."""
    lines: list[str] = []
    buffer: list[str] = []
    cur_speaker: str | None = None
    cur_start = 0.0
    prev_end = 0.0

    def flush(end: float) -> None:
        if not buffer:
            return
        who = cur_speaker or "Speaker"
        stamp = f"[{_fmt_time(cur_start)}–{_fmt_time(end)}]"
        lines.append(f"**{who}** {stamp}: {' '.join(buffer)}")

    for seg in transcript.segments:
        if seg.speaker != cur_speaker and buffer:
            flush(prev_end)
            buffer = []
        if not buffer:
            cur_speaker = seg.speaker
            cur_start = seg.start_s
        buffer.append(seg.text)
        prev_end = seg.end_s
    if buffer:
        flush(prev_end)
    return "\n\n".join(lines)
