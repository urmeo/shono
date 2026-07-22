"""Diarization behind an interface — pyannote is one implementation, not the contract.

The pipeline depends on the :class:`Diarizer` protocol, so tests drive it with a
fake and the gated pyannote model is swapped in for real runs. ``vad_intersection``
is the WhisperAlign trick that cut Bengali DER sharply: clip the diarizer's speaker
turns to the VAD's speech regions, so labels never bleed across the silence where
boundary hallucinations live. It is pure interval math and fully tested.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from shono.diarize.types import SpeakerSegment
from shono.transcribe.types import SpeechSegment


@runtime_checkable
class Diarizer(Protocol):
    """Assigns speaker turns to a recording."""

    def diarize(self, audio_path: str) -> list[SpeakerSegment]: ...


def vad_intersection(
    speakers: list[SpeakerSegment], speech: list[SpeechSegment]
) -> list[SpeakerSegment]:
    """Clip speaker turns to the VAD speech regions, dropping the silent overhang.

    A speaker turn is kept only where it overlaps detected speech; each overlap
    becomes its own (possibly shorter) segment. This removes the boundary regions
    over silence where diarizers and ASR both hallucinate.
    """
    speech_sorted = sorted(speech, key=lambda s: s.start_s)
    out: list[SpeakerSegment] = []
    for turn in speakers:
        for region in speech_sorted:
            lo = max(turn.start_s, region.start_s)
            hi = min(turn.end_s, region.end_s)
            if hi > lo:
                out.append(SpeakerSegment(lo, hi, turn.speaker))
    return sorted(out, key=lambda s: (s.start_s, s.speaker))


class PyannoteDiarizer:
    """pyannote speaker-diarization-community-1 (HF-gated; torch/GPU; lazy import)."""

    def __init__(self, model: str = "pyannote/speaker-diarization-community-1",
                 hf_token: str | None = None) -> None:
        self.model = model
        self.hf_token = hf_token
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(self.model, use_auth_token=self.hf_token)
        return self._pipeline

    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        annotation = self._load()(audio_path)
        return [
            SpeakerSegment(turn.start, turn.end, str(label))
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
