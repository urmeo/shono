"""Lazy pyannote 4 adapter and interval union/intersection helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from shono.data.audio_paths import validate_audio_window
from shono.diarize.types import SpeakerSegment
from shono.transcribe.chunking import union_speech
from shono.transcribe.types import SpeechSegment


@runtime_checkable
class Diarizer(Protocol):
    def diarize(self, audio_path: str) -> list[SpeakerSegment]: ...


def union_speakers(speakers: list[SpeakerSegment]) -> list[SpeakerSegment]:
    """Merge overlapping intervals for each label, retaining different-speaker overlap."""
    grouped: dict[str, list[SpeechSegment]] = {}
    for turn in speakers:
        if not isinstance(turn, SpeakerSegment):
            raise ValueError("speaker turns must contain SpeakerSegment values")
        grouped.setdefault(turn.speaker, []).append(SpeechSegment(turn.start_s, turn.end_s))
    result = [
        SpeakerSegment(s.start_s, s.end_s, label)
        for label, turns in grouped.items()
        for s in union_speech(turns)
    ]
    return sorted(result, key=lambda s: (s.start_s, s.end_s, s.speaker))


def vad_intersection(
    speakers: list[SpeakerSegment],
    speech: list[SpeechSegment],
) -> list[SpeakerSegment]:
    """Clip turns to disjoint VAD intervals without counting overlap twice."""
    regions = union_speech(speech)
    output: list[SpeakerSegment] = []
    for turn in union_speakers(speakers):
        for region in regions:
            lo, hi = max(turn.start_s, region.start_s), min(turn.end_s, region.end_s)
            if hi > lo:
                output.append(SpeakerSegment(lo, hi, turn.speaker))
    return union_speakers(output)


class PyannoteDiarizer:
    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-community-1",
        hf_token: str | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("pyannote model ID must be non-empty")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("pyannote device must be non-empty")
        self.model = model
        self.hf_token = hf_token
        self.device = device
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            import torch
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(self.model, token=self.hf_token)
            if pipeline is None:
                raise RuntimeError("pyannote pipeline unavailable; check model access and token")
            pipeline.to(torch.device(self.device))
            self._pipeline = pipeline
        return self._pipeline

    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        validate_audio_window(audio_path)
        output = self._load()(audio_path)
        try:
            annotation = output.speaker_diarization
        except AttributeError as exc:
            raise RuntimeError("pyannote 4 speaker_diarization output is required") from exc
        if hasattr(annotation, "itertracks"):
            turns = [(turn, label) for turn, _, label in annotation.itertracks(yield_label=True)]
        else:
            turns = list(annotation)
        return union_speakers(
            [SpeakerSegment(turn.start, turn.end, str(label)) for turn, label in turns]
        )
