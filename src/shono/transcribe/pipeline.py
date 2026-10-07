"""Run VAD, bounded chunks, confidence filtering and transcript merging."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from shono.data.audio_paths import validate_audio_window
from shono.transcribe._validation import count, finite_real, probability
from shono.transcribe.chunking import plan_chunks, union_speech
from shono.transcribe.convert import validate_ct2_checkpoint
from shono.transcribe.hallucination import FilterResult, HallucinationConfig, filter_hallucinations
from shono.transcribe.merge import merge_transcriptions
from shono.transcribe.types import ChunkTranscription, SpeechSegment, Transcript, Word

TARGET_SAMPLE_RATE = 16_000


@runtime_checkable
class VoiceActivityDetector(Protocol):
    def detect(self, audio_path: str) -> list[SpeechSegment]: ...


@runtime_checkable
class ChunkTranscriber(Protocol):
    def transcribe_chunk(
        self, audio_path: str, start_s: float, end_s: float
    ) -> ChunkTranscription: ...


def real_time_factor(processing_s: float, audio_s: float) -> float:
    elapsed = finite_real(processing_s, "processing_s", minimum=0)
    duration = finite_real(audio_s, "audio_s", minimum=0)
    if duration <= 0:
        raise ValueError("audio_s must be > 0")
    return finite_real(elapsed / duration, "real-time factor", minimum=0)


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: Transcript
    audio_duration_s: float
    processing_s: float
    n_chunks: int
    filtering: FilterResult
    speech: tuple[SpeechSegment, ...] = ()
    n_forced_splits: int = 0

    @property
    def rtf(self) -> float:
        return real_time_factor(self.processing_s, self.audio_duration_s)


class LongFormTranscriber:
    def __init__(
        self,
        vad: VoiceActivityDetector,
        transcriber: ChunkTranscriber,
        *,
        max_chunk_s: float = 28.0,
        pad_s: float = 0.2,
        hallucination: HallucinationConfig | None = None,
        max_chunks: int = 100_000,
    ) -> None:
        plan_chunks([], max_chunk_s=max_chunk_s, pad_s=pad_s, max_chunks=max_chunks)
        if max_chunk_s > 30:
            raise ValueError("Whisper windows must be <= 30 s including padding")
        self.vad = vad
        self.transcriber = transcriber
        self.max_chunk_s = max_chunk_s
        self.pad_s = pad_s
        self.max_chunks = max_chunks
        self.hallucination = hallucination or HallucinationConfig()

    def transcribe(
        self,
        audio_path: str,
        audio_duration_s: float,
        *,
        now: Callable[[], float] = perf_counter,
    ) -> TranscriptionResult:
        """Measure elapsed time, including any lazy loads during this call."""
        duration = finite_real(audio_duration_s, "audio_duration_s", minimum=0)
        if duration <= 0:
            raise ValueError("audio_duration_s must be > 0")
        started = finite_real(now(), "start clock", minimum=0)
        speech = union_speech(self.vad.detect(audio_path))
        chunks = plan_chunks(
            speech,
            max_chunk_s=self.max_chunk_s,
            pad_s=self.pad_s,
            audio_duration_s=duration,
            max_chunks=self.max_chunks,
        )
        transcriptions = [
            self.transcriber.transcribe_chunk(audio_path, c.start_s, c.end_s) for c in chunks
        ]
        if any(
            t.window_start_s != c.start_s or t.window_end_s != c.end_s
            for t, c in zip(transcriptions, chunks, strict=True)
        ):
            raise ValueError("transcriber returned a different audio window")
        filtered = filter_hallucinations(transcriptions, self.hallucination)
        transcript = merge_transcriptions(list(filtered.kept))
        processing_s = finite_real(now(), "end clock", minimum=0) - started
        real_time_factor(processing_s, duration)
        return TranscriptionResult(
            transcript,
            duration,
            processing_s,
            len(chunks),
            filtered,
            tuple(speech),
            sum(c.forced_split for c in chunks),
        )


class LongFormPipelineTranscriber:
    """Adapt complete recordings to the manifest transcriber protocol."""

    def __init__(self, pipeline: LongFormTranscriber) -> None:
        self.pipeline = pipeline

    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str:
        if start_s is not None and finite_real(start_s, "start_s", minimum=0) != 0:
            raise ValueError("long-form inference requires a complete recording, start_s=0")
        actual = validate_audio_window(audio_path, start_s, duration_s, whole_file=True)
        return self.pipeline.transcribe(audio_path, actual).transcript.text


class SileroVAD:
    """Lazy cached Silero torch-hub VAD; its first real use may download weights."""

    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 250) -> None:
        probability(threshold, "VAD threshold")
        count(min_speech_ms, "min_speech_ms")
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self._model = None
        self._utils = None

    def detect(self, audio_path: str) -> list[SpeechSegment]:
        validate_audio_window(audio_path)
        if self._model is None:
            import torch

            self._model, self._utils = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                trust_repo=True,
            )
        get_ts, _, read_audio, *_ = self._utils
        wav = read_audio(audio_path, sampling_rate=TARGET_SAMPLE_RATE)
        stamps = get_ts(
            wav,
            self._model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            sampling_rate=TARGET_SAMPLE_RATE,
        )
        return union_speech(
            [
                SpeechSegment(s["start"] / TARGET_SAMPLE_RATE, s["end"] / TARGET_SAMPLE_RATE)
                for s in stamps
            ]
        )


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "bn",
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _load(self):
        if self._model is None:
            model_path = validate_ct2_checkpoint(self.model_dir)
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                str(model_path), device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe_chunk(self, audio_path: str, start_s: float, end_s: float) -> ChunkTranscription:
        from shono.transcribe._validation import time_bounds

        time_bounds(start_s, end_s, "transcription window")
        if end_s - start_s > 30:
            raise ValueError("Whisper audio windows must be <= 30 s")
        actual = validate_audio_window(audio_path, start_s, end_s - start_s)
        import librosa
        import numpy as np

        audio, _ = librosa.load(
            audio_path, sr=TARGET_SAMPLE_RATE, offset=start_s, duration=actual, mono=True
        )
        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError("decoded audio must contain finite samples")
        segments, _info = self._load().transcribe(
            audio,
            language=self.language,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        texts, words, logprobs, comps, nosp = [], [], [], [], []
        for seg in segments:
            texts.append(seg.text)
            logprobs.append(seg.avg_logprob)
            comps.append(seg.compression_ratio)
            nosp.append(seg.no_speech_prob)
            words.extend(Word(w.start, w.end, w.word, w.probability) for w in (seg.words or []))

        def mean(values, default):
            return sum(values) / len(values) if values else default

        return ChunkTranscription(
            start_s,
            end_s,
            " ".join(t.strip() for t in texts).strip(),
            mean(logprobs, 0),
            mean(comps, 1),
            mean(nosp, 0),
            tuple(words),
        )
