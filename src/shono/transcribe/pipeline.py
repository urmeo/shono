"""Run VAD, bounded chunks, confidence filtering and transcript merging."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from shono.checkpoints import hash_checkpoint
from shono.data.audio_paths import validate_audio_window
from shono.transcribe._validation import count, finite_real, probability, time_bounds
from shono.transcribe.chunking import plan_chunks, union_speech
from shono.transcribe.hallucination import FilterResult, HallucinationConfig, filter_hallucinations
from shono.transcribe.merge import merge_transcriptions
from shono.transcribe.types import ChunkTranscription, SpeechSegment, Transcript, Word

TARGET_SAMPLE_RATE = 16_000
_SILERO_REPO = "snakers4/silero-vad:1e261b036686cd0017d500ee96acd1c4ba572a9d"


@runtime_checkable
class VoiceActivityDetector(Protocol):
    def detect(self, audio_path: str) -> list[SpeechSegment]: ...


@runtime_checkable
class ChunkTranscriber(Protocol):
    def transcribe_chunk(
        self, audio_path: str, start_s: float, end_s: float
    ) -> tuple[ChunkTranscription, ...]:
        """Return decoder segments within the requested recording window."""
        ...


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

    def observed_runtime_context(self) -> dict:
        observer = getattr(self.vad, "observed_runtime_context", None)
        decoder_observer = getattr(self.transcriber, "observed_runtime_context", None)
        return {
            "vad": observer() if callable(observer) else "unknown",
            "decoder": decoder_observer() if callable(decoder_observer) else "unknown",
            "hallucination": asdict(self.hallucination),
            "max_chunk_s": self.max_chunk_s,
            "pad_s": self.pad_s,
            "max_chunks": self.max_chunks,
        }

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
        transcriptions: list[ChunkTranscription] = []
        for chunk in chunks:
            segments = self.transcriber.transcribe_chunk(audio_path, chunk.start_s, chunk.end_s)
            if any(
                s.window_start_s < chunk.start_s - 1e-6 or s.window_end_s > chunk.end_s + 1e-6
                for s in segments
            ):
                raise ValueError("decoder segment lies outside its audio window")
            transcriptions.extend(segments)
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

    def observed_runtime_context(self) -> dict:
        return self.pipeline.observed_runtime_context()

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

    def observed_runtime_context(self) -> dict:
        repository, revision = _SILERO_REPO.split(":", 1)
        return {
            "repository": repository,
            "revision": revision,
            "entrypoint": "silero_vad",
            "threshold": self.threshold,
            "min_speech_ms": self.min_speech_ms,
            "sampling_rate": TARGET_SAMPLE_RATE,
        }

    def detect(self, audio_path: str) -> list[SpeechSegment]:
        validate_audio_window(audio_path)
        import librosa
        import numpy as np
        import torch

        audio, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
        if len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError("decoded audio must contain finite samples")
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        if self._model is None:
            self._model, self._utils = torch.hub.load(
                _SILERO_REPO,
                "silero_vad",
                trust_repo=True,
                # Hub validation only recognizes current branch/tag heads, not old commits.
                skip_validation=True,
            )
        get_ts = self._utils[0]
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
        self._checkpoint = None
        self._decoding_options = None

    def observed_runtime_context(self) -> dict:
        model = getattr(self._model, "model", None)
        return {
            "model_dir": self.model_dir,
            "checkpoint": deepcopy(self._checkpoint) if self._checkpoint is not None else "unknown",
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded_compute_type": getattr(model, "compute_type", "unknown"),
            "language": self.language,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "decoding_options": deepcopy(self._decoding_options)
            if self._decoding_options is not None
            else "unknown",
        }

    def _load(self):
        if self._model is None:
            from shono.transcribe.convert import validate_ct2_checkpoint

            model_path = validate_ct2_checkpoint(self.model_dir)
            identity = hash_checkpoint(model_path)
            from faster_whisper import WhisperModel

            model = WhisperModel(
                str(model_path), device=self.device, compute_type=self.compute_type
            )
            if hash_checkpoint(model_path) != identity:
                raise ValueError("local checkpoint changed while loading")
            self._model = model
            self._checkpoint = identity
        return self._model

    def transcribe_chunk(
        self, audio_path: str, start_s: float, end_s: float
    ) -> tuple[ChunkTranscription, ...]:
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
        segments, info = self._load().transcribe(
            audio,
            language=self.language,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        options = getattr(info, "transcription_options", None)
        self._decoding_options = asdict(options) if is_dataclass(options) else None
        transcriptions: list[ChunkTranscription] = []
        for segment in segments:
            time_bounds(segment.start, segment.end, "decoder segment")
            words = tuple(
                Word(w.start, w.end, w.word, w.probability) for w in (segment.words or ())
            )
            if segment.text.strip() and not any(w.word.strip() for w in words):
                words = ()
            # Word alignment can extend beyond the decoder's segment timestamps.
            segment_start = min([segment.start, *(w.start_s for w in words)])
            segment_end = max([segment.end, *(w.end_s for w in words)])
            if segment_end > end_s - start_s + 1e-6:
                raise ValueError("decoder segment lies outside its audio window")
            transcriptions.append(
                ChunkTranscription(
                    start_s + segment_start,
                    start_s + segment_end,
                    segment.text.strip(),
                    segment.avg_logprob,
                    segment.compression_ratio,
                    segment.no_speech_prob,
                    tuple(
                        Word(
                            w.start_s - segment_start,
                            w.end_s - segment_start,
                            w.word,
                            w.probability,
                        )
                        for w in words
                    ),
                )
            )
        return tuple(transcriptions)
