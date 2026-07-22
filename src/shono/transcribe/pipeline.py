"""The long-form pipeline: VAD → chunk → transcribe → de-hallucinate → merge.

The orchestration is pure and injectable: a :class:`VoiceActivityDetector` and a
:class:`ChunkTranscriber` are passed in, so tests drive the whole flow with fakes
and no audio, while production wires the lazy Silero-VAD and faster-whisper
implementations below. Real-time factor is measured here (processing time ÷ audio
duration) — the number decision S3 requires, computed, never assumed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from shono.transcribe.chunking import plan_chunks
from shono.transcribe.hallucination import FilterResult, HallucinationConfig, filter_hallucinations
from shono.transcribe.merge import merge_transcriptions
from shono.transcribe.types import ChunkTranscription, SpeechSegment, Transcript

TARGET_SAMPLE_RATE = 16_000


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Marks the speech regions of a recording."""

    def detect(self, audio_path: str) -> list[SpeechSegment]: ...


@runtime_checkable
class ChunkTranscriber(Protocol):
    """Transcribes one ``[start_s, end_s]`` window of a recording."""

    def transcribe_chunk(self, audio_path: str, start_s: float, end_s: float) -> ChunkTranscription:
        ...


def real_time_factor(processing_s: float, audio_s: float) -> float:
    """Processing time ÷ audio duration. < 1.0 means faster than real time."""
    if audio_s <= 0:
        raise ValueError(f"audio_s must be > 0, got {audio_s}")
    return processing_s / audio_s


@dataclass(frozen=True)
class TranscriptionResult:
    """A transcript plus the numbers that judge the run: RTF, chunk count, drops."""

    transcript: Transcript
    audio_duration_s: float
    processing_s: float
    n_chunks: int
    filtering: FilterResult

    @property
    def rtf(self) -> float:
        return real_time_factor(self.processing_s, self.audio_duration_s)


class LongFormTranscriber:
    """Turns a long recording into a merged, de-hallucinated, globally-timed transcript."""

    def __init__(
        self,
        vad: VoiceActivityDetector,
        transcriber: ChunkTranscriber,
        *,
        max_chunk_s: float = 28.0,
        pad_s: float = 0.2,
        hallucination: HallucinationConfig | None = None,
    ) -> None:
        self.vad = vad
        self.transcriber = transcriber
        self.max_chunk_s = max_chunk_s
        self.pad_s = pad_s
        self.hallucination = hallucination or HallucinationConfig()

    def transcribe(
        self,
        audio_path: str,
        audio_duration_s: float,
        *,
        now: Callable[[], float] = perf_counter,
    ) -> TranscriptionResult:
        """Transcribe ``audio_path`` end to end, measuring the real-time factor.

        The measured RTF includes the one-time model load (the lazy VAD/model are
        constructed on first use inside the timed span), so it is conservative — a
        steady-state RTF is lower. State this when reporting an RTF near 1.0.
        """
        started = now()
        speech = self.vad.detect(audio_path)
        chunks = plan_chunks(speech, max_chunk_s=self.max_chunk_s, pad_s=self.pad_s)
        transcriptions = [
            self.transcriber.transcribe_chunk(audio_path, c.start_s, c.end_s) for c in chunks
        ]
        filtered = filter_hallucinations(transcriptions, self.hallucination)
        transcript = merge_transcriptions(list(filtered.kept))
        processing_s = now() - started
        return TranscriptionResult(
            transcript=transcript,
            audio_duration_s=audio_duration_s,
            processing_s=processing_s,
            n_chunks=len(chunks),
            filtering=filtered,
        )


class LongFormPipelineTranscriber:
    """Adapt the long-form pipeline to the ``transcribe(path, …) -> str`` protocol.

    Runs the full VAD → chunk → transcribe → de-hallucinate → merge pipeline on a
    whole recording and returns the merged transcript text. This lets
    ``shono.api.run_over_manifest`` produce recording-level long-form predictions
    through the same one runner used for baselines and APIs. Score against a
    recording-level manifest (see ``shono.data.collapse_to_recordings``).
    """

    def __init__(self, pipeline: LongFormTranscriber) -> None:
        self.pipeline = pipeline

    def transcribe(
        self, audio_path: str, start_s: float | None = None, duration_s: float | None = None
    ) -> str:
        if duration_s is None:
            import soundfile as sf

            info = sf.info(audio_path)
            duration_s = info.frames / info.samplerate
        return self.pipeline.transcribe(audio_path, duration_s).transcript.text


# --- Lazy GPU implementations (Kaggle / faster-whisper; imported only when used) ---


class SileroVAD:
    """Silero-VAD speech detection (torch, downloaded on first use)."""

    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 250) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms

    def detect(self, audio_path: str) -> list[SpeechSegment]:
        import torch

        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        get_ts, _, read_audio, *_ = utils
        wav = read_audio(audio_path, sampling_rate=TARGET_SAMPLE_RATE)
        stamps = get_ts(
            wav, model, threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms, sampling_rate=TARGET_SAMPLE_RATE,
        )
        return [
            SpeechSegment(s["start"] / TARGET_SAMPLE_RATE, s["end"] / TARGET_SAMPLE_RATE)
            for s in stamps
        ]


class FasterWhisperTranscriber:
    """faster-whisper (CTranslate2) chunk transcription with word timestamps."""

    def __init__(self, model_dir: str, *, device: str = "cuda", compute_type: str = "float16",
                 language: str = "bn") -> None:
        self.model_dir = model_dir
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_dir, device=self.device,
                                       compute_type=self.compute_type)
        return self._model

    def transcribe_chunk(self, audio_path: str, start_s: float, end_s: float) -> ChunkTranscription:
        import librosa

        from shono.transcribe.types import Word

        audio, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, offset=start_s,
                                duration=max(0.0, end_s - start_s), mono=True)
        segments, _info = self._load().transcribe(
            audio, language=self.language, word_timestamps=True,
            condition_on_previous_text=False,  # a key hallucination guard
        )
        texts, words, logprobs, comps, nosp = [], [], [], [], []
        for seg in segments:
            texts.append(seg.text)
            logprobs.append(seg.avg_logprob)
            comps.append(seg.compression_ratio)
            nosp.append(seg.no_speech_prob)
            for w in (seg.words or []):
                words.append(Word(w.start, w.end, w.word, w.probability))

        def _mean(xs, default):
            return sum(xs) / len(xs) if xs else default

        text = " ".join(t.strip() for t in texts).strip()
        # Guarantee word timings when there is text, so the merger always uses its
        # precise dedup path rather than the coarse wordless fallback.
        if text and not words:
            words.append(Word(0.0, max(0.0, end_s - start_s), text))
        return ChunkTranscription(
            window_start_s=start_s,
            window_end_s=end_s,
            text=text,
            avg_logprob=_mean(logprobs, 0.0),
            compression_ratio=_mean(comps, 1.0),
            no_speech_prob=_mean(nosp, 0.0),
            words=tuple(words),
        )
