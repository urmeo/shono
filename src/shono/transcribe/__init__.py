"""Bounded transcription helpers with lazy optional model runtimes."""

from shono.transcribe.chunking import Chunk, plan_chunks
from shono.transcribe.hallucination import (
    Dropped,
    FilterResult,
    HallucinationConfig,
    filter_hallucinations,
)
from shono.transcribe.merge import merge_transcriptions
from shono.transcribe.pipeline import (
    ChunkTranscriber,
    FasterWhisperTranscriber,
    LongFormPipelineTranscriber,
    LongFormTranscriber,
    SileroVAD,
    TranscriptionResult,
    VoiceActivityDetector,
    real_time_factor,
)
from shono.transcribe.types import (
    ChunkTranscription,
    SpeechSegment,
    Transcript,
    TranscriptSegment,
    Word,
)
from shono.transcribe.whisper import ShortFormWhisperTranscriber

__all__ = [
    "Chunk",
    "ChunkTranscriber",
    "ChunkTranscription",
    "Dropped",
    "FasterWhisperTranscriber",
    "FilterResult",
    "HallucinationConfig",
    "LongFormPipelineTranscriber",
    "LongFormTranscriber",
    "ShortFormWhisperTranscriber",
    "SileroVAD",
    "SpeechSegment",
    "Transcript",
    "TranscriptSegment",
    "TranscriptionResult",
    "VoiceActivityDetector",
    "Word",
    "filter_hallucinations",
    "merge_transcriptions",
    "plan_chunks",
    "real_time_factor",
]
