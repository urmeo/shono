"""Long-form transcription: VAD chunking, hallucination guards, timestamp merging.

The chunk planner, hallucination filter, merger, and pipeline orchestration are
pure and imported eagerly; the Silero-VAD / faster-whisper implementations and the
CTranslate2 converter keep their heavy imports inside functions, so this package
loads without torch or a GPU.
"""

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

__all__ = [
    "Chunk",
    "ChunkTranscriber",
    "ChunkTranscription",
    "Dropped",
    "FasterWhisperTranscriber",
    "FilterResult",
    "HallucinationConfig",
    "LongFormTranscriber",
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
