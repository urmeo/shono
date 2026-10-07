"""Filter chunks using declared confidence thresholds and retain drop reasons."""

from __future__ import annotations

from dataclasses import dataclass

from shono.transcribe._validation import finite_real, probability
from shono.transcribe.types import ChunkTranscription


@dataclass(frozen=True)
class HallucinationConfig:
    """Thresholds for the confidence-based rejection guards."""

    min_avg_logprob: float = -1.0
    max_compression_ratio: float = 2.4
    max_no_speech_prob: float = 0.6

    def __post_init__(self) -> None:
        finite_real(self.min_avg_logprob, "min_avg_logprob")
        finite_real(self.max_compression_ratio, "max_compression_ratio", minimum=0)
        probability(self.max_no_speech_prob, "max_no_speech_prob")

    def reasons(self, chunk: ChunkTranscription) -> tuple[str, ...]:
        """Return threshold violations; an empty tuple means the chunk is kept."""
        out: list[str] = []
        if chunk.avg_logprob < self.min_avg_logprob:
            out.append(f"avg_logprob {chunk.avg_logprob:.2f} < {self.min_avg_logprob}")
        if chunk.compression_ratio > self.max_compression_ratio:
            out.append(
                f"compression_ratio {chunk.compression_ratio:.2f} > {self.max_compression_ratio}"
            )
        if chunk.no_speech_prob > self.max_no_speech_prob:
            out.append(f"no_speech_prob {chunk.no_speech_prob:.2f} > {self.max_no_speech_prob}")
        return tuple(out)


@dataclass(frozen=True)
class Dropped:
    """A chunk removed as a probable hallucination, with the reason(s) it tripped."""

    chunk: ChunkTranscription
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FilterResult:
    """The surviving chunks and the ones dropped (never silently discarded)."""

    kept: tuple[ChunkTranscription, ...]
    dropped: tuple[Dropped, ...]

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    def summary(self) -> str:
        if not self.dropped:
            return f"kept all {len(self.kept)} chunks"
        return f"kept {len(self.kept)}, dropped {self.n_dropped} as probable hallucinations"


def filter_hallucinations(
    chunks: list[ChunkTranscription], config: HallucinationConfig | None = None
) -> FilterResult:
    """Split ``chunks`` into kept and dropped by the confidence guards."""
    cfg = config or HallucinationConfig()
    kept: list[ChunkTranscription] = []
    dropped: list[Dropped] = []
    for chunk in chunks:
        reasons = cfg.reasons(chunk)
        if reasons:
            dropped.append(Dropped(chunk, reasons))
        else:
            kept.append(chunk)
    return FilterResult(kept=tuple(kept), dropped=tuple(dropped))
