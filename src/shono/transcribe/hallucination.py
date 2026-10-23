"""Filter decoder segments using confidence thresholds and retain drop reasons."""

from __future__ import annotations

from dataclasses import dataclass

from shono.transcribe.types import ChunkTranscription
from shono.validation import probability, real_number


@dataclass(frozen=True)
class HallucinationConfig:
    """Thresholds for the confidence-based rejection guards."""

    min_avg_logprob: float = -1.0
    max_compression_ratio: float = 2.4
    max_no_speech_prob: float = 0.6

    def __post_init__(self) -> None:
        real_number(self.min_avg_logprob, "min_avg_logprob")
        real_number(self.max_compression_ratio, "max_compression_ratio", minimum=0)
        probability(self.max_no_speech_prob, "max_no_speech_prob")

    def reasons(self, chunk: ChunkTranscription) -> tuple[str, ...]:
        """Return threshold violations; an empty tuple means the segment is kept."""
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
    """A rejected decoder segment and its threshold violations."""

    chunk: ChunkTranscription
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FilterResult:
    """Accepted decoder segments and rejected segments with reasons."""

    kept: tuple[ChunkTranscription, ...]
    dropped: tuple[Dropped, ...]

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    def summary(self) -> str:
        if not self.dropped:
            return f"kept all {len(self.kept)} decoder segments"
        return f"kept {len(self.kept)}, dropped {self.n_dropped} decoder segments"


def filter_hallucinations(
    chunks: list[ChunkTranscription], config: HallucinationConfig | None = None
) -> FilterResult:
    """Apply confidence guards to each decoder segment independently."""
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
