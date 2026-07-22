"""Reject Whisper hallucinations by confidence signals — and say what was dropped.

Whisper invents text on silence and loops on repetition. Three signals catch most
of it: a low ``avg_logprob`` (the model is guessing), a high ``compression_ratio``
(the output is repetitive — it gzip-compresses far too well), and a high
``no_speech_prob`` (the chunk is probably silence). This filter drops chunks that
trip the guards, but it *returns what it dropped* — a pipeline that silently
deletes output is exactly the kind of dishonesty this project avoids.
"""

from __future__ import annotations

from dataclasses import dataclass

from shono.transcribe.types import ChunkTranscription


@dataclass(frozen=True)
class HallucinationConfig:
    """Thresholds for the confidence-based rejection guards."""

    min_avg_logprob: float = -1.0
    max_compression_ratio: float = 2.4
    max_no_speech_prob: float = 0.6

    def reasons(self, chunk: ChunkTranscription) -> tuple[str, ...]:
        """Which guards (if any) this chunk trips — empty means it is kept."""
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
