"""Nearest-rank bootstrap intervals over recording or speaker blocks."""

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from shono.eval import score as _score

_METRICS = {"wer": _score.wer, "cer": _score.cer}
_MIN_RESAMPLES = 100


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate with a nearest-rank percentile bootstrap interval over blocks."""

    metric: str
    point: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    n_blocks: int
    n_segments: int
    samples: tuple[float, ...] | None = None


def validate_bootstrap_settings(n_resamples: int, confidence: float, seed: int) -> None:
    """Validate the bootstrap protocol before any work."""
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 100:
        raise ValueError("n_resamples must be an integer >= 100")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a finite real number in (0, 1)")
    if not 0 < confidence < 1 or not math.isfinite(confidence):
        raise ValueError("confidence must be a finite real number in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def _nearest_rank(sorted_samples: list[float], q: float) -> float:
    rank = max(1, math.ceil(q * len(sorted_samples)))
    return sorted_samples[rank - 1]


def blockwise_bootstrap_ci(
    records: list[tuple[str, str, str]],
    metric: str = "wer",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    keep_samples: bool = False,
) -> BootstrapCI:
    """Resample (block_id, reference, hypothesis) triples by whole block.

    Strings follow the raw whitespace-only scoring contract; normalize upstream.
    keep_samples=True retains the bootstrap draws.
    """
    validate_bootstrap_settings(n_resamples, confidence, seed)
    if not isinstance(metric, str) or metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(_METRICS)}")
    if not isinstance(keep_samples, bool):
        raise ValueError("keep_samples must be boolean")
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("records must be a nonempty sequence of triples")
    for item in records:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("each record must be a (block_id, reference, hypothesis) triple")
        if any(not isinstance(value, str) for value in item) or not item[0].strip():
            raise ValueError("record values must be strings with a nonempty block_id")

    blocks: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for block_id, reference, hypothesis in records:
        blocks[block_id].append((reference, hypothesis))
    if len(blocks) < 2:
        raise ValueError(
            f"blockwise bootstrap needs >= 2 blocks, got {len(blocks)}; "
            "with one recording there is no between-block variance to estimate"
        )

    compute = _METRICS[metric]
    refs = [r for _, r, _ in records]
    hyps = [h for _, _, h in records]
    point = compute(refs, hyps)

    block_ids = sorted(blocks)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        sample_refs: list[str] = []
        sample_hyps: list[str] = []
        for _ in block_ids:
            for ref, hyp in blocks[rng.choice(block_ids)]:
                sample_refs.append(ref)
                sample_hyps.append(hyp)
        samples.append(compute(sample_refs, sample_hyps))

    samples.sort()
    alpha = 1.0 - confidence
    return BootstrapCI(
        metric=metric,
        point=point,
        lower=_nearest_rank(samples, alpha / 2),
        upper=_nearest_rank(samples, 1.0 - alpha / 2),
        confidence=confidence,
        n_resamples=n_resamples,
        n_blocks=len(blocks),
        n_segments=len(records),
        samples=tuple(samples) if keep_samples else None,
    )
