"""Blockwise bootstrap confidence intervals for WER/CER.

Segments from the same recording are correlated, so resampling individual
segments underestimates variance (Liu et al., Interspeech 2020). This module
resamples whole *blocks* — one block per recording — with replacement, and
reports nearest-rank percentile intervals. Every per-slice number in a report
carries one of these intervals; a bare point estimate is not a result.
"""

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
    """Compute ``metric`` with a blockwise-bootstrap CI.

    ``records`` are ``(block_id, reference, hypothesis)`` triples, where
    ``block_id`` identifies the recording (or speaker) a segment came from.
    Strings are scored under the raw contract of :mod:`shono.eval.score`
    (whitespace-canonicalized, nothing else) — normalize upstream if
    normalized scores are wanted, so raw and normalized runs share one
    code path.
    ``keep_samples=True`` returns every resampled metric value for
    inspection or plotting.
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(_METRICS)}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < _MIN_RESAMPLES:
        raise ValueError(
            f"n_resamples must be >= {_MIN_RESAMPLES}, got {n_resamples}; "
            "percentile intervals from fewer resamples are noise"
        )

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
