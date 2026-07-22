"""Diarization Error Rate under one explicit protocol — self-contained, no scipy.

DER = (missed speech + false alarm + speaker confusion) / total reference speech.
A DER number is meaningless without its protocol, so this scorer states both
knobs and refuses to hide them:

    * **collar** — a forgiveness window around every *reference* boundary is
      excluded from scoring (annotators disagree on exact boundaries by tenths of
      a second). Default 0.25 s each side, the NIST/DIHARD convention.
    * **skip_overlap** — whether regions where more than one reference speaker is
      active are scored. Default False (overlap *is* scored — the honest, harder
      choice).

Confusion depends on mapping hypothesis speaker labels to reference labels
optimally; that is a max-weight bipartite matching, solved here with the
Hungarian algorithm over the ref×hyp co-occurrence matrix.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from shono.diarize.types import SpeakerSegment


@dataclass(frozen=True)
class DERConfig:
    """The DER protocol: collar width (per side) and whether overlap is scored."""

    collar_s: float = 0.25
    skip_overlap: bool = False

    def __post_init__(self) -> None:
        if self.collar_s < 0:
            raise ValueError(f"collar_s must be >= 0, got {self.collar_s}")


@dataclass(frozen=True)
class DERResult:
    """DER with its components broken out, so a bad number can be diagnosed."""

    der: float
    missed_s: float
    false_alarm_s: float
    confusion_s: float
    total_ref_s: float
    mapping: dict[str, str]  # hypothesis label -> reference label (optimal)

    def as_percent(self) -> float:
        return self.der * 100.0


# ---- Hungarian algorithm (max-weight assignment) -------------------------


def _min_cost_assignment(cost: list[list[float]]) -> list[int]:
    """Optimal square min-cost assignment (Kuhn-Munkres, O(n^3)); returns col per row."""
    n = len(cost)
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def _max_weight_matching(refs: list[str], hyps: list[str], overlap: dict) -> dict[str, str]:
    """Map each hyp label to a ref label to maximize total co-occurrence."""
    if not refs or not hyps:
        return {}
    n = max(len(refs), len(hyps))
    best = max((overlap.get((r, h), 0.0) for r in refs for h in hyps), default=0.0)
    cost = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            in_range = i < len(refs) and j < len(hyps)
            weight = overlap.get((refs[i], hyps[j]), 0.0) if in_range else 0.0
            cost[i][j] = best - weight
    assignment = _min_cost_assignment(cost)
    mapping: dict[str, str] = {}
    for i, j in enumerate(assignment):
        if i < len(refs) and j < len(hyps) and overlap.get((refs[i], hyps[j]), 0.0) > 0.0:
            mapping[hyps[j]] = refs[i]
    return mapping


# ---- timeline machinery --------------------------------------------------


def _active(segments: list[SpeakerSegment], a: float, b: float) -> set[str]:
    mid = (a + b) / 2.0
    return {s.speaker for s in segments if s.start_s <= mid < s.end_s}


def _no_score_boundaries(
    reference: list[SpeakerSegment], collar_s: float
) -> list[tuple[float, float]]:
    if collar_s <= 0:
        return []
    regions = []
    for s in reference:
        regions.append((s.start_s - collar_s, s.start_s + collar_s))
        regions.append((s.end_s - collar_s, s.end_s + collar_s))
    return regions


def _in_any(regions: list[tuple[float, float]], a: float, b: float) -> bool:
    mid = (a + b) / 2.0
    return any(lo <= mid < hi for lo, hi in regions)


def der(
    reference: list[SpeakerSegment],
    hypothesis: list[SpeakerSegment],
    config: DERConfig | None = None,
) -> DERResult:
    """Score ``hypothesis`` against ``reference`` under the given DER protocol."""
    cfg = config or DERConfig()
    if not reference:
        raise ValueError("cannot compute DER against an empty reference")

    refs = sorted({s.speaker for s in reference})
    hyps = sorted({s.speaker for s in hypothesis})

    all_segs = (*reference, *hypothesis)
    cuts = sorted({s.start_s for s in all_segs} | {s.end_s for s in all_segs})
    no_score = _no_score_boundaries(reference, cfg.collar_s)
    for lo, hi in no_score:
        cuts.extend([lo, hi])
    cuts = sorted(set(cuts))

    overlap: dict[tuple[str, str], float] = {}
    missed = false_alarm = 0.0
    total_ref = 0.0
    # first pass: co-occurrence for the optimal mapping
    for a, b in zip(cuts, cuts[1:], strict=False):
        d = b - a
        if d <= 0 or _in_any(no_score, a, b):
            continue
        r_active = _active(reference, a, b)
        if cfg.skip_overlap and len(r_active) > 1:
            continue
        h_active = _active(hypothesis, a, b)
        for r in r_active:
            for h in h_active:
                overlap[(r, h)] = overlap.get((r, h), 0.0) + d

    mapping = _max_weight_matching(refs, hyps, overlap)

    confusion = 0.0
    for a, b in zip(cuts, cuts[1:], strict=False):
        d = b - a
        if d <= 0 or _in_any(no_score, a, b):
            continue
        r_active = _active(reference, a, b)
        if cfg.skip_overlap and len(r_active) > 1:
            continue
        h_active = _active(hypothesis, a, b)
        total_ref += d * len(r_active)
        n_ref, n_sys = len(r_active), len(h_active)
        n_correct = sum(1 for r in r_active if any(mapping.get(h) == r for h in h_active))
        missed += d * max(0, n_ref - n_sys)
        false_alarm += d * max(0, n_sys - n_ref)
        confusion += d * (min(n_ref, n_sys) - n_correct)

    total_error = missed + false_alarm + confusion
    if total_ref > 0:
        der_value = total_error / total_ref
    elif total_error > 0:
        # Error with no scorable reference (all of it collar-excluded) is undefined —
        # never report it as a perfect 0.0, which would hide the error.
        der_value = float("nan")
    else:
        der_value = 0.0
    return DERResult(
        der=der_value,
        missed_s=missed,
        false_alarm_s=false_alarm,
        confusion_s=confusion,
        total_ref_s=total_ref,
        mapping=mapping,
    )


# ---- corpus-level DER + confidence interval ------------------------------

Recording = tuple[list[SpeakerSegment], list[SpeakerSegment]]


def corpus_der(recordings: list[Recording], config: DERConfig | None = None) -> DERResult:
    """Aggregate DER across recordings: total error over total reference speech.

    Each recording is a ``(reference, hypothesis)`` pair; speaker labels are mapped
    optimally *within* each recording (labels are not shared across recordings), and
    the error components are summed. This is how DER is reported on a multi-recording
    eval set — never a mean of per-recording rates, which would misweight short files.
    """
    if not recordings:
        raise ValueError("cannot compute corpus DER over zero recordings")
    missed = false_alarm = confusion = total_ref = 0.0
    for reference, hypothesis in recordings:
        r = der(reference, hypothesis, config)
        missed += r.missed_s
        false_alarm += r.false_alarm_s
        confusion += r.confusion_s
        total_ref += r.total_ref_s
    total_error = missed + false_alarm + confusion
    return DERResult(
        der=total_error / total_ref if total_ref > 0 else 0.0,
        missed_s=missed,
        false_alarm_s=false_alarm,
        confusion_s=confusion,
        total_ref_s=total_ref,
        mapping={},  # per-recording mappings are not comparable across the corpus
    )


@dataclass(frozen=True)
class DERInterval:
    """Corpus DER with a bootstrap CI over recordings — a DER without one is not a result."""

    point: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    n_recordings: int


def der_bootstrap_ci(
    recordings: list[Recording],
    config: DERConfig | None = None,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> DERInterval:
    """Blockwise bootstrap CI for corpus DER, resampling whole recordings."""
    if len(recordings) < 2:
        raise ValueError(
            f"bootstrap needs >= 2 recordings, got {len(recordings)}; "
            "there is no between-recording variance to estimate from one file"
        )
    point = corpus_der(recordings, config).der
    rng = random.Random(seed)
    n = len(recordings)
    samples: list[float] = []
    for _ in range(n_resamples):
        resampled = [recordings[rng.randrange(n)] for _ in range(n)]
        samples.append(corpus_der(resampled, config).der)
    samples.sort()
    alpha = 1.0 - confidence

    def _rank(q: float) -> float:
        return samples[max(1, math.ceil(q * len(samples))) - 1]

    return DERInterval(
        point=point,
        lower=_rank(alpha / 2),
        upper=_rank(1.0 - alpha / 2),
        confidence=confidence,
        n_resamples=n_resamples,
        n_recordings=n,
    )


def render_der_report(
    rows: list[tuple[str, DERResult, DERInterval | None]],
    config: DERConfig,
    *,
    title: str = "Diarization — DER",
    generated: str = "",
) -> str:
    """Render a Markdown DER report: one row per system, components broken out.

    ``rows`` are ``(system, corpus_der_result, ci_or_None)``. The collar/overlap
    protocol is stated in the header — a DER without it is not a result.
    """
    overlap = "scored" if not config.skip_overlap else "excluded"
    lines = [
        f"# {title}",
        "",
        f"- **Protocol:** collar {config.collar_s:g} s per boundary side · overlap {overlap}",
        f"- **Generated:** {generated or '—'}",
        "",
        "DER = (missed + false alarm + confusion) / reference speech; lower is better. "
        "Components are shown as a fraction of reference speech.",
        "",
        "| System | DER (95% CI) | Missed | False alarm | Confusion |",
        "|---|---|---|---|---|",
    ]
    for system, result, ci in rows:
        ref = result.total_ref_s or 1.0
        der_cell = (
            f"{result.der:.1%} [{ci.lower:.1%}, {ci.upper:.1%}]"
            if ci is not None
            else f"{result.der:.1%}"
        )
        lines.append(
            f"| {system} | {der_cell} | {result.missed_s / ref:.1%} "
            f"| {result.false_alarm_s / ref:.1%} | {result.confusion_s / ref:.1%} |"
        )
    return "\n".join(lines) + "\n"
