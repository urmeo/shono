"""Reference-time weighted DER with per-recording optimal speaker assignment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from shono.diarize.types import SpeakerSegment
from shono.transcribe._validation import count, finite_real, finite_sum


@dataclass(frozen=True)
class DERConfig:
    """The DER protocol: collar width (per side) and whether overlap is scored."""

    collar_s: float = 0.25
    skip_overlap: bool = False

    def __post_init__(self) -> None:
        finite_real(self.collar_s, "collar_s", minimum=0)
        if not isinstance(self.skip_overlap, bool):
            raise ValueError("skip_overlap must be boolean")


@dataclass(frozen=True)
class DERResult:
    """DER with its components broken out, so a bad number can be diagnosed."""

    der: float
    missed_s: float
    false_alarm_s: float
    confusion_s: float
    total_ref_s: float
    mapping: dict[str, str]  # hypothesis label -> reference label (optimal)
    config: DERConfig | None = None

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
            cost[i][j] = (best - weight) / best if best else 0.0
    assignment = _min_cost_assignment(cost)
    mapping: dict[str, str] = {}
    for i, j in enumerate(assignment):
        if i < len(refs) and j < len(hyps) and overlap.get((refs[i], hyps[j]), 0.0) > 0.0:
            mapping[hyps[j]] = refs[i]
    return mapping


# ---- timeline machinery --------------------------------------------------


def _active(segments: list[SpeakerSegment], a: float, b: float) -> set[str]:
    mid = a / 2.0 + b / 2.0
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
    if any(not math.isfinite(v) for region in regions for v in region):
        raise ValueError("collar boundaries must be finite")
    return regions


def _in_any(regions: list[tuple[float, float]], a: float, b: float) -> bool:
    mid = a / 2.0 + b / 2.0
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
    if any(not isinstance(s, SpeakerSegment) for s in (*reference, *hypothesis)):
        raise ValueError("DER inputs must contain speaker segments")

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
                finite_real(overlap[(r, h)], "speaker overlap", minimum=0)

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
        for value in (total_ref, missed, false_alarm, confusion):
            finite_real(value, "DER component", minimum=0)

    total_error = missed + false_alarm + confusion
    finite_real(total_error, "total DER error", minimum=0)
    if total_ref > 0:
        der_value = finite_real(total_error / total_ref, "DER", minimum=0)
    else:
        der_value = float("nan")
    return DERResult(
        der=der_value,
        missed_s=missed,
        false_alarm_s=false_alarm,
        confusion_s=confusion,
        total_ref_s=total_ref,
        mapping=mapping,
        config=cfg,
    )


# ---- corpus-level DER + confidence interval ------------------------------

Recording = tuple[list[SpeakerSegment], list[SpeakerSegment]]


def corpus_der(recordings: list[Recording], config: DERConfig | None = None) -> DERResult:
    """Sum errors and reference time with recording-local speaker mappings."""
    if not recordings:
        raise ValueError("cannot compute corpus DER over zero recordings")
    missed = false_alarm = confusion = total_ref = 0.0
    for reference, hypothesis in recordings:
        r = der(reference, hypothesis, config)
        missed += r.missed_s
        false_alarm += r.false_alarm_s
        confusion += r.confusion_s
        total_ref += r.total_ref_s
        for value in (total_ref, missed, false_alarm, confusion):
            finite_real(value, "corpus DER component", minimum=0)
    total_error = missed + false_alarm + confusion
    finite_real(total_error, "total corpus DER error", minimum=0)
    return DERResult(
        der=finite_real(total_error / total_ref, "corpus DER", minimum=0)
        if total_ref > 0
        else float("nan"),
        missed_s=missed,
        false_alarm_s=false_alarm,
        confusion_s=confusion,
        total_ref_s=total_ref,
        mapping={},  # per-recording mappings are not comparable across the corpus
        config=config or DERConfig(),
    )


@dataclass(frozen=True)
class DERInterval:
    """A confidence interval resampling complete recordings."""

    point: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    n_recordings: int
    config: DERConfig | None = None

    def __post_init__(self) -> None:
        for value in (self.point, self.lower, self.upper):
            finite_real(value, "DER interval", minimum=0)
        if self.lower > self.upper:
            raise ValueError("DER interval lower bound exceeds upper bound")
        if not 0 < finite_real(self.confidence, "confidence") < 1:
            raise ValueError("confidence must be in (0, 1)")
        count(self.n_resamples, "n_resamples", minimum=100)
        count(self.n_recordings, "n_recordings", minimum=2)


def der_bootstrap_ci(
    recordings: list[Recording],
    config: DERConfig | None = None,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> DERInterval:
    """Blockwise bootstrap CI for corpus DER, resampling whole recordings."""
    count(n_resamples, "n_resamples", minimum=100)
    level = finite_real(confidence, "confidence")
    if not 0 < level < 1:
        raise ValueError("confidence must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if len(recordings) < 2:
        raise ValueError(
            f"bootstrap needs >= 2 recordings, got {len(recordings)}; "
            "there is no between-recording variance to estimate from one file"
        )
    individual = [der(reference, hypothesis, config) for reference, hypothesis in recordings]
    if any(result.total_ref_s <= 0 for result in individual):
        raise ValueError("bootstrap requires scorable reference time in every recording")
    point = corpus_der(recordings, config).der
    rng = random.Random(seed)
    n = len(recordings)
    samples: list[float] = []
    for _ in range(n_resamples):
        resampled = [individual[rng.randrange(n)] for _ in range(n)]
        numerator = finite_sum(
            (r.missed_s + r.false_alarm_s + r.confusion_s for r in resampled), "resampled errors"
        )
        denominator = finite_sum((r.total_ref_s for r in resampled), "resampled reference time")
        samples.append(finite_real(numerator / denominator, "resampled DER", minimum=0))
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
        config=config or DERConfig(),
    )


def render_der_report(
    rows: list[tuple[str, DERResult, DERInterval | None]],
    config: DERConfig,
    *,
    title: str = "Diarization | DER",
    generated: str = "",
) -> str:
    """Render the actual confidence level and reject mismatched result protocols."""
    overlap = "scored" if not config.skip_overlap else "excluded"
    lines = [
        f"# {title}",
        "",
        f"- **Protocol:** collar {config.collar_s:g} s per boundary side · overlap {overlap}",
        f"- **Generated:** {generated or 'n/a'}",
        "",
        "DER = (missed + false alarm + confusion) / reference speech; lower is better. "
        "Components are shown as a fraction of reference speech.",
        "",
        "| System | DER (CI when available) | Missed | False alarm | Confusion |",
        "|---|---|---|---|---|",
    ]
    for system, result, ci in rows:
        if result.config is not None and result.config != config:
            raise ValueError("DER result protocol differs from report protocol")
        if ci is not None and (ci.config is not None and ci.config != config):
            raise ValueError("DER interval protocol differs from report protocol")
        if ci is not None and not math.isclose(result.der, ci.point, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("DER interval point differs from the reported result")
        if result.total_ref_s <= 0:
            lines.append(f"| {system} | n/a | n/a | n/a | n/a |")
            continue
        ref = result.total_ref_s
        der_cell = (
            f"{result.der:.1%} [{ci.lower:.1%}, {ci.upper:.1%}] ({ci.confidence * 100:g}% CI)"
            if ci is not None
            else f"{result.der:.1%}"
        )
        lines.append(
            f"| {system} | {der_cell} | {result.missed_s / ref:.1%} "
            f"| {result.false_alarm_s / ref:.1%} | {result.confusion_s / ref:.1%} |"
        )
    return "\n".join(lines) + "\n"
