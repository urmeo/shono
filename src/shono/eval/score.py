"""WER and CER scoring with explicit transforms — nothing happens implicitly.

jiwer's default transform quietly collapses repeated spaces and strips edges
before scoring, and its word splitter breaks only on single spaces — so a
newline in a hypothesis would count as an error. This module makes the raw
contract explicit instead: **raw = whitespace-canonicalized only** (split on
any whitespace, rejoin with single spaces — the one transform without which
word scoring is undefined), then explicit jiwer transforms with no other
munging: no case folding, no punctuation handling, nothing. "Normalized"
scores additionally run both sides — always both, never one — through the
frozen pipeline in :mod:`shono.eval.normalize`.
"""

from dataclasses import dataclass

import jiwer
from jiwer.transforms import Compose, ReduceToListOfListOfChars, ReduceToListOfListOfWords

from shono.eval.normalize import NORMALIZER_VERSION, normalize

_WER_TRANSFORM = Compose([ReduceToListOfListOfWords()])
_CER_TRANSFORM = Compose([ReduceToListOfListOfChars()])


def _canonicalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _validate(references: list[str], hypotheses: list[str]) -> None:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"got {len(references)} references but {len(hypotheses)} hypotheses; "
            "every hypothesis needs exactly one reference"
        )
    if not references:
        raise ValueError("cannot score an empty corpus")
    for i, ref in enumerate(references):
        if not ref.strip():
            raise ValueError(
                f"reference {i} is empty or whitespace-only; "
                "scoring against an empty reference is undefined — fix or drop the segment"
            )


def wer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level word error rate: total word edits / total reference words."""
    _validate(references, hypotheses)
    return jiwer.wer(
        [_canonicalize_whitespace(r) for r in references],
        [_canonicalize_whitespace(h) for h in hypotheses],
        reference_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )


def cer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level character error rate: total char edits / total reference chars."""
    _validate(references, hypotheses)
    return jiwer.cer(
        [_canonicalize_whitespace(r) for r in references],
        [_canonicalize_whitespace(h) for h in hypotheses],
        reference_transform=_CER_TRANSFORM,
        hypothesis_transform=_CER_TRANSFORM,
    )


@dataclass(frozen=True)
class ScoreReport:
    """Raw and normalized WER/CER for one corpus, stamped with the normalizer version."""

    wer_raw: float
    cer_raw: float
    wer_normalized: float
    cer_normalized: float
    normalizer_version: str
    n_segments: int


def score(references: list[str], hypotheses: list[str]) -> ScoreReport:
    """Score a corpus raw and under the frozen normalization pipeline."""
    _validate(references, hypotheses)
    norm_refs = [normalize(r) for r in references]
    norm_hyps = [normalize(h) for h in hypotheses]
    return ScoreReport(
        wer_raw=wer(references, hypotheses),
        cer_raw=cer(references, hypotheses),
        wer_normalized=wer(norm_refs, norm_hyps),
        cer_normalized=cer(norm_refs, norm_hyps),
        normalizer_version=NORMALIZER_VERSION,
        n_segments=len(references),
    )
