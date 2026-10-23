"""Corpus WER/CER with whitespace canonicalized before scoring."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jiwer
from jiwer.transforms import ReduceToListOfListOfChars, ReduceToListOfListOfWords

from shono.eval.normalize import NORMALIZER_VERSION, normalize

_WER_TRANSFORM = ReduceToListOfListOfWords()
_CER_TRANSFORM = ReduceToListOfListOfChars()


def _prepare(references: list[str], hypotheses: list[str]) -> tuple[list[str], list[str]]:
    for texts in (references, hypotheses):
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("references and hypotheses must be sequences of strings")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("references and hypotheses must contain only strings")
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
                "scoring against an empty reference is undefined; fix or drop the segment"
            )
    return (
        [" ".join(text.split()) for text in references],
        [" ".join(text.split()) for text in hypotheses],
    )


def _error_rate(corpus: tuple[list[str], list[str]], *, characters: bool = False) -> float:
    metric = jiwer.cer if characters else jiwer.wer
    transform = _CER_TRANSFORM if characters else _WER_TRANSFORM
    return metric(*corpus, reference_transform=transform, hypothesis_transform=transform)


def wer(references: list[str], hypotheses: list[str]) -> float:
    return _error_rate(_prepare(references, hypotheses))


def cer(references: list[str], hypotheses: list[str]) -> float:
    return _error_rate(_prepare(references, hypotheses), characters=True)


@dataclass(frozen=True)
class ScoreReport:
    wer_raw: float
    cer_raw: float
    wer_normalized: float
    cer_normalized: float
    normalizer_version: str
    n_segments: int


def score(
    references: list[str],
    hypotheses: list[str],
    *,
    normalizer: Callable[[str], str] = normalize,
    normalizer_version: str = NORMALIZER_VERSION,
) -> ScoreReport:
    raw = _prepare(references, hypotheses)
    normalized = _prepare(
        [normalizer(text) for text in references],
        [normalizer(text) for text in hypotheses],
    )
    return ScoreReport(
        wer_raw=_error_rate(raw),
        cer_raw=_error_rate(raw, characters=True),
        wer_normalized=_error_rate(normalized),
        cer_normalized=_error_rate(normalized, characters=True),
        normalizer_version=normalizer_version,
        n_segments=len(references),
    )
