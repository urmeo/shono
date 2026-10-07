"""Corpus WER/CER: total edits divided by total reference units.

Raw scoring canonicalizes whitespace only. Normalized scoring applies the
same stated normalizer to references and hypotheses."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jiwer
from jiwer.transforms import Compose, ReduceToListOfListOfChars, ReduceToListOfListOfWords

from shono.eval.normalize import NORMALIZER_VERSION, normalize

_WER_TRANSFORM = Compose([ReduceToListOfListOfWords()])
_CER_TRANSFORM = Compose([ReduceToListOfListOfChars()])


def _canonicalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _validate(references: list[str], hypotheses: list[str]) -> None:
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


def score(
    references: list[str],
    hypotheses: list[str],
    *,
    normalizer: Callable[[str], str] = normalize,
    normalizer_version: str = NORMALIZER_VERSION,
) -> ScoreReport:
    """Score a corpus raw and under a normalization pipeline (the frozen one by default).

    ``normalizer`` defaults to the frozen pipeline; the code-switch slice passes its
    documented variant (:func:`shono.eval.codeswitch.code_switch_normalize`) with the
    matching version, so the stamped ``normalizer_version`` always names the pipeline
    that produced the normalized numbers.
    """
    _validate(references, hypotheses)
    norm_refs = [normalizer(r) for r in references]
    norm_hyps = [normalizer(h) for h in hypotheses]
    return ScoreReport(
        wer_raw=wer(references, hypotheses),
        cer_raw=cer(references, hypotheses),
        wer_normalized=wer(norm_refs, norm_hyps),
        cer_normalized=cer(norm_refs, norm_hyps),
        normalizer_version=normalizer_version,
        n_segments=len(references),
    )
