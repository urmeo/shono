"""Hand-computed corpus scores and normalization boundaries."""

import json
from pathlib import Path

import pytest

from shono.eval import NORMALIZER_VERSION, cer, score, wer

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "wer_pairs.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_hand_computed_wer_cer(case):
    assert wer(case["references"], case["hypotheses"]) == pytest.approx(case["expected_wer"])
    if case["expected_cer"] is not None:
        assert cer(case["references"], case["hypotheses"]) == pytest.approx(case["expected_cer"])


def test_raw_wer_treats_any_whitespace_as_separator():
    assert wer(["আমি ভাত খাই"], ["আমি ভাত\nখাই"]) == 0.0
    assert wer(["আমি ভাত খাই"], ["আমি  ভাত\tখাই"]) == 0.0


def test_raw_scoring_never_folds_case():
    assert wer(["আমি Bank এ যাব"], ["আমি bank এ যাব"]) == pytest.approx(0.25)
    report = score(["আমি Bank এ যাব"], ["আমি bank এ যাব"])
    assert report.wer_normalized == pytest.approx(0.25)


def test_punctuation_mismatch_penalized_raw_but_not_normalized():
    report = score(["আমি ভাত খাই।"], ["আমি ভাত খাই"])
    assert report.wer_raw == pytest.approx(1 / 3)
    assert report.wer_normalized == 0.0
    assert report.normalizer_version == NORMALIZER_VERSION
    assert report.n_segments == 1


def test_normalization_applied_to_both_sides():
    report = score(["খাই।"], ["খাই।"])
    assert report.wer_raw == 0.0
    assert report.wer_normalized == 0.0


def test_taka_spacing_cannot_flip_normalized_scores():
    report = score(["৳ ১০০ দাম"], ["৳১০০ দাম"])
    assert report.wer_normalized == 0.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="references but"):
        wer(["আমি"], ["আমি", "তুমি"])


def test_empty_corpus_raises():
    with pytest.raises(ValueError, match="empty corpus"):
        wer([], [])


def test_empty_reference_raises():
    with pytest.raises(ValueError, match="reference 0 is empty"):
        wer([" "], ["আমি"])


def test_reference_empty_after_normalization_raises():
    with pytest.raises(ValueError, match="empty"):
        score(["৳!"], ["টাকা"])
