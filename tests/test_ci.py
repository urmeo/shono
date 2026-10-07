"""Blockwise bootstrap: correct point estimate, sane deterministic intervals, loud failures."""

import json
from pathlib import Path

import pytest

from shono.eval import blockwise_bootstrap_ci

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "ci_blocks.json").read_text(encoding="utf-8")
)
_RECORDS = [tuple(r) for r in _FIXTURE["records"]]

# Resampling 2 blocks with replacement can only produce {A,A}, {A,B}, {B,B}:
# corpus WERs 0, 1/6, 1/3. Any other value means resampling is not blockwise.
_BLOCKWISE_ONLY_VALUES = {0.0, 1 / 6, 1 / 3}


def test_point_estimate_matches_hand_count():
    result = blockwise_bootstrap_ci(_RECORDS, metric="wer", n_resamples=200, seed=0)
    assert result.point == pytest.approx(_FIXTURE["expected_point_wer"])
    assert result.n_blocks == 2
    assert result.n_segments == 4


def test_resampling_is_genuinely_blockwise():
    # Segment-wise resampling would produce values like 1/12 (one errorful
    # segment among three perfect ones) that block resampling cannot.
    result = blockwise_bootstrap_ci(
        _RECORDS, metric="wer", n_resamples=500, seed=7, keep_samples=True
    )
    assert result.samples is not None and len(result.samples) == 500
    for value in result.samples:
        assert any(value == pytest.approx(v) for v in _BLOCKWISE_ONLY_VALUES), (
            f"sample {value} is impossible under blockwise resampling"
        )


def test_interval_brackets_point_within_possible_range():
    result = blockwise_bootstrap_ci(_RECORDS, metric="wer", n_resamples=1000, seed=0)
    assert 0.0 <= result.lower <= result.point <= result.upper
    assert result.upper <= _FIXTURE["max_possible_wer"] + 1e-12


def test_deterministic_for_same_seed():
    a = blockwise_bootstrap_ci(_RECORDS, metric="wer", n_resamples=300, seed=42)
    b = blockwise_bootstrap_ci(_RECORDS, metric="wer", n_resamples=300, seed=42)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_samples_omitted_by_default():
    result = blockwise_bootstrap_ci(_RECORDS, metric="wer", n_resamples=100, seed=0)
    assert result.samples is None


def test_cer_metric_supported():
    result = blockwise_bootstrap_ci(_RECORDS, metric="cer", n_resamples=200, seed=0)
    assert 0.0 <= result.lower <= result.point <= result.upper


def test_single_block_raises():
    single = [r for r in _RECORDS if r[0] == "A"]
    with pytest.raises(ValueError, match=">= 2 blocks"):
        blockwise_bootstrap_ci(single, n_resamples=100)


def test_too_few_resamples_raises():
    with pytest.raises(ValueError, match="n_resamples"):
        blockwise_bootstrap_ci(_RECORDS, n_resamples=10)


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="unknown metric"):
        blockwise_bootstrap_ci(_RECORDS, metric="der")


def test_bad_confidence_raises():
    with pytest.raises(ValueError, match="confidence"):
        blockwise_bootstrap_ci(_RECORDS, confidence=1.5)


@pytest.mark.parametrize("confidence", [True, False, float("nan"), float("inf"), 0, 1, "0.95", 1j])
def test_confidence_must_be_real_finite_nonbool(confidence):
    with pytest.raises(ValueError, match="confidence"):
        blockwise_bootstrap_ci(_RECORDS, confidence=confidence)


@pytest.mark.parametrize("count", [True, False, 100.5, "100", float("inf"), None])
def test_resample_count_must_be_integer_nonbool(count):
    with pytest.raises(ValueError, match="n_resamples"):
        blockwise_bootstrap_ci(_RECORDS, n_resamples=count)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 0.5, "1", None])
def test_seed_contract(seed):
    with pytest.raises(ValueError, match="seed"):
        blockwise_bootstrap_ci(_RECORDS, seed=seed)


@pytest.mark.parametrize(
    "records",
    [
        None,
        "abc",
        [],
        [("r", "text")],
        [(1, "text", "hyp")],
        [("", "text", "hyp")],
        [("r", "text", None)],
    ],
)
def test_input_triples_are_validated(records):
    with pytest.raises(ValueError):
        blockwise_bootstrap_ci(records)


def test_keep_samples_must_be_boolean():
    with pytest.raises(ValueError, match="keep_samples"):
        blockwise_bootstrap_ci(_RECORDS, keep_samples="false")
