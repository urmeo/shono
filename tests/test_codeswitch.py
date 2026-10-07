"""Script-normalized code-switch scoring: case-insensitive Latin, Bengali untouched."""

import re

import pytest

from shono.data.manifest import Manifest, Segment
from shono.eval import NORMALIZER_VERSION, normalize
from shono.eval.codeswitch import (
    CS_NORMALIZER_VERSION,
    bengali_fraction,
    code_switch_normalize,
    is_code_switched,
    script_counts,
)
from shono.eval.report import Predictions, score_slice

# ---- the policy ----------------------------------------------------------


def test_latin_is_casefolded():
    assert code_switch_normalize("Machine Learning") == "machine learning"


def test_bengali_is_left_intact():
    # Bengali has no case; casefold must not touch matras or characters.
    assert code_switch_normalize("কি") == "কি"
    assert "ি" in code_switch_normalize("কি?")


def test_code_switch_makes_english_case_match():
    # "Bank" vs "bank" is not a transcription error in code-switch speech.
    assert code_switch_normalize("আমি Bank এ যাব") == code_switch_normalize("আমি bank এ যাব")
    # ...but the frozen normalizer (used elsewhere) keeps the case distinction.
    assert normalize("আমি Bank এ যাব") != normalize("আমি bank এ যাব")


def test_cross_script_is_not_credited():
    # A Latin word and its Bengali spelling are NOT treated as equal (no fabrication).
    assert code_switch_normalize("school") != code_switch_normalize("স্কুল")


def test_cs_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", CS_NORMALIZER_VERSION)


# ---- script analysis -----------------------------------------------------


def test_script_counts():
    counts = script_counts("আমি bank 2026")
    assert counts["bengali"] == 3  # আ ম ি
    assert counts["latin"] == 4  # b a n k
    assert counts["digit"] == 4


def test_bengali_digits_count_as_digits_not_letters():
    counts = script_counts("২০২৬")  # Bengali digits
    assert counts["digit"] == 4
    assert counts["bengali"] == 0


def test_bengali_fraction():
    assert bengali_fraction("শুধু বাংলা") == 1.0
    assert bengali_fraction("only english") == 0.0
    assert bengali_fraction("আমি bank") == pytest.approx(3 / 7)  # 3 Bengali, 4 Latin letters


def test_is_code_switched():
    assert is_code_switched("আমি bank এ যাব")
    assert not is_code_switched("আমি ব্যাংকে যাব")
    assert not is_code_switched("only english")


# ---- report integration: the CS slice uses the CS normalizer -------------


def _cs_manifest(domain):
    seg = Segment(
        id="s0",
        audio="a.wav",
        text="আজকের Lecture এ Machine Learning",
        duration_s=4.0,
        recording_id="r0",
    )
    seg2 = Segment(
        id="s1",
        audio="a.wav",
        text="Database এ Query চালাও",
        duration_s=3.0,
        recording_id="r1",
    )
    return Manifest(
        name="m",
        source="mucs_slr104",
        split="test",
        domain=domain,
        version="v",
        segments=(seg, seg2),
    )


def test_code_switch_slice_scores_case_insensitively():
    # Predictions differ from references only in English capitalization.
    hyps = {"s0": "আজকের lecture এ machine learning", "s1": "database এ query চালাও"}
    cs = score_slice(_cs_manifest("code-switch"), Predictions("s", "m", hyps), n_resamples=200)
    read = score_slice(_cs_manifest("read"), Predictions("s", "m", hyps), n_resamples=200)

    assert cs.score.wer_normalized == 0.0  # case-insensitive Latin → perfect
    assert cs.score.normalizer_version == CS_NORMALIZER_VERSION
    assert read.score.wer_normalized > 0.0  # frozen normalizer keeps the case difference
    assert read.score.normalizer_version == NORMALIZER_VERSION
