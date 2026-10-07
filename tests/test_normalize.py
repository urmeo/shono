"""Frozen Bengali normalization fixtures and Unicode boundaries."""

import json
import re
from pathlib import Path

import pytest

from shono.eval import NORMALIZER_VERSION, normalize

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "normalize_cases.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_fixture_case(case):
    assert normalize(case["input"]) == case["expected"]


def test_matras_survive_normalization():
    # Removing vowel marks would change the word.
    out = normalize("কি?")
    assert "ি" in out, "i-kar vowel mark was stripped"
    assert out == "কি"


def test_nfc_composes_decomposed_o_kar():
    # ka + e-kar (U+09C7) + aa-kar (U+09BE) must compose to ka + o-kar (U+09CB).
    decomposed = "\u0995\u09c7\u09be"
    composed = "\u0995\u09cb"
    assert decomposed != composed
    assert normalize(decomposed) == composed


def test_zero_width_characters_removed():
    for zw in ("\u200b", "\u200c", "\u200d", "\ufeff"):  # ZWSP, ZWNJ, ZWJ, BOM
        assert zw not in normalize(f"\u0986{zw}\u09ae\u09bf")  # আ<zw>মি


def test_idempotent():
    for case in _CASES:
        once = normalize(case["input"])
        assert normalize(once) == once


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", NORMALIZER_VERSION)
