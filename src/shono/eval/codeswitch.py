"""Script-normalized scoring for the Bengali-English code-switch slice.

No citable Bn-En code-switch WER convention exists, so Shono defines and documents
its own — and keeps it honest. The policy is **within-script** normalization:

    1. the frozen pipeline (:func:`shono.eval.normalize`) — Bengali repaired by
       bnunicodenormalizer, punctuation/digits handled — applied as everywhere else;
    2. plus case-folding of Latin (English) tokens, because English capitalization
       is not a distinction a Bengali speech transcriber should be judged on.

Implemented as ``normalize(text).casefold()``: ``casefold`` lowercases only cased
scripts (Latin), leaving Bengali — which has no case — untouched.

What this deliberately does **not** do: treat a word written in Latin as equal to
its Bengali transliteration ("school" vs "স্কুল"). Reliable Bn-En transliteration
equivalence would need a lexicon that does not exist; asserting it would fabricate
matches. That limitation is stated in the code-switch report, not hidden.
"""

from __future__ import annotations

import unicodedata

from shono.eval.normalize import normalize

# Bump only via a documented decision; a change regenerates every code-switch report.
CS_NORMALIZER_VERSION = "1.0.0"

_BENGALI_BLOCK = range(0x0980, 0x0A00)


def code_switch_normalize(text: str) -> str:
    """Frozen normalization plus case-insensitive Latin, for code-switch scoring."""
    return normalize(text).casefold()


def _script_of(char: str) -> str:
    if ord(char) in _BENGALI_BLOCK:
        return "bengali"
    if char.isascii() and char.isalpha():
        return "latin"
    if char.isdigit():
        return "digit"
    return "other"


def script_counts(text: str) -> dict[str, int]:
    """Count characters by script (bengali / latin / digit / other) after NFC."""
    counts = {"bengali": 0, "latin": 0, "digit": 0, "other": 0}
    for char in unicodedata.normalize("NFC", text):
        if not char.isspace():
            counts[_script_of(char)] += 1
    return counts


def bengali_fraction(text: str) -> float:
    """Fraction of letters that are Bengali (vs Latin); 1.0 = pure Bengali, 0.0 = pure Latin."""
    counts = script_counts(text)
    letters = counts["bengali"] + counts["latin"]
    return counts["bengali"] / letters if letters else 1.0


def is_code_switched(text: str) -> bool:
    """True when the text mixes Bengali and Latin letters — a code-switch candidate."""
    counts = script_counts(text)
    return counts["bengali"] > 0 and counts["latin"] > 0
