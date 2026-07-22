"""Frozen Bengali text normalization — the single pipeline behind every reported number.

Order (fixed; changing it invalidates and regenerates all reports):
    1. Unicode NFC composition.
    2. bnunicodenormalizer, word by word, with ``allow_english=True``. This
       step *repairs* malformed Bengali encodings (broken vowel forms like
       অ + া  → আ, invalid conjunct glue, stray nukta/zero-width sequences)
       and may therefore rewrite or drop codepoints inside a word; that is
       its purpose. Words it cannot process pass through unchanged.
    3. Punctuation and symbols — every codepoint in Unicode categories P*
       (includes the danda '।') and S* (includes the taka sign '৳') — are
       replaced with a space, never silently deleted, so 'ভাত,ডাল' splits
       into two tokens exactly like 'ভাত, ডাল'. Zero-width characters
       (ZWSP/ZWNJ/ZWJ/BOM) are removed.
    4. Whitespace collapse to single ASCII spaces, stripped at both ends.
    5. Digit canonicalization: Bengali digits ০-৯ map to ASCII 0-9, so
       '২০২৬' and '2026' score as equal. Digits are never spelled out.

Deliberate policy edges, so nobody discovers them in a report:
    - Latin (code-switched) tokens keep their case: 'Bank' ≠ 'bank'.
    - Currency/math symbols carry no lexical weight: '৳১০০' and '৳ ১০০'
      both normalize to '100'; '২+২' becomes '2 2'.
    - Bengali combining marks (matras, category Mn) and the visarga 'ঃ'
      (category Mc) are never touched: generic normalizers that strip
      Mark-category codepoints corrupt Bengali words and fabricate WER
      improvements, which is exactly what this module exists to prevent.
      OpenAI Whisper's BasicTextNormalizer must not be used anywhere in
      this codebase.
"""

import unicodedata

from bnunicodenormalizer import Normalizer as _BnNormalizer

# Stamped into every report; bump only via a new decision record, and
# regenerate every report in the same change.
NORMALIZER_VERSION = "1.1.0"

_bnorm = _BnNormalizer(allow_english=True)

_BN_TO_ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}  # ZWSP, ZWNJ, ZWJ, BOM


def _normalize_word(word: str) -> str:
    result = _bnorm(word)
    normalized = result.get("normalized") if result else None
    return normalized if normalized else word


def _replace_punct_and_symbols(text: str) -> str:
    out = []
    for c in text:
        if c in _ZERO_WIDTH:
            continue
        out.append(" " if unicodedata.category(c)[0] in "PS" else c)
    return "".join(out)


def normalize(text: str) -> str:
    """Apply the frozen pipeline to one reference or hypothesis string."""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(_normalize_word(w) for w in text.split())
    text = _replace_punct_and_symbols(text)
    text = " ".join(text.split())
    return text.translate(_BN_TO_ASCII_DIGITS)
