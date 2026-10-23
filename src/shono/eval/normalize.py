"""Frozen Bengali normalization, version 1.1.0.

NFC -> word-level encoding repair -> punctuation/symbols replaced with spaces
-> whitespace collapse -> Bengali digits mapped to ASCII. Zero-width marks
are removed; Latin case is retained. No transliteration or digit-to-word rule.
Changing this order requires a new version and regenerated scores."""

import unicodedata

from bnunicodenormalizer import Normalizer as _BnNormalizer

NORMALIZER_VERSION = "1.1.0"

_bnorm = _BnNormalizer(allow_english=True)

_BN_TO_ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}


def normalize(text: str) -> str:
    """Apply the frozen pipeline to one reference or hypothesis string."""
    text = unicodedata.normalize("NFC", text)
    words = [(_bnorm(word) or {}).get("normalized") or word for word in text.split()]
    text = "".join(
        " " if unicodedata.category(char)[0] in "PS" else char
        for char in " ".join(words)
        if char not in _ZERO_WIDTH
    )
    return " ".join(text.split()).translate(_BN_TO_ASCII_DIGITS)
