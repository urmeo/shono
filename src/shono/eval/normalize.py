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
