"""Synthetic inputs and hand-computed scoring cases."""

import wave
from pathlib import Path

from shono.data.license import DatasetLicense, LicenseRegistry


def synthetic_registry(*sources: str, status: str = "active") -> LicenseRegistry:
    return LicenseRegistry(
        DatasetLicense(
            id=source,
            name="Synthetic fixture",
            url="",
            license="fixture-only",
            redistributable_audio=False,
            attribution_required=False,
            share_alike=False,
            access="unit test",
            status=status,
            hours=None,
            domain="synthetic",
            version="fixture",
            notes="Synthetic test metadata, no real-source permission.",
            confidence="test",
        )
        for source in sources
    )


def write_wav(
    path: Path, *, sample_rate: int = 16_000, frames: int | None = None, sample: int = 0
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = sample.to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        stream.writeframes(frame * (sample_rate if frames is None else frames))
    return path


TINY_MANIFEST = """\
{"manifest": {"name": "tiny-longform-test", "source": "bengali_loop", "split": "test", "domain": "long-form", "version": "fixture-1", "language": "bn", "checksum": "sha256:demo"}}
{"id": "loop01-000", "audio": "rec-loop-01.wav", "text": "আজকে আমরা কথা বলব বাংলা ভাষা নিয়ে", "duration_s": 4.2, "recording_id": "rec-loop-01", "start_s": 0.0, "speaker": "S1"}
{"id": "loop01-001", "audio": "rec-loop-01.wav", "text": "ভাষা হল যোগাযোগের প্রধান মাধ্যম", "duration_s": 3.6, "recording_id": "rec-loop-01", "start_s": 4.2, "speaker": "S1"}
{"id": "loop01-002", "audio": "rec-loop-01.wav", "text": "চলুন শুরু করা যাক", "duration_s": 2.1, "recording_id": "rec-loop-01", "start_s": 7.8, "speaker": "S2"}
{"id": "loop02-000", "audio": "rec-loop-02.wav", "text": "দ্বিতীয় রেকর্ডিং এ স্বাগতম", "duration_s": 3.0, "recording_id": "rec-loop-02", "start_s": 0.0, "speaker": "S1"}
{"id": "loop02-001", "audio": "rec-loop-02.wav", "text": "আমরা দুইশো ছাব্বিশ সালে আছি", "duration_s": 3.9, "recording_id": "rec-loop-02", "start_s": 3.0, "speaker": "S1"}
"""


CI_BLOCKS = {
    "records": [
        ["A", "আমি ভাত খাই", "আমি ভাত খাই"],
        ["A", "সে ঘরে থাকে", "সে ঘরে থাকে"],
        ["B", "আমি ভাত খাই", "আমি ডাল খাই"],
        ["B", "তুমি ভাল আছ", "তুমি ভাল নাই"],
    ],
    "expected_point_wer": 0.16666666666666666,
    "max_possible_wer": 0.3333333333333333,
}

NORMALIZATION_CASES = [
    {"name": "matra_preserved_punct_stripped", "input": "কি?", "expected": "কি"},
    {"name": "danda_stripped", "input": "খাই।", "expected": "খাই"},
    {"name": "whitespace_collapsed", "input": "আমি   ভাত\tখাই", "expected": "আমি ভাত খাই"},
    {"name": "bengali_digits_to_ascii", "input": "২০২৬ সালে", "expected": "2026 সালে"},
    {"name": "english_token_kept", "input": "আমি bank এ যাব", "expected": "আমি bank এ যাব"},
    {"name": "bnorm_repairs_broken_vowel", "input": "অামি।", "expected": "আমি"},
    {"name": "taka_attached_and_spaced_agree", "input": "৳১০০", "expected": "100"},
    {"name": "taka_spaced", "input": "৳ ১০০", "expected": "100"},
    {"name": "punct_splits_tokens", "input": "ভাত,ডাল", "expected": "ভাত ডাল"},
    {"name": "math_symbol_splits", "input": "২+২", "expected": "2 2"},
    {"name": "latin_case_preserved", "input": "আমি Bank এ যাব", "expected": "আমি Bank এ যাব"},
]

SCORE_CASES = [
    {
        "name": "identical",
        "label": "Identical",
        "references": ["আমি ভাত খাই"],
        "hypotheses": ["আমি ভাত খাই"],
        "expected_wer": 0.0,
        "expected_cer": 0.0,
    },
    {
        "name": "one_word_substituted",
        "label": "1 word substituted",
        "references": ["আমি ভাত খাই"],
        "hypotheses": ["আমি ডাল খাই"],
        "expected_wer": 0.3333333333333333,
        "expected_cer": 0.18181818181818182,
    },
    {
        "name": "one_word_deleted",
        "label": "1 word deleted",
        "references": ["আমি ভাত খাই আজ"],
        "hypotheses": ["আমি ভাত খাই"],
        "expected_wer": 0.25,
        "expected_cer": 0.21428571428571427,
    },
    {
        "name": "code_switch_identical",
        "label": "Identical mixed script",
        "references": ["আমি bank এ যাব"],
        "hypotheses": ["আমি bank এ যাব"],
        "expected_wer": 0.0,
        "expected_cer": 0.0,
    },
    {
        "name": "corpus_pools_errors_across_segments",
        "label": "Pooled: equal lengths",
        "references": ["আমি ভাত খাই", "সে ঘরে থাকে"],
        "hypotheses": ["আমি ডাল খাই", "সে ঘরে থাকে"],
        "expected_wer": 0.16666666666666666,
        "expected_cer": 0.09090909090909091,
    },
    {
        "name": "pooled_not_averaged_unequal_segments",
        "label": "Pooled: unequal lengths",
        "_note": "6-word segment with 1 error + 2-word segment with 1 error: pooled 2/8 = 0.25; "
        "macro-average would be (1/6+1/2)/2 = 1/3.",
        "references": ["আমি আজ সকালে ভাত খাই না", "সে থাকে"],
        "hypotheses": ["আমি আজ সকালে ডাল খাই না", "সে যায়"],
        "expected_wer": 0.25,
        "expected_cer": None,
    },
]
