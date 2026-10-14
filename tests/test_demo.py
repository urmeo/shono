"""Demo orchestration, transcript formatting, and the model-card generator."""

import re

import pytest

from shono.demo import format_transcript, render_model_card, transcribe_recording
from shono.diarize import SpeakerSegment
from shono.transcribe import ChunkTranscription, LongFormTranscriber, SpeechSegment, Word
from shono.transcribe.types import Transcript, TranscriptSegment


class _FakeVAD:
    def detect(self, audio_path):
        return [SpeechSegment(0, 4)]


class _FakeTranscriber:
    def transcribe_chunk(self, audio_path, start_s, end_s):
        return (
            ChunkTranscription(
                start_s,
                end_s,
                "আমি ভালো",
                words=(Word(0.0, 1.0, "আমি"), Word(1.0, 2.0, "ভালো")),
            ),
        )


class _FakeDiarizer:
    def diarize(self, audio_path):
        return [SpeakerSegment(0, 4, "A")]


def test_transcribe_recording_attributes_speakers():
    pipeline = LongFormTranscriber(_FakeVAD(), _FakeTranscriber(), pad_s=0.0)
    transcript = transcribe_recording("rec.wav", 4.0, pipeline, _FakeDiarizer())
    assert transcript.text == "আমি ভালো"
    assert all(seg.speaker == "A" for seg in transcript.segments)


def test_transcribe_recording_without_diarizer_has_no_speakers():
    pipeline = LongFormTranscriber(_FakeVAD(), _FakeTranscriber(), pad_s=0.0)
    transcript = transcribe_recording("rec.wav", 4.0, pipeline, None)
    assert transcript.segments[0].speaker is None


def test_format_merges_consecutive_same_speaker():
    transcript = Transcript(
        segments=(
            TranscriptSegment(0, 3, "প্রথম", speaker="A"),
            TranscriptSegment(3, 6, "দ্বিতীয়", speaker="A"),
            TranscriptSegment(6, 9, "তৃতীয়", speaker="B"),
        )
    )
    out = format_transcript(transcript)
    assert out.count("**A**") == 1
    assert "প্রথম দ্বিতীয়" in out
    assert "**B**" in out
    assert "[0:00–0:06]" in out


def test_format_falls_back_when_no_speaker():
    transcript = Transcript(segments=(TranscriptSegment(0, 2, "কথা"),))
    assert "**Speaker**" in format_transcript(transcript)


def _report():
    return {
        "normalizer_version": "1.1.0",
        "slices": ["cv-bn-test", "bengali-loop-test"],
        "cells": [
            {
                "system": "ours",
                "slice": "cv-bn-test",
                "status": "scored",
                "model_kind": "pretrained",
                "normalization_policy": "bengali",
                "normalizer_version": "1.1.0",
                "wer_norm": {"point": 0.20, "lo": 0.15, "hi": 0.25},
            },
            {"system": "ours", "slice": "bengali-loop-test", "status": "pending"},
        ],
    }


def test_model_card_quotes_scored_numbers_and_dashes_pending():
    card = render_model_card(
        _report(),
        model_id="urmeo/shono-medium-bn",
        system="ours",
        base_model="base/whisper-medium",
        license_spdx="unknown",
        base_model_license="unknown",
        training_status="pending",
        repo_url="https://github.com/urmeo/shono",
        limitations=["bn-BD register only"],
    )
    assert "20.0% [15.0%, 25.0%]" in card
    assert "| bengali-loop-test | pending |" in card
    assert "v1.1.0" in card
    assert "bn-BD register only" in card


@pytest.mark.parametrize("model_kind", [None, "unknown"])
def test_model_card_allows_pending_unknown_kind_without_inventing_score(model_kind):
    report = {
        "normalizer_version": "1.1.0",
        "slices": ["s"],
        "cells": [{"system": "ours", "slice": "s", "status": "pending"}],
    }
    if model_kind is not None:
        report["cells"][0]["model_kind"] = model_kind
    card = render_model_card(
        report,
        model_id="m",
        system="ours",
        base_model="b",
        license_spdx="unknown",
        base_model_license="unknown",
        training_status="pending",
        repo_url="u",
    )
    assert "| s | pending |" in card
    assert re.search(r"\| s \| [\d.]+%", card) is None


@pytest.mark.parametrize(
    ("slice_name", "policy", "version", "expected"),
    [
        ("test|extra", "bengali", "1.1.0", r"test\|extra | 20.0% [15.0%, 25.0%] | bengali v1.1.0"),
        (
            "test\nextra\rrow",
            "bengali",
            "1.1.0",
            "test extra row | 20.0% [15.0%, 25.0%] | bengali v1.1.0",
        ),
        (
            "test",
            "bengali|custom\npolicy",
            "1.1.0",
            r"test | 20.0% [15.0%, 25.0%] | bengali\|custom policy v1.1.0",
        ),
        (
            "test",
            "bengali",
            "1|custom\rversion",
            r"test | 20.0% [15.0%, 25.0%] | bengali v1\|custom version",
        ),
    ],
)
def test_model_card_preserves_table_columns_and_rows(slice_name, policy, version, expected):
    report = _report()
    report["slices"][0] = slice_name
    report["cells"][0].update(
        slice=slice_name, normalization_policy=policy, normalizer_version=version
    )
    card = render_model_card(
        report,
        model_id="m",
        system="ours",
        base_model="b",
        license_spdx="unknown",
        base_model_license="unknown",
        training_status="unknown",
        repo_url="u",
    )
    table_lines = [line for line in card.splitlines() if line.startswith("|")]
    assert table_lines[2] == f"| {expected} |"
    assert len(table_lines) == 4
    assert all(len(re.split(r"(?<!\\)\|", line)) == 5 for line in table_lines)


def test_card_requires_explicit_training_and_weight_licenses():
    with pytest.raises(TypeError):
        render_model_card(
            _report(), model_id="m", system="ours", base_model="b", license_spdx="MIT", repo_url="u"
        )


def test_card_mixed_policy_is_per_cell_and_no_trained_claim():
    report = _report()
    report["cells"][1].update(normalization_policy="code-switch-script", normalizer_version="1.0.0")
    card = render_model_card(
        report,
        model_id="m",
        system="ours",
        base_model="b",
        license_spdx="unknown",
        base_model_license="unknown",
        training_status="unknown",
        repo_url="u",
    )
    assert "bengali v1.1.0" in card and "code-switch-script v1.0.0" in card
    assert "Training status: **unknown**" in card
    assert "fine-tuned from" not in card and "lecture" not in card
    assert "Model-weight license: **unknown**. Code license: MIT." in card


@pytest.mark.parametrize("point", [True, -1, float("nan"), float("inf"), 1e308, "0.2"])
def test_card_rejects_invalid_scores(point):
    report = _report()
    report["cells"][0]["wer_norm"]["point"] = point
    with pytest.raises(ValueError, match="WER"):
        render_model_card(
            report,
            model_id="m",
            system="ours",
            base_model="b",
            license_spdx="unknown",
            base_model_license="unknown",
            training_status="pending",
            repo_url="u",
        )


def test_card_extreme_integer_score_is_a_domain_error():
    report = _report()
    report["cells"][0]["wer_norm"]["point"] = 10**1000
    with pytest.raises(ValueError, match="WER"):
        render_model_card(
            report,
            model_id="m",
            system="ours",
            base_model="b",
            license_spdx="unknown",
            base_model_license="unknown",
            training_status="pending",
            repo_url="u",
        )


@pytest.mark.parametrize("system", ["fine-tuned Whisper", "Fine-tuned Whisper", "custom model"])
@pytest.mark.parametrize("audit", [None, [], "reviewed", {"status": "unreviewed"}])
def test_fine_tuned_card_cannot_publish_unreviewed_scored_cell(system, audit):
    report = _report()
    for cell in report["cells"]:
        cell["system"] = system
    report["cells"][0].update(model_kind="fine_tuned", leakage_audit=audit)
    with pytest.raises(ValueError, match="reviewed training audit"):
        render_model_card(
            report,
            model_id="m",
            system=system,
            base_model="b",
            license_spdx="unknown",
            base_model_license="unknown",
            training_status="unknown",
            repo_url="u",
        )


def test_fine_tuned_card_allows_reviewed_scored_cell():
    report = _report()
    report["cells"][0].update(model_kind="fine_tuned", leakage_audit={"status": "reviewed"})
    card = render_model_card(
        report,
        model_id="m",
        system="ours",
        base_model="b",
        license_spdx="unknown",
        base_model_license="unknown",
        training_status="trained",
        repo_url="u",
    )
    assert "20.0% [15.0%, 25.0%]" in card


@pytest.mark.parametrize("model_kind", ["pretrained", "provider"])
def test_card_uses_model_kind_rather_than_display_name(model_kind):
    report = _report()
    for cell in report["cells"]:
        cell["system"] = "fine-tuned Whisper"
    report["cells"][0]["model_kind"] = model_kind
    card = render_model_card(
        report,
        model_id="m",
        system="fine-tuned Whisper",
        base_model="b",
        license_spdx="unknown",
        base_model_license="unknown",
        training_status="unknown",
        repo_url="u",
    )
    assert "20.0% [15.0%, 25.0%]" in card


@pytest.mark.parametrize("model_kind", [None, "unknown", "Fine-tuned Whisper", False, []])
def test_scored_card_requires_declared_model_kind(model_kind):
    report = _report()
    if model_kind is None:
        del report["cells"][0]["model_kind"]
    else:
        report["cells"][0]["model_kind"] = model_kind
    with pytest.raises(ValueError, match="model_kind"):
        render_model_card(
            report,
            model_id="m",
            system="ours",
            base_model="b",
            license_spdx="unknown",
            base_model_license="unknown",
            training_status="unknown",
            repo_url="u",
        )
