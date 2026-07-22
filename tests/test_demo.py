"""Demo orchestration, transcript formatting, and the model-card generator."""
import re

from shono.demo import format_transcript, render_model_card, transcribe_recording
from shono.diarize import SpeakerSegment
from shono.transcribe import ChunkTranscription, LongFormTranscriber, SpeechSegment, Word
from shono.transcribe.types import Transcript, TranscriptSegment


class _FakeVAD:
    def detect(self, audio_path):
        return [SpeechSegment(0, 4)]


class _FakeTranscriber:
    def transcribe_chunk(self, audio_path, start_s, end_s):
        return ChunkTranscription(
            start_s, end_s, "আমি ভালো",
            words=(Word(0.0, 1.0, "আমি"), Word(1.0, 2.0, "ভালো")),
        )


class _FakeDiarizer:
    def diarize(self, audio_path):
        return [SpeakerSegment(0, 4, "A")]


# ---- orchestration -------------------------------------------------------


def test_transcribe_recording_attributes_speakers():
    pipeline = LongFormTranscriber(_FakeVAD(), _FakeTranscriber(), pad_s=0.0)
    transcript = transcribe_recording("rec.wav", 4.0, pipeline, _FakeDiarizer())
    assert transcript.text == "আমি ভালো"
    assert all(seg.speaker == "A" for seg in transcript.segments)


def test_transcribe_recording_without_diarizer_has_no_speakers():
    pipeline = LongFormTranscriber(_FakeVAD(), _FakeTranscriber(), pad_s=0.0)
    transcript = transcribe_recording("rec.wav", 4.0, pipeline, None)
    assert transcript.segments[0].speaker is None


# ---- formatting ----------------------------------------------------------


def test_format_merges_consecutive_same_speaker():
    transcript = Transcript(segments=(
        TranscriptSegment(0, 3, "প্রথম", speaker="A"),
        TranscriptSegment(3, 6, "দ্বিতীয়", speaker="A"),
        TranscriptSegment(6, 9, "তৃতীয়", speaker="B"),
    ))
    out = format_transcript(transcript)
    # A's two segments merge into one line; B is a separate line.
    assert out.count("**A**") == 1
    assert "প্রথম দ্বিতীয়" in out
    assert "**B**" in out
    assert "[0:00–0:06]" in out


def test_format_falls_back_when_no_speaker():
    transcript = Transcript(segments=(TranscriptSegment(0, 2, "কথা"),))
    assert "**Speaker**" in format_transcript(transcript)


# ---- model card ----------------------------------------------------------


def _report():
    return {
        "normalizer_version": "1.1.0",
        "slices": ["cv-bn-test", "bengali-loop-test"],
        "cells": [
            {"system": "ours", "slice": "cv-bn-test", "status": "scored",
             "wer_norm": {"point": 0.20, "lo": 0.15, "hi": 0.25}},
            {"system": "ours", "slice": "bengali-loop-test", "status": "pending"},
        ],
    }


def test_model_card_quotes_scored_numbers_and_dashes_pending():
    card = render_model_card(
        _report(), model_id="urme/shono-medium-bn", system="ours",
        base_model="base/whisper-medium", license_spdx="MIT",
        repo_url="https://github.com/urme/shono", limitations=["bn-BD register only"],
    )
    assert "20.0% [15.0%, 25.0%]" in card  # the scored slice
    assert "| bengali-loop-test | — |" in card  # the pending slice, honest dash
    assert "v1.1.0" in card  # frozen normalizer stamped
    assert "bn-BD register only" in card  # limitation carried through


def test_model_card_never_invents_a_number_for_pending_only_report():
    report = {"normalizer_version": "1.1.0", "slices": ["s"],
              "cells": [{"system": "ours", "slice": "s", "status": "pending"}]}
    card = render_model_card(
        report, model_id="m", system="ours", base_model="b",
        license_spdx="MIT", repo_url="u",
    )
    assert "| s | — |" in card
    # No fabricated number in the results table: the only slice row is a dash.
    assert re.search(r"\| s \| [\d.]+%", card) is None
